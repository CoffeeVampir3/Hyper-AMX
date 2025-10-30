# Necessary Operations for DeepSeek V3 Inference

## Operations Summary

| Operation | Needed For | Status |
|-----------|------------|--------|
| RMSNorm | Everywhere (3× per layer) | ❌ Need to implement |
| Add | Residual connections | ❌ Need to implement |
| Embedding lookup | Input layer | ❌ Need to implement |
| Linear (matmul) | All projections | ✅ Have matmul primitives |
| SwiGLU | MLP blocks | ✅ Have `silu_mul_requantize` |
| Flash Attention | Attention blocks | ✅ Assumed covered |
| Sigmoid | MoE router | ❌ Need to implement |
| Top-k | MoE router | ❌ Need to implement |
| Dispatch/Combine | MoE expert routing | ❌ Need to implement |
| All-reduce | Row-parallel layers | ✅ Have implementation |

## Model Architecture Flow

```
Input IDs [batch, seq]
  ↓
Embedding Lookup: vocab_embedding[input_ids] → [batch, seq, 7168]
  ↓
For each of 61 decoder layers:
  ┌─────────────────────────────────────┐
  │ ATTENTION BLOCK                      │
  ├─────────────────────────────────────┤
  │ residual = hidden_states             │
  │ hidden = RMSNorm(hidden)             │
  │ hidden = FlashAttention(hidden)      │
  │   - Q proj (with q_a + q_b)          │
  │   - KV proj (with kv_a + kv_b)       │
  │   - RMSNorm inside Q proj            │
  │   - RMSNorm inside KV proj           │
  │   - RoPE on q_rot, k_rot             │
  │   - Attention computation            │
  │   - O proj                           │
  │ hidden = residual + hidden           │
  └─────────────────────────────────────┘

  ┌─────────────────────────────────────┐
  │ MLP BLOCK                            │
  ├─────────────────────────────────────┤
  │ residual = hidden                    │
  │ hidden = RMSNorm(hidden)             │
  │                                      │
  │ if layer == 0:                       │
  │   Dense SwiGLU MLP                   │
  │   gate = Linear(hidden, 18432)       │
  │   up = Linear(hidden, 18432)         │
  │   hidden = Linear(SiLU(gate)*up, 7168)│
  │ else:                                │
  │   MoE (256 experts, 8 active)        │
  │   - Router (sigmoid + top-k)         │
  │   - Expert dispatch                  │
  │   - 8 SwiGLU experts computed        │
  │   - Expert combine                   │
  │   - Shared expert (always active)    │
  │                                      │
  │ hidden = residual + hidden           │
  └─────────────────────────────────────┘
  ↓
Final RMSNorm
  ↓
LM Head: Linear(hidden[:, -1], vocab_size)
```

## Implementation Phases

### Phase 1: Dense Layer (Layer 0)
**Goal:** Get first decoder layer working

Required operations:
1. **RMSNorm** - `x * rsqrt(mean(x^2) + eps) * weight`
2. **Element-wise Add** - Residual connections
3. **Embedding Lookup** - Initial token embeddings
4. **Flash Attention** - Complete attention mechanism (external)
5. **Linear wrappers** - Thin wrappers around existing matmul
   - `linear_column_parallel`
   - `linear_row_parallel` + all-reduce

Already have:
- SwiGLU: `silu_mul_requantize_parallel`
- Matmul: `matmul_amx_column_parallel`, `matmul_amx_row_parallel`
- All-reduce: `all_reduce_sum`

### Phase 2: MoE Layers (Layers ≥1)
**Goal:** Support mixture-of-experts routing

Required operations:
1. **Sigmoid** - `1 / (1 + exp(-x))` for router logits
2. **Group-based Top-k**
   - Reshape to [batch*seq, 8 groups, 32 experts per group]
   - Top-2 within each group, sum scores
   - Top-3 groups selected
   - Top-8 experts from selected groups
3. **Normalize** - `weights / sum(weights)` for expert weights
4. **Expert Dispatch** - Gather tokens for each expert
5. **Expert Combine** - Weighted scatter-add of expert outputs

## Critical Path Operations Detail

### 1. RMSNorm
**Frequency:** 3× per layer (61 layers = ~183 calls)
- Before attention
- Inside Q projection (q_a_layernorm)
- Inside KV projection (kv_a_layernorm)

**Computation:**
```cpp
variance = mean(x^2, dim=-1, keepdim=True)
normalized = x * rsqrt(variance + eps)
output = normalized * weight
```

**Tensor Parallel Strategy:** Column-partitioned
- Each socket computes independently on its partition
- No cross-socket communication needed

### 2. Element-wise Add
**Frequency:** 2× per layer (attention + MLP residuals)

**Computation:**
```cpp
output[i,j] = a[i,j] + b[i,j]
```

**Tensor Parallel Strategy:** Column-partitioned
- Element-wise on local partitions

### 3. Embedding Lookup
**Frequency:** Once at model entry

**Computation:**
```cpp
output[i,j] = embedding_table[input_ids[i,j], :]
```

**Tensor Parallel Strategy:** Column-partitioned embedding table
- Each socket has `[vocab_size, hidden_size/n_sockets]`
- Lookup produces column-partitioned output

### 4. MoE Router
**Frequency:** Layers 1-60 (60 layers)

**Computation:**
```cpp
// 1. Linear + sigmoid
logits = Linear(hidden.fp32, 256)
logits = sigmoid(logits) + e_score_correction_bias

// 2. Group-based selection
group_scores = logits.reshape(BS, 8, 32)
                     .topk(2, dim=-1)[0]
                     .sum(dim=-1)  // [BS, 8]
top_groups = group_scores.topk(3, dim=-1)  // Select 3 groups
mask = create_group_mask(top_groups)

// 3. Expert selection
masked_logits = logits * mask
top_k_indices, top_k_weights = topk(masked_logits, k=8)
top_k_weights = top_k_weights / sum(top_k_weights) * routed_scaling_factor
```

**Tensor Parallel Strategy:**
- Router operates on column-partitioned input
- Each socket computes full routing decision (replicated)
- Experts are partitioned

### 5. Expert Dispatch/Combine
**Frequency:** Layers 1-60

**Computation:**
```cpp
// Dispatch
for expert_id in 0..255:
    tokens = hidden[top_k_indices == expert_id]
    if tokens.empty(): continue

    // SwiGLU for this expert
    gate = Linear(tokens, moe_intermediate_size)
    up = Linear(tokens, moe_intermediate_size)
    expert_out = Linear(SiLU(gate) * up, 7168)

    // Combine with routing weights
    output[token_indices] += expert_out * top_k_weights[...]

// Shared expert (always active)
gate = Linear(hidden, moe_intermediate_size)
up = Linear(hidden, moe_intermediate_size)
shared_out = Linear(SiLU(gate) * up, 7168)
output += shared_out
```

**Tensor Parallel Strategy:** Column-parallel
- Each socket processes its partition of tokens
- Expert weights column-partitioned
- Shared expert also column-partitioned

## Data Types

- **Weight matrices:** int8 (use AMX matmul)
- **Activations:** int32 after matmul, may quantize back to int8
- **Attention scores:** fp32 (softmax requires float)
- **Router logits:** fp32 (sigmoid + top-k in float)
- **RMSNorm:** Compute in fp32, cast back to int8/int32

## Memory Layout Considerations

- **Column-parallel:** Split hidden dimension across sockets
- **Row-parallel:** Split reduction dimension, requires all-reduce
- **Attention:** Head dimension partitioned (128 heads / 2 sockets = 64 heads each)
- **MoE:** Experts partitioned or replicated depending on memory constraints
