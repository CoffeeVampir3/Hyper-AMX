# DeepSeek V3 Format Specification

## Data Types
- **AMXQ**: int8 quantized with per-tile (16×16) bias+scale params
- **bf16**: bfloat16
- **int32**: 32-bit integer activations
- **fp32**: 32-bit float (weights only)

Partitioning suffixes: `_cp` = column-partitioned, `_rp` = row-partitioned, `_rep` = replicated

## Operations

**Compute:**
- `embedding_lookup_column_parallel(token_ids, bf16_cp) -> int32_cp`
- `deepseek_rmsnorm_column_parallel(int32_cp, fp32_cp) -> int32_cp`
- `matmul_amx_column_parallel(int32_rep, AMXQ_cp) -> int32_cp`
- `matmul_amx_row_parallel(int32_cp, AMXQ_rp) -> int32_partials`
- `silu_mul_requantize_parallel(int32_cp, int32_cp) -> AMXQ_cp`
- `add_column_parallel(int32_cp, int32_cp) -> int32_cp`

**Communication:**
- `all_gather(int32_cp) -> int32_rep`
- `all_reduce_sum(int32_partials) -> int32_rep`

## Typical Layer Flow
```
int32_cp = rmsnorm(int32_cp)
int32_rep = all_gather(int32_cp)        // Communication: gather columns
int32_cp = matmul_column_parallel(int32_rep, AMXQ_cp)
int32_cp = add(int32_cp, residual_cp)   // Residual connection
```
