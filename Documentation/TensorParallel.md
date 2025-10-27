# Tensor Parallelism for AMX Matrix Multiplication

Tensor parallelism splits a **single matmul** `C[M, N] = A[M, K] @ B[K, N]` across multiple NUMA nodes/sockets to distribute memory and computation.

## Fundamental Math: Reduction vs. Non-Reduction Dimensions

Each output element is computed as:
```
C[i, j] = Σ(k=0 to K-1) A[i, k] * B[k, j]
```

Three dimensions with different properties:
- **M dimension** (rows of A/C): Non-reduction, freely parallelizable
- **N dimension** (columns of B/C): Non-reduction, freely parallelizable
- **K dimension**: Reduction dimension (the sum), requires special handling when split

**Key insight:** Splitting non-reduction dimensions produces independent complete results. Splitting the reduction dimension produces dependent partial results.

## Column-Parallel: Split N (Non-Reduction)

Each node computes **different columns** with **complete K reduction**:

```
Node 0: C[:, 0:N/2] = A[M, K] @ B[K, 0:N/2]
Node 1: C[:, N/2:N] = A[M, K] @ B[K, N/2:N]

Each node's computation:
for j in node_columns:
    C[i, j] = Σ(k=0 to K-1) A[i, k] * B[k, j]  ← FULL K reduction
                ^-----------------------^
                Complete sum, final result
```

**Properties:**
- Each node performs full K-loop (0 to K)
- Results per node are **final** (no partial sums)
- **Zero cross-node communication** (outputs are independent)
- A must be replicated or shared-read across nodes
- B is partitioned by columns across nodes
- Memory per node: `A[M,K] + B[K,N/n_nodes] + C[M,N/n_nodes]`

## Row-Parallel: Split K (Reduction Dimension)

Each node computes **partial sums for all columns** with **partial K reduction**:

```
Node 0: partial0 = A[:, 0:K/2] @ B[0:K/2, :]
Node 1: partial1 = A[:, K/2:K] @ B[K/2:K, :]
Final:  C = partial0 + partial1  ← ALL-REDUCE required

Each node's computation:
C_partial[i, j] = Σ(k=node_k_start to node_k_end) A[i, k] * B[k, j]  ← PARTIAL reduction
                  ^---------------------------------------^
                  Incomplete sum, not final
```

**Properties:**
- Each node performs partial K-loop (e.g., 0 to K/2)
- Results per node are **partial** (incomplete sums)
- **Requires all-reduce** to sum partials into final result
- A is partitioned by columns across nodes
- B is partitioned by rows across nodes
- Cross-socket communication cost (expensive on NUMA)

## VNNI Format Interaction

VNNI packing (groups of 4 consecutive K-elements for each N column) is **identical** for both patterns.

**Column-parallel:**
```cpp
// Slice B by columns first, then convert each slice independently
for (int node = 0; node < num_nodes; node++) {
    auto B_slice = B[:, node*N/2:(node+1)*N/2];
    B_vnni[node] = convert_to_vnni(B_slice);  // Independent conversion
}
```

**Row-parallel:**
```cpp
// Slice B by rows first, then convert each slice independently
for (int node = 0; node < num_nodes; node++) {
    auto B_slice = B[node*K/2:(node+1)*K/2, :];
    B_vnni[node] = convert_to_vnni(B_slice);  // Independent conversion
}
```

**No special handling needed:** Slice first, convert second. The VNNI format of a slice is exactly what you get by converting that slice - no relationship to the full matrix's VNNI format required.

**Constraints:**
- Column-parallel: Split N at TILE_N (16) or N_BLOCK (256) boundaries
- Row-parallel: Split K at TILE_K (64) or K_BLOCK (4096) boundaries

## Megatron Pattern: Alternating Column/Row

```
Layer 1 (column-parallel):
  Y1_part = X @ W1_part[d_in, d_hidden/n]
  Each node: full d_in reduction → final Y1_part ✓
  Communication: none

Layer 2 (row-parallel):
  Y2_partial = Y1_part @ W2_part[d_hidden/n, d_out]
  Each node: partial d_hidden reduction → Y2_partial
  Y2 = all_reduce_sum(Y2_partial)  ← Cross-node communication
  Communication: one all-reduce

Layer 3 (column-parallel):
  Y3_part = Y2 @ W3_part[d_out, d_next/n]
  Each node: full d_out reduction → final Y3_part ✓
  Communication: none (Y2 replicated/broadcast)
```

**Why alternate:** Minimizes all-reduce operations. Most layers are column-parallel (no communication), occasional row-parallel layers require one all-reduce each.

## Distribution Semantics Summary

| Aspect | Column-Parallel | Row-Parallel |
|--------|----------------|--------------|
| **Split dimension** | N (non-reduction) | K (reduction) |
| **K-loop per node** | Full (0 to K) | Partial (node_offset to node_offset+K/n) |
| **Result per node** | Final columns | Partial sums |
| **Cross-node comm** | None | All-reduce required |
| **A distribution** | Replicated | Partitioned by columns |
| **B distribution** | Partitioned by columns | Partitioned by rows |
| **C distribution** | Partitioned by columns | All nodes produce partials for all elements |
| **VNNI packing** | Same structure | Same structure |
| **When to use** | Large N, avoid communication | Large K, memory-constrained, can amortize all-reduce cost |

## Memory Trade-offs

**Column-parallel (2 nodes):**
- A replicated: 2 × M × K
- B partitioned: M × K total
- C partitioned: M × N total
- **Total:** 2 × (M × K) + M × K + M × N
- **Saves memory on B and C, duplicates A**

**Row-parallel (2 nodes):**
- A partitioned: M × K total
- B partitioned: K × N total
- C partials: 2 × M × N (temporary, becomes M × N after reduce)
- **Total:** M × K + K × N + M × N (minimal duplication)
- **Saves memory everywhere, but requires communication**

## Implementation Pattern

```cpp
// Column-parallel abstraction
template<typename View, int Dim>
struct PartitionedView {
    View base;
    int num_partitions;

    auto get_partition(int node_id) {
        // Return slice along dimension Dim for this node
    }
};

// Usage:
auto B_partitioned = PartitionedView<BVNNIView, /*Dim=*/1>(B_vnni, num_nodes);
auto C_partitioned = PartitionedView<CView, /*Dim=*/1>(C, num_nodes);

// Each node:
matmul_amx(A_replicated, B_partitioned.get_partition(node_id), C_partitioned.get_partition(node_id));
// Done - no communication
```

Key abstraction needed: `PartitionedView` (counterpart to existing `ReplicatedView`).
