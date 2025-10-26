# NUMA-Aware AMX Matmul: Dual-Socket SNC-2 Implementation

## Target Architecture

**Fixed topology:** Dual-socket Intel Xeon with SNC-2 (Sub-NUMA Clustering)
```
Socket 0 (L3: 320MB, BW: 150-200 GB/s)  │  Socket 1 (L3: 320MB, BW: 150-200 GB/s)
├─ NUMA Node 0                          │  ├─ NUMA Node 2
└─ NUMA Node 1                          │  └─ NUMA Node 3
Cross-socket QPI/UPI: 30-40 GB/s (BOTTLENECK)
```

**Bandwidth constraint:** Cross-socket is 4-5× slower than local, easily saturated by 128 cores.

**Problem with interleaving:** 75% remote accesses → QPI saturated → 2.5-4× slowdown.

## Tensor Parallel Solution: Two Patterns

### Pattern 1: Column-Parallel `C = A @ B`
```
A: REPLICATE per socket    (accessed for all output columns)
B: PARTITION by columns    (B[:, j] only for C[:, j])
C: PARTITION by columns    (output)

Socket 0: C[:, 0:N/2]   = A_local @ B[:, 0:N/2]
Socket 1: C[:, N/2:N]   = A_local @ B[:, N/2:N]
Cross-socket traffic: 0 bytes (after setup)
```

### Pattern 2: Row-Parallel `D = C @ W`
```
C: PARTITIONED (from previous col-parallel)
W: PARTITION by rows
D: ALL-REDUCE required

Socket 0: D_partial = C[:, 0:N/2]   @ W[0:N/2, :]
Socket 1: D_partial = C[:, N/2:N]   @ W[N/2:N, :]
D = All-Reduce(D_partial_0 + D_partial_1)
Cross-socket traffic: 1× output size
```

### Multi-Layer Alternation
```
Col-parallel → Row-parallel → Col-parallel
   (0 comm)      (All-Reduce)     (0 comm)
```
Column-parallel outputs feed row-parallel inputs with no data movement.

## Implementation Plan

### What's Fixed (Hardcoded)
```cpp
constexpr int NUM_SOCKETS = 2;
constexpr int NUM_NODES = 4;
constexpr int socket_for_node(int n) { return n / 2; }
constexpr int primary_node_for_socket(int s) { return s * 2; }  // 0 or 2
```

### What's Discovered (Runtime via libnuma)
```cpp
int cpus_per_node;                    // Could be 24, 32, 48, 64...
cpu_set_t node_cpusets[4];            // Actual CPU IDs (handles SMT, numbering)
int node_ids[4];                      // Physical NUMA node IDs (usually 0,1,2,3)
```

### Three Allocation Primitives
```cpp
// 1. Replicate data to both sockets (share within socket via L3)
T* replicate_to_socket(int socket, const T* source, size_t count);

// 2. Partition columns across sockets
std::pair<T*, T*> partition_columns(const T* source, size_t rows, size_t cols);

// 3. Allocate partitioned output (first-touch zero)
std::pair<T*, T*> alloc_partitioned_output(size_t rows, size_t cols);
```

### Column-Parallel Matmul
```cpp
void matmul_column_parallel(A_global, B_global, C_result) {
    // Setup
    auto A_repl = replicate_to_sockets(A_global);
    auto [B_left, B_right] = partition_columns(B_global);
    auto [C_left, C_right] = alloc_partitioned_output(M, N);

    // Compute on both sockets (parallel)
    #pragma omp parallel num_threads(2)
    {
        int socket = omp_get_thread_num();
        pin_to_socket(socket);
        matmul_amx_kernel(A_local, B_local, C_local);  // 100% local bandwidth
    }
}
```

### Row-Parallel Matmul (Multi-Layer)
```cpp
void matmul_row_parallel(C_partitioned, W_global, D_result) {
    auto [W_top, W_bottom] = partition_rows(W_global);

    // Compute partial results
    #pragma omp parallel num_threads(2)
    {
        int socket = omp_get_thread_num();
        matmul_amx_kernel(C_local, W_local, D_partial);
    }

    // All-Reduce (copy + sum)
    all_reduce_dual_socket(D_partial0, D_partial1, D_result);
}
```

### Performance Targets
- Column-parallel: **2.5-4× vs interleaved** (0 cross-socket traffic during compute)
- Row-parallel: ~2 ms All-Reduce overhead per layer @ 4096×4096 int32
- Code complexity: **~150 lines** (vs ~800 for hwloc auto-tuning)

### Critical Details
1. **Allocate on primary nodes (0, 2):** Nodes within socket share via L3 cache
2. **First-touch:** Zero C partitions with correct thread before compute
3. **VNNI conversion:** Apply AFTER partitioning B matrix (per-socket conversion)
4. **Alignment:** 64-byte aligned allocations required for AMX
5. **Validation:** Check numa_num_configured_nodes() == 4 at startup
