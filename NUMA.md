# NUMA-Aware Multiprocessing in Modern C++: A 2025 Technical Guide

Non-Uniform Memory Access programming in C++20/23 requires explicit attention to hardware topology and memory placement, as the standard provides no native NUMA support. However, **modern C++ features combined with specialized libraries like hwloc and Intel TBB can achieve 1.5x to 5.94x performance improvements**, with specialized optimizations reaching 14x speedup on NUMA systems. The critical challenge: C++ abstracts memory as a flat address space, leaving developers to manually bridge the gap between code and hardware topology through custom allocators, thread affinity control, and careful initialization patterns.

This disconnect between C++'s memory model and NUMA hardware realities creates both opportunity and peril. Systems programmed without NUMA awareness routinely suffer 50-200% performance degradation on multi-socket servers, yet the same workloads can achieve near-linear scaling with proper optimization. The path forward combines hwloc for portable topology discovery, modern C++20 primitives for better concurrency control, and explicit data placement strategies that respect memory locality.

## Why NUMA awareness is critical for modern multicore systems

Every modern dual-socket server creates a NUMA architecture where memory access times vary by 30-50% depending on whether threads access local versus remote memory. A typical two-socket Intel Xeon system shows **local memory access at ~100ns versus remote access at ~150ns**, representing a 50% latency penalty. Under heavy load with memory controller congestion, these differences explode: normal 300-cycle latencies can balloon to 1,200 cycles when multiple cores hammer a single node's memory controller.

Modern CPUs amplify these challenges through complex cache hierarchies. Intel Sandy Bridge and later architectures use distributed L3 caches carved into 2.5MB slices, accessed via a scalable ring interconnect. Each socket contains its integrated memory controller, private L1/L2 caches per core, and shared L3 cache. Memory addresses are hashed and distributed among L3 slices, creating variable on-die latencies before even considering cross-socket access. AMD's EPYC architecture adds another layer: multi-die designs with up to four "Zeppelin" dies per package create multiple NUMA domains within a single socket, configurable as NPS1, NPS2, or NPS4 (nodes per socket).

The bandwidth story is equally stark. **AMD EPYC provides approximately 145-150 GB/s per socket versus Intel Broadwell's 65 GB/s** - a 2.3x advantage. Remote sequential reads plummet to 6-20 GB/s compared to local 44-65 GB/s reads, a drop to one-seventh of local bandwidth. The QPI/UPI interconnects between sockets create fundamental bottlenecks with theoretical maximums around 36-40 GB/s, but practical limits near 30 GB/s in Home Snoop mode.

These hardware characteristics make NUMA awareness non-optional for performance-critical applications on multi-socket systems, particularly those that are memory-bandwidth-bound or exhibit random memory access patterns.

## The first-touch problem and C++'s lack of NUMA awareness

The fundamental challenge facing C++ programmers is stark: **C++ has no concept of data locality regarding NUMA topologies**. Memory is presented as a flat, homogeneous address space where `new`, `std::allocator`, and standard containers like `std::vector` operate without any awareness of physical memory location. This abstraction, while clean, collides violently with NUMA hardware realities.

Linux's default first-touch policy determines physical memory placement not when memory is allocated, but when it's first accessed. This creates a pernicious trap. Consider this common pattern:

```cpp
// Main thread allocates and initializes (may run on node 0)
std::vector<int> data(10'000'000);
for(int i = 0; i < data.size(); i++) {
    data[i] = initial_value;  // All pages allocated on node 0
}

// Worker threads distributed across all nodes
#pragma omp parallel for
for(int i = 0; i < data.size(); i++) {
    results[i] = process(data[i]);  // Remote access for threads on node 1!
}
```

All memory lands on NUMA node 0 because the main thread touched it first during initialization. Worker threads on node 1 now face remote memory access penalties for every operation, causing 50-100% performance degradation. Testing shows this pattern can result in **13% imbalance in page distribution** even with attempted careful initialization, sometimes requiring explicit `move_pages()` system calls to correct.

The correct NUMA-aware approach requires parallel initialization:

```cpp
std::vector<int> data(10'000'000);

// Parallel initialization - first touch by worker threads
#pragma omp parallel for
for(int i = 0; i < data.size(); i++) {
    data[i] = initial_value;  // Each thread touches its portion locally
}

// Now processing has local memory access
#pragma omp parallel for
for(int i = 0; i < data.size(); i++) {
    results[i] = process(data[i]);  // Local access
}
```

This pattern ensures each thread initializes the data it will later process, placing memory pages on the appropriate NUMA node through the first-touch mechanism. The performance difference is dramatic: measurements on 2-socket systems show 50% performance loss with naive initialization versus optimal with proper first-touch, escalating to 70% loss on 4-socket systems.

## False sharing amplification across NUMA boundaries

False sharing - where multiple cores write to different memory locations in the same cache line - becomes catastrophically worse across NUMA nodes. Cache line invalidation requires cross-socket coherency traffic, with cache line migration latencies of 50-100+ cycles consuming QPI/UPI bandwidth. Research documents cases where **57.3% of LLC misses hit remote modified cache lines**, creating 10x performance degradation during high concurrency.

Modern x86-64 systems use 64-byte cache lines, so any concurrent modifications within that boundary trigger false sharing:

```cpp
struct Counters {
    std::atomic<int> counter1;  // Offset 0
    std::atomic<int> counter2;  // Offset 4 - SAME CACHE LINE!
};

Counters counters;

// Thread on node 0
void thread0() {
    counters.counter1.fetch_add(1);  // Invalidates cache line
}

// Thread on node 1  
void thread1() {
    counters.counter2.fetch_add(1);  // Must wait for cache line
}
```

The solution requires explicit padding to force independent structures into separate cache lines:

```cpp
struct alignas(64) Counters {
    std::atomic<int> counter1;
    char padding[60];  // Force to different cache lines
    std::atomic<int> counter2;
};
```

C++17's `std::hardware_destructive_interference_size` provides a portable constant (typically 64 bytes) for this purpose, though ABI stability concerns mean it's often compile-time configurable. The performance impact of proper padding versus false sharing can mean the difference between scalable performance and complete collapse on multi-socket NUMA systems.

Even shared pointers create problems through reference counting. Each `std::shared_ptr` copy operation modifies the atomic reference count, creating a hot cache line that bounces between nodes. On NUMA systems with many threads, this reference count becomes a severe bottleneck, sometimes requiring architectural changes to avoid shared ownership across NUMA boundaries.

## Modern C++ libraries for NUMA-aware programming

The library landscape for NUMA programming in C++20/23 centers on hwloc for topology discovery and Intel TBB (oneTBB) for task parallelism, while memkind has been discontinued as of 2024. **hwloc emerges as the gold standard** for portable NUMA programming, actively maintained by Inria's TADaaM team with the latest version v2.12.2 released in July 2024.

hwloc provides comprehensive hardware topology information including CPU packages, cores, threads, NUMA nodes with memory sizes, complete cache hierarchy (L1/L2/L3), I/O devices (GPUs, NICs, storage), and even memory performance attributes via HMAT (bandwidth and latency). Its API allows explicit memory binding to specific NUMA nodes:

```c
hwloc_topology_t topology;
hwloc_topology_init(&topology);
hwloc_topology_load(topology);

// Get NUMA node and allocate memory on it
hwloc_obj_t node = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, 0);
void* mem = hwloc_alloc_membind(topology, size, node->nodeset, 
                                 HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_STRICT);
```

The library runs on Linux, Windows, macOS, Solaris, AIX, FreeBSD, and Android with a permissive BSD license. It's used internally by major HPC frameworks including OpenMPI, MPICH, SLURM, and serves as the foundation for Intel TBB's NUMA support.

**Intel TBB (now oneTBB as part of oneAPI)** provides high-level C++ abstractions for task-based parallelism with NUMA awareness through task arenas. The 2020 release extended the task_arena interface specifically for NUMA applications:

```cpp
#include <tbb/task_arena.h>

// Create task arena bound to specific NUMA node
tbb::task_arena arena(tbb::task_arena::constraints{}
    .set_numa_id(node_id)
    .set_max_concurrency(num_threads));

// Execute task in specific NUMA arena
arena.execute([&] {
    tbb::parallel_for(0, N, [&](int i) {
        process(local_data[i]);
    });
});
```

TBB requires hwloc v1.11+ for NUMA support (v2.5+ for hybrid CPU detection), demonstrating the ecosystem's interconnection. As of 2024, oneTBB remains very actively maintained under the UXL Foundation with Apache 2.0 licensing and excellent C++20/23 support. The library excels at high-level task parallelism with work-stealing load balancing, though NUMA features require explicit configuration and manual data placement for optimal performance.

**libnuma** provides Linux-specific direct access to kernel NUMA policies but suffers from acknowledged poor documentation and design issues. It's useful for Linux-only deployments needing direct kernel control, but hwloc's libnuma compatibility layer often provides a better alternative for new code. Key limitation: `numa_alloc_*()` functions are notably slower than malloc and can cause contention under multi-threaded allocation due to page-level operations.

**memkind**, previously Intel's sophisticated heap manager built on jemalloc for managing heterogeneous memory (DRAM, HBM, PMEM), is **no longer maintained as of 2024**. Intel recommends migrating to the Unified Memory Framework (UMF) as its successor. Existing memkind code continues functioning but receives no future updates.

## C++20 and C++23 features relevant to NUMA programming

Modern C++ standards provide no explicit NUMA support, but several features improve the foundation for building NUMA-aware systems. **std::jthread** introduced in C++20 represents the most significant practical improvement, automatically joining in its destructor and providing built-in cooperative cancellation via `std::stop_token`. This eliminates the common bug where programs terminate because threads weren't explicitly joined:

```cpp
void setAffinity(std::jthread& thread, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(thread.native_handle(), sizeof(cpu_set_t), &cpuset);
}

std::jthread worker([](std::stop_token st) {
    while (!st.stop_requested()) {
        // NUMA-local work
    }
});
setAffinity(worker, numa_node_cpu);
// Automatically joins on destruction
```

While std::jthread doesn't provide NUMA affinity controls directly, its improved RAII compliance and lifecycle management are crucial for the careful thread management NUMA systems require. The native_handle() method allows platform-specific affinity calls while maintaining exception safety and automatic cleanup.

**Atomic wait and notify operations** in C++20 significantly improve synchronization efficiency across NUMA boundaries. Traditional condition variables require syscalls; atomic waiting operates more efficiently:

```cpp
std::atomic<int> data{0};

// Thread 1: Wait for value change
void consumer() {
    data.wait(0, std::memory_order_acquire);
    // Value has changed from 0
}

// Thread 2: Modify and notify
void producer() {
    data.store(42, std::memory_order_release);
    data.notify_one(); // Wake waiting threads
}
```

Micro-benchmarks show atomic wait/notify can be **up to 30x faster than condition variables**, with reduced syscall overhead particularly beneficial on multi-socket NUMA systems with many threads. The memory ordering semantics become critical: `memory_order_release` and `memory_order_acquire` establish happens-before relationships without the expensive cross-socket synchronization required by `memory_order_seq_cst` (sequential consistency, the default).

On NUMA systems, release-acquire pairs allow efficient one-way data transfers between nodes. The writer uses `release` to ensure all prior writes are visible, while the reader uses `acquire` to synchronize. This avoids unnecessary bidirectional synchronization and reduces cache coherency traffic.

**C++20 coroutines** provide lightweight, stackless cooperative multitasking with potential NUMA benefits. Each coroutine requires minimal memory overhead (no large stack allocation), making them more cache-friendly when many tasks span NUMA nodes. Explicit scheduling capabilities allow coroutines to be dispatched to specific execution contexts, enabling NUMA-aware task placement:

```cpp
// Using a NUMA-aware scheduler (conceptual)
task<void> process_data(numa_scheduler& sched, std::span<data_t> partition) {
    co_await sched.schedule_on_node(preferred_node);
    // Work executes on NUMA-local resources
    process_local(partition);
    co_return;
}
```

However, coroutines provide no built-in NUMA support. The standard focuses on async I/O-bound tasks rather than CPU-bound NUMA workloads. Production libraries like libunifex (Meta), stdexec (NVIDIA), and HPX provide NUMA-aware coroutine schedulers, but these remain outside the standard.

The P2300 "std::execution" proposal, **targeted for C++26**, will bring sender/receiver abstractions for explicit execution resource management. This represents the most significant upcoming advancement for NUMA programming, allowing explicit transitions between execution contexts:

```cpp
using namespace std::execution;

scheduler auto cpu_sched = thread_pool.scheduler();
scheduler auto gpu_sched = gpu_thread_pool.scheduler();

sender auto work = schedule(cpu_sched)
    | then([]{ /* CPU work */ })
    | continues_on(gpu_sched)  // Explicit transition
    | then([]{ /* GPU work */ });

sync_wait(work);
```

The `continues_on()` operation makes cross-NUMA-node transitions explicit, with completion schedulers propagating through sender chains. This composable approach enables building complex pipelines across execution resources with zero-allocation potential through structured concurrency. However, P2300 doesn't expose NUMA topology directly - NUMA affinity must still be implemented in custom schedulers.

## Memory allocation strategies and custom allocators

NUMA-aware memory allocation requires moving beyond standard allocators to explicit node binding or interleaving strategies. **Linux provides five primary NUMA memory policies**: default (first-touch), bind (restrict to specific nodes), interleave (round-robin across nodes at cache-line granularity), preferred (try preferred node with fallback), and local (allocate from the requesting node).

First-touch performs best when initialization patterns match access patterns. Interleave distributes memory uniformly across nodes, typically 20-40% slower than optimal but providing consistent average latency for unpredictable access patterns. Bind offers the best performance when properly configured but worst performance when misconfigured, forcing all accesses to become remote. Benchmarks show interleave policy on single applications can provide **2.4x geomean speedup** when combined with disabled automatic NUMA migrations.

Custom allocators implement NUMA awareness through `std::pmr::memory_resource` interfaces introduced in C++17:

```cpp
#include <memory_resource>
#include <hwloc.h>

class numa_memory_resource : public std::pmr::memory_resource {
    hwloc_topology_t topology_;
    int numa_node_;
    
public:
    numa_memory_resource(int node) : numa_node_(node) {
        hwloc_topology_init(&topology_);
        hwloc_topology_load(topology_);
    }
    
    ~numa_memory_resource() {
        hwloc_topology_destroy(topology_);
    }
    
protected:
    void* do_allocate(size_t bytes, size_t alignment) override {
        auto node = hwloc_get_obj_by_type(topology_, 
                                          HWLOC_OBJ_NUMANODE, 
                                          numa_node_);
        return hwloc_alloc_membind(topology_, bytes, node->nodeset,
                                  HWLOC_MEMBIND_BIND,
                                  HWLOC_MEMBIND_STRICT);
    }
    
    void do_deallocate(void* p, size_t bytes, size_t) override {
        hwloc_free(topology_, p, bytes);
    }
    
    bool do_is_equal(const memory_resource& other) const noexcept override {
        return this == &other;
    }
};

// Usage with C++17 PMR
numa_memory_resource numa_mr(0);  // Node 0
std::pmr::vector<int> data(&numa_mr);
```

This pattern wraps hwloc's explicit memory binding in a standard-compliant allocator usable with any PMR-aware container. For shared read-only data, replication across nodes often outperforms single allocation:

```cpp
// Replicate on each node
for(int node = 0; node < num_nodes; node++) {
    auto data = numa_alloc_onnode(size, node);
    memcpy(data, source, size);
}
```

Thread-private data should allocate on the thread's local node, often using thread_local storage. Shared writable data presents a harder problem: partition for locality when access patterns are predictable, or interleave for uniform random access.

## Thread affinity and topology-aware scheduling

Thread migration destroys NUMA locality as previously local memory becomes remote and cache locality is lost. **Operating systems may migrate threads between cores and nodes**, causing unpredictable performance variations. Thread pinning prevents this:

```cpp
// Using pthread (Linux)
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(core_id, &cpuset);
pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
```

Intel's optimization guide states: "Thread migration from one core to another poses a problem for the NUMA shared memory architecture because of the way it disassociates a thread from its local memory allocations." The solution combines thread pinning with data placement. hwloc provides cross-platform APIs:

```cpp
hwloc_set_cpubind(topology, set, flags);
hwloc_set_proc_cpubind(topology, pid, set, flags);
hwloc_set_thread_cpubind(topology, thread, set, flags);
```

For HPC applications, the recommended pattern uses one MPI rank per NUMA node (typically per socket) with OpenMP threads equal to cores per node. This approach disaggregates at the memory controller level, avoiding cross-node NUMA effects within each rank while using message passing between sockets. Environment variables control this:

```bash
export OMP_PROC_BIND=true
export OMP_PLACES=sockets
mpirun --map-by socket --bind-to socket ./application
```

Intel TBB integrates thread affinity through task arena constraints:

```cpp
tbb::task_arena arena(
    tbb::task_arena::constraints{}
        .set_numa_id(node)
        .set_core_type(core_type)
        .set_max_concurrency(threads)
);
```

Combining TBB with hwloc enables sophisticated NUMA-aware executors:

```cpp
class numa_aware_executor {
    std::vector<tbb::task_arena> arenas_;
    hwloc_topology_t topology_;
    
public:
    numa_aware_executor() {
        hwloc_topology_init(&topology_);
        hwloc_topology_load(topology_);
        
        int num_nodes = hwloc_get_nbobjs_by_type(topology_, 
                                                 HWLOC_OBJ_NUMANODE);
        
        for (int i = 0; i < num_nodes; ++i) {
            arenas_.emplace_back(
                tbb::task_arena::constraints{}.set_numa_id(i)
            );
        }
    }
    
    template<typename F>
    void execute_on_node(int node, F&& func) {
        arenas_[node].execute(std::forward<F>(func));
    }
};
```

This pattern creates one task arena per NUMA node, allowing explicit scheduling of work to specific nodes while leveraging TBB's sophisticated work-stealing and load balancing within each arena.

## Performance profiling and measurement tools

Effective NUMA optimization requires measurement. **Intel VTune Profiler** provides platform-wide NUMA analysis through its Platform Profiler, visualizing local versus remote memory access and cross-socket UPI/QPI traffic. Case studies demonstrate identifying issues where threads migrate between sockets, transforming microsecond latencies into nanoseconds (1000x improvement) through proper pinning.

**numatop** offers real-time process monitoring with RMA (Remote Memory Access) and LMA (Local Memory Access) metrics. The critical metric is the RMA/LMA ratio: below 0.1 indicates excellent locality, 0.1-0.5 represents good locality, 0.5-1.0 shows poor locality, and above 1.0 signals severe NUMA issues. The tool identifies hot memory areas and provides per-thread statistics during live execution.

**NumaPerf** takes a novel predictive approach, focusing on memory sharing patterns rather than just remote accesses. This hardware-independent profiler detects issues missed by other tools and has achieved **up to 5.94x speedup** after applying its recommendations. It's particularly effective for architectural-level optimization decisions during development.

For false sharing detection, **perf c2c** (Cache-to-Cache) remains the definitive tool. It identifies cache lines experiencing false sharing and shows which threads read versus write each line. The key metric is LLC Misses to Remote Cache HITM (Hit In Modified): above 50% indicates severe false sharing requiring immediate attention. Research documents production cases with 57.3% of LLC misses hitting remote modified cache lines, creating 10x performance degradation.

**Linux's numastat** provides system-level statistics:

```bash
# Per-node memory usage
numastat -c

# Application-specific
numastat -p <pid>

# Hardware counters
perf stat -e node-loads,node-load-misses,node-stores <cmd>
```

The command-line tool **numactl** both sets NUMA policies and displays hardware configuration:

```bash
numactl --hardware
numactl --cpunodebind=0 --membind=0 ./application
numactl --interleave=all ./application
```

Together, these tools enable a diagnostic workflow: profile with numatop or VTune to establish baseline performance and RMA/LMA ratios, check for false sharing with perf c2c, implement optimizations, then verify reduced remote accesses (target below 10%) and improved instructions-per-cycle metrics.

## Benchmark results and real-world performance data

Performance improvements from NUMA optimization are substantial and measurable. The **PGASUS framework study** demonstrates average 1.56x speedup with peak 4.67x improvements on NUMA-optimized C++ code. **NumaPerf profiling** achieves up to 5.94x speedup by identifying and fixing memory sharing issues. Intel's research on automatic NUMA migration shows interleave policy with migrations disabled provides **2.4x geomean speedup**, while blocked policy achieves 1.6x improvement.

Database systems show significant NUMA sensitivity. PostgreSQL testing with HammerDB TPROC workload revealed that proper NUMA placement reduced performance variability to 0.3% while improving IPC. MySQL and MariaDB documentation recommends `numactl --interleave=all mysqld` for multi-socket deployments. In some configurations, unbalanced memory allocation causes **up to 20% performance degradation**.

The NAS Parallel Benchmarks illustrate initialization criticality. Before NUMA optimization, applications failed to scale across sockets. After implementing proper NUMA-aware initialization with parallel first-touch, near-linear scaling was achieved. The Carrefour algorithm, which focuses on congestion management rather than just locality, improved performance by **up to 3.6x** over the default Linux kernel with only 4% maximum degradation when optimization wasn't applicable.

Memory bandwidth comparisons between architectures show **AMD EPYC providing approximately 145-150 GB/s per socket versus Intel Broadwell's 65 GB/s** - a 2.3x advantage. Intel Skylake-SP improves to around 100 GB/s per socket but still lags AMD. Remote access bandwidth plummets: sequential reads drop from 44-65 GB/s locally to 6-20 GB/s remotely, a factor of three to seven depending on configuration.

Latency measurements on Intel Xeon E5-2680 v3 show L1 cache at 4 cycles, L2 at 12 cycles, L3 at 26-31 cycles, local DRAM at 190 cycles, and remote DRAM at 310 cycles - representing a **63% remote access penalty**. Under load, this gap widens dramatically. Intel Memory Latency Checker demonstrates unloaded DRAM latency around 79 nanoseconds expanding to 207 nanoseconds under high bandwidth utilization as memory controllers saturate.

False sharing performance impacts are severe. Controlled experiments show **2-10x slowdown** from false sharing on NUMA systems. JArena's partitioned memory approach for addressing false page sharing achieved **4.3x improvement on 256-core systems**. Thread pinning cases in VTune documentation demonstrate transforming microsecond-scale latencies to nanosecond-scale, literally thousand-fold improvements through eliminating cross-socket UPI traffic spikes.

## Recommended approaches and integration patterns

The optimal NUMA programming stack for new C++ projects in 2025 combines **hwloc for topology discovery and memory binding with oneTBB for task parallelism**, implementing custom PMR allocators for fine-grained control. This combination provides portability (hwloc runs everywhere), active maintenance (both projects actively developed), and excellent C++20/23 integration.

For Linux-only deployments, libnuma offers direct kernel control with lowest overhead, particularly for systems programming requiring explicit page-level policies. However, hwloc's libnuma compatibility layer often provides sufficient functionality with better portability.

Critical optimization patterns include parallel initialization to establish first-touch locality:

```cpp
std::vector<int> data(huge_size);

// BAD: Serial initialization causes all pages on one node
for (size_t i = 0; i < data.size(); ++i) data[i] = 0;

// GOOD: Parallel initialization distributes pages
#pragma omp parallel for
for (size_t i = 0; i < data.size(); ++i) data[i] = 0;
```

Thread pinning must match data placement. For parallel reductions, allocate per-node temporary storage and reduce locally before global aggregation:

```cpp
template<typename T>
T numa_aware_reduce(std::span<T> data, size_t num_nodes) {
    std::vector<std::jthread> threads;
    std::vector<T> per_node_results(num_nodes);
    
    size_t chunk = data.size() / num_nodes;
    for (size_t i = 0; i < num_nodes; ++i) {
        threads.emplace_back([=, &per_node_results] {
            pin_to_numa_node(i);
            auto start = data.begin() + i * chunk;
            auto end = (i == num_nodes - 1) ? data.end() : start + chunk;
            per_node_results[i] = std::reduce(start, end);
        });
    }
    // jthreads automatically join
    return std::reduce(per_node_results.begin(), per_node_results.end());
}
```

Prevent false sharing by padding concurrent data structures to cache line boundaries using `std::hardware_destructive_interference_size`:

```cpp
struct alignas(std::hardware_destructive_interference_size) separated_counter {
    std::atomic<int> count;
};

std::array<separated_counter, NUM_THREADS> per_thread_counters;
```

For memory-intensive workloads, diagnostic workflow starts with establishing baseline RMA/LMA ratios using numatop, checking false sharing with perf c2c, implementing first-touch parallel initialization, adding thread pinning, then validating improvements. Target metrics include RMA/LMA ratio below 0.1, remote accesses under 10% of total, and IPC improvements of 1.5-2x for memory-bound codes.

Shared read-only data benefits from replication across NUMA nodes despite memory overhead. Write-heavy shared data requires careful analysis: partition when access patterns are predictable, interleave for uniform random access, or implement reader-writer partitioning where feasible.

## Limitations and future directions

C++20 and C++23 provide no standard NUMA API, forcing reliance on platform-specific APIs or third-party libraries. Thread affinity remains non-standard, requiring `pthread_setaffinity_np()` on Linux, Windows-specific APIs, or portable abstractions through hwloc. The parallel STL introduced in C++17 lacks NUMA awareness entirely, making its guarantees about parallel execution largely irrelevant for NUMA optimization.

**C++26's std::execution (P2300) represents the most significant upcoming improvement**, enabling explicit execution resource control through schedulers, senders, and receivers. The `continues_on()` operation will make cross-resource transitions explicit, though NUMA topology exposure still requires custom scheduler implementations. P1928's std::simd addition also targets C++26, providing portable vectorization useful for NUMA-local data processing.

Current workarounds require combining hwloc or libnuma for topology and memory binding, TBB or HPX for NUMA-aware task parallelism, custom PMR allocators for containers, and explicit first-touch initialization patterns. Research frameworks like PGASUS demonstrate what standard library support could provide: **average 1.56x performance improvement through RAII-based memory placement** with significantly simpler code than manual topology management.

Key anti-patterns to avoid include serial initialization before parallel execution (causes all memory on one node), running without thread pinning on multi-socket systems (allows performance-destroying thread migration), ignoring cache line boundaries in concurrent structures (creates false sharing), using standard allocators for NUMA-sensitive workloads (no locality control), and enabling automatic NUMA balancing for single applications (2.4x overhead).

## Conclusion and key takeaways

NUMA-aware programming in modern C++ requires explicit attention to hardware topology, memory placement, and thread affinity that the standard does not provide. Performance improvements of 1.5-5.94x are consistently achievable through proper NUMA optimization, with specialized cases reaching 14x. The investment proves most justified for memory-intensive applications on multi-socket systems with high core counts, including databases, in-memory analytics, HPC applications, and long-running production workloads.

The current optimal approach combines hwloc for portable topology discovery and memory binding with Intel TBB for NUMA-aware task parallelism, supplemented by custom PMR allocators and explicit first-touch initialization patterns. C++20's std::jthread, atomic wait/notify, and coroutines provide better primitives for building NUMA-aware systems even without direct NUMA support. C++26's std::execution promises significant improvement through explicit scheduler control, though full NUMA awareness will likely require additional future proposals.

Critical optimization techniques focus on parallel initialization matching access patterns, thread pinning to prevent migration, cache line padding to prevent false sharing, and continuous measurement using tools like numatop, VTune, and perf c2c. The gap between C++'s flat memory model and NUMA hardware realities requires manual bridging through careful programming discipline, but the performance rewards on modern multi-socket servers make this effort worthwhile for performance-critical applications. As server architectures increasingly embrace multi-socket designs and heterogeneous memory hierarchies including HBM and persistent memory, NUMA awareness transitions from optimization to requirement for achieving acceptable performance.
