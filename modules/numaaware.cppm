module;
#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <array>
#include <optional>
#include <stdexcept>
#include <print>
#include <thread>
#include <vector>
#include <mdspan>
#include <sched.h>
#include <numa.h>
#include <numaif.h>
export module numaaware;
import tensor;
import matmul;
import tensor_utils;

export struct DualSocketConfig {
    static constexpr int NUM_SOCKETS = 2;
    static constexpr int NUM_NODES = 4;
    static constexpr int NODES_PER_SOCKET = 2;

    // 8: Uses CPUs 0-7, 32-39, 64-71, 96-103 (32 cores total)
    // 16: Uses CPUs 0-15, 32-47, 64-79, 96-111 (64 cores total)
    // 8 is experimentally optimal.
    static constexpr int CORES_PER_NODE = 8;

    int physical_cores_per_socket;
    int total_cpus;

    static constexpr int primary_node_for_socket(int socket) { return socket * NODES_PER_SOCKET; }
    static constexpr int socket_for_node(int node) { return node / NODES_PER_SOCKET; }

    // Map socket + local thread ID to physical core CPU ID (no hyperthreads)
    // Physical cores are interleaved across NUMA nodes:
    // Socket 0: CPUs 0-(CORES_PER_NODE-1) (node 0) + 32-(32+CORES_PER_NODE-1) (node 1)
    // Socket 1: CPUs 64-(64+CORES_PER_NODE-1) (node 2) + 96-(96+CORES_PER_NODE-1) (node 3)
    static constexpr int physical_core_id(int socket, int local_tid) {
        int node_within_socket = local_tid / CORES_PER_NODE;  // 0 or 1
        int core_within_node = local_tid % CORES_PER_NODE;    // 0 to (CORES_PER_NODE-1)
        int base_node = socket * NODES_PER_SOCKET;            // 0 or 2
        return (base_node + node_within_socket) * 32 + core_within_node;
    }

    static DualSocketConfig discover() {
        if (numa_available() < 0) {
            throw std::runtime_error("NUMA not available on this system");
        }
        int num_nodes = numa_num_configured_nodes();
        if (num_nodes != NUM_NODES) {
            throw std::runtime_error(std::format("Expected {} NUMA nodes (dual-socket SNC-2), found {}", NUM_NODES, num_nodes));
        }
        DualSocketConfig config;
        config.total_cpus = numa_num_configured_cpus();
        config.physical_cores_per_socket = CORES_PER_NODE * NODES_PER_SOCKET;

        std::println("Detected {} logical CPUs, using {} cores per socket ({} per node)",
                     config.total_cpus, config.physical_cores_per_socket, CORES_PER_NODE);

        return config;
    }
};


// Socket-replicated tensor: one complete copy per socket
// First-touch guaranteed by pinned threads during initialization
// Typical use: replicate activations for column-parallel matmul (no cross-socket communication)
export template<typename T, typename Extents, typename Layout>
struct SocketReplicated {
    static constexpr int NUM_SOCKETS = DualSocketConfig::NUM_SOCKETS;
    std::array<Tensor<T, Extents, Layout>, NUM_SOCKETS> replicas;

    template<typename InitFn>
    SocketReplicated(Extents e, const DualSocketConfig& config, InitFn&& init)
        : replicas{Tensor<T, Extents, Layout>(e, NumaNodeAllocator<T>{DualSocketConfig::primary_node_for_socket(0)}),
                   Tensor<T, Extents, Layout>(e, NumaNodeAllocator<T>{DualSocketConfig::primary_node_for_socket(1)})} {
        for (int socket = 0; socket < NUM_SOCKETS; socket++) {
            std::jthread([&, socket] {
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                int cpu = DualSocketConfig::physical_core_id(socket, 0);
                CPU_SET(cpu, &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
                init(replicas[socket].data_handle(), replicas[socket].mapping().required_span_size(), socket);
            }).join();
        }
    }

    // Optimize: reuse existing tensor on source_socket, allocate only for other socket
    // Typical use: activation already exists on one socket, replicate to other
    // Performance: 50% reduction (1 allocation + 1 init instead of 2)
    template<typename InitFn>
    SocketReplicated(Tensor<T, Extents, Layout>&& source, int source_socket, const DualSocketConfig& config, InitFn&& init) {
        replicas[source_socket] = std::move(source);

        int other_socket = 1 - source_socket;
        int node = DualSocketConfig::primary_node_for_socket(other_socket);
        replicas[other_socket] = Tensor<T, Extents, Layout>(replicas[source_socket].mapping().extents(), NumaNodeAllocator<T>{node});

        std::jthread([&, other_socket] {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            int cpu = DualSocketConfig::physical_core_id(other_socket, 0);
            CPU_SET(cpu, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            init(replicas[other_socket].data_handle(), replicas[other_socket].mapping().required_span_size(), other_socket);
        }).join();
    }

    SocketReplicated(Tensor<T, Extents, Layout>&& source, int source_socket, const DualSocketConfig& config)
        : SocketReplicated(std::move(source), source_socket, config,
            [this, source_socket](T* dest, size_t count, int) {
                std::memcpy(dest, replicas[source_socket].data_handle(), count * sizeof(T));
            }) {}

    SocketReplicated(Extents e, const DualSocketConfig& config)
        : SocketReplicated(e, config, [](T* p, size_t n, int) { std::memset(p, 0, n * sizeof(T)); }) {}

    SocketReplicated(const SocketReplicated&) = delete;
    SocketReplicated& operator=(const SocketReplicated&) = delete;
    SocketReplicated(SocketReplicated&&) = default;
    SocketReplicated& operator=(SocketReplicated&&) = default;

    auto& operator[](int socket) { return replicas[socket]; }
    const auto& operator[](int socket) const { return replicas[socket]; }
    auto view(int socket) { return replicas[socket].view(); }
    auto view(int socket) const { return replicas[socket].view(); }
};

// Socket-partitioned tensor: slice one dimension across sockets
// PartitionDimValue: PartitionDim::Rows for row-parallel (split K), PartitionDim::Columns for column-parallel (split N)
// First-touch guaranteed by pinned threads during initialization
// Typical use: partition weights/outputs for tensor parallelism
// Memory: Σ(partition_sizes) = total (no duplication unlike replication)
export template<typename T, typename Extents, typename Layout, int PartitionDimValue>
struct SocketPartitioned {
    static constexpr int MAX_SOCKETS = DualSocketConfig::NUM_SOCKETS;
    std::array<std::optional<Tensor<T, Extents, Layout>>, MAX_SOCKETS> partitions;
    int num_partitions;

    template<typename InitFn>
    SocketPartitioned(Extents full_extents, int n_parts, const DualSocketConfig& config, InitFn&& init)
        : num_partitions(n_parts) {

        auto part_extents = compute_partition_extents(full_extents, n_parts);

        for (int socket = 0; socket < n_parts; socket++) {
            int node = DualSocketConfig::primary_node_for_socket(socket);
            partitions[socket].emplace(part_extents, NumaNodeAllocator<T>{node});

            std::jthread([&, socket] {
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                int cpu = DualSocketConfig::physical_core_id(socket, 0);
                CPU_SET(cpu, &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
                init(partitions[socket]->data_handle(), partitions[socket]->mapping().required_span_size(), socket);
            }).join();
        }
    }

    SocketPartitioned(Extents full_extents, int n_parts, const DualSocketConfig& config)
        : SocketPartitioned(full_extents, n_parts, config,
            [](T* p, size_t n, int) { std::memset(p, 0, n * sizeof(T)); }) {}

    SocketPartitioned(const SocketPartitioned&) = delete;
    SocketPartitioned& operator=(const SocketPartitioned&) = delete;
    SocketPartitioned(SocketPartitioned&&) = default;
    SocketPartitioned& operator=(SocketPartitioned&&) = default;

    auto& operator[](int socket) { return *partitions[socket]; }
    const auto& operator[](int socket) const { return *partitions[socket]; }
    auto view(int socket) { return partitions[socket]->view(); }
    auto view(int socket) const { return partitions[socket]->view(); }

private:
    static Extents compute_partition_extents(Extents full, int n_parts) {
        if constexpr (Extents::rank() == 2) {
            size_t dim_size = full.extent(PartitionDimValue);
            size_t part_size = dim_size / n_parts;
            if (PartitionDimValue == 0) {
                return Extents{part_size, full.extent(1)};
            } else {
                return Extents{full.extent(0), part_size};
            }
        } else {
            static_assert(Extents::rank() == 2, "Only 2D tensors supported for partitioning");
        }
    }
};

export template<typename T, typename Extents, typename Layout>
using ColumnPartitioned = SocketPartitioned<T, Extents, Layout, static_cast<int>(PartitionDim::Columns)>;

export template<typename T, typename Extents, typename Layout>
using RowPartitioned = SocketPartitioned<T, Extents, Layout, static_cast<int>(PartitionDim::Rows)>;

export enum class LayoutConversion {
    None,
    ToVNNI
};

export template<typename T, typename Extents, typename Layout>
auto make_socket_replicated_from(auto src_view, const DualSocketConfig& config) {
    return SocketReplicated<T, Extents, Layout>(
        src_view.extents(), config,
        [&](T* dest, size_t count, int) {
            std::memcpy(dest, src_view.data_handle(), count * sizeof(T));
        });
}

export template<PartitionDim Dim, typename T, typename SourceExtents, typename SourceLayout, typename DestLayout = SourceLayout, LayoutConversion Conversion = LayoutConversion::None>
auto make_partitioned_from(auto src_view, int n_parts, const DualSocketConfig& config) {
    constexpr int dim_idx = static_cast<int>(Dim);
    size_t dim0 = src_view.extent(0);
    size_t dim1 = src_view.extent(1);
    size_t part_dim = src_view.extent(dim_idx);
    size_t part_size = part_dim / n_parts;

    return SocketPartitioned<T, SourceExtents, DestLayout, dim_idx>(
        SourceExtents{dim0, dim1}, n_parts, config,
        [&, part_size](T* dest, size_t, int socket) {
            auto src_slice = slice<dim_idx>(src_view, socket * part_size, part_size);
            SourceExtents dest_extents = (dim_idx == 0) ? SourceExtents{part_size, dim1} : SourceExtents{dim0, part_size};
            auto dest_view = std::mdspan<T, SourceExtents, DestLayout>(dest, dest_extents);
            if constexpr (Conversion == LayoutConversion::ToVNNI) {
                convert_to_vnni(src_slice, dest_view);
            } else {
                for (size_t i = 0; i < dest_view.extent(0); i++) {
                    for (size_t j = 0; j < dest_view.extent(1); j++) {
                        dest_view[i, j] = src_slice[i, j];
                    }
                }
            }
        });
}

export template<typename T, typename SourceExtents, typename SourceLayout, typename DestLayout = SourceLayout, LayoutConversion Conversion = LayoutConversion::None>
auto make_column_partitioned_from(auto src_view, int n_parts, const DualSocketConfig& config) {
    return make_partitioned_from<PartitionDim::Columns, T, SourceExtents, SourceLayout, DestLayout, Conversion>(src_view, n_parts, config);
}

export template<typename T, typename SourceExtents, typename SourceLayout, typename DestLayout = SourceLayout, LayoutConversion Conversion = LayoutConversion::None>
auto make_row_partitioned_from(auto src_view, int n_parts, const DualSocketConfig& config) {
    return make_partitioned_from<PartitionDim::Rows, T, SourceExtents, SourceLayout, DestLayout, Conversion>(src_view, n_parts, config);
}

// Parallel all-reduce sum: combines partial results from all sockets into final result
// Each socket produced partial sums for all output elements (row-parallel pattern)
// Result allocated on target_socket with proper NUMA first-touch
// Uses multi-threaded reduction with row-wise work distribution
export template<typename T, typename Extents, typename Layout>
auto all_reduce_sum(SocketReplicated<T, Extents, Layout>& partials,
                    int target_socket, const DualSocketConfig& config) {
    size_t M = partials[0].extent(0);
    size_t N = partials[0].extent(1);

    int node = DualSocketConfig::primary_node_for_socket(target_socket);
    auto result = make_tensor<T>(M, N, NumaNodeAllocator<T>{node});

    int num_threads = config.physical_cores_per_socket;
    std::vector<std::jthread> threads;
    threads.reserve(num_threads);

    for (int tid = 0; tid < num_threads; tid++) {
        threads.emplace_back([&, tid, num_threads] {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(DualSocketConfig::physical_core_id(target_socket, tid), &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

            size_t rows_per_thread = (M + num_threads - 1) / num_threads;
            size_t m_start = tid * rows_per_thread;
            size_t m_end = std::min(M, m_start + rows_per_thread);

            for (size_t i = m_start; i < m_end; i++) {
                for (size_t j = 0; j < N; j++) {
                    T sum = partials[0][i, j];
                    for (int socket = 1; socket < DualSocketConfig::NUM_SOCKETS; socket++) {
                        sum += partials[socket][i, j];
                    }
                    result[i, j] = sum;
                }
            }
        });
    }

    threads.clear();
    return result;
}

// Gathers partitions from all sockets into a single contiguous tensor (non-NUMA-pinned)
export template<typename T, typename Extents, typename Layout>
auto all_gather(ColumnPartitioned<T, Extents, Layout>& partitioned) {
    size_t M = partitioned[0].extent(0);
    size_t N_per_partition = partitioned[0].extent(1);
    size_t N_total = N_per_partition * partitioned.num_partitions;

    Extents full_extents{M, N_total};
    auto result = make_tensor<T>(M, N_total);

    for (int socket = 0; socket < partitioned.num_partitions; socket++) {
        size_t col_offset = socket * N_per_partition;
        auto src = partitioned.view(socket);

        for (size_t i = 0; i < M; i++) {
            T* dest_row = &result[i, col_offset];
            const T* src_row = &src[i, 0];
            size_t bytes_per_row = N_per_partition * sizeof(T);
            std::memcpy(dest_row, src_row, bytes_per_row);
        }
    }

    return result;
}

export enum class ParallelStrategy {
    ColumnParallel,
    RowParallel
};

export enum class IsQuantized {
    No,
    Yes
};

template<ParallelStrategy Strategy, IsQuantized Quantized>
void matmul_amx_parallel_impl(auto& A_container, auto& B_container, auto& C_container,
                               auto params_container, const DualSocketConfig& config) {
    int num_sockets = B_container.num_partitions;
    size_t N = C_container[0].extent(1);
    constexpr size_t N_STEP = 32;
    int max_threads_for_work = std::max(1, static_cast<int>(N / N_STEP));
    int threads_per_socket = std::min(config.physical_cores_per_socket, max_threads_for_work);

    std::vector<std::jthread> threads;
    threads.reserve(num_sockets * threads_per_socket);

    for (int socket = 0; socket < num_sockets; socket++) {
        for (int local_tid = 0; local_tid < threads_per_socket; local_tid++) {
            threads.emplace_back([&, socket, local_tid, threads_per_socket] {
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                int cpu = DualSocketConfig::physical_core_id(socket, local_tid);
                CPU_SET(cpu, &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

                if constexpr (Quantized == IsQuantized::Yes) {
                    matmul_amx_int8_blocked_quantized(
                        A_container.view(socket),
                        B_container.view(socket),
                        C_container.view(socket),
                        params_container->view(socket),
                        local_tid,
                        threads_per_socket
                    );
                } else {
                    matmul_amx_int8_blocked(
                        A_container.view(socket),
                        B_container.view(socket),
                        C_container.view(socket),
                        local_tid,
                        threads_per_socket
                    );
                }
            });
        }
    }
}

// Column-parallel NUMA matmul: C[:, partitions] = A[M,K] @ B[K, partitions]
// A: socket-replicated (one copy per socket)
// B: column-partitioned across sockets (each socket owns different columns)
// C: column-partitioned across sockets (each socket computes different output columns)
// Zero cross-socket communication (outputs independent, full K reduction per socket)
// Uses physical cores only (no hyperthreads) for optimal memory bandwidth
export template<typename TA, typename ExtentsA, typename LayoutA,
                typename TB, typename ExtentsB, typename LayoutB,
                typename TC, typename ExtentsC, typename LayoutC>
void matmul_amx_column_parallel(
    SocketReplicated<TA, ExtentsA, LayoutA>& A_repl,
    ColumnPartitioned<TB, ExtentsB, LayoutB>& B_part,
    ColumnPartitioned<TC, ExtentsC, LayoutC>& C_part,
    const DualSocketConfig& config) {
    matmul_amx_parallel_impl<ParallelStrategy::ColumnParallel, IsQuantized::No>(
        A_repl, B_part, C_part, nullptr, config);
};

export template<typename TA, typename ExtentsA, typename LayoutA,
                typename TB, typename ExtentsB, typename LayoutB,
                typename TC, typename ExtentsC, typename LayoutC,
                typename TParams, typename ExtentsP, typename LayoutP>
void matmul_amx_column_parallel_quantized(
    SocketReplicated<TA, ExtentsA, LayoutA>& A_repl,
    ColumnPartitioned<TB, ExtentsB, LayoutB>& B_part,
    ColumnPartitioned<TC, ExtentsC, LayoutC>& C_part,
    ColumnPartitioned<TParams, ExtentsP, LayoutP>& params_part,
    const DualSocketConfig& config) {
    matmul_amx_parallel_impl<ParallelStrategy::ColumnParallel, IsQuantized::Yes>(
        A_repl, B_part, C_part, &params_part, config);
};

// Row-parallel NUMA matmul: C[M, N] = A[M, partitions] @ B[partitions, N]
// A: column-partitioned across sockets (each socket owns different columns = partial K)
// B: row-partitioned across sockets (each socket owns different rows = partial K)
// C_partials: each socket produces partial sums for all M×N elements
// Requires all_reduce_sum() to combine partials into final result
// Each socket performs partial K reduction → outputs are incomplete
export template<typename TA, typename ExtentsA, typename LayoutA,
                typename TB, typename ExtentsB, typename LayoutB,
                typename TC, typename ExtentsC, typename LayoutC>
void matmul_amx_row_parallel(
    ColumnPartitioned<TA, ExtentsA, LayoutA>& A_part,
    RowPartitioned<TB, ExtentsB, LayoutB>& B_part,
    SocketReplicated<TC, ExtentsC, LayoutC>& C_partials,
    const DualSocketConfig& config) {
    matmul_amx_parallel_impl<ParallelStrategy::RowParallel, IsQuantized::No>(
        A_part, B_part, C_partials, nullptr, config);
};

export template<typename TA, typename ExtentsA, typename LayoutA,
                typename TB, typename ExtentsB, typename LayoutB,
                typename TC, typename ExtentsC, typename LayoutC,
                typename TParams, typename ExtentsP, typename LayoutP>
void matmul_amx_row_parallel_quantized(
    ColumnPartitioned<TA, ExtentsA, LayoutA>& A_part,
    RowPartitioned<TB, ExtentsB, LayoutB>& B_part,
    SocketReplicated<TC, ExtentsC, LayoutC>& C_partials,
    SocketReplicated<TParams, ExtentsP, LayoutP>& params_partials,
    const DualSocketConfig& config) {
    matmul_amx_parallel_impl<ParallelStrategy::RowParallel, IsQuantized::Yes>(
        A_part, B_part, C_partials, &params_partials, config);
};
