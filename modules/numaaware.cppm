module;
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
                init(replicas[socket].data(), replicas[socket].mapping().required_span_size(), socket);
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
            init(replicas[other_socket].data(), replicas[other_socket].mapping().required_span_size(), other_socket);
        }).join();
    }

    SocketReplicated(Tensor<T, Extents, Layout>&& source, int source_socket, const DualSocketConfig& config)
        : SocketReplicated(std::move(source), source_socket, config,
            [this, source_socket](T* dest, size_t count, int) {
                std::memcpy(dest, replicas[source_socket].data(), count * sizeof(T));
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
                init(partitions[socket]->data(), partitions[socket]->mapping().required_span_size(), socket);
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

    int num_sockets = B_part.num_partitions;

    // Matmul kernel processes N_STEP=32 columns at a time, so limit threads accordingly
    size_t N_per_partition = C_part[0].extent(1);
    constexpr size_t N_STEP = 32;
    int max_threads_for_work = std::max(1, static_cast<int>(N_per_partition / N_STEP));
    int threads_per_socket = std::min(config.physical_cores_per_socket, max_threads_for_work);

    int total_threads = num_sockets * threads_per_socket;

    std::vector<std::jthread> threads;
    threads.reserve(total_threads);

    for (int socket = 0; socket < num_sockets; socket++) {
        for (int local_tid = 0; local_tid < threads_per_socket; local_tid++) {
            threads.emplace_back([&, socket, local_tid, threads_per_socket] {
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                // Pin to physical core (skips hyperthreads)
                int cpu = DualSocketConfig::physical_core_id(socket, local_tid);
                CPU_SET(cpu, &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

                matmul_amx_int8_blocked(
                    A_repl.view(socket),
                    B_part.view(socket),
                    C_part.view(socket),
                    local_tid,
                    threads_per_socket
                );
            });
        }
    }
};

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
            for (size_t j = 0; j < N_per_partition; j++) {
                result[i, col_offset + j] = src[i, j];
            }
        }
    }

    return result;
}

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

    int num_sockets = B_part.num_partitions;

    // Matmul kernel processes N_STEP=32 columns at a time, so limit threads accordingly
    size_t N_per_partition = C_part[0].extent(1);
    constexpr size_t N_STEP = 32;
    int max_threads_for_work = std::max(1, static_cast<int>(N_per_partition / N_STEP));
    int threads_per_socket = std::min(config.physical_cores_per_socket, max_threads_for_work);

    int total_threads = num_sockets * threads_per_socket;

    std::vector<std::jthread> threads;
    threads.reserve(total_threads);

    for (int socket = 0; socket < num_sockets; socket++) {
        for (int local_tid = 0; local_tid < threads_per_socket; local_tid++) {
            threads.emplace_back([&, socket, local_tid, threads_per_socket] {
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                int cpu = DualSocketConfig::physical_core_id(socket, local_tid);
                CPU_SET(cpu, &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

                matmul_amx_int8_blocked_quantized(
                    A_repl.view(socket),
                    B_part.view(socket),
                    C_part.view(socket),
                    params_part.view(socket),
                    local_tid,
                    threads_per_socket
                );
            });
        }
    }
};
