module;
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <array>
#include <optional>
#include <stdexcept>
#include <format>
#include <print>
#include <thread>
#include <vector>
#include <memory>
#include <sched.h>
#include <mdspan>
#include <numa.h>
#include <numaif.h>
export module modernnuma;
import moderntensor;
import modernlayout;
import modernmatmul;

export namespace Numa {

struct DualSocketConfig {
    static constexpr int NUM_SOCKETS = 2;
    static constexpr int NUM_NODES = 4;
    static constexpr int NODES_PER_SOCKET = 2;
    static constexpr int CORES_PER_NODE = 8;

    int physical_cores_per_socket;
    int total_cpus;

    static constexpr int primary_node_for_socket(int socket) { return socket * NODES_PER_SOCKET; }
    static constexpr int socket_for_node(int node) { return node / NODES_PER_SOCKET; }

    static constexpr int physical_core_id(int socket, int local_tid) {
        int node_within_socket = local_tid / CORES_PER_NODE;
        int core_within_node = local_tid % CORES_PER_NODE;
        int base_node = socket * NODES_PER_SOCKET;
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

enum class PartitionDim {
    Rows = 0,
    Columns = 1
};

template<typename T>
struct NumaAllocator {
    int node;
    T* allocate(size_t count) const {
        return static_cast<T*>(numa_alloc_onnode(count * sizeof(T), node));
    }
    void deallocate(T* ptr, size_t count) const {
        numa_free(ptr, count * sizeof(T));
    }
};

inline void pin_to_socket(int socket, int local_tid = 0) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int cpu = DualSocketConfig::physical_core_id(socket, local_tid);
    CPU_SET(cpu, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

template<typename Fn>
void execute_on_socket(int socket, int local_tid, Fn&& fn) {
    std::jthread([&, socket, local_tid] {
        pin_to_socket(socket, local_tid);
        fn();
    }).join();
}

template<typename T, typename Extents, typename Layout>
struct Replicated {
    static constexpr int NUM_SOCKETS = DualSocketConfig::NUM_SOCKETS;
    using storage_type = Tensor<T, Extents, Layout>;
    std::array<std::optional<storage_type>, NUM_SOCKETS> replicas;

    Replicated(Extents extents, const DualSocketConfig& config) {
        for (int socket = 0; socket < NUM_SOCKETS; socket++) {
            int node = DualSocketConfig::primary_node_for_socket(socket);
            replicas[socket].emplace(extents, NumaAllocator<T>{node});
        }
    }

    template<TensorStorage Source>
    Replicated(const Source& source, const DualSocketConfig& config) {
        auto extents = source.view().extents();
        for (int socket = 0; socket < NUM_SOCKETS; socket++) {
            int node = DualSocketConfig::primary_node_for_socket(socket);
            replicas[socket].emplace(extents, NumaAllocator<T>{node});
            execute_on_socket(socket, 0, [&, socket] {
                source.copy_to(replicas[socket].value());
            });
        }
    }

    //Optimization specifically for all reduce sum case where we pin to a socket. We can avoid making a copy
    //for the socket we just all-reduced to by specifying the source socket. Ensure to use this in tandem with
    //all reduce sum.
    Replicated(Tensor<T, Extents, Layout>&& source, int source_socket, const DualSocketConfig& config) {
        replicas[source_socket].emplace(std::move(source));

        auto extents = replicas[source_socket]->view().extents();
        for (int socket = 0; socket < NUM_SOCKETS; socket++) {
            if (socket == source_socket) continue;

            int node = DualSocketConfig::primary_node_for_socket(socket);
            replicas[socket].emplace(extents, NumaAllocator<T>{node});
            execute_on_socket(socket, 0, [&, socket] {
                replicas[source_socket]->copy_to(replicas[socket].value());
            });
        }
    }

    Replicated(const Replicated&) = delete;
    Replicated& operator=(const Replicated&) = delete;
    Replicated(Replicated&&) = default;
    Replicated& operator=(Replicated&&) = default;

    auto& operator[](int socket) { return *replicas[socket]; }
    const auto& operator[](int socket) const { return *replicas[socket]; }
    auto view(int socket) { return replicas[socket]->view(); }
    auto view(int socket) const { return replicas[socket]->view(); }
};

template<typename T, typename Extents, typename Layout, PartitionDim Dim>
struct Partitioned {
    static constexpr int MAX_SOCKETS = DualSocketConfig::NUM_SOCKETS;
    static constexpr int partition_dim = static_cast<int>(Dim);
    using storage_type = Tensor<T, Extents, Layout>;
    std::array<std::optional<storage_type>, MAX_SOCKETS> partitions;
    int num_partitions;

    Partitioned(Extents extents, int n_parts, const DualSocketConfig& config)
        : num_partitions(n_parts) {
        auto part_extents = compute_partition_extents(extents, n_parts);
        for (int socket = 0; socket < n_parts; socket++) {
            int node = DualSocketConfig::primary_node_for_socket(socket);
            partitions[socket].emplace(part_extents, NumaAllocator<T>{node});
        }
    }

    template<TensorStorage Source>
    Partitioned(const Source& source, int n_parts, const DualSocketConfig& config)
        : num_partitions(n_parts) {
        size_t dim_size = source.extent(partition_dim);
        size_t part_size = dim_size / n_parts;

        auto src_extents = source.view().extents();
        auto part_extents = compute_partition_extents(src_extents, n_parts);

        for (int socket = 0; socket < n_parts; socket++) {
            int node = DualSocketConfig::primary_node_for_socket(socket);
            partitions[socket].emplace(part_extents, NumaAllocator<T>{node});
            execute_on_socket(socket, 0, [&, socket, part_size] {
                source.copy_slice_to(partitions[socket].value(), partition_dim, socket * part_size, part_size);
            });
        }
    }

    Partitioned(const Partitioned&) = delete;
    Partitioned& operator=(const Partitioned&) = delete;
    Partitioned(Partitioned&&) = default;
    Partitioned& operator=(Partitioned&&) = default;

    auto& operator[](int socket) { return *partitions[socket]; }
    const auto& operator[](int socket) const { return *partitions[socket]; }
    auto view(int socket) { return partitions[socket]->view(); }
    auto view(int socket) const { return partitions[socket]->view(); }

private:
    static Extents compute_partition_extents(Extents full, int n_parts) {
        if constexpr (Extents::rank() == 2) {
            size_t dim_size = full.extent(partition_dim);
            size_t part_size = dim_size / n_parts;
            if constexpr (partition_dim == 0) {
                return Extents{part_size, full.extent(1)};
            } else {
                return Extents{full.extent(0), part_size};
            }
        } else {
            static_assert(Extents::rank() == 2, "Only 2D tensors supported for partitioning");
        }
    }
};

template<typename T, typename Extents, typename Layout>
using ColumnPartitioned = Partitioned<T, Extents, Layout, PartitionDim::Columns>;

template<typename T, typename Extents, typename Layout>
using RowPartitioned = Partitioned<T, Extents, Layout, PartitionDim::Rows>;

template<typename T, typename Extents, typename Layout>
auto all_reduce_sum(Replicated<T, Extents, Layout>& partials, int target_socket, const DualSocketConfig& config) {
    size_t M = partials[0].extent(0);
    size_t N = partials[0].extent(1);

    int node = DualSocketConfig::primary_node_for_socket(target_socket);
    Tensor<T, Extents, Layout> result(Extents{M, N}, NumaAllocator<T>{node});

    int num_threads = config.physical_cores_per_socket;
    std::vector<std::jthread> threads;
    threads.reserve(num_threads);

    for (int tid = 0; tid < num_threads; tid++) {
        threads.emplace_back([&, tid, num_threads] {
            pin_to_socket(target_socket, tid);

            size_t rows_per_thread = (M + num_threads - 1) / num_threads;
            size_t m_start = tid * rows_per_thread;
            size_t m_end = std::min(M, m_start + rows_per_thread);

            auto result_view = result.view();
            for (size_t i = m_start; i < m_end; i++) {
                for (size_t j = 0; j < N; j++) {
                    auto sum = partials[0].view()[i, j];
                    for (int socket = 1; socket < DualSocketConfig::NUM_SOCKETS; socket++) {
                        sum += partials[socket].view()[i, j];
                    }
                    result_view[i, j] = sum;
                }
            }
        });
    }

    threads.clear();
    return result;
}

template<typename T, typename Extents, typename Layout>
auto all_gather(ColumnPartitioned<T, Extents, Layout>& partitioned) {
    size_t M = partitioned[0].extent(0);
    size_t N_per_partition = partitioned[0].extent(1);
    size_t N_total = N_per_partition * partitioned.num_partitions;

    Tensor<T, Extents, Layout> result(Extents{M, N_total});
    auto result_view = result.view();

    for (int socket = 0; socket < partitioned.num_partitions; socket++) {
        size_t col_offset = socket * N_per_partition;
        auto src = partitioned.view(socket);

        for (size_t i = 0; i < M; i++) {
            T* dest_row = &result_view[i, col_offset];
            const T* src_row = &src[i, 0];
            std::memcpy(dest_row, src_row, N_per_partition * sizeof(T));
        }
    }

    return result;
}

void matmul_amx_parallel_impl(auto& A_container, auto& B_container, auto& C_container,
                              const DualSocketConfig& config)
{
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
                pin_to_socket(socket, local_tid);
                matmul_amx_int8_blocked(
                    A_container.view(socket),
                    B_container.view(socket),
                    C_container.view(socket),
                    local_tid,
                    threads_per_socket
                );
            });
        }
    }
}

template<typename TA, typename ExtentsA, typename LayoutA,
         typename TB, typename ExtentsB, typename LayoutB,
         typename TC, typename ExtentsC, typename LayoutC>
void matmul_amx_column_parallel(
    Replicated<TA, ExtentsA, LayoutA>& A_repl,
    ColumnPartitioned<TB, ExtentsB, LayoutB>& B_part,
    ColumnPartitioned<TC, ExtentsC, LayoutC>& C_part,
    const DualSocketConfig& config)
{
    matmul_amx_parallel_impl(A_repl, B_part, C_part, config);
}

template<typename TA, typename ExtentsA, typename LayoutA,
         typename TB, typename ExtentsB, typename LayoutB,
         typename TC, typename ExtentsC, typename LayoutC>
void matmul_amx_row_parallel(
    ColumnPartitioned<TA, ExtentsA, LayoutA>& A_part,
    RowPartitioned<TB, ExtentsB, LayoutB>& B_part,
    Replicated<TC, ExtentsC, LayoutC>& C_partials,
    const DualSocketConfig& config)
{
    matmul_amx_parallel_impl(A_part, B_part, C_partials, config);
}

}
