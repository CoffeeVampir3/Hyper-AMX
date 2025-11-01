module;
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <memory>
#include <optional>
#include <print>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>
#include <sched.h>
#include <numa.h>
#include <numaif.h>
export module hyperamx.numa;

import hyperamx.tensor;
import hyperamx.avx512;

export namespace numa {

using namespace tensor;
using namespace avx512;

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

template<typename Alloc>
Alloc make_node_allocator(int node) {
    if constexpr (requires { Alloc{node}; }) {
        return Alloc{node};
    } else if constexpr (std::is_default_constructible_v<Alloc>) {
        return Alloc{};
    } else {
        static_assert(sizeof(Alloc) == 0, "Allocator must be constructible with NUMA node id");
    }
}

template<typename TensorType>
auto allocate_on_node(const typename TensorType::extents_type& extents, int node) {
    if constexpr (tensor::QuantTensor<TensorType>) {
        using Extents = typename TensorType::extents_type;
        using Layout = typename TensorType::layout_type;
        using DataTensor = typename TensorType::data_type;
        using ScaleTensor = typename TensorType::scale_type;
        using DataValue = typename DataTensor::element_type;
        using ScaleValue = typename ScaleTensor::element_type;
        constexpr size_t TileRows = TensorType::TILE_ROWS;
        constexpr size_t TileCols = TensorType::TILE_COLS;
        if constexpr (DataTensor::has_allocator && ScaleTensor::has_allocator) {
            using DataAlloc = typename DataTensor::allocator_type;
            using ScaleAlloc = typename ScaleTensor::allocator_type;
            return make_quantized_tensor<DataValue, Extents, Layout, ScaleValue, TileRows, TileCols>(
                extents,
                make_node_allocator<DataAlloc>(node),
                make_node_allocator<ScaleAlloc>(node));
        } else {
            return make_quantized_tensor<DataValue, Extents, Layout, ScaleValue, TileRows, TileCols>(extents);
        }
    } else {
        static_assert(TensorType::has_allocator, "TensorType must own storage for NUMA allocation");
        using Alloc = typename TensorType::allocator_type;
        return TensorType(extents, make_node_allocator<Alloc>(node));
    }
}

template<typename TensorType>
struct Replicated {
    static constexpr int NUM_SOCKETS = DualSocketConfig::NUM_SOCKETS;
    using extents_type = typename TensorType::extents_type;

    std::array<std::optional<TensorType>, NUM_SOCKETS> replicas;

    Replicated(const extents_type& extents, const DualSocketConfig& config) {
        for (int socket = 0; socket < NUM_SOCKETS; socket++) {
            int node = DualSocketConfig::primary_node_for_socket(socket);
            replicas[socket].emplace(allocate_on_node<TensorType>(extents, node));
        }
    }

    Replicated(TensorType& source, const DualSocketConfig& config) {
        auto extents = source.extents();
        for (int socket = 0; socket < NUM_SOCKETS; socket++) {
            int node = DualSocketConfig::primary_node_for_socket(socket);
            replicas[socket].emplace(allocate_on_node<TensorType>(extents, node));
            copy_tensor(source, replicas[socket].value());
        }
    }

    Replicated(TensorType&& source, int source_socket, const DualSocketConfig& config) {
        replicas[source_socket].emplace(std::move(source));
        auto extents = replicas[source_socket]->extents();
        for (int socket = 0; socket < NUM_SOCKETS; socket++) {
            if (socket == source_socket) continue;
            int node = DualSocketConfig::primary_node_for_socket(socket);
            replicas[socket].emplace(allocate_on_node<TensorType>(extents, node));
            copy_tensor(replicas[source_socket].value(), replicas[socket].value());
        }
    }

    Replicated(const Replicated&) = delete;
    Replicated& operator=(const Replicated&) = delete;
    Replicated(Replicated&&) = default;
    Replicated& operator=(Replicated&&) = default;

    auto& operator[](int socket) { return *replicas[socket]; }
    const auto& operator[](int socket) const { return *replicas[socket]; }
    auto span(int socket) { return replicas[socket]->span(); }
    auto span(int socket) const { return replicas[socket]->span(); }
};

template<typename TensorType, PartitionDim Dim>
struct Partitioned {
    static constexpr int MAX_SOCKETS = DualSocketConfig::NUM_SOCKETS;
    static constexpr int partition_dim = static_cast<int>(Dim);
    using extents_type = typename TensorType::extents_type;

    std::array<std::optional<TensorType>, MAX_SOCKETS> partitions;
    int num_partitions;

    Partitioned(const extents_type& extents, int n_parts, const DualSocketConfig& config)
        : num_partitions(n_parts)
    {
        auto part_extents = compute_partition_extents(extents, n_parts);
        for (int socket = 0; socket < n_parts; socket++) {
            int node = DualSocketConfig::primary_node_for_socket(socket);
            partitions[socket].emplace(allocate_on_node<TensorType>(part_extents, node));
        }
    }

    Partitioned(TensorType& source, int n_parts, const DualSocketConfig& config)
        : num_partitions(n_parts)
    {
        size_t dim_size = source.extent(partition_dim);
        size_t part_size = dim_size / n_parts;

        auto src_extents = source.extents();
        auto part_extents = compute_partition_extents(src_extents, n_parts);

        for (int socket = 0; socket < n_parts; socket++) {
            int node = DualSocketConfig::primary_node_for_socket(socket);
            partitions[socket].emplace(allocate_on_node<TensorType>(part_extents, node));
            copy_tensor_slice(source, partitions[socket].value(), partition_dim, socket * part_size, part_size);
        }
    }

    Partitioned(const Partitioned&) = delete;
    Partitioned& operator=(const Partitioned&) = delete;
    Partitioned(Partitioned&&) = default;
    Partitioned& operator=(Partitioned&&) = default;

    auto& operator[](int socket) { return *partitions[socket]; }
    const auto& operator[](int socket) const { return *partitions[socket]; }
    auto span(int socket) { return partitions[socket]->span(); }
    auto span(int socket) const { return partitions[socket]->span(); }

private:
    static extents_type compute_partition_extents(extents_type full, int n_parts) {
        if constexpr (extents_type::rank() == 2) {
            size_t dim_size = full.extent(partition_dim);
            size_t part_size = dim_size / n_parts;
            if constexpr (partition_dim == 0) {
                return extents_type{part_size, full.extent(1)};
            } else {
                return extents_type{full.extent(0), part_size};
            }
        } else {
            static_assert(extents_type::rank() == 2, "Only 2D tensors supported for partitioning");
        }
    }
};

template<typename TensorType>
using ColumnPartitioned = Partitioned<TensorType, PartitionDim::Columns>;

template<typename TensorType>
using RowPartitioned = Partitioned<TensorType, PartitionDim::Rows>;

template<typename TensorType>
void all_reduce_sum(Replicated<TensorType>& partials,
                    TensorType& result,
                    int target_socket,
                    const DualSocketConfig& config)
{
    using Value = typename TensorType::element_type;
    size_t M = partials[0].extent(0);
    size_t N = partials[0].extent(1);

    int num_threads = config.physical_cores_per_socket;
    std::vector<std::jthread> threads;
    threads.reserve(num_threads);

    for (int tid = 0; tid < num_threads; tid++) {
        threads.emplace_back([&, tid, num_threads] {
            pin_to_socket(target_socket, tid);

            size_t rows_per_thread = (M + num_threads - 1) / num_threads;
            size_t m_start = tid * rows_per_thread;
            size_t m_end = std::min(M, m_start + rows_per_thread);

            auto result_view = result.span();
            auto first = partials[0].span();
            for (size_t i = m_start; i < m_end; i++) {
                std::memcpy(&result_view[i, 0], &first[i, 0], N * sizeof(Value));
                for (int socket = 1; socket < DualSocketConfig::NUM_SOCKETS; socket++) {
                    auto partial_view = partials[socket].span();
                    add(&result_view[i, 0], &partial_view[i, 0], &result_view[i, 0], N);
                }
            }
        });
    }

    threads.clear();
}

template<typename TensorType>
void all_gather(ColumnPartitioned<TensorType>& partitioned,
                TensorType& result)
{
    size_t M = partitioned[0].extent(0);
    size_t N_per_partition = partitioned[0].extent(1);

    using Value = typename TensorType::element_type;
    auto result_view = result.span();

    for (int socket = 0; socket < partitioned.num_partitions; socket++) {
        size_t col_offset = socket * N_per_partition;
        auto src = partitioned.span(socket);

        for (size_t i = 0; i < M; i++) {
            Value* dest_row = &result_view[i, col_offset];
            const Value* src_row = &src[i, 0];
            std::memcpy(dest_row, src_row, N_per_partition * sizeof(Value));
        }
    }
}

} // namespace numa
