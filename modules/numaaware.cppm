module;
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <array>
#include <memory>
#include <optional>
#include <stdexcept>
#include <print>
#include <thread>
#include <mdspan>
#include <sched.h>
#include <numa.h>
#include <numaif.h>
export module numaaware;
import tensor;
import matmul;

// Dual-socket SNC-2 configuration and primitives
export struct DualSocketConfig {
    static constexpr int NUM_SOCKETS = 2;
    static constexpr int NUM_NODES = 4;
    static constexpr int NODES_PER_SOCKET = 2;

    int cpus_per_node;
    std::array<cpu_set_t, NUM_NODES> node_cpusets;
    std::array<int, NUM_NODES> node_ids;

    static constexpr int socket_for_node(int node) { return node / 2; }
    static constexpr int primary_node_for_socket(int socket) { return socket * 2; }

    static DualSocketConfig discover() {
        if (numa_available() < 0) {
            throw std::runtime_error("NUMA not available on this system");
        }

        int num_nodes = numa_num_configured_nodes();
        if (num_nodes != NUM_NODES) {
            throw std::runtime_error(std::format("Expected {} NUMA nodes (dual-socket SNC-2), found {}", NUM_NODES, num_nodes));
        }

        DualSocketConfig config;

        // Query CPU masks for each node
        for (int node = 0; node < NUM_NODES; node++) {
            struct bitmask* mask = numa_allocate_cpumask();
            numa_node_to_cpus(node, mask);

            CPU_ZERO(&config.node_cpusets[node]);
            for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
                if (numa_bitmask_isbitset(mask, cpu)) {
                    CPU_SET(cpu, &config.node_cpusets[node]);
                }
            }
            numa_free_cpumask(mask);
            config.node_ids[node] = node;
        }

        config.cpus_per_node = CPU_COUNT(&config.node_cpusets[0]);

        // Validate SNC-2 topology via distance matrix
        for (int i = 0; i < NUM_NODES; i++) {
            for (int j = 0; j < NUM_NODES; j++) {
                int distance = numa_distance(i, j);
                bool same_socket = (socket_for_node(i) == socket_for_node(j));

                if (same_socket && distance > 20) {
                    std::println(stderr, "Warning: Unexpected NUMA distance within socket ({}->{}: {})", i, j, distance);
                }
                if (!same_socket && distance < 20) {
                    std::println(stderr, "Warning: Unexpected NUMA distance across sockets ({}->{}: {})", i, j, distance);
                }
            }
        }

        std::println("NUMA topology discovered: {} nodes, {} CPUs per node", NUM_NODES, config.cpus_per_node);
        return config;
    }

    void pin_to_node(int node) const {
        if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &node_cpusets[node]) != 0) {
            std::println(stderr, "Warning: Failed to pin thread to node {}", node);
        }
    }

    void pin_to_socket(int socket) const {
        int primary_node = primary_node_for_socket(socket);
        pin_to_node(primary_node);
    }
};

// NUMA memory allocator helper
template<typename T>
struct NumaAllocator {
    static T* alloc_on_node(size_t count, int node) {
        void* ptr = numa_alloc_onnode(count * sizeof(T), node);
        if (!ptr) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    static void free(T* ptr, size_t count) {
        numa_free(ptr, count * sizeof(T));
    }
};

// Socket-replicated data: one copy per socket (shared within socket via L3)
export template<typename T>
struct SocketReplicated {
    std::array<T*, DualSocketConfig::NUM_SOCKETS> socket_data;
    size_t count;

    SocketReplicated(size_t n, const T* source, const DualSocketConfig& config) : count(n) {
        // Allocate on primary node of each socket
        for (int socket = 0; socket < DualSocketConfig::NUM_SOCKETS; socket++) {
            int primary_node = DualSocketConfig::primary_node_for_socket(socket);
            socket_data[socket] = NumaAllocator<T>::alloc_on_node(count, primary_node);

            // First-touch copy on correct socket
            std::thread([&, socket] {
                config.pin_to_socket(socket);
                std::memcpy(socket_data[socket], source, count * sizeof(T));
            }).join();
        }
    }

    ~SocketReplicated() {
        for (int socket = 0; socket < DualSocketConfig::NUM_SOCKETS; socket++) {
            if (socket_data[socket]) {
                NumaAllocator<T>::free(socket_data[socket], count);
            }
        }
    }

    SocketReplicated(const SocketReplicated&) = delete;
    SocketReplicated& operator=(const SocketReplicated&) = delete;
    SocketReplicated(SocketReplicated&&) = default;
    SocketReplicated& operator=(SocketReplicated&&) = default;

    T* get_for_socket(int socket) { return socket_data[socket]; }
    const T* get_for_socket(int socket) const { return socket_data[socket]; }
};

// Replicated VNNI data: full B_vnni on each socket
export template<typename VNNIView>
struct ReplicatedVNNI {
    using T = typename VNNIView::element_type;
    using Layout = typename VNNIView::layout_type;
    using Extents = typename VNNIView::extents_type;
    using Mapping = typename VNNIView::mapping_type;

    std::array<T*, DualSocketConfig::NUM_SOCKETS> socket_data;
    Mapping map;
    size_t total_size;

    ReplicatedVNNI(VNNIView source, const DualSocketConfig& config)
        : map(source.mapping()), total_size(map.required_span_size()) {

        for (int socket = 0; socket < DualSocketConfig::NUM_SOCKETS; socket++) {
            int primary_node = DualSocketConfig::primary_node_for_socket(socket);
            socket_data[socket] = NumaAllocator<T>::alloc_on_node(total_size, primary_node);

            std::thread([&, socket] {
                config.pin_to_socket(socket);
                std::memcpy(socket_data[socket], source.data_handle(), total_size * sizeof(T));
            }).join();
        }
    }

    ~ReplicatedVNNI() {
        for (int socket = 0; socket < DualSocketConfig::NUM_SOCKETS; socket++) {
            if (socket_data[socket]) {
                NumaAllocator<T>::free(socket_data[socket], total_size);
            }
        }
    }

    ReplicatedVNNI(const ReplicatedVNNI&) = delete;
    ReplicatedVNNI& operator=(const ReplicatedVNNI&) = delete;
    ReplicatedVNNI(ReplicatedVNNI&&) = default;
    ReplicatedVNNI& operator=(ReplicatedVNNI&&) = default;

    auto get_for_socket(int socket) {
        return std::mdspan<T, Extents, Layout>(socket_data[socket], map);
    }
};

// NUMA-aware matmul: replicate A and B_vnni, partition threads across sockets
export template<typename VNNIView, MdspanLike TC>
void matmul_amx_numa(
    SocketReplicated<std::int8_t>& A_repl,
    ReplicatedVNNI<VNNIView>& B_vnni_repl,
    TC C,
    const DualSocketConfig& config) {

    const size_t M = C.extent(0);
    const size_t K = A_repl.count / M;

    // Use only physical cores (not hyperthreads) to avoid bandwidth contention
    const int threads_per_socket = (config.cpus_per_node * DualSocketConfig::NODES_PER_SOCKET) / 2;
    const int total_threads = threads_per_socket * DualSocketConfig::NUM_SOCKETS;

    std::vector<std::jthread> threads;
    threads.reserve(total_threads);

    for (int tid = 0; tid < total_threads; tid++) {
        threads.emplace_back([&, tid] {
            int socket = tid / threads_per_socket;

            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(tid, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

            auto A_local = A_repl.get_for_socket(socket);
            auto B_vnni_local = B_vnni_repl.get_for_socket(socket);

            using Extents = std::dextents<size_t, 2>;
            auto A_view = std::mdspan<std::int8_t, Extents, std::layout_right>(A_local, Extents{M, K});

            // Use global thread ID so threads don't overlap on columns
            matmul_amx_int8_blocked(A_view, B_vnni_local, C, tid, total_threads);
        });
    }
}
