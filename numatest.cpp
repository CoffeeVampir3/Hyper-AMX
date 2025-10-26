#include <print>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstddef>
#include <string_view>
#include <sys/syscall.h>
#include <unistd.h>
import tensor;
import matmul;
import numaaware;

constexpr auto ARCH_GET_XCOMP_PERM = 0x1022;
constexpr auto ARCH_REQ_XCOMP_PERM = 0x1023;
constexpr auto XFEATURE_XTILEDATA = 18;

bool request_amx_permission() {
    unsigned long bitmask = 0;
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        std::println(stderr, "Failed to enable AMX");
        return false;
    }
    if (syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask)) {
        std::println(stderr, "Failed to get AMX permissions");
        return false;
    }
    return bitmask & (1 << XFEATURE_XTILEDATA);
}

template<typename T, typename Extents, typename Layout>
void fill(Tensor<T, Extents, Layout>& t, T value) {
    for (std::size_t i = 0; i < t.extent(0); ++i)
        for (std::size_t j = 0; j < t.extent(1); ++j)
            t[i, j] = value;
}

template<typename T, typename Extents, typename Layout, typename Fn>
void fill(Tensor<T, Extents, Layout>& t, Fn&& fn) {
    for (std::size_t i = 0; i < t.extent(0); ++i)
        for (std::size_t j = 0; j < t.extent(1); ++j)
            t[i, j] = fn(i, j);
}

template<typename Fn, typename T, typename Extents, typename Layout>
bool check_result(const Tensor<T, Extents, Layout>& c, Fn&& expected_fn, std::string_view test_name) {
    for (std::size_t i = 0; i < c.extent(0); ++i) {
        for (std::size_t j = 0; j < c.extent(1); ++j) {
            auto expected = expected_fn(i, j);
            if (c[i, j] != expected) {
                std::println("  FAIL [{}]: C[{},{}] = {} (expected {})", test_name, i, j, c[i, j], expected);
                return false;
            }
        }
    }
    std::println("  PASS [{}]", test_name);
    return true;
}


void benchmark_numa_vs_baseline(const DualSocketConfig& config) {
    constexpr int M = 4096, N = 4096, K = 4096;
    constexpr int NUM_ITERS = 100;

    std::println("\n=== NUMA vs Baseline Benchmark: {}x{}x{}, {} iterations ===", M, K, N, NUM_ITERS);

    using ATensor = decltype(make_tensor<std::int8_t>(M, K));
    using BTensor = decltype(make_tensor<std::int8_t>(K, N));
    using BVNNITensor = decltype(convert_to_vnni(std::declval<BTensor>().view()));
    using CTensor = decltype(make_tensor<std::int32_t>(M, N));

    std::println("Allocating {} unique tensor sets ({:.2f} GB)...", NUM_ITERS,
                 NUM_ITERS * (16.0 + 16.0 + 64.0) / 1024.0);

    std::vector<ATensor> a_buffers;
    std::vector<BTensor> b_buffers;
    std::vector<BVNNITensor> b_vnni_buffers;
    std::vector<CTensor> c_buffers_baseline;
    std::vector<CTensor> c_buffers_numa;

    // NUMA setup objects (created once per buffer)
    std::vector<SocketReplicated<std::int8_t>> a_repl_buffers;
    std::vector<ReplicatedVNNI<decltype(b_vnni_buffers[0].view())>> b_vnni_repl_buffers;

    a_buffers.reserve(NUM_ITERS);
    b_buffers.reserve(NUM_ITERS);
    b_vnni_buffers.reserve(NUM_ITERS);
    c_buffers_baseline.reserve(NUM_ITERS);
    c_buffers_numa.reserve(NUM_ITERS);
    a_repl_buffers.reserve(NUM_ITERS);
    b_vnni_repl_buffers.reserve(NUM_ITERS);

    std::int8_t a_val = 3, b_val = 5;
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        a_buffers.push_back(make_tensor<std::int8_t>(M, K));
        b_buffers.push_back(make_tensor<std::int8_t>(K, N));
        c_buffers_baseline.push_back(make_tensor<std::int32_t>(M, N));
        c_buffers_numa.push_back(make_tensor<std::int32_t>(M, N));

        fill(a_buffers[iter], [&](std::size_t, std::size_t) { return a_val++; });
        fill(b_buffers[iter], [&](std::size_t, std::size_t) { return b_val++; });
        fill(c_buffers_baseline[iter], std::int32_t{0});
        fill(c_buffers_numa[iter], std::int32_t{0});

        // Convert to VNNI once
        b_vnni_buffers.push_back(convert_to_vnni(b_buffers[iter].view()));

        // Create NUMA setup objects once per buffer
        a_repl_buffers.emplace_back(M * K, a_buffers[iter].data(), config);
        b_vnni_repl_buffers.emplace_back(b_vnni_buffers[iter].view(), config);
    }

    // Baseline: existing multi-threaded implementation
    std::println("\n--- Baseline (matmul_amx_int8_blocked_mt) ---");
    auto baseline_start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        matmul_amx_int8_blocked_mt(a_buffers[iter].view(),
                                   b_vnni_buffers[iter].view(),
                                   c_buffers_baseline[iter].view());
    }
    auto baseline_end = std::chrono::high_resolution_clock::now();

    auto baseline_duration = std::chrono::duration_cast<std::chrono::microseconds>(baseline_end - baseline_start);
    double baseline_avg_ms = baseline_duration.count() / static_cast<double>(NUM_ITERS) / 1000.0;
    double baseline_gflops = (2.0 * M * N * K) / (baseline_duration.count() / static_cast<double>(NUM_ITERS)) / 1000.0;

    std::println("Average: {:.2f} ms/iter, Throughput: {:.2f} GFLOPS", baseline_avg_ms, baseline_gflops);

    // NUMA-aware implementation
    std::println("\n--- NUMA-Aware (matmul_amx_numa) ---");
    auto numa_start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        matmul_amx_numa(a_repl_buffers[iter],
                        b_vnni_repl_buffers[iter],
                        c_buffers_numa[iter].view(),
                        config);
    }
    auto numa_end = std::chrono::high_resolution_clock::now();

    auto numa_duration = std::chrono::duration_cast<std::chrono::microseconds>(numa_end - numa_start);
    double numa_avg_ms = numa_duration.count() / static_cast<double>(NUM_ITERS) / 1000.0;
    double numa_gflops = (2.0 * M * N * K) / (numa_duration.count() / static_cast<double>(NUM_ITERS)) / 1000.0;

    std::println("Average: {:.2f} ms/iter, Throughput: {:.2f} GFLOPS", numa_avg_ms, numa_gflops);

    // Verify correctness (spot check last iteration)
    std::println("\n--- Verifying NUMA correctness against baseline ---");
    bool results_match = true;
    const auto& c_base = c_buffers_baseline[NUM_ITERS - 1];
    const auto& c_numa = c_buffers_numa[NUM_ITERS - 1];

    for (std::size_t i = 0; i < M && results_match; ++i) {
        for (std::size_t j = 0; j < N && results_match; ++j) {
            if (c_base[i, j] != c_numa[i, j]) {
                std::println("Mismatch at [{},{}]: baseline={}, numa={}", i, j, c_base[i, j], c_numa[i, j]);
                results_match = false;
            }
        }
    }

    if (results_match) {
        std::println("Correctness: PASS (results match baseline)");
    } else {
        std::println("Correctness: FAIL (results differ from baseline)");
        std::exit(1);
    }

    // Summary
    double speedup = baseline_avg_ms / numa_avg_ms;
    std::println("\n=== Performance Summary ===");
    std::println("Baseline:     {:.2f} ms/iter, {:.2f} GFLOPS", baseline_avg_ms, baseline_gflops);
    std::println("NUMA-aware:   {:.2f} ms/iter, {:.2f} GFLOPS", numa_avg_ms, numa_gflops);
    std::println("Speedup:      {:.2f}x", speedup);

    if (speedup < 1.5) {
        std::println("\nWarning: Expected 2.5-4x speedup. Possible issues:");
        std::println("  - NUMA topology not as expected");
        std::println("  - Memory already well-distributed");
        std::println("  - Insufficient memory bandwidth saturation");
    }
}


int main() {
    if (!request_amx_permission()) {
        std::println(stderr, "AMX not available");
        return 1;
    }

    std::println("=== NUMA-Aware AMX Matmul Test Suite ===");
    std::println("Testing dual-socket SNC-2 column-parallel implementation\n");

    try {
        // Discover NUMA topology once
        auto config = DualSocketConfig::discover();
        std::println("Topology validated: {} sockets, {} nodes, {} CPUs/node\n",
                     DualSocketConfig::NUM_SOCKETS,
                     DualSocketConfig::NUM_NODES,
                     config.cpus_per_node);

        // Run benchmark
        benchmark_numa_vs_baseline(config);

        std::println("\n=== Benchmark completed successfully ===");
        return 0;

    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        return 1;
    }
}
