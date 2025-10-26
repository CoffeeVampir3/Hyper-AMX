module;
#include <print>
#include <vector>
#include <optional>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstddef>
#include <thread>
#include <mdspan>
#include <sys/syscall.h>
#include <unistd.h>
export module numa_matmul_test;
import tensor;
import matmul;
import numaaware;
import tensor_utils;
import quantization;

using namespace amx;

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

void benchmark_column_parallel_numa(const DualSocketConfig& config) {
    constexpr int M = 4096, N = 4096, K = 4096;
    constexpr int NUM_PARTITIONS = 2;  // 2 sockets
    constexpr int NUM_ITERS = 1000;

    std::println("\n=== Column-Parallel NUMA Matmul (Quantized int8 output): {}x{}x{}, {} partitions, {} iterations ===",
                 M, K, N, NUM_PARTITIONS, NUM_ITERS);

    using VNNILayout = vnni_layout<256, 4096>;

    double mem_per_iter_gb = (2.0 * M * K + K * N + M * N) * sizeof(std::int8_t) / (1024.0 * 1024.0 * 1024.0);
    double total_mem_gb = mem_per_iter_gb * NUM_ITERS;

    // THIS IS NOT BY ACCIDENT! WE NEED THIS FOR THE BENCHMARK TO BE CORRECT!
    std::println("Pre-allocating {} tensor sets ({:.2f} GB per set, {:.2f} GB total)...",
                 NUM_ITERS, mem_per_iter_gb, total_mem_gb);

    std::vector<std::optional<SocketReplicated<std::int8_t, Extents2D, RowMajor2D>>> A_replicas(NUM_ITERS);
    std::vector<std::optional<ColumnPartitioned<std::int8_t, Extents2D, VNNILayout>>> B_parts(NUM_ITERS);
    std::vector<std::optional<ColumnPartitioned<std::int8_t, Extents2D, RowMajor2D>>> C_parts(NUM_ITERS);

    // Per-tile quantization params (M/TILE_M × N/TILE_N tiles)
    std::vector<std::optional<ColumnPartitioned<QuantizationParams, Extents2D, RowMajor2D>>> params_parts(NUM_ITERS);

    constexpr int ALLOC_THREADS = 50;
    std::vector<std::jthread> alloc_threads;
    alloc_threads.reserve(ALLOC_THREADS);

    std::atomic<int> progress_counter{0};

    for (int tid = 0; tid < ALLOC_THREADS; tid++) {
        alloc_threads.emplace_back([&, tid] {
            for (int iter = tid; iter < NUM_ITERS; iter += ALLOC_THREADS) {
                std::int8_t a_val = 3 + iter;
                std::int8_t b_val = 5 + iter;

                auto A_source = make_tensor<std::int8_t>(M, K);
                fill(A_source, [&](std::size_t, std::size_t) { return a_val++; });

                auto B_source = make_tensor<std::int8_t>(K, N);
                fill(B_source, [&](std::size_t, std::size_t) { return b_val++; });

                // A: socket-replicated (row-major)
                A_replicas[iter].emplace(
                    Extents2D{M, K},
                    config,
                    [&](std::int8_t* dest, std::size_t count, int socket) {
                        std::memcpy(dest, A_source.data(), count * sizeof(std::int8_t));
                    }
                );

                // B: column-partitioned with VNNI conversion
                B_parts[iter].emplace(
                    Extents2D{K, N},
                    NUM_PARTITIONS,
                    config,
                    [&](std::int8_t* dest, std::size_t count, int socket) {
                        std::size_t cols_per_socket = N / NUM_PARTITIONS;
                        auto src_slice = slice<1>(B_source.view(), socket * cols_per_socket, cols_per_socket);
                        auto dest_view = std::mdspan<std::int8_t, Extents2D, VNNILayout>(
                            dest,
                            Extents2D{K, cols_per_socket}
                        );
                        convert_to_vnni(src_slice, dest_view);
                    }
                );

                // C: column-partitioned output (zero-init, int8)
                C_parts[iter].emplace(Extents2D{M, N}, NUM_PARTITIONS, config);

                // Params: column-partitioned, zero-initialized (computed during matmul)
                params_parts[iter].emplace(
                    Extents2D{M / TILE_M, N / TILE_N},
                    NUM_PARTITIONS,
                    config,
                    [](QuantizationParams* dest, std::size_t count, int socket) {
                        std::fill(dest, dest + count, QuantizationParams{0, 0.0f16});
                    }
                );

                int current = progress_counter.fetch_add(1) + 1;
                if (current % 50 == 0) {
                    std::println("  Allocated {}/{} tensor sets", current, NUM_ITERS);
                }
            }
        });
    }

    alloc_threads.clear();

    std::println("\nRunning column-parallel NUMA matmul benchmark (with fused quantization)...");
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        matmul_amx_column_parallel_quantized(*A_replicas[iter], *B_parts[iter], *C_parts[iter], *params_parts[iter], config);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_ms = duration.count() / static_cast<double>(NUM_ITERS) / 1000.0;
    double gflops = (2.0 * M * N * K) / (duration.count() / static_cast<double>(NUM_ITERS)) / 1000.0;

    std::println("\nResults:");
    std::println("  Average: {:.2f} ms/iter", avg_ms);
    std::println("  Throughput: {:.2f} GFLOPS", gflops);
    std::println("  Output: int8 (fused quantization, 4× bandwidth reduction vs int32)");
    std::println("  Memory pattern: A replicated on {} sockets, B+C partitioned across {} sockets",
                 DualSocketConfig::NUM_SOCKETS, NUM_PARTITIONS);
    std::println("  Threads: {} physical cores per socket ({} total), hyperthreads excluded",
                 config.physical_cores_per_socket, config.physical_cores_per_socket * DualSocketConfig::NUM_SOCKETS);

    // Validation: verify quantization params and reconstruction accuracy
    std::println("\n=== Validating quantization correctness ===");
    constexpr int VALIDATE_ITER = 0;

    // Sample first tile (0,0) from socket 0
    auto params_view = params_parts[VALIDATE_ITER]->view(0);
    auto C_view = C_parts[VALIDATE_ITER]->view(0);

    QuantizationParams p00 = params_view[0, 0];
    std::println("Tile[0,0] quantization params: bias={}, scale={}", p00.bias, (float)p00.scale);

    // Verify scale is non-zero (should be computed, not zero-initialized)
    if ((float)p00.scale == 0.0f) {
        std::println("  ERROR: Quantization params were not computed (scale is zero)!");
    } else {
        std::println("  ✓ Quantization params computed successfully");

        // Sample and dequantize a few values from the first tile
        std::println("  Sample dequantized values from tile[0,0]:");
        for (int i = 0; i < 3; i++) {
            int8_t quantized = C_view[i, 0];
            int32_t dequantized = dequantize_scalar(quantized, p00.bias, p00.scale);
            std::println("    C[{},0]: quantized={:4d}, dequantized={:8d}", i, quantized, dequantized);
        }
    }
}

export void run_numa_matmul_test() {
    if (!request_amx_permission()) {
        std::println(stderr, "AMX not available");
        return;
    }

    std::println("=== NUMA-Aware AMX Matmul Test ===");

    try {
        auto config = DualSocketConfig::discover();
        std::println("Topology: {} sockets, {} NUMA nodes total\n",
                     DualSocketConfig::NUM_SOCKETS,
                     DualSocketConfig::NUM_NODES);

        benchmark_column_parallel_numa(config);

        std::println("\n=== Test completed successfully ===");

    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
    }
}
