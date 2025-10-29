module;
#include <print>
#include <chrono>
#include <vector>
#include <mdspan>
export module numa_matmul_bench;
import tensor;
import layout;
import amx_gemms;
import numa;
import tensor_utils;

using namespace Numa;

constexpr size_t M = 4096;
constexpr size_t K = 4096;
constexpr size_t N = 4096;
constexpr size_t NUM_ITERS = 1000;
constexpr int NUM_WARMUP = 10;

using Extents2D = std::dextents<size_t, 2>;
using VNNILayout = Layout::VNNI<256, 4096>;

export void run_numa_matmul_bench() {
    if (!utils::request_amx()) {
        std::println("Failed to request AMX permissions");
        return;
    }
    auto config = DualSocketConfig::discover();
    std::println("Configuration:");
    std::println("  Matrix dimensions: {}x{}x{}", M, K, N);
    std::println("  Iterations: {}", NUM_ITERS);
    std::println("  NUMA nodes: {}", DualSocketConfig::NUM_NODES);
    std::println("  Sockets: {}", DualSocketConfig::NUM_SOCKETS);

    Tensor<int8_t, Extents2D, Layout::RowMajor> A_src(Extents2D{M, K});
    Tensor<int8_t, Extents2D, Layout::RowMajor> B_src(Extents2D{K, N});
    utils::fill(A_src, [](size_t i, size_t j) { return static_cast<int8_t>((i + j) % 127 + 1); });
    utils::fill(B_src, [](size_t i, size_t j) { return static_cast<int8_t>((i * j) % 127 + 1); });

    Tensor<int8_t, Extents2D, VNNILayout> B_vnni_src(Extents2D{K, N});
    auto B_vnni_view = B_vnni_src.view();
    VNNILayout::copy_from(B_src.view(), B_vnni_view, 1, 0, N);

    double mem_per_set_mb = (M*K*3.0 + K*N) / (1024*1024);
    double total_mem_gb = (mem_per_set_mb * NUM_ITERS) / 1024;
    double c_mem_mb = (M*N*4.0) / (1024*1024);

    std::vector<Replicated<int8_t, Extents2D, Layout::RowMajor>> A_batch;
    std::vector<ColumnPartitioned<int8_t, Extents2D, VNNILayout>> B_batch;
    A_batch.reserve(NUM_ITERS);
    B_batch.reserve(NUM_ITERS);

    auto alloc_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < NUM_ITERS; i++) {
        if (i % 100 == 0)
            std::println("  Allocated {}/{} sets...", i, NUM_ITERS);
        A_batch.emplace_back(A_src, config);
        B_batch.emplace_back(B_vnni_src, 2, config);
    }
    auto alloc_end = std::chrono::high_resolution_clock::now();
    auto alloc_time = std::chrono::duration_cast<std::chrono::milliseconds>(alloc_end - alloc_start);
    std::println("Allocation complete in {:.2f} seconds.\n", alloc_time.count() / 1e3);

    ColumnPartitioned<int32_t, Extents2D, Layout::RowMajor> C(Extents2D{M, N}, 2, config);

    for (int i = 0; i < NUM_WARMUP; i++)
        matmul_amx_column_parallel(A_batch[i], B_batch[i], C, config);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < NUM_ITERS; i++)
        matmul_amx_column_parallel(A_batch[i], B_batch[i], C, config);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double total_time_s = duration.count() / 1e6;
    double time_per_iter_ms = (duration.count() / 1e3) / NUM_ITERS;
    double total_ops = 2.0 * M * N * K * NUM_ITERS;
    double tops = (total_ops / total_time_s) / 1e12;
    double bytes_per_gemm = M*K*1.0 + K*N*1.0 + M*N*4.0;
    double total_bytes = bytes_per_gemm * NUM_ITERS;
    double bandwidth_gbs = (total_bytes / total_time_s) / 1e9;
    double arithmetic_intensity = (2.0*M*N*K) / bytes_per_gemm;

    std::println("\n=== Benchmark Results ===");
    std::println("Total time:           {:.3f} s", total_time_s);
    std::println("Time per iteration:   {:.3f} ms", time_per_iter_ms);
    std::println("Throughput:           {:.2f} TOPS", tops);
    std::println("Bandwidth:            {:.2f} GB/s", bandwidth_gbs);
    std::println("Arithmetic Intensity: {:.2f} FLOP/byte", arithmetic_intensity);
}
