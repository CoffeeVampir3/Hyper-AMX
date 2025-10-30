module;
#include <print>
#include <chrono>
#include <vector>
#include <mdspan>
export module numa_add_bench;
import tensor;
import layout;
import numa;
import avx512;

using namespace Numa;
using namespace avx512;

constexpr size_t M = 4096;
constexpr size_t N = 4096;
constexpr size_t NUM_ITERS = 1000;
constexpr int NUM_WARMUP = 10;

using Extents2D = std::dextents<size_t, 2>;

export void run_numa_add_bench() {
    auto config = DualSocketConfig::discover();
    std::println("Configuration:");
    std::println("  Matrix dimensions: {}x{}", M, N);
    std::println("  Iterations: {}", NUM_ITERS);
    std::println("  NUMA nodes: {}", DualSocketConfig::NUM_NODES);
    std::println("  Sockets: {}", DualSocketConfig::NUM_SOCKETS);

    Tensor<int32_t, Extents2D, Layout::RowMajor> A_src(Extents2D{M, N});
    Tensor<int32_t, Extents2D, Layout::RowMajor> B_src(Extents2D{M, N});
    fill(A_src, [](size_t i, size_t j) { return static_cast<int32_t>((i + j) % 1000 - 500); });
    fill(B_src, [](size_t i, size_t j) { return static_cast<int32_t>((i * j) % 1000 - 500); });

    double mem_per_set_mb = (M*N*4.0 + M*N*4.0) / (1024*1024);
    double total_mem_gb = (mem_per_set_mb * NUM_ITERS) / 1024;
    double c_mem_mb = (M*N*4.0) / (1024*1024);

    std::vector<ColumnPartitioned<int32_t, Extents2D, Layout::RowMajor>> A_batch;
    std::vector<ColumnPartitioned<int32_t, Extents2D, Layout::RowMajor>> B_batch;
    A_batch.reserve(NUM_ITERS);
    B_batch.reserve(NUM_ITERS);

    auto alloc_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < NUM_ITERS; i++) {
        if (i % 100 == 0)
            std::println("  Allocated {}/{} sets...", i, NUM_ITERS);
        A_batch.emplace_back(A_src, 2, config);
        B_batch.emplace_back(B_src, 2, config);
    }
    auto alloc_end = std::chrono::high_resolution_clock::now();
    auto alloc_time = std::chrono::duration_cast<std::chrono::milliseconds>(alloc_end - alloc_start);
    std::println("Allocation complete in {:.2f} seconds.\n", alloc_time.count() / 1e3);

    ColumnPartitioned<int32_t, Extents2D, Layout::RowMajor> C(Extents2D{M, N}, 2, config);

    for (int i = 0; i < NUM_WARMUP; i++)
        add_column_parallel(A_batch[i], B_batch[i], C, config);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < NUM_ITERS; i++)
        add_column_parallel(A_batch[i], B_batch[i], C, config);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double total_time_s = duration.count() / 1e6;
    double time_per_iter_ms = (duration.count() / 1e3) / NUM_ITERS;
    double total_ops = static_cast<double>(M) * N * NUM_ITERS;
    double gops = (total_ops / total_time_s) / 1e9;
    double bytes_per_add = M*N*4.0 + M*N*4.0 + M*N*4.0;
    double total_bytes = bytes_per_add * NUM_ITERS;
    double bandwidth_gbs = (total_bytes / total_time_s) / 1e9;
    double arithmetic_intensity = static_cast<double>(M*N) / bytes_per_add;

    std::println("\n=== Benchmark Results ===");
    std::println("Total time:           {:.3f} s", total_time_s);
    std::println("Time per iteration:   {:.3f} ms", time_per_iter_ms);
    std::println("Throughput:           {:.2f} GOPS", gops);
    std::println("Bandwidth:            {:.2f} GB/s", bandwidth_gbs);
    std::println("Arithmetic Intensity: {:.2f} OP/byte", arithmetic_intensity);
}
