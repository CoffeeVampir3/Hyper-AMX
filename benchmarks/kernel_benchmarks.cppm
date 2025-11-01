module;
#include <print>
#include <chrono>
#include <vector>
#include <mdspan>
export module kernel_benchmarks;
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

export void run_kernel_benchmarks() {
    auto config = DualSocketConfig::discover();
    std::println("silu_mul_requantize_parallel: Configuration:");
    std::println("  Matrix dimensions: {}x{}", M, N);
    std::println("  Iterations: {}", NUM_ITERS);
    std::println("  NUMA nodes: {}", DualSocketConfig::NUM_NODES);
    std::println("  Sockets: {}", DualSocketConfig::NUM_SOCKETS);

    Tensor<int32_t, Extents2D, Layout::RowMajor> gate_src(Extents2D{M, N});
    Tensor<int32_t, Extents2D, Layout::RowMajor> up_src(Extents2D{M, N});
    fill(gate_src, [](size_t i, size_t j) { return static_cast<int32_t>((i + j) % 2000 - 1000); });
    fill(up_src, [](size_t i, size_t j) { return static_cast<int32_t>((i * j) % 2000 - 1000); });

    double mem_per_set_mb = (M*N*4.0 * 2 + M*N*1.0) / (1024*1024);
    double total_mem_gb = (mem_per_set_mb * NUM_ITERS) / 1024;

    std::vector<ColumnPartitioned<int32_t, Extents2D, Layout::RowMajor>> gate_batch;
    std::vector<ColumnPartitioned<int32_t, Extents2D, Layout::RowMajor>> up_batch;
    std::vector<ColumnPartitioned<int8_t, Extents2D, Layout::RowMajor>> out_batch;
    gate_batch.reserve(NUM_ITERS);
    up_batch.reserve(NUM_ITERS);
    out_batch.reserve(NUM_ITERS);

    auto alloc_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < NUM_ITERS; i++) {
        if (i % 100 == 0)
            std::println("  Allocated {}/{} sets...", i, NUM_ITERS);
        gate_batch.emplace_back(gate_src, 2, config);
        up_batch.emplace_back(up_src, 2, config);
        out_batch.emplace_back(Extents2D{M, N}, 2, config);
    }
    auto alloc_end = std::chrono::high_resolution_clock::now();
    auto alloc_time = std::chrono::duration_cast<std::chrono::milliseconds>(alloc_end - alloc_start);
    std::println("Allocation complete in {:.2f} seconds.\n", alloc_time.count() / 1e3);

    for (int i = 0; i < NUM_WARMUP; i++)
        silu_mul_requantize_parallel(gate_batch[i], up_batch[i], out_batch[i], config);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < NUM_ITERS; i++)
        silu_mul_requantize_parallel(gate_batch[i], up_batch[i], out_batch[i], config);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double total_time_s = duration.count() / 1e6;
    double time_per_iter_ms = (duration.count() / 1e3) / NUM_ITERS;
    double ops_per_kernel = M * N * (8.0 + 2.0 + 1.0);
    double total_ops = ops_per_kernel * NUM_ITERS;
    double gops = (total_ops / total_time_s) / 1e9;
    double bytes_per_kernel = M*N*4.0 + M*N*4.0 + M*N*1.0;
    double total_bytes = bytes_per_kernel * NUM_ITERS;
    double bandwidth_gbs = (total_bytes / total_time_s) / 1e9;
    double arithmetic_intensity = ops_per_kernel / bytes_per_kernel;

    std::println("\n=== Benchmark Results ===");
    std::println("Total time:           {:.3f} s", total_time_s);
    std::println("Time per iteration:   {:.3f} ms", time_per_iter_ms);
    std::println("Throughput:           {:.2f} GOPS", gops);
    std::println("Bandwidth:            {:.2f} GB/s", bandwidth_gbs);
    std::println("Arithmetic Intensity: {:.2f} OP/byte", arithmetic_intensity);
}
