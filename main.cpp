#include <print>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstddef>
#include <string_view>
#include <sys/syscall.h>
#include <unistd.h>
#include <numa.h>
import tensor;
import matmul;

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

void test_matmul_correctness() {
    std::println("=== Correctness Tests ===\n");
    bool all_passed = true;
    {
        constexpr int M = 64, K = 128, N = 64;
        auto a = make_tensor<std::int8_t>(M, K);
        auto b = make_tensor<std::int8_t>(K, N);
        auto c = make_tensor<std::int32_t>(M, N);
        fill(a, std::int8_t{1});
        fill(b, std::int8_t{1});
        fill(c, std::int32_t{0});
        auto b_vnni = convert_to_vnni(b.view());
        matmul_amx_int8_blocked_mt(a.view(), b_vnni.view(), c.view());
        all_passed &= check_result(c, [K](size_t, size_t) { return K; }, "All ones");
    }
    {
        constexpr int M = 32, K = 64, N = 32;
        auto a = make_tensor<std::int8_t>(M, K);
        auto b = make_tensor<std::int8_t>(K, N);
        auto c = make_tensor<std::int32_t>(M, N);
        fill(a, [](std::size_t, std::size_t k) { return static_cast<std::int8_t>(k + 1); });
        fill(b, [](std::size_t k, std::size_t) { return static_cast<std::int8_t>(k + 1); });
        fill(c, std::int32_t{0});
        auto b_vnni = convert_to_vnni(b.view());
        matmul_amx_int8_blocked_mt(a.view(), b_vnni.view(), c.view());
        std::int32_t expected = 0;
        for (int k = 0; k < K; ++k) expected += (k + 1) * (k + 1);
        all_passed &= check_result(c, [expected](size_t, size_t) { return expected; }, "Sequential pattern");
    }
    {
        constexpr int M = 128, K = 256, N = 128;
        auto a = make_tensor<std::int8_t>(M, K);
        auto b = make_tensor<std::int8_t>(K, N);
        auto c = make_tensor<std::int32_t>(M, N);
        fill(a, [](std::size_t i, std::size_t) { return static_cast<std::int8_t>(i % 127); });
        fill(b, [](std::size_t, std::size_t j) { return static_cast<std::int8_t>(j % 127); });
        fill(c, std::int32_t{0});
        auto b_vnni = convert_to_vnni(b.view());
        matmul_amx_int8_blocked_mt(a.view(), b_vnni.view(), c.view());
        all_passed &= check_result(c, [K](size_t i, size_t j) { return K * ((i % 127) * (j % 127)); }, "Diagonal pattern");
    }
    std::println("\n=== {} ===\n", all_passed ? "All tests PASSED" : "Some tests FAILED");
    if (!all_passed) std::exit(1);
}

void benchmark_single_thread() {
    constexpr int M = 4096, N = 4096, K = 4096;
    constexpr int NUM_ITERS = 100;
    std::println("\n=== Single-threaded Benchmark: {}x{}x{}, {} iterations ===", M, K, N, NUM_ITERS);

    using ATensor = decltype(make_tensor<std::int8_t>(M, K));
    using BTensor = decltype(make_tensor<std::int8_t>(K, N));
    using BVNNITensor = decltype(convert_to_vnni(std::declval<BTensor>().view()));
    using CTensor = decltype(make_tensor<std::int32_t>(M, N));

    std::println("Allocating {} unique tensor sets ({:.2f} GB)...", NUM_ITERS,
                 NUM_ITERS * (16.0 + 16.0 + 64.0) / 1024.0);

    std::vector<ATensor> a_buffers;
    std::vector<BTensor> b_buffers;
    std::vector<BVNNITensor> b_vnni_buffers;
    std::vector<CTensor> c_buffers;
    a_buffers.reserve(NUM_ITERS);
    b_buffers.reserve(NUM_ITERS);
    b_vnni_buffers.reserve(NUM_ITERS);
    c_buffers.reserve(NUM_ITERS);

    std::int8_t a_val = 3, b_val = 5;
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        a_buffers.push_back(make_tensor<std::int8_t>(M, K));
        b_buffers.push_back(make_tensor<std::int8_t>(K, N));
        c_buffers.push_back(make_tensor<std::int32_t>(M, N));
        fill(a_buffers[iter], [&](std::size_t, std::size_t) { return a_val++; });
        fill(b_buffers[iter], [&](std::size_t, std::size_t) { return b_val++; });
        fill(c_buffers[iter], std::int32_t{0});
        b_vnni_buffers.push_back(convert_to_vnni(b_buffers[iter].view()));
    }

    std::println("Running benchmark...");
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        matmul_amx_int8_blocked(a_buffers[iter].view(), b_vnni_buffers[iter].view(), c_buffers[iter].view());
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_ms = duration.count() / static_cast<double>(NUM_ITERS) / 1000.0;
    double gflops = (2.0 * M * N * K) / (duration.count() / static_cast<double>(NUM_ITERS)) / 1000.0;
    std::println("Average: {:.2f} ms/iter, Throughput: {:.2f} GFLOPS", avg_ms, gflops);
}

void benchmark_multi_thread() {
    constexpr int M = 4096, N = 4096, K = 4096;
    constexpr int NUM_ITERS = 1000;
    constexpr int NUM_THREADS = 16;
    std::println("\n=== Multi-threaded Benchmark ({} threads): {}x{}x{}, {} iterations ===", NUM_THREADS, M, K, N, NUM_ITERS);

    using ATensor = decltype(make_tensor<std::int8_t>(M, K));
    using BTensor = decltype(make_tensor<std::int8_t>(K, N));
    using BVNNITensor = decltype(convert_to_vnni(std::declval<BTensor>().view()));
    using CTensor = decltype(make_tensor<std::int32_t>(M, N));

    std::println("Allocating {} unique tensor sets ({:.2f} GB)...", NUM_ITERS,
                 NUM_ITERS * (16.0 + 16.0 + 64.0) / 1024.0);

    std::vector<ATensor> a_buffers;
    std::vector<BTensor> b_buffers;
    std::vector<BVNNITensor> b_vnni_buffers;
    std::vector<CTensor> c_buffers;
    a_buffers.reserve(NUM_ITERS);
    b_buffers.reserve(NUM_ITERS);
    b_vnni_buffers.reserve(NUM_ITERS);
    c_buffers.reserve(NUM_ITERS);

    std::int8_t a_val = 3, b_val = 5;
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        a_buffers.push_back(make_tensor<std::int8_t>(M, K));
        b_buffers.push_back(make_tensor<std::int8_t>(K, N));
        c_buffers.push_back(make_tensor<std::int32_t>(M, N));
        fill(a_buffers[iter], [&](std::size_t, std::size_t) { return a_val++; });
        fill(b_buffers[iter], [&](std::size_t, std::size_t) { return b_val++; });
        fill(c_buffers[iter], std::int32_t{0});
        b_vnni_buffers.push_back(convert_to_vnni(b_buffers[iter].view()));
    }

    std::println("Running benchmark...");
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        matmul_amx_int8_blocked_mt(a_buffers[iter].view(), b_vnni_buffers[iter].view(), c_buffers[iter].view(), NUM_THREADS);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_ms = duration.count() / static_cast<double>(NUM_ITERS) / 1000.0;
    double gflops = (2.0 * M * N * K) / (duration.count() / static_cast<double>(NUM_ITERS)) / 1000.0;
    std::println("Average: {:.2f} ms/iter, Throughput: {:.2f} GFLOPS", avg_ms, gflops);
}

int main() {
    if (!request_amx_permission()) {
        std::println(stderr, "AMX not available");
        return 1;
    }
    test_matmul_correctness();
    benchmark_single_thread();
    benchmark_multi_thread();
    return 0;
}
