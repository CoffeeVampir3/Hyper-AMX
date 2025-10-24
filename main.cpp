#include <print>
#include <sys/syscall.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <thread>
#include <numa.h>
import tensor;
import matmul;
import amx_buffer;

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

template<typename T>
void fill(TensorOwning<T, TensorLayout::RowMajor>& t, T value) {
    for (size_t i = 0; i < t.shape()[0]; ++i)
        for (size_t j = 0; j < t.shape()[1]; ++j)
            t(i, j) = value;
}

template<typename T, typename Fn>
void fill(TensorOwning<T, TensorLayout::RowMajor>& t, Fn&& fn) {
    for (size_t i = 0; i < t.shape()[0]; ++i)
        for (size_t j = 0; j < t.shape()[1]; ++j)
            t(i, j) = fn(i, j);
}

template<typename Fn>
bool check_result(const TensorOwning<int32_t, TensorLayout::RowMajor>& c, Fn&& expected_fn, std::string_view test_name) {
    for (size_t i = 0; i < c.shape()[0]; ++i) {
        for (size_t j = 0; j < c.shape()[1]; ++j) {
            auto expected = expected_fn(i, j);
            if (c(i, j) != expected) {
                std::println("  FAIL [{}]: C[{},{}] = {} (expected {})", test_name, i, j, c(i, j), expected);
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
        auto a = make_tensor<int8_t>(M, K);
        auto b = make_tensor<int8_t>(K, N);
        auto c = make_tensor<int32_t>(M, N);
        fill(a, int8_t{1});
        fill(b, int8_t{1});
        fill(c, int32_t{0});
        auto b_blocked = AMXWeightBuffer<>::from_rowmajor(b.view());
        matmul_amx_int8_blocked_mt(a.view(), b_blocked, c.view());
        all_passed &= check_result(c, [K](size_t, size_t) { return K; }, "All ones");
    }
    {
        constexpr int M = 32, K = 64, N = 32;
        auto a = make_tensor<int8_t>(M, K);
        auto b = make_tensor<int8_t>(K, N);
        auto c = make_tensor<int32_t>(M, N);
        fill(a, [](size_t, size_t k) { return static_cast<int8_t>(k + 1); });
        fill(b, [](size_t k, size_t) { return static_cast<int8_t>(k + 1); });
        fill(c, int32_t{0});
        auto b_blocked = AMXWeightBuffer<>::from_rowmajor(b.view());
        matmul_amx_int8_blocked_mt(a.view(), b_blocked, c.view());
        int32_t expected = 0;
        for (int k = 0; k < K; ++k) expected += (k + 1) * (k + 1);
        all_passed &= check_result(c, [expected](size_t, size_t) { return expected; }, "Sequential pattern");
    }
    {
        constexpr int M = 128, K = 256, N = 128;
        auto a = make_tensor<int8_t>(M, K);
        auto b = make_tensor<int8_t>(K, N);
        auto c = make_tensor<int32_t>(M, N);
        fill(a, [](size_t i, size_t) { return static_cast<int8_t>(i % 127); });
        fill(b, [](size_t, size_t j) { return static_cast<int8_t>(j % 127); });
        fill(c, int32_t{0});
        auto b_blocked = AMXWeightBuffer<>::from_rowmajor(b.view());
        matmul_amx_int8_blocked_mt(a.view(), b_blocked, c.view());
        all_passed &= check_result(c, [K](size_t i, size_t j) { return K * ((i % 127) * (j % 127)); }, "Diagonal pattern");
    }
    std::println("\n=== {} ===\n", all_passed ? "All tests PASSED" : "Some tests FAILED");
    if (!all_passed) std::exit(1);
}

void benchmark_matmul() {
    constexpr int M = 4096, N = 4096, K = 4096;
    constexpr int NUM_ITERS = 1000;
    constexpr size_t CACHE_FLUSH_SIZE = 2ULL * 1024 * 1024 * 1024;
    std::println("=== Benchmark: {}x{}x{}, {} iterations ===", M, K, N, NUM_ITERS);
    std::vector<TensorOwning<int8_t, TensorLayout::RowMajor>> a_buffers, b_buffers;
    std::vector<AMXWeightBuffer<>> b_blocked_buffers;
    std::vector<TensorOwning<int32_t, TensorLayout::RowMajor>> c_buffers;
    a_buffers.reserve(NUM_ITERS);
    b_buffers.reserve(NUM_ITERS);
    b_blocked_buffers.reserve(NUM_ITERS);
    c_buffers.reserve(NUM_ITERS);
    int8_t a_val = 3, b_val = 5;
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        a_buffers.push_back(make_tensor<int8_t>(M, K));
        b_buffers.push_back(make_tensor<int8_t>(K, N));
        c_buffers.push_back(make_tensor<int32_t>(M, N));
        fill(a_buffers[iter], [&](size_t, size_t) { return a_val++; });
        fill(b_buffers[iter], [&](size_t, size_t) { return b_val++; });
        fill(c_buffers[iter], int32_t{0});
        b_blocked_buffers.push_back(AMXWeightBuffer<>::from_rowmajor(b_buffers[iter].view()));
    }
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        matmul_amx_int8_blocked_mt(a_buffers[iter].view(), b_blocked_buffers[iter], c_buffers[iter].view(), 16);
    }
    auto cache_flush = make_tensor<int64_t>(CACHE_FLUSH_SIZE / sizeof(int64_t), 1);
    fill(cache_flush, [](size_t i, size_t) { return static_cast<int64_t>(i * 7 + 13); });
    volatile int64_t flush_sink = 0;
    for (size_t i = 0; i < CACHE_FLUSH_SIZE / sizeof(int64_t); i += 8)
        flush_sink += cache_flush(i, 0);
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        matmul_amx_int8_blocked_mt(a_buffers[iter].view(), b_blocked_buffers[iter], c_buffers[iter].view(), 16);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_us = duration.count() / static_cast<double>(NUM_ITERS);
    double tops = (2.0 * M * N * K * NUM_ITERS) / (duration.count() * 1e6);
    std::println("Average: {:.2f} µs, Throughput: {:.2f} TOPS", avg_us, tops);
}

int main() {
    if (!request_amx_permission()) {
        std::println(stderr, "AMX not available");
        return 1;
    }
    test_matmul_correctness();
    benchmark_matmul();
    return 0;
}
