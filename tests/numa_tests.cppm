module;
#include <print>
#include <cstdlib>
#include <cmath>
#include <mdspan>
export module numa_tests;
import tensor;
import layout;
import numa;
import avx512;
import kernel_tests;

using namespace Numa;
using Extents2D = std::dextents<size_t, 2>;

void test_add_column_parallel_basic() {
    std::println("Basic NUMA add test");
    constexpr size_t M = 16, N = 512;
    auto config = DualSocketConfig::discover();
    Tensor<int32_t, Extents2D, Layout::RowMajor> a_full(Extents2D{M, N});
    Tensor<int32_t, Extents2D, Layout::RowMajor> b_full(Extents2D{M, N});
    int32_t val = 1;
    avx512::fill(a_full, [&](auto...) { return (val++ % 1000) - 500; });
    val = 1000;
    avx512::fill(b_full, [&](auto...) { return (val++ % 1000) - 500; });
    Tensor<int32_t, Extents2D, Layout::RowMajor> result_ref(Extents2D{M, N});
    auto a_view = a_full.view();
    auto b_view = b_full.view();
    auto ref_view = result_ref.view();
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            ref_view[i, j] = a_view[i, j] + b_view[i, j];
        }
    }
    ColumnPartitioned<int32_t, Extents2D, Layout::RowMajor> a_part(a_full, 2, config);
    ColumnPartitioned<int32_t, Extents2D, Layout::RowMajor> b_part(b_full, 2, config);
    ColumnPartitioned<int32_t, Extents2D, Layout::RowMajor> output_part(Extents2D{M, N}, 2, config);
    add_column_parallel(a_part, b_part, output_part, config);
    Tensor<int32_t, Extents2D, Layout::RowMajor> result_gathered(Extents2D{M, N});
    all_gather(output_part, result_gathered);
    if (!avx512::check_approximate_equal(result_gathered.view(), result_ref.view(), int32_t{0}, "NUMA add basic")) {
        std::exit(1);
    }
    std::println("   ✓ Basic NUMA add correctness\n");
}

void test_deepseek_rmsnorm_column_parallel_basic() {
    std::println("Basic NUMA DeepSeek RMSNorm test");
    constexpr size_t M = 8, N = 256;
    constexpr float eps = 1e-6f;

    auto config = DualSocketConfig::discover();

    Tensor<int32_t, Extents2D, Layout::RowMajor> input_full(Extents2D{M, N});
    Tensor<float, Extents2D, Layout::RowMajor> weight_full(Extents2D{1, N});

    int32_t val = 1;
    avx512::fill(input_full, [&](auto...) { return (val++ % 2000) - 1000; });
    for (size_t i = 0; i < N; i++) {
        weight_full.view()[0, i] = 1.0f + (static_cast<float>(i % 10) / 100.0f);
    }

    Tensor<int32_t, Extents2D, Layout::RowMajor> result_ref(Extents2D{M, N});
    reference_rmsnorm(input_full.view(), weight_full.view(), result_ref.view(), eps);

    ColumnPartitioned<int32_t, Extents2D, Layout::RowMajor> input_part(input_full, 2, config);
    ColumnPartitioned<float, Extents2D, Layout::RowMajor> weight_part(weight_full, 2, config);
    ColumnPartitioned<int32_t, Extents2D, Layout::RowMajor> output_part(Extents2D{M, N}, 2, config);

    deepseek_rmsnorm_column_parallel(input_part, weight_part, output_part, config, eps);

    Tensor<int32_t, Extents2D, Layout::RowMajor> result_gathered(Extents2D{M, N});
    all_gather(output_part, result_gathered);

    if (!avx512::check_approximate_equal(result_gathered.view(), result_ref.view(), int32_t{1}, "NUMA DeepSeek RMSNorm basic")) {
        std::exit(1);
    }
    std::println("   ✓ Basic NUMA DeepSeek RMSNorm correctness\n");
}

void test_deepseek_rmsnorm_column_parallel_large() {
    std::println("Large NUMA DeepSeek RMSNorm test");
    constexpr size_t M = 128, N = 8192;
    constexpr float eps = 1e-6f;

    auto config = DualSocketConfig::discover();

    Tensor<int32_t, Extents2D, Layout::RowMajor> input_full(Extents2D{M, N});
    Tensor<float, Extents2D, Layout::RowMajor> weight_full(Extents2D{1, N});

    int32_t val = 1;
    avx512::fill(input_full, [&](auto...) { return (val++ % 10000) - 5000; });
    for (size_t i = 0; i < N; i++) {
        weight_full.view()[0, i] = 0.9f + (static_cast<float>(i % 20) / 100.0f);
    }

    Tensor<int32_t, Extents2D, Layout::RowMajor> result_ref(Extents2D{M, N});
    reference_rmsnorm(input_full.view(), weight_full.view(), result_ref.view(), eps);

    ColumnPartitioned<int32_t, Extents2D, Layout::RowMajor> input_part(input_full, 2, config);
    ColumnPartitioned<float, Extents2D, Layout::RowMajor> weight_part(weight_full, 2, config);
    ColumnPartitioned<int32_t, Extents2D, Layout::RowMajor> output_part(Extents2D{M, N}, 2, config);

    deepseek_rmsnorm_column_parallel(input_part, weight_part, output_part, config, eps);

    Tensor<int32_t, Extents2D, Layout::RowMajor> result_gathered(Extents2D{M, N});
    all_gather(output_part, result_gathered);

    if (!avx512::check_approximate_equal(result_gathered.view(), result_ref.view(), int32_t{1}, "NUMA DeepSeek RMSNorm large")) {
        std::exit(1);
    }
    std::println("   ✓ Large NUMA DeepSeek RMSNorm correctness\n");
}

export void run_numa_tests() {
    try {
        test_add_column_parallel_basic();
        test_deepseek_rmsnorm_column_parallel_basic();
        test_deepseek_rmsnorm_column_parallel_large();
    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        std::exit(1);
    }
}
