module;
#include <print>
#include <cstdlib>
#include <cmath>
#include <mdspan>
export module kernel_tests;
import tensor;
import layout;
import kernel;
import quantization;
import tensor_utils;

using Extents2D = std::dextents<size_t, 2>;

void reference_silu_mul_requantize(
    auto gate_view, auto up_view, auto out_view)
{
    size_t M = gate_view.extent(0);
    size_t N = gate_view.extent(1);
    constexpr size_t TILE_SIZE = 16;
    for (size_t tile_m = 0; tile_m < M; tile_m += TILE_SIZE) {
        for (size_t tile_n = 0; tile_n < N; tile_n += TILE_SIZE) {
            size_t m_end = std::min(tile_m + TILE_SIZE, M);
            size_t n_end = std::min(tile_n + TILE_SIZE, N);
            float temp_f32[TILE_SIZE][TILE_SIZE];
            int32_t temp_i32[TILE_SIZE][TILE_SIZE];
            for (size_t i = tile_m; i < m_end; i++) {
                for (size_t j = tile_n; j < n_end; j++) {
                    float gate_f = static_cast<float>(gate_view[i, j]);
                    float up_f = static_cast<float>(up_view[i, j]);
                    float silu = gate_f / (1.0f + std::exp(-gate_f));
                    float result = silu * up_f;
                    temp_i32[i - tile_m][j - tile_n] = static_cast<int32_t>(std::round(result));
                }
            }
            auto temp_span = std::mdspan<const int32_t, std::extents<size_t, TILE_SIZE, TILE_SIZE>>(
                &temp_i32[0][0]
            );
            auto params = AMXQ::compute_quantization_params(temp_span);
            for (size_t i = tile_m; i < m_end; i++) {
                for (size_t j = tile_n; j < n_end; j++) {
                    int32_t val = temp_i32[i - tile_m][j - tile_n];
                    out_view[i, j] = AMXQ::quantize_scalar(val, params.bias, params.scale);
                }
            }
        }
    }
}

void test_silu_mul_requantize_basic() {
    std::println("Basic SiLU multiply requantize test");
    constexpr size_t M = 32, N = 32;
    Tensor<int32_t, Extents2D, Layout::RowMajor> gate(Extents2D{M, N});
    Tensor<int32_t, Extents2D, Layout::RowMajor> up(Extents2D{M, N});
    int32_t val = 1;
    utils::fill(gate, [&](auto...) { return (val++ % 1000) - 500; });
    val = 1;
    utils::fill(up, [&](auto...) { return (val++ % 1000) - 500; });
    Tensor<int8_t, Extents2D, Layout::RowMajor> result_ref(Extents2D{M, N});
    reference_silu_mul_requantize(gate.view(), up.view(), result_ref.view());
    Tensor<int8_t, Extents2D, Layout::RowMajor> result_kernel(Extents2D{M, N});
    kernel::silu_mul_requantize(gate.view(), up.view(), result_kernel.view());
    if (!utils::check_approximate_equal(result_kernel.view(), result_ref.view(), int8_t{1}, "SiLU mul requantize")) {
        std::exit(1);
    }
    std::println("   ✓ Basic kernel correctness\n");
}

void test_silu_mul_requantize_large() {
    std::println("Large matrix SiLU multiply requantize test");
    constexpr size_t M = 512, N = 256;
    Tensor<int32_t, Extents2D, Layout::RowMajor> gate(Extents2D{M, N});
    Tensor<int32_t, Extents2D, Layout::RowMajor> up(Extents2D{M, N});
    int32_t val = 1;
    utils::fill(gate, [&](auto...) { return (val++ % 4000) - 2000; });
    val = 1;
    utils::fill(up, [&](auto...) { return (val++ % 4000) - 2000; });
    Tensor<int8_t, Extents2D, Layout::RowMajor> result_ref(Extents2D{M, N});
    reference_silu_mul_requantize(gate.view(), up.view(), result_ref.view());
    Tensor<int8_t, Extents2D, Layout::RowMajor> result_kernel(Extents2D{M, N});
    kernel::silu_mul_requantize(gate.view(), up.view(), result_kernel.view());
    if (!utils::check_approximate_equal(result_kernel.view(), result_ref.view(), int8_t{1}, "Large matrix")) {
        std::exit(1);
    }
    std::println("   ✓ Large matrix correctness\n");
}

export void run_kernel_tests() {
    try {
        test_silu_mul_requantize_basic();
        test_silu_mul_requantize_large();
    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        std::exit(1);
    }
}
