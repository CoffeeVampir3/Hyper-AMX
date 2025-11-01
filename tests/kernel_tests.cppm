module;
#include <print>
#include <cstdlib>
#include <cmath>
#include <mdspan>
export module kernel_tests;
import tensor;
import layout;
import avx512;

using namespace quantization;
using namespace kernel;
using namespace avx512;
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
            auto params = compute_quantization_params(temp_span);
            for (size_t i = tile_m; i < m_end; i++) {
                for (size_t j = tile_n; j < n_end; j++) {
                    int32_t val = temp_i32[i - tile_m][j - tile_n];
                    out_view[i, j] = quantize_scalar(val, params.bias, params.scale);
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
    fill(gate, [&](auto...) { return (val++ % 1000) - 500; });
    val = 1;
    fill(up, [&](auto...) { return (val++ % 1000) - 500; });
    Tensor<int8_t, Extents2D, Layout::RowMajor> result_ref(Extents2D{M, N});
    reference_silu_mul_requantize(gate.view(), up.view(), result_ref.view());
    Tensor<int8_t, Extents2D, Layout::RowMajor> result_kernel(Extents2D{M, N});
    silu_mul_requantize(gate.view(), up.view(), result_kernel.view());
    if (!check_approximate_equal(result_kernel.view(), result_ref.view(), int8_t{1}, "SiLU mul requantize")) {
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
    fill(gate, [&](auto...) { return (val++ % 4000) - 2000; });
    val = 1;
    fill(up, [&](auto...) { return (val++ % 4000) - 2000; });
    Tensor<int8_t, Extents2D, Layout::RowMajor> result_ref(Extents2D{M, N});
    reference_silu_mul_requantize(gate.view(), up.view(), result_ref.view());
    Tensor<int8_t, Extents2D, Layout::RowMajor> result_kernel(Extents2D{M, N});
    silu_mul_requantize(gate.view(), up.view(), result_kernel.view());
    if (!check_approximate_equal(result_kernel.view(), result_ref.view(), int8_t{1}, "Large matrix")) {
        std::exit(1);
    }
    std::println("   ✓ Large matrix correctness\n");
}

export void reference_rmsnorm(auto input_view, auto weight_view, auto output_view, float eps) {
    size_t M = input_view.extent(0);
    size_t N = input_view.extent(1);
    for (size_t i = 0; i < M; i++) {
        float sum_sq = 0.0f;
        for (size_t j = 0; j < N; j++) {
            float x = static_cast<float>(input_view[i, j]);
            sum_sq += x * x;
        }
        float mean_sq = sum_sq / static_cast<float>(N);
        float rstd = 1.0f / std::sqrt(mean_sq + eps);
        for (size_t j = 0; j < N; j++) {
            float x = static_cast<float>(input_view[i, j]);
            float normalized = x * rstd;
            float scaled = normalized * weight_view[0, j];
            output_view[i, j] = static_cast<int32_t>(std::round(scaled));
        }
    }
}

void test_rmsnorm_basic() {
    std::println("Basic DeepSeek RMSNorm test");
    constexpr size_t M = 8, N = 128;
    constexpr float eps = 1e-6f;
    Tensor<int32_t, Extents2D, Layout::RowMajor> input(Extents2D{M, N});
    Tensor<float, Extents2D, Layout::RowMajor> weight(Extents2D{1, N});
    int32_t val = 1;
    fill(input, [&](auto...) { return (val++ % 2000) - 1000; });
    for (size_t i = 0; i < N; i++) {
        weight.view()[0, i] = 1.0f + (static_cast<float>(i % 10) / 100.0f);
    }
    Tensor<int32_t, Extents2D, Layout::RowMajor> result_ref(Extents2D{M, N});
    reference_rmsnorm(input.view(), weight.view(), result_ref.view(), eps);
    Tensor<int32_t, Extents2D, Layout::RowMajor> result_kernel(Extents2D{M, N});
    for (size_t i = 0; i < M; i++) {
        float sum_sq = deepseek_rmsnorm_compute_sum_sq_row_i32(&input.view()[i, 0], N);
        float mean_sq = sum_sq / static_cast<float>(N);
        float rstd = 1.0f / std::sqrt(mean_sq + eps);
        deepseek_rmsnorm_apply_row_i32(&input.view()[i, 0], &weight.view()[0, 0], &result_kernel.view()[i, 0], N, rstd);
    }
    if (!check_approximate_equal(result_kernel.view(), result_ref.view(), int32_t{1}, "DeepSeek RMSNorm basic")) {
        std::exit(1);
    }
    std::println("   ✓ Basic DeepSeek RMSNorm correctness\n");
}

void test_rmsnorm_large() {
    std::println("Large DeepSeek RMSNorm test");
    constexpr size_t M = 128, N = 4096;
    constexpr float eps = 1e-6f;
    Tensor<int32_t, Extents2D, Layout::RowMajor> input(Extents2D{M, N});
    Tensor<float, Extents2D, Layout::RowMajor> weight(Extents2D{1, N});
    int32_t val = 1;
    fill(input, [&](auto...) { return (val++ % 10000) - 5000; });
    for (size_t i = 0; i < N; i++) {
        weight.view()[0, i] = 0.9f + (static_cast<float>(i % 20) / 100.0f);
    }
    Tensor<int32_t, Extents2D, Layout::RowMajor> result_ref(Extents2D{M, N});
    reference_rmsnorm(input.view(), weight.view(), result_ref.view(), eps);
    Tensor<int32_t, Extents2D, Layout::RowMajor> result_kernel(Extents2D{M, N});
    for (size_t i = 0; i < M; i++) {
        float sum_sq = deepseek_rmsnorm_compute_sum_sq_row_i32(&input.view()[i, 0], N);
        float mean_sq = sum_sq / static_cast<float>(N);
        float rstd = 1.0f / std::sqrt(mean_sq + eps);
        deepseek_rmsnorm_apply_row_i32(&input.view()[i, 0], &weight.view()[0, 0], &result_kernel.view()[i, 0], N, rstd);
    }
    if (!check_approximate_equal(result_kernel.view(), result_ref.view(), int32_t{1}, "DeepSeek RMSNorm large")) {
        std::exit(1);
    }
    std::println("   ✓ Large DeepSeek RMSNorm correctness\n");
}

void test_embedding_lookup_basic() {
    std::println("Basic embedding lookup test");
    constexpr size_t vocab_size = 256;
    constexpr size_t hidden_size = 128;
    constexpr size_t num_tokens = 8;
    Tensor<__bf16, Extents2D, Layout::RowMajor> embedding_table(Extents2D{vocab_size, hidden_size});
    for (size_t i = 0; i < vocab_size; i++) {
        for (size_t j = 0; j < hidden_size; j++) {
            embedding_table.view()[i, j] = static_cast<__bf16>(i * hidden_size + j);
        }
    }
    Tensor<int32_t, std::dextents<size_t, 1>, Layout::RowMajor> token_ids(std::dextents<size_t, 1>{num_tokens});
    token_ids.view()[0] = 0;
    token_ids.view()[1] = 1;
    token_ids.view()[2] = 10;
    token_ids.view()[3] = 50;
    token_ids.view()[4] = 100;
    token_ids.view()[5] = 150;
    token_ids.view()[6] = 200;
    token_ids.view()[7] = 255;
    Tensor<int32_t, Extents2D, Layout::RowMajor> output(Extents2D{num_tokens, hidden_size});
    kernel::embedding_lookup_bf16(token_ids.view(), embedding_table.view(), output.view());
    for (size_t tok = 0; tok < num_tokens; tok++) {
        int32_t token_id = token_ids.view()[tok];
        for (size_t i = 0; i < hidden_size; i++) {
            int32_t expected = static_cast<int32_t>(std::round(static_cast<float>(embedding_table.view()[token_id, i])));
            int32_t actual = output.view()[tok, i];
            if (std::abs(actual - expected) > 1) {
                std::println(stderr, "   ✗ Embedding lookup FAILED at token {} pos {}: {} vs {}", tok, i, actual, expected);
                std::exit(1);
            }
        }
    }
    std::println("   ✓ Basic embedding lookup correctness\n");
}

export void run_kernel_tests() {
    try {
        test_silu_mul_requantize_basic();
        test_silu_mul_requantize_large();
        test_rmsnorm_basic();
        test_rmsnorm_large();
        test_embedding_lookup_basic();
    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        std::exit(1);
    }
}
