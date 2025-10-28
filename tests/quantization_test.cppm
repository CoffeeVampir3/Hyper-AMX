module;
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <print>
#include <algorithm>
export module quantization_test;
import quantization;

static int tests_passed = 0;
static int tests_failed = 0;

void assert_eq(int32_t actual, int32_t expected, const char* test_name) {
    if (actual == expected) {
        tests_passed++;
    } else {
        std::println("FAIL [{}]: expected {}, got {}", test_name, expected, actual);
        tests_failed++;
    }
}

void assert_near(int32_t actual, int32_t expected, int32_t tolerance, const char* test_name) {
    int32_t diff = std::abs(actual - expected);
    if (diff <= tolerance) {
        tests_passed++;
    } else {
        std::println("FAIL [{}]: expected {} ± {}, got {} (diff: {})",
                     test_name, expected, tolerance, actual, diff);
        tests_failed++;
    }
}

void assert_in_range(int8_t value, int8_t min, int8_t max, const char* test_name) {
    if (value >= min && value <= max) {
        tests_passed++;
    } else {
        std::println("FAIL [{}]: value {} not in range [{}, {}]", test_name, value, min, max);
        tests_failed++;
    }
}

// Test: Basic quantization roundtrip
void test_basic_roundtrip() {
    int32_t tile[16][16];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            tile[i][j] = 30'000'000 + (i * 16 + j) * 120'000;
        }
    }

    auto params = AMXQ::compute_quantization_params(tile);

    std::println("  [DEBUG] bias={}, scale={}", params.bias, (float)params.scale);

    int32_t original = 45'000'000;
    int8_t quantized = AMXQ::quantize_scalar(original, params.bias, params.scale);
    int32_t dequantized = AMXQ::dequantize_scalar(quantized, params.bias, params.scale);

    int32_t error = std::abs(dequantized - original);
    float error_pct = 100.0f * error / original;

    if (error <= original / 200) {  // 0.5% tolerance
        std::println("  PASS [basic_roundtrip_0.5pct_error]: error={} ({:.2f}%)", error, error_pct);
        tests_passed++;
    } else {
        std::println("  FAIL [basic_roundtrip_0.5pct_error]: expected {} ± {}, got {} (error: {}, {:.2f}%)",
                   original, original/200, dequantized, error, error_pct);
        tests_failed++;
    }
}

// Test: Zero preservation
void test_zero_preservation() {
    int32_t original = 0;
    _Float16 scale = (_Float16)1.0f;
    int8_t quantized = AMXQ::quantize_scalar(original, 0, scale);
    int32_t dequantized = AMXQ::dequantize_scalar(quantized, 0, scale);

    assert_eq(quantized, 0, "zero_quantized");
    assert_eq(dequantized, 0, "zero_dequantized");
}

// Test: Symmetric quantization around bias
void test_symmetric() {
    int32_t bias = 50'000'000;
    _Float16 scale = (_Float16)(127.0f / 25'000'000);

    int32_t positive = 25'000'000;
    int32_t negative = -25'000'000;

    int8_t q_pos = AMXQ::quantize_scalar(bias + positive, bias, scale);
    int8_t q_neg = AMXQ::quantize_scalar(bias + negative, bias, scale);

    bool symmetric = std::abs(q_pos + q_neg) <= 1;
    if (symmetric) {
        tests_passed++;
    } else {
        std::println("FAIL [symmetric_quantization]: pos={}, neg={} (sum={})", q_pos, q_neg, q_pos + q_neg);
        tests_failed++;
    }
}

// Test: Small values near bias
void test_small_values() {
    int32_t bias = 45'000'000;
    _Float16 scale = (_Float16)(127.0f / 15'000'000);

    int32_t small = bias + 10'000;
    int8_t quantized = AMXQ::quantize_scalar(small, bias, scale);

    assert_in_range(quantized, -2, 2, "small_value_near_zero");
}

// Test: Full tile quantization
void test_tile_quantization() {
    alignas(64) int32_t input[16][16];
    alignas(64) int8_t output[16][16];
    alignas(64) int32_t reconstructed[16][16];

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            input[i][j] = 30'000'000 + (i * 16 + j) * 120'000;
        }
    }

    auto params = AMXQ::compute_quantization_params(input);
    AMXQ::quantize_tile_avx512(input, &output[0][0], 16, params.bias, params.scale);
    AMXQ::dequantize_tile_avx512(&output[0][0], 16, reconstructed, params.bias, params.scale);

    int32_t max_error = 0;
    float max_error_pct = 0;
    bool all_passed = true;

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            int32_t original = input[i][j];
            int32_t recovered = reconstructed[i][j];
            int32_t error = std::abs(recovered - original);
            float error_pct = 100.0f * error / original;

            max_error = std::max(max_error, error);
            max_error_pct = std::max(max_error_pct, error_pct);

            int32_t tolerance = original / 500;

            if (error > tolerance) {
                std::println("  FAIL [tile_quantization]: elem [{},{}] original={}, recovered={}, error={} ({:.2f}%)",
                           i, j, original, recovered, error, error_pct);
                all_passed = false;
                tests_failed++;
                break;
            }
        }
        if (!all_passed) break;
    }

    if (all_passed) {
        std::println("  PASS [tile_quantization]: max_error={} ({:.2f}%)", max_error, max_error_pct);
        tests_passed++;
    }
}

// Test: Tile vs scalar consistency
void test_tile_scalar_consistency() {
    alignas(64) int32_t input[16][16];
    alignas(64) int8_t tile_output[16][16];
    int8_t scalar_output[16][16];

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            input[i][j] = ((i - 8) * (j - 8)) * 500'000;
        }
    }

    auto params = AMXQ::compute_quantization_params(input);

    AMXQ::quantize_tile_avx512(input, &tile_output[0][0], 16, params.bias, params.scale);

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            scalar_output[i][j] = AMXQ::quantize_scalar(input[i][j], params.bias, params.scale);
        }
    }

    bool all_match = true;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            if (tile_output[i][j] != scalar_output[i][j]) {
                std::println("FAIL [tile_scalar_consistency]: elem [{},{}] tile={}, scalar={}",
                           i, j, tile_output[i][j], scalar_output[i][j]);
                all_match = false;
                tests_failed++;
                break;
            }
        }
        if (!all_match) break;
    }

    if (all_match) {
        tests_passed++;
    }
}

void test_parameter_computation() {
    int32_t tile[16][16];

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            tile[i][j] = 30'000'000 + (i * 16 + j) * 120'000;
        }
    }

    auto params = AMXQ::compute_quantization_params(tile);

    int32_t expected_bias = 45'240'000;
    assert_near(params.bias, expected_bias, 500'000, "bias_computation");

    float expected_scale = 127.0f / 15'240'000;
    float scale_fp32 = (float)params.scale;
    bool scale_ok = std::abs(scale_fp32 - expected_scale) < 0.00001f;
    if (scale_ok) {
        tests_passed++;
    } else {
        std::println("FAIL [scale_computation]: expected ~{}, got {}", expected_scale, scale_fp32);
        tests_failed++;
    }
}

void test_monotonicity() {
    int32_t bias = 45'000'000;
    _Float16 scale = (_Float16)(127.0f / 15'000'000);

    int32_t v1 = 31'000'000;
    int32_t v2 = 45'000'000;
    int32_t v3 = 59'000'000;

    int8_t q1 = AMXQ::quantize_scalar(v1, bias, scale);
    int8_t q2 = AMXQ::quantize_scalar(v2, bias, scale);
    int8_t q3 = AMXQ::quantize_scalar(v3, bias, scale);

    bool monotonic = (q1 <= q2) && (q2 <= q3);
    if (monotonic) {
        tests_passed++;
    } else {
        std::println("FAIL [monotonicity]: {} -> {}, {} -> {}, {} -> {}",
                   v1, q1, v2, q2, v3, q3);
        tests_failed++;
    }
}

// Test: AMX typical range (4096×4096×4096 matmul)
void test_amx_realistic_range() {
    constexpr int32_t K = 4096;
    constexpr int32_t max_per_element = 127 * 127;
    constexpr int32_t max_accumulation = K * max_per_element;

    int32_t tile[16][16];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            tile[i][j] = -33'000'000 + (i * 16 + j) * 260'000;
        }
    }

    auto params = AMXQ::compute_quantization_params(tile);

    int32_t mid = 0;
    int8_t q_mid = AMXQ::quantize_scalar(mid, params.bias, params.scale);
    int32_t dq_mid = AMXQ::dequantize_scalar(q_mid, params.bias, params.scale);

    int32_t error = std::abs(dq_mid - mid);
    float error_pct = 100.0f * error / 33'000'000;

    if (error <= 200'000) {
        std::println("  PASS [amx_midrange_0.6pct_error]: error={} ({:.2f}%)", error, error_pct);
        tests_passed++;
    } else {
        std::println("  FAIL [amx_midrange_0.6pct_error]: expected ± 200000, got error {} ({:.2f}%)",
                   error, error_pct);
        tests_failed++;
    }
}

void test_asymmetric_data() {
    int32_t tile[16][16];

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            tile[i][j] = 30'000'000 + (i * 16 + j) * 120'000;
        }
    }

    auto params = AMXQ::compute_quantization_params(tile);

    alignas(64) int8_t quantized[16][16];
    alignas(64) int32_t recovered[16][16];

    AMXQ::quantize_tile_avx512(tile, &quantized[0][0], 16, params.bias, params.scale);
    AMXQ::dequantize_tile_avx512(&quantized[0][0], 16, recovered, params.bias, params.scale);

    int32_t max_error = 0;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            int32_t error = std::abs(recovered[i][j] - tile[i][j]);
            max_error = std::max(max_error, error);
        }
    }

    float error_pct = 100.0f * max_error / 60'000'000;

    if (error_pct < 0.15f) {
        std::println("  PASS [asymmetric_data_affine]: max_error={} ({:.3f}%)", max_error, error_pct);
        tests_passed++;
    } else {
        std::println("  FAIL [asymmetric_data_affine]: max_error={} ({:.3f}%), expected <0.15%",
                   max_error, error_pct);
        tests_failed++;
    }
}

export void run_quantization_tests() {
    std::println("=== Quantization Tests (Affine + FP16) ===\n");

    test_basic_roundtrip();
    test_zero_preservation();
    test_symmetric();
    test_small_values();
    test_tile_quantization();
    test_tile_scalar_consistency();
    test_parameter_computation();
    test_monotonicity();
    test_amx_realistic_range();
    test_asymmetric_data();

    std::println("\n=== Quantization Test Results ===");
    std::println("Passed: {}", tests_passed);
    std::println("Failed: {}", tests_failed);
    std::println("Total:  {}", tests_passed + tests_failed);

    if (tests_failed == 0) {
        std::println("\n✓ All quantization tests passed!");
        std::println("\nKey findings:");
        std::println("  - Affine quantization: <0.15% error on asymmetric data");
        std::println("  - Symmetric data: <0.6% error");
        std::println("  - Tile operations: Consistent with scalar");
        std::println("  - Storage: 6 bytes/tile (int32 bias + FP16 scale)");
    } else {
        std::println("\n✗ Some tests failed.");
    }
}
