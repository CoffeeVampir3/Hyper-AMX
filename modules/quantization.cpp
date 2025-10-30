module;
#include <immintrin.h>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <mdspan>
#include <bit>
module quantization;

// This .cpp segmentation exists purely because of immintrin.h being poorly designed.
// TLDR: importing immintrin.h in multiple places causes redeclaration of static inline garbage.

namespace AMXQ {

__m512 broadcast_fp16_to_fp32(_Float16 val) {
    uint16_t fp16_bits = std::bit_cast<uint16_t>(val);
    __m256i fp16_vec = _mm256_set1_epi16(fp16_bits);
    return _mm512_cvtph_ps(fp16_vec);
}

__m512 fast_reciprocal_ps(__m512 a) {
    __m512 x0 = _mm512_rcp14_ps(a);
    __m512 two = _mm512_set1_ps(2.0f);
    __m512 ax0 = _mm512_mul_ps(a, x0);
    __m512 correction = _mm512_sub_ps(two, ax0);
    return _mm512_mul_ps(x0, correction);
}

QuantizationParams compute_quantization_params(
    std::mdspan<const int32_t, std::extents<size_t, 16, 16>> tile)
{
    QuantizationParams params;
    __m512i sum_vec = _mm512_setzero_si512();

    for (int i = 0; i < 16; i++) {
        __m512i row = _mm512_loadu_si512(&tile[i, 0]);
        sum_vec = _mm512_add_epi32(sum_vec, row);
    }

    int32_t sum_arr[16];
    _mm512_storeu_si512(sum_arr, sum_vec);
    int64_t total_sum = 0;
    for (int i = 0; i < 16; i++) {
        total_sum += sum_arr[i];
    }
    params.bias = (int32_t)(total_sum / 256);

    __m512i bias_vec = _mm512_set1_epi32(params.bias);
    __m512i max_delta_vec = _mm512_setzero_si512();

    for (int i = 0; i < 16; i++) {
        __m512i row = _mm512_loadu_si512(&tile[i, 0]);
        __m512i delta = _mm512_sub_epi32(row, bias_vec);
        __m512i abs_delta = _mm512_abs_epi32(delta);
        max_delta_vec = _mm512_max_epi32(max_delta_vec, abs_delta);
    }

    int32_t max_arr[16];
    _mm512_storeu_si512(max_arr, max_delta_vec);
    int32_t max_delta = 0;
    for (int i = 0; i < 16; i++) {
        max_delta = std::max(max_delta, max_arr[i]);
    }

    float scale_fp32 = (max_delta > 0) ? (127.0f / max_delta) : 1.0f;
    params.scale = (_Float16)scale_fp32;

    return params;
}

int8_t quantize_scalar(int32_t value, int32_t bias, _Float16 scale) {
    int32_t delta = value - bias;
    float scale_fp32 = (float)scale;
    int32_t quantized = (int32_t)__builtin_roundf(delta * scale_fp32);
    if (quantized > 127) return 127;
    if (quantized < -128) return -128;
    return (int8_t)quantized;
}

int32_t dequantize_scalar(int8_t value, int32_t bias, _Float16 scale) {
    float scale_fp32 = (float)scale;
    float inv_scale = 1.0f / scale_fp32;
    int32_t delta = (int32_t)__builtin_roundf(value * inv_scale);
    return bias + delta;
}

void quantize_tile_avx512(
    std::mdspan<const int32_t, std::extents<size_t, 16, 16>> temp,
    int8_t* out_base,
    size_t out_stride,
    int32_t bias,
    _Float16 scale)
{
    __m512i bias_vec = _mm512_set1_epi32(bias);
    __m512 scale_vec = broadcast_fp16_to_fp32(scale);

    for (int row = 0; row < 16; row++) {
        __m512i v_i32 = _mm512_loadu_si512(&temp[row, 0]);
        __m512i delta_i32 = _mm512_sub_epi32(v_i32, bias_vec);
        __m512 delta_f32 = _mm512_cvtepi32_ps(delta_i32);
        __m512 quantized_f32 = _mm512_mul_ps(delta_f32, scale_vec);
        __m512i quantized_i32 = _mm512_cvt_roundps_epi32(quantized_f32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512i min_vec = _mm512_set1_epi32(-128);
        __m512i max_vec = _mm512_set1_epi32(127);
        quantized_i32 = _mm512_max_epi32(quantized_i32, min_vec);
        quantized_i32 = _mm512_min_epi32(quantized_i32, max_vec);
        __m128i packed8 = _mm512_cvtsepi32_epi8(quantized_i32);
        _mm_stream_si128((__m128i*)(out_base + row * out_stride), packed8);
    }
}

void dequantize_tile_avx512(
    const int8_t* in_base,
    size_t in_stride,
    std::mdspan<int32_t, std::extents<size_t, 16, 16>> temp,
    int32_t bias,
    _Float16 scale)
{
    __m512i bias_vec = _mm512_set1_epi32(bias);
    float scale_fp32 = (float)scale;
    __m512 scale_fp32_vec = _mm512_set1_ps(scale_fp32);
    __m512 inv_scale_vec = fast_reciprocal_ps(scale_fp32_vec);

    for (int row = 0; row < 16; row++) {
        __m128i packed8 = _mm_loadu_si128((__m128i*)(in_base + row * in_stride));
        __m512i v_i32 = _mm512_cvtepi8_epi32(packed8);
        __m512 v_f32 = _mm512_cvtepi32_ps(v_i32);
        __m512 delta_f32 = _mm512_mul_ps(v_f32, inv_scale_vec);
        __m512i delta_i32 = _mm512_cvt_roundps_epi32(delta_f32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512i result = _mm512_add_epi32(bias_vec, delta_i32);
        _mm512_storeu_si512(&temp[row, 0], result);
    }
}

}
