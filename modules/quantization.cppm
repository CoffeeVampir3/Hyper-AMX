module;
#include <cstdint>
#include <mdspan>
export module quantization;

// The implementation of quantization.cppm is in quantization.cpp because of immintrin.h being cancer.

export namespace AMXQ {

struct QuantizationParams {
    int32_t bias;
    _Float16 scale;
};

QuantizationParams compute_quantization_params(
    std::mdspan<const int32_t, std::extents<size_t, 16, 16>> tile);

int8_t quantize_scalar(int32_t value, int32_t bias, _Float16 scale);

int32_t dequantize_scalar(int8_t value, int32_t bias, _Float16 scale);

void quantize_tile_avx512(
    std::mdspan<const int32_t, std::extents<size_t, 16, 16>> temp,
    int8_t* out_base,
    size_t out_stride,
    int32_t bias,
    _Float16 scale);

void dequantize_tile_avx512(
    const int8_t* in_base,
    size_t in_stride,
    std::mdspan<int32_t, std::extents<size_t, 16, 16>> temp,
    int32_t bias,
    _Float16 scale);

}
