module;
#include <immintrin.h>
#include <cstdint>
#include <mdspan>
export module kernel;
import quantization;

constexpr size_t TILE_SIZE = 16;

inline __m512 fast_exp_avx512(__m512 x) {
    const __m512 log2e = _mm512_set1_ps(1.44269504088896341f);
    const __m512 x_scaled = _mm512_mul_ps(x, log2e);
    const __m512 fx = _mm512_floor_ps(x_scaled);
    const __m512 X = _mm512_sub_ps(x_scaled, fx);
    const __m512 c0 = _mm512_set1_ps(1.0f);
    const __m512 c1 = _mm512_set1_ps(0.693147181f);
    const __m512 c2 = _mm512_set1_ps(0.240226507f);
    const __m512 c3 = _mm512_set1_ps(0.0558530818f);
    const __m512 c4 = _mm512_set1_ps(0.00898934f);
    const __m512 c5 = _mm512_set1_ps(0.00187682f);
    __m512 y = _mm512_fmadd_ps(c5, X, c4);
    y = _mm512_fmadd_ps(y, X, c3);
    y = _mm512_fmadd_ps(y, X, c2);
    y = _mm512_fmadd_ps(y, X, c1);
    y = _mm512_fmadd_ps(y, X, c0);
    return _mm512_scalef_ps(y, fx);
}

inline __m512 silu_avx512(__m512 x) {
    __m512 neg_x = _mm512_mul_ps(x, _mm512_set1_ps(-1.0f));
    __m512 exp_neg_x = fast_exp_avx512(neg_x);
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 denom = _mm512_add_ps(one, exp_neg_x);
    return _mm512_div_ps(x, denom);
}

void silu_mul_quantize_tile(
    const int32_t* gate_base, size_t gate_stride,
    const int32_t* up_base, size_t up_stride,
    int8_t* out_base, size_t out_stride,
    size_t tile_m, size_t tile_n,
    std::mdspan<int32_t, std::extents<size_t, TILE_SIZE, TILE_SIZE>> temp_i32_span)
{
    for (size_t i = 0; i < tile_m; i++) {
        for (size_t j = 0; j < tile_n; j += 16) {
            __m512i gate_i32 = _mm512_loadu_si512(&gate_base[i * gate_stride + j]);
            __m512i up_i32 = _mm512_loadu_si512(&up_base[i * up_stride + j]);
            __m512 gate_f32 = _mm512_cvtepi32_ps(gate_i32);
            __m512 up_f32 = _mm512_cvtepi32_ps(up_i32);
            __m512 silu_result = silu_avx512(gate_f32);
            __m512 result_f32 = _mm512_mul_ps(silu_result, up_f32);
            __m512i result_i32 = _mm512_cvt_roundps_epi32(result_f32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm512_storeu_si512(&temp_i32_span[i, j], result_i32);
        }
    }
    auto params = AMXQ::compute_quantization_params(temp_i32_span);
    AMXQ::quantize_tile_avx512(temp_i32_span, out_base, out_stride, params.bias, params.scale);
}

export namespace kernel {

// Used in the MLP for the Gate + Up project in the swiglu
template<typename GateView, typename UpView, typename OutView>
void silu_mul_requantize(GateView gate, UpView up, OutView out,
                         int thread_id = 0, int num_threads = 1)
{
    size_t M = gate.extent(0);
    size_t N = gate.extent(1);
    size_t rows_per_thread = (M + num_threads - 1) / num_threads;
    size_t m_start = thread_id * rows_per_thread;
    size_t m_end = std::min(M, m_start + rows_per_thread);
    m_start = (m_start / TILE_SIZE) * TILE_SIZE;
    m_end = ((m_end + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    m_end = std::min(M, m_end);
    alignas(64) int32_t temp_i32[TILE_SIZE][TILE_SIZE];
    auto temp_i32_span = std::mdspan<int32_t, std::extents<size_t, TILE_SIZE, TILE_SIZE>>(&temp_i32[0][0]);
    for (size_t m = m_start; m < m_end; m += TILE_SIZE) {
        for (size_t n = 0; n < N; n += TILE_SIZE) {
            size_t tile_m = std::min(TILE_SIZE, M - m);
            size_t tile_n = std::min(TILE_SIZE, N - n);
            silu_mul_quantize_tile(
                &gate[m, n], N,
                &up[m, n], N,
                &out[m, n], N,
                tile_m, tile_n,
                temp_i32_span
            );
        }
    }
}

}
