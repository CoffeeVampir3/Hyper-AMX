module;
#include <immintrin.h>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <print>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstring>
#include <mdspan>
#include <concepts>
#include <utility>
#include <algorithm>
#include <bit>
export module avx512;
import tensor;

// The reason this is a singular file is because immintrin.h is absolute cancer so we need a containment module for it.

export namespace avx512math {

enum class StoreHint {
    Temporal,
    NonTemporal
};

inline __m512 broadcast_fp16_to_fp32(_Float16 val) {
    uint16_t fp16_bits = std::bit_cast<uint16_t>(val);
    __m256i fp16_vec = _mm256_set1_epi16(fp16_bits);
    return _mm512_cvtph_ps(fp16_vec);
}

inline __m512 fast_reciprocal_ps(__m512 a) {
    __m512 x0 = _mm512_rcp14_ps(a);
    __m512 two = _mm512_set1_ps(2.0f);
    __m512 ax0 = _mm512_mul_ps(a, x0);
    __m512 correction = _mm512_sub_ps(two, ax0);
    return _mm512_mul_ps(x0, correction);
}

inline __m512 fast_rsqrt_ps(__m512 a) {
    __m512 x0 = _mm512_rsqrt14_ps(a);
    __m512 half = _mm512_set1_ps(0.5f);
    __m512 three_half = _mm512_set1_ps(1.5f);
    __m512 x0_sq = _mm512_mul_ps(x0, x0);
    __m512 a_x0_sq = _mm512_mul_ps(a, x0_sq);
    __m512 half_a_x0_sq = _mm512_mul_ps(half, a_x0_sq);
    __m512 correction = _mm512_sub_ps(three_half, half_a_x0_sq);
    return _mm512_mul_ps(x0, correction);
}

inline __m512 fast_exp_ps(__m512 x) {
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
    __m512 exp_neg_x = fast_exp_ps(neg_x);
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 denom = _mm512_add_ps(one, exp_neg_x);
    return _mm512_div_ps(x, denom);
}

template<size_t ElemsPerVec, auto LoadFn, auto AddFn, auto StoreFn, typename T>
inline void add_impl(const T* a, const T* b, T* out, size_t count) {
    size_t i = 0;
    for (; i + ElemsPerVec <= count; i += ElemsPerVec) {
        auto va = LoadFn(&a[i]);
        auto vb = LoadFn(&b[i]);
        auto vout = AddFn(va, vb);
        StoreFn(&out[i], vout);
    }
    for (; i < count; i++) {
        out[i] = a[i] + b[i];
    }
}

template<StoreHint hint = StoreHint::Temporal>
inline void add(const int32_t* a, const int32_t* b, int32_t* out, size_t count) {
    if constexpr (hint == StoreHint::NonTemporal) {
        add_impl<16, _mm512_loadu_si512, _mm512_add_epi32, _mm512_stream_si512>(a, b, out, count);
    } else {
        add_impl<16, _mm512_loadu_si512, _mm512_add_epi32, _mm512_storeu_si512>(a, b, out, count);
    }
}

template<StoreHint hint = StoreHint::Temporal>
inline void add(const int8_t* a, const int8_t* b, int8_t* out, size_t count) {
    if constexpr (hint == StoreHint::NonTemporal) {
        add_impl<64, _mm512_loadu_si512, _mm512_add_epi8, _mm512_stream_si512>(a, b, out, count);
    } else {
        add_impl<64, _mm512_loadu_si512, _mm512_add_epi8, _mm512_storeu_si512>(a, b, out, count);
    }
}

template<StoreHint hint = StoreHint::Temporal>
inline void add(const float* a, const float* b, float* out, size_t count) {
    if constexpr (hint == StoreHint::NonTemporal) {
        add_impl<16, _mm512_loadu_ps, _mm512_add_ps, _mm512_stream_ps>(a, b, out, count);
    } else {
        add_impl<16, _mm512_loadu_ps, _mm512_add_ps, _mm512_storeu_ps>(a, b, out, count);
    }
}

template<StoreHint hint = StoreHint::Temporal>
inline void add(const __bf16* a, const __bf16* b, __bf16* out, size_t count) {
    if constexpr (hint == StoreHint::NonTemporal) {
        add_impl<32, _mm512_loadu_pbh, _mm512_add_pbh, _mm512_stream_si512>(a, b, out, count);
    } else {
        add_impl<32, _mm512_loadu_pbh, _mm512_add_pbh, _mm512_storeu_pbh>(a, b, out, count);
    }
}

// If using non-temporal ensure to use _mm_sfence() at the appropriate level.
template<StoreHint hint = StoreHint::Temporal, typename ViewA, typename ViewB, typename ViewOut>
inline void add(ViewA a, ViewB b, ViewOut out) {
    using T = typename ViewOut::element_type;
    size_t rows = out.extent(0);
    size_t cols = out.extent(1);
    for (size_t i = 0; i < rows; i++) {
        add<hint>(&a[i, 0], &b[i, 0], &out[i, 0], cols);
    }
}

}

export namespace quantization {

struct QuantizationParams {
    int32_t bias;
    _Float16 scale;
};

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
    __m512 scale_vec = avx512math::broadcast_fp16_to_fp32(scale);

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
    __m512 inv_scale_vec = avx512math::fast_reciprocal_ps(scale_fp32_vec);

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

export namespace kernel {

constexpr size_t TILE_SIZE = 16;

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
            __m512 silu_result = avx512math::silu_avx512(gate_f32);
            __m512 result_f32 = _mm512_mul_ps(silu_result, up_f32);
            __m512i result_i32 = _mm512_cvt_roundps_epi32(result_f32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm512_storeu_si512(&temp_i32_span[i, j], result_i32);
        }
    }
    auto params = quantization::compute_quantization_params(temp_i32_span);
    quantization::quantize_tile_avx512(temp_i32_span, out_base, out_stride, params.bias, params.scale);
}

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

// Compute sum of squares
float deepseek_rmsnorm_compute_sum_sq_row_i32(const int32_t* input, size_t N) {
    constexpr size_t vec_width = 16;
    __m512 sum_sq_vec = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + vec_width <= N; i += vec_width) {
        __m512i x_i32 = _mm512_loadu_si512(&input[i]);
        __m512 x_f32 = _mm512_cvtepi32_ps(x_i32);
        __m512 x_sq = _mm512_mul_ps(x_f32, x_f32);
        sum_sq_vec = _mm512_add_ps(sum_sq_vec, x_sq);
    }

    float sum_sq = _mm512_reduce_add_ps(sum_sq_vec);
    for (; i < N; i++) {
        float x = static_cast<float>(input[i]);
        sum_sq += x * x;
    }

    return sum_sq;
}

void deepseek_rmsnorm_apply_row_i32(const int32_t* input, const float* weight, int32_t* output, size_t N, float rstd) {
    constexpr size_t vec_width = 16;
    __m512 rstd_vec = _mm512_set1_ps(rstd);

    size_t i = 0;
    for (; i + vec_width <= N; i += vec_width) {
        __m512i x_i32 = _mm512_loadu_si512(&input[i]);
        __m512 x_f32 = _mm512_cvtepi32_ps(x_i32);
        __m512 w_f32 = _mm512_loadu_ps(&weight[i]);
        __m512 normalized = _mm512_mul_ps(x_f32, rstd_vec);
        __m512 scaled = _mm512_mul_ps(normalized, w_f32);
        __m512i out_i32 = _mm512_cvt_roundps_epi32(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm512_storeu_si512(&output[i], out_i32);
    }

    for (; i < N; i++) {
        float x = static_cast<float>(input[i]);
        float normalized = x * rstd;
        float scaled = normalized * weight[i];
        output[i] = static_cast<int32_t>(std::round(scaled));
    }
}

template<typename TokenView, typename EmbeddingView, typename OutputView>
void embedding_lookup_bf16(TokenView token_ids, EmbeddingView embedding_table, OutputView output) {
    constexpr size_t vec_width = 32;
    size_t num_tokens = token_ids.extent(0);
    size_t hidden_size = embedding_table.extent(1);
    for (size_t tok = 0; tok < num_tokens; tok++) {
        int32_t token_id = token_ids[tok];
        size_t i = 0;
        for (; i + vec_width <= hidden_size; i += vec_width) {
            __m512i bf16_raw = _mm512_loadu_si512((__m512i*)&embedding_table[token_id, i]);
            __m256bh bf16_lo = _mm512_castsi512_si256(bf16_raw);
            __m256bh bf16_hi = _mm512_extracti64x4_epi64(bf16_raw, 1);
            __m512 f32_lo = _mm512_cvtpbh_ps(bf16_lo);
            __m512 f32_hi = _mm512_cvtpbh_ps(bf16_hi);
            __m512i i32_lo = _mm512_cvt_roundps_epi32(f32_lo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m512i i32_hi = _mm512_cvt_roundps_epi32(f32_hi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm512_stream_si512((__m512i*)&output[tok, i], i32_lo);
            _mm512_stream_si512((__m512i*)&output[tok, i + 16], i32_hi);
        }
        for (; i < hidden_size; i++) {
            float f = static_cast<float>(embedding_table[token_id, i]);
            output[tok, i] = static_cast<int32_t>(std::round(f));
        }
    }
}

}

export namespace avx512 {

inline bool request_amx() {
    constexpr auto ARCH_REQ_XCOMP_PERM = 0x1023;
    constexpr auto XFEATURE_XTILEDATA = 18;
    return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0;
}

template<typename T>
void contiguous_copy(T* dst, const T* src, size_t element_count) {
    std::memcpy(dst, src, element_count * sizeof(T));
}

template<typename SrcView, typename DstView>
void element_wise_copy_2d(const SrcView& src, DstView& dst,
                          size_t src_offset_dim0, size_t src_offset_dim1,
                          size_t copy_extent0, size_t copy_extent1) {
    using T = typename DstView::element_type;
    constexpr bool src_is_row_major = std::is_same_v<typename SrcView::layout_type, std::layout_right>;
    constexpr bool dst_is_row_major = std::is_same_v<typename DstView::layout_type, std::layout_right>;
    constexpr bool src_is_vnni = requires { typename SrcView::layout_type::is_vnni_layout; };
    constexpr bool dst_is_vnni = requires { typename DstView::layout_type::is_vnni_layout; };

    if constexpr (src_is_row_major && dst_is_row_major) {
        for (size_t i = 0; i < copy_extent0; i++) {
            std::memcpy(&dst[i, 0], &src[src_offset_dim0 + i, src_offset_dim1], copy_extent1 * sizeof(T));
        }
    } else if constexpr (sizeof(T) == 1 && src_is_vnni && dst_is_vnni) {
        constexpr size_t VNNI_TILE_K = 4;
        constexpr size_t VNNI_TILE_N = 16;
        constexpr size_t TILE_BYTES = 64;
        constexpr size_t UNROLL = 4;
        const bool use_nontemporal = (copy_extent0 * copy_extent1) > 262144;

        for (size_t k = 0; k < copy_extent0; k += VNNI_TILE_K) {
            size_t k_end = std::min(k + VNNI_TILE_K, copy_extent0);
            for (size_t n = 0; n < copy_extent1; n += VNNI_TILE_N * UNROLL) {
                if (k + VNNI_TILE_K < copy_extent0) {
                    const void* prefetch_addr = &src[src_offset_dim0 + k + VNNI_TILE_K, src_offset_dim1 + n];
                    _mm_prefetch((const char*)prefetch_addr, _MM_HINT_T0);
                }
                for (size_t tile = 0; tile < UNROLL && n + tile * VNNI_TILE_N < copy_extent1; tile++) {
                    size_t nn = n + tile * VNNI_TILE_N;
                    size_t n_end = std::min(nn + VNNI_TILE_N, copy_extent1);
                    if (k_end - k == VNNI_TILE_K && n_end - nn == VNNI_TILE_N) {
                        const T* src_tile = &src[src_offset_dim0 + k, src_offset_dim1 + nn];
                        T* dst_tile = &dst[k, nn];
                        __m512i data = _mm512_loadu_si512((__m512i*)src_tile);
                        if (use_nontemporal) {
                            _mm512_stream_si512((__m512i*)dst_tile, data);
                        } else {
                            _mm512_storeu_si512((__m512i*)dst_tile, data);
                        }
                    } else {
                        size_t bytes_to_copy = (k_end - k) * (n_end - nn);
                        const T* src_tile = &src[src_offset_dim0 + k, src_offset_dim1 + nn];
                        T* dst_tile = &dst[k, nn];
                        __mmask64 mask = (1ULL << bytes_to_copy) - 1;
                        __m512i data = _mm512_maskz_loadu_epi8(mask, src_tile);
                        _mm512_mask_storeu_epi8(dst_tile, mask, data);
                    }
                }
            }
        }
        if (use_nontemporal) {
            _mm_sfence();
        }
    } else if constexpr (sizeof(T) == 1 && src_is_row_major && dst_is_vnni) {
        for (size_t i = 0; i < copy_extent0; i++) {
            const T* src_row = &src[src_offset_dim0 + i, src_offset_dim1];
            for (size_t j = 0; j + 64 <= copy_extent1; j += 64) {
                __m512i vec = _mm512_loadu_si512((__m512i*)(src_row + j));
                alignas(64) T buf[64];
                _mm512_storeu_si512((__m512i*)buf, vec);
                for (size_t k = 0; k < 64; k++) {
                    dst[i, j + k] = buf[k];
                }
            }
            for (size_t j = (copy_extent1 / 64) * 64; j < copy_extent1; j++) {
                dst[i, j] = src_row[j];
            }
        }
    }  else {
        for (size_t i = 0; i < copy_extent0; i++) {
            for (size_t j = 0; j < copy_extent1; j++) {
                dst[i, j] = src[src_offset_dim0 + i, src_offset_dim1 + j];
            }
        }
    }
}

template<TensorStorage T, typename Fn>
void fill(T& tensor, Fn&& fn) {
    auto view = tensor.view();
    if constexpr (requires { view.extent(0); view.extent(1); }) {
        for (size_t i = 0; i < view.extent(0); i++) {
            for (size_t j = 0; j < view.extent(1); j++) {
                view[i, j] = fn(i, j);
            }
        }
    }
}

template<auto Value>
void fill_constant_avx512(auto view) {
    using elem_t = typename decltype(view)::element_type;
    constexpr size_t elems_per_vec = 64 / sizeof(elem_t);
    size_t rows = view.extent(0);
    size_t cols = view.extent(1);
    auto make_vec = []() {
        if constexpr (Value == 0) {
            if constexpr (std::same_as<elem_t, _Float16>) {
                return _mm512_setzero_ph();
            } else {
                return _mm512_setzero_si512();
            }
        } else {
            if constexpr (std::same_as<elem_t, int8_t> || std::same_as<elem_t, uint8_t>) {
                return _mm512_set1_epi8(Value);
            } else if constexpr (std::same_as<elem_t, int32_t>) {
                return _mm512_set1_epi32(Value);
            } else if constexpr (std::same_as<elem_t, _Float16>) {
                return _mm512_set1_ph(Value);
            }
        }
    };
    auto vec = make_vec();
    for (size_t i = 0; i < rows; i++) {
        elem_t* row_ptr = &view[i, 0];
        size_t j = 0;
        for (; j + elems_per_vec <= cols; j += elems_per_vec) {
            if constexpr (std::same_as<elem_t, _Float16>) {
                _mm512_stream_si512((__m512i*)(row_ptr + j), _mm512_castph_si512(vec));
            } else {
                _mm512_stream_si512((__m512i*)(row_ptr + j), vec);
            }
        }
        for (; j < cols; j++) {
            row_ptr[j] = Value;
        }
    }
    _mm_sfence();
}

template<TensorStorage T>
void zero(T& tensor) {
    auto view = tensor.view();
    if constexpr (requires { view.extent(0); view.extent(1); }) {
        using elem_t = typename decltype(view)::element_type;
        if constexpr (std::same_as<elem_t, int8_t> || std::same_as<elem_t, uint8_t> ||
                      std::same_as<elem_t, int32_t> || std::same_as<elem_t, _Float16>) {
            fill_constant_avx512<elem_t{0}>(view);
        } else {
            fill(tensor, [](auto...) { return 0; });
        }
    } else {
        fill(tensor, [](auto...) { return 0; });
    }
}

template<TensorStorage T>
void ones(T& tensor) {
    auto view = tensor.view();
    if constexpr (requires { view.extent(0); view.extent(1); }) {
        using elem_t = typename decltype(view)::element_type;
        if constexpr (std::same_as<elem_t, int8_t> || std::same_as<elem_t, uint8_t> ||
                      std::same_as<elem_t, int32_t> || std::same_as<elem_t, _Float16>) {
            fill_constant_avx512<elem_t{1}>(view);
        } else {
            fill(tensor, [](auto...) { return 1; });
        }
    } else {
        fill(tensor, [](auto...) { return 1; });
    }
}

template<typename View1, typename View2, typename T>
bool check_approximate_equal(View1 v1, View2 v2, T tolerance, const char* name) {
    for (size_t i = 0; i < v1.extent(0); i++) {
        for (size_t j = 0; j < v1.extent(1); j++) {
            auto diff = v1[i, j] - v2[i, j];
            if (std::abs(diff) > tolerance) {
                std::println(stderr, "   ✗ {} FAILED at [{}, {}]: {} vs {}", name, i, j, v1[i, j], v2[i, j]);
                return false;
            }
        }
    }
    return true;
}

inline void reference_matmul(const auto& A, const auto& B, auto C) {
    for (size_t i = 0; i < A.extent(0); i++) {
        for (size_t j = 0; j < B.extent(1); j++) {
            int32_t sum = 0;
            for (size_t k = 0; k < A.extent(1); k++) {
                sum += static_cast<int32_t>(A[i, k]) * static_cast<int32_t>(B[k, j]);
            }
            C[i, j] = sum;
        }
    }
}

}
