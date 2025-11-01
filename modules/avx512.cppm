module;
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <type_traits>
export module hyperamx.avx512;

export namespace avx512 {

namespace detail {

template<typename T>
concept HasSpan = requires(T& t) { t.span(); };

template<typename T>
concept HasView = requires(T& t) { t.view(); };

template<typename T>
auto as_view(T& t) {
    if constexpr (HasSpan<T>) {
        return t.span();
    } else if constexpr (HasView<T>) {
        return t.view();
    } else {
        return t;
    }
}

template<typename View>
concept View2D = requires(View view) {
    { view.extent(0) } -> std::convertible_to<size_t>;
    { view.extent(1) } -> std::convertible_to<size_t>;
    { view[0, 0] };
};

} // namespace detail

enum class StoreHint {
    Temporal,
    NonTemporal
};

inline void stream_fence() { _mm_sfence(); }
inline void prefetch_t0(const void* addr) { _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0); }


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

template<StoreHint Hint>
inline void store_vec(int8_t* ptr, __m512i vec) {
    if constexpr (Hint == StoreHint::NonTemporal) {
        _mm512_stream_si512(reinterpret_cast<__m512i*>(ptr), vec);
    } else {
        _mm512_storeu_si512(reinterpret_cast<void*>(ptr), vec);
    }
}

template<StoreHint Hint>
inline void store_vec(int32_t* ptr, __m512i vec) {
    if constexpr (Hint == StoreHint::NonTemporal) {
        _mm512_stream_si512(reinterpret_cast<__m512i*>(ptr), vec);
    } else {
        _mm512_storeu_si512(reinterpret_cast<void*>(ptr), vec);
    }
}

template<StoreHint Hint>
inline void store_vec(uint16_t* ptr, __m512i vec) {
    if constexpr (Hint == StoreHint::NonTemporal) {
        _mm512_stream_si512(reinterpret_cast<__m512i*>(ptr), vec);
    } else {
        _mm512_storeu_si512(reinterpret_cast<void*>(ptr), vec);
    }
}

template<StoreHint Hint>
inline void copy_row_i8(const int8_t* src, int8_t* dst, size_t count) {
    size_t i = 0;
    constexpr size_t width = 64;
    for (; i + width <= count; i += width) {
        __m512i vec = _mm512_loadu_si512(reinterpret_cast<const void*>(src + i));
        store_vec<Hint>(dst + i, vec);
    }
    size_t remaining = count - i;
    if (remaining) {
        __mmask64 mask = (remaining >= width) ? ~0ULL : ((__mmask64(1) << remaining) - 1);
        __m512i vec = _mm512_maskz_loadu_epi8(mask, src + i);
        _mm512_mask_storeu_epi8(dst + i, mask, vec);
    }
}

template<StoreHint Hint>
inline void copy_row_i32(const int32_t* src, int32_t* dst, size_t count) {
    size_t i = 0;
    constexpr size_t width = 16;
    for (; i + width <= count; i += width) {
        __m512i vec = _mm512_loadu_si512(reinterpret_cast<const void*>(src + i));
        store_vec<Hint>(dst + i, vec);
    }
    size_t remaining = count - i;
    if (remaining) {
        __mmask16 mask = (remaining >= width) ? 0xFFFF : ((__mmask16(1) << remaining) - 1);
        __m512i vec = _mm512_maskz_loadu_epi32(mask, src + i);
        _mm512_mask_storeu_epi32(dst + i, mask, vec);
    }
}

template<StoreHint Hint>
inline void copy_row_bf16(const __bf16* src, __bf16* dst, size_t count) {
    const uint16_t* src16 = reinterpret_cast<const uint16_t*>(src);
    uint16_t* dst16 = reinterpret_cast<uint16_t*>(dst);
    size_t i = 0;
    constexpr size_t width = 32;
    for (; i + width <= count; i += width) {
        __m512i vec = _mm512_loadu_si512(reinterpret_cast<const void*>(src16 + i));
        store_vec<Hint>(dst16 + i, vec);
    }
    size_t remaining = count - i;
    if (remaining) {
        __mmask32 mask = (remaining >= width) ? 0xFFFFFFFFu : ((__mmask32(1) << remaining) - 1);
        __m512i vec = _mm512_maskz_loadu_epi16(mask, src16 + i);
        _mm512_mask_storeu_epi16(dst16 + i, mask, vec);
    }
}

template<typename T, StoreHint Hint>
inline void copy_row(const T* src, T* dst, size_t count) {
    if constexpr (std::is_same_v<T, int8_t>) {
        copy_row_i8<Hint>(src, dst, count);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        copy_row_i32<Hint>(src, dst, count);
    } else if constexpr (std::is_same_v<T, __bf16>) {
        copy_row_bf16<Hint>(src, dst, count);
    } else {
        std::memcpy(dst, src, count * sizeof(T));
    }
}

template<StoreHint Hint, typename SrcView, typename DstView>
void copy_row_major(const SrcView& src, DstView& dst, int dim, size_t offset, size_t size) {
    using T = typename DstView::element_type;
    size_t rows = src.extent(0);
    size_t cols = src.extent(1);

    if (dim == 0) {
        size_t copy_rows = std::min(size, rows - offset);
        for (size_t r = 0; r < copy_rows; r++) {
            const T* src_row = &src[offset + r, 0];
            T* dst_row = &dst[r, 0];
            copy_row<T, Hint>(src_row, dst_row, cols);
        }
    } else {
        size_t copy_cols = std::min(size, cols - offset);
        for (size_t r = 0; r < rows; r++) {
            const T* src_row = &src[r, offset];
            T* dst_row = &dst[r, 0];
            copy_row<T, Hint>(src_row, dst_row, copy_cols);
        }
    }
}

template<StoreHint Hint, typename SrcView, typename DstView>
void copy_column_major(const SrcView& src, DstView& dst, int dim, size_t offset, size_t size) {
    using T = typename DstView::element_type;
    size_t rows = src.extent(0);
    size_t cols = src.extent(1);

    if (dim == 1) {
        size_t copy_cols = std::min(size, cols - offset);
        const T* src_ptr = src.data_handle() + offset * rows;
        T* dst_ptr = dst.data_handle();
        copy_row<T, Hint>(src_ptr, dst_ptr, rows * copy_cols);
    } else {
        size_t copy_rows = std::min(size, rows - offset);
        for (size_t r = 0; r < copy_rows; r++) {
            for (size_t c = 0; c < cols; c++) {
                dst[r, c] = src[offset + r, c];
            }
        }
    }
}

template<StoreHint Hint, typename SrcView, typename DstView>
void copy_vnni(const SrcView& src, DstView& dst, int dim, size_t offset, size_t size) {
    using T = typename DstView::element_type;
    size_t rows = src.extent(0);
    size_t cols = src.extent(1);

    if constexpr (!std::is_same_v<T, int8_t>) {
        copy_row_major<Hint>(src, dst, dim, offset, size);
    } else {
        constexpr size_t TILE_K = 4;
        constexpr size_t TILE_N = 16;
        constexpr size_t UNROLL = 4;

        auto copy_region = [&](size_t row_base, size_t col_base, size_t row_count, size_t col_count) {
            for (size_t k = 0; k < row_count; k += TILE_K) {
                size_t k_end = std::min(k + TILE_K, row_count);
                for (size_t n = 0; n < col_count; n += TILE_N * UNROLL) {
                    if (k + TILE_K < row_count) {
                        prefetch_t0(&src[row_base + k + TILE_K, col_base + n]);
                    }
                    for (size_t tile = 0; tile < UNROLL && n + tile * TILE_N < col_count; tile++) {
                        size_t nn = n + tile * TILE_N;
                        size_t n_end = std::min(nn + TILE_N, col_count);
                        size_t tile_rows = k_end - k;
                        size_t tile_cols = n_end - nn;
                        const int8_t* src_tile = &src[row_base + k, col_base + nn];
                        int8_t* dst_tile = &dst[row_base + k, col_base + nn];
                        if (tile_rows == TILE_K && tile_cols == TILE_N) {
                            __m512i vec = _mm512_loadu_si512(reinterpret_cast<const void*>(src_tile));
                            store_vec<Hint>(dst_tile, vec);
                        } else {
                            size_t bytes = tile_rows * tile_cols;
                            __mmask64 mask = (bytes >= 64) ? ~0ULL : ((__mmask64(1) << bytes) - 1);
                            __m512i vec = _mm512_maskz_loadu_epi8(mask, src_tile);
                            _mm512_mask_storeu_epi8(dst_tile, mask, vec);
                        }
                    }
                }
            }
        };

        if (dim == 0) {
            size_t copy_rows = std::min(size, rows - offset);
            copy_region(offset, 0, copy_rows, cols);
        } else {
            size_t copy_cols = std::min(size, cols - offset);
            copy_region(0, offset, rows, copy_cols);
        }
    }
}

template<auto Value, typename View>
void fill_constant_impl(View view) {
    using elem_t = typename std::remove_reference_t<View>::element_type;
    constexpr elem_t converted = static_cast<elem_t>(Value);
    size_t rows = view.extent(0);
    size_t cols = view.extent(1);
    for (size_t i = 0; i < rows; i++) {
        elem_t* row_ptr = &view[i, 0];
        size_t j = 0;
        if constexpr (std::same_as<elem_t, int8_t> || std::same_as<elem_t, uint8_t>) {
            __m512i vec = _mm512_set1_epi8(static_cast<int>(converted));
            constexpr size_t width = 64;
            for (; j + width <= cols; j += width) {
                _mm512_storeu_si512(reinterpret_cast<void*>(row_ptr + j), vec);
            }
            size_t remaining = cols - j;
            if (remaining) {
                __mmask64 mask = (remaining == 64) ? ~__mmask64{0} : ((__mmask64{1} << remaining) - 1);
                _mm512_mask_storeu_epi8(row_ptr + j, mask, vec);
                j = cols;
            }
        } else if constexpr (std::same_as<elem_t, int32_t>) {
            __m512i vec = _mm512_set1_epi32(static_cast<int>(converted));
            constexpr size_t width = 16;
            for (; j + width <= cols; j += width) {
                _mm512_storeu_si512(reinterpret_cast<void*>(row_ptr + j), vec);
            }
            size_t remaining = cols - j;
            if (remaining) {
                __mmask16 mask = (remaining == 16) ? static_cast<__mmask16>(0xFFFF) : ((__mmask16{1} << remaining) - 1);
                _mm512_mask_storeu_epi32(row_ptr + j, mask, vec);
                j = cols;
            }
        } else if constexpr (std::same_as<elem_t, float>) {
            __m512 vec = _mm512_set1_ps(static_cast<float>(converted));
            constexpr size_t width = 16;
            for (; j + width <= cols; j += width) {
                _mm512_storeu_ps(row_ptr + j, vec);
            }
            size_t remaining = cols - j;
            if (remaining) {
                __mmask16 mask = (remaining == 16) ? static_cast<__mmask16>(0xFFFF) : ((__mmask16{1} << remaining) - 1);
                _mm512_mask_storeu_ps(row_ptr + j, mask, vec);
                j = cols;
            }
        } else if constexpr (std::same_as<elem_t, _Float16>) {
            __m512h vec = _mm512_set1_ph(static_cast<_Float16>(converted));
            constexpr size_t width = 32;
            for (; j + width <= cols; j += width) {
                _mm512_storeu_ph(row_ptr + j, vec);
            }
            size_t remaining = cols - j;
            if (remaining) {
                __mmask32 mask = (remaining == 32) ? static_cast<__mmask32>(0xFFFFFFFFu) : ((__mmask32{1} << remaining) - 1);
                _mm512_mask_storeu_ph(row_ptr + j, mask, vec);
                j = cols;
            }
        }
        for (; j < cols; j++) {
            row_ptr[j] = converted;
        }
    }
}

template<auto Value, typename ViewLike>
void fill_constant_avx512(ViewLike&& view_like) {
    auto view = detail::as_view(view_like);
    using view_type = decltype(view);
    static_assert(detail::View2D<view_type>, "fill_constant_avx512 requires a 2D view");
    fill_constant_impl<Value>(view);
}

} // namespace avx512
