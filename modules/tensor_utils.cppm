module;
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <print>
#include <sys/syscall.h>
#include <unistd.h>
#include <immintrin.h>
#include <cstring>
#include <mdspan>
#include <concepts>
#include <utility>
export module tensor_utils;
import tensor;

export namespace utils {

inline bool request_amx() {
    constexpr auto ARCH_REQ_XCOMP_PERM = 0x1023;
    constexpr auto XFEATURE_XTILEDATA = 18;
    return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0;
}

template<typename T>
void contiguous_copy(T* dst, const T* src, size_t element_count) {
    std::memcpy(dst, src, element_count * sizeof(T));
}

//Mostly used when slicing
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
        // RowMajor → RowMajor: Column slicing via ColumnPartitioned
        for (size_t i = 0; i < copy_extent0; i++) {
            std::memcpy(&dst[i, 0], &src[src_offset_dim0 + i, src_offset_dim1], copy_extent1 * sizeof(T));
        }
    } else if constexpr (sizeof(T) == 1 && src_is_vnni && dst_is_vnni) {
        // VNNI → VNNI Partitioned
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
        // RowMajor → VNNI: Initial packing for AMX
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
        // Generic fallback
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
