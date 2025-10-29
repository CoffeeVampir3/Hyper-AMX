module;
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <print>
#include <sys/syscall.h>
#include <unistd.h>
#include <immintrin.h>
export module tensor_utils;
import tensor;

export namespace utils {

constexpr auto ARCH_REQ_XCOMP_PERM = 0x1023;
constexpr auto XFEATURE_XTILEDATA = 18;

bool request_amx() {
    return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0;
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

void reference_matmul(const auto& A, const auto& B, auto C) {
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
