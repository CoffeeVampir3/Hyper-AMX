module;
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>
export module hyperamx.tensor_utils;

import hyperamx.tensor;
import hyperamx.avx512;

export namespace tensor_utils {

namespace detail {

template<typename T>
concept HasSpan = requires(T& t) { t.span(); };

template<typename T>
concept HasView = requires(T& t) { t.view(); };

} // namespace detail

template<typename T>
auto view_of(T& tensor) {
    if constexpr (detail::HasSpan<T>) {
        return tensor.span();
    } else if constexpr (detail::HasView<T>) {
        return tensor.view();
    } else {
        return tensor;
    }
}

template<typename Tensor, typename F>
void fill(Tensor& tensor, F&& filler) {
    auto view = view_of(tensor);
    for (size_t i = 0; i < view.extent(0); i++) {
        for (size_t j = 0; j < view.extent(1); j++) {
            view[i, j] = std::invoke(filler, i, j);
        }
    }
}

template<typename Tensor, typename Fn>
void for_each(Tensor&& tensor, Fn&& fn) {
    auto view = view_of(std::forward<Tensor>(tensor));
    for (size_t i = 0; i < view.extent(0); i++) {
        for (size_t j = 0; j < view.extent(1); j++) {
            std::invoke(fn, i, j, view[i, j]);
        }
    }
}

template<typename Lhs, typename Rhs>
bool approximately_equal(Lhs& lhs, Rhs& rhs, double tolerance = 0.0) {
    auto lhs_view = view_of(lhs);
    auto rhs_view = view_of(rhs);
    if (lhs_view.extent(0) != rhs_view.extent(0) || lhs_view.extent(1) != rhs_view.extent(1)) {
        return false;
    }
    for (size_t i = 0; i < lhs_view.extent(0); i++) {
        for (size_t j = 0; j < lhs_view.extent(1); j++) {
            double diff = std::abs(static_cast<double>(lhs_view[i, j]) - static_cast<double>(rhs_view[i, j]));
            if (diff > tolerance) {
                return false;
            }
        }
    }
    return true;
}

template<auto Value, typename Tensor>
void fill(Tensor& tensor) {
    avx512::fill_constant_avx512<Value>(tensor);
}

template<typename Lhs, typename Rhs>
bool equal(Lhs& lhs, Rhs& rhs) {
    return approximately_equal(lhs, rhs, 0.0);
}

} // namespace tensor_utils
