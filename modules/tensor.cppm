module;
#include <memory>
#include <mdspan>
#include <cstdlib>
#include <cstdint>
export module tensor;

export enum class bfloat16 : uint16_t {};

export enum class TensorLayout { RowMajor, ColMajor, VNNI };

export template<typename T, TensorLayout L = TensorLayout::RowMajor>
using Tensor = std::mdspan<T, std::dextents<size_t, 2>>;

export template<typename T>
concept RawStorage = std::same_as<T, uint8_t> || std::same_as<T, uint16_t> ||
                     std::same_as<T, int8_t> || std::same_as<T, int32_t> ||
                     std::same_as<T, float> || std::same_as<T, bfloat16>;

export template<typename T>
struct TensorOwning {
    using value_type = T;
    std::unique_ptr<T[], decltype(&std::free)> owned_data;
    size_t rows, cols, row_stride;
    static constexpr TensorLayout layout = TensorLayout::RowMajor;

    TensorOwning(size_t r, size_t c, size_t stride, size_t alignment)
        : owned_data(static_cast<T*>(std::aligned_alloc(alignment, r * stride * sizeof(T))), &std::free),
          rows(r), cols(c), row_stride(stride) {}

    constexpr auto view() const {
        return std::mdspan(owned_data.get(), rows, cols);
    }
    constexpr T& operator()(size_t i, size_t j) const {
        return owned_data[i * row_stride + j];
    }
    constexpr T* data() const { return owned_data.get(); }
    constexpr size_t stride_bytes() const { return row_stride * sizeof(T); }
};

export template<RawStorage T>
auto make_tensor(size_t rows, size_t cols, size_t alignment = 64) {
    return TensorOwning<T>(rows, cols, cols, alignment);
}

export template<RawStorage T>
auto make_tensor_strided(size_t rows, size_t cols, size_t row_stride, size_t alignment = 64) {
    return TensorOwning<T>(rows, cols, row_stride, alignment);
}

export template<typename T>
concept HasRowMajorLayout = requires {
    { T::layout } -> std::same_as<const TensorLayout&>;
    requires T::layout == TensorLayout::RowMajor;
};

export template<typename T>
concept HasColMajorLayout = requires {
    { T::layout } -> std::same_as<const TensorLayout&>;
    requires T::layout == TensorLayout::ColMajor;
};

export template<typename T>
concept HasVNNILayout = requires {
    { T::layout } -> std::same_as<const TensorLayout&>;
    requires T::layout == TensorLayout::VNNI;
};
