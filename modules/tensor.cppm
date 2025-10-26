module;
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <mdspan>
#include <memory>
#include <functional>
#include <concepts>
#include <numa.h>
export module tensor;
export import layout;
import quantization;

export enum class bfloat16 : uint16_t {};

export template<typename T>
concept RawStorage = std::same_as<T, uint8_t> || std::same_as<T, uint16_t> ||
                     std::same_as<T, int8_t> || std::same_as<T, int32_t> ||
                     std::same_as<T, int64_t> || std::same_as<T, float> ||
                     std::same_as<T, bfloat16> || std::same_as<T, QuantizationParams>;

export template<typename A>
concept TensorAllocator = requires(A a, size_t n) {
    typename A::pointer;
    { a.allocate(n, size_t{64}) } -> std::same_as<typename A::pointer>;
    { a.deallocate(std::declval<typename A::pointer>(), n) } -> std::same_as<void>;
};

export template<typename T>
struct AlignedAllocator {
    using pointer = T*;
    T* allocate(size_t count, size_t alignment = 64) const {
        return static_cast<T*>(std::aligned_alloc(alignment, count * sizeof(T)));
    }
    void deallocate(T* ptr, size_t count) const {
        std::free(ptr);
    }
};

export template<typename T>
struct NumaNodeAllocator {
    using pointer = T*;
    int node;
    T* allocate(size_t count, size_t alignment = 64) const {
        return static_cast<T*>(numa_alloc_onnode(count * sizeof(T), node));
    }
    void deallocate(T* ptr, size_t count) const {
        numa_free(ptr, count * sizeof(T));
    }
};

export template<RawStorage T, typename Extents, typename Layout>
struct Tensor {
    using element_type = T;
    using value_type = T;
    using extents_type = Extents;
    using layout_type = Layout;
    using mapping_type = typename Layout::template mapping<Extents>;
    using mdspan_type = std::mdspan<T, Extents, Layout>;

    struct Deleter {
        std::function<void(T*, size_t)> del_fn;
        size_t size;
        void operator()(T* p) const {
            if (p && del_fn) del_fn(p, size);
        }
    };

    mapping_type map;
    std::unique_ptr<T[], Deleter> owned_data;

    template<TensorAllocator Alloc>
    Tensor(Extents e, Alloc allocator, size_t alignment = 64)
        : map(e) {
        size_t total_size = map.required_span_size();
        T* ptr = allocator.allocate(total_size, alignment);
        Deleter del{
            [allocator](T* p, size_t sz) mutable { allocator.deallocate(p, sz); },
            total_size
        };
        owned_data = std::unique_ptr<T[], Deleter>(ptr, std::move(del));
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;

    constexpr T* data() { return owned_data.get(); }
    constexpr const T* data() const { return owned_data.get(); }
    constexpr T* data_handle() { return owned_data.get(); }
    constexpr const T* data_handle() const { return owned_data.get(); }

    constexpr const mapping_type& mapping() const { return map; }

    constexpr size_t extent(size_t dim) const { return map.extents().extent(dim); }

    constexpr auto view() { return mdspan_type{data(), map}; }
    constexpr auto view() const { return std::mdspan<const T, Extents, Layout>{data(), map}; }

    constexpr operator mdspan_type() { return view(); }
    constexpr operator std::mdspan<const T, Extents, Layout>() const { return view(); }

    constexpr T& operator[](size_t i, size_t j) { return data()[map(i, j)]; }
    constexpr const T& operator[](size_t i, size_t j) const { return data()[map(i, j)]; }
};

export template<RawStorage T, TensorAllocator Alloc>
auto make_tensor(size_t rows, size_t cols, Alloc allocator, size_t alignment = 64) {
    using Extents = std::dextents<size_t, 2>;
    using Layout = std::layout_right;
    return Tensor<T, Extents, Layout>(Extents{rows, cols}, allocator, alignment);
}

export template<RawStorage T>
auto make_tensor(size_t rows, size_t cols, size_t alignment = 64) {
    using Extents = std::dextents<size_t, 2>;
    using Layout = std::layout_right;
    return Tensor<T, Extents, Layout>(Extents{rows, cols}, AlignedAllocator<T>{}, alignment);
}

export template<RawStorage T, TensorAllocator Alloc>
auto make_tensor_strided(size_t rows, size_t cols, size_t row_stride, Alloc allocator, size_t alignment = 64) {
    using Extents = std::dextents<size_t, 2>;
    using Layout = std::layout_stride;
    std::array<size_t, 2> strides{row_stride, 1};
    return Tensor<T, Extents, Layout>(typename Layout::template mapping<Extents>{Extents{rows, cols}, strides}, allocator, alignment);
}
