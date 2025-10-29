module;
#include <cstddef>
#include <cstdlib>
#include <concepts>
#include <mdspan>
#include <memory>
#include <functional>
export module tensor;

export template<typename T>
concept TensorStorage = requires(T t, int dim) {
    typename T::extents_type;
    { t.extent(dim) } -> std::same_as<size_t>;
    { t.view() };
};

export template<typename Src, typename Dst>
concept CompatibleTensorStorage = TensorStorage<Src> && TensorStorage<Dst> &&
    requires(const Src& src, Dst& dst, int dim, size_t offset, size_t size) {
        { src.copy_to(dst) } -> std::same_as<void>;
        { src.copy_slice_to(dst, dim, offset, size) } -> std::same_as<void>;
    };

export template<typename A, typename T>
concept Allocator = requires(A a, size_t n) {
    { a.allocate(n) } -> std::same_as<T*>;
    { a.deallocate(std::declval<T*>(), n) } -> std::same_as<void>;
};

export template<typename T>
struct DefaultAllocator {
    T* allocate(size_t count) const {
        return static_cast<T*>(std::aligned_alloc(64, count * sizeof(T)));
    }
    void deallocate(T* ptr, size_t) const {
        std::free(ptr);
    }
};

export template<typename T, typename Extents, typename Layout>
struct Tensor {
    using self_type = Tensor<T, Extents, Layout>;
    using extents_type = Extents;
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
    std::unique_ptr<T[], Deleter> data;

    Tensor(Extents extents) : map(extents) {
        size_t size = map.required_span_size();
        T* ptr = static_cast<T*>(std::aligned_alloc(64, size * sizeof(T)));
        Deleter del{[](T* p, size_t) { std::free(p); }, size};
        data = std::unique_ptr<T[], Deleter>(ptr, std::move(del));
    }

    template<Allocator<T> Alloc>
    Tensor(Extents extents, Alloc allocator) : map(extents) {
        size_t size = map.required_span_size();
        T* ptr = allocator.allocate(size);
        Deleter del{
            [allocator](T* p, size_t sz) mutable { allocator.deallocate(p, sz); },
            size
        };
        data = std::unique_ptr<T[], Deleter>(ptr, std::move(del));
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;

    size_t extent(size_t dim) const { return map.extents().extent(dim); }
    auto view() { return mdspan_type{data.get(), map}; }
    auto view() const { return std::mdspan<const T, Extents, Layout>{data.get(), map}; }

    template<TensorStorage Dest>
    void copy_to(Dest& dest) const {
        auto src = view();
        auto dst = dest.view();
        for (size_t i = 0; i < src.extent(0); i++) {
            for (size_t j = 0; j < src.extent(1); j++) {
                dst[i, j] = src[i, j];
            }
        }
    }

    template<TensorStorage Dest>
    void copy_slice_to(Dest& dest, int dim, size_t offset, size_t size) const {
        auto src = view();
        auto dst = dest.view();
        Layout::copy_from(src, dst, dim, offset, size);
    }
};

export template<typename T, typename DataExtents, typename DataLayout, typename QuantScaleType, size_t TileRows = 16, size_t TileCols = 16>
struct QuantizedTensor {
    using self_type = QuantizedTensor<T, DataExtents, DataLayout, QuantScaleType, TileRows, TileCols>;
    using extents_type = DataExtents;
    using scale_extents_type = std::dextents<size_t, 2>;

    static constexpr size_t TILE_M = TileRows;
    static constexpr size_t TILE_N = TileCols;

    static constexpr size_t scale_for_dim(int dim) {
        return (dim == 0) ? TILE_M : TILE_N;
    }

    Tensor<T, DataExtents, DataLayout> data;
    Tensor<QuantScaleType, scale_extents_type, DataLayout> group_quant_scales;

    QuantizedTensor(DataExtents data_extents)
        : data(data_extents)
        , group_quant_scales(scale_extents_type{data_extents.extent(0) / TILE_M, data_extents.extent(1) / TILE_N})
    {}

    template<Allocator<T> DataAlloc, Allocator<QuantScaleType> ScaleAlloc>
    QuantizedTensor(DataExtents data_extents, DataAlloc data_alloc, ScaleAlloc scale_alloc)
        : data(data_extents, data_alloc)
        , group_quant_scales(scale_extents_type{data_extents.extent(0) / TILE_M, data_extents.extent(1) / TILE_N}, scale_alloc)
    {}

    QuantizedTensor(const QuantizedTensor&) = delete;
    QuantizedTensor& operator=(const QuantizedTensor&) = delete;
    QuantizedTensor(QuantizedTensor&&) = default;
    QuantizedTensor& operator=(QuantizedTensor&&) = default;

    size_t extent(size_t dim) const { return data.extent(dim); }
    auto view() { return data.view(); }
    auto view() const { return data.view(); }
    auto scales_view() { return group_quant_scales.view(); }
    auto scales_view() const { return group_quant_scales.view(); }

    void copy_to(QuantizedTensor& dest) const {
        data.copy_to(dest.data);
        group_quant_scales.copy_to(dest.group_quant_scales);
    }

    void copy_slice_to(QuantizedTensor& dest, int dim, size_t offset, size_t size) const {
        size_t scale = scale_for_dim(dim);
        data.copy_slice_to(dest.data, dim, offset, size);
        group_quant_scales.copy_slice_to(dest.group_quant_scales, dim, offset / scale, size / scale);
    }
};
