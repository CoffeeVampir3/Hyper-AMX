module;
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <concepts>
#include <mdspan>
#include <memory>
#include <functional>
#include <type_traits>
#include <utility>
export module tensor;

export template<typename T>
concept TensorStorage = requires(T t, int dim) {
    typename T::extents_type;
    { t.extent(dim) } -> std::same_as<size_t>;
    { t.view() };
};

template<typename Layout>
concept VNNILayout = requires {
    typename Layout::is_vnni_layout;
};

template<typename Layout>
concept SliceableLayout = !VNNILayout<Layout>;

namespace detail {

template<typename T, typename Extents, typename Layout>
using Mdspan = std::mdspan<T, Extents, Layout>;

template<typename Derived, typename Layout>
struct TensorCopyOps {
    constexpr const Derived& self() const {
        return static_cast<const Derived&>(*this);
    }

    template<TensorStorage Dest>
    void copy_to(Dest& dest) const {
        auto src = self().view();
        auto dst = dest.view();
        Layout::copy_from(src, dst, 0, 0, src.extent(0));
    }

    template<TensorStorage Dest>
    void copy_slice_to(Dest& dest, int dim, size_t offset, size_t size) const {
        Layout::copy_from(self().view(), dest.view(), dim, offset, size);
    }
};

template<typename Derived, typename Layout, size_t TileRows, size_t TileCols>
struct QuantizedCopyOps {
    static constexpr size_t tile_rows = TileRows;
    static constexpr size_t tile_cols = TileCols;

    static constexpr size_t scale_for_dim(int dim) {
        return dim == 0 ? tile_rows : tile_cols;
    }

    constexpr const Derived& self() const {
        return static_cast<const Derived&>(*this);
    }

    template<typename Dest>
    void copy_to(Dest& dest) const requires requires(Dest& d) { d.view(); d.scales_view(); } {
        auto src_data = self().view();
        auto dst_data = dest.view();
        Layout::copy_from(src_data, dst_data, 0, 0, src_data.extent(0));

        auto src_scales = self().scales_view();
        auto dst_scales = dest.scales_view();
        Layout::copy_from(src_scales, dst_scales, 0, 0, src_scales.extent(0));
    }

    template<typename Dest>
    void copy_slice_to(Dest& dest, int dim, size_t offset, size_t size) const requires requires(Dest& d) { d.view(); d.scales_view(); } {
        auto src_data = self().view();
        auto dst_data = dest.view();
        Layout::copy_from(src_data, dst_data, dim, offset, size);

        auto src_scales = self().scales_view();
        auto dst_scales = dest.scales_view();
        auto scale = scale_for_dim(dim);
        Layout::copy_from(src_scales, dst_scales, dim, offset / scale, size / scale);
    }
};

}

export template<typename T, typename Extents, typename Layout>
struct TensorView : detail::TensorCopyOps<TensorView<T, Extents, Layout>, Layout> {
    using base = detail::TensorCopyOps<TensorView<T, Extents, Layout>, Layout>;
    using extents_type = Extents;
    using mdspan_type = detail::Mdspan<T, Extents, Layout>;
    using const_mdspan_type = detail::Mdspan<std::add_const_t<T>, Extents, Layout>;
    using base::copy_slice_to;
    using base::copy_to;

    mdspan_type view_;

    TensorView(mdspan_type view) : view_(view) {}

    size_t extent(size_t dim) const { return view_.extent(dim); }
    auto view() { return view_; }
    auto view() const { return const_mdspan_type(view_); }
};

export template<typename T, typename DataExtents, typename DataLayout, typename ScaleExtents, typename QuantScaleType, size_t TileRows, size_t TileCols>
struct QuantizedTensorView : detail::QuantizedCopyOps<QuantizedTensorView<T, DataExtents, DataLayout, ScaleExtents, QuantScaleType, TileRows, TileCols>, DataLayout, TileRows, TileCols> {
    using base = detail::QuantizedCopyOps<QuantizedTensorView<T, DataExtents, DataLayout, ScaleExtents, QuantScaleType, TileRows, TileCols>, DataLayout, TileRows, TileCols>;
    using extents_type = DataExtents;
    using data_mdspan_type = detail::Mdspan<T, DataExtents, DataLayout>;
    using const_data_mdspan_type = detail::Mdspan<std::add_const_t<T>, DataExtents, DataLayout>;
    using scales_mdspan_type = detail::Mdspan<QuantScaleType, ScaleExtents, DataLayout>;
    using const_scales_mdspan_type = detail::Mdspan<std::add_const_t<QuantScaleType>, ScaleExtents, DataLayout>;
    using base::copy_slice_to;
    using base::copy_to;
    using base::scale_for_dim;

    static constexpr size_t TILE_M = TileRows;
    static constexpr size_t TILE_N = TileCols;

    data_mdspan_type data_view_;
    scales_mdspan_type scales_view_;

    QuantizedTensorView(data_mdspan_type data_view, scales_mdspan_type scales_view)
        : data_view_(data_view), scales_view_(scales_view) {}

    size_t extent(size_t dim) const { return data_view_.extent(dim); }
    auto view() { return data_view_; }
    auto view() const { return const_data_mdspan_type(data_view_); }
    auto scales_view() { return scales_view_; }
    auto scales_view() const { return const_scales_mdspan_type(scales_view_); }
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
        size_t byte_size = count * sizeof(T);
        size_t aligned_size = (byte_size + 63) & ~size_t(63);
        return static_cast<T*>(std::aligned_alloc(64, aligned_size));
    }
    void deallocate(T* ptr, size_t) const {
        std::free(ptr);
    }
};

export template<typename T, typename Extents, typename Layout>
struct Tensor : detail::TensorCopyOps<Tensor<T, Extents, Layout>, Layout> {
    using base = detail::TensorCopyOps<Tensor<T, Extents, Layout>, Layout>;
    using self_type = Tensor<T, Extents, Layout>;
    using extents_type = Extents;
    using mapping_type = typename Layout::template mapping<Extents>;
    using mdspan_type = detail::Mdspan<T, Extents, Layout>;
    using const_mdspan_type = detail::Mdspan<std::add_const_t<T>, Extents, Layout>;
    using base::copy_slice_to;
    using base::copy_to;

    struct Deleter {
        std::function<void(T*, size_t)> del_fn;
        size_t size;
        void operator()(T* p) const {
            if (p && del_fn) {
                del_fn(p, size);
            }
        }
    };

    mapping_type map;
    std::unique_ptr<T[], Deleter> data;

    Tensor(Extents extents) : Tensor(extents, DefaultAllocator<T>{}) {}

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
    auto view() const { return const_mdspan_type{data.get(), map}; }

    auto subview(size_t row_start, size_t row_count) {
        static_assert(SliceableLayout<Layout>, "Cannot slice VNNI layout tensors");
        auto v = view();
        T* slice_ptr = v.data_handle() + row_start * v.extent(1);
        using slice_extents = std::dextents<size_t, 2>;
        auto sliced = std::mdspan<T, slice_extents, Layout>(slice_ptr, row_count, v.extent(1));
        return TensorView<T, slice_extents, Layout>(sliced);
    }

    auto subview(size_t row_start, size_t row_count) const {
        static_assert(SliceableLayout<Layout>, "Cannot slice VNNI layout tensors");
        auto v = view();
        const T* slice_ptr = v.data_handle() + row_start * v.extent(1);
        using slice_extents = std::dextents<size_t, 2>;
        auto sliced = std::mdspan<const T, slice_extents, Layout>(slice_ptr, row_count, v.extent(1));
        return TensorView<const T, slice_extents, Layout>(sliced);
    }

};

export template<typename T, typename DataExtents, typename DataLayout, typename QuantScaleType, size_t TileRows = 16, size_t TileCols = 16>
struct QuantizedTensor : detail::QuantizedCopyOps<QuantizedTensor<T, DataExtents, DataLayout, QuantScaleType, TileRows, TileCols>, DataLayout, TileRows, TileCols> {
    using base = detail::QuantizedCopyOps<QuantizedTensor<T, DataExtents, DataLayout, QuantScaleType, TileRows, TileCols>, DataLayout, TileRows, TileCols>;
    using self_type = QuantizedTensor<T, DataExtents, DataLayout, QuantScaleType, TileRows, TileCols>;
    using extents_type = DataExtents;
    using scale_extents_type = std::dextents<size_t, 2>;
    using data_tensor_type = Tensor<T, DataExtents, DataLayout>;
    using scales_tensor_type = Tensor<QuantScaleType, scale_extents_type, DataLayout>;
    using base::copy_slice_to;
    using base::copy_to;
    using base::scale_for_dim;

    static constexpr size_t TILE_M = TileRows;
    static constexpr size_t TILE_N = TileCols;

    data_tensor_type data;
    scales_tensor_type group_quant_scales;

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

    auto subview(size_t row_start, size_t row_count) {
        static_assert(SliceableLayout<DataLayout>, "Cannot slice VNNI layout tensors");
        auto data_slice = data.subview(row_start, row_count);
        auto scales_slice = group_quant_scales.subview(row_start / TILE_M, row_count / TILE_M);
        using data_slice_extents = typename decltype(data_slice)::extents_type;
        using scales_slice_extents = typename decltype(scales_slice)::extents_type;
        return QuantizedTensorView<T, data_slice_extents, DataLayout, scales_slice_extents, QuantScaleType, TILE_M, TILE_N>(
            data_slice.view(), scales_slice.view());
    }

    auto subview(size_t row_start, size_t row_count) const {
        static_assert(SliceableLayout<DataLayout>, "Cannot slice VNNI layout tensors");
        auto data_slice = data.subview(row_start, row_count);
        auto scales_slice = group_quant_scales.subview(row_start / TILE_M, row_count / TILE_M);
        using data_slice_extents = typename decltype(data_slice)::extents_type;
        using scales_slice_extents = typename decltype(scales_slice)::extents_type;
        return QuantizedTensorView<const T, data_slice_extents, DataLayout, scales_slice_extents, const QuantScaleType, TILE_M, TILE_N>(
            data_slice.view(), scales_slice.view());
    }
};
