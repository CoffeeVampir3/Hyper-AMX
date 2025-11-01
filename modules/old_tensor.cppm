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
#include <stdexcept>
#include <format>
export module tensor;
import avx512;

using std::mdspan;
using dextents2d = std::dextents<size_t, 2>;
using layout_right = std::layout_right;
using layout_left = std::layout_left;

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

template<typename Layout>
concept ContiguousRowMajorLayout = requires {
    typename Layout::template mapping<std::dextents<size_t, 2>>;
} && std::same_as<
    typename Layout::template mapping<std::dextents<size_t, 2>>,
    typename std::layout_right::template mapping<std::dextents<size_t, 2>>
>;

export constexpr size_t align_up(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

export constexpr bool is_aligned(size_t value, size_t alignment) {
    return (value % alignment) == 0;
}

namespace detail {

template<typename T, typename Extents, typename Layout>
using Mdspan = std::mdspan<T, Extents, Layout>;

constexpr void validate_tile_alignment(size_t row_start, size_t row_count, size_t tile_m) {
    if (row_start % tile_m != 0) {
        throw std::invalid_argument(std::format(
            "QuantizedTensor row slice start {} must be aligned to TILE_M={}",
            row_start, tile_m));
    }
    if (row_count % tile_m != 0) {
        throw std::invalid_argument(std::format(
            "QuantizedTensor row slice count {} must be multiple of TILE_M={}",
            row_count, tile_m));
    }
}

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
        auto src = self().view();
        auto dst = dest.view();
        Layout::copy_from(src, dst, dim, offset, size);
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
        auto scale = scale_for_dim(dim);
        if (offset % scale != 0) {
            throw std::invalid_argument(std::format(
                "QuantizedTensor slice offset {} must be aligned to tile boundary {}",
                offset, scale));
        }
        if (size % scale != 0) {
            throw std::invalid_argument(std::format(
                "QuantizedTensor slice size {} must be multiple of tile size {}",
                size, scale));
        }

        auto src_data = self().view();
        auto dst_data = dest.view();
        Layout::copy_from(src_data, dst_data, dim, offset, size);

        auto src_scales = self().scales_view();
        auto dst_scales = dest.scales_view();
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
        return static_cast<T*>(::operator new[](count * sizeof(T), std::align_val_t{64}));
    }
    void deallocate(T* ptr, size_t count) const {
        ::operator delete[](ptr, count * sizeof(T), std::align_val_t{64});
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

    template<typename Self>
    auto subview(this Self&& self, size_t row_start, size_t row_count) {
        static_assert(SliceableLayout<Layout>, "Cannot slice VNNI layout tensors");
        auto v = self.view();
        using elem_type = std::remove_reference_t<decltype(*v.data_handle())>;
        elem_type* slice_ptr = v.data_handle() + row_start * v.extent(1);
        using slice_extents = std::dextents<size_t, 2>;
        auto sliced = std::mdspan<elem_type, slice_extents, Layout>(slice_ptr, row_count, v.extent(1));
        return TensorView<elem_type, slice_extents, Layout>(sliced);
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

    template<typename Self>
    auto subview(this Self&& self, size_t row_start, size_t row_count) {
        static_assert(SliceableLayout<DataLayout>, "Cannot slice VNNI layout tensors");
        detail::validate_tile_alignment(row_start, row_count, TILE_M);
        auto data_slice = self.data.subview(row_start, row_count);
        auto scales_slice = self.group_quant_scales.subview(row_start / TILE_M, row_count / TILE_M);
        using data_slice_extents = typename decltype(data_slice)::extents_type;
        using scales_slice_extents = typename decltype(scales_slice)::extents_type;
        using data_elem_type = std::remove_reference_t<decltype(*data_slice.view().data_handle())>;
        using scales_elem_type = std::remove_reference_t<decltype(*scales_slice.view().data_handle())>;
        return QuantizedTensorView<data_elem_type, data_slice_extents, DataLayout, scales_slice_extents, scales_elem_type, TILE_M, TILE_N>(
            data_slice.view(), scales_slice.view());
    }
};

export namespace tensor_ops {

namespace detail {
template<typename T, typename Extents, typename Layout>
void pad_implementation(const Tensor<T, Extents, Layout>& source,
                       Tensor<T, Extents, Layout>& padded,
                       size_t orig_M, size_t orig_N,
                       size_t padded_M, size_t padded_N,
                       T fill_value)
{
    auto src_view = source.view();
    auto dst_view = padded.view();

    for (size_t i = 0; i < orig_M; i++) {
        avx512::contiguous_copy(&dst_view[i, 0], &src_view[i, 0], orig_N);
    }

    if constexpr (std::same_as<T, int8_t> || std::same_as<T, int32_t> || std::same_as<T, _Float16>) {
        if (padded_N > orig_N) {
            avx512::fill_padding_cols(&dst_view[0, 0], padded_N, orig_M, orig_N, padded_N - orig_N, fill_value);
        }
        if (padded_M > orig_M) {
            avx512::fill_padding_rows(&dst_view[orig_M, 0], padded_M - orig_M, padded_N, fill_value);
        }
    } else {
        for (size_t i = 0; i < orig_M; i++) {
            for (size_t j = orig_N; j < padded_N; j++) {
                dst_view[i, j] = fill_value;
            }
        }
        for (size_t i = orig_M; i < padded_M; i++) {
            for (size_t j = 0; j < padded_N; j++) {
                dst_view[i, j] = fill_value;
            }
        }
    }
}
}

template<typename T, typename Extents, typename Layout>
Tensor<T, Extents, Layout> pad_to_alignment(
    const Tensor<T, Extents, Layout>& source,
    size_t row_alignment,
    size_t col_alignment,
    T fill_value = T{0})
{
    static_assert(Extents::rank() == 2, "pad_to_alignment only supports 2D tensors");
    static_assert(SliceableLayout<Layout>, "Cannot pad VNNI layout tensors directly");
    static_assert(ContiguousRowMajorLayout<Layout>, "pad_to_alignment requires contiguous row-major layout");

    size_t orig_M = source.extent(0);
    size_t orig_N = source.extent(1);
    size_t padded_M = align_up(orig_M, row_alignment);
    size_t padded_N = align_up(orig_N, col_alignment);

    auto padded = Tensor<T, Extents, Layout>(Extents{padded_M, padded_N});
    detail::pad_implementation(source, padded, orig_M, orig_N, padded_M, padded_N, fill_value);
    return padded;
}

template<typename T, typename Extents, typename Layout, typename Alloc>
Tensor<T, Extents, Layout> pad_to_alignment(
    const Tensor<T, Extents, Layout>& source,
    size_t row_alignment,
    size_t col_alignment,
    Alloc allocator,
    T fill_value = T{0})
{
    static_assert(Extents::rank() == 2, "pad_to_alignment only supports 2D tensors");
    static_assert(SliceableLayout<Layout>, "Cannot pad VNNI layout tensors directly");
    static_assert(ContiguousRowMajorLayout<Layout>, "pad_to_alignment requires contiguous row-major layout");

    size_t orig_M = source.extent(0);
    size_t orig_N = source.extent(1);
    size_t padded_M = align_up(orig_M, row_alignment);
    size_t padded_N = align_up(orig_N, col_alignment);

    auto padded = Tensor<T, Extents, Layout>(Extents{padded_M, padded_N}, allocator);
    detail::pad_implementation(source, padded, orig_M, orig_N, padded_M, padded_N, fill_value);
    return padded;
}

template<typename T, typename Extents, typename Layout>
Tensor<T, Extents, Layout> gather_rows(
    const Tensor<T, Extents, Layout>& source,
    const size_t* row_indices,
    size_t num_rows)
{
    static_assert(Extents::rank() == 2, "gather_rows only supports 2D tensors");
    static_assert(ContiguousRowMajorLayout<Layout>, "gather_rows requires contiguous row-major layout");

    size_t N = source.extent(1);
    auto gathered = Tensor<T, Extents, Layout>(Extents{num_rows, N});

    auto src_view = source.view();
    auto dst_view = gathered.view();

    avx512::gather_rows_impl(&src_view[0, 0], N, &dst_view[0, 0], N, row_indices, num_rows);

    return gathered;
}

template<typename T, typename Extents, typename Layout, typename IndexContainer>
Tensor<T, Extents, Layout> gather_rows(
    const Tensor<T, Extents, Layout>& source,
    const IndexContainer& row_indices)
{
    return gather_rows(source, row_indices.data(), row_indices.size());
}

}
