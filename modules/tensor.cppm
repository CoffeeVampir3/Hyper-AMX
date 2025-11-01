module;
#include <array>
#include <cstddef>
#include <concepts>
#include <cstdlib>
#include <mdspan>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
export module hyperamx.tensor;

import hyperamx.layout;

export namespace tensor {

template<typename T>
struct DefaultAllocator {
    using value_type = T;
    T* allocate(size_t count) {
        size_t bytes = count * sizeof(T);
        size_t aligned = (bytes + 63) & ~size_t(63);
        if (aligned == 0) aligned = 64;
        if (auto* ptr = static_cast<T*>(std::aligned_alloc(64, aligned))) return ptr;
        throw std::bad_alloc();
    }
    void deallocate(T* ptr, size_t) {
        std::free(ptr);
    }
};

template<typename Layout>
concept VNNILayout = requires { typename Layout::is_vnni_layout; };

template<typename Layout>
concept SliceableLayout = !VNNILayout<Layout>;

template<typename Extents>
constexpr size_t compute_required_span_size(Extents extents, std::array<size_t, Extents::rank()> strides) {
    constexpr size_t rank = Extents::rank();
    size_t span_size = 0;
    bool zero_extent = false;
    for (size_t i = 0; i < rank; i++) {
        if (extents.extent(i) == 0) {
            zero_extent = true;
            break;
        }
    }
    if (!zero_extent) {
        size_t max_index = 0;
        for (size_t i = 0; i < rank; i++) {
            max_index += (extents.extent(i) - 1) * strides[i];
        }
        span_size = max_index + 1;
    }
    return span_size;
}

template<typename T, typename Allocator = DefaultAllocator<T>>
struct OwnedStorage {
    using element_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using allocator_type = Allocator;
    static_assert(std::is_move_constructible_v<Allocator> || std::is_copy_constructible_v<Allocator>);
    static_assert(std::is_move_assignable_v<Allocator> || std::is_copy_assignable_v<Allocator>);

    OwnedStorage() = default;
    explicit OwnedStorage(size_t count, Allocator alloc = {})
        : alloc_(std::move(alloc)), ptr_(alloc_.allocate(count)), count_(count) {}

    OwnedStorage(const OwnedStorage&) = delete;
    OwnedStorage& operator=(const OwnedStorage&) = delete;

    OwnedStorage(OwnedStorage&& other) noexcept
        : alloc_(steal_allocator(other.alloc_)), ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    OwnedStorage& operator=(OwnedStorage&& other) noexcept {
        if (this != &other) {
            release();
            assign_allocator(other.alloc_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    ~OwnedStorage() {
        release();
    }

    pointer data() { return ptr_; }
    const_pointer data() const { return ptr_; }
    size_t size() { return count_; }
    size_t size() const { return count_; }
    allocator_type& allocator() { return alloc_; }
    const allocator_type& allocator() const { return alloc_; }

private:
    void release() {
        if (ptr_) {
            alloc_.deallocate(ptr_, count_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    static allocator_type steal_allocator(allocator_type& alloc) {
        if constexpr (std::is_move_constructible_v<allocator_type>) {
            return std::move(alloc);
        } else {
            return alloc;
        }
    }

    void assign_allocator(allocator_type& alloc) {
        if constexpr (std::is_move_assignable_v<allocator_type>) {
            alloc_ = std::move(alloc);
        } else {
            alloc_ = alloc;
        }
    }

    allocator_type alloc_{};
    pointer ptr_{nullptr};
    size_t count_{0};
};

template<typename T>
struct BorrowedStorage {
    using element_type = T;
    using pointer = T*;
    using const_pointer = const T*;

    BorrowedStorage() = default;
    explicit BorrowedStorage(pointer ptr) : ptr_(ptr) {}

    pointer data() { return ptr_; }
    const_pointer data() const { return ptr_; }

private:
    pointer ptr_{nullptr};
};

namespace detail {

struct NoAllocator {};

struct stride_init_tag {};
struct mapping_init_tag {};

template<typename Storage>
struct storage_traits {
    using allocator_type = NoAllocator;
    static constexpr bool owns = false;
};

template<typename T, typename Alloc>
struct storage_traits<OwnedStorage<T, Alloc>> {
    using allocator_type = Alloc;
    static constexpr bool owns = true;

    template<typename Extents, typename Strides>
    static auto make_with_strides(Extents extents,
                                  Strides strides,
                                  allocator_type alloc) {
        size_t span_size = compute_required_span_size(extents, strides);
        return OwnedStorage<T, Alloc>(span_size, std::move(alloc));
    }

    template<typename Mapping>
    static auto make_with_mapping(Mapping mapping, allocator_type alloc) {
        return OwnedStorage<T, Alloc>(mapping.required_span_size(), std::move(alloc));
    }
};

} // namespace detail

template<typename Storage>
concept StoragePolicy = requires(Storage storage) {
    typename Storage::element_type;
    typename Storage::pointer;
    { storage.data() } -> std::same_as<typename Storage::pointer>;
};

template<typename T, typename Extents, typename Layout, StoragePolicy Storage>
struct BasicTensor {
    using element_type = T;
    using extents_type = Extents;
    using layout_type = Layout;
    using storage_type = Storage;
    static constexpr size_t rank = Extents::rank();
    static constexpr bool uses_stride =
        std::same_as<Layout, ::Layout::RowMajor> || std::same_as<Layout, ::Layout::ColumnMajor>;
    using strides_type = std::array<size_t, rank>;
    using mapping_type = typename Layout::template mapping<Extents>;
    using mdspan_type = std::conditional_t<uses_stride,
        std::mdspan<T, Extents, std::layout_stride>,
        std::mdspan<T, Extents, Layout>>;
    using const_mdspan_type = std::conditional_t<uses_stride,
        std::mdspan<const T, Extents, std::layout_stride>,
        std::mdspan<const T, Extents, Layout>>;
    using storage_traits = detail::storage_traits<storage_type>;
    using allocator_type = typename storage_traits::allocator_type;
    static constexpr bool has_allocator = storage_traits::owns;

    BasicTensor() = default;

    BasicTensor(storage_type storage, Extents extents, strides_type strides)
        requires uses_stride
        : storage_(std::move(storage)), extents_(std::move(extents)), strides_(std::move(strides)) {}

    BasicTensor(storage_type storage, mapping_type mapping)
        requires (!uses_stride)
        : storage_(std::move(storage)), extents_(mapping.extents()), mapping_(std::move(mapping)) {}

    BasicTensor(Extents extents)
        requires (has_allocator && uses_stride)
        : BasicTensor(detail::stride_init_tag{}, extents, Layout::default_strides(extents), allocator_type{}) {}

    BasicTensor(Extents extents, allocator_type alloc)
        requires (has_allocator && uses_stride)
        : BasicTensor(detail::stride_init_tag{}, extents, Layout::default_strides(extents), std::move(alloc)) {}

    BasicTensor(Extents extents, strides_type strides)
        requires (has_allocator && uses_stride)
        : BasicTensor(detail::stride_init_tag{}, extents, strides, allocator_type{}) {}

    BasicTensor(Extents extents, strides_type strides, allocator_type alloc)
        requires (has_allocator && uses_stride)
        : BasicTensor(detail::stride_init_tag{}, extents, strides, std::move(alloc)) {}

    BasicTensor(Extents extents)
        requires (has_allocator && !uses_stride)
        : BasicTensor(detail::mapping_init_tag{}, mapping_type(extents), allocator_type{}) {}

    BasicTensor(Extents extents, allocator_type alloc)
        requires (has_allocator && !uses_stride)
        : BasicTensor(detail::mapping_init_tag{}, mapping_type(extents), std::move(alloc)) {}

    size_t extent(size_t dim) {
        if constexpr (uses_stride) {
            return extents_.extent(dim);
        } else {
            return mapping_.extents().extent(dim);
        }
    }

    size_t extent(size_t dim) const {
        if constexpr (uses_stride) {
            return extents_.extent(dim);
        } else {
            return mapping_.extents().extent(dim);
        }
    }

    auto extents() {
        if constexpr (uses_stride) {
            return extents_;
        } else {
            return mapping_.extents();
        }
    }

    auto extents() const {
        if constexpr (uses_stride) {
            return extents_;
        } else {
            return mapping_.extents();
        }
    }

    auto& strides() requires uses_stride { return strides_; }
    const auto& strides() const requires uses_stride { return strides_; }
    size_t stride(size_t dim) requires uses_stride { return strides_[dim]; }
    size_t stride(size_t dim) const requires uses_stride { return strides_[dim]; }

    mdspan_type span() {
        if constexpr (uses_stride) {
            auto mapping = std::layout_stride::mapping<Extents>(extents_, strides_);
            return mdspan_type{storage_.data(), mapping};
        } else {
            return mdspan_type{storage_.data(), mapping_};
        }
    }

    const_mdspan_type span() const {
        if constexpr (uses_stride) {
            auto mapping = std::layout_stride::mapping<Extents>(extents_, strides_);
            return const_mdspan_type{storage_.data(), mapping};
        } else {
            return const_mdspan_type{storage_.data(), mapping_};
        }
    }

    storage_type& storage() { return storage_; }
    const storage_type& storage() const { return storage_; }

    storage_type storage_{};
    Extents extents_{};
    strides_type strides_{};
    mapping_type mapping_{Extents{}};

private:
    BasicTensor(detail::stride_init_tag,
                Extents extents,
                strides_type strides,
                allocator_type alloc)
        requires (has_allocator && uses_stride)
        : storage_(storage_traits::make_with_strides(extents, strides, std::move(alloc))),
          extents_(std::move(extents)),
          strides_(std::move(strides)) {}

    BasicTensor(detail::mapping_init_tag,
                mapping_type mapping,
                allocator_type alloc)
        requires (has_allocator && !uses_stride)
        : storage_(storage_traits::make_with_mapping(mapping, std::move(alloc))),
          extents_(mapping.extents()),
          mapping_(std::move(mapping)) {}
};

template<typename Tensor>
concept TensorHandle = requires(Tensor tensor, size_t dim) {
    typename Tensor::element_type;
    typename Tensor::extents_type;
    typename Tensor::layout_type;
    { tensor.extent(dim) } -> std::convertible_to<size_t>;
    { tensor.span() };
};

template<typename T, typename Extents, typename Layout, typename Allocator = DefaultAllocator<T>>
auto make_tensor(Extents extents, Allocator alloc = {}) {
    using storage_type = OwnedStorage<T, Allocator>;
    using tensor_type = BasicTensor<T, Extents, Layout, storage_type>;
    return tensor_type(extents, std::move(alloc));
}

template<typename T, typename Extents, typename Layout, typename Allocator = DefaultAllocator<T>>
auto make_tensor(Extents extents, std::array<size_t, Extents::rank()> strides, Allocator alloc = {}) {
    using storage_type = OwnedStorage<T, Allocator>;
    using tensor_type = BasicTensor<T, Extents, Layout, storage_type>;
    static_assert(tensor_type::uses_stride, "Custom stride construction requires stride-based layout");
    return tensor_type(extents, strides, std::move(alloc));
}

template<typename T, typename Extents, typename Layout>
auto make_tensor_view(std::mdspan<T, Extents, Layout> view) {
    using storage_type = BorrowedStorage<T>;
    using tensor_type = BasicTensor<T, Extents, Layout, storage_type>;
    storage_type storage(view.data_handle());
    if constexpr (tensor_type::uses_stride) {
        constexpr size_t rank = Extents::rank();
        std::array<size_t, rank> strides{};
        auto mapping = view.mapping();
        for (size_t i = 0; i < rank; i++) {
            strides[i] = mapping.stride(i);
        }
        return tensor_type{std::move(storage), view.extents(), strides};
    } else {
        using mapping_type = typename tensor_type::mapping_type;
        mapping_type mapping(view.mapping());
        return tensor_type{std::move(storage), mapping};
    }
}

template<typename T, typename Extents, typename Layout, typename Allocator = DefaultAllocator<T>>
using Tensor = BasicTensor<T, Extents, Layout, OwnedStorage<T, Allocator>>;

template<typename T, typename Extents, typename Layout>
using MutTensorView = BasicTensor<T, Extents, Layout, BorrowedStorage<T>>;

template<typename T, typename Extents, typename Layout>
using TensorView = BasicTensor<const T, Extents, Layout, BorrowedStorage<const T>>;

template<TensorHandle Tensor>
requires SliceableLayout<typename Tensor::layout_type>
auto slice_rows(Tensor& tensor, size_t offset, size_t rows) {
    constexpr size_t rank = Tensor::extents_type::rank();
    static_assert(rank == 2);
    static_assert(Tensor::uses_stride, "slice_rows requires stride-based layout");
    if (offset + rows > tensor.extent(0)) {
        throw std::out_of_range("row slice exceeds tensor extent");
    }
    using layout_type = typename Tensor::layout_type;
    using element_type = typename Tensor::element_type;
    using slice_extents = std::dextents<size_t, rank>;
    slice_extents extents{rows, tensor.extent(1)};
    auto base = tensor.span();
    auto strides = tensor.strides();
    size_t linear = offset * strides[0];
    BorrowedStorage<element_type> storage(base.data_handle() + linear);
    using view_type = BasicTensor<element_type, slice_extents, layout_type, BorrowedStorage<element_type>>;
    return view_type{std::move(storage), extents, strides};
}

template<TensorHandle Src, TensorHandle Dst>
requires std::same_as<typename Src::layout_type, typename Dst::layout_type>
void copy_tensor(Src& src, Dst& dst) {
    static_assert(Src::extents_type::rank() == Dst::extents_type::rank());
    static_assert(Src::extents_type::rank() == 2);
    auto src_view = src.span();
    auto dst_view = dst.span();
    if (src_view.extent(0) != dst_view.extent(0) || src_view.extent(1) != dst_view.extent(1)) {
        throw std::invalid_argument("tensor extents must match for copy");
    }
    using layout_type = typename Src::layout_type;
    layout_type::copy_from(src_view, dst_view, 0, 0, src_view.extent(0));
}

template<TensorHandle Src, TensorHandle Dst>
requires std::same_as<typename Src::layout_type, typename Dst::layout_type>
void copy_tensor_slice(Src& src, Dst& dst, int dim, size_t offset, size_t size) {
    static_assert(Src::extents_type::rank() == Dst::extents_type::rank());
    static_assert(Src::extents_type::rank() == 2);
    auto src_view = src.span();
    auto dst_view = dst.span();
    if (dim < 0 || dim > 1) {
        throw std::invalid_argument("invalid copy dimension");
    }
    if (offset + size > src_view.extent(dim) || size > dst_view.extent(dim)) {
        throw std::out_of_range("copy slice exceeds tensor extent");
    }
    using layout_type = typename Src::layout_type;
    layout_type::copy_from(src_view, dst_view, dim, offset, size);
}

template<typename DataTensor, typename ScaleTensor, size_t TileRows, size_t TileCols>
struct QuantizedTensor {
    using data_type = DataTensor;
    using scale_type = ScaleTensor;
    using element_type = typename DataTensor::element_type;
    using extents_type = typename DataTensor::extents_type;
    using layout_type = typename DataTensor::layout_type;
    using self_type = QuantizedTensor<DataTensor, ScaleTensor, TileRows, TileCols>;
    template<typename NewData, typename NewScale>
    using rebind = QuantizedTensor<NewData, NewScale, TileRows, TileCols>;
    static constexpr size_t TILE_ROWS = TileRows;
    static constexpr size_t TILE_COLS = TileCols;

    DataTensor data;
    ScaleTensor scales;

    size_t extent(size_t dim) { return data.extent(dim); }
    size_t extent(size_t dim) const { return data.extent(dim); }
    auto span() { return data.span(); }
    auto span() const { return data.span(); }
    auto scale_span() { return scales.span(); }
    auto scale_span() const { return scales.span(); }
};

template<size_t TileRows, size_t TileCols>
constexpr bool quantized_slice_aligned(int dim, size_t offset, size_t size);

template<size_t TileRows, size_t TileCols>
constexpr std::pair<size_t, size_t> quantized_scale_slice(int dim, size_t offset, size_t size);

template<typename QT>
auto make_quantized_slice(QT&& tensor, size_t offset, size_t rows) {
    using tensor_type = std::remove_reference_t<QT>;
    static_assert(!std::is_const_v<tensor_type>, "quantized slices require mutable tensor handles");
    constexpr size_t TileRows = tensor_type::TILE_ROWS;
    constexpr size_t TileCols = tensor_type::TILE_COLS;
    if (!quantized_slice_aligned<TileRows, TileCols>(0, offset, rows)) {
        throw std::invalid_argument("quantized tensor slice must align to tile rows");
    }

    auto data_view = slice_rows(tensor.data, offset, rows);
    auto [scale_offset, scale_rows] = quantized_scale_slice<TileRows, TileCols>(0, offset, rows);
    auto scale_view = slice_rows(tensor.scales, scale_offset, scale_rows);

    using data_view_type = decltype(data_view);
    using scale_view_type = decltype(scale_view);
    using view_type = typename tensor_type::template rebind<data_view_type, scale_view_type>;
    return view_type{std::move(data_view), std::move(scale_view)};
}

template<typename T>
concept QuantTensor = requires(T tensor) {
    typename T::data_type;
    typename T::scale_type;
    typename T::layout_type;
    { T::TILE_ROWS } -> std::convertible_to<size_t>;
    { T::TILE_COLS } -> std::convertible_to<size_t>;
    tensor.data;
    tensor.scales;
};

template<typename Src, typename Dst>
concept QuantTensorPair =
    QuantTensor<Src> &&
    QuantTensor<Dst> &&
    (Src::TILE_ROWS == Dst::TILE_ROWS) &&
    (Src::TILE_COLS == Dst::TILE_COLS) &&
    std::same_as<typename Src::layout_type, typename Dst::layout_type>;

template<typename Src, typename Dst>
requires QuantTensorPair<Src, Dst>
void copy_quantized_tensor(Src& src, Dst& dst) {
    copy_tensor(src.data, dst.data);
    copy_tensor(src.scales, dst.scales);
}

template<size_t TileRows, size_t TileCols>
constexpr bool quantized_slice_aligned(int dim, size_t offset, size_t size) {
    if (dim == 0) {
        return (offset % TileRows == 0) && (size % TileRows == 0);
    } else if (dim == 1) {
        return (offset % TileCols == 0) && (size % TileCols == 0);
    }
    return false;
}

template<size_t TileRows, size_t TileCols>
constexpr std::pair<size_t, size_t> quantized_scale_slice(int dim, size_t offset, size_t size) {
    if (dim == 0) {
        return {offset / TileRows, size / TileRows};
    } else {
        return {offset / TileCols, size / TileCols};
    }
}

template<typename Src, typename Dst>
requires QuantTensorPair<Src, Dst>
void copy_quantized_tensor_slice(Src& src, Dst& dst,
                                 int dim, size_t offset, size_t size) {
    constexpr size_t tile_rows = Src::TILE_ROWS;
    constexpr size_t tile_cols = Src::TILE_COLS;
    if (!quantized_slice_aligned<tile_rows, tile_cols>(dim, offset, size)) {
        throw std::invalid_argument("quantized tensor slice must align to tile size");
    }
    copy_tensor_slice(src.data, dst.data, dim, offset, size);
    auto [scale_offset, scale_size] = quantized_scale_slice<tile_rows, tile_cols>(dim, offset, size);
    copy_tensor_slice(src.scales, dst.scales, dim, scale_offset, scale_size);
}

template<typename T, typename Extents, typename Layout, typename ScaleType, size_t TileRows = 16, size_t TileCols = 16,
         typename DataAlloc = DefaultAllocator<T>, typename ScaleAlloc = DefaultAllocator<ScaleType>>
auto make_quantized_tensor(Extents data_extents, DataAlloc data_alloc = {}, ScaleAlloc scale_alloc = {}) {
    static_assert(Extents::rank() == 2);
    if (data_extents.extent(0) % TileRows != 0 || data_extents.extent(1) % TileCols != 0) {
        throw std::invalid_argument("quantized tensor extents must align to tile sizes");
    }
    using data_tensor = BasicTensor<T, Extents, Layout, OwnedStorage<T, DataAlloc>>;
    using scale_extents = std::dextents<size_t, 2>;
    scale_extents scale_dims{data_extents.extent(0) / TileRows, data_extents.extent(1) / TileCols};
    using scale_tensor = BasicTensor<ScaleType, scale_extents, Layout, OwnedStorage<ScaleType, ScaleAlloc>>;
    auto data = make_tensor<T, Extents, Layout, DataAlloc>(data_extents, std::move(data_alloc));
    auto scales = make_tensor<ScaleType, scale_extents, Layout, ScaleAlloc>(scale_dims, std::move(scale_alloc));
    using quant_type = QuantizedTensor<data_tensor, scale_tensor, TileRows, TileCols>;
    return quant_type{std::move(data), std::move(scales)};
}

template<typename DataTensor, typename ScaleTensor, size_t TileRows, size_t TileCols, typename Tensor>
auto slice_rows(Tensor&& tensor, size_t offset, size_t rows)
    requires std::same_as<std::remove_cvref_t<Tensor>, QuantizedTensor<DataTensor, ScaleTensor, TileRows, TileCols>>
{
    return make_quantized_slice(std::forward<Tensor>(tensor), offset, rows);
}

} // namespace tensor
