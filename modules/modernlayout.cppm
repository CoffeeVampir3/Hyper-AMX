module;
#include <cstddef>
#include <cstring>
#include <concepts>
#include <mdspan>
export module modernlayout;

export namespace Layout {

template<typename L, typename SrcView, typename DstView>
concept TensorLayout = requires(SrcView src, DstView dst, int dim, size_t offset, size_t size) {
    typename L::template mapping<std::dextents<size_t, 2>>;
    { L::copy_from(src, dst, dim, offset, size) } -> std::same_as<void>;
};

struct RowMajor {
    template<typename Extents>
    using mapping = typename std::layout_right::template mapping<Extents>;

    template<typename SrcView, typename DstView>
    static void copy_from(const SrcView& src, DstView& dst, int dim, size_t offset, size_t size) {
        using T = typename DstView::element_type;

        if (dim == 0) {
            size_t row_bytes = src.extent(1) * sizeof(T);
            std::memcpy(dst.data_handle(), src.data_handle() + offset * src.extent(1), size * row_bytes);
        } else {
            for (size_t i = 0; i < src.extent(0); i++) {
                for (size_t j = 0; j < size; j++) {
                    dst[i, j] = src[i, offset + j];
                }
            }
        }
    }
};

struct ColumnMajor {
    template<typename Extents>
    using mapping = typename std::layout_left::template mapping<Extents>;

    template<typename SrcView, typename DstView>
    static void copy_from(const SrcView& src, DstView& dst, int dim, size_t offset, size_t size) {
        using T = typename DstView::element_type;

        if (dim == 1) {
            size_t col_bytes = src.extent(0) * sizeof(T);
            std::memcpy(dst.data_handle(), src.data_handle() + offset * src.extent(0), size * col_bytes);
        } else {
            for (size_t i = 0; i < size; i++) {
                for (size_t j = 0; j < src.extent(1); j++) {
                    dst[i, j] = src[offset + i, j];
                }
            }
        }
    }
};

template<size_t BlockM, size_t BlockN>
struct VNNI {
    template<typename Extents>
    struct mapping {
        Extents extents_;
        constexpr mapping(Extents e) : extents_(e) {}
        constexpr const Extents& extents() const { return extents_; }
        constexpr size_t required_span_size() const {
            return extents_.extent(0) * extents_.extent(1);
        }
        constexpr size_t operator()(size_t k, size_t n) const {
            size_t k_outer = k / 4;
            size_t k_inner = k % 4;
            size_t n_block = n / BlockN;
            size_t n_offset = n % BlockN;
            size_t blocks_per_row = (extents_.extent(1) + BlockN - 1) / BlockN;
            return (k_outer * blocks_per_row * BlockN * 4) + (n_block * BlockN * 4) + (n_offset * 4) + k_inner;
        }
    };

    template<typename SrcView, typename DstView>
    static void copy_from(const SrcView& src, DstView& dst, int dim, size_t offset, size_t size) {
        if (dim == 0) {
            for (size_t k = 0; k < size; k++) {
                for (size_t n = 0; n < src.extent(1); n++) {
                    dst[k, n] = src[offset + k, n];
                }
            }
        } else {
            for (size_t k = 0; k < src.extent(0); k++) {
                for (size_t n = 0; n < size; n++) {
                    dst[k, n] = src[k, offset + n];
                }
            }
        }
    }
};

}
