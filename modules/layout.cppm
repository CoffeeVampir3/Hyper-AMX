module;
#include <cstddef>
#include <cstring>
#include <concepts>
#include <mdspan>
export module layout;

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

template<size_t N_BLOCK, size_t K_BLOCK, size_t TILE_N = 16, size_t TILE_K = 64, size_t VNNI_BLK = 4>
struct VNNI {
    using is_vnni_layout = std::true_type;

    template<typename Extents>
    struct mapping {
        using is_vnni_layout = std::true_type;

        Extents extents_;

        static constexpr size_t TILE_N_BYTES = TILE_N * VNNI_BLK;

        constexpr mapping(Extents e) : extents_(e) {}
        constexpr const Extents& extents() const { return extents_; }
        constexpr size_t required_span_size() const {
            return extents_.extent(0) * extents_.extent(1);
        }
        constexpr size_t operator()(size_t k, size_t n) const {
            size_t K = extents_.extent(0);
            size_t N = extents_.extent(1);

            constexpr size_t effective_tile_n = (TILE_N > N_BLOCK) ? N_BLOCK : TILE_N;
            constexpr size_t effective_vnni_blk = (VNNI_BLK > K_BLOCK) ? K_BLOCK : VNNI_BLK;
            constexpr size_t tiles_per_n_block = N_BLOCK / effective_tile_n;
            constexpr size_t vnni_per_k_block = K_BLOCK / effective_vnni_blk;

            size_t n_block_idx = n / N_BLOCK;
            size_t n_tile_idx = (tiles_per_n_block > 1) ? ((n / effective_tile_n) % tiles_per_n_block) : 0;
            size_t n_in_tile = n % effective_tile_n;

            size_t k_block_idx = k / K_BLOCK;
            size_t k_vnni_idx = (vnni_per_k_block > 1) ? ((k / effective_vnni_blk) % vnni_per_k_block) : 0;
            size_t k_in_vnni = k % effective_vnni_blk;

            size_t n_block_size = (N - n_block_idx * N_BLOCK < N_BLOCK) ? (N - n_block_idx * N_BLOCK) : N_BLOCK;
            size_t k_block_size = (K - k_block_idx * K_BLOCK < K_BLOCK) ? (K - k_block_idx * K_BLOCK) : K_BLOCK;

            constexpr size_t effective_tile_n_bytes = effective_tile_n * effective_vnni_blk;

            return n_block_idx * K
                 + k_block_idx * (n_block_size * k_block_size)
                 + n_tile_idx * (effective_tile_n * k_block_size)
                 + k_vnni_idx * effective_tile_n_bytes
                 + (n_in_tile * effective_vnni_blk)
                 + k_in_vnni;
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

template<typename View>
concept Int8RowMajor = requires(View v) {
    requires std::same_as<typename View::element_type, int8_t>;
    requires std::same_as<typename View::layout_type, RowMajor>;
};

template<typename View>
concept Int8VNNI = requires(View v) {
    requires std::same_as<typename View::element_type, int8_t>;
    requires requires { typename View::layout_type::is_vnni_layout; };
};

template<typename View>
concept Int32RowMajor = requires(View v) {
    requires std::same_as<typename View::element_type, int32_t>;
    requires std::same_as<typename View::layout_type, RowMajor>;
};

}
