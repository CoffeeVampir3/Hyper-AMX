module;
#include <cstddef>
#include <mdspan>
#include <bit>
#include <algorithm>
#include <concepts>
export module layout;

export template<size_t N_BLOCK = 256, size_t K_BLOCK = 4096,
                size_t TILE_N = 16, size_t TILE_K = 64, size_t VNNI_BLK = 4>
struct vnni_layout {
    static constexpr size_t n_block = N_BLOCK;
    static constexpr size_t k_block = K_BLOCK;
    static constexpr size_t tile_n = TILE_N;
    static constexpr size_t tile_k = TILE_K;
    static constexpr size_t vnni_blk = VNNI_BLK;

    template<typename Extents>
    struct mapping {
        using extents_type = Extents;
        using index_type = typename extents_type::index_type;
        using size_type = typename extents_type::size_type;
        using rank_type = typename extents_type::rank_type;
        using layout_type = vnni_layout;

        static constexpr int N_BLOCK_SHIFT = std::countr_zero(N_BLOCK);
        static constexpr int K_BLOCK_SHIFT = std::countr_zero(K_BLOCK);
        static constexpr int TILE_N_SHIFT = std::countr_zero(TILE_N);
        static constexpr int VNNI_SHIFT = std::countr_zero(VNNI_BLK);
        static constexpr size_t TILE_N_BYTES = TILE_N << VNNI_SHIFT;

        constexpr mapping() = default;
        constexpr mapping(const extents_type& e) : ext(e) {}

        constexpr const extents_type& extents() const { return ext; }

        constexpr index_type required_span_size() const {
            if (ext.extent(0) == 0 || ext.extent(1) == 0) return 0;
            return ext.extent(0) * ext.extent(1);
        }

        static constexpr bool is_always_unique() { return true; }
        static constexpr bool is_always_exhaustive() { return false; }
        static constexpr bool is_always_strided() { return false; }

        constexpr bool is_unique() const { return true; }
        constexpr bool is_exhaustive() const { return false; }
        constexpr bool is_strided() const { return false; }

        template<typename... Indices>
        constexpr index_type operator()(Indices... idxs) const {
            return compute_offset(static_cast<index_type>(idxs)...);
        }

    private:
        constexpr index_type compute_offset(index_type k_begin, index_type n_begin) const {
            index_type K = ext.extent(0);
            index_type N = ext.extent(1);
            index_type n_block_idx = n_begin >> N_BLOCK_SHIFT;
            index_type n_tile_idx = (n_begin >> TILE_N_SHIFT) & ((N_BLOCK >> TILE_N_SHIFT) - 1);
            index_type n_in_tile = n_begin & (TILE_N - 1);
            index_type k_block_idx = k_begin >> K_BLOCK_SHIFT;
            index_type k_vnni_idx = (k_begin >> VNNI_SHIFT) & ((K_BLOCK >> VNNI_SHIFT) - 1);
            index_type k_in_vnni = k_begin & (VNNI_BLK - 1);
            index_type n_block_size = std::min(static_cast<index_type>(N_BLOCK), N - (n_block_idx << N_BLOCK_SHIFT));
            index_type k_block_size = std::min(static_cast<index_type>(K_BLOCK), K - (k_block_idx << K_BLOCK_SHIFT));
            return n_block_idx * K
                 + k_block_idx * (n_block_size * k_block_size)
                 + n_tile_idx * (TILE_N * k_block_size)
                 + k_vnni_idx * TILE_N_BYTES
                 + (n_in_tile << VNNI_SHIFT)
                 + k_in_vnni;
        }

        extents_type ext;
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

export template<typename T>
concept MdspanLike = requires(T t) {
    typename T::element_type;
    typename T::extents_type;
    typename T::layout_type;
    { t.data_handle() };
    { t.extent(0) };
};

export template<typename Layout>
concept IsVNNI = requires { Layout::n_block; Layout::tile_n; Layout::vnni_blk; };

export template<typename Layout>
concept HasStride = std::same_as<Layout, std::layout_right> ||
                    std::same_as<Layout, std::layout_stride>;

export template<MdspanLike M>
    requires HasStride<typename M::layout_type>
constexpr size_t stride_bytes(const M& m, size_t elem_size) {
    if constexpr (std::same_as<typename M::layout_type, std::layout_right>) {
        return m.extent(1) * elem_size;
    } else {
        return m.stride(0) * elem_size;
    }
}

export template<MdspanLike M>
    requires HasStride<typename M::layout_type>
constexpr size_t stride_bytes(const M& m) {
    return stride_bytes(m, sizeof(typename M::element_type));
}
