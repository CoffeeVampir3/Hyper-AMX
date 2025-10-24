module;
#include <cstddef>
#include <bit>
#include <algorithm>
#include <concepts>
export module layout;

export template<typename L>
concept Layout = requires(const L layout, size_t i, size_t j, size_t dim) {
    { layout.extent(dim) } -> std::same_as<size_t>;
    { layout(i, j) } -> std::same_as<size_t>;
};

export template<typename L>
concept StridedLayout = Layout<L> && requires(const L layout, size_t dim) {
    { layout.stride(dim) } -> std::same_as<size_t>;
};

export template<Layout L>
constexpr bool is_row_major(const L& layout) {
    if constexpr (StridedLayout<L>) {
        return layout.stride(0) >= layout.stride(1);
    }
    return false;
}

export template<Layout L>
constexpr bool is_col_major(const L& layout) {
    if constexpr (StridedLayout<L>) {
        return layout.stride(1) >= layout.stride(0);
    }
    return false;
}

export template<Layout L>
constexpr bool is_contiguous(const L& layout) {
    if constexpr (StridedLayout<L>) {
        return (is_row_major(layout) && layout.stride(0) == layout.extent(1)) ||
               (is_col_major(layout) && layout.stride(1) == layout.extent(0));
    }
    return false;
}

export template<Layout L>
constexpr bool has_vnni_blocking(const L& layout) {
    return requires { L::n_block; L::tile_n; L::vnni_blk; };
}

export struct RowMajorLayout {
    size_t rows, cols, row_stride;

    constexpr size_t extent(size_t dim) const {
        return dim == 0 ? rows : cols;
    }
    constexpr size_t operator()(size_t i, size_t j) const {
        return i * row_stride + j;
    }
    constexpr size_t stride(size_t dim) const {
        return dim == 0 ? row_stride : 1;
    }
    constexpr size_t stride_bytes(size_t elem_size) const {
        return row_stride * elem_size;
    }
};

export struct ColMajorLayout {
    size_t rows, cols, col_stride;

    constexpr size_t extent(size_t dim) const {
        return dim == 0 ? rows : cols;
    }
    constexpr size_t operator()(size_t i, size_t j) const {
        return j * col_stride + i;
    }
    constexpr size_t stride(size_t dim) const {
        return dim == 0 ? 1 : col_stride;
    }
    constexpr size_t stride_bytes(size_t elem_size) const {
        return col_stride * elem_size;
    }
};

export template<size_t N_BLOCK = 256, size_t K_BLOCK = 4096,
                size_t TILE_N = 16, size_t TILE_K = 64, size_t VNNI_BLK = 4>
struct VNNILayout {
    static constexpr size_t n_block = N_BLOCK;
    static constexpr size_t k_block = K_BLOCK;
    static constexpr size_t tile_n = TILE_N;
    static constexpr size_t tile_k = TILE_K;
    static constexpr size_t vnni_blk = VNNI_BLK;

    static constexpr int N_BLOCK_SHIFT = std::countr_zero(N_BLOCK);
    static constexpr int K_BLOCK_SHIFT = std::countr_zero(K_BLOCK);
    static constexpr int TILE_N_SHIFT = std::countr_zero(TILE_N);
    static constexpr int VNNI_SHIFT = std::countr_zero(VNNI_BLK);
    static constexpr size_t TILE_N_BYTES = TILE_N << VNNI_SHIFT;

    size_t K, N;

    constexpr size_t extent(size_t dim) const {
        return dim == 0 ? K : N;
    }
    constexpr size_t operator()(size_t k_begin, size_t n_begin) const {
        size_t n_block_idx = n_begin >> N_BLOCK_SHIFT;
        size_t n_tile_idx = (n_begin >> TILE_N_SHIFT) & ((N_BLOCK >> TILE_N_SHIFT) - 1);
        size_t n_in_tile = n_begin & (TILE_N - 1);
        size_t k_block_idx = k_begin >> K_BLOCK_SHIFT;
        size_t k_vnni_idx = (k_begin >> VNNI_SHIFT) & ((K_BLOCK >> VNNI_SHIFT) - 1);
        size_t n_block_size = std::min(N_BLOCK, N - (n_block_idx << N_BLOCK_SHIFT));
        size_t k_block_size = std::min(K_BLOCK, K - (k_block_idx << K_BLOCK_SHIFT));
        return n_block_idx * K
             + k_block_idx * (n_block_size * k_block_size)
             + n_tile_idx * (TILE_N * k_block_size)
             + k_vnni_idx * TILE_N_BYTES
             + (n_in_tile << VNNI_SHIFT);
    }
};
