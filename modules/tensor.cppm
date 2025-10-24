module;
#include <memory>
#include <mdspan>
#include <cstdlib>
#include <cstdint>
#include <bit>
export module tensor;

export enum class bfloat16 : uint16_t {};

export template<typename T>
concept RawStorage = std::same_as<T, uint8_t> || std::same_as<T, uint16_t> ||
                     std::same_as<T, int8_t> || std::same_as<T, int32_t> ||
                     std::same_as<T, int64_t> || std::same_as<T, float> ||
                     std::same_as<T, bfloat16>;

export template<RawStorage T, typename Layout>
struct TensorView;

export struct RowMajorLayout {
    size_t rows, cols, row_stride;

    constexpr size_t compute_offset(size_t i, size_t j) const {
        return i * row_stride + j;
    }
    constexpr size_t stride_bytes(size_t elem_size) const { return row_stride * elem_size; }
};

export struct ColMajorLayout {
    size_t rows, cols, col_stride;

    constexpr size_t compute_offset(size_t i, size_t j) const {
        return j * col_stride + i;
    }
    constexpr size_t stride_bytes(size_t elem_size) const { return col_stride * elem_size; }
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

    constexpr size_t compute_offset(size_t n_begin, size_t k_begin) const {
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

export template<RawStorage T, typename Layout>
struct Tensor {
    using value_type = T;
    using layout_type = Layout;
    std::unique_ptr<T[], decltype(&std::free)> owned_data;
    Layout layout;

    constexpr Tensor(Layout l, size_t alignment = 64)
        : layout(l), owned_data(nullptr, &std::free) {
        size_t total_size;
        if constexpr (requires { layout.rows; layout.cols; layout.row_stride; }) {
            total_size = layout.rows * layout.row_stride;
        } else if constexpr (requires { layout.K; layout.N; }) {
            total_size = layout.K * layout.N;
        }
        owned_data.reset(static_cast<T*>(std::aligned_alloc(alignment, total_size * sizeof(T))));
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;

    constexpr T* data() const { return owned_data.get(); }
    constexpr T* tile_ptr(size_t i, size_t j) const {
        return data() + layout.compute_offset(i, j);
    }
    constexpr T& operator()(size_t i, size_t j) const { return *tile_ptr(i, j); }

    constexpr auto view() const {
        if constexpr (requires { layout.rows; layout.cols; }) {
            return TensorView<T, Layout>{
                std::mdspan<T, std::dextents<size_t, 2>>(data(), layout.rows, layout.cols),
                layout,
                layout.rows,
                layout.cols,
                layout.row_stride
            };
        } else {
            return *this;
        }
    }
};

export template<RawStorage T, typename Layout>
struct TensorView {
    using value_type = T;
    using layout_type = Layout;
    std::mdspan<T, std::dextents<size_t, 2>> span;
    Layout layout;
    size_t rows, cols, row_stride;

    constexpr T& operator[](size_t i, size_t j) const { return span[i, j]; }
    constexpr T& operator()(size_t i, size_t j) const { return span[i, j]; }
    constexpr auto shape() const { return span.extents(); }
    constexpr T* data() const { return span.data_handle(); }
    constexpr T* row(size_t i) const { return span.data_handle() + i * row_stride; }
    constexpr size_t stride_bytes() const { return layout.stride_bytes(sizeof(T)); }
};

export template<RawStorage T>
auto make_tensor(size_t rows, size_t cols, size_t alignment = 64) {
    return Tensor<T, RowMajorLayout>(RowMajorLayout{rows, cols, cols}, alignment);
}

export template<RawStorage T>
auto make_tensor_strided(size_t rows, size_t cols, size_t row_stride, size_t alignment = 64) {
    return Tensor<T, RowMajorLayout>(RowMajorLayout{rows, cols, row_stride}, alignment);
}

export template<typename T>
concept IsRowMajor = requires { typename T::layout_type; } &&
                     std::same_as<typename T::layout_type, RowMajorLayout>;

export template<typename T>
concept IsColMajor = requires { typename T::layout_type; } &&
                     std::same_as<typename T::layout_type, ColMajorLayout>;

export template<typename T>
concept IsVNNI = requires { typename T::layout_type; } &&
                 requires { T::layout_type::n_block; };

export template<typename TensorSrc, size_t N_BLOCK = 256, size_t K_BLOCK = 4096>
    requires IsRowMajor<TensorSrc>
auto convert_to_vnni(const TensorSrc& src) {
    using T = typename TensorSrc::value_type;
    static_assert(std::same_as<T, int8_t>, "VNNI conversion only supports int8_t");

    constexpr size_t TILE_N = 16;
    constexpr size_t TILE_K = 64;
    constexpr size_t VNNI_BLK = 4;

    size_t K = src.layout.rows;
    size_t N = src.layout.cols;

    Tensor<T, VNNILayout<N_BLOCK, K_BLOCK>> result(VNNILayout<N_BLOCK, K_BLOCK>{K, N});

    for (size_t n_block_start = 0; n_block_start < N; n_block_start += N_BLOCK) {
        size_t n_block_size = std::min(N_BLOCK, N - n_block_start);
        for (size_t k_block_start = 0; k_block_start < K; k_block_start += K_BLOCK) {
            size_t k_block_size = std::min(K_BLOCK, K - k_block_start);
            for (size_t n_tile = 0; n_tile < n_block_size; n_tile += TILE_N) {
                size_t n_tile_size = std::min(TILE_N, n_block_size - n_tile);
                for (size_t k_tile = 0; k_tile < k_block_size; k_tile += TILE_K) {
                    size_t k_tile_size = std::min(TILE_K, k_block_size - k_tile);
                    for (size_t k = 0; k < k_tile_size; k += VNNI_BLK) {
                        size_t global_k = k_block_start + k_tile + k;
                        T* dst_base = result.tile_ptr(n_block_start + n_tile, global_k);
                        for (size_t n = 0; n < n_tile_size; ++n) {
                            size_t global_n = n_block_start + n_tile + n;
                            dst_base[n * VNNI_BLK + 0] = src(global_k + 0, global_n);
                            dst_base[n * VNNI_BLK + 1] = src(global_k + 1, global_n);
                            dst_base[n * VNNI_BLK + 2] = src(global_k + 2, global_n);
                            dst_base[n * VNNI_BLK + 3] = src(global_k + 3, global_n);
                        }
                    }
                }
            }
        }
    }
    return result;
}
