module;
#include <memory>
#include <mdspan>
#include <cstdlib>
#include <cstdint>
export module tensor;

export enum class bfloat16 : uint16_t {};

export enum class TensorLayout { RowMajor, ColMajor, VNNI };

export template<typename T>
concept RawStorage = std::same_as<T, uint8_t> || std::same_as<T, uint16_t> ||
                     std::same_as<T, int8_t> || std::same_as<T, int32_t> ||
                     std::same_as<T, int64_t> || std::same_as<T, float> ||
                     std::same_as<T, bfloat16>;

export template<RawStorage T, TensorLayout L, size_t Rows, size_t Cols>
struct TensorView {
    using value_type = T;
    std::mdspan<T, std::extents<size_t, Rows, Cols>> span;
    static constexpr TensorLayout layout = L;
    size_t rows, cols, row_stride;

    constexpr T& operator[](size_t i, size_t j) const { return span[i, j]; }
    constexpr T& operator()(size_t i, size_t j) const { return span[i, j]; }
    constexpr auto shape() const { return span.extents(); }
    constexpr T* data() const { return span.data_handle(); }
    constexpr T* row(size_t i) const { return span.data_handle() + i * row_stride; }
    constexpr size_t stride_bytes() const { return row_stride * sizeof(T); }
};

export template<RawStorage T, TensorLayout L = TensorLayout::RowMajor,
                size_t Rows = std::dynamic_extent, size_t Cols = std::dynamic_extent>
struct Tensor {
    using value_type = T;
    std::unique_ptr<T[], decltype(&std::free)> owned_data;
    size_t rows, cols, row_stride;
    static constexpr TensorLayout layout = L;

    Tensor(size_t r, size_t c, size_t stride, size_t alignment = 64)
        : owned_data(static_cast<T*>(std::aligned_alloc(alignment, r * stride * sizeof(T))), &std::free),
          rows(r), cols(c), row_stride(stride) {}

    constexpr auto view() const {
        return TensorView<T, L, Rows, Cols>{
            std::mdspan<T, std::extents<size_t, Rows, Cols>>(owned_data.get(), rows, cols),
            rows, cols, row_stride
        };
    }
    constexpr T& operator[](size_t i, size_t j) const {
        if constexpr (L == TensorLayout::RowMajor || L == TensorLayout::VNNI) {
            return owned_data[i * row_stride + j];
        } else {
            return owned_data[j * row_stride + i];
        }
    }
    constexpr T& operator()(size_t i, size_t j) const { return (*this)[i, j]; }
    constexpr T* data() const { return owned_data.get(); }
    constexpr T* row(size_t i) const { return owned_data.get() + i * row_stride; }
    constexpr size_t stride_bytes() const { return row_stride * sizeof(T); }
    constexpr auto shape() const {
        if constexpr (Rows != std::dynamic_extent && Cols != std::dynamic_extent) {
            return std::extents<size_t, Rows, Cols>{};
        } else {
            return std::dextents<size_t, 2>{rows, cols};
        }
    }
};

export template<RawStorage T, TensorLayout L = TensorLayout::RowMajor>
auto make_tensor(size_t rows, size_t cols, size_t alignment = 64) {
    return Tensor<T, L>(rows, cols, cols, alignment);
}

export template<RawStorage T, TensorLayout L = TensorLayout::RowMajor,
                size_t Rows = std::dynamic_extent, size_t Cols = std::dynamic_extent>
auto make_tensor_static(size_t alignment = 64) {
    return Tensor<T, L, Rows, Cols>(Rows, Cols, Cols, alignment);
}

export template<RawStorage T, TensorLayout L = TensorLayout::RowMajor>
auto make_tensor_strided(size_t rows, size_t cols, size_t row_stride, size_t alignment = 64) {
    return Tensor<T, L>(rows, cols, row_stride, alignment);
}

export template<typename T>
concept IsRowMajor = requires { T::layout; } && T::layout == TensorLayout::RowMajor;

export template<typename T>
concept IsColMajor = requires { T::layout; } && T::layout == TensorLayout::ColMajor;

export template<typename T>
concept IsVNNI = requires { T::layout; } && T::layout == TensorLayout::VNNI;

export template<size_t N_BLOCK, size_t K_BLOCK, size_t TILE_N = 16, size_t TILE_K = 64, size_t VNNI_BLK = 4>
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

    static constexpr size_t compute_offset(size_t K, size_t N, size_t n_begin, size_t k_begin) {
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

export template<RawStorage T, size_t N_BLOCK = 256, size_t K_BLOCK = 4096>
struct VNNITensor {
    using value_type = T;
    using layout_type = VNNILayout<N_BLOCK, K_BLOCK>;
    static constexpr TensorLayout layout = TensorLayout::VNNI;
    std::unique_ptr<T[], decltype(&std::free)> owned_data;
    size_t K, N;

    VNNITensor(size_t k, size_t n, size_t alignment = 64) : K(k), N(n),
        owned_data(static_cast<T*>(std::aligned_alloc(alignment, k * n * sizeof(T))), &std::free) {}

    VNNITensor(const VNNITensor&) = delete;
    VNNITensor& operator=(const VNNITensor&) = delete;
    VNNITensor(VNNITensor&&) = default;
    VNNITensor& operator=(VNNITensor&&) = default;

    constexpr T* data() const { return owned_data.get(); }
    constexpr T* get_tile_ptr(size_t n, size_t k) const {
        return data() + layout_type::compute_offset(K, N, n, k);
    }
};

export template<IsRowMajor TensorSrc, size_t N_BLOCK = 256, size_t K_BLOCK = 4096>
auto convert_to_vnni(const TensorSrc& src) {
    using T = typename TensorSrc::value_type;
    static_assert(std::same_as<T, int8_t>, "VNNI conversion only supports int8_t");

    constexpr size_t TILE_N = 16;
    constexpr size_t TILE_K = 64;
    constexpr size_t VNNI_BLK = 4;

    size_t K = src.rows;
    size_t N = src.cols;
    VNNITensor<T, N_BLOCK, K_BLOCK> result(K, N);

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
                        T* dst_base = result.get_tile_ptr(n_block_start + n_tile, global_k);
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
