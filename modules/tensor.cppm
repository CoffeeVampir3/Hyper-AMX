module;
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <memory>
export module tensor;
export import layout;

export enum class bfloat16 : uint16_t {};

export template<typename T>
concept RawStorage = std::same_as<T, uint8_t> || std::same_as<T, uint16_t> ||
                     std::same_as<T, int8_t> || std::same_as<T, int32_t> ||
                     std::same_as<T, int64_t> || std::same_as<T, float> ||
                     std::same_as<T, bfloat16>;

export template<typename T, Layout L>
struct TensorView;

export template<RawStorage T, Layout L>
struct Tensor {
    using value_type = T;
    using layout_type = L;
    std::unique_ptr<T[], decltype(&std::free)> owned_data;
    L layout;

    constexpr Tensor(L l, size_t alignment = 64)
        : layout(l), owned_data(nullptr, &std::free) {
        size_t total_size = layout.extent(0) * layout.extent(1);
        owned_data.reset(static_cast<T*>(std::aligned_alloc(alignment, total_size * sizeof(T))));
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;

    constexpr T* data() { return owned_data.get(); }
    constexpr const T* data() const { return owned_data.get(); }

    constexpr size_t extent(size_t dim) const { return layout.extent(dim); }

    constexpr T& operator()(size_t i, size_t j) {
        return data()[layout(i, j)];
    }
    constexpr const T& operator()(size_t i, size_t j) const {
        return data()[layout(i, j)];
    }

    constexpr auto view() {
        return TensorView<T, L>{data(), layout};
    }
    constexpr auto view() const {
        return TensorView<const T, L>{data(), layout};
    }
};

export template<typename T, Layout L>
struct TensorView {
    using value_type = std::remove_const_t<T>;
    using layout_type = L;

    T* data_ptr;
    L layout;

    constexpr TensorView(T* ptr, L l) : data_ptr(ptr), layout(l) {}

    constexpr T* data() { return data_ptr; }
    constexpr const T* data() const { return data_ptr; }

    constexpr size_t extent(size_t dim) const { return layout.extent(dim); }

    constexpr T& operator()(size_t i, size_t j) {
        return data_ptr[layout(i, j)];
    }
    constexpr const T& operator()(size_t i, size_t j) const {
        return data_ptr[layout(i, j)];
    }

    constexpr T* row(size_t i)
        requires StridedLayout<L>
    {
        return data_ptr + layout(i, 0);
    }
    constexpr const T* row(size_t i) const
        requires StridedLayout<L>
    {
        return data_ptr + layout(i, 0);
    }

    constexpr size_t stride_bytes() const
        requires requires(L l, size_t s) { l.stride_bytes(s); }
    {
        return layout.stride_bytes(sizeof(value_type));
    }
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
concept HasLayout = requires { typename T::layout_type; };

export template<typename T>
concept IsRowMajor = HasLayout<T> &&
                     std::same_as<typename T::layout_type, RowMajorLayout>;

export template<typename T>
concept IsColMajor = HasLayout<T> &&
                     std::same_as<typename T::layout_type, ColMajorLayout>;

export template<typename T>
concept IsVNNI = HasLayout<T> &&
                 requires { T::layout_type::n_block; };

export template<typename TensorSrc, size_t N_BLOCK = 256, size_t K_BLOCK = 4096>
    requires IsRowMajor<TensorSrc>
auto convert_to_vnni(const TensorSrc& src) {
    using T = typename TensorSrc::value_type;
    static_assert(std::same_as<T, int8_t>, "VNNI conversion only supports int8_t");

    constexpr size_t TILE_N = 16;
    constexpr size_t TILE_K = 64;
    constexpr size_t VNNI_BLK = 4;

    size_t K = src.extent(0);
    size_t N = src.extent(1);

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
                        size_t global_n = n_block_start + n_tile;
                        T* dst_base = result.data() + result.layout(global_k, global_n);
                        for (size_t n = 0; n < n_tile_size; ++n) {
                            dst_base[n * VNNI_BLK + 0] = src(global_k + 0, global_n + n);
                            dst_base[n * VNNI_BLK + 1] = src(global_k + 1, global_n + n);
                            dst_base[n * VNNI_BLK + 2] = src(global_k + 2, global_n + n);
                            dst_base[n * VNNI_BLK + 3] = src(global_k + 3, global_n + n);
                        }
                    }
                }
            }
        }
    }
    return result;
}
