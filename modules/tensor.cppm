module;
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <mdspan>
#include <memory>
export module tensor;
export import layout;

export enum class bfloat16 : uint16_t {};

export template<typename T>
concept RawStorage = std::same_as<T, uint8_t> || std::same_as<T, uint16_t> ||
                     std::same_as<T, int8_t> || std::same_as<T, int32_t> ||
                     std::same_as<T, int64_t> || std::same_as<T, float> ||
                     std::same_as<T, bfloat16>;

export template<RawStorage T, typename Extents, typename Layout>
struct Tensor {
    using element_type = T;
    using value_type = T;
    using extents_type = Extents;
    using layout_type = Layout;
    using mapping_type = typename Layout::template mapping<Extents>;
    using mdspan_type = std::mdspan<T, Extents, Layout>;

    std::unique_ptr<T[], decltype(&std::free)> owned_data;
    mapping_type map;

    constexpr Tensor(Extents e, size_t alignment = 64)
        : map(e), owned_data(nullptr, &std::free) {
        size_t total_size = map.required_span_size();
        owned_data.reset(static_cast<T*>(std::aligned_alloc(alignment, total_size * sizeof(T))));
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;

    constexpr T* data() { return owned_data.get(); }
    constexpr const T* data() const { return owned_data.get(); }
    constexpr T* data_handle() { return owned_data.get(); }
    constexpr const T* data_handle() const { return owned_data.get(); }

    constexpr const mapping_type& mapping() const { return map; }

    constexpr size_t extent(size_t dim) const { return map.extents().extent(dim); }

    constexpr auto view() { return mdspan_type{data(), map}; }
    constexpr auto view() const { return std::mdspan<const T, Extents, Layout>{data(), map}; }

    constexpr operator mdspan_type() { return view(); }
    constexpr operator std::mdspan<const T, Extents, Layout>() const { return view(); }

    constexpr T& operator[](size_t i, size_t j) { return data()[map(i, j)]; }
    constexpr const T& operator[](size_t i, size_t j) const { return data()[map(i, j)]; }
};

export template<RawStorage T>
auto make_tensor(size_t rows, size_t cols, size_t alignment = 64) {
    using Extents = std::dextents<size_t, 2>;
    using Layout = std::layout_right;
    return Tensor<T, Extents, Layout>(Extents{rows, cols}, alignment);
}

export template<RawStorage T>
auto make_tensor_strided(size_t rows, size_t cols, size_t row_stride, size_t alignment = 64) {
    using Extents = std::dextents<size_t, 2>;
    using Layout = std::layout_stride;
    std::array<size_t, 2> strides{row_stride, 1};
    return Tensor<T, Extents, Layout>(typename Layout::template mapping<Extents>{Extents{rows, cols}, strides}, alignment);
}

export template<MdspanLike M, size_t N_BLOCK = 256, size_t K_BLOCK = 4096>
    requires std::same_as<typename M::layout_type, std::layout_right>
auto convert_to_vnni(const M& src) {
    using T = typename M::element_type;
    static_assert(std::same_as<T, int8_t>, "VNNI conversion only supports int8_t");

    constexpr size_t TILE_N = 16;
    constexpr size_t TILE_K = 64;
    constexpr size_t VNNI_BLK = 4;

    size_t K = src.extent(0);
    size_t N = src.extent(1);

    using VNNILayout = vnni_layout<N_BLOCK, K_BLOCK>;
    using Extents = std::dextents<size_t, 2>;
    Tensor<T, Extents, VNNILayout> result(Extents{K, N});
    auto dst = result.view();

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
                        T* dst_base = result.data() + dst.mapping()(global_k, global_n);
                        for (size_t n = 0; n < n_tile_size; ++n) {
                            dst_base[n * VNNI_BLK + 0] = src.data_handle()[src.mapping()(global_k + 0, global_n + n)];
                            dst_base[n * VNNI_BLK + 1] = src.data_handle()[src.mapping()(global_k + 1, global_n + n)];
                            dst_base[n * VNNI_BLK + 2] = src.data_handle()[src.mapping()(global_k + 2, global_n + n)];
                            dst_base[n * VNNI_BLK + 3] = src.data_handle()[src.mapping()(global_k + 3, global_n + n)];
                        }
                    }
                }
            }
        }
    }
    return result;
}
