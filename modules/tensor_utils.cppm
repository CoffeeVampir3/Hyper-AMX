module;
#include <cstddef>
#include <string_view>
#include <print>
#include <concepts>
#include <mdspan>
export module tensor_utils;
import tensor;
import quantization;

// Common type aliases
export using Extents2D = std::dextents<std::size_t, 2>;
export using RowMajor2D = std::layout_right;

// Named tensor concepts for clarity
export template<typename T>
concept Int8RowMajor = MdspanLike<T> &&
                       std::same_as<typename T::element_type, int8_t> &&
                       !IsVNNI<typename T::layout_type>;

export template<typename T>
concept Int8VNNI = MdspanLike<T> &&
                   std::same_as<typename T::element_type, int8_t> &&
                   IsVNNI<typename T::layout_type>;

export template<typename T>
concept Int32RowMajor = MdspanLike<T> &&
                        std::same_as<typename T::element_type, int32_t> &&
                        !IsVNNI<typename T::layout_type>;

export template<typename T>
concept QuantParamsGrid = MdspanLike<T> &&
                          std::same_as<typename T::element_type, QuantizationParams>;

// Partition dimension for socket-aware tensors
export enum class PartitionDim : int {
    Rows = 0,
    Columns = 1
};

export template<typename T>
void fill(auto& t, T value) requires (!std::invocable<T, std::size_t, std::size_t>) {
    for (std::size_t i = 0; i < t.extent(0); ++i)
        for (std::size_t j = 0; j < t.extent(1); ++j)
            t[i, j] = value;
}

export template<typename Fn>
void fill(auto& t, Fn&& fn) requires std::invocable<Fn, std::size_t, std::size_t> {
    for (std::size_t i = 0; i < t.extent(0); ++i)
        for (std::size_t j = 0; j < t.extent(1); ++j)
            t[i, j] = fn(i, j);
}

export void zero(auto& t) {
    using T = typename std::remove_reference_t<decltype(t)>::element_type;
    fill(t, T{0});
}

export void ones(auto& t) {
    using T = typename std::remove_reference_t<decltype(t)>::element_type;
    fill(t, T{1});
}

export template<typename Fn>
bool check_result(const auto& c, Fn&& expected_fn, std::string_view test_name) {
    for (std::size_t i = 0; i < c.extent(0); ++i) {
        for (std::size_t j = 0; j < c.extent(1); ++j) {
            auto expected = expected_fn(i, j);
            if (c[i, j] != expected) {
                std::println("  FAIL [{}]: C[{},{}] = {} (expected {})", test_name, i, j, c[i, j], expected);
                return false;
            }
        }
    }
    std::println("  PASS [{}]", test_name);
    return true;
}

export template<typename T>
bool check_approximate_equal(const auto& actual, const auto& expected, T tolerance, std::string_view test_name) {
    if (actual.extent(0) != expected.extent(0) || actual.extent(1) != expected.extent(1)) {
        std::println("  FAIL [{}]: Shape mismatch {}x{} vs {}x{}", test_name,
                     actual.extent(0), actual.extent(1), expected.extent(0), expected.extent(1));
        return false;
    }

    std::size_t error_count = 0;
    T max_error = 0;
    std::size_t max_error_i = 0, max_error_j = 0;

    for (std::size_t i = 0; i < actual.extent(0); ++i) {
        for (std::size_t j = 0; j < actual.extent(1); ++j) {
            T diff = std::abs(static_cast<T>(actual[i, j]) - static_cast<T>(expected[i, j]));
            if (diff > tolerance) {
                error_count++;
                if (error_count <= 5) {
                    std::println("  ERROR at [{},{}]: actual={}, expected={}, diff={}",
                               i, j, actual[i, j], expected[i, j], diff);
                }
            }
            if (diff > max_error) {
                max_error = diff;
                max_error_i = i;
                max_error_j = j;
            }
        }
    }

    if (error_count > 0) {
        std::println("  FAIL [{}]: {} errors (showing first 5), max error: {} at [{},{}]",
                     test_name, error_count, max_error, max_error_i, max_error_j);
        return false;
    }

    std::println("  PASS [{}]: max error = {}", test_name, max_error);
    return true;
}

export template<size_t Dim, MdspanLike M>
    requires (!IsVNNI<typename M::layout_type>) &&
             std::same_as<typename M::layout_type, std::layout_right>
auto slice(M m, std::size_t start, std::size_t count) {
    using T = typename M::element_type;
    using Extents = std::dextents<std::size_t, 2>;

    if constexpr (Dim == 0) {
        // Row slice: contiguous in memory for row-major layout
        auto new_extents = Extents{count, m.extent(1)};
        auto offset = m.mapping()(start, 0);
        return std::mdspan<T, Extents, std::layout_right>(m.data_handle() + offset, new_extents);
    } else {
        // Column slice: NOT contiguous for row-major, need strided layout
        auto new_extents = Extents{m.extent(0), count};
        auto offset = m.mapping()(0, start);
        std::array<std::size_t, 2> strides{m.extent(1), 1};  // stride_0 = original column count
        return std::mdspan<T, Extents, std::layout_stride>(
            m.data_handle() + offset,
            std::layout_stride::mapping{new_extents, strides}
        );
    }
}

// In-place VNNI conversion: layout_right[K,N] or layout_stride[K,N] → vnni_layout[K,N]
// Zero-copy operation, writes directly to dst for NUMA first-touch
// Usage: convert_to_vnni(source.view(), dest.view())
export template<MdspanLike SrcView, MdspanLike DstView>
    requires (std::same_as<typename SrcView::layout_type, std::layout_right> ||
              std::same_as<typename SrcView::layout_type, std::layout_stride>) &&
             IsVNNI<typename DstView::layout_type> &&
             std::same_as<typename SrcView::element_type, typename DstView::element_type> &&
             std::same_as<typename SrcView::element_type, int8_t>
void convert_to_vnni(const SrcView& src, DstView dst) {
    using T = typename SrcView::element_type;
    using Layout = typename DstView::layout_type;

    constexpr size_t N_BLOCK = Layout::n_block;
    constexpr size_t K_BLOCK = Layout::k_block;
    constexpr size_t TILE_N = Layout::tile_n;
    constexpr size_t TILE_K = Layout::tile_k;
    constexpr size_t VNNI_BLK = Layout::vnni_blk;

    size_t K = src.extent(0);
    size_t N = src.extent(1);

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
                        T* dst_base = dst.data_handle() + dst.mapping()(global_k, global_n);
                        for (size_t n = 0; n < n_tile_size; ++n) {
                            dst_base[n * VNNI_BLK + 0] = src[global_k + 0, global_n + n];
                            dst_base[n * VNNI_BLK + 1] = src[global_k + 1, global_n + n];
                            dst_base[n * VNNI_BLK + 2] = src[global_k + 2, global_n + n];
                            dst_base[n * VNNI_BLK + 3] = src[global_k + 3, global_n + n];
                        }
                    }
                }
            }
        }
    }
}
