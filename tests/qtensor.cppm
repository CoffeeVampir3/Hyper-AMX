module;
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <concepts>
#include <mdspan>
#include <memory>
#include <cstdint>
#include <print>
#include <cmath>
export module qtensor;
import quantization;

export template<typename L, typename SrcView, typename DstView>
concept TensorLayout = requires(SrcView src, DstView dst, int dim, size_t offset, size_t size) {
    typename L::template mapping<std::dextents<size_t, 2>>;
    { L::copy_from(src, dst, dim, offset, size) } -> std::same_as<void>;
};

export struct RowMajorLayout {
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

export struct ColumnMajorLayout {
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

export template<size_t BlockM, size_t BlockN>
struct VNNILayout {
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

export template<typename T>
concept TensorStorage = requires(T t, const T ct, typename T::self_type& dest, int dim, size_t offset, size_t size) {
    typename T::extents_type;
    typename T::self_type;
    { t.extent(dim) } -> std::same_as<size_t>;
    { t.view() };
    { ct.copy_to(dest) } -> std::same_as<void>;
    { ct.copy_slice_to(dest, dim, offset, size) } -> std::same_as<void>;
};

export template<typename T, typename Extents, typename Layout>
struct Tensor {
    using self_type = Tensor<T, Extents, Layout>;
    using extents_type = Extents;
    using mapping_type = typename Layout::template mapping<Extents>;
    using mdspan_type = std::mdspan<T, Extents, Layout>;

    struct Deleter {
        void operator()(T* p) const { std::free(p); }
    };

    mapping_type map;
    std::unique_ptr<T[], Deleter> data;

    Tensor(Extents extents) : map(extents) {
        size_t size = map.required_span_size();
        T* ptr = static_cast<T*>(std::aligned_alloc(64, size * sizeof(T)));
        data = std::unique_ptr<T[], Deleter>(ptr);
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;

    size_t extent(size_t dim) const { return map.extents().extent(dim); }
    auto view() { return mdspan_type{data.get(), map}; }
    auto view() const { return std::mdspan<const T, Extents, Layout>{data.get(), map}; }

    void copy_to(Tensor& dest) const {
        std::memcpy(dest.data.get(), data.get(), map.required_span_size() * sizeof(T));
    }

    void copy_slice_to(Tensor& dest, int dim, size_t offset, size_t size) const {
        auto src = view();
        auto dst = dest.view();
        Layout::copy_from(src, dst, dim, offset, size);
    }
};

export template<typename T, typename DataExtents, typename DataLayout, typename QuantScaleType, size_t TileRows = 16, size_t TileCols = 16>
struct QuantizedTensor {
    using self_type = QuantizedTensor<T, DataExtents, DataLayout, QuantScaleType, TileRows, TileCols>;
    using extents_type = DataExtents;
    using scale_extents_type = std::dextents<size_t, 2>;

    static constexpr size_t TILE_M = TileRows;
    static constexpr size_t TILE_N = TileCols;

    static constexpr size_t scale_for_dim(int dim) {
        return (dim == 0) ? TILE_M : TILE_N;
    }

    Tensor<T, DataExtents, DataLayout> data;
    Tensor<QuantScaleType, scale_extents_type, DataLayout> group_quant_scales;

    QuantizedTensor(DataExtents data_extents)
        : data(data_extents)
        , group_quant_scales(scale_extents_type{data_extents.extent(0) / TILE_M, data_extents.extent(1) / TILE_N})
    {}

    QuantizedTensor(const QuantizedTensor&) = delete;
    QuantizedTensor& operator=(const QuantizedTensor&) = delete;
    QuantizedTensor(QuantizedTensor&&) = default;
    QuantizedTensor& operator=(QuantizedTensor&&) = default;

    size_t extent(size_t dim) const { return data.extent(dim); }
    auto view() { return data.view(); }
    auto view() const { return data.view(); }
    auto scales_view() { return group_quant_scales.view(); }
    auto scales_view() const { return group_quant_scales.view(); }

    void copy_to(QuantizedTensor& dest) const {
        data.copy_to(dest.data);
        group_quant_scales.copy_to(dest.group_quant_scales);
    }

    void copy_slice_to(QuantizedTensor& dest, int dim, size_t offset, size_t size) const {
        size_t scale = scale_for_dim(dim);
        data.copy_slice_to(dest.data, dim, offset, size);
        group_quant_scales.copy_slice_to(dest.group_quant_scales, dim, offset / scale, size / scale);
    }
};

template<typename QTensor, typename RefView>
void quantize_from_reference(QTensor& qtensor, const RefView& ref_view) {
    constexpr size_t TILE_SIZE = 16;
    size_t M = qtensor.extent(0);
    size_t N = qtensor.extent(1);
    auto data_view = qtensor.data.view();
    auto scales_view = qtensor.scales_view();

    for (size_t tile_i = 0; tile_i < M / TILE_SIZE; tile_i++) {
        for (size_t tile_j = 0; tile_j < N / TILE_SIZE; tile_j++) {
            int32_t tile_data[16][16];
            for (size_t i = 0; i < TILE_SIZE; i++) {
                for (size_t j = 0; j < TILE_SIZE; j++) {
                    tile_data[i][j] = ref_view[tile_i * TILE_SIZE + i, tile_j * TILE_SIZE + j];
                }
            }
            auto params = AMXQ::compute_quantization_params(tile_data);
            scales_view[tile_i, tile_j] = params;

            for (size_t i = 0; i < TILE_SIZE; i++) {
                for (size_t j = 0; j < TILE_SIZE; j++) {
                    data_view[tile_i * TILE_SIZE + i, tile_j * TILE_SIZE + j] =
                        AMXQ::quantize_scalar(tile_data[i][j], params.bias, params.scale);
                }
            }
        }
    }
}

template<typename QTensorA, typename QTensorB, typename Extents>
auto compute_quantized_matmul(const QTensorA& A_q, const QTensorB& B_q, size_t M, size_t K, size_t N) {
    constexpr size_t TILE_SIZE = 16;
    auto C = Tensor<int32_t, Extents, RowMajorLayout>(Extents{M, N});

    for (size_t i = 0; i < M; i++) {
        for (size_t n = 0; n < N; n++) {
            int32_t sum = 0;
            for (size_t k = 0; k < K; k++) {
                size_t tile_i = i / TILE_SIZE;
                size_t tile_k = k / TILE_SIZE;
                size_t tile_n = n / TILE_SIZE;

                auto a_params = A_q.scales_view()[tile_i, tile_k];
                auto b_params = B_q.scales_view()[tile_k, tile_n];

                int8_t a_q = A_q.data.view()[i, k];
                int8_t b_q = B_q.data.view()[k, n];

                int32_t a_deq = AMXQ::dequantize_scalar(a_q, a_params.bias, a_params.scale);
                int32_t b_deq = AMXQ::dequantize_scalar(b_q, b_params.bias, b_params.scale);

                sum += a_deq * b_deq;
            }
            C.view()[i, n] = sum;
        }
    }
    return C;
}

export void test_quantized_matmul() {
    std::println("=== Testing Quantized Matmul Accuracy ===");

    constexpr size_t M = 64, K = 64, N = 64;
    using Extents = std::dextents<size_t, 2>;

    auto A_ref = Tensor<int32_t, Extents, RowMajorLayout>(Extents{M, K});
    auto B_ref = Tensor<int32_t, Extents, RowMajorLayout>(Extents{K, N});

    auto A_ref_view = A_ref.view();
    auto B_ref_view = B_ref.view();

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < K; j++) {
            A_ref_view[i, j] = static_cast<int32_t>((i * K + j) % 777777 - 10);
        }
    }

    for (size_t k = 0; k < K; k++) {
        for (size_t n = 0; n < N; n++) {
            B_ref_view[k, n] = static_cast<int32_t>((k * N + n) % 333333 - 10);
        }
    }

    auto C_ref = Tensor<int32_t, Extents, RowMajorLayout>(Extents{M, N});
    auto C_ref_view = C_ref.view();

    for (size_t i = 0; i < M; i++) {
        for (size_t n = 0; n < N; n++) {
            int32_t sum = 0;
            for (size_t k = 0; k < K; k++) {
                sum += A_ref_view[i, k] * B_ref_view[k, n];
            }
            C_ref_view[i, n] = sum;
        }
    }

    using QTensor = QuantizedTensor<int8_t, Extents, RowMajorLayout, AMXQ::QuantizationParams, 16, 16>;
    auto A_quant = QTensor(Extents{M, K});
    auto B_quant = QTensor(Extents{K, N});

    quantize_from_reference(A_quant, A_ref_view);
    quantize_from_reference(B_quant, B_ref_view);

    auto C_quant = compute_quantized_matmul<QTensor, QTensor, Extents>(A_quant, B_quant, M, K, N);
    auto C_quant_view = C_quant.view();

    double max_error = 0.0;
    double avg_error = 0.0;
    size_t count = 0;

    for (size_t i = 0; i < M; i++) {
        for (size_t n = 0; n < N; n++) {
            double error = std::abs(static_cast<double>(C_ref_view[i, n] - C_quant_view[i, n]));
            double rel_error = error / (std::abs(static_cast<double>(C_ref_view[i, n])) + 1.0);
            max_error = std::max(max_error, rel_error);
            avg_error += rel_error;
            count++;
        }
    }

    avg_error /= count;

    std::println("Max relative error: {:.6f}", max_error);
    std::println("Avg relative error: {:.6f}", avg_error);

    if (max_error < 0.1) {
        std::println("✓ Quantized matmul accuracy acceptable");
    } else {
        std::println("✗ Quantized matmul error too high");
    }
}

export void test_quantized_vnni_slicing() {
    std::println("=== Testing Quantized VNNI Matmul with Slicing ===");

    constexpr size_t M = 64, K = 128, N = 64;
    using Extents = std::dextents<size_t, 2>;

    auto A_ref = Tensor<int32_t, Extents, RowMajorLayout>(Extents{M, K});
    auto B_ref = Tensor<int32_t, Extents, RowMajorLayout>(Extents{K, N});

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < K; j++) {
            A_ref.view()[i, j] = static_cast<int32_t>((i * K + j) % 333333 - 100);
        }
    }
    for (size_t k = 0; k < K; k++) {
        for (size_t n = 0; n < N; n++) {
            B_ref.view()[k, n] = static_cast<int32_t>((k * N + n) % 777777 - 100);
        }
    }

    using QTensor = QuantizedTensor<int8_t, Extents, RowMajorLayout, AMXQ::QuantizationParams, 16, 16>;
    using QTensorVNNI = QuantizedTensor<int8_t, Extents, VNNILayout<4, 16>, AMXQ::QuantizationParams, 16, 16>;

    auto A_quant = QTensor(Extents{M, K});
    auto B_quant = QTensorVNNI(Extents{K, N});

    quantize_from_reference(A_quant, A_ref.view());
    quantize_from_reference(B_quant, B_ref.view());

    constexpr size_t k_offset = 32;
    constexpr size_t k_slice_size = 64;

    auto A_sliced = QTensor(Extents{M, k_slice_size});
    A_quant.copy_slice_to(A_sliced, 1, k_offset, k_slice_size);

    auto B_sliced = QTensorVNNI(Extents{k_slice_size, N});
    B_quant.copy_slice_to(B_sliced, 0, k_offset, k_slice_size);

    auto C_sliced = compute_quantized_matmul<QTensor, QTensorVNNI, Extents>(A_sliced, B_sliced, M, k_slice_size, N);

    auto C_ref_partial = Tensor<int32_t, Extents, RowMajorLayout>(Extents{M, N});
    for (size_t i = 0; i < M; i++) {
        for (size_t n = 0; n < N; n++) {
            int32_t sum = 0;
            for (size_t k = k_offset; k < k_offset + k_slice_size; k++) {
                int32_t a_val = A_ref.view()[i, k];
                int32_t b_val = B_ref.view()[k, n];
                sum += a_val * b_val;
            }
            C_ref_partial.view()[i, n] = sum;
        }
    }

    double max_error = 0.0;
    for (size_t i = 0; i < M; i++) {
        for (size_t n = 0; n < N; n++) {
            double ref = static_cast<double>(C_ref_partial.view()[i, n]);
            double sliced = static_cast<double>(C_sliced.view()[i, n]);
            double error = std::abs(ref - sliced);
            double rel_error = error / (std::abs(ref) + 1.0);
            max_error = std::max(max_error, rel_error);
        }
    }

    std::println("Max relative error (sliced quantized matmul): {:.6f}", max_error);

    if (max_error < 0.2) {
        std::println("✓ QuantizedTensor slicing preserves scale alignment");
    } else {
        std::println("✗ QuantizedTensor slicing has misaligned scales!");
    }
}

export void run_qtensor_tests() {
    test_quantized_matmul();
    test_quantized_vnni_slicing();
}
