module;
#include <cstddef>
#include <cstdlib>
#include <concepts>
#include <mdspan>
#include <memory>
#include <cstdint>
#include <print>
#include <cmath>
#include <vector>
export module tensor_tests;
import avx512;
import tensor;
import layout;

using namespace quantization;

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
            std::mdspan<const int32_t, std::extents<size_t, 16, 16>> tile_view(&tile_data[0][0]);
            auto params = compute_quantization_params(tile_view);
            scales_view[tile_i, tile_j] = params;

            for (size_t i = 0; i < TILE_SIZE; i++) {
                for (size_t j = 0; j < TILE_SIZE; j++) {
                    data_view[tile_i * TILE_SIZE + i, tile_j * TILE_SIZE + j] =
                        quantize_scalar(tile_data[i][j], params.bias, params.scale);
                }
            }
        }
    }
}

template<typename QTensorA, typename QTensorB, typename Extents>
auto compute_quantized_matmul(const QTensorA& A_q, const QTensorB& B_q, size_t M, size_t K, size_t N) {
    constexpr size_t TILE_SIZE = 16;
    auto C = Tensor<int32_t, Extents, Layout::RowMajor>(Extents{M, N});

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

                int32_t a_deq = dequantize_scalar(a_q, a_params.bias, a_params.scale);
                int32_t b_deq = dequantize_scalar(b_q, b_params.bias, b_params.scale);

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

    auto A_ref = Tensor<int32_t, Extents, Layout::RowMajor>(Extents{M, K});
    auto B_ref = Tensor<int32_t, Extents, Layout::RowMajor>(Extents{K, N});

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

    auto C_ref = Tensor<int32_t, Extents, Layout::RowMajor>(Extents{M, N});
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

    using QTensor = QuantizedTensor<int8_t, Extents, Layout::RowMajor, QuantizationParams, 16, 16>;
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

    auto A_ref = Tensor<int32_t, Extents, Layout::RowMajor>(Extents{M, K});
    auto B_ref = Tensor<int32_t, Extents, Layout::RowMajor>(Extents{K, N});

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

    using QTensor = QuantizedTensor<int8_t, Extents, Layout::RowMajor, QuantizationParams, 16, 16>;
    using QTensorVNNI = QuantizedTensor<int8_t, Extents, Layout::VNNI<4, 16>, QuantizationParams, 16, 16>;

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

    auto C_ref_partial = Tensor<int32_t, Extents, Layout::RowMajor>(Extents{M, N});
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

void test_tensor_subview_rowmajor() {
    std::println("=== Testing Tensor subview (RowMajor) ===");

    using Extents = std::dextents<size_t, 2>;
    constexpr size_t M = 64, N = 128;

    auto tensor = Tensor<int32_t, Extents, Layout::RowMajor>(Extents{M, N});

    int32_t val = 0;
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            tensor.view()[i, j] = val++;
        }
    }

    constexpr size_t row_start = 16;
    constexpr size_t row_count = 32;
    auto subview = tensor.subview(row_start, row_count);

    if (subview.extent(0) != row_count) {
        std::println("  ✗ Subview row extent incorrect: expected {}, got {}", row_count, subview.extent(0));
        std::exit(1);
    }
    if (subview.extent(1) != N) {
        std::println("  ✗ Subview col extent incorrect: expected {}, got {}", N, subview.extent(1));
        std::exit(1);
    }

    bool data_correct = true;
    for (size_t i = 0; i < row_count; i++) {
        for (size_t j = 0; j < N; j++) {
            int32_t expected = static_cast<int32_t>((row_start + i) * N + j);
            if (subview.view()[i, j] != expected) {
                std::println("  ✗ Subview data mismatch at [{},{}]: expected {}, got {}",
                             i, j, expected, subview.view()[i, j]);
                data_correct = false;
                break;
            }
        }
        if (!data_correct) break;
    }

    if (!data_correct) {
        std::exit(1);
    }

    for (size_t i = 0; i < row_count; i++) {
        for (size_t j = 0; j < N; j++) {
            subview.view()[i, j] = -1000 - static_cast<int32_t>(i * N + j);
        }
    }

    bool view_mutates_original = true;
    for (size_t i = 0; i < row_count; i++) {
        for (size_t j = 0; j < N; j++) {
            int32_t expected = -1000 - static_cast<int32_t>(i * N + j);
            if (tensor.view()[row_start + i, j] != expected) {
                std::println("  ✗ Subview mutation did not affect original tensor");
                view_mutates_original = false;
                break;
            }
        }
        if (!view_mutates_original) break;
    }

    if (!view_mutates_original) {
        std::exit(1);
    }

    std::println("   ✓ Tensor subview (RowMajor) correctness");
    std::println("   ✓ Subview correctly references original data\n");
}

void test_quantized_tensor_subview() {
    std::println("=== Testing QuantizedTensor subview ===");

    using Extents = std::dextents<size_t, 2>;
    constexpr size_t M = 64, N = 128;
    constexpr size_t TILE_M = 16, TILE_N = 16;

    using QTensor = QuantizedTensor<int8_t, Extents, Layout::RowMajor, QuantizationParams, TILE_M, TILE_N>;
    auto qtensor = QTensor(Extents{M, N});

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            qtensor.view()[i, j] = static_cast<int8_t>((i + j) % 127);
        }
    }

    for (size_t tile_i = 0; tile_i < M / TILE_M; tile_i++) {
        for (size_t tile_j = 0; tile_j < N / TILE_N; tile_j++) {
            qtensor.scales_view()[tile_i, tile_j] = QuantizationParams{
                static_cast<int32_t>(tile_i * 100 + tile_j),
                static_cast<_Float16>(tile_i + tile_j + 1.0f)
            };
        }
    }

    constexpr size_t row_start = 32;
    constexpr size_t row_count = 16;
    auto subview = qtensor.subview(row_start, row_count);

    if (subview.extent(0) != row_count) {
        std::println("  ✗ QuantizedTensor subview row extent incorrect: expected {}, got {}",
                     row_count, subview.extent(0));
        std::exit(1);
    }
    if (subview.extent(1) != N) {
        std::println("  ✗ QuantizedTensor subview col extent incorrect: expected {}, got {}",
                     N, subview.extent(1));
        std::exit(1);
    }

    bool data_correct = true;
    for (size_t i = 0; i < row_count; i++) {
        for (size_t j = 0; j < N; j++) {
            int8_t expected = static_cast<int8_t>((row_start + i + j) % 127);
            if (subview.view()[i, j] != expected) {
                std::println("  ✗ QuantizedTensor subview data mismatch at [{},{}]", i, j);
                data_correct = false;
                break;
            }
        }
        if (!data_correct) break;
    }

    if (!data_correct) {
        std::exit(1);
    }

    size_t scale_tile_start = row_start / TILE_M;
    size_t scale_tile_count = row_count / TILE_M;

    bool scales_correct = true;
    for (size_t tile_i = 0; tile_i < scale_tile_count; tile_i++) {
        for (size_t tile_j = 0; tile_j < N / TILE_N; tile_j++) {
            auto expected_params = qtensor.scales_view()[scale_tile_start + tile_i, tile_j];
            auto actual_params = subview.scales_view()[tile_i, tile_j];

            if (expected_params.bias != actual_params.bias ||
                std::abs(static_cast<float>(expected_params.scale) - static_cast<float>(actual_params.scale)) > 1e-6f) {
                std::println("  ✗ QuantizedTensor subview scale mismatch at tile [{},{}]", tile_i, tile_j);
                scales_correct = false;
                break;
            }
        }
        if (!scales_correct) break;
    }

    if (!scales_correct) {
        std::exit(1);
    }

    std::println("   ✓ QuantizedTensor subview data correctness");
    std::println("   ✓ QuantizedTensor subview scales correctness\n");
}

void test_quantized_tensor_alignment_validation() {
    std::println("=== Testing QuantizedTensor Alignment Validation ===");

    using Extents = std::dextents<size_t, 2>;
    constexpr size_t M = 64, N = 64;
    constexpr size_t TILE_M = 16, TILE_N = 16;

    using QTensor = QuantizedTensor<int8_t, Extents, Layout::RowMajor, QuantizationParams, TILE_M, TILE_N>;
    auto qtensor = QTensor(Extents{M, N});

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            qtensor.view()[i, j] = static_cast<int8_t>((i + j) % 127);
        }
    }

    bool caught_misaligned_start = false;
    try {
        auto subview = qtensor.subview(10, 16);
    } catch (const std::invalid_argument&) {
        caught_misaligned_start = true;
    }

    if (!caught_misaligned_start) {
        std::println("  ✗ Failed to catch misaligned row_start");
        std::exit(1);
    }

    bool caught_misaligned_count = false;
    try {
        auto subview = qtensor.subview(16, 10);
    } catch (const std::invalid_argument&) {
        caught_misaligned_count = true;
    }

    if (!caught_misaligned_count) {
        std::println("  ✗ Failed to catch misaligned row_count");
        std::exit(1);
    }

    bool caught_copy_slice_misaligned_offset = false;
    try {
        auto dest = QTensor(Extents{M, 32});
        qtensor.copy_slice_to(dest, 1, 10, 32);
    } catch (const std::invalid_argument&) {
        caught_copy_slice_misaligned_offset = true;
    }

    if (!caught_copy_slice_misaligned_offset) {
        std::println("  ✗ Failed to catch misaligned copy_slice_to offset");
        std::exit(1);
    }

    bool caught_copy_slice_misaligned_size = false;
    try {
        auto dest = QTensor(Extents{M, 10});
        qtensor.copy_slice_to(dest, 1, 16, 10);
    } catch (const std::invalid_argument&) {
        caught_copy_slice_misaligned_size = true;
    }

    if (!caught_copy_slice_misaligned_size) {
        std::println("  ✗ Failed to catch misaligned copy_slice_to size");
        std::exit(1);
    }

    auto aligned_subview = qtensor.subview(16, 32);
    if (aligned_subview.extent(0) != 32 || aligned_subview.extent(1) != N) {
        std::println("  ✗ Aligned subview has incorrect extents");
        std::exit(1);
    }

    std::println("   ✓ QuantizedTensor alignment validation works correctly\n");
}

void test_tensor_padding() {
    std::println("=== Testing Tensor Padding ===");

    using Extents = std::dextents<size_t, 2>;

    auto test_padding_int8 = []() {
        constexpr size_t M = 30, N = 50;
        auto source = Tensor<int8_t, Extents, Layout::RowMajor>(Extents{M, N});

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                source.view()[i, j] = static_cast<int8_t>((i * N + j) % 127);
            }
        }

        auto padded = tensor_ops::pad_to_alignment(source, 32, 64, int8_t{-1});

        if (padded.extent(0) != 32) {
            std::println("  ✗ Padded row extent incorrect: expected 32, got {}", padded.extent(0));
            std::exit(1);
        }
        if (padded.extent(1) != 64) {
            std::println("  ✗ Padded col extent incorrect: expected 64, got {}", padded.extent(1));
            std::exit(1);
        }

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                int8_t expected = static_cast<int8_t>((i * N + j) % 127);
                if (padded.view()[i, j] != expected) {
                    std::println("  ✗ Padded data mismatch at [{},{}]: expected {}, got {}",
                                 i, j, expected, padded.view()[i, j]);
                    std::exit(1);
                }
            }
        }

        for (size_t i = 0; i < M; i++) {
            for (size_t j = N; j < 64; j++) {
                if (padded.view()[i, j] != -1) {
                    std::println("  ✗ Column padding incorrect at [{},{}]: expected -1, got {}",
                                 i, j, padded.view()[i, j]);
                    std::exit(1);
                }
            }
        }

        for (size_t i = M; i < 32; i++) {
            for (size_t j = 0; j < 64; j++) {
                if (padded.view()[i, j] != -1) {
                    std::println("  ✗ Row padding incorrect at [{},{}]: expected -1, got {}",
                                 i, j, padded.view()[i, j]);
                    std::exit(1);
                }
            }
        }
    };

    auto test_padding_int32 = []() {
        constexpr size_t M = 25, N = 40;
        auto source = Tensor<int32_t, Extents, Layout::RowMajor>(Extents{M, N});

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                source.view()[i, j] = static_cast<int32_t>(i * 1000 + j);
            }
        }

        auto padded = tensor_ops::pad_to_alignment(source, 32, 48, int32_t{9999});

        if (padded.extent(0) != 32 || padded.extent(1) != 48) {
            std::println("  ✗ int32_t padded extents incorrect");
            std::exit(1);
        }

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                int32_t expected = static_cast<int32_t>(i * 1000 + j);
                if (padded.view()[i, j] != expected) {
                    std::println("  ✗ int32_t padded data mismatch at [{},{}]", i, j);
                    std::exit(1);
                }
            }
        }

        for (size_t i = M; i < 32; i++) {
            for (size_t j = 0; j < 48; j++) {
                if (padded.view()[i, j] != 9999) {
                    std::println("  ✗ int32_t padding value incorrect at [{},{}]", i, j);
                    std::exit(1);
                }
            }
        }
    };

    test_padding_int8();
    test_padding_int32();

    std::println("   ✓ Tensor padding works correctly\n");
}

void test_tensor_gather_rows() {
    std::println("=== Testing Tensor Gather Rows ===");

    using Extents = std::dextents<size_t, 2>;
    constexpr size_t M = 100, N = 64;

    auto source = Tensor<int32_t, Extents, Layout::RowMajor>(Extents{M, N});

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            source.view()[i, j] = static_cast<int32_t>(i * 10000 + j);
        }
    }

    std::vector<size_t> indices = {5, 10, 15, 20, 25, 30, 99, 0, 50};
    auto gathered = tensor_ops::gather_rows(source, indices);

    if (gathered.extent(0) != indices.size()) {
        std::println("  ✗ Gathered tensor row count incorrect: expected {}, got {}",
                     indices.size(), gathered.extent(0));
        std::exit(1);
    }
    if (gathered.extent(1) != N) {
        std::println("  ✗ Gathered tensor col count incorrect: expected {}, got {}",
                     N, gathered.extent(1));
        std::exit(1);
    }

    for (size_t i = 0; i < indices.size(); i++) {
        size_t src_row = indices[i];
        for (size_t j = 0; j < N; j++) {
            int32_t expected = static_cast<int32_t>(src_row * 10000 + j);
            if (gathered.view()[i, j] != expected) {
                std::println("  ✗ Gathered data mismatch at [{},{}]: expected {}, got {}",
                             i, j, expected, gathered.view()[i, j]);
                std::exit(1);
            }
        }
    }

    size_t single_indices[] = {42, 7, 88};
    auto gathered2 = tensor_ops::gather_rows(source, single_indices, 3);

    if (gathered2.extent(0) != 3) {
        std::println("  ✗ Gathered2 tensor row count incorrect");
        std::exit(1);
    }

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < N; j++) {
            int32_t expected = static_cast<int32_t>(single_indices[i] * 10000 + j);
            if (gathered2.view()[i, j] != expected) {
                std::println("  ✗ Gathered2 data mismatch at [{},{}]", i, j);
                std::exit(1);
            }
        }
    }

    std::println("   ✓ Tensor gather_rows works correctly\n");
}

void test_subview_mutation_preserves_stride() {
    std::println("=== Testing Subview Stride Correctness ===");

    using Extents = std::dextents<size_t, 2>;
    constexpr size_t M = 64, N = 128;

    auto tensor = Tensor<int32_t, Extents, Layout::RowMajor>(Extents{M, N});

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            tensor.view()[i, j] = static_cast<int32_t>(i * 1000 + j);
        }
    }

    auto subview1 = tensor.subview(10, 20);
    auto subview2 = tensor.subview(30, 10);

    for (size_t i = 0; i < 20; i++) {
        for (size_t j = 0; j < N; j++) {
            subview1.view()[i, j] = static_cast<int32_t>(-(i * 1000 + j));
        }
    }

    for (size_t i = 10; i < 30; i++) {
        for (size_t j = 0; j < N; j++) {
            int32_t expected = static_cast<int32_t>(-(i - 10) * 1000 - j);
            if (tensor.view()[i, j] != expected) {
                std::println("  ✗ Subview mutation didn't propagate correctly at [{},{}]", i, j);
                std::exit(1);
            }
        }
    }

    for (size_t i = 0; i < 10; i++) {
        for (size_t j = 0; j < N; j++) {
            subview2.view()[i, j] = 777;
        }
    }

    for (size_t i = 30; i < 40; i++) {
        for (size_t j = 0; j < N; j++) {
            if (tensor.view()[i, j] != 777) {
                std::println("  ✗ Second subview mutation didn't propagate at [{},{}]", i, j);
                std::exit(1);
            }
        }
    }

    for (size_t i = 0; i < 10; i++) {
        for (size_t j = 0; j < N; j++) {
            int32_t expected = static_cast<int32_t>(i * 1000 + j);
            if (tensor.view()[i, j] != expected) {
                std::println("  ✗ Unmodified region was corrupted at [{},{}]", i, j);
                std::exit(1);
            }
        }
    }

    std::println("   ✓ Subview stride and mutation correctness verified\n");
}

export void run_qtensor_tests() {
    test_quantized_matmul();
    test_quantized_vnni_slicing();
    test_tensor_subview_rowmajor();
    test_quantized_tensor_subview();
    test_quantized_tensor_alignment_validation();
    test_tensor_padding();
    test_tensor_gather_rows();
    test_subview_mutation_preserves_stride();
}
