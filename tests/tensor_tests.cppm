module;
#include <array>
#include <cstdint>
#include <cstdlib>
#include <mdspan>
#include <print>
#include <utility>
#include <vector>
export module hyperamx.tensor_tests;

import hyperamx.layout;
import hyperamx.tensor;
import hyperamx.tensor_utils;

namespace {

using namespace tensor;
using namespace Layout;
using namespace tensor_utils;

using Extents2D = std::dextents<size_t, 2>;
using RowMajor = Layout::RowMajor;
using ColumnMajor = Layout::ColumnMajor;

template<typename Fn>
void run_case(const char* label, Fn&& fn) {
    std::println("=== {} ===", label);
    fn();
    std::println("   ✓ pass\n");
}

[[noreturn]] void fail(const char* msg) {
    std::println("   ✗ {}", msg);
    std::exit(1);
}

void test_tensor_allocation_rowmajor() {
    constexpr size_t M = 64;
    constexpr size_t N = 48;
    auto tensor = make_tensor<int32_t, Extents2D, RowMajor>(Extents2D{M, N});
    if (tensor.extent(0) != M || tensor.extent(1) != N) fail("extent mismatch");
    fill(tensor, [count = int32_t{0}](size_t, size_t) mutable { return count++; });
    int32_t value = 0;
    auto view = view_of(tensor);
    for_each(view, [&](size_t, size_t, int32_t v) {
        if (v != value++) fail("initialization mismatch");
    });
}

void test_slice_rows_mutates_storage() {
    constexpr size_t M = 32;
    constexpr size_t N = 16;
    auto tensor = make_tensor<int32_t, Extents2D, RowMajor>(Extents2D{M, N});
    fill(tensor, [value = int32_t{1}](size_t, size_t) mutable { return value++; });
    constexpr size_t offset = 8;
    constexpr size_t rows = 12;
    auto slice = slice_rows(tensor, offset, rows);
    auto slice_view = view_of(slice);
    fill(slice, [](size_t i, size_t j) { return -static_cast<int32_t>((i + 1) * 100 + j); });
    auto full = view_of(tensor);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < N; j++) {
            int32_t expected = -static_cast<int32_t>((i + 1) * 100 + j);
            if (full[offset + i, j] != expected) fail("slice mutation not reflected");
        }
    }
}

void test_copy_tensor_rowmajor() {
    constexpr size_t M = 40;
    constexpr size_t N = 24;
    auto src = make_tensor<int32_t, Extents2D, RowMajor>(Extents2D{M, N});
    auto dst = make_tensor<int32_t, Extents2D, RowMajor>(Extents2D{M, N});
    fill(src, [](size_t i, size_t j) { return static_cast<int32_t>((i + 3) * 100 + j); });
    fill<0>(dst);
    copy_tensor(src, dst);
    if (!approximately_equal(src, dst)) fail("copy_tensor mismatch");
}

void test_copy_tensor_slice_dims() {
    constexpr size_t M = 20;
    constexpr size_t N = 18;
    auto src = make_tensor<int32_t, Extents2D, RowMajor>(Extents2D{M, N});
    fill(src, [value = int32_t{0}](size_t, size_t) mutable { return value++; });
    auto src_view = view_of(src);
    constexpr size_t row_offset = 4;
    constexpr size_t row_count = 8;
    auto row_dst = make_tensor<int32_t, Extents2D, RowMajor>(Extents2D{row_count, N});
    fill<0>(row_dst);
    copy_tensor_slice(src, row_dst, 0, row_offset, row_count);
    auto row_view = view_of(row_dst);
    for_each(row_view, [&](size_t i, size_t j, int32_t v) {
        if (v != src_view[row_offset + i, j]) fail("row slice copy mismatch");
    });
    constexpr size_t col_offset = 5;
    constexpr size_t col_count = 6;
    auto col_dst = make_tensor<int32_t, Extents2D, RowMajor>(Extents2D{M, col_count});
    fill<0>(col_dst);
    copy_tensor_slice(src, col_dst, 1, col_offset, col_count);
    auto col_view = view_of(col_dst);
    for_each(col_view, [&](size_t i, size_t j, int32_t v) {
        if (v != src_view[i, col_offset + j]) fail("column slice copy mismatch");
    });
}

void test_column_major_support() {
    constexpr size_t M = 12;
    constexpr size_t N = 10;
    auto tensor = make_tensor<int32_t, Extents2D, ColumnMajor>(Extents2D{M, N});
    fill(tensor, [](size_t i, size_t j) { return static_cast<int32_t>(i + j * 100); });
    auto src_view = view_of(tensor);
    auto dst = make_tensor<int32_t, Extents2D, ColumnMajor>(Extents2D{M, N});
    copy_tensor(tensor, dst);
    if (!approximately_equal(tensor, dst)) fail("column-major copy mismatch");
    auto slice = slice_rows(tensor, 3, 6);
    auto slice_view = view_of(slice);
    for_each(slice_view, [&](size_t i, size_t j, int32_t v) {
        int32_t expected = src_view[i + 3, j];
        if (v != expected) fail("column-major slice mismatch");
    });
}

void test_row_major_with_padding() {
    constexpr size_t M = 6;
    constexpr size_t N = 5;
    constexpr size_t stride_row = 8;
    std::array<size_t, Extents2D::rank()> strides{stride_row, 1};
    auto tensor = make_tensor<int32_t, Extents2D, RowMajor>(Extents2D{M, N}, strides);
    fill(tensor, [value = int32_t{0}](size_t, size_t) mutable { return value++; });
    auto view = view_of(tensor);
    if (tensor.storage().size() != stride_row * (M - 1) + (N - 1) + 1) fail("padded tensor span size incorrect");
    auto slice = slice_rows(tensor, 2, 3);
    auto slice_view = view_of(slice);
    for_each(slice_view, [&](size_t i, size_t j, int32_t v) {
        if (v != view[i + 2, j]) fail("padded slice mismatch");
    });
    auto dst = make_tensor<int32_t, Extents2D, RowMajor>(Extents2D{M, N});
    fill<0>(dst);
    copy_tensor(tensor, dst);
    if (!approximately_equal(tensor, dst)) fail("padded copy mismatch");
}

void test_make_tensor_view_bridge() {
    constexpr size_t M = 8;
    constexpr size_t N = 5;
    std::vector<int32_t> backing(M * N);
    int32_t value = 1;
    for (auto& v : backing) v = value++;
    std::mdspan<int32_t, std::extents<size_t, M, N>, std::layout_right> md(backing.data());
    auto borrowed = make_tensor_view(md);
    auto view = view_of(borrowed);
    for_each(view, [&](size_t i, size_t j, int32_t v) {
        int32_t expected = static_cast<int32_t>(i * N + j + 1);
        if (v != expected) fail("borrowed view initial mismatch");
    });
    fill(borrowed, [](size_t i, size_t j) { return -static_cast<int32_t>(i * 10 + j); });
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            int32_t expected = -static_cast<int32_t>(i * 10 + j);
            if (backing[i * N + j] != expected) fail("borrowed view did not write back");
        }
    }
}

void test_quantized_tensor_copy() {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    constexpr size_t Tile = 16;
    auto src = make_quantized_tensor<int8_t, Extents2D, RowMajor, float, Tile, Tile>(Extents2D{M, N});
    auto dst = make_quantized_tensor<int8_t, Extents2D, RowMajor, float, Tile, Tile>(Extents2D{M, N});
    fill(src.data, [](size_t i, size_t j) { return static_cast<int8_t>((i * 17 + j * 3) % 64 - 32); });
    fill(src.scales, [cols = view_of(src.scales).extent(1)](size_t i, size_t j) {
        return 0.5f + static_cast<float>(i * cols + j) * 0.01f;
    });
    fill<0>(dst.data);
    fill<0.0f>(dst.scales);
    copy_quantized_tensor(src, dst);
    if (!approximately_equal(src.data, dst.data)) fail("quantized data copy mismatch");
    if (!approximately_equal(src.scales, dst.scales, 1e-6)) fail("quantized scale copy mismatch");
}

void test_quantized_tensor_slice_copy() {
    constexpr size_t M = 64;
    constexpr size_t N = 32;
    constexpr size_t TileRows = 16;
    constexpr size_t TileCols = 16;
    auto src = make_quantized_tensor<int8_t, Extents2D, RowMajor, float, TileRows, TileCols>(Extents2D{M, N});
    auto dst = make_quantized_tensor<int8_t, Extents2D, RowMajor, float, TileRows, TileCols>(Extents2D{M, N});
    fill(src.data, [](size_t i, size_t j) { return static_cast<int8_t>((i * 5 + j) % 32 - 16); });
    fill(src.scales, [](size_t i, size_t j) { return 1.0f + static_cast<float>(i + j) * 0.05f; });
    fill<0>(dst.data);
    fill<0.0f>(dst.scales);
    constexpr size_t slice_offset = 16;
    constexpr size_t slice_rows = 32;
    copy_quantized_tensor_slice(src, dst, 0, slice_offset, slice_rows);
    auto src_slice = make_quantized_slice(src, slice_offset, slice_rows);
    auto dst_slice = make_quantized_slice(dst, 0, slice_rows);
    if (!approximately_equal(src_slice.data, dst_slice.data)) fail("quantized slice data mismatch");
    if (!approximately_equal(src_slice.scales, dst_slice.scales, 1e-6)) fail("quantized slice scales mismatch");
    auto dst_full = view_of(dst.data);
    for (size_t i = slice_rows; i < dst_full.extent(0); i++) {
        for (size_t j = 0; j < dst_full.extent(1); j++) {
            if (dst_full[i, j] != 0) fail("quantized suffix should remain zero");
        }
    }
    auto dst_scale_full = view_of(dst.scales);
    constexpr size_t scale_rows = slice_rows / TileRows;
    for (size_t i = scale_rows; i < dst_scale_full.extent(0); i++) {
        for (size_t j = 0; j < dst_scale_full.extent(1); j++) {
            if (dst_scale_full[i, j] != 0.0f) fail("quantized scale suffix should remain zero");
        }
    }
}

} // namespace

export void run_tensor_tests() {
    run_case("Tensor allocation row-major", test_tensor_allocation_rowmajor);
    run_case("Row slices mutate backing storage", test_slice_rows_mutates_storage);
    run_case("copy_tensor row-major", test_copy_tensor_rowmajor);
    run_case("copy_tensor_slice across dimensions", test_copy_tensor_slice_dims);
    run_case("Column-major support", test_column_major_support);
    run_case("Row-major with padding", test_row_major_with_padding);
    run_case("Borrowed tensor view bridge", test_make_tensor_view_bridge);
    run_case("Quantized tensor copy", test_quantized_tensor_copy);
    run_case("Quantized tensor slice copy", test_quantized_tensor_slice_copy);
}
