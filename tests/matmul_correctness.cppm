module;
#include <print>
#include <cstdlib>
#include <mdspan>
export module matmul_correctness;
import tensor;
import layout;
import amx_gemms;
import numa;
import tensor_utils;

using namespace Numa;

using Extents2D = std::dextents<size_t, 2>;
using VNNILayout = Layout::VNNI<256, 4096>;

void test_core_amx() {
    std::println("Core AMX kernel");
    if (!utils::request_amx()) std::exit(1);

    constexpr size_t M = 32, K = 64, N = 32;
    Tensor<int8_t, Extents2D, Layout::RowMajor> A(Extents2D{M, K});
    Tensor<int8_t, Extents2D, Layout::RowMajor> B_row(Extents2D{K, N});
    int8_t val = 1;
    utils::fill(A, [&](auto...) { return (val++ % 126) + 1; });
    val = 1;
    utils::fill(B_row, [&](auto...) { return (val++ % 126) + 1; });

    Tensor<int32_t, Extents2D, Layout::RowMajor> C_ref(Extents2D{M, N});
    utils::zero(C_ref);
    utils::reference_matmul(A.view(), B_row.view(), C_ref.view());

    Tensor<int8_t, Extents2D, VNNILayout> B_vnni(Extents2D{K, N});
    auto B_vnni_view = B_vnni.view();
    VNNILayout::copy_from(B_row.view(), B_vnni_view, 1, 0, N);

    Tensor<int32_t, Extents2D, Layout::RowMajor> C_amx(Extents2D{M, N});
    utils::zero(C_amx);
    cpugemm::i8_i8_i32_blocked(A.view(), B_vnni.view(), C_amx.view());

    if (!utils::check_approximate_equal(C_amx.view(), C_ref.view(), int32_t{0}, "Core AMX")) std::exit(1);
    std::println("   ✓ AMX intrinsics\n");
}

void test_vnni_layout() {
    std::println("VNNI layout indexing");
    if (!utils::request_amx()) std::exit(1);

    constexpr size_t M = 32, K = 64, N = 32;
    Tensor<int8_t, Extents2D, Layout::RowMajor> A(Extents2D{M, K});
    Tensor<int8_t, Extents2D, Layout::RowMajor> B_row(Extents2D{K, N});
    int8_t val = 1;
    utils::fill(A, [&](auto...) { return (val++ % 126) + 1; });
    val = 1;
    utils::fill(B_row, [&](auto...) { return (val++ % 126) + 1; });

    Tensor<int32_t, Extents2D, Layout::RowMajor> C_ref(Extents2D{M, N});
    utils::zero(C_ref);
    utils::reference_matmul(A.view(), B_row.view(), C_ref.view());

    Tensor<int8_t, Extents2D, VNNILayout> B_vnni(Extents2D{K, N});
    for (size_t k = 0; k < K; k++)
        for (size_t n = 0; n < N; n++)
            B_vnni.view()[k, n] = B_row.view()[k, n];

    Tensor<int32_t, Extents2D, Layout::RowMajor> C_amx(Extents2D{M, N});
    utils::zero(C_amx);
    cpugemm::i8_i8_i32_blocked(A.view(), B_vnni.view(), C_amx.view());

    if (!utils::check_approximate_equal(C_amx.view(), C_ref.view(), int32_t{0}, "VNNI layout")) std::exit(1);
    std::println("   ✓ VNNI layout mapping\n");
}

void test_partitioning() {
    std::println("Column partitioning");
    if (!utils::request_amx()) std::exit(1);

    constexpr size_t M = 32, K = 64, N = 64, PARTS = 2;
    Tensor<int8_t, Extents2D, Layout::RowMajor> A(Extents2D{M, K});
    Tensor<int8_t, Extents2D, Layout::RowMajor> B_row(Extents2D{K, N});
    int8_t val = 1;
    utils::fill(A, [&](auto...) { return (val++ % 126) + 1; });
    val = 1;
    utils::fill(B_row, [&](auto...) { return (val++ % 126) + 1; });

    Tensor<int32_t, Extents2D, Layout::RowMajor> C_ref(Extents2D{M, N});
    utils::zero(C_ref);
    utils::reference_matmul(A.view(), B_row.view(), C_ref.view());

    Tensor<int8_t, Extents2D, VNNILayout> B_vnni_full(Extents2D{K, N});
    auto B_vnni_full_view = B_vnni_full.view();
    VNNILayout::copy_from(B_row.view(), B_vnni_full_view, 1, 0, N);

    auto config = DualSocketConfig::discover();
    ColumnPartitioned<int8_t, Extents2D, VNNILayout> B_part(B_vnni_full, PARTS, config);

    std::array<Tensor<int32_t, Extents2D, Layout::RowMajor>, PARTS> C_parts = {
        Tensor<int32_t, Extents2D, Layout::RowMajor>(Extents2D{M, N/PARTS}),
        Tensor<int32_t, Extents2D, Layout::RowMajor>(Extents2D{M, N/PARTS})
    };

    for (size_t p = 0; p < PARTS; p++) {
        utils::zero(C_parts[p]);
        cpugemm::i8_i8_i32_blocked(A.view(), B_part.view(p), C_parts[p].view());
    }

    Tensor<int32_t, Extents2D, Layout::RowMajor> C_gathered(Extents2D{M, N});
    for (size_t p = 0; p < PARTS; p++)
        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < N/PARTS; j++)
                C_gathered.view()[i, p * (N/PARTS) + j] = C_parts[p].view()[i, j];

    if (!utils::check_approximate_equal(C_gathered.view(), C_ref.view(), int32_t{0}, "Partitioning")) std::exit(1);
    std::println("   ✓ Column slicing and gather\n");
}

void test_numa_types() {
    std::println("NUMA types (single-threaded)");
    if (!utils::request_amx()) std::exit(1);

    constexpr size_t M = 32, K = 64, N = 64;
    auto config = DualSocketConfig::discover();

    Tensor<int8_t, Extents2D, Layout::RowMajor> A_src(Extents2D{M, K});
    Tensor<int8_t, Extents2D, Layout::RowMajor> B_src(Extents2D{K, N});
    int8_t val = 1;
    utils::fill(A_src, [&](auto...) { return (val++ % 126) + 1; });
    val = 1;
    utils::fill(B_src, [&](auto...) { return (val++ % 126) + 1; });

    Tensor<int32_t, Extents2D, Layout::RowMajor> C_ref(Extents2D{M, N});
    utils::zero(C_ref);
    utils::reference_matmul(A_src.view(), B_src.view(), C_ref.view());

    Replicated<int8_t, Extents2D, Layout::RowMajor> A_repl(A_src, config);

    Tensor<int8_t, Extents2D, VNNILayout> B_vnni_full(Extents2D{K, N});
    auto B_vnni_full_view = B_vnni_full.view();
    VNNILayout::copy_from(B_src.view(), B_vnni_full_view, 1, 0, N);

    ColumnPartitioned<int8_t, Extents2D, VNNILayout> B_part(B_vnni_full, 2, config);
    ColumnPartitioned<int32_t, Extents2D, Layout::RowMajor> C_part(Extents2D{M, N}, 2, config);

    for (int s = 0; s < 2; s++)
        cpugemm::i8_i8_i32_blocked(A_repl.view(s), B_part.view(s), C_part.view(s));

    Tensor<int32_t, Extents2D, Layout::RowMajor> C_gathered(Extents2D{M, N});
    all_gather(C_part, C_gathered);
    if (!utils::check_approximate_equal(C_gathered.view(), C_ref.view(), int32_t{0}, "NUMA types")) std::exit(1);
    std::println("   ✓ Replicated and ColumnPartitioned\n");
}

void test_numa_multithreaded() {
    std::println("Full multi-threaded NUMA");
    if (!utils::request_amx()) std::exit(1);

    constexpr size_t M = 64, K = 128, N = 64;
    auto config = DualSocketConfig::discover();

    Tensor<int8_t, Extents2D, Layout::RowMajor> A_src(Extents2D{M, K});
    Tensor<int8_t, Extents2D, Layout::RowMajor> B_src(Extents2D{K, N});
    int8_t val = 1;
    utils::fill(A_src, [&](auto...) { return (val++ % 126) + 1; });
    val = 1;
    utils::fill(B_src, [&](auto...) { return (val++ % 126) + 1; });

    Tensor<int32_t, Extents2D, Layout::RowMajor> C_ref(Extents2D{M, N});
    utils::zero(C_ref);
    utils::reference_matmul(A_src.view(), B_src.view(), C_ref.view());

    Replicated<int8_t, Extents2D, Layout::RowMajor> A_repl(A_src, config);

    Tensor<int8_t, Extents2D, VNNILayout> B_vnni_full(Extents2D{K, N});
    auto B_vnni_full_view = B_vnni_full.view();
    VNNILayout::copy_from(B_src.view(), B_vnni_full_view, 1, 0, N);

    ColumnPartitioned<int8_t, Extents2D, VNNILayout> B_part(B_vnni_full, 2, config);
    ColumnPartitioned<int32_t, Extents2D, Layout::RowMajor> C_part(Extents2D{M, N}, 2, config);

    matmul_amx_column_parallel(A_repl, B_part, C_part, config);

    Tensor<int32_t, Extents2D, Layout::RowMajor> C_gathered(Extents2D{M, N});
    all_gather(C_part, C_gathered);
    if (!utils::check_approximate_equal(C_gathered.view(), C_ref.view(), int32_t{0}, "Multi-threaded")) std::exit(1);
    std::println("   ✓ Multi-threaded Column-parallel\n");
}

void test_row_parallel() {
    std::println("Row-parallel (K-split) matmul");
    if (!utils::request_amx()) std::exit(1);

    constexpr size_t M = 256, K = 512, N = 256;
    auto config = DualSocketConfig::discover();

    Tensor<int8_t, Extents2D, Layout::RowMajor> A_src(Extents2D{M, K});
    Tensor<int8_t, Extents2D, Layout::RowMajor> B_src(Extents2D{K, N});
    int8_t val = 1;
    utils::fill(A_src, [&](auto...) { return (val++ % 126) + 1; });
    val = 1;
    utils::fill(B_src, [&](auto...) { return (val++ % 126) + 1; });

    Tensor<int32_t, Extents2D, Layout::RowMajor> C_ref(Extents2D{M, N});
    utils::zero(C_ref);
    utils::reference_matmul(A_src.view(), B_src.view(), C_ref.view());

    ColumnPartitioned<int8_t, Extents2D, Layout::RowMajor> A_part(A_src, 2, config);

    Tensor<int8_t, Extents2D, VNNILayout> B_vnni_full(Extents2D{K, N});
    auto B_vnni_full_view = B_vnni_full.view();
    VNNILayout::copy_from(B_src.view(), B_vnni_full_view, 1, 0, N);

    RowPartitioned<int8_t, Extents2D, VNNILayout> B_part(B_vnni_full, 2, config);
    Replicated<int32_t, Extents2D, Layout::RowMajor> C_partials(Extents2D{M, N}, config);

    matmul_amx_row_parallel(A_part, B_part, C_partials, config);
    int node = DualSocketConfig::primary_node_for_socket(0);
    Tensor<int32_t, Extents2D, Layout::RowMajor> C_result(Extents2D{M, N}, NumaAllocator<int32_t>{node});
    all_reduce_sum(C_partials, C_result, 0, config);

    if (!utils::check_approximate_equal(C_result.view(), C_ref.view(), int32_t{0}, "Row-parallel")) std::exit(1);
    std::println("   ✓ Multi-threaded Row-parallel\n");
}

export void run_modern_matmul_correctness() {
    try {
        test_core_amx();
        test_vnni_layout();
        test_partitioning();
        test_numa_types();
        test_numa_multithreaded();
        test_row_parallel();
    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        std::exit(1);
    }
}
