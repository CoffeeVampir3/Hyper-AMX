module;
#include <print>
#include <cstdlib>
#include <sys/syscall.h>
#include <unistd.h>
#include <cmath>
#include <mdspan>
export module modern_matmul_correctness;
import moderntensor;
import modernlayout;
import modernmatmul;
import modernnuma;

using namespace Numa;

constexpr auto ARCH_REQ_XCOMP_PERM = 0x1023;
constexpr auto XFEATURE_XTILEDATA = 18;

using Extents2D = std::dextents<size_t, 2>;
using VNNILayout = Layout::VNNI<256, 4096>;

bool request_amx() {
    return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0;
}

template<TensorStorage T, typename Fn>
void fill(T& tensor, Fn&& fn) {
    auto view = tensor.view();
    if constexpr (requires { view.extent(0); view.extent(1); }) {
        for (size_t i = 0; i < view.extent(0); i++) {
            for (size_t j = 0; j < view.extent(1); j++) {
                view[i, j] = fn(i, j);
            }
        }
    }
}

template<TensorStorage T>
void zero(T& tensor) {
    fill(tensor, [](auto...) { return 0; });
}

template<typename View1, typename View2, typename T>
bool check_approximate_equal(View1 v1, View2 v2, T tolerance, const char* name) {
    for (size_t i = 0; i < v1.extent(0); i++) {
        for (size_t j = 0; j < v1.extent(1); j++) {
            auto diff = v1[i, j] - v2[i, j];
            if (std::abs(diff) > tolerance) {
                std::println(stderr, "   ✗ {} FAILED at [{}, {}]: {} vs {}", name, i, j, v1[i, j], v2[i, j]);
                return false;
            }
        }
    }
    return true;
}

void reference_matmul(const auto& A, const auto& B, auto C) {
    for (size_t i = 0; i < A.extent(0); i++) {
        for (size_t j = 0; j < B.extent(1); j++) {
            int32_t sum = 0;
            for (size_t k = 0; k < A.extent(1); k++) {
                sum += static_cast<int32_t>(A[i, k]) * static_cast<int32_t>(B[k, j]);
            }
            C[i, j] = sum;
        }
    }
}

void test_core_amx() {
    std::println("1. Core AMX kernel");
    if (!request_amx()) std::exit(1);

    constexpr size_t M = 32, K = 64, N = 32;
    Tensor<int8_t, Extents2D, Layout::RowMajor> A(Extents2D{M, K});
    Tensor<int8_t, Extents2D, Layout::RowMajor> B_row(Extents2D{K, N});
    int8_t val = 1;
    fill(A, [&](auto...) { return (val++ % 126) + 1; });
    val = 1;
    fill(B_row, [&](auto...) { return (val++ % 126) + 1; });

    Tensor<int32_t, Extents2D, Layout::RowMajor> C_ref(Extents2D{M, N});
    zero(C_ref);
    reference_matmul(A.view(), B_row.view(), C_ref.view());

    Tensor<int8_t, Extents2D, VNNILayout> B_vnni(Extents2D{K, N});
    auto B_vnni_view = B_vnni.view();
    VNNILayout::copy_from(B_row.view(), B_vnni_view, 1, 0, N);

    Tensor<int32_t, Extents2D, Layout::RowMajor> C_amx(Extents2D{M, N});
    zero(C_amx);
    matmul_amx_int8_blocked(A.view(), B_vnni.view(), C_amx.view());

    if (!check_approximate_equal(C_amx.view(), C_ref.view(), int32_t{0}, "Core AMX")) std::exit(1);
    std::println("   ✓ AMX intrinsics work correctly\n");
}

void test_vnni_layout() {
    std::println("2. VNNI layout indexing");
    if (!request_amx()) std::exit(1);

    constexpr size_t M = 32, K = 64, N = 32;
    Tensor<int8_t, Extents2D, Layout::RowMajor> A(Extents2D{M, K});
    Tensor<int8_t, Extents2D, Layout::RowMajor> B_row(Extents2D{K, N});
    int8_t val = 1;
    fill(A, [&](auto...) { return (val++ % 126) + 1; });
    val = 1;
    fill(B_row, [&](auto...) { return (val++ % 126) + 1; });

    Tensor<int32_t, Extents2D, Layout::RowMajor> C_ref(Extents2D{M, N});
    zero(C_ref);
    reference_matmul(A.view(), B_row.view(), C_ref.view());

    Tensor<int8_t, Extents2D, VNNILayout> B_vnni(Extents2D{K, N});
    for (size_t k = 0; k < K; k++)
        for (size_t n = 0; n < N; n++)
            B_vnni.view()[k, n] = B_row.view()[k, n];

    Tensor<int32_t, Extents2D, Layout::RowMajor> C_amx(Extents2D{M, N});
    zero(C_amx);
    matmul_amx_int8_blocked(A.view(), B_vnni.view(), C_amx.view());

    if (!check_approximate_equal(C_amx.view(), C_ref.view(), int32_t{0}, "VNNI layout")) std::exit(1);
    std::println("   ✓ VNNI layout mapping is correct\n");
}

void test_partitioning() {
    std::println("3. Column partitioning");
    if (!request_amx()) std::exit(1);

    constexpr size_t M = 32, K = 64, N = 64, PARTS = 2;
    Tensor<int8_t, Extents2D, Layout::RowMajor> A(Extents2D{M, K});
    Tensor<int8_t, Extents2D, Layout::RowMajor> B_row(Extents2D{K, N});
    int8_t val = 1;
    fill(A, [&](auto...) { return (val++ % 126) + 1; });
    val = 1;
    fill(B_row, [&](auto...) { return (val++ % 126) + 1; });

    Tensor<int32_t, Extents2D, Layout::RowMajor> C_ref(Extents2D{M, N});
    zero(C_ref);
    reference_matmul(A.view(), B_row.view(), C_ref.view());

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
        zero(C_parts[p]);
        matmul_amx_int8_blocked(A.view(), B_part.view(p), C_parts[p].view());
    }

    Tensor<int32_t, Extents2D, Layout::RowMajor> C_gathered(Extents2D{M, N});
    for (size_t p = 0; p < PARTS; p++)
        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < N/PARTS; j++)
                C_gathered.view()[i, p * (N/PARTS) + j] = C_parts[p].view()[i, j];

    if (!check_approximate_equal(C_gathered.view(), C_ref.view(), int32_t{0}, "Partitioning")) std::exit(1);
    std::println("   ✓ Column slicing and gather work correctly\n");
}

void test_numa_types() {
    std::println("4. NUMA types (single-threaded)");
    if (!request_amx()) std::exit(1);

    constexpr size_t M = 32, K = 64, N = 64;
    auto config = DualSocketConfig::discover();

    Tensor<int8_t, Extents2D, Layout::RowMajor> A_src(Extents2D{M, K});
    Tensor<int8_t, Extents2D, Layout::RowMajor> B_src(Extents2D{K, N});
    int8_t val = 1;
    fill(A_src, [&](auto...) { return (val++ % 126) + 1; });
    val = 1;
    fill(B_src, [&](auto...) { return (val++ % 126) + 1; });

    Tensor<int32_t, Extents2D, Layout::RowMajor> C_ref(Extents2D{M, N});
    zero(C_ref);
    reference_matmul(A_src.view(), B_src.view(), C_ref.view());

    Replicated<int8_t, Extents2D, Layout::RowMajor> A_repl(A_src, config);

    Tensor<int8_t, Extents2D, VNNILayout> B_vnni_full(Extents2D{K, N});
    auto B_vnni_full_view = B_vnni_full.view();
    VNNILayout::copy_from(B_src.view(), B_vnni_full_view, 1, 0, N);

    ColumnPartitioned<int8_t, Extents2D, VNNILayout> B_part(B_vnni_full, 2, config);
    ColumnPartitioned<int32_t, Extents2D, Layout::RowMajor> C_part(Extents2D{M, N}, 2, config);

    for (int s = 0; s < 2; s++)
        matmul_amx_int8_blocked(A_repl.view(s), B_part.view(s), C_part.view(s));

    auto C_gathered = all_gather(C_part);
    if (!check_approximate_equal(C_gathered.view(), C_ref.view(), int32_t{0}, "NUMA types")) std::exit(1);
    std::println("   ✓ Replicated and ColumnPartitioned work\n");
}

void test_numa_multithreaded() {
    std::println("5. Full multi-threaded NUMA");
    if (!request_amx()) std::exit(1);

    constexpr size_t M = 64, K = 128, N = 64;
    auto config = DualSocketConfig::discover();

    Tensor<int8_t, Extents2D, Layout::RowMajor> A_src(Extents2D{M, K});
    Tensor<int8_t, Extents2D, Layout::RowMajor> B_src(Extents2D{K, N});
    int8_t val = 1;
    fill(A_src, [&](auto...) { return (val++ % 126) + 1; });
    val = 1;
    fill(B_src, [&](auto...) { return (val++ % 126) + 1; });

    Tensor<int32_t, Extents2D, Layout::RowMajor> C_ref(Extents2D{M, N});
    zero(C_ref);
    reference_matmul(A_src.view(), B_src.view(), C_ref.view());

    Replicated<int8_t, Extents2D, Layout::RowMajor> A_repl(A_src, config);

    Tensor<int8_t, Extents2D, VNNILayout> B_vnni_full(Extents2D{K, N});
    auto B_vnni_full_view = B_vnni_full.view();
    VNNILayout::copy_from(B_src.view(), B_vnni_full_view, 1, 0, N);

    ColumnPartitioned<int8_t, Extents2D, VNNILayout> B_part(B_vnni_full, 2, config);
    ColumnPartitioned<int32_t, Extents2D, Layout::RowMajor> C_part(Extents2D{M, N}, 2, config);

    matmul_amx_column_parallel(A_repl, B_part, C_part, config);

    auto C_gathered = all_gather(C_part);
    if (!check_approximate_equal(C_gathered.view(), C_ref.view(), int32_t{0}, "Multi-threaded")) std::exit(1);
    std::println("   ✓ Multi-threaded column-parallel works\n");
}

void test_row_parallel() {
    std::println("6. Row-parallel (K-split) matmul");
    if (!request_amx()) std::exit(1);

    constexpr size_t M = 256, K = 512, N = 256;
    auto config = DualSocketConfig::discover();

    Tensor<int8_t, Extents2D, Layout::RowMajor> A_src(Extents2D{M, K});
    Tensor<int8_t, Extents2D, Layout::RowMajor> B_src(Extents2D{K, N});
    int8_t val = 1;
    fill(A_src, [&](auto...) { return (val++ % 126) + 1; });
    val = 1;
    fill(B_src, [&](auto...) { return (val++ % 126) + 1; });

    Tensor<int32_t, Extents2D, Layout::RowMajor> C_ref(Extents2D{M, N});
    zero(C_ref);
    reference_matmul(A_src.view(), B_src.view(), C_ref.view());

    ColumnPartitioned<int8_t, Extents2D, Layout::RowMajor> A_part(A_src, 2, config);

    Tensor<int8_t, Extents2D, VNNILayout> B_vnni_full(Extents2D{K, N});
    auto B_vnni_full_view = B_vnni_full.view();
    VNNILayout::copy_from(B_src.view(), B_vnni_full_view, 1, 0, N);

    RowPartitioned<int8_t, Extents2D, VNNILayout> B_part(B_vnni_full, 2, config);
    Replicated<int32_t, Extents2D, Layout::RowMajor> C_partials(Extents2D{M, N}, config);

    matmul_amx_row_parallel(A_part, B_part, C_partials, config);
    auto C_result = all_reduce_sum(C_partials, 0, config);

    if (!check_approximate_equal(C_result.view(), C_ref.view(), int32_t{0}, "Row-parallel")) std::exit(1);
    std::println("   ✓ Row-parallel matmul with all-reduce works\n");
}

export void run_modern_matmul_correctness() {
    std::println("=== Modern Incremental Validation Tests ===\n");

    try {
        test_core_amx();
        test_vnni_layout();
        test_partitioning();
        test_numa_types();
        test_numa_multithreaded();
        test_row_parallel();

        std::println("=== All modern correctness tests passed! ===");
    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        std::exit(1);
    }
}
