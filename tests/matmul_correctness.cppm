module;
#include <print>
#include <cstdlib>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>
export module matmul_correctness;
import tensor;
import matmul;
import numaaware;
import tensor_utils;
import quantization;

using namespace amx;

constexpr auto ARCH_REQ_XCOMP_PERM = 0x1023;
constexpr auto XFEATURE_XTILEDATA = 18;

using VNNILayout = vnni_layout<256, 4096>;
inline auto vnni_column_partition(auto view, size_t parts, const auto& config) {
    return make_column_partitioned_from<int8_t, Extents2D, RowMajor2D, VNNILayout, LayoutConversion::ToVNNI>(
        view, parts, config
    );
}

inline auto vnni_row_partition(auto view, size_t parts, const auto& config) {
    return make_row_partitioned_from<int8_t, Extents2D, RowMajor2D, VNNILayout, LayoutConversion::ToVNNI>(
        view, parts, config
    );
}

bool request_amx() {
    return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0;
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
    auto A = make_tensor<int8_t>(M, K);
    auto B_row = make_tensor<int8_t>(K, N);
    int8_t val = 1;
    fill(A, [&](auto...) { return (val++ % 126) + 1; });
    val = 1;
    fill(B_row, [&](auto...) { return (val++ % 126) + 1; });

    auto C_ref = make_tensor<int32_t>(M, N);
    zero(C_ref);
    reference_matmul(A.view(), B_row.view(), C_ref.view());

    auto B_vnni = Tensor<int8_t, Extents2D, VNNILayout>(Extents2D{K, N}, AlignedAllocator<int8_t>{});
    convert_to_vnni(B_row.view(), B_vnni.view());

    auto C_amx = make_tensor<int32_t>(M, N);
    zero(C_amx);
    matmul_amx_int8_blocked(A.view(), B_vnni.view(), C_amx.view());

    if (!check_approximate_equal(C_amx.view(), C_ref.view(), int32_t{0}, "Core AMX")) std::exit(1);
    std::println("   ✓ AMX intrinsics work correctly\n");
}

void test_vnni_layout() {
    std::println("2. VNNI layout indexing");
    if (!request_amx()) std::exit(1);

    constexpr size_t M = 32, K = 64, N = 32;
    auto A = make_tensor<int8_t>(M, K);
    auto B_row = make_tensor<int8_t>(K, N);
    int8_t val = 1;
    fill(A, [&](auto...) { return (val++ % 126) + 1; });
    val = 1;
    fill(B_row, [&](auto...) { return (val++ % 126) + 1; });

    auto C_ref = make_tensor<int32_t>(M, N);
    zero(C_ref);
    reference_matmul(A.view(), B_row.view(), C_ref.view());

    auto B_vnni = Tensor<int8_t, Extents2D, VNNILayout>(Extents2D{K, N}, AlignedAllocator<int8_t>{});
    for (size_t k = 0; k < K; k++)
        for (size_t n = 0; n < N; n++)
            B_vnni[k, n] = B_row[k, n];

    auto C_amx = make_tensor<int32_t>(M, N);
    zero(C_amx);
    matmul_amx_int8_blocked(A.view(), B_vnni.view(), C_amx.view());

    if (!check_approximate_equal(C_amx.view(), C_ref.view(), int32_t{0}, "VNNI layout")) std::exit(1);
    std::println("   ✓ VNNI layout mapping is correct\n");
}

void test_partitioning() {
    std::println("3. Column partitioning");
    if (!request_amx()) std::exit(1);

    constexpr size_t M = 32, K = 64, N = 64, PARTS = 2;
    auto A = make_tensor<int8_t>(M, K);
    auto B_row = make_tensor<int8_t>(K, N);
    int8_t val = 1;
    fill(A, [&](auto...) { return (val++ % 126) + 1; });
    val = 1;
    fill(B_row, [&](auto...) { return (val++ % 126) + 1; });

    auto C_ref = make_tensor<int32_t>(M, N);
    zero(C_ref);
    reference_matmul(A.view(), B_row.view(), C_ref.view());

    std::vector<Tensor<int8_t, Extents2D, VNNILayout>> B_parts;
    std::vector<Tensor<int32_t, Extents2D, RowMajor2D>> C_parts;

    for (size_t p = 0; p < PARTS; p++) {
        auto B_slice = slice<1>(B_row.view(), p * (N/PARTS), N/PARTS);
        B_parts.emplace_back(Extents2D{K, N/PARTS}, AlignedAllocator<int8_t>{});
        convert_to_vnni(B_slice, B_parts.back().view());
        C_parts.emplace_back(Extents2D{M, N/PARTS}, AlignedAllocator<int32_t>{});
        zero(C_parts.back());
        matmul_amx_int8_blocked(A.view(), B_parts[p].view(), C_parts[p].view());
    }

    auto C_gathered = make_tensor<int32_t>(M, N);
    for (size_t p = 0; p < PARTS; p++)
        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < N/PARTS; j++)
                C_gathered[i, p * (N/PARTS) + j] = C_parts[p][i, j];

    if (!check_approximate_equal(C_gathered.view(), C_ref.view(), int32_t{0}, "Partitioning")) std::exit(1);
    std::println("   ✓ Column slicing and gather work correctly\n");
}

void test_numa_types() {
    std::println("4. NUMA types (single-threaded)");
    if (!request_amx()) std::exit(1);

    constexpr size_t M = 32, K = 64, N = 64;
    auto config = DualSocketConfig::discover();

    auto A_src = make_tensor<int8_t>(M, K);
    auto B_src = make_tensor<int8_t>(K, N);
    int8_t val = 1;
    fill(A_src, [&](auto...) { return (val++ % 126) + 1; });
    val = 1;
    fill(B_src, [&](auto...) { return (val++ % 126) + 1; });

    auto C_ref = make_tensor<int32_t>(M, N);
    zero(C_ref);
    reference_matmul(A_src.view(), B_src.view(), C_ref.view());

    auto A_repl = make_socket_replicated_from<int8_t, Extents2D, RowMajor2D>(A_src.view(), config);

    auto B_part = vnni_column_partition(B_src.view(), 2, config);

    ColumnPartitioned<int32_t, Extents2D, RowMajor2D> C_part(Extents2D{M, N}, 2, config);

    for (int s = 0; s < 2; s++)
        matmul_amx_int8_blocked(A_repl.view(s), B_part.view(s), C_part.view(s));

    auto C_gathered = all_gather(C_part);
    if (!check_approximate_equal(C_gathered.view(), C_ref.view(), int32_t{0}, "NUMA types")) std::exit(1);
    std::println("   ✓ SocketReplicated and ColumnPartitioned work\n");
}

void test_numa_multithreaded() {
    std::println("5. Full multi-threaded NUMA");
    if (!request_amx()) std::exit(1);

    constexpr size_t M = 64, K = 128, N = 64;
    auto config = DualSocketConfig::discover();

    auto A_src = make_tensor<int8_t>(M, K);
    auto B_src = make_tensor<int8_t>(K, N);
    int8_t val = 1;
    fill(A_src, [&](auto...) { return (val++ % 126) + 1; });
    val = 1;
    fill(B_src, [&](auto...) { return (val++ % 126) + 1; });

    auto C_ref = make_tensor<int32_t>(M, N);
    zero(C_ref);
    reference_matmul(A_src.view(), B_src.view(), C_ref.view());

    auto A_repl = make_socket_replicated_from<int8_t, Extents2D, RowMajor2D>(A_src.view(), config);

    auto B_part = vnni_column_partition(B_src.view(), 2, config);

    ColumnPartitioned<int32_t, Extents2D, RowMajor2D> C_part(Extents2D{M, N}, 2, config);

    matmul_amx_column_parallel(A_repl, B_part, C_part, config);

    auto C_gathered = all_gather(C_part);
    if (!check_approximate_equal(C_gathered.view(), C_ref.view(), int32_t{0}, "Multi-threaded")) std::exit(1);
    std::println("   ✓ Multi-threaded column-parallel works\n");
}

void test_quantized_matmul() {
    std::println("6. Quantized matmul");
    if (!request_amx()) std::exit(1);

    constexpr size_t M = 64, K = 128, N = 64;
    auto config = DualSocketConfig::discover();

    auto A_src = make_tensor<int8_t>(M, K);
    auto B_src = make_tensor<int8_t>(K, N);
    int8_t val = 1;
    fill(A_src, [&](auto...) { return (val++ % 126) + 1; });
    val = 1;
    fill(B_src, [&](auto...) { return (val++ % 126) + 1; });

    auto C_ref = make_tensor<int32_t>(M, N);
    zero(C_ref);
    reference_matmul(A_src.view(), B_src.view(), C_ref.view());

    auto A_repl = make_socket_replicated_from<int8_t, Extents2D, RowMajor2D>(A_src.view(), config);

    auto B_part = vnni_column_partition(B_src.view(), 2, config);

    ColumnPartitioned<int8_t, Extents2D, RowMajor2D> C_part(Extents2D{M, N}, 2, config);

    ColumnPartitioned<AMXQ::QuantizationParams, Extents2D, RowMajor2D> params_part(
        Extents2D{M / TILE_M, N / TILE_N}, 2, config
    );

    matmul_amx_column_parallel_quantized(A_repl, B_part, C_part, params_part, config);

    auto C_quantized = all_gather(C_part);
    auto params_gathered = all_gather(params_part);

    auto C_dequantized = make_tensor<int32_t>(M, N);
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            size_t tile_i = i / TILE_M;
            size_t tile_j = j / TILE_N;
            auto params = params_gathered[tile_i, tile_j];
            C_dequantized[i, j] = AMXQ::dequantize_scalar(C_quantized[i, j], params.bias, params.scale);
        }
    }

    int32_t tolerance = 1000;
    if (!check_approximate_equal(C_dequantized.view(), C_ref.view(), tolerance, "Quantized")) std::exit(1);
    std::println("   ✓ Quantized matmul within tolerance\n");
}

void test_row_parallel() {
    std::println("7. Row-parallel (K-split) matmul");
    if (!request_amx()) std::exit(1);

    constexpr size_t M = 256, K = 512, N = 256;
    auto config = DualSocketConfig::discover();

    auto A_src = make_tensor<int8_t>(M, K);
    auto B_src = make_tensor<int8_t>(K, N);
    int8_t val = 1;
    fill(A_src, [&](auto...) { return (val++ % 126) + 1; });
    val = 1;
    fill(B_src, [&](auto...) { return (val++ % 126) + 1; });

    auto C_ref = make_tensor<int32_t>(M, N);
    zero(C_ref);
    reference_matmul(A_src.view(), B_src.view(), C_ref.view());

    auto A_part = make_column_partitioned_from<int8_t, Extents2D, RowMajor2D>(A_src.view(), 2, config);
    auto B_part = vnni_row_partition(B_src.view(), 2, config);

    SocketReplicated<int32_t, Extents2D, RowMajor2D> C_partials(Extents2D{M, N}, config);

    matmul_amx_row_parallel(A_part, B_part, C_partials, config);
    auto C_result = all_reduce_sum(C_partials, 0, config);

    if (!check_approximate_equal(C_result.view(), C_ref.view(), int32_t{0}, "Row-parallel")) std::exit(1);
    std::println("   ✓ Row-parallel matmul with all-reduce works\n");
}

export void run_matmul_correctness() {
    std::println("=== Incremental Validation Tests ===\n");

    try {
        test_core_amx();
        test_vnni_layout();
        test_partitioning();
        test_numa_types();
        test_numa_multithreaded();
        test_quantized_matmul();
        test_row_parallel();

        std::println("=== All correctness tests passed! ===");
    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        std::exit(1);
    }
}
