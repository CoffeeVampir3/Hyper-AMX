module;
#include <immintrin.h>
#include <cstdint>
#include <thread>
#include <vector>
#include <mdspan>
#include <sched.h>
export module amx_gemms;
import tensor;
import layout;
import avx512;

namespace amx {
    constexpr size_t TILE_M = 16;
    constexpr size_t TILE_K = 64;
    constexpr size_t TILE_N = 16;
    constexpr size_t TILE_STRIDE_BYTES = 64;
    constexpr size_t ACCUM_STRIDE_BYTES = 128;
}

struct TileConfig {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
};

template<typename View>
concept Int8RowMajor = requires(View v) {
    requires std::same_as<typename View::element_type, int8_t>;
    requires std::same_as<typename View::layout_type, Layout::RowMajor>;
};

template<typename View>
concept Int8VNNI = requires(View v) {
    requires std::same_as<typename View::element_type, int8_t>;
    requires requires { typename View::layout_type::is_vnni_layout; };
};

template<typename View>
concept Int32RowMajor = requires(View v) {
    requires std::same_as<typename View::element_type, int32_t>;
    requires std::same_as<typename View::layout_type, Layout::RowMajor>;
};

namespace detail {
template<typename View>
constexpr size_t stride_bytes(const View& v) {
    return v.stride(0) * sizeof(typename View::element_type);
}

struct Range { size_t start, end; };

constexpr Range partition_range(size_t total, int thread_id, int num_threads) {
    size_t per_thread = (total + num_threads - 1) / num_threads;
    size_t start = thread_id * per_thread;
    return {start, std::min(total, start + per_thread)};
}

constexpr TileConfig make_i8_gemm_config() {
    TileConfig cfg{};
    cfg.palette_id = 1;
    cfg.rows[0] = cfg.rows[1] = amx::TILE_M;
    cfg.colsb[0] = cfg.colsb[1] = amx::TILE_K;
    cfg.rows[2] = cfg.rows[3] = amx::TILE_K / 4;
    cfg.colsb[2] = cfg.colsb[3] = amx::TILE_N * 4;
    for (int i = 4; i < 8; i++) {
        cfg.rows[i] = amx::TILE_M;
        cfg.colsb[i] = amx::TILE_N * 4;
    }
    return cfg;
}
}

export namespace cpugemm {

void i8_i8_i32_blocked(Int8RowMajor auto A, Int8VNNI auto B, Int32RowMajor auto C,
                       int thread_id = 0, int num_threads = 1)
{
    using namespace amx;
    constexpr size_t M_STEP = 32;
    constexpr size_t N_STEP = 32;
    constexpr size_t K_BLOCK = 8096;
    size_t M = A.extent(0);
    size_t K = A.extent(1);
    size_t N = C.extent(1);
    auto [n_start, n_end] = detail::partition_range(N, thread_id, num_threads);

    auto A_stride = detail::stride_bytes(A);
    auto C_stride = detail::stride_bytes(C);

    TileConfig cfg = detail::make_i8_gemm_config();
    _tile_loadconfig(&cfg);

    for (size_t kb = 0; kb < K; kb += K_BLOCK) {
        size_t k_block_end = std::min(kb + K_BLOCK, K);
        for (size_t m = 0; m < M; m += M_STEP) {
            for (size_t n = n_start; n < n_end; n += N_STEP) {
                if (kb == 0) {
                    _tile_zero(4);
                    _tile_zero(5);
                    _tile_zero(6);
                    _tile_zero(7);
                } else {
                    _tile_loadd(4, &(C[m, n]), C_stride);
                    _tile_loadd(5, &(C[m, n + TILE_N]), C_stride);
                    _tile_loadd(6, &(C[m + TILE_M, n]), C_stride);
                    _tile_loadd(7, &(C[m + TILE_M, n + TILE_N]), C_stride);
                }
                for (size_t k = kb; k < k_block_end; k += TILE_K) {
                    auto a0_ptr = &A[m, k];
                    auto a1_ptr = &A[m + TILE_M, k];
                    auto b0_ptr = &B[k, n];
                    auto b1_ptr = &B[k, n + TILE_N];
                    _tile_loadd(0, a0_ptr, A_stride);
                    _tile_loadd(1, a1_ptr, A_stride);
                    _tile_loadd(2, b0_ptr, TILE_N * 4);
                    _tile_loadd(3, b1_ptr, TILE_N * 4);
                    _tile_dpbssd(4, 0, 2);
                    _tile_dpbssd(5, 0, 3);
                    _tile_dpbssd(6, 1, 2);
                    _tile_dpbssd(7, 1, 3);
                }

                _tile_stored(4, &(C[m, n]), C_stride);
                _tile_stored(5, &(C[m, n + TILE_N]), C_stride);
                _tile_stored(6, &(C[m + TILE_M, n]), C_stride);
                _tile_stored(7, &(C[m + TILE_M, n + TILE_N]), C_stride);
            }
        }
    }
    _tile_release();
}

}
