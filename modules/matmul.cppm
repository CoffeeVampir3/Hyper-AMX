module;
#include <immintrin.h>
#include <cstdint>
#include <thread>
#include <vector>
#include <sched.h>
export module matmul;
import tensor;
import quantization;
import tensor_utils;

export namespace amx {
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

template<int tmm_register>
inline void store_and_quantize_tile(
    int32_t tile_temp[16][16],
    auto C, auto params_out,
    size_t m, size_t n, size_t tile_m, size_t tile_n, size_t C_stride)
{
    _tile_stored(tmm_register, tile_temp, amx::TILE_STRIDE_BYTES);
    auto params = compute_quantization_params(tile_temp);
    params_out[tile_m, tile_n] = params;
    quantize_tile_avx512(tile_temp, C.data_handle() + C.mapping()(m, n),
                        C_stride, params.bias, params.scale);
}

template<bool Quantized>
void matmul_amx_int8_blocked_impl(Int8RowMajor auto A, Int8VNNI auto B, auto C,
                                   auto params_out,
                                   int32_t (*tile_temp)[16],
                                   int32_t (*accum_temp)[32],
                                   int thread_id = 0, int num_threads = 1)
{
    using namespace amx;
    constexpr size_t M_STEP = 32;
    constexpr size_t N_STEP = 32;
    constexpr size_t K_BLOCK = 8096;
    size_t M = A.extent(0);
    size_t K = A.extent(1);
    size_t N = C.extent(1);
    size_t n_per_thread = (N + num_threads - 1) / num_threads;
    size_t n_start = thread_id * n_per_thread;
    size_t n_end = std::min(N, n_start + n_per_thread);

    auto A_stride = stride_bytes(A);
    auto C_stride = stride_bytes(C);

    TileConfig cfg{};
    cfg.palette_id = 1;
    cfg.rows[0] = TILE_M;
    cfg.colsb[0] = TILE_K;
    cfg.rows[1] = TILE_M;
    cfg.colsb[1] = TILE_K;
    cfg.rows[2] = TILE_K / 4;
    cfg.colsb[2] = TILE_N * 4;
    cfg.rows[3] = TILE_K / 4;
    cfg.colsb[3] = TILE_N * 4;
    cfg.rows[4] = TILE_M;
    cfg.colsb[4] = TILE_N * 4;
    cfg.rows[5] = TILE_M;
    cfg.colsb[5] = TILE_N * 4;
    cfg.rows[6] = TILE_M;
    cfg.colsb[6] = TILE_N * 4;
    cfg.rows[7] = TILE_M;
    cfg.colsb[7] = TILE_N * 4;
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
                    if constexpr (Quantized) {
                        // If quantized, load to a temporary accumulator.
                        _tile_loadd(4, &accum_temp[0][0], ACCUM_STRIDE_BYTES);
                        _tile_loadd(5, &accum_temp[0][16], ACCUM_STRIDE_BYTES);
                        _tile_loadd(6, &accum_temp[16][0], ACCUM_STRIDE_BYTES);
                        _tile_loadd(7, &accum_temp[16][16], ACCUM_STRIDE_BYTES);
                    } else {
                        //Otherwise load directly from c
                        _tile_loadd(4, C.data_handle() + C.mapping()(m, n), C_stride);
                        _tile_loadd(5, C.data_handle() + C.mapping()(m, n + TILE_N), C_stride);
                        _tile_loadd(6, C.data_handle() + C.mapping()(m + TILE_M, n), C_stride);
                        _tile_loadd(7, C.data_handle() + C.mapping()(m + TILE_M, n + TILE_N), C_stride);
                    }
                }
                for (size_t k = kb; k < k_block_end; k += TILE_K) {
                    auto a0_ptr = A.data_handle() + A.mapping()(m, k);
                    auto a1_ptr = A.data_handle() + A.mapping()(m + TILE_M, k);
                    auto b0_ptr = B.data_handle() + B.mapping()(k, n);
                    auto b1_ptr = B.data_handle() + B.mapping()(k, n + TILE_N);
                    _tile_loadd(0, a0_ptr, A_stride);
                    _tile_loadd(1, a1_ptr, A_stride);
                    _tile_loadd(2, b0_ptr, TILE_N * 4);
                    _tile_loadd(3, b1_ptr, TILE_N * 4);
                    _tile_dpbssd(4, 0, 2);
                    _tile_dpbssd(5, 0, 3);
                    _tile_dpbssd(6, 1, 2);
                    _tile_dpbssd(7, 1, 3);
                }

                if constexpr (Quantized) {
                    // If quantized, write to the accumulator unless it's the last iteration, then write the accumulation to C.
                    if (kb + K_BLOCK < K) {
                        _tile_stored(4, &accum_temp[0][0], ACCUM_STRIDE_BYTES);
                        _tile_stored(5, &accum_temp[0][16], ACCUM_STRIDE_BYTES);
                        _tile_stored(6, &accum_temp[16][0], ACCUM_STRIDE_BYTES);
                        _tile_stored(7, &accum_temp[16][16], ACCUM_STRIDE_BYTES);
                    } else {
                        size_t tile_m = m / TILE_M;
                        size_t tile_n = n / TILE_N;
                        store_and_quantize_tile<4>(tile_temp, C, params_out, m, n, tile_m, tile_n, C_stride);
                        store_and_quantize_tile<5>(tile_temp, C, params_out, m, n + TILE_N, tile_m, tile_n + 1, C_stride);
                        store_and_quantize_tile<6>(tile_temp, C, params_out, m + TILE_M, n, tile_m + 1, tile_n, C_stride);
                        store_and_quantize_tile<7>(tile_temp, C, params_out, m + TILE_M, n + TILE_N, tile_m + 1, tile_n + 1, C_stride);
                    }
                } else {
                    // Directly write to C
                    _tile_stored(4, C.data_handle() + C.mapping()(m, n), C_stride);
                    _tile_stored(5, C.data_handle() + C.mapping()(m, n + TILE_N), C_stride);
                    _tile_stored(6, C.data_handle() + C.mapping()(m + TILE_M, n), C_stride);
                    _tile_stored(7, C.data_handle() + C.mapping()(m + TILE_M, n + TILE_N), C_stride);
                }
            }
        }
    }
    _tile_release();
}

export void matmul_amx_int8_blocked(Int8RowMajor auto A, Int8VNNI auto B, Int32RowMajor auto C,
                                    int thread_id = 0, int num_threads = 1)
{
    matmul_amx_int8_blocked_impl<false>(A, B, C, nullptr, nullptr, nullptr, thread_id, num_threads);
}

export void matmul_amx_int8_blocked_quantized(Int8RowMajor auto A, Int8VNNI auto B, Int8RowMajor auto C,
                                              QuantParamsGrid auto params_out,
                                              int thread_id = 0, int num_threads = 1)
{
    alignas(64) int32_t tile_temp[16][16];
    alignas(64) int32_t accum_temp[32][32];
    matmul_amx_int8_blocked_impl<true>(A, B, C, params_out, tile_temp, accum_temp, thread_id, num_threads);
}
