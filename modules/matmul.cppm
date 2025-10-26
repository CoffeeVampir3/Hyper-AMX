module;
#include <cstdint>
#include <immintrin.h>
#include <thread>
#include <vector>
#include <sched.h>
export module matmul;
import tensor;

struct TileConfig {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
};

export template<MdspanLike TA, MdspanLike TB, MdspanLike TC>
    requires (!IsVNNI<typename TA::layout_type>) &&
             IsVNNI<typename TB::layout_type> &&
             (!IsVNNI<typename TC::layout_type>) &&
             std::same_as<typename TA::element_type, int8_t> &&
             std::same_as<typename TC::element_type, int32_t>
void matmul_amx_int8_blocked(TA A, TB B, TC C, int thread_id = 0, int num_threads = 1)
{
    constexpr size_t TILE_M = 16;
    constexpr size_t TILE_K = 64;
    constexpr size_t TILE_N = 16;
    constexpr size_t M_STEP = 32;
    constexpr size_t N_STEP = 32;
    constexpr size_t K_BLOCK = 3584;
    size_t M = A.extent(0);
    size_t K = A.extent(1);
    size_t N = C.extent(1);
    size_t n_per_thread = (N + num_threads - 1) / num_threads;
    size_t n_start = thread_id * n_per_thread;
    size_t n_end = std::min(N, n_start + n_per_thread);

    auto A_stride = stride_bytes(A, sizeof(typename TA::element_type));
    auto C_stride = stride_bytes(C, sizeof(typename TC::element_type));

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
                    _tile_loadd(4, C.data_handle() + C.mapping()(m, n), C_stride);
                    _tile_loadd(5, C.data_handle() + C.mapping()(m, n + TILE_N), C_stride);
                    _tile_loadd(6, C.data_handle() + C.mapping()(m + TILE_M, n), C_stride);
                    _tile_loadd(7, C.data_handle() + C.mapping()(m + TILE_M, n + TILE_N), C_stride);
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
                _tile_stored(4, C.data_handle() + C.mapping()(m, n), C_stride);
                _tile_stored(5, C.data_handle() + C.mapping()(m, n + TILE_N), C_stride);
                _tile_stored(6, C.data_handle() + C.mapping()(m + TILE_M, n), C_stride);
                _tile_stored(7, C.data_handle() + C.mapping()(m + TILE_M, n + TILE_N), C_stride);
            }
        }
    }
    _tile_release();
}

export template<MdspanLike TA, MdspanLike TB, MdspanLike TC>
    requires (!IsVNNI<typename TA::layout_type>) &&
             IsVNNI<typename TB::layout_type> &&
             (!IsVNNI<typename TC::layout_type>) &&
             std::same_as<typename TA::element_type, int8_t> &&
             std::same_as<typename TC::element_type, int32_t>
void matmul_amx_int8_blocked_mt(TA A, TB B, TC C, int num_threads = 0)
{
    constexpr size_t N_STEP = 32;
    size_t N = C.extent(1);
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    num_threads = std::min(num_threads, static_cast<int>(N / N_STEP));
    if (num_threads <= 1) {
        matmul_amx_int8_blocked(A, B, C, 0, 1);
        return;
    }
    auto worker = [&](int tid) {
        int cpu_id = tid;
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        matmul_amx_int8_blocked(A, B, C, tid, num_threads);
    };
    std::vector<std::jthread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }
}
