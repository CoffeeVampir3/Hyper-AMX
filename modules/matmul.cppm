module;
#include <immintrin.h>
#include <cstdint>
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

export template<typename TA, typename TB, typename TC>
    requires IsRowMajor<TA> && IsVNNI<TB> && IsRowMajor<TC> &&
             std::same_as<typename TA::value_type, int8_t> &&
             std::same_as<typename TC::value_type, int32_t>
void matmul_amx_int8_blocked(TA A, const TB& B, TC C, int thread_id = 0, int num_threads = 1)
{
    constexpr size_t TILE_M = 16;
    constexpr size_t TILE_K = 64;
    constexpr size_t TILE_N = 16;
    constexpr size_t M_STEP = 32;
    constexpr size_t N_STEP = 32;
    size_t M = A.rows;
    size_t K = A.cols;
    size_t N = C.cols;
    size_t n_per_thread = (N + num_threads - 1) / num_threads;
    size_t n_start = thread_id * n_per_thread;
    size_t n_end = std::min(N, n_start + n_per_thread);
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
    for (size_t m = 0; m < M; m += M_STEP) {
        for (size_t n = n_start; n < n_end; n += N_STEP) {
            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);
            for (size_t k = 0; k < K; k += TILE_K) {
                auto a0_ptr = A.row(m) + k;
                auto a1_ptr = A.row(m + TILE_M) + k;
                auto b0_ptr = B.get_tile_ptr(n, k);
                auto b1_ptr = B.get_tile_ptr(n + TILE_N, k);
                _tile_loadd(0, a0_ptr, A.stride_bytes());
                _tile_loadd(1, a1_ptr, A.stride_bytes());
                _tile_loadd(2, b0_ptr, TILE_N * 4);
                _tile_loadd(3, b1_ptr, TILE_N * 4);
                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(5, 0, 3);
                _tile_dpbssd(6, 1, 2);
                _tile_dpbssd(7, 1, 3);
            }
            _tile_stored(4, C.row(m) + n, C.stride_bytes());
            _tile_stored(5, C.row(m) + n + TILE_N, C.stride_bytes());
            _tile_stored(6, C.row(m + TILE_M) + n, C.stride_bytes());
            _tile_stored(7, C.row(m + TILE_M) + n + TILE_N, C.stride_bytes());
        }
    }
    _tile_release();
}

export template<typename TA, typename TB, typename TC>
    requires IsRowMajor<TA> && IsVNNI<TB> && IsRowMajor<TC> &&
             std::same_as<typename TA::value_type, int8_t> &&
             std::same_as<typename TC::value_type, int32_t>
void matmul_amx_int8_blocked_mt(TA A, const TB& B, TC C, int num_threads = 0)
{
    constexpr size_t TILE_M = 16;
    constexpr size_t TILE_K = 64;
    constexpr size_t TILE_N = 16;
    constexpr size_t M_STEP = 32;
    constexpr size_t N_STEP = 32;
    size_t M = A.rows;
    size_t K = A.cols;
    size_t N = C.cols;
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    auto worker = [&](int tid) {
        int cpu_id = tid;
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        size_t M_blocks = (M + M_STEP - 1) / M_STEP;
        size_t blocks_per_thread = (M_blocks + num_threads - 1) / num_threads;
        size_t block_start = tid * blocks_per_thread;
        size_t block_end = std::min(M_blocks, block_start + blocks_per_thread);
        if (block_start >= M_blocks) return;
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
        for (size_t mb = block_start; mb < block_end; ++mb) {
            size_t m = mb * M_STEP;
            if (m >= M) break;
            for (size_t n = 0; n < N; n += N_STEP) {
                _tile_zero(4);
                _tile_zero(5);
                _tile_zero(6);
                _tile_zero(7);
                for (size_t k = 0; k < K; k += TILE_K) {
                    auto a0_ptr = A.row(m) + k;
                    auto a1_ptr = A.row(m + TILE_M) + k;
                    auto b0_ptr = B.get_tile_ptr(n, k);
                    auto b1_ptr = B.get_tile_ptr(n + TILE_N, k);
                    _tile_loadd(0, a0_ptr, A.stride_bytes());
                    _tile_loadd(1, a1_ptr, A.stride_bytes());
                    _tile_loadd(2, b0_ptr, TILE_N * 4);
                    _tile_loadd(3, b1_ptr, TILE_N * 4);
                    _tile_dpbssd(4, 0, 2);
                    _tile_dpbssd(5, 0, 3);
                    _tile_dpbssd(6, 1, 2);
                    _tile_dpbssd(7, 1, 3);
                }
                _tile_stored(4, C.row(m) + n, C.stride_bytes());
                _tile_stored(5, C.row(m) + n + TILE_N, C.stride_bytes());
                _tile_stored(6, C.row(m + TILE_M) + n, C.stride_bytes());
                _tile_stored(7, C.row(m + TILE_M) + n + TILE_N, C.stride_bytes());
            }
        }
        _tile_release();
    };
    std::vector<std::jthread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }
}
