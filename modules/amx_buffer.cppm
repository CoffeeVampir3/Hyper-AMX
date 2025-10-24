module;
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <immintrin.h>
#include <bit>
export module amx_buffer;
import tensor;

export template<size_t N_BLOCK_ = 256, size_t K_BLOCK_ = 4096>
struct AMXWeightBuffer {
    static constexpr size_t N_BLOCK = N_BLOCK_;
    static constexpr size_t K_BLOCK = K_BLOCK_;
    static constexpr size_t TILE_N = 16;
    static constexpr size_t TILE_K = 64;
    static constexpr size_t VNNI_BLK = 4;
    static_assert((N_BLOCK & (N_BLOCK - 1)) == 0, "N_BLOCK must be power of 2");
    static_assert((K_BLOCK & (K_BLOCK - 1)) == 0, "K_BLOCK must be power of 2");
    static_assert((TILE_N & (TILE_N - 1)) == 0, "TILE_N must be power of 2");
    static_assert((VNNI_BLK & (VNNI_BLK - 1)) == 0, "VNNI_BLK must be power of 2");
    static constexpr int N_BLOCK_SHIFT = std::countr_zero(N_BLOCK);
    static constexpr int K_BLOCK_SHIFT = std::countr_zero(K_BLOCK);
    static constexpr int TILE_N_SHIFT = std::countr_zero(TILE_N);
    static constexpr int VNNI_SHIFT = std::countr_zero(VNNI_BLK);
    int8_t* data{nullptr};
    size_t K{0};
    size_t N{0};
    std::unique_ptr<int8_t[], decltype(&std::free)> owned_data{nullptr, &std::free};
    AMXWeightBuffer(size_t k, size_t n) : K(k), N(n) {
        size_t total_size = k * n;
        void* ptr = std::aligned_alloc(64, total_size);
        owned_data.reset(static_cast<int8_t*>(ptr));
        data = owned_data.get();
    }
    template<IsRowMajor Tsor>
        requires std::same_as<typename Tsor::value_type, int8_t>
    static AMXWeightBuffer<N_BLOCK_, K_BLOCK_> from_rowmajor(Tsor src) {
        size_t K = src.rows;
        size_t N = src.cols;
        AMXWeightBuffer<N_BLOCK_, K_BLOCK_> buf(K, N);
        for (size_t n_block_start = 0; n_block_start < N; n_block_start += N_BLOCK) {
            size_t n_block_size = std::min(N_BLOCK, N - n_block_start);
            for (size_t k_block_start = 0; k_block_start < K; k_block_start += K_BLOCK) {
                size_t k_block_size = std::min(K_BLOCK, K - k_block_start);
                for (size_t n_tile = 0; n_tile < n_block_size; n_tile += TILE_N) {
                    size_t n_tile_size = std::min(TILE_N, n_block_size - n_tile);
                    for (size_t k_tile = 0; k_tile < k_block_size; k_tile += TILE_K) {
                        size_t k_tile_size = std::min(TILE_K, k_block_size - k_tile);
                        for (size_t k = 0; k < k_tile_size; k += VNNI_BLK) {
                            size_t global_k = k_block_start + k_tile + k;
                            int8_t* dst_base = buf.get_submat_ptr(n_block_start + n_tile, global_k);
                            for (size_t n = 0; n < n_tile_size; ++n) {
                                size_t global_n = n_block_start + n_tile + n;
                                dst_base[n * VNNI_BLK + 0] = src(global_k + 0, global_n);
                                dst_base[n * VNNI_BLK + 1] = src(global_k + 1, global_n);
                                dst_base[n * VNNI_BLK + 2] = src(global_k + 2, global_n);
                                dst_base[n * VNNI_BLK + 3] = src(global_k + 3, global_n);
                            }
                        }
                    }
                }
            }
        }
        return buf;
    }
    int8_t* get_submat_ptr(size_t n_begin, size_t k_begin) const {
        constexpr size_t TILE_N_BYTES = TILE_N << VNNI_SHIFT;
        size_t n_block_idx = n_begin >> N_BLOCK_SHIFT;
        size_t n_tile_idx = (n_begin >> TILE_N_SHIFT) & ((N_BLOCK >> TILE_N_SHIFT) - 1);
        size_t n_in_tile = n_begin & (TILE_N - 1);
        size_t k_block_idx = k_begin >> K_BLOCK_SHIFT;
        size_t k_vnni_idx = (k_begin >> VNNI_SHIFT) & ((K_BLOCK >> VNNI_SHIFT) - 1);
        size_t n_block_size = std::min(N_BLOCK, N - (n_block_idx << N_BLOCK_SHIFT));
        size_t k_block_size = std::min(K_BLOCK, K - (k_block_idx << K_BLOCK_SHIFT));
        size_t offset = n_block_idx * K
                      + k_block_idx * (n_block_size * k_block_size)
                      + n_tile_idx * (TILE_N * k_block_size)
                      + k_vnni_idx * TILE_N_BYTES
                      + (n_in_tile << VNNI_SHIFT);
        return data + offset;
    }
};
