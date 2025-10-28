module;
#include <cstdint>
#include <chrono>
#include <print>
#include <random>
#include <cstring>
export module quantization_benchmark;
import quantization;

template<typename T>
inline void do_not_optimize(T&& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

inline void clobber_memory() {
    asm volatile("" : : : "memory");
}

export void run_quantization_benchmark() {
    std::println("=== Quantization Performance Benchmark ===\n");

    constexpr int NUM_RUNS = 10000;
    constexpr int TILE_SIZE_BYTES = 16 * 16 * sizeof(int32_t);
    constexpr int TILE_SIZE_INT8 = 16 * 16;

    std::println("Allocating test data ({} tiles, ~{} MB)...", NUM_RUNS,
                 (NUM_RUNS * TILE_SIZE_BYTES) / (1024 * 1024));

    alignas(64) int32_t* input_tiles = new (std::align_val_t{64}) int32_t[NUM_RUNS * 16 * 16];

    alignas(64) int8_t* output_tiles = new (std::align_val_t{64}) int8_t[NUM_RUNS * 16 * 16];

    alignas(64) int32_t* recon_tiles = new (std::align_val_t{64}) int32_t[NUM_RUNS * 16 * 16];

    AMXQ::QuantizationParams* params = new AMXQ::QuantizationParams[NUM_RUNS];

    std::println("Initializing test data...");
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(30'000'000, 60'000'000);

    for (int run = 0; run < NUM_RUNS; run++) {
        int32_t* tile = &input_tiles[run * 16 * 16];

        for (int i = 0; i < 256; i++) {
            tile[i] = dist(rng);
        }

        int32_t (*tile_2d)[16] = reinterpret_cast<int32_t(*)[16]>(tile);
        params[run] = AMXQ::compute_quantization_params(tile_2d);
    }

    std::println("Data initialized. Starting benchmarks...\n");

    std::println("--- Quantization Benchmark (int32 → int8) ---");

    for (int run = 0; run < 100; run++) {
        int32_t (*tile_in)[16] = reinterpret_cast<int32_t(*)[16]>(&input_tiles[run * 16 * 16]);
        int8_t* tile_out = &output_tiles[run * 16 * 16];
        AMXQ::quantize_tile_avx512(tile_in, tile_out, 16, params[run].bias, params[run].scale);
    }
    clobber_memory();

    auto quant_start = std::chrono::high_resolution_clock::now();

    for (int run = 0; run < NUM_RUNS; run++) {
        int32_t (*tile_in)[16] = reinterpret_cast<int32_t(*)[16]>(&input_tiles[run * 16 * 16]);
        int8_t* tile_out = &output_tiles[run * 16 * 16];

        AMXQ::quantize_tile_avx512(tile_in, tile_out, 16, params[run].bias, params[run].scale);
        do_not_optimize(tile_out);
    }
    clobber_memory();

    auto quant_end = std::chrono::high_resolution_clock::now();
    auto quant_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(quant_end - quant_start);

    double quant_ns_per_tile = quant_duration.count() / (double)NUM_RUNS;
    double quant_tiles_per_sec = 1e9 / quant_ns_per_tile;
    double quant_gb_per_sec = (quant_tiles_per_sec * TILE_SIZE_BYTES) / (1024.0 * 1024.0 * 1024.0);

    std::println("  Total time:        {:.2f} ms", quant_duration.count() / 1e6);
    std::println("  Time per tile:     {:.0f} ns", quant_ns_per_tile);
    std::println("  Throughput:        {:.2f} M tiles/sec", quant_tiles_per_sec / 1e6);
    std::println("  Bandwidth:         {:.2f} GB/s (input)", quant_gb_per_sec);
    std::println("  Compression ratio: {:.1f}× (1024B → 256B)", TILE_SIZE_BYTES / (float)TILE_SIZE_INT8);

    std::println("\n--- Dequantization Benchmark (int8 → int32) ---");

    for (int run = 0; run < 100; run++) {
        int8_t* tile_in = &output_tiles[run * 16 * 16];
        int32_t (*tile_out)[16] = reinterpret_cast<int32_t(*)[16]>(&recon_tiles[run * 16 * 16]);
        AMXQ::dequantize_tile_avx512(tile_in, 16, tile_out, params[run].bias, params[run].scale);
    }
    clobber_memory();

    auto dequant_start = std::chrono::high_resolution_clock::now();

    for (int run = 0; run < NUM_RUNS; run++) {
        int8_t* tile_in = &output_tiles[run * 16 * 16];
        int32_t (*tile_out)[16] = reinterpret_cast<int32_t(*)[16]>(&recon_tiles[run * 16 * 16]);

        AMXQ::dequantize_tile_avx512(tile_in, 16, tile_out, params[run].bias, params[run].scale);
        do_not_optimize(tile_out);
    }
    clobber_memory();

    auto dequant_end = std::chrono::high_resolution_clock::now();
    auto dequant_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(dequant_end - dequant_start);

    double dequant_ns_per_tile = dequant_duration.count() / (double)NUM_RUNS;
    double dequant_tiles_per_sec = 1e9 / dequant_ns_per_tile;
    double dequant_gb_per_sec = (dequant_tiles_per_sec * TILE_SIZE_BYTES) / (1024.0 * 1024.0 * 1024.0);

    std::println("  Total time:        {:.2f} ms", dequant_duration.count() / 1e6);
    std::println("  Time per tile:     {:.0f} ns", dequant_ns_per_tile);
    std::println("  Throughput:        {:.2f} M tiles/sec", dequant_tiles_per_sec / 1e6);
    std::println("  Bandwidth:         {:.2f} GB/s (output)", dequant_gb_per_sec);
    std::println("  Expansion ratio:   {:.1f}× (256B → 1024B)", TILE_SIZE_BYTES / (float)TILE_SIZE_INT8);

    std::println("\n--- Accuracy Verification ---");

    int64_t total_error = 0;
    int32_t max_error = 0;

    for (int run = 0; run < NUM_RUNS; run++) {
        int32_t* original = &input_tiles[run * 16 * 16];
        int32_t* reconstructed = &recon_tiles[run * 16 * 16];

        for (int i = 0; i < 256; i++) {
            int32_t error = std::abs(reconstructed[i] - original[i]);
            total_error += error;
            max_error = std::max(max_error, error);
        }
    }

    double avg_error = total_error / (double)(NUM_RUNS * 256);
    double avg_error_pct = 100.0 * avg_error / 45'000'000;
    double max_error_pct = 100.0 * max_error / 60'000'000;

    std::println("  Average error:     {:.0f} ({:.3f}%)", avg_error, avg_error_pct);
    std::println("  Maximum error:     {} ({:.3f}%)", max_error, max_error_pct);

    std::println("\n=== Performance Summary ===");
    std::println("  Quantization:      {:.0f} ns/tile ({:.2f} GB/s)", quant_ns_per_tile, quant_gb_per_sec);
    std::println("  Dequantization:    {:.0f} ns/tile ({:.2f} GB/s)", dequant_ns_per_tile, dequant_gb_per_sec);
    std::println("  Total roundtrip:   {:.0f} ns/tile", quant_ns_per_tile + dequant_ns_per_tile);
    std::println("  Compression:       4× (1024B → 256B + 6B params)");

    delete[] input_tiles;
    delete[] output_tiles;
    delete[] recon_tiles;
    delete[] params;
}
