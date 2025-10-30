#include <print>
import tensor_tests;
import matmul_correctness;
import kernel_tests;
import quantization_test;

import kernel_benchmarks;
import numa_matmul_bench;
import quantization_benchmark;

int main() {
    std::println("=== AMX Test Suite ===\n");

    run_qtensor_tests();
    run_modern_matmul_correctness();
    run_kernel_tests();
    run_quantization_tests();

    //run_quantization_benchmark();
    //run_kernel_benchmarks();
    //run_numa_matmul_bench();
    return 0;
}
