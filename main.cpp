#include <print>
import kernel_tests;
import matmul_correctness;
import numa_tests;
import quantization_test;
import tensor_tests;

import kernel_benchmarks;
import numa_add_bench;
import numa_matmul_bench;
import quantization_benchmark;

int main() {
    run_kernel_tests();
    run_modern_matmul_correctness();
    run_numa_tests();
    run_quantization_tests();
    run_qtensor_tests();

    run_quantization_benchmark();
    run_kernel_benchmarks();
    run_numa_add_bench();
    run_numa_matmul_bench();
    return 0;
}
