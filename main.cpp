#include <print>
import tensor_tests;
import matmul_correctness;
import numa_matmul_bench;
import kernel_tests;
import kernel_benchmarks;

int main() {
    std::println("=== AMX Test Suite ===\n");

    run_qtensor_tests();
    run_modern_matmul_correctness();
    run_kernel_tests();
    run_kernel_benchmarks();
    //run_numa_matmul_bench();
    return 0;
}
