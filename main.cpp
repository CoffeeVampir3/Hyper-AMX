#include <print>
import tensor_tests;
import matmul_correctness;
import numa_matmul_bench;

int main() {
    std::println("=== AMX Test Suite ===\n");

    run_qtensor_tests();
    run_modern_matmul_correctness();
    run_numa_matmul_bench();
    return 0;
}
