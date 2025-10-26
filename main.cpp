#include <print>
import quantization_test;
import quantization_benchmark;
import matmul_correctness;
import numa_matmul_test;

int main() {
    std::println("=== AMX Test Suite ===\n");

    run_matmul_correctness();
    std::println("");
    run_quantization_tests();
    std::println("");
    run_quantization_benchmark();
    std::println("");
    //run_numa_matmul_test();
    std::println("");
    return 0;
}
