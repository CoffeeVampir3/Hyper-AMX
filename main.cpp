#include <print>
import quantization_test;
import quantization_benchmark;
import matmul_correctness;
import numa_matmul_test;
import qtensor;

int main() {
    std::println("=== AMX Test Suite ===\n");

    run_qtensor_tests();
    return 0;
}
