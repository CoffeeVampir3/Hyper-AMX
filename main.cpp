#include <print>
import qtensor;
import modern_matmul_correctness;

int main() {
    std::println("=== AMX Test Suite ===\n");

    run_qtensor_tests();
    run_modern_matmul_correctness();
    return 0;
}
