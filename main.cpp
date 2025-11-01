#include <print>
import hyperamx.tensor_tests;
import hyperamx.numa_tests;

int main() {
    run_tensor_tests();
    run_numa_tests();
    return 0;
}
