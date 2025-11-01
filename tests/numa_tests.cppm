module;
#include <cerrno>
#include <cstdlib>
#include <mdspan>
#include <numa.h>
#include <numaif.h>
#include <print>
#include <system_error>
export module hyperamx.numa_tests;

import hyperamx.tensor;
import hyperamx.layout;
import hyperamx.tensor_utils;
import hyperamx.numa;

using namespace tensor;
using namespace tensor_utils;
using namespace numa;

namespace {

using Extents2D = std::dextents<size_t, 2>;

template<typename T>
using RowTensor = Tensor<T, Extents2D, Layout::RowMajor, NumaAllocator<T>>;

template<typename Fn>
void run_case(const char* label, Fn&& fn) {
    std::println("=== {} ===", label);
    fn();
    std::println("   ✓ pass\n");
}

[[noreturn]] void fail(const char* msg) {
    std::println("   ✗ {}", msg);
    std::exit(1);
}

int node_for_address(const void* ptr) {
    void* addr = const_cast<void*>(ptr);
    int status = -1;
    if (numa_move_pages(0, 1, &addr, nullptr, &status, 0) != 0) {
        throw std::system_error(errno, std::generic_category(), "numa_move_pages failed");
    }
    return status;
}

void test_replicated_tensor_basic(const DualSocketConfig& config) {
    constexpr size_t M = 32;
    constexpr size_t N = 64;
    RowTensor<int32_t> source = make_tensor<int32_t, Extents2D, Layout::RowMajor>(
        Extents2D{M, N}, NumaAllocator<int32_t>{DualSocketConfig::primary_node_for_socket(0)});
    tensor_utils::fill(source, [](size_t i, size_t j) { return static_cast<int32_t>(i * 100 + j); });

    Replicated<RowTensor<int32_t>> replicas(source, config);
    for (int socket = 0; socket < DualSocketConfig::NUM_SOCKETS; socket++) {
        if (!tensor_utils::equal(replicas[socket], source)) {
            fail("Replicated tensor data mismatch");
        }
        int node = node_for_address(replicas[socket].span().data_handle());
        int expected_node = DualSocketConfig::primary_node_for_socket(socket);
        if (node != expected_node) {
            fail("Replicated tensor allocated on unexpected NUMA node");
        }
    }
}

void test_column_partition_basic(const DualSocketConfig& config) {
    constexpr size_t M = 16;
    constexpr size_t N = 64;
    RowTensor<int32_t> full = make_tensor<int32_t, Extents2D, Layout::RowMajor>(
        Extents2D{M, N}, NumaAllocator<int32_t>{DualSocketConfig::primary_node_for_socket(0)});
    tensor_utils::fill(full, [](size_t i, size_t j) { return static_cast<int32_t>(i * 10 + j); });

    ColumnPartitioned<RowTensor<int32_t>> partitions(full, DualSocketConfig::NUM_SOCKETS, config);
    size_t expected_cols = N / DualSocketConfig::NUM_SOCKETS;

    for (int socket = 0; socket < partitions.num_partitions; socket++) {
        auto view = partitions.span(socket);
        if (view.extent(0) != M || view.extent(1) != expected_cols) {
            fail("Partition extents incorrect");
        }
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < expected_cols; j++) {
                int32_t expected = static_cast<int32_t>(i * 10 + (socket * expected_cols + j));
                if (view[i, j] != expected) {
                    fail("Partition data mismatch");
                }
            }
        }
        int node = node_for_address(view.data_handle());
        int expected_node = DualSocketConfig::primary_node_for_socket(socket);
        if (node != expected_node) {
            fail("Partition data allocated on unexpected NUMA node");
        }
    }
}

void test_all_reduce_sum_basic(const DualSocketConfig& config) {
    constexpr size_t M = 8;
    constexpr size_t N = 16;
    Extents2D extents{M, N};

    Replicated<RowTensor<int32_t>> partials(extents, config);
    for (int socket = 0; socket < DualSocketConfig::NUM_SOCKETS; socket++) {
        tensor_utils::fill(partials[socket], [socket](size_t i, size_t j) {
            return static_cast<int32_t>(i * 100 + j + socket);
        });
    }

    RowTensor<int32_t> result = make_tensor<int32_t, Extents2D, Layout::RowMajor>(
        extents, NumaAllocator<int32_t>{DualSocketConfig::primary_node_for_socket(0)});
    tensor_utils::fill<0>(result);

    all_reduce_sum(partials, result, 0, config);

    auto res_view = result.span();
    int sockets = DualSocketConfig::NUM_SOCKETS;
    int socket_sum = sockets * (sockets - 1) / 2;
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            int base = static_cast<int>(i * 100 + j);
            int expected = sockets * base + socket_sum;
            if (res_view[i, j] != expected) {
                fail("all_reduce_sum produced incorrect value");
            }
        }
    }
}

void test_all_gather_basic(const DualSocketConfig& config) {
    constexpr size_t M = 12;
    constexpr size_t N = 48;
    RowTensor<int32_t> full = make_tensor<int32_t, Extents2D, Layout::RowMajor>(
        Extents2D{M, N}, NumaAllocator<int32_t>{DualSocketConfig::primary_node_for_socket(0)});
    tensor_utils::fill(full, [](size_t i, size_t j) { return static_cast<int32_t>(i * 50 + j); });

    ColumnPartitioned<RowTensor<int32_t>> partitions(full, DualSocketConfig::NUM_SOCKETS, config);
    RowTensor<int32_t> gathered = make_tensor<int32_t, Extents2D, Layout::RowMajor>(
        Extents2D{M, N}, NumaAllocator<int32_t>{DualSocketConfig::primary_node_for_socket(0)});
    tensor_utils::fill<0>(gathered);

    all_gather(partitions, gathered);
    if (!tensor_utils::equal(gathered, full)) {
        fail("all_gather did not reconstruct the original tensor");
    }
}

} // namespace

export void run_numa_tests() {
    auto config = DualSocketConfig::discover();
    run_case("Replicated tensor basic", [&] { test_replicated_tensor_basic(config); });
    run_case("Column partition basic", [&] { test_column_partition_basic(config); });
    run_case("all_reduce_sum basic", [&] { test_all_reduce_sum_basic(config); });
    run_case("all_gather basic", [&] { test_all_gather_basic(config); });
}
