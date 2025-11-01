module;
#include <print>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <mdspan>
export module moe_tests;
import tensor;
import layout;
import numa;
import moe;
import avx512;

using namespace MoE;
using Extents2D = std::dextents<size_t, 2>;

void reference_group_based_routing(
    const Tensor2D<float>& router_logits,
    const Tensor2D<float>& correction_bias,
    RoutingResult& result,
    int n_group,
    int topk_group,
    int top_k,
    float routed_scaling_factor)
{
    size_t batch_size = router_logits.extent(0);
    size_t n_experts = router_logits.extent(1);
    size_t experts_per_group = n_experts / n_group;

    auto logits_view = router_logits.view();
    auto bias_view = correction_bias.view();

    for (size_t b = 0; b < batch_size; b++) {
        std::vector<float> sigmoid_probs(n_experts);
        for (size_t e = 0; e < n_experts; e++) {
            sigmoid_probs[e] = 1.0f / (1.0f + std::exp(-logits_view[b, e]));
        }

        std::vector<float> group_scores(n_group);
        for (int g = 0; g < n_group; g++) {
            std::vector<float> group_vals;
            for (size_t e = 0; e < experts_per_group; e++) {
                group_vals.push_back(sigmoid_probs[g * experts_per_group + e] + bias_view[0, g * experts_per_group + e]);
            }
            std::sort(group_vals.begin(), group_vals.end(), std::greater<float>());
            group_scores[g] = group_vals[0] + group_vals[1];
        }

        std::vector<std::pair<float, int>> group_pairs;
        for (int g = 0; g < n_group; g++) {
            group_pairs.push_back({group_scores[g], g});
        }
        std::partial_sort(group_pairs.begin(), group_pairs.begin() + topk_group, group_pairs.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        std::vector<bool> expert_mask(n_experts, false);
        for (int i = 0; i < topk_group; i++) {
            int g = group_pairs[i].second;
            for (size_t e = 0; e < experts_per_group; e++) {
                expert_mask[g * experts_per_group + e] = true;
            }
        }

        std::vector<std::pair<float, int>> expert_pairs;
        for (size_t e = 0; e < n_experts; e++) {
            if (expert_mask[e]) {
                expert_pairs.push_back({sigmoid_probs[e] + bias_view[0, e], static_cast<int>(e)});
            }
        }
        std::partial_sort(expert_pairs.begin(), expert_pairs.begin() + top_k, expert_pairs.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        float sum_weights = 0.0f;
        for (int k = 0; k < top_k; k++) {
            sum_weights += sigmoid_probs[expert_pairs[k].second];
        }

        for (int k = 0; k < top_k; k++) {
            result.indices.view()[b, k] = expert_pairs[k].second;
            result.weights.view()[b, k] = (sigmoid_probs[expert_pairs[k].second] / sum_weights) * routed_scaling_factor;
        }
    }
}

void test_group_based_routing_basic() {
    std::println("Basic group-based routing test");
    constexpr size_t n_experts = 64;
    constexpr int n_group = 8;
    constexpr int topk_group = 3;
    constexpr int top_k = 8;
    constexpr float routed_scaling_factor = 1.0f;

    Tensor2D<float> router_logits(Extents2D{1, n_experts});
    Tensor2D<float> correction_bias(Extents2D{1, n_experts});

    float val = 0.0f;
    avx512::fill(router_logits, [&](auto...) { return (val += 0.1f) - 3.2f; });
    val = 0.0f;
    avx512::fill(correction_bias, [&](auto...) { return (val += 0.01f) - 0.32f; });

    RoutingResult result(top_k);
    RoutingResult result_ref(top_k);

    group_based_routing(router_logits, correction_bias, result, n_group, topk_group, top_k, routed_scaling_factor);
    reference_group_based_routing(router_logits, correction_bias, result_ref, n_group, topk_group, top_k, routed_scaling_factor);

    bool indices_match = true;
    for (int k = 0; k < top_k; k++) {
        if (result.indices.view()[0, k] != result_ref.indices.view()[0, k]) {
            indices_match = false;
            std::println("  Mismatch at k {}: got {}, expected {}",
                         k, result.indices.view()[0, k], result_ref.indices.view()[0, k]);
        }
    }

    if (!indices_match) {
        std::println("  ✗ Routing indices mismatch");
        std::exit(1);
    }

    if (!avx512::check_approximate_equal(result.weights.view(), result_ref.weights.view(), 1e-5f, "Routing weights")) {
        std::exit(1);
    }

    std::println("   ✓ Basic group-based routing correctness\n");
}

void test_moe_column_parallel_basic() {
    std::println("Basic MoE column-parallel test");
    constexpr size_t hidden_dim = 4096;
    constexpr size_t hidden_dim_local = hidden_dim / 2;
    constexpr size_t intermediate_size = 4096;
    constexpr size_t intermediate_size_local = intermediate_size / 2;
    constexpr size_t n_routed_experts = 8;
    constexpr size_t n_shared_experts = 2;
    constexpr int n_group = 4;
    constexpr int topk_group = 2;
    constexpr int num_experts_per_tok = 2;

    auto config = Numa::DualSocketConfig::discover();

    Tensor2D<int8_t> hidden_full(Extents2D{1, hidden_dim});
    int8_t val = 1;
    avx512::fill(hidden_full, [&](auto...) { return static_cast<int8_t>((val++ % 20) - 10); });

    CPart<int8_t> hidden_cp(hidden_full, 2, config);

    Tensor2D<float> router_weight(Extents2D{n_routed_experts, hidden_dim});
    float fval = 0.01f;
    avx512::fill(router_weight, [&](auto...) { return (fval += 0.001f) - 0.5f; });

    Tensor2D<float> correction_bias(Extents2D{1, n_routed_experts});
    fval = 0.0f;
    avx512::fill(correction_bias, [&](auto...) { return (fval += 0.01f) - 0.04f; });

    std::vector<Tensor2DVNNI<int8_t>> expert_gate_full;
    std::vector<Tensor2DVNNI<int8_t>> expert_up_full;
    std::vector<Tensor2DVNNI<int8_t>> expert_down_full;
    std::vector<CPartVNNI<int8_t>> expert_gate_weights;
    std::vector<CPartVNNI<int8_t>> expert_up_weights;
    std::vector<RPartVNNI<int8_t>> expert_down_weights;

    expert_gate_full.reserve(n_routed_experts);
    expert_up_full.reserve(n_routed_experts);
    expert_down_full.reserve(n_routed_experts);
    expert_gate_weights.reserve(n_routed_experts);
    expert_up_weights.reserve(n_routed_experts);
    expert_down_weights.reserve(n_routed_experts);

    for (size_t i = 0; i < n_routed_experts; i++) {
        expert_gate_full.emplace_back(Extents2D{intermediate_size, hidden_dim});
        expert_up_full.emplace_back(Extents2D{intermediate_size, hidden_dim});
        expert_down_full.emplace_back(Extents2D{hidden_dim, intermediate_size});

        val = static_cast<int8_t>(i + 1);
        avx512::fill(expert_gate_full.back(), [&](auto...) { return static_cast<int8_t>((val++ % 10) - 5); });
        val = static_cast<int8_t>(i + 10);
        avx512::fill(expert_up_full.back(), [&](auto...) { return static_cast<int8_t>((val++ % 10) - 5); });
        val = static_cast<int8_t>(i + 20);
        avx512::fill(expert_down_full.back(), [&](auto...) { return static_cast<int8_t>((val++ % 10) - 5); });

        expert_gate_weights.emplace_back(expert_gate_full.back(), 2, config);
        expert_up_weights.emplace_back(expert_up_full.back(), 2, config);
        expert_down_weights.emplace_back(expert_down_full.back(), 2, config);
    }

    Tensor2DVNNI<int8_t> shared_gate_full(Extents2D{intermediate_size * n_shared_experts, hidden_dim});
    Tensor2DVNNI<int8_t> shared_up_full(Extents2D{intermediate_size * n_shared_experts, hidden_dim});
    Tensor2DVNNI<int8_t> shared_down_full(Extents2D{hidden_dim, intermediate_size * n_shared_experts});

    val = 100;
    avx512::fill(shared_gate_full, [&](auto...) { return static_cast<int8_t>((val++ % 10) - 5); });
    val = 110;
    avx512::fill(shared_up_full, [&](auto...) { return static_cast<int8_t>((val++ % 10) - 5); });
    val = 120;
    avx512::fill(shared_down_full, [&](auto...) { return static_cast<int8_t>((val++ % 10) - 5); });

    CPartVNNI<int8_t> shared_gate_weight(shared_gate_full, 2, config);
    CPartVNNI<int8_t> shared_up_weight(shared_up_full, 2, config);
    RPartVNNI<int8_t> shared_down_weight(shared_down_full, 2, config);

    CPart<int32_t> output_cp(Extents2D{1, hidden_dim}, 2, config);

    MoEScratchBuffers<int8_t> scratch(
        hidden_dim_local,
        n_routed_experts,
        num_experts_per_tok,
        config
    );

    deepseek_moe_column_parallel(
        hidden_cp,
        router_weight,
        correction_bias,
        expert_gate_weights,
        expert_up_weights,
        expert_down_weights,
        shared_gate_weight,
        shared_up_weight,
        shared_down_weight,
        output_cp,
        scratch,
        config,
        intermediate_size,
        intermediate_size * n_shared_experts,
        n_group,
        topk_group,
        num_experts_per_tok,
        1.0f
    );

    Tensor2D<int32_t> result_gathered(Extents2D{1, hidden_dim});
    Numa::all_gather(output_cp, result_gathered);

    bool has_nonzero = false;
    for (size_t h = 0; h < hidden_dim; h++) {
        if (result_gathered.view()[0, h] != 0) {
            has_nonzero = true;
            break;
        }
    }

    if (!has_nonzero) {
        std::println("  ✗ MoE output is all zeros - likely a bug");
        std::exit(1);
    }

    std::println("   ✓ Basic MoE column-parallel runs without errors");
    std::println("   ✓ Output contains non-zero values\n");
}

export void run_moe_tests() {
    try {
        test_group_based_routing_basic();
        test_moe_column_parallel_basic();
    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        std::exit(1);
    }
}
