module;
#include <cstddef>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>
#include <mdspan>
#include <print>
export module moe;
import tensor;
import layout;
import numa;
import amx_gemms;
import avx512;

export namespace MoE {

using Extents2D = std::dextents<size_t, 2>;
using VNNILayout = Layout::VNNI<16, 64, 16, 64, 4>;

template<typename T>
using CPart = Numa::ColumnPartitioned<T, Extents2D, Layout::RowMajor>;

template<typename T>
using CPartVNNI = Numa::ColumnPartitioned<T, Extents2D, VNNILayout>;

template<typename T>
using RPart = Numa::RowPartitioned<T, Extents2D, Layout::RowMajor>;

template<typename T>
using RPartVNNI = Numa::RowPartitioned<T, Extents2D, VNNILayout>;

template<typename T>
using Repl = Numa::Replicated<T, Extents2D, Layout::RowMajor>;

template<typename T>
using Tensor2D = Tensor<T, Extents2D, Layout::RowMajor>;

template<typename T>
using Tensor2DVNNI = Tensor<T, Extents2D, VNNILayout>;

struct RoutingResult {
    Tensor2D<int32_t> indices;
    Tensor2D<float> weights;

    RoutingResult(size_t top_k)
        : indices(Extents2D{1, top_k})
        , weights(Extents2D{1, top_k}) {}
};

inline void topk_indices(const float* scores, size_t n, int32_t* indices, float* values, size_t k) {
    std::vector<std::pair<float, int32_t>> pairs(n);
    for (size_t i = 0; i < n; i++) {
        pairs[i] = {scores[i], static_cast<int32_t>(i)};
    }
    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < k; i++) {
        values[i] = pairs[i].first;
        indices[i] = pairs[i].second;
    }
}

void group_based_routing(
    const Tensor2D<float>& router_logits,
    const Tensor2D<float>& correction_bias,
    RoutingResult& result,
    int n_group = 8,
    int topk_group = 3,
    int top_k = 8,
    float routed_scaling_factor = 1.0f)
{
    size_t n_experts = router_logits.extent(1);
    size_t experts_per_group = n_experts / n_group;

    Tensor2D<float> logits_corrected(Extents2D{1, n_experts});

    auto logits_view = router_logits.view();
    auto corrected_view = logits_corrected.view();
    auto bias_view = correction_bias.view();

    for (size_t e = 0; e < n_experts; e++) {
        float logit = logits_view[0, e];
        float prob = 1.0f / (1.0f + std::exp(-logit));
        corrected_view[0, e] = prob + bias_view[0, e];
    }

    std::vector<float> group_scores(n_group);
    std::vector<int32_t> group_indices(topk_group);
    std::vector<float> group_values(topk_group);
    std::vector<bool> expert_mask(n_experts);
    std::vector<float> masked_scores(n_experts);
    std::vector<int32_t> final_indices(top_k);
    std::vector<float> final_values(top_k);

    for (int g = 0; g < n_group; g++) {
        float top2_sum = 0.0f;
        float max1 = -INFINITY, max2 = -INFINITY;
        for (size_t e = 0; e < experts_per_group; e++) {
            float val = corrected_view[0, g * experts_per_group + e];
            if (val > max1) {
                max2 = max1;
                max1 = val;
            } else if (val > max2) {
                max2 = val;
            }
        }
        group_scores[g] = max1 + max2;
    }

    topk_indices(group_scores.data(), n_group, group_indices.data(), group_values.data(), topk_group);

    std::fill(expert_mask.begin(), expert_mask.end(), false);
    for (int i = 0; i < topk_group; i++) {
        int g = group_indices[i];
        for (size_t e = 0; e < experts_per_group; e++) {
            expert_mask[g * experts_per_group + e] = true;
        }
    }

    for (size_t e = 0; e < n_experts; e++) {
        masked_scores[e] = expert_mask[e] ? corrected_view[0, e] : 0.0f;
    }

    topk_indices(masked_scores.data(), n_experts, final_indices.data(), final_values.data(), top_k);

    float sum_weights = 0.0f;
    for (int k = 0; k < top_k; k++) {
        float original_prob = 1.0f / (1.0f + std::exp(-logits_view[0, final_indices[k]]));
        final_values[k] = original_prob;
        sum_weights += original_prob;
    }

    for (int k = 0; k < top_k; k++) {
        result.indices.view()[0, k] = final_indices[k];
        result.weights.view()[0, k] = (final_values[k] / sum_weights) * routed_scaling_factor;
    }
}

template<TensorStorage TInput, TensorStorage TOutput>
void compute_swiglu_mlp_batch(
    TInput& hidden,
    CPartVNNI<int8_t>& gate_weight,
    CPartVNNI<int8_t>& up_weight,
    RPartVNNI<int8_t>& down_weight,
    int socket,
    size_t intermediate_size,
    TOutput& output)
{
    size_t n_tokens = hidden.extent(0);
    size_t hidden_dim_local = hidden.extent(1);

    Tensor2D<int32_t> gate_out(Extents2D{n_tokens, intermediate_size});
    Tensor2D<int32_t> up_out(Extents2D{n_tokens, intermediate_size});
    Tensor2D<int8_t> intermediate(Extents2D{n_tokens, intermediate_size});

    cpugemm::i8_i8_i32_vector_by_matrix_blocked(hidden.view(), gate_weight.view(socket), gate_out.view(), 0, 1);
    cpugemm::i8_i8_i32_vector_by_matrix_blocked(hidden.view(), up_weight.view(socket), up_out.view(), 0, 1);
    kernel::silu_mul_requantize(gate_out.view(), up_out.view(), intermediate.view(), 0, 1);
    cpugemm::i8_i8_i32_vector_by_matrix_blocked(intermediate.view(), down_weight.view(socket), output.view(), 0, 1);
}

template<typename T>
struct MoEScratchBuffers {
    Repl<float> router_logits_partials;
    Tensor2D<float> router_logits;
    RoutingResult routing;
    std::vector<std::vector<std::pair<size_t, size_t>>> expert_tokens;

    MoEScratchBuffers(size_t hidden_dim_local, size_t n_experts, size_t num_experts_per_tok, const Numa::DualSocketConfig& config)
        : router_logits_partials(Extents2D{1, n_experts}, config)
        , router_logits(Extents2D{1, n_experts})
        , routing(num_experts_per_tok)
        , expert_tokens(n_experts) {}
};

template<typename TInput, typename TOutput>
void deepseek_moe_column_parallel(
    CPart<TInput>& hidden_cp,
    Tensor2D<float>& router_weight,
    Tensor2D<float>& correction_bias,
    std::vector<CPartVNNI<int8_t>>& expert_gate_weights,
    std::vector<CPartVNNI<int8_t>>& expert_up_weights,
    std::vector<RPartVNNI<int8_t>>& expert_down_weights,
    CPartVNNI<int8_t>& shared_gate_weight,
    CPartVNNI<int8_t>& shared_up_weight,
    RPartVNNI<int8_t>& shared_down_weight,
    CPart<TOutput>& output_cp,
    MoEScratchBuffers<TInput>& scratch,
    Numa::DualSocketConfig& config,
    size_t expert_intermediate_size,
    size_t shared_intermediate_size,
    int n_group = 8,
    int topk_group = 3,
    int num_experts_per_tok = 8,
    float routed_scaling_factor = 1.0f)
{
    constexpr int NUM_SOCKETS = Numa::DualSocketConfig::NUM_SOCKETS;
    size_t hidden_dim_local = hidden_cp[0].extent(1);
    size_t n_experts = router_weight.extent(0);

    std::vector<std::jthread> threads;
    threads.reserve(NUM_SOCKETS);

    for (int socket = 0; socket < NUM_SOCKETS; socket++) {
        threads.emplace_back([&, socket] {
            Numa::pin_to_socket(socket, 0);
            auto hidden_view = hidden_cp.view(socket);
            auto logits_view = scratch.router_logits_partials.view(socket);

            for (size_t e = 0; e < n_experts; e++) {
                float sum = 0.0f;
                for (size_t h = 0; h < hidden_dim_local; h++) {
                    size_t global_h = socket * hidden_dim_local + h;
                    sum += static_cast<float>(hidden_view[0, h]) * router_weight.view()[e, global_h];
                }
                logits_view[0, e] = sum;
            }
        });
    }
    threads.clear();

    Numa::all_reduce_sum(scratch.router_logits_partials, scratch.router_logits, 0, config);

    group_based_routing(scratch.router_logits, correction_bias, scratch.routing, n_group, topk_group, num_experts_per_tok, routed_scaling_factor);

    for (int socket = 0; socket < NUM_SOCKETS; socket++) {
        auto output_view = output_cp.view(socket);
        std::memset(output_view.data_handle(), 0, output_view.size() * sizeof(TOutput));
    }

    for (auto& expert_list : scratch.expert_tokens) {
        expert_list.clear();
    }
    for (size_t k = 0; k < num_experts_per_tok; k++) {
        int32_t expert_id = scratch.routing.indices.view()[0, k];
        scratch.expert_tokens[expert_id].push_back({0, k});
    }

    threads.reserve(NUM_SOCKETS * config.physical_cores_per_socket);
    for (int socket = 0; socket < NUM_SOCKETS; socket++) {
        threads.emplace_back([&, socket] {
            Numa::pin_to_socket(socket, 0);
            auto hidden_view = hidden_cp.view(socket);
            auto output_view = output_cp.view(socket);

            Tensor2D<TInput> expert_hidden(Extents2D{1, hidden_dim_local});
            Tensor2D<int32_t> expert_output(Extents2D{1, hidden_dim_local});

            for (size_t expert_id = 0; expert_id < n_experts; expert_id++) {
                const auto& token_list = scratch.expert_tokens[expert_id];
                if (token_list.empty()) continue;

                size_t k_idx = token_list[0].second;
                auto expert_hidden_view = expert_hidden.view();

                std::memcpy(&expert_hidden_view[0, 0], &hidden_view[0, 0], hidden_dim_local * sizeof(TInput));

                compute_swiglu_mlp_batch(expert_hidden, expert_gate_weights[expert_id], expert_up_weights[expert_id], expert_down_weights[expert_id], socket, expert_intermediate_size, expert_output);

                float weight = scratch.routing.weights.view()[0, k_idx];
                auto expert_out_view = expert_output.view();

                for (size_t h = 0; h < hidden_dim_local; h++) {
                    output_view[0, h] += static_cast<TOutput>(expert_out_view[0, h] * weight);
                }
            }
        });
    }
    threads.clear();
    threads.reserve(NUM_SOCKETS);
    for (int socket = 0; socket < NUM_SOCKETS; socket++) {
        threads.emplace_back([&, socket] {
            Numa::pin_to_socket(socket, 0);
            auto output_view = output_cp.view(socket);

            Tensor2D<int32_t> shared_output(Extents2D{1, hidden_dim_local});

            compute_swiglu_mlp_batch(hidden_cp[socket], shared_gate_weight, shared_up_weight, shared_down_weight, socket, shared_intermediate_size, shared_output);

            auto shared_output_view = shared_output.view();
            avx512math::add(&output_view[0, 0], &shared_output_view[0, 0], &output_view[0, 0], hidden_dim_local);
        });
    }
    threads.clear();
}

}
