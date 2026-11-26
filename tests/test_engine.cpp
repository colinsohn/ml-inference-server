#include "ml_inference/engine.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void test_matmul_and_activations() {
    // 2x3 input * 2x3 weights^T + bias(2)
    const float input[6] = {
        1.0f, 2.0f, 3.0f,
        -1.0f, 0.5f, 2.0f
    };
    const float weights[6] = {
        0.5f, -0.5f, 1.0f,   // out 0
        -1.0f, 2.0f, 0.25f   // out 1
    };
    const float bias[2] = {0.1f, -0.2f};
    float output[4] = {0.0f};

    ml_inference::op_matmul_bias(input, 2, 3, weights, bias, 2, output);

    // Row 0, Out 0: 1*0.5 + 2*(-0.5) + 3*1.0 + 0.1 = 2.6
    // Row 0, Out 1: 1*(-1) + 2*2 + 3*0.25 - 0.2 = 3.55
    assert(std::abs(output[0] - 2.6f) < 1e-6f);
    assert(std::abs(output[1] - 3.55f) < 1e-6f);

    ml_inference::op_relu_inplace(output, 4);
    for (float v : output) {
        assert(v >= 0.0f);
    }
}

void test_layer_norm_invariance() {
    std::vector<float> row = {2.0f, 4.0f, 6.0f, 8.0f};
    std::vector<float> gamma = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> beta = {0.0f, 0.0f, 0.0f, 0.0f};

    ml_inference::op_layer_norm_inplace(row.data(), 1, 4, gamma.data(), beta.data(), 1e-6f);

    float mean = 0.0f;
    float var = 0.0f;
    for (float v : row) {
        mean += v;
        var += v * v;
    }
    mean /= 4.0f;
    var /= 4.0f;
    assert(std::abs(mean) < 1e-6f);
    assert(std::abs(var - 1.0f) < 1e-4f);
}

void test_softmax_numerical_stability() {
    std::vector<float> extreme_logits = {1000.0f, 1001.0f, 999.0f};
    ml_inference::op_softmax_inplace(extreme_logits.data(), 1, 3);
    float sum = 0.0f;
    for (float p : extreme_logits) {
        assert(std::isfinite(p));
        assert(p > 0.0f && p < 1.0f);
        sum += p;
    }
    assert(std::abs(sum - 1.0f) < 1e-6f);
    assert(extreme_logits[1] > extreme_logits[0]);
    assert(extreme_logits[0] > extreme_logits[2]);
}

void test_fused_vs_unfused_graph_equivalence_and_arena_zero_growth() {
    ml_inference::InferenceEngine engine;

    ml_inference::LinearLayer l1;
    l1.in_features = 4;
    l1.out_features = 6;
    l1.has_layer_norm = true;
    l1.activation = ml_inference::ActivationType::GELU;
    l1.weights.resize(24);
    for (std::size_t i = 0; i < 24; ++i) {
        l1.weights[i] = static_cast<float>(static_cast<int>(i) - 10) * 0.1f;
    }
    l1.bias = {0.1f, -0.2f, 0.05f, 0.3f, -0.1f, 0.2f};
    l1.ln_gamma = {1.1f, 0.9f, 1.0f, 1.05f, 0.95f, 1.0f};
    l1.ln_beta = {0.01f, -0.02f, 0.0f, 0.03f, -0.01f, 0.0f};

    ml_inference::LinearLayer l2;
    l2.in_features = 6;
    l2.out_features = 3;
    l2.has_layer_norm = false;
    l2.activation = ml_inference::ActivationType::NONE;
    l2.weights.resize(18);
    for (std::size_t i = 0; i < 18; ++i) {
        l2.weights[i] = static_cast<float>(static_cast<int>(i) - 8) * 0.15f;
    }
    l2.bias = {-0.1f, 0.25f, 0.05f};

    engine.add_layer(l1);
    engine.add_layer(l2);

    ml_inference::Tensor input(5, 4);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input.data[i] = static_cast<float>(static_cast<int>(i) - 9) * 0.2f;
    }

    auto unfused = engine.forward_unfused(input);
    ml_inference::ActivationArena arena(1024);
    ml_inference::Tensor fused(5, 3);
    engine.forward_fused_into(input, fused, arena);

    const std::size_t first_used = arena.used_floats();
    assert(first_used == 5 * 6);

    // Second pass reuses the exact same arena buffer without growing capacity
    engine.forward_fused_into(input, fused, arena);
    assert(arena.used_floats() == first_used);
    assert(arena.capacity_floats() == 1024);

    for (std::size_t i = 0; i < fused.size(); ++i) {
        assert(std::abs(fused.data[i] - unfused.data[i]) < 1e-6f);
    }

    // Verify invalid input shape throws std::invalid_argument
    ml_inference::Tensor bad_input(2, 5);
    bool threw = false;
    try {
        (void)engine.forward_fused(bad_input);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

}  // namespace

int main() {
    test_matmul_and_activations();
    test_layer_norm_invariance();
    test_softmax_numerical_stability();
    test_fused_vs_unfused_graph_equivalence_and_arena_zero_growth();
    std::cout << "[PASS] All C++17 inference engine unit tests succeeded (4/4 suites).\n";
    return 0;
}
