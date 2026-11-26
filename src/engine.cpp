#include "ml_inference/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace ml_inference {

namespace {

constexpr uint32_t kMagicHeader = 0x53494C4D;  // "MLIS" in little-endian
constexpr uint32_t kFormatVersion = 1;
constexpr float kInvSqrt2 = 0.70710678118654752440f;

inline float gelu_scalar(float x) noexcept {
    return 0.5f * x * (1.0f + std::erf(x * kInvSqrt2));
}

uint32_t fnv1a_hash_bytes(const void* ptr, std::size_t num_bytes, uint32_t seed = 2166136261u) noexcept {
    const auto* bytes = static_cast<const uint8_t*>(ptr);
    uint32_t h = seed;
    for (std::size_t i = 0; i < num_bytes; ++i) {
        h ^= static_cast<uint32_t>(bytes[i]);
        h *= 16777619u;
    }
    return h;
}

}  // namespace

void op_matmul_bias(
    const float* input,
    std::size_t batch_size,
    std::size_t in_features,
    const float* weights_out_major,
    const float* bias,
    std::size_t out_features,
    float* output
) {
    for (std::size_t b = 0; b < batch_size; ++b) {
        const float* x_row = input + b * in_features;
        float* y_row = output + b * out_features;

        for (std::size_t o = 0; o < out_features; ++o) {
            const float* w_row = weights_out_major + o * in_features;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;

            std::size_t k = 0;
            const std::size_t k_unroll = (in_features / 4) * 4;
            for (; k < k_unroll; k += 4) {
                acc0 += x_row[k + 0] * w_row[k + 0];
                acc1 += x_row[k + 1] * w_row[k + 1];
                acc2 += x_row[k + 2] * w_row[k + 2];
                acc3 += x_row[k + 3] * w_row[k + 3];
            }
            float sum = (acc0 + acc1) + (acc2 + acc3);
            for (; k < in_features; ++k) {
                sum += x_row[k] * w_row[k];
            }
            y_row[o] = sum + (bias ? bias[o] : 0.0f);
        }
    }
}

void op_layer_norm_inplace(
    float* data,
    std::size_t batch_size,
    std::size_t features,
    const float* gamma,
    const float* beta,
    float eps
) {
    if (features == 0) {
        return;
    }
    const float inv_n = 1.0f / static_cast<float>(features);
    for (std::size_t b = 0; b < batch_size; ++b) {
        float* row = data + b * features;
        float sum = 0.0f;
        for (std::size_t j = 0; j < features; ++j) {
            sum += row[j];
        }
        const float mean = sum * inv_n;

        float var_sum = 0.0f;
        for (std::size_t j = 0; j < features; ++j) {
            const float d = row[j] - mean;
            var_sum += d * d;
        }
        const float inv_std = 1.0f / std::sqrt(var_sum * inv_n + eps);

        for (std::size_t j = 0; j < features; ++j) {
            const float norm = (row[j] - mean) * inv_std;
            const float g = gamma ? gamma[j] : 1.0f;
            const float bt = beta ? beta[j] : 0.0f;
            row[j] = norm * g + bt;
        }
    }
}

void op_relu_inplace(float* data, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        data[i] = data[i] > 0.0f ? data[i] : 0.0f;
    }
}

void op_gelu_inplace(float* data, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        data[i] = gelu_scalar(data[i]);
    }
}

void op_softmax_inplace(float* data, std::size_t batch_size, std::size_t classes) {
    if (classes == 0) {
        return;
    }
    for (std::size_t b = 0; b < batch_size; ++b) {
        float* row = data + b * classes;
        float max_val = row[0];
        for (std::size_t c = 1; c < classes; ++c) {
            if (row[c] > max_val) {
                max_val = row[c];
            }
        }
        float sum_exp = 0.0f;
        for (std::size_t c = 0; c < classes; ++c) {
            row[c] = std::exp(row[c] - max_val);
            sum_exp += row[c];
        }
        const float inv_sum = 1.0f / sum_exp;
        for (std::size_t c = 0; c < classes; ++c) {
            row[c] *= inv_sum;
        }
    }
}

void InferenceEngine::add_layer(LinearLayer layer) {
    if (layer.weights.size() != layer.in_features * layer.out_features) {
        throw std::invalid_argument("LinearLayer weight dimension mismatch");
    }
    if (layer.bias.size() != layer.out_features) {
        throw std::invalid_argument("LinearLayer bias dimension mismatch");
    }
    if (layer.has_layer_norm) {
        if (layer.ln_gamma.size() != layer.out_features || layer.ln_beta.size() != layer.out_features) {
            throw std::invalid_argument("LinearLayer LayerNorm parameter mismatch");
        }
    }
    if (!layers_.empty() && layers_.back().out_features != layer.in_features) {
        throw std::invalid_argument("Adjacent layer feature dimension mismatch");
    }
    layers_.push_back(std::move(layer));
}

InferenceEngine InferenceEngine::load_from_binary(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open binary model file: " + path);
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t num_layers = 0;
    uint32_t expected_checksum = 0;

    in.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&num_layers), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&expected_checksum), sizeof(uint32_t));

    if (!in || magic != kMagicHeader) {
        throw std::runtime_error("Invalid model binary header magic");
    }
    if (version != kFormatVersion) {
        throw std::runtime_error("Unsupported model binary version");
    }
    if (num_layers == 0 || num_layers > 256) {
        throw std::runtime_error("Invalid number of layers in model binary");
    }

    InferenceEngine engine;
    uint32_t rolling_hash = 2166136261u;

    for (uint32_t i = 0; i < num_layers; ++i) {
        uint32_t in_f = 0;
        uint32_t out_f = 0;
        uint32_t has_ln = 0;
        uint32_t act_code = 0;

        in.read(reinterpret_cast<char*>(&in_f), sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(&out_f), sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(&has_ln), sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(&act_code), sizeof(uint32_t));

        LinearLayer layer;
        layer.in_features = in_f;
        layer.out_features = out_f;
        layer.has_layer_norm = (has_ln != 0);
        layer.activation = static_cast<ActivationType>(act_code);

        layer.weights.resize(static_cast<std::size_t>(in_f) * out_f);
        layer.bias.resize(out_f);
        in.read(reinterpret_cast<char*>(layer.weights.data()), layer.weights.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(layer.bias.data()), layer.bias.size() * sizeof(float));

        rolling_hash = fnv1a_hash_bytes(layer.weights.data(), layer.weights.size() * sizeof(float), rolling_hash);
        rolling_hash = fnv1a_hash_bytes(layer.bias.data(), layer.bias.size() * sizeof(float), rolling_hash);

        if (layer.has_layer_norm) {
            layer.ln_gamma.resize(out_f);
            layer.ln_beta.resize(out_f);
            in.read(reinterpret_cast<char*>(layer.ln_gamma.data()), layer.ln_gamma.size() * sizeof(float));
            in.read(reinterpret_cast<char*>(layer.ln_beta.data()), layer.ln_beta.size() * sizeof(float));
            rolling_hash = fnv1a_hash_bytes(layer.ln_gamma.data(), layer.ln_gamma.size() * sizeof(float), rolling_hash);
            rolling_hash = fnv1a_hash_bytes(layer.ln_beta.data(), layer.ln_beta.size() * sizeof(float), rolling_hash);
        }

        if (!in) {
            throw std::runtime_error("Unexpected EOF while reading model binary layer " + std::to_string(i));
        }
        engine.add_layer(std::move(layer));
    }

    if (expected_checksum != 0 && rolling_hash != expected_checksum) {
        throw std::runtime_error("Model weight checksum verification failed");
    }
    engine.checksum_ = rolling_hash;
    return engine;
}

Tensor InferenceEngine::forward_unfused(const Tensor& input) const {
    if (layers_.empty()) {
        throw std::runtime_error("InferenceEngine has no layers");
    }
    if (input.cols != input_dim()) {
        throw std::invalid_argument("Input tensor column dimension mismatch");
    }

    Tensor current = input;
    for (const auto& layer : layers_) {
        Tensor next(current.rows, layer.out_features);
        // Step 1: Pure MatMul without bias
        op_matmul_bias(
            current.data.data(),
            current.rows,
            layer.in_features,
            layer.weights.data(),
            nullptr,
            layer.out_features,
            next.data.data()
        );
        // Step 2: Separate bias addition pass
        for (std::size_t r = 0; r < next.rows; ++r) {
            for (std::size_t c = 0; c < next.cols; ++c) {
                next.at(r, c) += layer.bias[c];
            }
        }
        // Step 3: LayerNorm if enabled
        if (layer.has_layer_norm) {
            op_layer_norm_inplace(
                next.data.data(),
                next.rows,
                next.cols,
                layer.ln_gamma.data(),
                layer.ln_beta.data()
            );
        }
        // Step 4: Activation pass
        if (layer.activation == ActivationType::RELU) {
            op_relu_inplace(next.data.data(), next.size());
        } else if (layer.activation == ActivationType::GELU) {
            op_gelu_inplace(next.data.data(), next.size());
        }
        current = std::move(next);
    }

    op_softmax_inplace(current.data.data(), current.rows, current.cols);
    return current;
}

void InferenceEngine::forward_fused_into(
    const Tensor& input,
    Tensor& output,
    ActivationArena& arena
) const {
    if (layers_.empty()) {
        throw std::runtime_error("InferenceEngine has no layers");
    }
    if (input.cols != input_dim()) {
        throw std::invalid_argument("Input tensor column dimension mismatch");
    }

    arena.reset();
    const std::size_t batch_size = input.rows;
    const float* cur_ptr = input.data.data();
    std::size_t cur_cols = input.cols;

    for (std::size_t idx = 0; idx < layers_.size(); ++idx) {
        const auto& layer = layers_[idx];
        const bool is_last = (idx + 1 == layers_.size());
        float* out_ptr = nullptr;
        if (is_last) {
            if (output.rows != batch_size || output.cols != layer.out_features) {
                output = Tensor(batch_size, layer.out_features);
            }
            out_ptr = output.data.data();
        } else {
            out_ptr = arena.allocate(batch_size * layer.out_features);
        }

        // Fused MatMul + Bias directly into pre-allocated arena buffer (zero heap allocations)
        op_matmul_bias(
            cur_ptr,
            batch_size,
            cur_cols,
            layer.weights.data(),
            layer.bias.data(),
            layer.out_features,
            out_ptr
        );

        // Cache-resident row-local LayerNorm + Activation before row leaves L1 cache
        for (std::size_t b = 0; b < batch_size; ++b) {
            float* y_row = out_ptr + b * layer.out_features;
            if (layer.has_layer_norm) {
                op_layer_norm_inplace(
                    y_row,
                    1,
                    layer.out_features,
                    layer.ln_gamma.data(),
                    layer.ln_beta.data()
                );
            }
            if (layer.activation == ActivationType::RELU) {
                op_relu_inplace(y_row, layer.out_features);
            } else if (layer.activation == ActivationType::GELU) {
                op_gelu_inplace(y_row, layer.out_features);
            }
        }

        cur_ptr = out_ptr;
        cur_cols = layer.out_features;
    }

    op_softmax_inplace(output.data.data(), batch_size, output.cols);
}

Tensor InferenceEngine::forward_fused(const Tensor& input) const {
    ActivationArena arena(std::max<std::size_t>(65536, input.rows * 1024));
    Tensor output(input.rows, output_dim());
    forward_fused_into(input, output, arena);
    return output;
}

}  // namespace ml_inference
