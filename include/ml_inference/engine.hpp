#pragma once

#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ml_inference {

struct Tensor {
    std::size_t rows{0};
    std::size_t cols{0};
    std::vector<float> data;

    Tensor() = default;
    Tensor(std::size_t r, std::size_t c, float init_val = 0.0f)
        : rows(r), cols(c), data(r * c, init_val) {}

    inline float& at(std::size_t r, std::size_t c) {
        return data[r * cols + c];
    }

    inline float at(std::size_t r, std::size_t c) const {
        return data[r * cols + c];
    }

    inline std::size_t size() const {
        return data.size();
    }
};

class ActivationArena {
public:
    explicit ActivationArena(std::size_t capacity_floats = 262144)
        : buffer_(capacity_floats, 0.0f),
          offset_(0),
          peak_used_(0),
          reset_count_(0),
          allocation_calls_(0),
          heap_reallocations_(0) {}

    void reset() noexcept {
        offset_ = 0;
        ++reset_count_;
    }

    float* allocate(std::size_t count) {
        if (offset_ + count > buffer_.size()) {
            throw std::runtime_error("ActivationArena capacity exceeded");
        }
        float* ptr = buffer_.data() + offset_;
        offset_ += count;
        if (offset_ > peak_used_) {
            peak_used_ = offset_;
        }
        ++allocation_calls_;
        return ptr;
    }

    std::size_t used_floats() const noexcept { return offset_; }
    std::size_t peak_used_floats() const noexcept { return peak_used_; }
    std::size_t capacity_floats() const noexcept { return buffer_.size(); }
    std::size_t reset_count() const noexcept { return reset_count_; }
    std::size_t allocation_calls() const noexcept { return allocation_calls_; }
    std::size_t heap_reallocations() const noexcept { return heap_reallocations_; }

private:
    std::vector<float> buffer_;
    std::size_t offset_;
    std::size_t peak_used_;
    std::size_t reset_count_;
    std::size_t allocation_calls_;
    std::size_t heap_reallocations_;
};

enum class ActivationType : uint32_t {
    NONE = 0,
    RELU = 1,
    GELU = 2
};

struct LinearLayer {
    std::size_t in_features{0};
    std::size_t out_features{0};
    bool has_layer_norm{false};
    ActivationType activation{ActivationType::NONE};
    // Weight matrix stored row-major as [out_features, in_features] for unit-stride dot products
    std::vector<float> weights;
    std::vector<float> bias;
    std::vector<float> ln_gamma;
    std::vector<float> ln_beta;
};

class InferenceEngine {
public:
    InferenceEngine() = default;

    static InferenceEngine load_from_binary(const std::string& path);

    void add_layer(LinearLayer layer);

    // Unfused baseline execution (separate matmul, bias, norm, activation passes)
    Tensor forward_unfused(const Tensor& input) const;

    // Fused execution (single cache-resident pass per layer using pre-allocated buffers)
    Tensor forward_fused(const Tensor& input) const;
    void forward_fused_into(const Tensor& input, Tensor& output, ActivationArena& arena) const;

    // Static ONNX JSON graph importer & shape inference validator against loaded layers
    void verify_onnx_json_topology(const std::string& onnx_json_path) const;

    std::size_t num_layers() const noexcept { return layers_.size(); }
    std::size_t input_dim() const noexcept { return layers_.empty() ? 0 : layers_.front().in_features; }
    std::size_t output_dim() const noexcept { return layers_.empty() ? 0 : layers_.back().out_features; }
    uint32_t model_checksum() const noexcept { return checksum_; }

private:
    std::vector<LinearLayer> layers_;
    uint32_t checksum_{0};
};

// Standalone operator primitives exposed for unit testing
void op_matmul_bias(
    const float* input,
    std::size_t batch_size,
    std::size_t in_features,
    const float* weights_out_major,
    const float* bias,
    std::size_t out_features,
    float* output
);

void op_layer_norm_inplace(
    float* data,
    std::size_t batch_size,
    std::size_t features,
    const float* gamma,
    const float* beta,
    float eps = 1e-5f
);

void op_relu_inplace(float* data, std::size_t count);
void op_gelu_inplace(float* data, std::size_t count);
void op_softmax_inplace(float* data, std::size_t batch_size, std::size_t classes);

}  // namespace ml_inference
