#ifndef MLIS_INFERENCE_ENGINE_HPP
#define MLIS_INFERENCE_ENGINE_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace mlis {

enum class Precision : uint8_t {
    FP32 = 0,
    INT8 = 1,
};

enum class Activation : uint8_t {
    None = 0,
    ReLU = 1,
    GELU = 2,
};

struct Tensor {
    size_t rows = 0;
    size_t cols = 0;
    std::vector<float> data;

    Tensor() = default;
    Tensor(size_t r, size_t c, float fill = 0.0f);
    Tensor(size_t r, size_t c, std::vector<float> values);

    float& at(size_t r, size_t c);
    float at(size_t r, size_t c) const;
    size_t size() const noexcept { return data.size(); }
};

float max_abs_diff(const Tensor& a, const Tensor& b);
void relu_inplace(Tensor& x);
void gelu_inplace(Tensor& x);
void layer_norm_inplace(Tensor& x, const std::vector<float>& gamma, const std::vector<float>& beta, float eps = 1e-5f);
void softmax_inplace(Tensor& x);

struct LinearLayer {
    size_t in_features = 0;
    size_t out_features = 0;
    Activation activation = Activation::None;
    bool use_layer_norm = false;

    std::vector<float> weights;     // out_features * in_features (row-major)
    std::vector<float> bias;        // out_features
    std::vector<float> ln_gamma;    // out_features
    std::vector<float> ln_beta;     // out_features

    std::vector<int8_t> qweights;   // out_features * in_features (symmetric per-channel INT8)
    std::vector<float> qscales;     // out_features

    LinearLayer() = default;
    LinearLayer(
        size_t in_dim,
        size_t out_dim,
        std::vector<float> w,
        std::vector<float> b,
        Activation act = Activation::None,
        bool layer_norm = false
    );

    void quantize_per_channel();
    Tensor forward(const Tensor& input, Precision precision) const;
};

class SequentialModel {
public:
    SequentialModel() = default;
    explicit SequentialModel(std::vector<LinearLayer> layers, bool final_softmax = true);

    static SequentialModel create_deterministic_mlp(
        const std::vector<size_t>& dims,
        Activation hidden_activation = Activation::GELU,
        bool use_layer_norm = true,
        bool final_softmax = true,
        uint64_t seed = 42
    );

    Tensor forward(const Tensor& input, Precision precision = Precision::FP32) const;

    void save(const std::string& path) const;
    static SequentialModel load(const std::string& path);

    size_t input_dim() const;
    size_t output_dim() const;
    size_t num_layers() const noexcept { return layers_.size(); }
    size_t fp32_weight_bytes() const noexcept;
    size_t int8_weight_bytes() const noexcept;
    const std::vector<LinearLayer>& layers() const noexcept { return layers_; }

private:
    std::vector<LinearLayer> layers_;
    bool final_softmax_ = true;
};

struct SchedulerStats {
    uint64_t total_requests = 0;
    uint64_t total_batches = 0;
    uint64_t total_items_inferred = 0;
    uint64_t rejected_overloaded = 0;
};

class OverloadError : public std::runtime_error {
public:
    explicit OverloadError(const std::string& msg) : std::runtime_error(msg) {}
};

class DynamicBatchScheduler {
public:
    DynamicBatchScheduler(
        const SequentialModel& model,
        Precision precision,
        size_t max_batch_size,
        std::chrono::microseconds batch_timeout,
        size_t max_queue_depth = 1024
    );
    ~DynamicBatchScheduler();

    DynamicBatchScheduler(const DynamicBatchScheduler&) = delete;
    DynamicBatchScheduler& operator=(const DynamicBatchScheduler&) = delete;

    std::future<Tensor> submit(Tensor single_or_microbatch);
    void shutdown();
    SchedulerStats stats() const noexcept;

private:
    struct WorkItem {
        Tensor input;
        std::promise<Tensor> promise;
    };

    void worker_loop();

    const SequentialModel& model_;
    Precision precision_;
    size_t max_batch_size_;
    std::chrono::microseconds batch_timeout_;
    size_t max_queue_depth_;

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<WorkItem> queue_;
    bool stop_ = false;
    std::thread worker_;

    std::atomic<uint64_t> total_requests_{0};
    std::atomic<uint64_t> total_batches_{0};
    std::atomic<uint64_t> total_items_inferred_{0};
    std::atomic<uint64_t> rejected_overloaded_{0};
};

}  // namespace mlis

#endif  // MLIS_INFERENCE_ENGINE_HPP
