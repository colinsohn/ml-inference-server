#include "inference_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>

namespace mlis {

namespace {

constexpr char kMagicHeader[8] = {'M', 'L', 'I', 'S', 'v', '0', '0', '1'};

uint64_t fnv1a_update(uint64_t hash, const void* bytes, size_t len) {
    const auto* ptr = static_cast<const uint8_t*>(bytes);
    for (size_t i = 0; i < len; ++i) {
        hash ^= static_cast<uint64_t>(ptr[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

}  // namespace

Tensor::Tensor(size_t r, size_t c, float fill) : rows(r), cols(c), data(r * c, fill) {
    if (r == 0 || c == 0) {
        throw std::invalid_argument("Tensor dimensions must be positive");
    }
}

Tensor::Tensor(size_t r, size_t c, std::vector<float> values) : rows(r), cols(c), data(std::move(values)) {
    if (r == 0 || c == 0 || data.size() != r * c) {
        throw std::invalid_argument("Tensor data size does not match rows * cols");
    }
}

float& Tensor::at(size_t r, size_t c) {
    if (r >= rows || c >= cols) {
        throw std::out_of_range("Tensor index out of range");
    }
    return data[r * cols + c];
}

float Tensor::at(size_t r, size_t c) const {
    if (r >= rows || c >= cols) {
        throw std::out_of_range("Tensor index out of range");
    }
    return data[r * cols + c];
}

float max_abs_diff(const Tensor& a, const Tensor& b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        throw std::invalid_argument("Tensor shape mismatch in max_abs_diff");
    }
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.data.size(); ++i) {
        float d = std::fabs(a.data[i] - b.data[i]);
        if (d > max_diff) {
            max_diff = d;
        }
    }
    return max_diff;
}

void relu_inplace(Tensor& x) {
    for (float& v : x.data) {
        if (v < 0.0f) {
            v = 0.0f;
        }
    }
}

void gelu_inplace(Tensor& x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kCubicCoeff = 0.044715f;
    for (float& v : x.data) {
        float inner = kSqrt2OverPi * (v + kCubicCoeff * v * v * v);
        v = 0.5f * v * (1.0f + std::tanh(inner));
    }
}

void layer_norm_inplace(Tensor& x, const std::vector<float>& gamma, const std::vector<float>& beta, float eps) {
    if (gamma.size() != x.cols || beta.size() != x.cols || eps <= 0.0f) {
        throw std::invalid_argument("invalid LayerNorm parameters");
    }
    for (size_t r = 0; r < x.rows; ++r) {
        float* row = &x.data[r * x.cols];
        float mean = 0.0f;
        for (size_t c = 0; c < x.cols; ++c) {
            mean += row[c];
        }
        mean /= static_cast<float>(x.cols);

        float var = 0.0f;
        for (size_t c = 0; c < x.cols; ++c) {
            float diff = row[c] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(x.cols);
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (size_t c = 0; c < x.cols; ++c) {
            row[c] = (row[c] - mean) * inv_std * gamma[c] + beta[c];
        }
    }
}

void softmax_inplace(Tensor& x) {
    for (size_t r = 0; r < x.rows; ++r) {
        float* row = &x.data[r * x.cols];
        float max_val = row[0];
        for (size_t c = 1; c < x.cols; ++c) {
            if (row[c] > max_val) {
                max_val = row[c];
            }
        }
        float sum = 0.0f;
        for (size_t c = 0; c < x.cols; ++c) {
            row[c] = std::exp(row[c] - max_val);
            sum += row[c];
        }
        float inv_sum = 1.0f / sum;
        for (size_t c = 0; c < x.cols; ++c) {
            row[c] *= inv_sum;
        }
    }
}

LinearLayer::LinearLayer(
    size_t in_dim,
    size_t out_dim,
    std::vector<float> w,
    std::vector<float> b,
    Activation act,
    bool layer_norm
)
    : in_features(in_dim),
      out_features(out_dim),
      activation(act),
      use_layer_norm(layer_norm),
      weights(std::move(w)),
      bias(std::move(b)),
      ln_gamma(out_dim, 1.0f),
      ln_beta(out_dim, 0.0f) {
    if (in_dim == 0 || out_dim == 0 || weights.size() != in_dim * out_dim || bias.size() != out_dim) {
        throw std::invalid_argument("LinearLayer weight/bias dimension mismatch");
    }
    quantize_per_channel();
}

void LinearLayer::quantize_per_channel() {
    qweights.resize(out_features * in_features);
    qscales.resize(out_features);
    for (size_t o = 0; o < out_features; ++o) {
        const float* row = &weights[o * in_features];
        float max_abs = 0.0f;
        for (size_t i = 0; i < in_features; ++i) {
            float a = std::fabs(row[i]);
            if (a > max_abs) {
                max_abs = a;
            }
        }
        float scale = (max_abs > 1e-12f) ? (max_abs / 127.0f) : 1.0f;
        qscales[o] = scale;
        float inv_scale = 1.0f / scale;
        int8_t* qrow = &qweights[o * in_features];
        for (size_t i = 0; i < in_features; ++i) {
            int q = static_cast<int>(std::lround(row[i] * inv_scale));
            q = std::clamp(q, -127, 127);
            qrow[i] = static_cast<int8_t>(q);
        }
    }
}

Tensor LinearLayer::forward(const Tensor& input, Precision precision) const {
    if (input.cols != in_features) {
        throw std::invalid_argument("LinearLayer input dimension mismatch");
    }
    Tensor out(input.rows, out_features);
    if (precision == Precision::FP32) {
        for (size_t b = 0; b < input.rows; ++b) {
            const float* in_row = &input.data[b * in_features];
            float* out_row = &out.data[b * out_features];
            for (size_t o = 0; o < out_features; ++o) {
                const float* w_row = &weights[o * in_features];
                float acc = bias[o];
                for (size_t i = 0; i < in_features; ++i) {
                    acc += in_row[i] * w_row[i];
                }
                out_row[o] = acc;
            }
        }
    } else {
        for (size_t b = 0; b < input.rows; ++b) {
            const float* in_row = &input.data[b * in_features];
            float* out_row = &out.data[b * out_features];
            for (size_t o = 0; o < out_features; ++o) {
                const int8_t* qw_row = &qweights[o * in_features];
                float acc = 0.0f;
                for (size_t i = 0; i < in_features; ++i) {
                    acc += in_row[i] * static_cast<float>(qw_row[i]);
                }
                out_row[o] = acc * qscales[o] + bias[o];
            }
        }
    }

    if (use_layer_norm) {
        layer_norm_inplace(out, ln_gamma, ln_beta);
    }
    if (activation == Activation::ReLU) {
        relu_inplace(out);
    } else if (activation == Activation::GELU) {
        gelu_inplace(out);
    }
    return out;
}

SequentialModel::SequentialModel(std::vector<LinearLayer> layers, bool final_softmax)
    : layers_(std::move(layers)), final_softmax_(final_softmax) {
    if (layers_.empty()) {
        throw std::invalid_argument("SequentialModel requires at least one layer");
    }
    for (size_t i = 1; i < layers_.size(); ++i) {
        if (layers_[i - 1].out_features != layers_[i].in_features) {
            throw std::invalid_argument("Adjacent layer dimensions do not match");
        }
    }
}

SequentialModel SequentialModel::create_deterministic_mlp(
    const std::vector<size_t>& dims,
    Activation hidden_activation,
    bool use_layer_norm,
    bool final_softmax,
    uint64_t seed
) {
    if (dims.size() < 2) {
        throw std::invalid_argument("dims must contain at least input and output dimensions");
    }
    uint64_t state = seed ? seed : 1ULL;
    auto next_uniform = [&state]() -> float {
        // SplitMix64 deterministic PRNG
        state += 0x9e3779b97f4a7c15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        double u = static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
        return static_cast<float>(u * 2.0 - 1.0);
    };

    std::vector<LinearLayer> layers;
    layers.reserve(dims.size() - 1);
    for (size_t idx = 0; idx + 1 < dims.size(); ++idx) {
        size_t in_d = dims[idx];
        size_t out_d = dims[idx + 1];
        bool is_last = (idx + 2 == dims.size());
        float limit = std::sqrt(6.0f / static_cast<float>(in_d + out_d));

        std::vector<float> w(in_d * out_d);
        for (float& val : w) {
            val = next_uniform() * limit;
        }
        std::vector<float> b(out_d);
        for (float& val : b) {
            val = next_uniform() * 0.05f;
        }
        layers.emplace_back(
            in_d,
            out_d,
            std::move(w),
            std::move(b),
            is_last ? Activation::None : hidden_activation,
            is_last ? false : use_layer_norm
        );
    }
    return SequentialModel(std::move(layers), final_softmax);
}

Tensor SequentialModel::forward(const Tensor& input, Precision precision) const {
    Tensor current = input;
    for (const auto& layer : layers_) {
        current = layer.forward(current, precision);
    }
    if (final_softmax_) {
        softmax_inplace(current);
    }
    return current;
}

size_t SequentialModel::input_dim() const { return layers_.front().in_features; }
size_t SequentialModel::output_dim() const { return layers_.back().out_features; }

size_t SequentialModel::fp32_weight_bytes() const noexcept {
    size_t total = 0;
    for (const auto& l : layers_) {
        total += l.weights.size() * sizeof(float);
    }
    return total;
}

size_t SequentialModel::int8_weight_bytes() const noexcept {
    size_t total = 0;
    for (const auto& l : layers_) {
        total += l.qweights.size() * sizeof(int8_t) + l.qscales.size() * sizeof(float);
    }
    return total;
}

void SequentialModel::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open output model file: " + path);
    }

    uint64_t hash = 14695981039346656037ULL;
    auto write_checked = [&](const void* ptr, size_t len) {
        out.write(static_cast<const char*>(ptr), static_cast<std::streamsize>(len));
        if (!out) {
            throw std::runtime_error("failed while writing model file");
        }
        hash = fnv1a_update(hash, ptr, len);
    };

    write_checked(kMagicHeader, sizeof(kMagicHeader));
    uint32_t n_layers = static_cast<uint32_t>(layers_.size());
    uint8_t softmax_flag = final_softmax_ ? 1 : 0;
    write_checked(&n_layers, sizeof(n_layers));
    write_checked(&softmax_flag, sizeof(softmax_flag));

    for (const auto& l : layers_) {
        uint32_t in_d = static_cast<uint32_t>(l.in_features);
        uint32_t out_d = static_cast<uint32_t>(l.out_features);
        uint8_t act = static_cast<uint8_t>(l.activation);
        uint8_t ln = l.use_layer_norm ? 1 : 0;
        write_checked(&in_d, sizeof(in_d));
        write_checked(&out_d, sizeof(out_d));
        write_checked(&act, sizeof(act));
        write_checked(&ln, sizeof(ln));
        write_checked(l.weights.data(), l.weights.size() * sizeof(float));
        write_checked(l.bias.data(), l.bias.size() * sizeof(float));
        write_checked(l.ln_gamma.data(), l.ln_gamma.size() * sizeof(float));
        write_checked(l.ln_beta.data(), l.ln_beta.size() * sizeof(float));
    }

    out.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
    if (!out) {
        throw std::runtime_error("failed to write model checksum");
    }
}

SequentialModel SequentialModel::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open model file: " + path);
    }

    uint64_t hash = 14695981039346656037ULL;
    auto read_checked = [&](void* ptr, size_t len) {
        in.read(static_cast<char*>(ptr), static_cast<std::streamsize>(len));
        if (!in || static_cast<size_t>(in.gcount()) != len) {
            throw std::runtime_error("truncated or unreadable model file");
        }
        hash = fnv1a_update(hash, ptr, len);
    };

    char magic[8];
    read_checked(magic, sizeof(magic));
    if (std::memcmp(magic, kMagicHeader, sizeof(kMagicHeader)) != 0) {
        throw std::runtime_error("invalid model file magic header");
    }

    uint32_t n_layers = 0;
    uint8_t softmax_flag = 0;
    read_checked(&n_layers, sizeof(n_layers));
    read_checked(&softmax_flag, sizeof(softmax_flag));
    if (n_layers == 0 || n_layers > 256) {
        throw std::runtime_error("invalid layer count in model header");
    }

    std::vector<LinearLayer> layers;
    layers.reserve(n_layers);
    for (uint32_t idx = 0; idx < n_layers; ++idx) {
        uint32_t in_d = 0;
        uint32_t out_d = 0;
        uint8_t act = 0;
        uint8_t ln = 0;
        read_checked(&in_d, sizeof(in_d));
        read_checked(&out_d, sizeof(out_d));
        read_checked(&act, sizeof(act));
        read_checked(&ln, sizeof(ln));
        if (in_d == 0 || out_d == 0 || in_d > 65536 || out_d > 65536 || act > 2) {
            throw std::runtime_error("corrupt layer descriptor in model file");
        }

        std::vector<float> w(static_cast<size_t>(in_d) * out_d);
        std::vector<float> b(out_d);
        std::vector<float> gamma(out_d);
        std::vector<float> beta(out_d);
        read_checked(w.data(), w.size() * sizeof(float));
        read_checked(b.data(), b.size() * sizeof(float));
        read_checked(gamma.data(), gamma.size() * sizeof(float));
        read_checked(beta.data(), beta.size() * sizeof(float));

        LinearLayer layer(in_d, out_d, std::move(w), std::move(b), static_cast<Activation>(act), ln != 0);
        layer.ln_gamma = std::move(gamma);
        layer.ln_beta = std::move(beta);
        layers.push_back(std::move(layer));
    }

    uint64_t expected_hash = 0;
    in.read(reinterpret_cast<char*>(&expected_hash), sizeof(expected_hash));
    if (!in || static_cast<size_t>(in.gcount()) != sizeof(expected_hash) || expected_hash != hash) {
        throw std::runtime_error("model checksum verification failed");
    }
    if (in.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("unexpected trailing bytes in model file");
    }

    return SequentialModel(std::move(layers), softmax_flag != 0);
}

DynamicBatchScheduler::DynamicBatchScheduler(
    const SequentialModel& model,
    Precision precision,
    size_t max_batch_size,
    std::chrono::microseconds batch_timeout,
    size_t max_queue_depth
)
    : model_(model),
      precision_(precision),
      max_batch_size_(max_batch_size),
      batch_timeout_(batch_timeout),
      max_queue_depth_(max_queue_depth) {
    if (max_batch_size_ == 0 || max_queue_depth_ == 0) {
        throw std::invalid_argument("DynamicBatchScheduler capacities must be positive");
    }
    worker_ = std::thread(&DynamicBatchScheduler::worker_loop, this);
}

DynamicBatchScheduler::~DynamicBatchScheduler() {
    shutdown();
}

std::future<Tensor> DynamicBatchScheduler::submit(Tensor single_or_microbatch) {
    if (single_or_microbatch.cols != model_.input_dim()) {
        throw std::invalid_argument("submitted input columns do not match model input_dim");
    }
    std::promise<Tensor> prom;
    std::future<Tensor> fut = prom.get_future();
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (stop_) {
            throw std::runtime_error("scheduler is shut down");
        }
        if (queue_.size() >= max_queue_depth_) {
            rejected_overloaded_.fetch_add(1, std::memory_order_relaxed);
            throw OverloadError("inference queue capacity exceeded");
        }
        total_requests_.fetch_add(1, std::memory_order_relaxed);
        queue_.push_back(WorkItem{std::move(single_or_microbatch), std::move(prom)});
    }
    cv_.notify_one();
    return fut;
}

void DynamicBatchScheduler::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (stop_) {
            return;
        }
        stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

SchedulerStats DynamicBatchScheduler::stats() const noexcept {
    return SchedulerStats{
        total_requests_.load(std::memory_order_relaxed),
        total_batches_.load(std::memory_order_relaxed),
        total_items_inferred_.load(std::memory_order_relaxed),
        rejected_overloaded_.load(std::memory_order_relaxed),
    };
}

void DynamicBatchScheduler::worker_loop() {
    const size_t in_cols = model_.input_dim();
    const size_t out_cols = model_.output_dim();

    while (true) {
        std::vector<WorkItem> batch;
        size_t total_rows = 0;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [&]() { return stop_ || !queue_.empty(); });
            if (stop_ && queue_.empty()) {
                return;
            }

            auto deadline = std::chrono::steady_clock::now() + batch_timeout_;
            while (total_rows < max_batch_size_) {
                while (!queue_.empty() && total_rows < max_batch_size_) {
                    if (total_rows > 0 && total_rows + queue_.front().input.rows > max_batch_size_) {
                        break;
                    }
                    total_rows += queue_.front().input.rows;
                    batch.push_back(std::move(queue_.front()));
                    queue_.pop_front();
                }
                if (total_rows >= max_batch_size_ || stop_ || batch_timeout_.count() <= 0) {
                    break;
                }
                if (cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
                    break;
                }
            }
        }

        if (batch.empty()) {
            continue;
        }

        try {
            Tensor combined(total_rows, in_cols);
            size_t row_cursor = 0;
            for (const auto& item : batch) {
                std::memcpy(
                    &combined.data[row_cursor * in_cols],
                    item.input.data.data(),
                    item.input.rows * in_cols * sizeof(float)
                );
                row_cursor += item.input.rows;
            }

            Tensor batch_out = model_.forward(combined, precision_);
            total_batches_.fetch_add(1, std::memory_order_relaxed);
            total_items_inferred_.fetch_add(total_rows, std::memory_order_relaxed);

            size_t out_cursor = 0;
            for (auto& item : batch) {
                Tensor slice(item.input.rows, out_cols);
                std::memcpy(
                    slice.data.data(),
                    &batch_out.data[out_cursor * out_cols],
                    item.input.rows * out_cols * sizeof(float)
                );
                out_cursor += item.input.rows;
                item.promise.set_value(std::move(slice));
            }
        } catch (...) {
            auto ex = std::current_exception();
            for (auto& item : batch) {
                item.promise.set_exception(ex);
            }
        }
    }
}

}  // namespace mlis
