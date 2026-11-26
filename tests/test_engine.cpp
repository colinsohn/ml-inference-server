#include "inference_engine.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

using namespace mlis;

namespace {

void test_tensor_and_activations() {
    Tensor x(2, 4, {-2.0f, 0.0f, 1.0f, 4.0f, 1000.0f, 1001.0f, 999.0f, 1000.0f});
    Tensor r = x;
    relu_inplace(r);
    assert(r.at(0, 0) == 0.0f);
    assert(r.at(0, 2) == 1.0f);

    Tensor g = x;
    gelu_inplace(g);
    assert(g.at(0, 0) < 0.0f && g.at(0, 0) > -0.1f);
    assert(g.at(0, 2) > 0.8f && g.at(0, 2) < 0.85f);

    std::vector<float> gamma(4, 1.0f);
    std::vector<float> beta(4, 0.0f);
    Tensor ln = x;
    layer_norm_inplace(ln, gamma, beta);
    float row0_mean = (ln.at(0, 0) + ln.at(0, 1) + ln.at(0, 2) + ln.at(0, 3)) / 4.0f;
    assert(std::fabs(row0_mean) < 1e-5f);

    // Numerically stable softmax must not NaN/overflow on +1000 logits
    Tensor sm = x;
    softmax_inplace(sm);
    float sum_row1 = 0.0f;
    for (size_t c = 0; c < 4; ++c) {
        assert(std::isfinite(sm.at(1, c)));
        assert(sm.at(1, c) > 0.0f && sm.at(1, c) < 1.0f);
        sum_row1 += sm.at(1, c);
    }
    assert(std::fabs(sum_row1 - 1.0f) < 1e-5f);
    assert(sm.at(1, 1) > sm.at(1, 0) && sm.at(1, 0) > sm.at(1, 2));
}

void test_fp32_int8_parity_and_serialization() {
    SequentialModel model = SequentialModel::create_deterministic_mlp({32, 64, 32, 8}, Activation::GELU, true, true, 12345);
    assert(model.fp32_weight_bytes() > model.int8_weight_bytes() * 3);

    Tensor input(4, 32);
    for (size_t i = 0; i < input.size(); ++i) {
        input.data[i] = std::sin(static_cast<float>(i) * 0.17f);
    }

    Tensor out_fp32 = model.forward(input, Precision::FP32);
    Tensor out_int8 = model.forward(input, Precision::INT8);
    assert(out_fp32.rows == 4 && out_fp32.cols == 8);
    assert(out_int8.rows == 4 && out_int8.cols == 8);

    float diff = max_abs_diff(out_fp32, out_int8);
    // Symmetric per-channel INT8 quantization after softmax should stay within bounded tolerance
    assert(diff < 0.025f);

    const std::string path = "/tmp/mlis_unit_test_model.mlis";
    model.save(path);
    SequentialModel loaded = SequentialModel::load(path);
    Tensor out_loaded = loaded.forward(input, Precision::FP32);
    assert(max_abs_diff(out_fp32, out_loaded) == 0.0f);

    // Corrupt a single byte in the saved file and verify checksum rejection
    {
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(24);
        char b = 0;
        f.read(&b, 1);
        b ^= 0x5A;
        f.seekp(24);
        f.write(&b, 1);
    }
    bool caught_corruption = false;
    try {
        (void)SequentialModel::load(path);
    } catch (const std::runtime_error&) {
        caught_corruption = true;
    }
    std::remove(path.c_str());
    assert(caught_corruption);
}

void test_dynamic_batch_scheduler_and_backpressure() {
    SequentialModel model = SequentialModel::create_deterministic_mlp({16, 32, 4}, Activation::ReLU, true, true, 77);
    DynamicBatchScheduler scheduler(model, Precision::FP32, 16, std::chrono::microseconds(2000), 64);

    constexpr size_t kNumClients = 32;
    std::vector<Tensor> inputs;
    std::vector<std::future<Tensor>> futures;
    inputs.reserve(kNumClients);
    futures.reserve(kNumClients);

    for (size_t i = 0; i < kNumClients; ++i) {
        Tensor single(1, 16);
        for (size_t c = 0; c < 16; ++c) {
            single.at(0, c) = static_cast<float>((i + c) % 11) * 0.1f;
        }
        inputs.push_back(single);
        futures.push_back(scheduler.submit(single));
    }

    for (size_t i = 0; i < kNumClients; ++i) {
        Tensor batched_out = futures[i].get();
        Tensor direct_out = model.forward(inputs[i], Precision::FP32);
        assert(max_abs_diff(batched_out, direct_out) < 1e-5f);
    }

    SchedulerStats st = scheduler.stats();
    assert(st.total_requests == kNumClients);
    assert(st.total_items_inferred == kNumClients);
    assert(st.total_batches >= 2 && st.total_batches <= kNumClients);
    scheduler.shutdown();
}

}  // namespace

int main() {
    test_tensor_and_activations();
    test_fp32_int8_parity_and_serialization();
    test_dynamic_batch_scheduler_and_backpressure();
    std::cout << "PASS: all C++20 inference engine, quantization, checksum, and dynamic batching tests passed.\n";
    return 0;
}
