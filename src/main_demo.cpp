#include "inference_engine.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace mlis;

int main(int argc, char** argv) {
    SequentialModel model;
    std::string source_desc = "built-in deterministic MLP [64 -> 128 -> 64 -> 8]";
    if (argc > 1) {
        model = SequentialModel::load(argv[1]);
        source_desc = std::string("loaded binary model ") + argv[1];
    } else {
        model = SequentialModel::create_deterministic_mlp({64, 128, 64, 8}, Activation::GELU, true, true, 42);
    }

    Tensor sample(2, model.input_dim());
    for (size_t i = 0; i < sample.size(); ++i) {
        sample.data[i] = std::cos(static_cast<float>(i) * 0.13f) * 0.5f;
    }

    Tensor fp32_out = model.forward(sample, Precision::FP32);
    Tensor int8_out = model.forward(sample, Precision::INT8);
    float max_diff = max_abs_diff(fp32_out, int8_out);

    DynamicBatchScheduler scheduler(model, Precision::INT8, 16, std::chrono::microseconds(1000), 128);
    std::vector<std::future<Tensor>> futures;
    for (int i = 0; i < 8; ++i) {
        futures.push_back(scheduler.submit(sample));
    }
    for (auto& f : futures) {
        (void)f.get();
    }
    scheduler.shutdown();
    SchedulerStats st = scheduler.stats();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "{\n";
    std::cout << "  \"model_source\": \"" << source_desc << "\",\n";
    std::cout << "  \"layers\": " << model.num_layers() << ",\n";
    std::cout << "  \"input_dim\": " << model.input_dim() << ",\n";
    std::cout << "  \"output_dim\": " << model.output_dim() << ",\n";
    std::cout << "  \"fp32_weight_bytes\": " << model.fp32_weight_bytes() << ",\n";
    std::cout << "  \"int8_weight_bytes\": " << model.int8_weight_bytes() << ",\n";
    std::cout << "  \"compression_ratio\": "
              << (static_cast<double>(model.fp32_weight_bytes()) / static_cast<double>(model.int8_weight_bytes()))
              << ",\n";
    std::cout << "  \"fp32_vs_int8_max_abs_diff\": " << max_diff << ",\n";
    std::cout << "  \"scheduler_requests\": " << st.total_requests << ",\n";
    std::cout << "  \"scheduler_coalesced_batches\": " << st.total_batches << ",\n";
    std::cout << "  \"scheduler_items_inferred\": " << st.total_items_inferred << "\n";
    std::cout << "}\n";
    return 0;
}
