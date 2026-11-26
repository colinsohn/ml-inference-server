#include "inference_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace mlis;

namespace {

struct BenchResult {
    std::string mode;
    std::string precision;
    size_t batch_size = 0;
    size_t iterations = 0;
    size_t total_items = 0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    double p99_us = 0.0;
    double mean_us = 0.0;
    double items_per_sec = 0.0;
};

double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) {
        return 0.0;
    }
    size_t idx = static_cast<size_t>(std::ceil(p * static_cast<double>(sorted.size())));
    if (idx > 0) {
        --idx;
    }
    if (idx >= sorted.size()) {
        idx = sorted.size() - 1;
    }
    return sorted[idx];
}

BenchResult run_direct_bench(
    const SequentialModel& model,
    Precision precision,
    size_t batch_size,
    size_t iterations,
    const std::string& mode_label
) {
    Tensor input(batch_size, model.input_dim());
    for (size_t i = 0; i < input.size(); ++i) {
        input.data[i] = std::sin(static_cast<float>(i) * 0.07f) * 0.5f;
    }

    // Warm up caches
    for (size_t w = 0; w < 10; ++w) {
        (void)model.forward(input, precision);
    }

    std::vector<double> latencies_us;
    latencies_us.reserve(iterations);
    auto t0 = std::chrono::steady_clock::now();
    for (size_t it = 0; it < iterations; ++it) {
        auto s0 = std::chrono::steady_clock::now();
        Tensor out = model.forward(input, precision);
        if (out.data[0] < -1.0f) {
            std::cerr << "impossible softmax probability\n";
        }
        auto s1 = std::chrono::steady_clock::now();
        latencies_us.push_back(std::chrono::duration<double, std::micro>(s1 - s0).count());
    }
    auto t1 = std::chrono::steady_clock::now();
    double wall_sec = std::chrono::duration<double>(t1 - t0).count();

    std::sort(latencies_us.begin(), latencies_us.end());
    double sum_us = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
    size_t total_items = batch_size * iterations;

    return BenchResult{
        mode_label,
        precision == Precision::FP32 ? "FP32" : "INT8",
        batch_size,
        iterations,
        total_items,
        percentile(latencies_us, 0.50),
        percentile(latencies_us, 0.95),
        percentile(latencies_us, 0.99),
        sum_us / static_cast<double>(iterations),
        static_cast<double>(total_items) / wall_sec,
    };
}

}  // namespace

int main(int argc, char** argv) {
    std::string report_path;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--report") {
            report_path = argv[i + 1];
        }
    }

    SequentialModel model = SequentialModel::create_deterministic_mlp({128, 256, 128, 16}, Activation::GELU, true, true, 42);

    Tensor parity_input(32, 128);
    for (size_t i = 0; i < parity_input.size(); ++i) {
        parity_input.data[i] = std::cos(static_cast<float>(i) * 0.11f) * 0.4f;
    }
    float max_diff = max_abs_diff(
        model.forward(parity_input, Precision::FP32),
        model.forward(parity_input, Precision::INT8)
    );

    std::vector<BenchResult> results = {
        run_direct_bench(model, Precision::FP32, 1, 2000, "single_request_fp32"),
        run_direct_bench(model, Precision::INT8, 1, 2000, "single_request_int8"),
        run_direct_bench(model, Precision::FP32, 32, 400, "batched_32_fp32"),
        run_direct_bench(model, Precision::INT8, 32, 400, "batched_32_int8"),
    };

    std::ostringstream json;
    json << std::fixed << std::setprecision(3);
    json << "{\n";
    json << "  \"format_version\": 1,\n";
    json << "  \"architecture\": \"MLP [128 -> 256 -> 128 -> 16] with LayerNorm + GELU + Softmax\",\n";
    json << "  \"fp32_weight_bytes\": " << model.fp32_weight_bytes() << ",\n";
    json << "  \"int8_weight_bytes\": " << model.int8_weight_bytes() << ",\n";
    json << "  \"weight_compression_ratio\": "
         << (static_cast<double>(model.fp32_weight_bytes()) / static_cast<double>(model.int8_weight_bytes())) << ",\n";
    json << std::setprecision(6);
    json << "  \"fp32_vs_int8_max_abs_probability_diff\": " << max_diff << ",\n";
    json << std::setprecision(3);
    json << "  \"benchmarks\": [\n";
    for (size_t idx = 0; idx < results.size(); ++idx) {
        const auto& r = results[idx];
        json << "    {\n";
        json << "      \"mode\": \"" << r.mode << "\",\n";
        json << "      \"precision\": \"" << r.precision << "\",\n";
        json << "      \"batch_size\": " << r.batch_size << ",\n";
        json << "      \"iterations\": " << r.iterations << ",\n";
        json << "      \"total_items\": " << r.total_items << ",\n";
        json << "      \"p50_latency_us\": " << r.p50_us << ",\n";
        json << "      \"p95_latency_us\": " << r.p95_us << ",\n";
        json << "      \"p99_latency_us\": " << r.p99_us << ",\n";
        json << "      \"mean_latency_us\": " << r.mean_us << ",\n";
        json << "      \"throughput_items_per_sec\": " << r.items_per_sec << "\n";
        json << "    }" << (idx + 1 < results.size() ? "," : "") << "\n";
    }
    json << "  ]\n";
    json << "}\n";

    if (!report_path.empty()) {
        std::ofstream out(report_path, std::ios::trunc);
        out << json.str();
    }
    std::cout << json.str();
    return 0;
}
