#include "ml_inference/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

double percentile(const std::vector<double>& sorted_us, double pct) {
    if (sorted_us.empty()) {
        return 0.0;
    }
    if (sorted_us.size() == 1) {
        return sorted_us.front();
    }
    const double rank = (pct / 100.0) * static_cast<double>(sorted_us.size() - 1);
    const auto low = static_cast<std::size_t>(std::floor(rank));
    const auto high = static_cast<std::size_t>(std::ceil(rank));
    const double weight = rank - static_cast<double>(low);
    return sorted_us[low] * (1.0 - weight) + sorted_us[high] * weight;
}

int run_predict_binary(const std::string& model_path, const std::string& input_bin_path, const std::string& output_json_path) {
    auto engine = ml_inference::InferenceEngine::load_from_binary(model_path);

    std::ifstream in(input_bin_path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open input tensor binary: " << input_bin_path << "\n";
        return 1;
    }
    uint32_t rows = 0;
    uint32_t cols = 0;
    in.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
    if (!in || cols != engine.input_dim()) {
        std::cerr << "Invalid input binary dimensions\n";
        return 1;
    }

    ml_inference::Tensor input(rows, cols);
    in.read(reinterpret_cast<char*>(input.data.data()), input.size() * sizeof(float));

    ml_inference::Tensor fused_out = engine.forward_fused(input);
    ml_inference::Tensor unfused_out = engine.forward_unfused(input);

    double max_fused_vs_unfused_diff = 0.0;
    for (std::size_t i = 0; i < fused_out.size(); ++i) {
        max_fused_vs_unfused_diff = std::max<double>(
            max_fused_vs_unfused_diff,
            std::abs(static_cast<double>(fused_out.data[i]) - static_cast<double>(unfused_out.data[i]))
        );
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(8);
    oss << "{\n";
    oss << "  \"rows\": " << fused_out.rows << ",\n";
    oss << "  \"cols\": " << fused_out.cols << ",\n";
    oss << "  \"model_checksum\": " << engine.model_checksum() << ",\n";
    oss << "  \"max_fused_vs_unfused_diff\": " << max_fused_vs_unfused_diff << ",\n";
    oss << "  \"predictions\": [\n";
    for (std::size_t r = 0; r < fused_out.rows; ++r) {
        oss << "    [";
        for (std::size_t c = 0; c < fused_out.cols; ++c) {
            oss << fused_out.at(r, c);
            if (c + 1 < fused_out.cols) {
                oss << ", ";
            }
        }
        oss << "]" << (r + 1 < fused_out.rows ? "," : "") << "\n";
    }
    oss << "  ]\n}\n";

    if (!output_json_path.empty()) {
        std::ofstream out(output_json_path);
        out << oss.str();
    } else {
        std::cout << oss.str();
    }
    return 0;
}

int run_benchmark(const std::string& model_path, int iterations, const std::string& output_json_path) {
    auto engine = ml_inference::InferenceEngine::load_from_binary(model_path);
    const std::vector<std::size_t> batch_sizes = {1, 8, 16, 32, 64};

    std::mt19937 rng(20261126);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "{\n";
    oss << "  \"benchmark\": \"ml-inference-server-cpp17-day01\",\n";
    oss << "  \"model\": {\n";
    oss << "    \"num_layers\": " << engine.num_layers() << ",\n";
    oss << "    \"input_dim\": " << engine.input_dim() << ",\n";
    oss << "    \"output_dim\": " << engine.output_dim() << ",\n";
    oss << "    \"checksum\": " << engine.model_checksum() << "\n";
    oss << "  },\n";
    oss << "  \"iterations_per_config\": " << iterations << ",\n";
    oss << "  \"results\": [\n";

    for (std::size_t b_idx = 0; b_idx < batch_sizes.size(); ++b_idx) {
        const std::size_t bs = batch_sizes[b_idx];
        ml_inference::Tensor input(bs, engine.input_dim());
        for (float& v : input.data) {
            v = dist(rng);
        }

        ml_inference::ActivationArena arena(262144);
        ml_inference::Tensor output(bs, engine.output_dim());

        // Warmup
        for (int w = 0; w < 200; ++w) {
            engine.forward_fused_into(input, output, arena);
        }

        std::vector<double> fused_lat_us;
        fused_lat_us.reserve(iterations);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            auto s = std::chrono::steady_clock::now();
            engine.forward_fused_into(input, output, arena);
            auto e = std::chrono::steady_clock::now();
            fused_lat_us.push_back(
                std::chrono::duration<double, std::micro>(e - s).count()
            );
        }
        auto t1 = std::chrono::steady_clock::now();
        const double fused_total_sec = std::chrono::duration<double>(t1 - t0).count();

        std::vector<double> unfused_lat_us;
        unfused_lat_us.reserve(iterations);
        auto u0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            auto s = std::chrono::steady_clock::now();
            auto u_out = engine.forward_unfused(input);
            auto e = std::chrono::steady_clock::now();
            if (u_out.data[0] < -1.0f) {
                std::cerr << "unreachable\n";
            }
            unfused_lat_us.push_back(
                std::chrono::duration<double, std::micro>(e - s).count()
            );
        }
        auto u1 = std::chrono::steady_clock::now();
        const double unfused_total_sec = std::chrono::duration<double>(u1 - u0).count();

        std::sort(fused_lat_us.begin(), fused_lat_us.end());
        std::sort(unfused_lat_us.begin(), unfused_lat_us.end());

        const double fused_mean = std::accumulate(fused_lat_us.begin(), fused_lat_us.end(), 0.0) / fused_lat_us.size();
        const double unfused_mean = std::accumulate(unfused_lat_us.begin(), unfused_lat_us.end(), 0.0) / unfused_lat_us.size();
        const double total_items = static_cast<double>(iterations) * static_cast<double>(bs);
        const double fused_items_per_sec = total_items / std::max(fused_total_sec, 1e-9);
        const double unfused_items_per_sec = total_items / std::max(unfused_total_sec, 1e-9);
        const double speedup = unfused_mean / std::max(fused_mean, 1e-9);

        oss << "    {\n";
        oss << "      \"batch_size\": " << bs << ",\n";
        oss << "      \"fused_p50_us\": " << percentile(fused_lat_us, 50.0) << ",\n";
        oss << "      \"fused_p95_us\": " << percentile(fused_lat_us, 95.0) << ",\n";
        oss << "      \"fused_p99_us\": " << percentile(fused_lat_us, 99.0) << ",\n";
        oss << "      \"fused_mean_us\": " << fused_mean << ",\n";
        oss << "      \"fused_items_per_sec\": " << fused_items_per_sec << ",\n";
        oss << "      \"unfused_p50_us\": " << percentile(unfused_lat_us, 50.0) << ",\n";
        oss << "      \"unfused_p95_us\": " << percentile(unfused_lat_us, 95.0) << ",\n";
        oss << "      \"unfused_mean_us\": " << unfused_mean << ",\n";
        oss << "      \"unfused_items_per_sec\": " << unfused_items_per_sec << ",\n";
        oss << "      \"fusion_speedup_ratio\": " << speedup << ",\n";
        oss << "      \"arena_floats_used\": " << arena.used_floats() << "\n";
        oss << "    }" << (b_idx + 1 < batch_sizes.size() ? "," : "") << "\n";
    }

    oss << "  ]\n}\n";

    if (!output_json_path.empty()) {
        std::ofstream out(output_json_path);
        out << oss.str();
    }
    std::cout << oss.str();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage:\n"
                  << "  ml-infer-cli predict <model.bin> <input.bin> [output.json]\n"
                  << "  ml-infer-cli benchmark <model.bin> [iterations] [output.json]\n";
        return 1;
    }

    const std::string mode = argv[1];
    const std::string model_path = argv[2];

    if (mode == "predict") {
        if (argc < 4) {
            std::cerr << "predict mode requires <input.bin>\n";
            return 1;
        }
        const std::string input_bin = argv[3];
        const std::string output_json = argc >= 5 ? argv[4] : "";
        return run_predict_binary(model_path, input_bin, output_json);
    }

    if (mode == "benchmark") {
        const int iterations = argc >= 4 ? std::stoi(argv[3]) : 2500;
        const std::string output_json = argc >= 5 ? argv[4] : "";
        return run_benchmark(model_path, iterations, output_json);
    }

    std::cerr << "Unknown mode: " << mode << "\n";
    return 1;
}
