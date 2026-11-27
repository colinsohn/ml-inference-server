# Work Log (`ml-inference-server`)

## November 26, 2026 — Day 1: C++17 Tensor & Fused Operator Engine, Python/PyTorch/ONNX Exporter, Numerical Parity Suite & Latency Benchmark

- Initialized `ml-inference-server` (Active New Project Slot 2/2) with `ARCHITECTURE.md`, 21-day `ROADMAP.md`, `AGENTS.md`, `WORKLOG.md`, `Makefile`, `Dockerfile`, and `.github/workflows/ci.yml`.
- Implemented C++17 inference runtime (`include/ml_inference/engine.hpp`, `src/engine.cpp`, `src/main.cpp`) featuring contiguous row-major `Tensor`, zero-allocation `ActivationArena`, 4-way loop-unrolled `op_matmul_bias`, `op_layer_norm_inplace`, `op_relu_inplace`, exact `erf`-based `op_gelu_inplace`, numerically stable `op_softmax_inplace`, fused `forward_fused_into` vs. `forward_unfused` execution paths, and FNV-1a checksum verification on binary model load.
- Implemented Python/PyTorch/ONNX exporter and IEEE-754 float32 reference evaluator (`python/export_model.py`), C++17 unit test suite (`tests/test_engine.cpp`), Python-to-C++ numerical parity integration test suite (`tests/test_parity.py`), and unified build/test/benchmark runner (`scripts/build_and_test.py`).

## November 27, 2026 — Days 2–3: ONNX Static Shape Inference Verifier & Arena Zero-Allocation Counters

- Added `InferenceEngine::verify_onnx_json_topology` (`include/ml_inference/engine.hpp`, `src/engine.cpp`) verifying layer input/output dimensions, terminal `Softmax` node presence, and FNV-1a weight checksum parity against `artifacts/fraud_scorer_v1.onnx.json`.
- Added `ActivationArena` counters (`peak_used_floats`, `reset_count`, `allocation_calls`, `heap_reallocations`) and a 5th C++17 unit test suite (`tests/test_engine.cpp`) verifying zero heap reallocations across repeated `forward_fused_into` passes.
