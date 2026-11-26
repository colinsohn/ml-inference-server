# Work Log — ML Inference Server

## November 26, 2026 (Pacific) — Project Initialization & Milestones 1–2

- Established system architecture (`ARCHITECTURE.md`), 21-day development roadmap (`ROADMAP.md`), persistent agent instructions (`AGENTS.md`), `Makefile`, and GitHub Actions CI workflow (`.github/workflows/ci.yml`).
- Implemented the C++20 inference engine (`include/inference_engine.hpp`, `src/inference_engine.cpp`):
  - Row-major `Tensor`, numerically stable `LayerNorm`, `ReLU`, `GELU`, and `Softmax`, `FP32` and symmetric per-output-channel `INT8` quantized `LinearLayer` (~3.88x weight compression), and FNV-1a 64-bit checksum-guarded `.mlis` binary serialization.
  - Thread-safe `DynamicBatchScheduler` coalescing concurrent requests into batched forward passes with bounded queue backpressure (`OverloadError`).
- Implemented the Python weight exporter and cross-language golden parity test suite (`python/exporter.py`, `python/test_exporter.py`), C++20 unit test suite (`tests/test_engine.cpp`), CLI demo (`src/main_demo.cpp`), and latency/throughput benchmark (`src/benchmark.cpp`, `reports/inference_benchmark.json`).
