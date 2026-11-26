# ML Inference Server

A high-performance **C++20 neural network inference engine and dynamic batching scheduler** with a **Python model weight serializer and cross-language parity verifier**.

---

## Implemented Features

- **C++20 Tensor & Operator Core ([`include/inference_engine.hpp`](include/inference_engine.hpp), [`src/inference_engine.cpp`](src/inference_engine.cpp))**:
  - Contiguous row-major `Tensor`, `LinearLayer` (`GEMM` with bias), fused `LayerNorm`, `ReLU`, exact-tanh `GELU`, and overflow-safe `Softmax`.
  - **Symmetric Per-Channel `INT8` Weight Quantization**: Automatically computes per-output-channel scales $s_o = \max_i |W_{o,i}| / 127.0$, compressing weights by **3.88x** (`540,672` bytes $\to$ `139,264` bytes on `[128 -> 256 -> 128 -> 16]`) while keeping maximum post-softmax probability deviation under `0.008`.
  - **Checksum-Guarded Binary Model Format (`MLISv001`)**: 64-bit FNV-1a integrity verification rejects truncated or corrupted weight files on load.
- **Thread-Safe Dynamic Batching ([`DynamicBatchScheduler`](include/inference_engine.hpp))**:
  - Coalesces concurrent single-sample and micro-batch requests up to `max_batch_size` within a configurable microsecond timeout window and enforces bounded queue backpressure (`OverloadError`).
- **Python Weight Exporter & Golden Reference Verifier ([`python/exporter.py`](python/exporter.py), [`python/test_exporter.py`](python/test_exporter.py))**:
  - Exports multi-layer weights into the `.mlis` binary layout and verifies round-trip loading and inference through the compiled C++20 engine.

---

## Documentation & Project Files

- Technical Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- 21-Day Development Roadmap: [`ROADMAP.md`](ROADMAP.md)
- Persistent Agent Guidelines: [`AGENTS.md`](AGENTS.md)
- Engineering Work Log: [`WORKLOG.md`](WORKLOG.md)
- Measured Benchmark Report: [`reports/inference_benchmark.json`](reports/inference_benchmark.json)

---

## Build, Test & Benchmark

Requires a C++20 compiler (`clang++` or `g++`) and Python 3.9+:

```sh
# 1. Compile C++20 engine, unit tests, demo, and benchmark with -O3 -Wall -Wextra -Werror
make all

# 2. Run C++20 unit and multi-threaded scheduler tests
make test

# 3. Run Python exporter and cross-language C++ binary parity tests
python3 -B -m unittest discover -s python -p "test_*.py" -v

# 4. Run CLI demonstration
make demo

# 5. Run latency and throughput benchmark
./build/benchmark --report reports/inference_benchmark.json
```
