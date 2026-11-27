# 21-Day Development Roadmap (`ml-inference-server`)

Timeline: **November 26 – December 16, 2026** (Active New Project Slot 2/2).

| Day | Date | Status | Milestone | Verification Command |
| --- | --- | :---: | --- | --- |
| **Day 1** | Nov 26 | `[x]` | **C++17 Tensor & Fused Operator Engine, Python/PyTorch/ONNX Exporter, Numerical Parity Suite & Latency Benchmark**: Contiguous `Tensor` & zero-allocation `ActivationArena`, 4-way unrolled `MatMul`, fused `Linear+Bias+LayerNorm+GELU/ReLU`, stable `Softmax`, FNV-1a checksummed binary loader, Python/PyTorch/ONNX exporter (`python/export_model.py`), C++ unit tests (`tests/test_engine.cpp`), Python-to-C++ parity tests (`< 1e-6` max error), and batch latency/fusion benchmark (`reports/benchmark_day01.json`). | `python3 -B scripts/build_and_test.py` |
| **Day 2** | Nov 27 | `[x]` | **ONNX Graph Importer & Static Shape Inference**: Parse ONNX node topologies (`Gemm`, `MatMul`, `Add`, `LayerNorm`, `Relu`, `Gelu`, `Softmax`) and validate tensor shapes at load time (`verify_onnx_json_topology`). | `python3 -B scripts/build_and_test.py` |
| **Day 3** | Nov 28 | `[x]` | **Thread-Local Arena Pool & Allocation Counters**: Verify zero heap allocations during steady-state inference via custom allocator counters (`reset_count`, `allocation_calls`, `heap_reallocations`). | `python3 -B scripts/build_and_test.py` |
| **Day 4** | Nov 29 | `[ ]` | **Ahead-of-Time Graph Optimizer & Pattern Fusion Pass**: Automated graph rewrite pass merging sequential `MatMul + Add + Activation` nodes. | `python3 -B scripts/build_and_test.py` |
| **Day 5** | Nov 30 | `[ ]` | **Dynamic Batching Scheduler with Bounded Queue Deadlines**: Coalesce concurrent single-item requests into batch tensors under `max_batch_size` and `max_queue_delay_us`. | `python3 -B scripts/build_and_test.py` |
| **Day 6** | Dec 1 | `[ ]` | **Multi-Threaded Worker Pool & Request Cancellation**: Bounded C++ thread pool with per-worker arenas and deadline cancellation. | `python3 -B scripts/build_and_test.py` |
| **Day 7** | Dec 2 | `[ ]` | **Hardware SIMD Micro-Kernels (ARM NEON / AVX2 + Scalar Fallback)**: Register-tiled `float32` matrix multiplication with runtime/compile-time SIMD dispatch. | `python3 -B scripts/build_and_test.py` |
| **Day 8** | Dec 3 | `[ ]` | **INT8 Symmetric Weight Quantization & Calibration**: Per-channel `INT8` weight quantization and C++ `LinearInt8` kernel with accuracy-vs-latency benchmark. | `python3 -B scripts/build_and_test.py` |
| **Day 9** | Dec 4 | `[ ]` | **Cross-Runtime PyTorch & ONNX Runtime 1,000-Sample Parity Suite**: Automated statistical parity suite comparing FP32 and INT8 error distributions. | `python3 -B scripts/build_and_test.py` |
| **Day 10** | Dec 5 | `[ ]` | **HTTP/JSON & Binary Framing C++ Inference Server**: Network serving endpoints (`POST /v1/models/{model}:predict`, `GET /healthz`, `GET /metrics`). | `python3 -B scripts/build_and_test.py` |
| **Day 11** | Dec 6 | `[ ]` | **Multi-Model Registry & Atomic Zero-Downtime Hot-Reload**: Lock-free/atomic `std::shared_ptr<const InferenceEngine>` model version swaps. | `python3 -B scripts/build_and_test.py` |
| **Day 12** | Dec 7 | `[ ]` | **Scaled Dot-Product Self-Attention & KV-Cache Operator**: Transformer attention block with pre-allocated sequence KV-cache in C++17. | `python3 -B scripts/build_and_test.py` |
| **Day 13** | Dec 8 | `[ ]` | **Admission Control, Overload Shedding & SLA Enforcement**: Queue-delay estimator shedding requests that cannot meet client timeout budgets. | `python3 -B scripts/build_and_test.py` |
| **Day 14** | Dec 9 | `[ ]` | **Per-Operator Nanosecond Profiler & Chrome Trace Exporter**: Operator-level latency breakdown and `trace.json` timeline generator. | `python3 -B scripts/build_and_test.py` |
| **Day 15** | Dec 10 | `[ ]` | **ASan, UBSan & TSan Sanitizer Verification**: Full C++ unit and concurrency suite under Address, UndefinedBehavior, and Thread Sanitizers. | `python3 -B scripts/build_and_test.py` |
| **Day 16** | Dec 11 | `[ ]` | **Python Client SDK & Closed-Loop Load Generator**: Measure SLA-compliant goodput vs. concurrency across batching policies. | `python3 -B scripts/build_and_test.py` |
| **Day 17** | Dec 12 | `[ ]` | **Multi-Stage Container Build & Cloud Run Scale-to-Zero Guide**: Minimal runtime container image and `DEPLOYMENT.md` cost review. | `python3 -B scripts/build_and_test.py` |
| **Day 18** | Dec 13 | `[ ]` | **Reproducible Colab Export & Quantization Notebook**: End-to-end PyTorch training, ONNX export, INT8 calibration, and parity visualization notebook. | `python3 -B scripts/build_and_test.py` |
| **Day 19** | Dec 14 | `[ ]` | **Batch Size vs. p99 Latency Pareto Frontier Benchmark**: Full sweep across batch sizes, threads, and precision (`reports/pareto_benchmark.json`). | `python3 -B scripts/build_and_test.py` |
| **Day 20** | Dec 15 | `[ ]` | **Fresh-Clone Reproduction & Interactive Walkthrough**: Single-command fresh-clone build, test, and server demo. | `python3 -B scripts/build_and_test.py` |
| **Day 21** | Dec 16 | `[ ]` | **Final Release Audit & Capacity Transition**: Verify all acceptance criteria and release concurrency slot. | `python3 -B scripts/build_and_test.py` |
