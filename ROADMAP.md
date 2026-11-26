# 21-Day Development Roadmap — ML Inference Server

| Day | Milestone | Status |
| --- | --- | --- |
| **Day 1** | C++20 tensor core (`Tensor`), numerically stable `LayerNorm`, `ReLU`, `GELU`, and `Softmax`, `FP32` and symmetric per-channel `INT8` quantized `LinearLayer`, and FNV-1a checksum-guarded `.mlis` binary serialization. | **Completed (Nov 26, 2026)** |
| **Day 2** | Thread-safe `DynamicBatchScheduler` with microsecond coalescing and overload backpressure, Python weight exporter and cross-language parity tests (`python/exporter.py`), C++20 benchmark CLI (`reports/inference_benchmark.json`), and GitHub Actions CI. | **Completed (Nov 26, 2026)** |
| **Day 3** | Cache-blocked / SIMD-tiled (`AVX2` / ARM `NEON`) matrix multiplication kernels with benchmark comparison against baseline loops. | Pending |
| **Day 4** | Dynamic activation quantization (`INT8 x INT8 -> INT32` accumulation with per-channel rescaling) and calibration histogram collector. | Pending |
| **Day 5** | ONNX protobuf / JSON graph importer supporting general DAG topologies and residual skip connections (`Add` + `LayerNorm`). | Pending |
| **Day 6** | Multi-head self-attention (`QKV` projection, scaled dot-product attention, causal masking, and KV-cache arena allocator). | Pending |
| **Day 7** | Custom slab/arena memory pool eliminating per-request heap allocations (`std::vector` malloc/free) along the hot inference path. | Pending |
| **Day 8** | Multi-worker thread-pool execution engine for parallel intra-batch and inter-batch scheduling. | Pending |
| **Day 9** | HTTP/JSON and binary TCP inference server endpoint (`/v1/models/{name}:predict`, `/healthz`, `/metrics`) using non-blocking sockets. | Pending |
| **Day 10** | Priority-aware request scheduling and per-request deadline propagation (cancelling expired queued requests before execution). | Pending |
| **Day 11** | Zero-copy `mmap` weight loading with page-aligned tensor offsets in `MLISv002` format. | Pending |
| **Day 12** | Operator fusion pass (fusing `Linear + Bias + ReLU/GELU` and `Add + LayerNorm` into single-pass memory kernels). | Pending |
| **Day 13** | Hot-reload model registry supporting atomic version swaps and shadow A/B traffic splitting without dropping in-flight batches. | Pending |
| **Day 14** | Python PyTorch (`torch.nn.Module`) and ONNX Runtime end-to-end accuracy and latency parity suite. | Pending |
| **Day 15** | Tail-latency (`p99` / `p99.9`) load generator under Poisson and bursty arrival distributions across batch timeouts. | Pending |
| **Day 16** | Token-streaming autoregressive generation loop with KV-cache reuse and top-$k$ / nucleus ($p$) sampling. | Pending |
| **Day 17** | Speculative / continuous batching scheduler for variable-length sequence requests. | Pending |
| **Day 18** | AddressSanitizer (`-fsanitize=address,undefined`) and ThreadSanitizer (`-fsanitize=thread`) CI regression gates. | Pending |
| **Day 19** | Multi-stage production `Dockerfile` and containerized latency benchmark. | Pending |
| **Day 20** | Fresh-clone build verification and interactive C++/Python CLI walkthrough. | Pending |
| **Day 21** | Final release verification, roof-line performance analysis, and portfolio summary. | Pending |
