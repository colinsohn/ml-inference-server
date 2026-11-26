# Agent Guidelines (`ml-inference-server`)

Updated November 26, 2026 (Pacific).

1. **Repository Ownership**: Managed by **Autonomous Portfolio Engineering** under `/Users/colinsohn/Documents/ChatGPT/Career/portfolio-build/ml-inference-server`. Always check `git status -sb` and `WORKLOG.md` before starting a milestone.
2. **Unified Build, Test & Benchmark Command**:
   ```sh
   python3 -B scripts/build_and_test.py
   ```
   This compiles the C++17 engine and unit tests (`clang++`/`g++ -std=c++17 -O3`), executes the C++ unit tests, runs the Python/PyTorch/ONNX model exporter, verifies Python-to-C++ numerical parity (`< 1e-6`), and updates `reports/benchmark_day01.json`.
3. **Never Commit Compiled Binaries**: Keep `build/`, `*.o`, and `artifacts/*.bin` out of Git via `.gitignore`; commit source code, JSON specs, tests, and measured JSON benchmark summaries under `reports/`.
4. **Commit & Push Discipline**:
   - Commit as `Colin Sohn <183441695+colinsohn@users.noreply.github.com>`.
   - Verify remote `origin/main` matches local `HEAD` and GitHub Actions CI passes (`gh run list`).
