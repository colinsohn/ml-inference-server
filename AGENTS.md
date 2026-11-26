# Agent Guidelines (`AGENTS.md`) — ML Inference Server

- **Repository Owner**: **Autonomous Portfolio Engineering** (`colinsohn/ml-inference-server`).
- **Commit Identity**: `Colin Sohn <183441695+colinsohn@users.noreply.github.com>`.
- **Verification Before Commit**:
  1. `make clean && make all` (compiles with `-std=c++20 -O3 -Wall -Wextra -Werror -pthread`)
  2. `make test` (runs C++20 unit and multi-threaded scheduler tests)
  3. `make demo` (runs C++20 CLI demonstration)
  4. `./build/benchmark --report reports/inference_benchmark.json`
  5. `python3 -B -m unittest discover -s python -p "test_*.py" -v` (verifies Python exporter and C++ binary loading parity)
- **Guardrails**:
  - Never commit compiled binaries (`build/`), temporary `.mlis` weight artifacts, or secrets.
  - After every `git push origin main`, verify `git ls-remote --heads origin refs/heads/main` matches `git rev-parse HEAD` and confirm GitHub Actions CI passes (`gh run list`).
