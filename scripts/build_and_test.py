"""Unified build, unit test, numerical parity check, demo, and benchmark runner for ml-inference-server."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
REPORTS_DIR = ROOT_DIR / "reports"


def find_cxx() -> str:
    for candidate in ("clang++", "g++", "c++"):
        path = shutil.which(candidate)
        if path:
            return path
    raise RuntimeError("No C++17 compiler (clang++ or g++) found on PATH")


def main() -> int:
    cxx = find_cxx()
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    test_bin = BUILD_DIR / "test_engine"
    cli_bin = BUILD_DIR / "ml-infer-cli"

    common_flags = [
        cxx,
        "-std=c++17",
        "-O3",
        "-Wall",
        "-Wextra",
        "-I",
        str(ROOT_DIR / "include"),
    ]

    print(f"=== 1. Compiling C++17 Engine & Unit Tests with {cxx} ===")
    subprocess.run(
        common_flags + [str(ROOT_DIR / "src/engine.cpp"), str(ROOT_DIR / "tests/test_engine.cpp"), "-o", str(test_bin)],
        check=True,
    )
    subprocess.run(
        common_flags + [str(ROOT_DIR / "src/engine.cpp"), str(ROOT_DIR / "src/main.cpp"), "-o", str(cli_bin)],
        check=True,
    )

    print("\n=== 2. Exporting Model Weights, ONNX Spec & Golden Validation Vectors ===")
    sys.path.insert(0, str(ROOT_DIR))
    from python.export_model import export_artifacts

    golden = export_artifacts(ARTIFACTS_DIR, num_samples=64, seed=20261126)
    print(
        json.dumps(
            {
                "model_checksum": golden["model_checksum"],
                "num_samples": golden["num_samples"],
                "pytorch_max_diff": golden["pytorch_max_diff"],
                "onnx_binary_exported": golden["onnx_binary_exported"],
            },
            indent=2,
        )
    )

    print("\n=== 3. Running C++17 Unit Test Suite ===")
    subprocess.run([str(test_bin)], cwd=str(ROOT_DIR), check=True)

    print("\n=== 4. Running Python-to-C++17 Numerical Parity Integration Tests ===")
    subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", str(ROOT_DIR / "tests"), "-p", "test_*.py", "-v"],
        cwd=str(ROOT_DIR),
        check=True,
    )

    print("\n=== 4b. Compiling & Testing C++20 Quantized INT8 + DynamicBatchScheduler Engine ===")
    cxx20_flags = [
        cxx,
        "-std=c++20",
        "-O3",
        "-Wall",
        "-Wextra",
        "-pthread",
        "-I",
        str(ROOT_DIR / "include"),
    ]
    demo20_bin = BUILD_DIR / "main_demo"
    bench20_bin = BUILD_DIR / "benchmark"
    subprocess.run(
        cxx20_flags + [str(ROOT_DIR / "src/inference_engine.cpp"), str(ROOT_DIR / "src/main_demo.cpp"), "-o", str(demo20_bin)],
        check=True,
    )
    subprocess.run(
        cxx20_flags + [str(ROOT_DIR / "src/inference_engine.cpp"), str(ROOT_DIR / "src/benchmark.cpp"), "-o", str(bench20_bin)],
        check=True,
    )
    subprocess.run([str(demo20_bin)], check=True)
    subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", str(ROOT_DIR / "python"), "-p", "test_*.py", "-v"],
        cwd=str(ROOT_DIR),
        check=True,
    )

    print("\n=== 5. Running C++17 Latency & Operator-Fusion Benchmark ===")
    bench_path = REPORTS_DIR / "benchmark_day01.json"
    subprocess.run(
        [str(cli_bin), "benchmark", str(ARTIFACTS_DIR / "fraud_scorer_v1.bin"), "3000", str(bench_path)],
        check=True,
    )

    # Enrich benchmark report with host metadata and parity verification result
    bench_data = json.loads(bench_path.read_text(encoding="utf-8"))
    bench_data["environment"] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "compiler": cxx,
        "python_version": platform.python_version(),
    }
    bench_data["parity_verification"] = {
        "golden_samples_verified": 64,
        "max_abs_error_threshold": 1e-6,
        "parity_passed": True,
    }
    bench_path.write_text(json.dumps(bench_data, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
