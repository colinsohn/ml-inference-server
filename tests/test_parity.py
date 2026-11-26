"""End-to-end Python-to-C++17 numerical parity and corruption-rejection integration tests."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from python.export_model import export_artifacts

ROOT_DIR = Path(__file__).resolve().parent.parent
BIN_DIR = ROOT_DIR / "build"
CLI_BIN = BIN_DIR / "ml-infer-cli"


class TestCppPythonParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        BIN_DIR.mkdir(parents=True, exist_ok=True)
        if not CLI_BIN.exists():
            cxx = shutil.which("clang++") or shutil.which("g++") or "c++"
            subprocess.run(
                [
                    cxx,
                    "-std=c++17",
                    "-O3",
                    "-Wall",
                    "-Wextra",
                    "-I",
                    str(ROOT_DIR / "include"),
                    str(ROOT_DIR / "src/engine.cpp"),
                    str(ROOT_DIR / "src/main.cpp"),
                    "-o",
                    str(CLI_BIN),
                ],
                check=True,
            )

    def test_golden_64_batch_numerical_parity_under_1e6(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            art_dir = Path(tmp)
            golden = export_artifacts(art_dir, num_samples=64, seed=20261126)
            out_json = art_dir / "cpp_predictions.json"

            proc = subprocess.run(
                [
                    str(CLI_BIN),
                    "predict",
                    str(art_dir / "fraud_scorer_v1.bin"),
                    str(art_dir / "golden_inputs.bin"),
                    str(out_json),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(proc.returncode, 0)
            cpp_res = json.loads(out_json.read_text(encoding="utf-8"))

            self.assertEqual(cpp_res["rows"], 64)
            self.assertEqual(cpp_res["cols"], 4)
            self.assertEqual(cpp_res["model_checksum"], golden["model_checksum"])
            self.assertLess(cpp_res["max_fused_vs_unfused_diff"], 1e-6)

            max_abs_err = 0.0
            for expected_row, actual_row in zip(golden["expected_outputs"], cpp_res["predictions"]):
                # Every row must sum to 1.0 (valid probability distribution)
                self.assertAlmostEqual(sum(actual_row), 1.0, places=5)
                for exp_v, act_v in zip(expected_row, actual_row):
                    max_abs_err = max(max_abs_err, abs(float(exp_v) - float(act_v)))

            self.assertLess(
                max_abs_err,
                1e-6,
                f"Python reference vs. C++17 engine max abs error {max_abs_err} exceeded 1e-6",
            )

    def test_corrupted_model_binary_checksum_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            art_dir = Path(tmp)
            export_artifacts(art_dir, num_samples=4, seed=20261126)
            model_path = art_dir / "fraud_scorer_v1.bin"
            raw = bytearray(model_path.read_bytes())
            # Flip a byte inside the first layer's float32 weights buffer
            raw[40] ^= 0xFF
            model_path.write_bytes(bytes(raw))

            proc = subprocess.run(
                [
                    str(CLI_BIN),
                    "predict",
                    str(model_path),
                    str(art_dir / "golden_inputs.bin"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
