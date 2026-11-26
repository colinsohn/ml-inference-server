import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

from python.exporter import (
    ACT_GELU,
    ACT_NONE,
    ACT_RELU,
    export_mlis_model,
    python_golden_forward,
)


class ExporterTests(unittest.TestCase):
    def test_export_binary_format_and_cpp_engine_load(self):
        layers = [
            {
                "in_features": 8,
                "out_features": 16,
                "activation": ACT_GELU,
                "use_layer_norm": True,
                "weights": [math.sin(i * 0.1) * 0.3 for i in range(8 * 16)],
                "bias": [0.01 * i for i in range(16)],
            },
            {
                "in_features": 16,
                "out_features": 4,
                "activation": ACT_NONE,
                "use_layer_norm": False,
                "weights": [math.cos(i * 0.2) * 0.25 for i in range(16 * 4)],
                "bias": [-0.02 * i for i in range(4)],
            },
        ]
        sample_batch = [[math.cos((r + c) * 0.15) for c in range(8)] for r in range(2)]
        golden = python_golden_forward(layers, sample_batch, final_softmax=True)
        for row in golden:
            self.assertAlmostEqual(sum(row), 1.0, places=6)

        repo_root = Path(__file__).resolve().parents[1]
        demo_bin = repo_root / "build" / "main_demo"
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "exported.mlis"
            meta = export_mlis_model(layers, model_path, final_softmax=True)
            self.assertEqual(meta["num_layers"], 2)
            self.assertTrue(model_path.exists())

            if demo_bin.exists():
                proc = subprocess.run(
                    [str(demo_bin), str(model_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                payload = json.loads(proc.stdout)
                self.assertEqual(payload["layers"], 2)
                self.assertEqual(payload["input_dim"], 8)
                self.assertEqual(payload["output_dim"], 4)
                self.assertLess(payload["fp32_vs_int8_max_abs_diff"], 0.05)

    def test_invalid_layer_dimensions_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                export_mlis_model([], Path(tmp) / "bad.mlis")
            with self.assertRaises(ValueError):
                export_mlis_model(
                    [{"in_features": 4, "out_features": 4, "weights": [1.0] * 10, "bias": [0.0] * 4}],
                    Path(tmp) / "bad2.mlis",
                )


if __name__ == "__main__":
    unittest.main()
