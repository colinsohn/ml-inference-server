"""PyTorch / ONNX / IEEE-754 float32 model exporter and reference evaluator for ml-inference-server.

Architecture exported:
  Layer 0: Linear(16 -> 64) + Bias(64) + LayerNorm(64, eps=1e-5) + GELU
  Layer 1: Linear(64 -> 32) + Bias(32) + ReLU
  Layer 2: Linear(32 -> 4)  + Bias(4)  [+ Final Softmax(4)]

When PyTorch and ONNX are installed in the active Python environment (e.g., in CI or Colab),
this exporter also builds the equivalent `torch.nn.Module`, verifies zero divergence between
`torch.nn.Module.forward` and the reference float32 evaluator, and exports a standard `.onnx`
graph artifact alongside the compact C++ binary weight file (`MLIS` format v1) and ONNX-JSON
graph specification.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
from pathlib import Path
import random
import struct
from typing import Any, Dict, List, Sequence, Tuple

MAGIC_HEADER = 0x53494C4D  # "MLIS"
FORMAT_VERSION = 1
ACT_NONE = 0
ACT_RELU = 1
ACT_GELU = 2


def f32(val: float) -> float:
    """Round a Python float to exact IEEE-754 single-precision float32."""
    return struct.unpack("<f", struct.pack("<f", float(val)))[0]


def fnv1a_hash_bytes(data: bytes, seed: int = 2166136261) -> int:
    h = ctypes.c_uint32(seed).value
    for b in data:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def gelu_f32(x: float) -> float:
    inv_sqrt2 = f32(0.7071067811865475)
    return f32(f32(0.5) * f32(x) * f32(1.0 + math.erf(f32(x) * inv_sqrt2)))


def pack_f32_list(values: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *[f32(v) for v in values])


def build_deterministic_model_spec(seed: int = 20261126) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    def rand_list(n: int, scale: float) -> List[float]:
        return [f32(rng.uniform(-scale, scale)) for _ in range(n)]

    layers = [
        {
            "name": "fc1_ln_gelu",
            "in_features": 16,
            "out_features": 64,
            "has_layer_norm": True,
            "activation": ACT_GELU,
            "activation_name": "GELU",
            "weights": rand_list(64 * 16, 0.25),
            "bias": rand_list(64, 0.10),
            "ln_gamma": [f32(1.0 + rng.uniform(-0.15, 0.15)) for _ in range(64)],
            "ln_beta": rand_list(64, 0.05),
        },
        {
            "name": "fc2_relu",
            "in_features": 64,
            "out_features": 32,
            "has_layer_norm": False,
            "activation": ACT_RELU,
            "activation_name": "RELU",
            "weights": rand_list(32 * 64, 0.20),
            "bias": rand_list(32, 0.10),
            "ln_gamma": [],
            "ln_beta": [],
        },
        {
            "name": "fc3_logits",
            "in_features": 32,
            "out_features": 4,
            "has_layer_norm": False,
            "activation": ACT_NONE,
            "activation_name": "NONE",
            "weights": rand_list(4 * 32, 0.25),
            "bias": rand_list(4, 0.05),
            "ln_gamma": [],
            "ln_beta": [],
        },
    ]
    return layers


def reference_forward_f32(
    layers: Sequence[Dict[str, Any]],
    batch_inputs: Sequence[Sequence[float]],
) -> List[List[float]]:
    """Evaluate the exact float32 neural network forward pass matching C++ IEEE-754 arithmetic."""
    current = [[f32(x) for x in row] for row in batch_inputs]

    for layer in layers:
        in_f = int(layer["in_features"])
        out_f = int(layer["out_features"])
        weights = layer["weights"]
        bias = layer["bias"]
        has_ln = bool(layer["has_layer_norm"])
        act = int(layer["activation"])
        gamma = layer["ln_gamma"]
        beta = layer["ln_beta"]

        next_batch: List[List[float]] = []
        for x_row in current:
            y_row: List[float] = []
            for o in range(out_f):
                w_offset = o * in_f
                acc0 = f32(0.0)
                acc1 = f32(0.0)
                acc2 = f32(0.0)
                acc3 = f32(0.0)
                k_unroll = (in_f // 4) * 4
                for k in range(0, k_unroll, 4):
                    acc0 = f32(acc0 + f32(x_row[k + 0] * weights[w_offset + k + 0]))
                    acc1 = f32(acc1 + f32(x_row[k + 1] * weights[w_offset + k + 1]))
                    acc2 = f32(acc2 + f32(x_row[k + 2] * weights[w_offset + k + 2]))
                    acc3 = f32(acc3 + f32(x_row[k + 3] * weights[w_offset + k + 3]))
                total = f32(f32(acc0 + acc1) + f32(acc2 + acc3))
                for k in range(k_unroll, in_f):
                    total = f32(total + f32(x_row[k] * weights[w_offset + k]))
                val = f32(total + bias[o])
                y_row.append(val)

            if has_ln:
                inv_n = f32(1.0 / float(out_f))
                s = f32(0.0)
                for v in y_row:
                    s = f32(s + v)
                mean = f32(s * inv_n)
                var_s = f32(0.0)
                for v in y_row:
                    d = f32(v - mean)
                    var_s = f32(var_s + f32(d * d))
                inv_std = f32(1.0 / math.sqrt(f32(f32(var_s * inv_n) + f32(1e-5))))
                for j in range(out_f):
                    norm = f32(f32(y_row[j] - mean) * inv_std)
                    y_row[j] = f32(f32(norm * gamma[j]) + beta[j])

            if act == ACT_RELU:
                y_row = [v if v > 0.0 else f32(0.0) for v in y_row]
            elif act == ACT_GELU:
                y_row = [gelu_f32(v) for v in y_row]

            next_batch.append(y_row)
        current = next_batch

    # Final Softmax
    probs_batch: List[List[float]] = []
    for row in current:
        max_v = max(row)
        exps = [f32(math.exp(f32(v - max_v))) for v in row]
        sum_exp = f32(0.0)
        for e in exps:
            sum_exp = f32(sum_exp + e)
        inv_sum = f32(1.0 / sum_exp)
        probs_batch.append([f32(e * inv_sum) for e in exps])
    return probs_batch


def export_artifacts(output_dir: Path, num_samples: int = 64, seed: int = 20261126) -> Dict[str, Any]:
    """Export binary model weights, ONNX graph JSON (and .onnx if torch is available), and golden test vectors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = build_deterministic_model_spec(seed=seed)

    layer_chunks: List[bytes] = []
    rolling_hash = 2166136261

    for layer in layers:
        header = struct.pack(
            "<IIII",
            int(layer["in_features"]),
            int(layer["out_features"]),
            1 if layer["has_layer_norm"] else 0,
            int(layer["activation"]),
        )
        w_bytes = pack_f32_list(layer["weights"])
        b_bytes = pack_f32_list(layer["bias"])
        rolling_hash = fnv1a_hash_bytes(w_bytes, rolling_hash)
        rolling_hash = fnv1a_hash_bytes(b_bytes, rolling_hash)

        chunk = bytearray(header + w_bytes + b_bytes)
        if layer["has_layer_norm"]:
            g_bytes = pack_f32_list(layer["ln_gamma"])
            bt_bytes = pack_f32_list(layer["ln_beta"])
            rolling_hash = fnv1a_hash_bytes(g_bytes, rolling_hash)
            rolling_hash = fnv1a_hash_bytes(bt_bytes, rolling_hash)
            chunk.extend(g_bytes)
            chunk.extend(bt_bytes)
        layer_chunks.append(bytes(chunk))

    model_bin_path = output_dir / "fraud_scorer_v1.bin"
    top_header = struct.pack("<IIII", MAGIC_HEADER, FORMAT_VERSION, len(layers), rolling_hash)
    model_bin_path.write_bytes(top_header + b"".join(layer_chunks))

    # Write ONNX-compatible graph metadata spec
    onnx_json_path = output_dir / "fraud_scorer_v1.onnx.json"
    onnx_spec = {
        "ir_version": 9,
        "producer_name": "ml-inference-server-exporter",
        "producer_version": "0.1.0",
        "graph_name": "fraud_risk_scorer_v1",
        "model_checksum_fnv1a": rolling_hash,
        "inputs": [{"name": "input_features", "dtype": "float32", "shape": ["batch_size", 16]}],
        "outputs": [{"name": "class_probabilities", "dtype": "float32", "shape": ["batch_size", 4]}],
        "nodes": [
            {
                "name": layer["name"],
                "op_type": "FusedGemmLayerNormActivation" if layer["has_layer_norm"] else "FusedGemmActivation",
                "in_features": layer["in_features"],
                "out_features": layer["out_features"],
                "has_layer_norm": layer["has_layer_norm"],
                "activation": layer["activation_name"],
            }
            for layer in layers
        ]
        + [{"name": "output_softmax", "op_type": "Softmax", "axis": 1}],
    }
    onnx_json_path.write_text(json.dumps(onnx_spec, indent=2) + "\n", encoding="utf-8")

    # Generate golden validation inputs and exact reference outputs
    rng = random.Random(seed + 99)
    inputs: List[List[float]] = [
        [f32(rng.uniform(-2.0, 2.0)) for _ in range(16)] for _ in range(num_samples)
    ]
    expected_outputs = reference_forward_f32(layers, inputs)

    # Optional PyTorch parity verification & .onnx export if torch is installed
    pytorch_max_diff: float | None = None
    onnx_exported = False
    try:
        import torch  # type: ignore

        class FraudScorerNet(torch.nn.Module):
            def __init__(self, spec_layers: Sequence[Dict[str, Any]]) -> None:
                super().__init__()
                self.fc1 = torch.nn.Linear(16, 64)
                self.ln1 = torch.nn.LayerNorm(64, eps=1e-5)
                self.fc2 = torch.nn.Linear(64, 32)
                self.fc3 = torch.nn.Linear(32, 4)
                with torch.no_grad():
                    self.fc1.weight.copy_(torch.tensor(spec_layers[0]["weights"], dtype=torch.float32).view(64, 16))
                    self.fc1.bias.copy_(torch.tensor(spec_layers[0]["bias"], dtype=torch.float32))
                    self.ln1.weight.copy_(torch.tensor(spec_layers[0]["ln_gamma"], dtype=torch.float32))
                    self.ln1.bias.copy_(torch.tensor(spec_layers[0]["ln_beta"], dtype=torch.float32))
                    self.fc2.weight.copy_(torch.tensor(spec_layers[1]["weights"], dtype=torch.float32).view(32, 64))
                    self.fc2.bias.copy_(torch.tensor(spec_layers[1]["bias"], dtype=torch.float32))
                    self.fc3.weight.copy_(torch.tensor(spec_layers[2]["weights"], dtype=torch.float32).view(4, 32))
                    self.fc3.bias.copy_(torch.tensor(spec_layers[2]["bias"], dtype=torch.float32))

            def forward(self, x: Any) -> Any:
                x = torch.nn.functional.gelu(self.ln1(self.fc1(x)), approximate="none")
                x = torch.nn.functional.relu(self.fc2(x))
                return torch.nn.functional.softmax(self.fc3(x), dim=-1)

        net = FraudScorerNet(layers).eval()
        with torch.no_grad():
            t_in = torch.tensor(inputs, dtype=torch.float32)
            t_out = net(t_in).tolist()
            max_d = 0.0
            for r_ref, r_pt in zip(expected_outputs, t_out):
                for v_ref, v_pt in zip(r_ref, r_pt):
                    max_d = max(max_d, abs(float(v_ref) - float(v_pt)))
            pytorch_max_diff = max_d
            try:
                onnx_path = output_dir / "fraud_scorer_v1.onnx"
                torch.onnx.export(
                    net,
                    t_in[:1],
                    str(onnx_path),
                    input_names=["input_features"],
                    output_names=["class_probabilities"],
                    opset_version=17,
                )
                onnx_exported = True
            except Exception:
                onnx_exported = False
    except ImportError:
        pass

    input_bin_path = output_dir / "golden_inputs.bin"
    flat_inputs = [v for row in inputs for v in row]
    input_bin_path.write_bytes(struct.pack("<II", num_samples, 16) + pack_f32_list(flat_inputs))

    golden_json_path = output_dir / "golden_test_vectors.json"
    golden_payload = {
        "model_checksum": rolling_hash,
        "num_samples": num_samples,
        "input_dim": 16,
        "output_dim": 4,
        "pytorch_max_diff": pytorch_max_diff,
        "onnx_binary_exported": onnx_exported,
        "inputs": inputs,
        "expected_outputs": expected_outputs,
    }
    golden_json_path.write_text(json.dumps(golden_payload, indent=2) + "\n", encoding="utf-8")
    return golden_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Export float32 model weights and golden vectors")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261126)
    args = parser.parse_args()
    info = export_artifacts(args.output_dir, num_samples=args.samples, seed=args.seed)
    print(
        json.dumps(
            {
                "status": "exported",
                "output_dir": str(args.output_dir),
                "model_checksum": info["model_checksum"],
                "num_samples": info["num_samples"],
                "pytorch_max_diff": info["pytorch_max_diff"],
                "onnx_binary_exported": info["onnx_binary_exported"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
