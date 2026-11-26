"""PyTorch / ONNX weight serializer and golden reference verifier for the C++20 MLIS engine."""
import math
from pathlib import Path
import struct
from typing import Dict, List, Sequence

MAGIC_HEADER = b"MLISv001"
ACT_NONE = 0
ACT_RELU = 1
ACT_GELU = 2


def fnv1a_64(data: bytes, seed: int = 14695981039346656037) -> int:
    h = seed
    for b in data:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def pack_f32_array(values: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *[float(v) for v in values])


def export_mlis_model(layers: List[Dict], output_path: Path, final_softmax: bool = True) -> Dict:
    """Serialize model layers into the checksum-guarded MLISv001 binary format."""
    if not layers:
        raise ValueError("layers must be nonempty")

    chunks = [
        MAGIC_HEADER,
        struct.pack("<IB", len(layers), 1 if final_softmax else 0),
    ]
    total_params = 0
    for idx, layer in enumerate(layers):
        in_d = int(layer["in_features"])
        out_d = int(layer["out_features"])
        act = int(layer.get("activation", ACT_NONE))
        use_ln = bool(layer.get("use_layer_norm", False))
        weights = layer["weights"]
        bias = layer["bias"]
        gamma = layer.get("ln_gamma", [1.0] * out_d)
        beta = layer.get("ln_beta", [0.0] * out_d)

        if in_d <= 0 or out_d <= 0 or len(weights) != in_d * out_d or len(bias) != out_d:
            raise ValueError(f"invalid dimensions in layer {idx}")
        if idx > 0 and int(layers[idx - 1]["out_features"]) != in_d:
            raise ValueError("adjacent layer dimensions do not match")

        chunks.append(struct.pack("<IIBB", in_d, out_d, act, 1 if use_ln else 0))
        chunks.append(pack_f32_array(weights))
        chunks.append(pack_f32_array(bias))
        chunks.append(pack_f32_array(gamma))
        chunks.append(pack_f32_array(beta))
        total_params += len(weights) + len(bias)

    payload = b"".join(chunks)
    checksum = fnv1a_64(payload)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload + struct.pack("<Q", checksum))

    return {
        "path": str(output_path),
        "num_layers": len(layers),
        "total_parameters": total_params,
        "file_bytes": output_path.stat().st_size,
        "fnv1a_checksum": hex(checksum),
    }


def python_golden_forward(layers: List[Dict], batch: List[List[float]], final_softmax: bool = True) -> List[List[float]]:
    """Reference FP32 forward pass in Python for cross-language parity verification against C++20."""
    current = [list(map(float, row)) for row in batch]
    for layer in layers:
        in_d = int(layer["in_features"])
        out_d = int(layer["out_features"])
        w = layer["weights"]
        b = layer["bias"]
        act = int(layer.get("activation", ACT_NONE))
        use_ln = bool(layer.get("use_layer_norm", False))
        gamma = layer.get("ln_gamma", [1.0] * out_d)
        beta = layer.get("ln_beta", [0.0] * out_d)

        next_batch = []
        for row in current:
            if len(row) != in_d:
                raise ValueError("input row width mismatch")
            out_row = []
            for o in range(out_d):
                base = o * in_d
                acc = float(b[o])
                for i in range(in_d):
                    acc += row[i] * float(w[base + i])
                out_row.append(acc)

            if use_ln:
                mean = sum(out_row) / out_d
                var = sum((x - mean) ** 2 for x in out_row) / out_d
                inv_std = 1.0 / math.sqrt(var + 1e-5)
                out_row = [(out_row[c] - mean) * inv_std * float(gamma[c]) + float(beta[c]) for c in range(out_d)]

            if act == ACT_RELU:
                out_row = [max(0.0, x) for x in out_row]
            elif act == ACT_GELU:
                k0 = 0.7978845608028654
                out_row = [0.5 * x * (1.0 + math.tanh(k0 * (x + 0.044715 * (x ** 3)))) for x in out_row]

            next_batch.append(out_row)
        current = next_batch

    if final_softmax:
        sm_batch = []
        for row in current:
            m = max(row)
            exps = [math.exp(x - m) for x in row]
            s = sum(exps)
            sm_batch.append([e / s for e in exps])
        current = sm_batch
    return current
