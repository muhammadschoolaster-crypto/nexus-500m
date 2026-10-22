"""Export a NEXUS checkpoint to the mobile `.nexus` format.

.nexus = a ZIP archive containing:
  - manifest.json : format version, full model config, quantization info
                    (per-tensor int8 scales), tensor order + shapes,
                    tokenizer (merges), EOS id, metadata
  - weights.bin   : all tensors concatenated as little-endian int8, in the
                    manifest-declared order (symmetric per-tensor quantization)

The Android app ships a pure-Kotlin reader for this exact format, so the same
file that PyTorch trained is the file the phone runs — no third-party runtime.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig
from .model import TransformerLM
from .tokenizer import BPETokenizer
from .train import load_checkpoint

NEXUS_MAGIC = "NEXUS1"
NEXUS_FORMAT_VERSION = 1


def quantize_tensor(t: torch.Tensor) -> tuple:
    """Symmetric per-tensor int8: q = round(w / scale), scale = max|w| / 127."""
    w = t.detach().to(torch.float32).numpy()
    max_abs = float(np.max(np.abs(w))) if w.size else 0.0
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
    return q, scale


def export_nexus(ckpt_path: str | Path, tokenizer_path: str | Path,
                out_path: str | Path, model_name: str = "nexus-demo",
                version: str = "0.1.0", notes: str = "") -> dict:
    ck = load_checkpoint(ckpt_path)
    cfg = ModelConfig.from_dict(ck["config"])
    tok = BPETokenizer.load_json(tokenizer_path)
    assert tok.vocab_size == cfg.vocab_size, "tokenizer/model vocab mismatch"

    model = TransformerLM(cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    tensors = [(name, p) for name, p in model.named_parameters()]
    blobs, entries, scales = [], [], {}
    buf = io.BytesIO()
    for name, p in tensors:
        q, scale = quantize_tensor(p)
        shapes = list(p.shape)
        entries.append({"name": name, "shape": shapes, "dtype": "int8",
                        "offset": buf.tell(), "n_bytes": int(q.nbytes), "scale": scale})
        scales[name] = scale
        buf.write(q.tobytes(order="C"))
    weights_bytes = buf.getvalue()

    manifest = {
        "magic": NEXUS_MAGIC,
        "format_version": NEXUS_FORMAT_VERSION,
        "model_name": model_name,
        "model_version": version,
        "notes": notes,
        "config": cfg.to_dict(),
        "quantization": {"type": "int8_symmetric_per_tensor", "scales": scales},
        "tensors": entries,
        "tokenizer": {
            "version": tok.version,
            "merges": [list(m) for m in tok.merges],
            "vocab_size": tok.vocab_size,
            "eos_token_id": tok.eos_token_id,
        },
        "training": {
            "checkpoint_step": ck.get("step"),
            "val_loss": ck.get("val_loss"),
            "torch_version": ck.get("torch_version"),
        },
        "weights_sha256": hashlib.sha256(weights_bytes).hexdigest(),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        zf.writestr("weights.bin", weights_bytes)

    info = {
        "path": str(out_path),
        "size_bytes": out_path.stat().st_size,
        "sha256": file_sha256(out_path),
        "params": model.num_parameters(),
        "model_name": model_name,
        "version": version,
    }
    return info


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_nexus_state(nexus_path: str | Path) -> tuple:
    """Read .nexus back: returns (ModelConfig, dequantized state_dict, eos_id).

    Used by parity tests and the golden-data generator: the SAME quantized
    weights the phone will use are loaded back into the torch model.
    """
    with zipfile.ZipFile(nexus_path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        weights = zf.read("weights.bin")
    assert manifest["magic"] == NEXUS_MAGIC, "not a .nexus file"
    assert manifest["format_version"] == NEXUS_FORMAT_VERSION, "unsupported format version"
    assert hashlib.sha256(weights).hexdigest() == manifest["weights_sha256"], "weights.bin corrupted"

    state = {}
    for e in manifest["tensors"]:
        raw = weights[e["offset"]: e["offset"] + e["n_bytes"]]
        q = np.frombuffer(raw, dtype=np.int8).reshape(e["shape"])
        w = q.astype(np.float32) * np.float32(e["scale"])
        state[e["name"]] = torch.from_numpy(w)
    cfg = ModelConfig.from_dict(manifest["config"])
    eos = manifest["tokenizer"]["eos_token_id"]
    return cfg, state, eos


def verify_export(ckpt_path: str | Path, nexus_path: str | Path) -> dict:
    """Check the quantization error bound: |w_dequant - w| <= max|w| / 254."""
    ck = load_checkpoint(ckpt_path)
    model = TransformerLM(ModelConfig.from_dict(ck["config"]))
    model.load_state_dict(ck["model"])
    _, state, _ = load_nexus_state(nexus_path)

    worst = 0.0
    for name, p in model.named_parameters():
        orig = p.detach().numpy()
        deq = state[name].numpy()
        max_abs = float(np.max(np.abs(orig))) if orig.size else 0.0
        bound = max_abs / 254.0 if max_abs > 0 else 1e-8
        err = float(np.max(np.abs(orig - deq))) if orig.size else 0.0
        worst = max(worst, err)
        assert err <= bound + 1e-6, f"quantization error too large for {name}: {err} > {bound}"
    return {"worst_abs_error": worst, "tensors": len(state)}


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Export checkpoint to .nexus mobile format")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-name", default="nexus-demo")
    p.add_argument("--version", default="0.1.0")
    p.add_argument("--notes", default="")
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    info = export_nexus(a.checkpoint, a.tokenizer, a.out, a.model_name, a.version, a.notes)
    print(json.dumps(info, indent=2))
    if a.verify:
        print(json.dumps(verify_export(a.checkpoint, a.out), indent=2))


if __name__ == "__main__":
    main()
