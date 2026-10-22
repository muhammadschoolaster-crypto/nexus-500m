"""Exact parameter counting, verified on PyTorch's `meta` device.

The meta device builds the full module tree (real shapes) without allocating
any weight memory, so the 500M configuration can be verified exactly even on
a machine with far less RAM than the ~2 GB the float32 weights would need.

Run:  python -m nexus.count_params
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import PRESETS, ModelConfig, get_preset
from .model import TransformerLM


def exact_param_table(cfg: ModelConfig) -> tuple[dict, int]:
    """Build the model on the meta device and count every parameter exactly."""
    try:
        with torch.device("meta"):
            model = TransformerLM(cfg)
    except AttributeError:  # pragma: no cover - very old torch
        raise RuntimeError("PyTorch >= 2.0 required for meta-device parameter counting")
    table = {name: p.numel() for name, p in model.named_parameters()}
    total = sum(table.values())
    return table, total


def main() -> None:
    report = {}
    for name in PRESETS:
        cfg = get_preset(name)
        table, total = exact_param_table(cfg)
        by_component = {
            "token_embeddings": sum(v for k, v in table.items() if k.startswith("wte")),
            "position_embeddings": sum(v for k, v in table.items() if k.startswith("wpe")),
            "attention": sum(v for k, v in table.items() if ".attn." in k),
            "mlp": sum(v for k, v in table.items() if ".mlp." in k),
            "layernorms": sum(v for k, v in table.items() if ".ln" in k or k == "ln_f"),
        }
        report[name] = {
            "config": cfg.to_dict(),
            "total_parameters": total,
            "total_million": round(total / 1e6, 3),
            "by_component": by_component,
        }
        print(f"{name:12s} d={cfg.d_model:5d} L={cfg.n_layers:3d} H={cfg.n_heads:3d} "
              f"V={cfg.vocab_size:6d} ctx={cfg.ctx_len:5d} -> "
              f"{total:,} parameters ({total/1e6:.2f} M)")

    five = report["nexus_500m"]["total_parameters"]
    assert 490_000_000 <= five <= 520_000_000, "nexus_500m config must land near 500M parameters"
    print(f"\nnexus_500m target check: {five:,} params is within [490M, 520M] -> OK")

    out = Path("artifacts/param_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"parameter report written to {out}")


if __name__ == "__main__":
    main()
