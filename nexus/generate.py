"""Sampling/generation utilities + golden-data export for Android parity tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig
from .model import TransformerLM
from .train import load_checkpoint, set_seed


@torch.no_grad()
def generate(model: TransformerLM, idx: torch.Tensor, max_new_tokens: int,
             temperature: float = 0.8, top_k: int = 0, greedy: bool = False,
             eos_id: int | None = None, rng: np.random.Generator | None = None):
    """Token-by-token generation (no KV cache; window crops to ctx_len — fine at demo scale,
    and the identical cropping rule is implemented in Kotlin)."""
    model.eval()
    out = []
    for _ in range(max_new_tokens):
        idx_c = idx[:, -model.cfg.ctx_len:]
        logits, _ = model(idx_c)
        logits = logits[0, -1, :]
        if greedy:
            nxt = int(torch.argmax(logits).item())
        else:
            logits = logits / max(temperature, 1e-6)
            if top_k and top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.numel()))
                mask = torch.full_like(logits, float("-inf"))
                mask[kth.indices] = kth.values
                logits = mask
            probs = torch.softmax(logits, dim=-1).numpy().astype(np.float64)
            g = rng or np.random.default_rng(0)
            nxt = int(g.choice(len(probs), p=probs / probs.sum()))
        if eos_id is not None and nxt == eos_id:
            break
        idx = torch.cat([idx, torch.tensor([[nxt]], dtype=idx.dtype)], dim=1)
        out.append(nxt)
    return out


def load_model_from_checkpoint(ckpt_path: str | Path) -> tuple:
    ck = load_checkpoint(ckpt_path)
    cfg = ModelConfig.from_dict(ck["config"])
    model = TransformerLM(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck


def load_model_from_nexus(nexus_path: str | Path):
    """Load a quantized .nexus export back into the torch model (for parity checks)."""
    from .export import load_nexus_state
    cfg, state, eos_id = load_nexus_state(nexus_path)
    model = TransformerLM(cfg)
    model.load_state_dict(state)
    model.eval()
    return model, cfg, eos_id


def main() -> None:
    import argparse
    from .tokenizer import BPETokenizer

    p = argparse.ArgumentParser(description="NEXUS generation / golden export")
    p.add_argument("--checkpoint", default="", help="torch .pt checkpoint")
    p.add_argument("--nexus", default="", help="quantized .nexus file")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompt", default="### Instruction:\nSay hello.\n### Response:\n")
    p.add_argument("--max-new", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--golden-out", default="", help="write golden data JSON for Android tests")
    a = p.parse_args()

    if a.nexus:
        model, cfg, eos_id = load_model_from_nexus(a.nexus)
    else:
        model, cfg, ck = load_model_from_checkpoint(a.checkpoint)
        eos_id = None
    tok = BPETokenizer.load_json(a.tokenizer)
    eos = eos_id if eos_id is not None else tok.eos_token_id

    set_seed(a.seed)
    ids = tok.encode(a.prompt)
    x = torch.tensor([ids], dtype=torch.long)
    new_ids = generate(model, x, a.max_new, temperature=a.temperature, top_k=a.top_k,
                       greedy=a.greedy, eos_id=eos, rng=np.random.default_rng(a.seed))
    full = ids + new_ids
    text = tok.decode(full)
    print("--- GENERATED ---")
    print(text)
    print("-----------------")

    if a.golden_out:
        golden = {
            "model_format": "nexus" if a.nexus else "pt",
            "prompt": a.prompt,
            "prompt_token_ids": ids,
            "sampled_token_ids": new_ids,
            "temperature": a.temperature, "top_k": a.top_k, "greedy": a.greedy,
            "seed": a.seed,
            "first_step_logits": None,
        }
        with torch.no_grad():
            logits, _ = model(torch.tensor([ids], dtype=torch.long))
        golden["first_step_logits"] = [round(float(v), 4) for v in logits[0, -1, :].tolist()]
        Path(a.golden_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.golden_out).write_text(json.dumps(golden), encoding="utf-8")
        print(f"golden data -> {a.golden_out}")


if __name__ == "__main__":
    main()
