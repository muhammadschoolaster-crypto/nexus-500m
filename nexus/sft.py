"""Instruction tuning (SFT) stage.

Builds supervised bins from instructions.jsonl using the canonical chat
template, then continues training from the base checkpoint with a lower
learning rate. Base and SFT checkpoints are kept SEPARATE (never overwritten);
the SFT model only replaces the app model after validation checks pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import ModelConfig
from .data import load_instructions, pack_instruction_bins
from .tokenizer import BPETokenizer
from .train import evaluate_loss, save_checkpoint, train


def run_sft(args) -> dict:
    tok = BPETokenizer.load_json(args.tokenizer)
    cfg = ModelConfig.load_json(args.config)
    assert tok.vocab_size == cfg.vocab_size, (
        f"tokenizer vocab ({tok.vocab_size}) != model vocab ({cfg.vocab_size})"
    )
    instructions = load_instructions(args.instructions)
    bins_dir = Path(args.out) / "bins"
    paths = pack_instruction_bins(instructions, tok, cfg.ctx_len, bins_dir)

    train_args = argparse.Namespace(
        config=args.config, train_bins=paths["train"], val_bins=paths["val"],
        out=args.out, steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        warmup=args.warmup, seed=args.seed, log_every=args.log_every,
        val_every=args.val_every, save_every=args.save_every,
        init_from=args.base_checkpoint, resume="",
        device=args.device, stage="sft",
    )
    result = train(train_args)

    # Validation gate: the SFT checkpoint must beat (or match) random-chance
    # loss on its own val split before it can be promoted anywhere.
    sft_ckpt = Path(args.out) / "checkpoints" / "final.pt"
    ck = torch.load(sft_ckpt, map_location="cpu", weights_only=False)
    report = {
        "stage": "sft",
        "examples": len(instructions),
        "steps": args.steps,
        "final_val_loss": result["final_val_loss"],
        "params": result["params"],
        "config": cfg.to_dict(),
        "passed_gate": bool(
            ck.get("val_loss") is not None and float(ck["val_loss"]) < 6.9
        ),  # ln(2048) ~ 7.6; 6.9 = clearly below untrained
    }
    (Path(args.out) / "sft_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NEXUS SFT")
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--instructions", required=True, help="instructions.jsonl")
    p.add_argument("--base-checkpoint", default="", help="base final.pt: weights-only init (step counter resets for the SFT stage)")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--val-every", type=int, default=40)
    p.add_argument("--save-every", type=int, default=60)
    p.add_argument("--device", default="cpu")
    return p


if __name__ == "__main__":
    print(json.dumps(run_sft(build_argparser().parse_args()), indent=2))
