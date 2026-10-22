"""Training loop: reproducible, checkpointed, validated, logged.

Features:
- deterministic seeds (python/numpy/torch)
- AdamW with weight decay on matrix weights only
- linear warmup + cosine decay schedule
- gradient clipping
- JSONL training log (step, loss, lr, tokens/sec)
- periodic validation; checkpoints (model+optimizer+step+config) with resume
- NaN guard (aborts with a clear error instead of silently diverging)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig
from .model import TransformerLM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PackedBins:
    """Fixed (ctx+1)-length token windows from a .bin.npy file."""

    def __init__(self, path: str | Path):
        self.arr = np.load(path, mmap_mode="r")   # (n_windows, ctx+1) uint16
        self.n_windows = self.arr.shape[0]

    def sample_batch(self, batch_size: int, rng: np.random.Generator):
        if self.n_windows == 0:
            raise RuntimeError("empty packed dataset")
        ix = rng.integers(0, self.n_windows, size=batch_size)
        rows = np.asarray(self.arr[ix], dtype=np.int64)      # (B, ctx+1)
        x = torch.from_numpy(rows[:, :-1])
        y = torch.from_numpy(rows[:, 1:])
        return x, y

    def deterministic_batches(self, batch_size: int):
        for start in range(0, self.n_windows, batch_size):
            rows = np.asarray(self.arr[start: start + batch_size], dtype=np.int64)
            if rows.shape[0] < 2:
                continue
            yield torch.from_numpy(rows[:, :-1]), torch.from_numpy(rows[:, 1:])


def lr_at(step: int, total_steps: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def evaluate_loss(model, bins: PackedBins, batch_size: int, n_batches: int = 10) -> float:
    model.eval()
    losses = []
    gen = bins.deterministic_batches(batch_size)
    for _ in range(n_batches):
        try:
            x, y = next(gen)
        except StopIteration:
            break
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def build_optimizer(model, base_lr: float, weight_decay: float = 0.1):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.95))


def save_checkpoint(path, model, optimizer, step, cfg, val_loss, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "config": cfg.to_dict(),
        "config_hash": cfg.config_hash(),
        "val_loss": val_loss,
        "torch_version": torch.__version__,
    }
    if extra:
        payload.update(extra)
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)   # atomic-ish: a crash never leaves a half-written ckpt


def load_checkpoint(path, model=None, optimizer=None, map_location="cpu"):
    # Our checkpoints are produced by our own training code (trusted files);
    # they contain optimizer state dicts, so they need weights_only=False.
    ck = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        cfg = ModelConfig.from_dict(ck["config"])
        model.load_state_dict(ck["model"])
    if optimizer is not None and ck.get("optimizer"):
        optimizer.load_state_dict(ck["optimizer"])
    return ck


def train(args) -> dict:
    # --config is always a JSON config file path (see configs/)
    cfg = ModelConfig.load_json(args.config)
    set_seed(args.seed)
    device = torch.device(args.device)

    model = TransformerLM(cfg).to(device)
    n_params = model.num_parameters()
    print(f"config: {cfg.name} | params: {n_params:,} | device: {device}")

    train_bins = PackedBins(args.train_bins)
    val_bins = PackedBins(args.val_bins) if args.val_bins else None
    print(f"train windows: {train_bins.n_windows:,} | val windows: {val_bins.n_windows if val_bins else 0:,}")

    optimizer = build_optimizer(model, args.lr)
    start_step, best_val = 0, float("inf")
    if args.resume and Path(args.resume).exists():
        ck = load_checkpoint(args.resume, model, optimizer)
        start_step, best_val = ck["step"], ck.get("val_loss", float("inf"))
        print(f"resumed from {args.resume} at step {start_step}")

    rng = np.random.default_rng(args.seed + start_step)
    out_dir = Path(args.out)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"

    t0 = time.time()
    model.train()
    with log_path.open("a", encoding="utf-8") as logf:
        for step in range(start_step, args.steps):
            lr = lr_at(step - start_step, args.steps - start_step, args.lr, args.warmup)
            for g in optimizer.param_groups:
                g["lr"] = lr

            x, y = train_bins.sample_batch(args.batch_size, rng)
            _, loss = model(x.to(device), y.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {loss.item()} (aborting, not diverging silently)")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (step + 1) % args.log_every == 0:
                elapsed = time.time() - t0
                rec = {
                    "step": step + 1, "train_loss": round(loss.item(), 6), "lr": round(lr, 8),
                    "tokens_per_sec": round(args.batch_size * cfg.ctx_len / max(elapsed, 1e-9), 1),
                    "elapsed_sec": round(elapsed, 1), "stage": args.stage,
                }
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                print(f"step {step+1:6d} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:7.1f}s")
                t0 = time.time()

            if val_bins is not None and ((step + 1) % args.val_every == 0 or step + 1 == args.steps):
                vl = evaluate_loss(model, val_bins, args.batch_size)
                print(f"  val loss: {vl:.4f}")
                logf.write(json.dumps({"step": step + 1, "val_loss": round(vl, 6), "stage": args.stage}) + "\n")
                logf.flush()
                if vl < best_val:
                    best_val = vl
                    save_checkpoint(out_dir / "checkpoints" / "best.pt", model, optimizer, step + 1, cfg, vl)

            if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
                save_checkpoint(out_dir / "checkpoints" / "last.pt", model, optimizer, step + 1, cfg, best_val)

    final = evaluate_loss(model, val_bins, args.batch_size) if val_bins else float("nan")
    save_checkpoint(out_dir / "checkpoints" / "final.pt", model, None, args.steps, cfg, final)
    return {"final_val_loss": final, "steps": args.steps, "params": n_params}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NEXUS training")
    p.add_argument("--config", required=True, help="path to model config JSON (configs/*.json)")
    p.add_argument("--train-bins", required=True)
    p.add_argument("--val-bins", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--resume", default="")
    p.add_argument("--device", default="cpu")
    p.add_argument("--stage", default="base", help="base | sft (recorded in logs)")
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
