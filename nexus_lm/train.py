"""NEXUS-LM pretraining loop: AdamW + cosine LR + warmup, checkpoint/resume,
CSV logging, periodic validation loss + sample generation.

Real training only. No fabricated numbers: everything logged comes from
actual forward/backward passes on this machine.
"""
import argparse
import csv
import json
import math
import os
import time

import torch

from .data import get_batch
from .model import NexusLM, NexusConfig


def lr_schedule(step, warmup, total, base_lr, min_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def evaluate(model, prefix, ctx, batch_size, iters=10):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(iters):
            x, y = get_batch("val", prefix, ctx, batch_size)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prefix", required=True, help="bin file prefix (train/val .npy)")
    ap.add_argument("--out", required=True, help="run dir")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--ckpt_every", type=int, default=250)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_seconds", type=float, default=1e9,
                    help="stop+checkpoint gracefully after this many seconds")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = NexusConfig.from_json_file(args.config)
    model = NexusLM(cfg)
    counts = model.exact_param_count()
    print(f"config: {cfg.to_json()}")
    print(f"params: total={counts['total']:,} non_embedding={counts['non_embedding']:,} "
          f"(formula check={counts['config_check']:,})")
    assert counts["total"] == counts["config_check"], "param formula mismatch!"

    device = "cpu"
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=False)

    start_step = 0
    best_val = float("inf")
    ckpt_path = os.path.join(args.out, "nexus_ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"] + 1
        best_val = ck.get("best_val", best_val)
        print(f"resumed from step {start_step}")

    csv_path = os.path.join(args.out, "train_log.csv")
    write_header = not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0)
    logf = open(csv_path, "a", newline="")
    writer = csv.writer(logf)
    if write_header:
        writer.writerow(["step", "loss", "lr", "tokens_per_sec", "elapsed_s"])

    t_start = time.time()
    tokens_seen = 0
    model.train()
    print(f"training up to {args.steps} steps (batch={args.batch}, ctx={cfg.block_size}) on {device}")
    stop = False
    for step in range(start_step, args.steps):
        if time.time() - t_start > args.max_seconds and step > start_step:
            print(f"time budget reached at step {step}; checkpointing and exiting")
            stop = True
        if stop:
            break
        lr = lr_schedule(step, args.warmup, args.steps, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        t0 = time.time()
        x, y = get_batch("train", args.prefix, cfg.block_size, args.batch)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        dt = time.time() - t0
        tokens_seen += args.batch * cfg.block_size
        if step % args.log_every == 0 or step == args.steps - 1:
            tps = tokens_seen / (time.time() - t_start)
            print(f"step {step:6d}/{args.steps} loss {loss.item():.4f} lr {lr:.2e} "
                  f"{tokens_seen/1e6:.2f}M tok seen ({tps:.0f} tok/s)", flush=True)
            writer.writerow([step, f"{loss.item():.5f}", f"{lr:.3e}", f"{tps:.1f}",
                             f"{time.time()-t_start:.1f}"])
            logf.flush()
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            vl = evaluate(model, args.prefix, cfg.block_size, args.batch)
            print(f"  [val] step {step+1} val_loss {vl:.4f} "
                  f"perplexity {math.exp(min(vl, 20)):.2f}", flush=True)
            with open(os.path.join(args.out, "val_log.csv"), "a") as vf:
                vf.write(f"{step+1},{vl:.5f}\n")
            if vl < best_val:
                best_val = vl
                torch.save({"model": model.state_dict(), "step": step, "val": vl},
                           os.path.join(args.out, "nexus_best.pt"))
        if (step + 1) % args.ckpt_every == 0 or step == args.steps - 1:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "best_val": best_val, "cfg": cfg.__dict__}, ckpt_path)

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step - 1 if stop else args.steps - 1,
                "best_val": best_val, "cfg": cfg.__dict__}, ckpt_path)
    writer.writerow(["DONE" if not stop else "CHUNK_END", "", "", "",
                     f"best_val={best_val:.4f}"])
    logf.close()
    print(("DONE." if not stop else "CHUNK_END. ") + f" best_val_loss={best_val:.4f} "
          f"total_time={(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
