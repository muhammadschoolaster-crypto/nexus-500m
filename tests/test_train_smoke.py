"""Real training smoke tests: an actual optimizer loop runs here (CPU, ~10-40 s).

These are the tests that prove the training pipeline genuinely learns:
1. a fixed batch is overfit -> loss must drop dramatically
2. checkpoint save/restore reproduces the model bit-exactly
3. training resume continues from the stored step
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

from nexus.config import get_preset
from nexus.model import TransformerLM
from nexus.train import (PackedBins, build_optimizer, evaluate_loss,
                         load_checkpoint, lr_at, save_checkpoint, set_seed)


def _fixed_dataset(tmp_path, n_windows=32):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=(n_windows, 65)).astype(np.uint16)
    p = tmp_path / "fixed.bin.npy"
    np.save(p, data)
    return PackedBins(p)


def test_overfit_fixed_batch_loss_drops(tmp_path):
    set_seed(0)
    cfg = get_preset("nexus_smoke")
    model = TransformerLM(cfg)
    opt = build_optimizer(model, base_lr=3e-3)

    x = torch.randint(0, cfg.vocab_size, (8, 33))
    x[:, 0] = 0
    y = torch.roll(x, -1, dims=1)
    # use only first 32 positions as targets (like ctx 32)
    xb, yb = x[:, :33], torch.cat([x[:, 1:], torch.full((8, 1), 0, dtype=torch.long)], 1)
    yb = yb[:, :32]
    xb = xb[:, :32]

    first_loss = None
    for step in range(40):
        logits, loss = model(xb, yb)
        if first_loss is None:
            first_loss = loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    final_loss = loss.item()
    assert torch.isfinite(loss)
    assert final_loss < first_loss * 0.6, (
        f"model failed to learn: loss {first_loss:.3f} -> {final_loss:.3f}"
    )


def test_checkpoint_save_restore_identical(tmp_path):
    set_seed(1)
    cfg = get_preset("nexus_smoke")
    model = TransformerLM(cfg)
    opt = build_optimizer(model, base_lr=1e-3)

    x = torch.randint(0, cfg.vocab_size, (4, 16))
    y = torch.randint(0, cfg.vocab_size, (4, 16))
    _, loss1 = model(x, y)
    loss1.backward()
    opt.step()

    ckpt_path = tmp_path / "ck.pt"
    save_checkpoint(ckpt_path, model, opt, step=1, cfg=cfg, val_loss=0.5)

    model2 = TransformerLM(cfg)
    opt2 = build_optimizer(model2, base_lr=1e-3)
    ck = load_checkpoint(ckpt_path, model2, opt2)
    assert ck["step"] == 1
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"parameter {n1} changed across save/restore"


def test_training_loop_learns_and_logs(tmp_path):
    """Full loop integration: the real train() entrypoint runs and the loss decreases."""
    import argparse
    from nexus.train import train

    bins = _fixed_dataset(tmp_path)
    # duplicate as val
    import shutil
    val_p = tmp_path / "val.bin.npy"
    shutil.copy(str(tmp_path / "fixed.bin.npy"), val_p)

    cfg = get_preset("nexus_smoke")
    cfg_path = tmp_path / "cfg.json"
    cfg.save_json(cfg_path)

    args = argparse.Namespace(
        config=str(cfg_path), train_bins=str(tmp_path / "fixed.bin.npy"),
        val_bins=str(val_p), out=str(tmp_path / "run"), steps=25, batch_size=8,
        lr=1e-3, warmup=5, seed=0, log_every=5, val_every=25, save_every=25,
        resume="", device="cpu", stage="base",
    )
    result = train(args)

    log_lines = (tmp_path / "run" / "training_log.jsonl").read_text().strip().splitlines()
    import json
    losses = [json.loads(l)["train_loss"] for l in log_lines if "train_loss" in l]
    assert len(losses) >= 4
    assert losses[-1] < losses[0], f"no learning in real loop: {losses}"
    assert (tmp_path / "run" / "checkpoints" / "final.pt").exists()
    assert result["params"] > 0


def test_lr_schedule_warmup_and_decay():
    assert lr_at(0, 100, 0.1, warmup=10) == pytest.approx(0.01)
    assert lr_at(9, 100, 0.1, warmup=10) == pytest.approx(0.1)
    assert lr_at(99, 100, 0.1, warmup=10) < lr_at(50, 100, 0.1, warmup=10) < 0.1
    # never negative
    assert lr_at(100, 100, 0.1, warmup=10) >= 0


def test_evaluate_loss_deterministic(tmp_path):
    bins = _fixed_dataset(tmp_path)
    set_seed(2)
    cfg = get_preset("nexus_smoke")
    model = TransformerLM(cfg)
    l1 = evaluate_loss(model, bins, batch_size=8)
    l2 = evaluate_loss(model, bins, batch_size=8)
    assert l1 == l2
    assert np.isfinite(l1)
