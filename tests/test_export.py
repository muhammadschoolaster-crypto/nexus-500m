"""Export round-trip tests: .nexus files must faithfully represent the model.

The Android app reads the exact same .nexus file these tests verify, so these
tests are the contract between the training side and the phone side.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

from nexus.config import get_preset
from nexus.export import export_nexus, file_sha256, load_nexus_state, verify_export
from nexus.model import TransformerLM
from nexus.tokenizer import BPETokenizer
from nexus.train import save_checkpoint, set_seed


@pytest.fixture(scope="module")
def tiny_trained_checkpoint(tmp_path_factory):
    """A genuinely tiny trained model: 60 steps on one fixed batch (fast on CPU)."""
    d = tmp_path_factory.mktemp("ckpt")
    set_seed(3)
    cfg = get_preset("nexus_smoke")
    model = TransformerLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, cfg.vocab_size, (4, 32))
    y = torch.roll(x, -1, dims=1)
    for _ in range(60):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    ckpt = d / "tiny.pt"
    save_checkpoint(ckpt, model, None, step=60, cfg=cfg, val_loss=float(loss.item()))
    tok = BPETokenizer()
    tok_p = d / "tokenizer.json"
    tok.save_json(tok_p)
    return ckpt, tok_p, float(loss.item())


def test_export_creates_file_with_manifest(tiny_trained_checkpoint, tmp_path):
    ckpt, tok_p, _ = tiny_trained_checkpoint
    out = tmp_path / "model.nexus"
    info = export_nexus(ckpt, tok_p, out, model_name="test-model", version="9.9.9")
    assert out.exists() and out.stat().st_size > 1000
    assert len(info["sha256"]) == 64
    assert file_sha256(out) == info["sha256"]
    import zipfile, json
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "manifest.json" in names and "weights.bin" in names
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["magic"] == "NEXUS1"
    assert manifest["model_name"] == "test-model"
    assert manifest["tokenizer"]["vocab_size"] == 257
    assert manifest["format_version"] == 1


def test_quantization_error_within_bound(tiny_trained_checkpoint, tmp_path):
    ckpt, tok_p, _ = tiny_trained_checkpoint
    out = tmp_path / "model.nexus"
    export_nexus(ckpt, tok_p, out)
    rep = verify_export(ckpt, out)
    assert rep["worst_abs_error"] >= 0
    assert rep["tensors"] > 0


def test_dequantized_model_matches_logits(tiny_trained_checkpoint, tmp_path):
    """Dequantized .nexus weights must produce (nearly) the same logits as the original."""
    import zipfile, json
    ckpt, tok_p, _ = tiny_trained_checkpoint
    out = tmp_path / "model.nexus"
    export_nexus(ckpt, tok_p, out)

    from nexus.train import load_checkpoint
    from nexus.config import ModelConfig
    ck = load_checkpoint(ckpt)
    cfg = ModelConfig.from_dict(ck["config"])
    model = TransformerLM(cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    cfg2, state, eos = load_nexus_state(out)
    model2 = TransformerLM(cfg2)
    model2.load_state_dict(state)
    model2.eval()
    assert eos == 256

    x = torch.randint(0, cfg.vocab_size, (1, 32))
    y = torch.roll(x, -1, dims=1)
    with torch.no_grad():
        l1, loss1 = model(x, y)
        l2, loss2 = model2(x, y)
    diff = (l1 - l2).abs().max().item()
    assert diff < 0.25, f"dequantized logits drifted too far: max diff {diff}"
    assert abs(loss1.item() - loss2.item()) < 0.15


def test_manifest_tokenizer_matches_saved_tokenizer(tiny_trained_checkpoint, tmp_path):
    ckpt, tok_p, _ = tiny_trained_checkpoint
    out = tmp_path / "model.nexus"
    export_nexus(ckpt, tok_p, out)
    import zipfile, json
    with zipfile.ZipFile(out) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    tok = BPETokenizer.load_json(tok_p)
    assert manifest["tokenizer"]["merges"] == [list(m) for m in tok.merges]
    assert manifest["tokenizer"]["eos_token_id"] == tok.eos_token_id
    assert manifest["config"]["vocab_size"] == tok.vocab_size


def test_corrupted_weights_detected(tiny_trained_checkpoint, tmp_path):
    """Integrity check must catch a tampered weights.bin."""
    import zipfile, json, hashlib
    ckpt, tok_p, _ = tiny_trained_checkpoint
    out = tmp_path / "model.nexus"
    export_nexus(ckpt, tok_p, out)

    with zipfile.ZipFile(out) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        weights = bytearray(zf.read("weights.bin"))
    weights[10] ^= 0xFF  # flip one byte
    tampered = tmp_path / "tampered.nexus"
    with zipfile.ZipFile(tampered, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        zf.writestr("weights.bin", bytes(weights))
    with pytest.raises(AssertionError):
        load_nexus_state(tampered)
