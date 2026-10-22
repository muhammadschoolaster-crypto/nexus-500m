import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from nexus.config import ModelConfig, get_preset
from nexus.model import TransformerLM


def _analytic_params(cfg: ModelConfig, include_pos: bool = True) -> int:
    d, L, V, T = cfg.d_model, cfg.n_layers, cfg.vocab_size, cfg.ctx_len
    per_block = 12 * d * d + 4 * d          # attn(3d^2+d^2) + mlp(4d^2+4d^2) + 2 LN(2d each)
    total = L * per_block + V * d           # token embeddings (tied head adds nothing)
    if include_pos:
        total += T * d                     # learned positional embeddings
    total += 2 * d                          # final LayerNorm
    return total


def test_demo_param_count_exact():
    cfg = get_preset("nexus_demo")
    model = TransformerLM(cfg)
    assert model.num_parameters() == _analytic_params(cfg)


def test_smoke_param_count_exact():
    cfg = get_preset("nexus_smoke")
    model = TransformerLM(cfg)
    assert model.num_parameters() == _analytic_params(cfg)


def test_500m_param_count_via_meta_device():
    """The 500M config is verified EXACTLY without allocating real weights."""
    pytest.importorskip("torch")
    from nexus.count_params import exact_param_table
    cfg = get_preset("nexus_500m")
    table, total = exact_param_table(cfg)
    expected = _analytic_params(cfg)
    assert total == expected
    assert 490_000_000 <= total <= 520_000_000
    # sanity: tied embeddings mean the unembedding shares storage
    assert sum(v for k, v in table.items() if k.startswith("wte")) == cfg.vocab_size * cfg.d_model


def test_causal_masking():
    """Changing future tokens must not change logits at earlier positions."""
    torch.manual_seed(0)
    cfg = get_preset("nexus_smoke")
    model = TransformerLM(cfg)
    model.eval()
    x1 = torch.randint(0, cfg.vocab_size, (1, 16))
    x2 = x1.clone()
    x2[0, 12:] = (x2[0, 12:] + 7) % cfg.vocab_size  # change only positions >= 12
    with torch.no_grad():
        l1, _ = model(x1)
        l2, _ = model(x2)
    assert torch.allclose(l1[0, :12], l2[0, :12], atol=1e-5), "future tokens leaked into past logits"
    assert not torch.allclose(l1[0, 12:], l2[0, 12:], atol=1e-5)


def test_output_shapes_and_loss():
    torch.manual_seed(1)
    cfg = get_preset("nexus_smoke")
    model = TransformerLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, cfg.ctx_len))
    y = torch.randint(0, cfg.vocab_size, (3, cfg.ctx_len))
    logits, loss = model(x, y)
    assert logits.shape == (3, cfg.ctx_len, cfg.vocab_size)
    assert torch.isfinite(loss)
    # untrained model loss should be near ln(V) = ln(257) ~ 5.55
    assert 4.5 < loss.item() < 7.0


def test_ctx_len_guard():
    cfg = get_preset("nexus_smoke")
    model = TransformerLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, cfg.ctx_len + 1))
    with pytest.raises(ValueError):
        model(x)


def test_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(d_model=100, n_heads=3)      # 100 % 3 != 0
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=200)              # below 256 bytes + EOS


def test_config_hash_stable_and_sensitive():
    a = get_preset("nexus_demo")
    b = get_preset("nexus_demo")
    assert a.config_hash() == b.config_hash()
    c = ModelConfig(**{**a.to_dict(), "d_model": 132})   # still divisible by 4 heads
    assert a.config_hash() != c.config_hash()


def test_json_config_files_match_presets():
    for name in ("nexus_500m", "nexus_demo", "nexus_smoke"):
        disk = ModelConfig.load_json(Path(__file__).resolve().parents[1] / "configs" / f"{name}.json")
        code = get_preset(name)
        assert disk == code, f"configs/{name}.json drifted from code preset"
