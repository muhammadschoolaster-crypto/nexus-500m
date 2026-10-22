"""Model configuration system for NEXUS models.

Every checkpoint records its full ModelConfig so any model file can always be
reconstructed exactly and compatibility-checked before loading.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    name: str = "nexus"
    vocab_size: int = 2048          # bytes(256) + learned merges + 1 EOS token
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    d_ff: int = 0                   # 0 -> 4 * d_model
    ctx_len: int = 256
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_ff <= 0:
            self.d_ff = 4 * self.d_model
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.vocab_size < 257:
            # 256 byte tokens + at least the EOS token
            raise ValueError(f"vocab_size ({self.vocab_size}) must be >= 257 (256 bytes + EOS)")

    # --- serialization -------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load_json(cls, path: str | Path) -> "ModelConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


PRESETS = {
    # The full target configuration (~500M parameters; exact count printed by
    # ``python -m nexus.count_params`` and stored in artifacts/param_report.json).
    "nexus_500m": dict(
        name="nexus_500m", vocab_size=32768, d_model=1760, n_layers=12,
        n_heads=16, ctx_len=2048, dropout=0.0,
    ),
    # The scale actually trained in this CPU-only environment and shipped in
    # the Android app as the offline demo model.
    "nexus_demo": dict(
        name="nexus_demo", vocab_size=2048, d_model=128, n_layers=3,
        n_heads=4, ctx_len=256, dropout=0.0,
    ),
    # Even smaller; used by fast unit tests (raw byte vocabulary + EOS).
    "nexus_smoke": dict(
        name="nexus_smoke", vocab_size=257, d_model=64, n_layers=2,
        n_heads=4, ctx_len=64, dropout=0.0,
    ),
}


def get_preset(name: str) -> ModelConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset '{name}' (available: {sorted(PRESETS)})")
    return ModelConfig(**PRESETS[name])
