"""NEXUS-LM: GPT-style causal Transformer language model.

Implemented from scratch (random init, no pretrained weights, no third-party LLM).
Untied embedding/head, pre-LayerNorm blocks, causal self-attention, GELU MLP,
learned absolute position embeddings (chosen for exact portability to on-device
Kotlin inference), GPT-2-style weight init.
"""
import json
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NexusConfig:
    # Target production config = configs/nexus_500m.json (~500M params)
    vocab_size: int = 8192
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    d_ff: int = 1024
    block_size: int = 256
    dropout: float = 0.0
    layer_norm_epsilon: float = 1e-5

    def num_params(self, include_embedding: bool = True) -> int:
        d, L, V = self.d_model, self.n_layers, self.vocab_size
        # per layer: attn (4d^2 + 4d bias) + mlp (2*d*d_ff + d + d_ff bias) + 2 LN (2d each)
        attn = 4 * d * d + 4 * d
        mlp = d * self.d_ff + self.d_ff + self.d_ff * d + d
        ln = 4 * d
        per_layer = attn + mlp + ln
        total = L * per_layer
        total += V * d  # token embedding
        total += V * d  # lm head (untied)
        total += 2 * d + self.block_size * d  # final LN (w+b) + pos emb
        if include_embedding:
            return total
        return total - 2 * V * d

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json_file(path: str) -> "NexusConfig":
        with open(path) as f:
            d = json.load(f)
        valid = {f.name for f in NexusConfig.__dataclass_fields__.values()} if hasattr(NexusConfig, "__dataclass_fields__") else None
        if valid is not None:
            d = {k: v for k, v in d.items() if k in valid}
        return NexusConfig(**d)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: NexusConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                             .view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # memory-efficient fused kernel on CPU (torch >= 2.0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


def cfg_split(C, n):
    return [C] * n


class Block(nn.Module):
    def __init__(self, cfg: NexusConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_epsilon)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_epsilon)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NexusLM(nn.Module):
    """Decoder-only Transformer LM. Trained from random init."""

    def __init__(self, cfg: NexusConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_epsilon)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"T={T} > block_size={self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int = 40):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

    def exact_param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        non_emb = total - self.tok_emb.weight.numel() - self.head.weight.numel()
        return {"total": total, "non_embedding": non_emb,
                "config_check": self.cfg.num_params()}
