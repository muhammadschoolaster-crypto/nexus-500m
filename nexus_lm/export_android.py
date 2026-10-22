"""Export a trained NEXUS checkpoint for on-device Kotlin inference.

Outputs (into --outdir):
  nexus_model.bin    - all weights, fp32 little-endian, in the exact tensor
                       order documented in TensorPlan (see HEADER below)
  nexus_config.json  - architecture + tokenizer files copied alongside
  tokenizer_vocab.json, tokenizer_merges.json
  export_manifest.json - tensor shapes/offsets so the loader can be verified
"""
import argparse
import json
import os
import struct

import numpy as np
import torch

from .model import NexusLM, NexusConfig
from .tokenizer import NexusTokenizer

# Tensor plan (loader reads in this exact order):
#   tok_emb [V, d] (row-major)
#   pos_emb [T, d]
#   per layer i in 0..L-1:
#     ln1_w [d], ln1_b [d]
#     qkv_w [3d, d], qkv_b [3d]      (PyTorch Linear weight is [out, in])
#     proj_w [d, d], proj_b [d]
#     ln2_w [d], ln2_b [d]
#     fc_w [ff, d], fc_b [ff]
#     proj2_w [d, ff], proj2_b [d]
#   lnf_w [d], lnf_b [d]
#   head_w [V, d]  (no bias)
TENSOR_PLAN = """tok_emb[V,d]; pos_emb[T,d];
for each layer: ln1_w[d],ln1_b[d], qkv_w[3d,d],qkv_b[3d], proj_w[d,d],proj_b[d],
ln2_w[d],ln2_b[d], fc_w[ff,d],fc_b[ff], proj2_w[d,ff],proj2_b[d];
lnf_w[d],lnf_b[d]; head_w[V,d]"""


def export_all(ckpt_path, tok_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = NexusConfig(**ck["cfg"])
    model = NexusLM(cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    sd = model.state_dict()
    d, V, T, L, ff = cfg.d_model, cfg.vocab_size, cfg.block_size, cfg.n_layers, cfg.d_ff

    tensors = []
    tensors.append(("tok_emb", sd["tok_emb.weight"].float().numpy()))
    tensors.append(("pos_emb", sd["pos_emb.weight"].float().numpy()))
    for i in range(L):
        p = f"blocks.{i}."
        tensors.append((f"L{i}.ln1_w", sd[p + "ln1.weight"].numpy()))
        tensors.append((f"L{i}.ln1_b", sd[p + "ln1.bias"].numpy()))
        tensors.append((f"L{i}.qkv_w", sd[p + "attn.qkv.weight"].numpy()))
        tensors.append((f"L{i}.qkv_b", sd[p + "attn.qkv.bias"].numpy()))
        tensors.append((f"L{i}.proj_w", sd[p + "attn.proj.weight"].numpy()))
        tensors.append((f"L{i}.proj_b", sd[p + "attn.proj.bias"].numpy()))
        tensors.append((f"L{i}.ln2_w", sd[p + "ln2.weight"].numpy()))
        tensors.append((f"L{i}.ln2_b", sd[p + "ln2.bias"].numpy()))
        tensors.append((f"L{i}.fc_w", sd[p + "mlp.0.weight"].numpy()))
        tensors.append((f"L{i}.fc_b", sd[p + "mlp.0.bias"].numpy()))
        tensors.append((f"L{i}.proj2_w", sd[p + "mlp.2.weight"].numpy()))
        tensors.append((f"L{i}.proj2_b", sd[p + "mlp.2.bias"].numpy()))
    tensors.append(("lnf_w", sd["ln_f.weight"].numpy()))
    tensors.append(("lnf_b", sd["ln_f.bias"].numpy()))
    tensors.append(("head_w", sd["head.weight"].numpy()))

    n_floats = sum(int(np.prod(t.shape)) for _, t in tensors)
    bin_path = os.path.join(outdir, "nexus_model.bin")
    with open(bin_path, "wb") as f:
        for name, arr in tensors:
            flat = np.ascontiguousarray(arr, dtype=np.float32).ravel()
            f.write(flat.tobytes())
    size = os.path.getsize(bin_path)
    expected = n_floats * 4
    assert size == expected, f"size mismatch {size} != {expected}"

    cfg_d = {
        "name": "NEXUS-PoC",
        "architecture": "gpt-style decoder-only transformer (pre-LN, GELU, untied head, learned pos emb)",
        "d_model": d, "n_layers": L, "n_heads": cfg.n_heads, "d_ff": ff,
        "block_size": T, "vocab_size": V,
        "params_total": model.exact_param_count()["total"],
        "bin_bytes": size, "dtype": "fp32-le",
        "tensor_plan": TENSOR_PLAN,
    }
    with open(os.path.join(outdir, "nexus_config.json"), "w") as f:
        json.dump(cfg_d, f, indent=2)

    tok = NexusTokenizer.load(tok_path)
    with open(os.path.join(outdir, "tokenizer_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(tok.vocab, f, ensure_ascii=False)
    with open(os.path.join(outdir, "tokenizer_merges.json"), "w", encoding="utf-8") as f:
        json.dump(tok.merges, f, ensure_ascii=False)

    manifest = {"tensors": [{"name": n, "shape": list(t.shape), "dtype": "fp32",
                            "offset_floats": 0} for n, t in tensors],
                "n_floats": n_floats, "bytes": size}
    off = 0
    for m, (_, t) in zip(manifest["tensors"], tensors):
        m["offset_floats"] = off
        off += int(np.prod(t.shape))
    with open(os.path.join(outdir, "export_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"exported {len(tensors)} tensors, {n_floats:,} floats, {size:,} bytes -> {outdir}")
    return cfg_d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    export_all(a.ckpt, a.tokenizer, a.outdir)
