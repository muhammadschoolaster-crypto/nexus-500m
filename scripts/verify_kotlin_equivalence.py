#!/usr/bin/env python3
"""Kotlin-equivalence verification.

Replicates the EXACT algorithm of android NexusEngine.kt in Python:
  - reads nexus_model.bin as a flat fp32 array per the tensor plan
  - layerNorm / matvec / GELU with the SAME Abramowitz-Stegun erf approx
  - incremental single-token forward with K/V cache
Compares final-token logits against the PyTorch NexusLM forward on the same
prompt. If they agree within fp32 tolerance, the on-device engine math is
proven equivalent (modulo accumulation order).
"""
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = "/home/z/my-project/nexus-500m"
sys.path.insert(0, ROOT)
from nexus_lm.model import NexusLM, NexusConfig
from nexus_lm.tokenizer import NexusTokenizer

ASSETS = os.path.join(ROOT, "android", "NexusAssistant", "app", "src", "main", "assets")

cfg = json.load(open(os.path.join(ASSETS, "nexus_config.json")))
tok = NexusTokenizer.load(os.path.join(ROOT, "training", "tokenizer.json"))
d, L, H, ff, T, V = (cfg[k] for k in ("d_model", "n_layers", "n_heads", "d_ff",
                                       "block_size", "vocab_size"))
dh = d // H
w = np.fromfile(os.path.join(ASSETS, "nexus_model.bin"), dtype="<f4")
assert w.size == cfg["bin_bytes"] // 4, "size mismatch"

# ---- offsets (mirror computeOffsets) ----
o = 0
oTok = o; o += V * d
oPos = o; o += T * d
oL = []
for _ in range(L):
    row = [o]; o += d          # ln1w
    row.append(o); o += d      # ln1b
    row.append(o); o += 3*d*d  # qkvW
    row.append(o); o += 3*d    # qkvB
    row.append(o); o += d*d    # projW
    row.append(o); o += d      # projB
    row.append(o); o += d      # ln2w
    row.append(o); o += d      # ln2b
    row.append(o); o += ff*d   # fcW
    row.append(o); o += ff     # fcB
    row.append(o); o += d*ff   # proj2W
    row.append(o); o += d      # proj2B
    oL.append(row)
oLnfw = o; o += d
oLnfb = o; o += d
oHead = o; o += V * d
assert o == w.size, f"offset plan mismatch {o} != {w.size}"

def erf_as(x):  # A&S 7.1.26 exactly as in Kotlin (float32)
    x = np.float64(x)
    t = 1.0 / (1.0 + 0.5 * abs(x))
    tau = t * math.exp(-x*x - 1.26551223 + t*(1.00002368 +
          t*(0.37409196 + t*(0.09678418 + t*(-0.18628806 +
          t*(0.27886807 + t*(-1.13520398 + t*(1.48851587 +
          t*(-0.82215223 + t*0.17087277)))))))))
    return np.float32(1.0 - tau if x >= 0 else tau - 1.0)

def layer_norm(x, wOff, bOff):
    mean = np.float32(x.mean())
    v = x - mean
    vr = np.float32((v.astype(np.float64)*v).sum() / d)  # order like Kotlin loop
    inv = np.float32(1.0 / math.sqrt(float(vr) + 1e-5))
    return (x - mean) * inv * w[wOff:wOff+d] + w[bOff:bOff+d]

def matvec_bias(bOff, wOff, x, outDim, inDim):
    out = w[bOff:bOff+outDim].copy()
    W = w[wOff:wOff+outDim*inDim].reshape(outDim, inDim)
    out += (W @ x).astype(np.float32)
    return out

# ---- Kotlin forward (incremental, per token) ----
kCache = [np.zeros((T, d), dtype=np.float32) for _ in range(L)]
vCache = [np.zeros((T, d), dtype=np.float32) for _ in range(L)]

def forward_token(id_, p):
    x = w[oTok + id_*d : oTok + (id_+1)*d].copy() + w[oPos + p*d : oPos + (p+1)*d]
    scale = np.float32(1.0 / math.sqrt(dh))
    for l in range(L):
        r = oL[l]
        h = layer_norm(x, r[0], r[1])
        qkv = matvec_bias(r[3], r[2], h, 3*d, d)
        kCache[l][p] = qkv[d:2*d]
        vCache[l][p] = qkv[2*d:3*d]
        attn = np.zeros(d, dtype=np.float32)
        for hh in range(H):
            ho = hh*dh
            scores = (kCache[l][:p+1, ho:ho+dh] @ qkv[ho:ho+dh]) * scale
            mx = scores.max()
            e = np.exp(scores - mx).astype(np.float32)
            attn[ho:ho+dh] = ((e[:, None] * vCache[l][:p+1, ho:ho+dh]).sum(0)
                              / e.sum()).astype(np.float32)
        hproj = matvec_bias(r[5], r[4], attn, d, d)
        x = x + hproj
        h2 = layer_norm(x, r[6], r[7])
        f = matvec_bias(r[9], r[8], h2, ff, d)
        g = (np.float32(0.5) * f * (np.float32(1.0) + np.array(
            [erf_as(v / np.float32(math.sqrt(2.0))) for v in f]))).astype(np.float32)
        out2 = matvec_bias(r[11], r[10], g, d, ff)
        x = x + out2
    hf = layer_norm(x, oLnfw, oLnfb)
    logits = (w[oHead:oHead+V*d].reshape(V, d) @ hf).astype(np.float32)
    return logits

# ---- compare with torch on a real prompt ----
prompt = "\n<|text|>\nThe little robot booted its offline language model and"
ids = tok.encode(prompt)
print(f"prompt tokens: {len(ids)}")

ck = torch.load(os.path.join(ROOT, "training", "runs", "instruct", "nexus_best.pt"),
                map_location="cpu")
mcfg = NexusConfig(**ck["cfg"])
model = NexusLM(mcfg)
model.load_state_dict(ck["model"])
model.eval()
with torch.no_grad():
    ref_logits, _ = model(torch.tensor([ids]))
ref_last = ref_logits[0, -1].numpy()

for p, tid in enumerate(ids):
    kt_logits = forward_token(tid, p)

diff = np.abs(kt_logits.astype(np.float64) - ref_last.astype(np.float64))
top5_torch = np.argsort(ref_last)[-5:][::-1]
top5_kotlin = np.argsort(kt_logits)[-5:][::-1]
print(f"max |logit diff| (kotlin-sim vs torch): {diff.max():.6f}")
print(f"mean |logit diff|: {diff.mean():.6f}")
print(f"top-5 torch : {top5_torch.tolist()}")
print(f"top-5 kotlin: {top5_kotlin.tolist()}")
agreement = set(top5_torch.tolist()) & set(top5_kotlin.tolist())
print(f"top-5 overlap: {len(agreement)}/5")
assert diff.max() < 5e-3, "LOGIT DIVERGENCE TOO LARGE"
assert top5_torch.tolist() == top5_kotlin.tolist(), "ARGMAX/TOP5 MISMATCH"
print("KOTLIN ENGINE EQUIVALENCE: VERIFIED ✓")
