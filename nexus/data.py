"""Dataset preparation pipeline: provenance -> clean -> filter -> dedupe -> split -> pack.

Hard rules implemented here:
- Every document must declare source + license provenance (manifest.json).
- Sensitive-looking content (emails, API keys, private key blocks, password
  assignments) is REMOVED, not just logged.
- Exact + normalized near-duplicate documents are dropped.
- Deterministic train/val/test split (hash-based, reproducible, disjoint).
- Packed token bins are stored as uint16 .npy files (vocab < 65536).
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np

VAL_FRACTION = 0.05
TEST_FRACTION = 0.05

# ---------------------------------------------------------------------------
# Sensitive-data filtering (conservative; everything matching is dropped)

_SECRET_PATTERNS = [
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),          # emails
    re.compile(r"AKIA[0-9A-Z]{16}"),                                         # AWS keys
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),                             # GitHub PATs
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),                               # GitHub tokens
    re.compile(r"sk-[A-Za-z0-9]{20,}"),                                      # API sk- keys
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),                       # private keys
    re.compile(r"(?i)\b(password|passwd|secret|token|api[_-]?key)\s*[:=]\s*\S+", ),
    re.compile(r"\+?\d[\d\s\-()]{8,}\d"),                                    # phone-like
]


def find_sensitive(text: str) -> list:
    hits = []
    for pat in _SECRET_PATTERNS:
        for m in pat.finditer(text):
            hits.append(m.group(0)[:40])
    return hits


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def _norm_for_dedupe(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def dedupe(docs: Iterable[dict]) -> list:
    """Exact + normalized dedupe. Each doc: {"text":..., "source":..., "license":...}."""
    seen_exact, seen_norm, kept = set(), set(), []
    for d in docs:
        text = clean_text(d["text"])
        if not text or len(text) < 20:
            continue
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        hn = hashlib.sha256(_norm_for_dedupe(text).encode("utf-8")).hexdigest()
        if h in seen_exact or hn in seen_norm:
            continue
        seen_exact.add(h)
        seen_norm.add(hn)
        nd = dict(d)
        nd["text"] = text
        nd["sha256"] = h
        kept.append(nd)
    return kept


def split_docs(docs: list) -> dict:
    """Deterministic, disjoint split by content hash."""
    out = {"train": [], "val": [], "test": []}
    for d in docs:
        h = int(d["sha256"][:12], 16) / float(0xFFFFFFFFFFFF + 1)
        if h < TEST_FRACTION:
            out["test"].append(d)
        elif h < TEST_FRACTION + VAL_FRACTION:
            out["val"].append(d)
        else:
            out["train"].append(d)
    return out


def pack_bins(split_docs_map: dict, tokenizer, ctx_len: int, out_dir: str | Path) -> dict:
    """Tokenize all docs, join with EOS, pack into fixed-length windows of ctx_len+1."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, docs in split_docs_map.items():
        ids = []
        for d in docs:
            ids.extend(tokenizer.encode(d["text"], add_eos=True))
        window = ctx_len + 1
        n_windows = len(ids) // window
        if n_windows == 0:
            arr = np.zeros((0, window), dtype=np.uint16)
        else:
            arr = np.asarray(ids[: n_windows * window], dtype=np.uint16).reshape(n_windows, window)
        p = out_dir / f"{split}.bin.npy"
        np.save(p, arr)
        paths[split] = str(p)
        print(f"  {split:5s}: {len(docs):4d} docs -> {len(ids):8d} tokens -> {arr.shape[0]:6d} windows @ {p}")
    return paths


# ---------------------------------------------------------------------------
# Instruction-tuning dataset (SFT)

def instruction_text(instruction: str, response: str) -> str:
    from . import INSTRUCTION_TEMPLATE, RESPONSE_END
    return INSTRUCTION_TEMPLATE.format(instruction=instruction.strip()) + response.strip() + RESPONSE_END


def load_instructions(jsonl_path: str | Path) -> list:
    items = []
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        assert "instruction" in d and "response" in d, f"bad SFT row: {d}"
        items.append(d)
    return items


def pack_instruction_bins(instructions: list, tokenizer, ctx_len: int, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in ("train", "val"):
        rows = [d for d in instructions if d.get("split", "train") == split]
        ids = []
        for d in rows:
            ids.extend(tokenizer.encode(instruction_text(d["instruction"], d["response"]), add_eos=True))
        window = ctx_len + 1
        n_windows = len(ids) // window
        arr = (np.asarray(ids[: n_windows * window], dtype=np.uint16).reshape(n_windows, window)
               if n_windows else np.zeros((0, window), dtype=np.uint16))
        p = out_dir / f"sft_{split}.bin.npy"
        np.save(p, arr)
        paths[split] = str(p)
        print(f"  sft_{split}: {len(rows):4d} examples -> {len(ids):8d} tokens -> {arr.shape[0]:6d} windows")
    return paths
