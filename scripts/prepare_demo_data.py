#!/usr/bin/env python3
"""Prepare the NEXUS demo dataset end to end (deterministic):

  seeds -> clean -> sensitive-filter -> dedupe -> split -> BPE tokenizer
        -> packed uint16 bins + instructions.jsonl + provenance manifest

Everything is deterministic (fixed seeds), so the same inputs always produce
byte-identical outputs. Corpus expansion is SYNTHETIC and labeled as such in
the manifest: paragraph-order shuffles of project-authored seed documents.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexus.data import (VAL_FRACTION, TEST_FRACTION, clean_text, dedupe,
                        find_sensitive, pack_bins, pack_instruction_bins)
from nexus.tokenizer import BPETokenizer

SEED = 20260101
DEMO_VOCAB = 2048          # 256 bytes + 1791 merges + 1 EOS
EXPANSION_VARIANTS = 8     # paragraph-shuffled variants per eligible document

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "data" / "demo" / "seeds"
OUT = ROOT / "data" / "demo" / "prepared"


def load_seed_docs() -> list:
    """Each '## ' section of each seed file becomes one document
    (explanation + code together), keeping documents substantial."""
    import re
    docs = []
    for path in sorted(SEEDS.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for sec in re.split(r"\n(?=## )", text):
            if len(sec.strip()) > 250:
                docs.append({
                    "text": sec.strip(),
                    "source": f"data/demo/seeds/{path.name}",
                    "license": "MIT (project-authored)",
                    "origin": "seed",
                })
    return docs


def expand(docs: list, rng: random.Random) -> list:
    """Deterministic synthetic expansion: shuffle paragraph order within docs
    that have >= 3 paragraphs. Honest label: 'synthetic_expansion'."""
    out = []
    for d in docs:
        out.append(d)
        paras = [p for p in d["text"].split("\n\n") if p.strip()]
        if len(paras) < 3:
            continue
        for _ in range(EXPANSION_VARIANTS):
            variant = paras[:]
            rng.shuffle(variant)
            out.append({
                "text": "\n\n".join(variant),
                "source": d["source"],
                "license": d["license"],
                "origin": "synthetic_expansion",
            })
    return out


def main() -> None:
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/6] loading seed documents ...")
    raw = load_seed_docs()
    print(f"      {len(raw)} seed documents")

    print("[2/6] cleaning + sensitive filtering ...")
    clean = []
    dropped = 0
    for d in raw:
        d["text"] = clean_text(d["text"])
        hits = find_sensitive(d["text"])
        if hits:
            dropped += 1
            continue
        clean.append(d)
    print(f"      kept {len(clean)}, dropped {dropped} (sensitive/empty)")

    print("[3/6] synthetic expansion (paragraph shuffles) ...")
    docs = expand(clean, rng)
    print(f"      {len(docs)} documents total")

    print("[4/6] dedupe ...")
    docs = dedupe(docs)
    print(f"      {len(docs)} unique documents")

    # deterministic split
    split_map = {"train": [], "val": [], "test": []}
    for d in docs:
        h = int(d["sha256"][:12], 16) / float(0xFFFFFFFFFFFF + 1)
        if h < TEST_FRACTION:
            split_map["test"].append(d)
        elif h < TEST_FRACTION + VAL_FRACTION:
            split_map["val"].append(d)
        else:
            split_map["train"].append(d)
    print(f"      split: train={len(split_map['train'])} val={len(split_map['val'])} test={len(split_map['test'])}")

    print("[5/6] training byte-level BPE tokenizer (vocab 2048) ...")
    tok = BPETokenizer()
    tok.train((d["text"] for d in split_map["train"]), vocab_size=DEMO_VOCAB)
    print(f"      vocab_size={tok.vocab_size} merges={len(tok.merges)} eos_id={tok.eos_token_id}")
    tok.save_json(OUT / "tokenizer.json")

    print("[6/6] packing token bins ...")
    bins = pack_bins(split_map, tok, ctx_len=256, out_dir=OUT)

    # SFT instructions: deterministic split by hash
    sft_rows = []
    for line in (SEEDS / "sft_instructions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        h = int(hashlib.sha256(d["instruction"].encode()).hexdigest()[:12], 16) / float(0xFFFFFFFFFFFF + 1)
        d["split"] = "val" if h < 0.12 else ("test" if h < 0.2 else "train")
        sft_rows.append(d)
    with (OUT / "instructions.jsonl").open("w", encoding="utf-8") as f:
        for d in sft_rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    sft_bins = pack_instruction_bins(sft_rows, tok, ctx_len=256, out_dir=OUT)

    manifest = {
        "dataset": "nexus-demo-corpus",
        "version": "0.1.0",
        "seed": SEED,
        "provenance": {
            "origin": "100% project-authored seed documents (code examples with explanations, "
                      "professional emails/letters, technical knowledge paragraphs, short stories, "
                      "instruction/response pairs) written for this project",
            "licenses": ["MIT (project)"],
            "expansion": "deterministic synthetic expansion by paragraph-order shuffling "
                         f"({EXPANSION_VARIANTS} variants per eligible document); every expanded "
                         "document is labeled origin=synthetic_expansion and traces back to its seed",
            "external_sources": "none. No web scraping, no third-party datasets.",
        },
        "pipeline": ["clean", "sensitive_filter", "dedupe_exact+normalized", "hash_split", "bpe_train", "pack"],
        "counts": {k: len(v) for k, v in split_map.items()},
        "sft_counts": {s: sum(1 for r in sft_rows if r["split"] == s) for s in ("train", "val", "test")},
        "tokenizer": {"vocab_size": tok.vocab_size, "merges": len(tok.merges), "eos_token_id": tok.eos_token_id},
        "bins": {**bins, **sft_bins},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("manifest ->", OUT / "manifest.json")
    print("DONE.")


if __name__ == "__main__":
    main()
