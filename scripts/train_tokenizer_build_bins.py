#!/usr/bin/env python3
"""Train the NEXUS BPE tokenizer on the corpus, then build train/val bins."""
import os
import sys
import json
import time

ROOT = "/home/z/my-project/nexus-500m"
sys.path.insert(0, ROOT)
from nexus_lm.tokenizer import NexusTokenizer
from nexus_lm.data import build_bin

CORPUS = os.path.join(ROOT, "training", "data", "corpus")
TOK_PATH = os.path.join(ROOT, "training", "tokenizer.json")
PREFIX = os.path.join(ROOT, "training", "data", "nexus_poc")

texts = []
for dirpath, _, files in os.walk(CORPUS):
    for fn in sorted(files):
        with open(os.path.join(dirpath, fn), encoding="utf-8", errors="replace") as f:
            texts.append(f.read())
print(f"reading corpus: {len(texts)} files, {sum(len(t) for t in texts)/1e6:.2f}M chars")

t0 = time.time()
tok = NexusTokenizer.train(texts, target_vocab=4096)
tok.save(TOK_PATH)
print("saved tokenizer")

stats = tok.stats(texts)
print("tokenizer stats:", json.dumps(stats, indent=1))

# roundtrip integrity check on a sample
sample = "def hello():\n    print('nexus is online')\n"
assert tok.decode(tok.encode(sample)) == sample, "roundtrip failed"
print("roundtrip check OK")

info = build_bin(CORPUS, tok, PREFIX)
with open(os.path.join(ROOT, "training", "data", "dataset_info.json"), "w") as f:
    json.dump(info, f, indent=2)
print(f"train tokens: {info['train_tokens']:,}  val tokens: {info['val_tokens']:,}")
print(f"unique pretokens covered: {len(tok.vocab)} vocab entries")
