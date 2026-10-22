#!/usr/bin/env python3
"""Build instruct-only bins, then run a short real fine-tune (instruction tuning)."""
import os
import sys

ROOT = "/home/z/my-project/nexus-500m"
sys.path.insert(0, ROOT)
import numpy as np
from nexus_lm.tokenizer import NexusTokenizer
from nexus_lm.data import DOMAIN_TAGS

CORPUS = os.path.join(ROOT, "training", "data", "corpus")
PREFIX = os.path.join(ROOT, "training", "data", "nexus_instruct")

tok = NexusTokenizer.load(os.path.join(ROOT, "training", "tokenizer.json"))
path = os.path.join(CORPUS, "instruct", "synthetic_instructions.txt")
text = open(path, encoding="utf-8").read()
ids = np.array(tok.encode("\n<|instruct|>\n" + text + "\n<|end|>\n"), dtype=np.uint16)
n_val = max(1, int(len(ids) * 0.05))
val = ids[len(ids) - n_val:]
train = ids[:len(ids) - n_val]
np.save(PREFIX + "_train.bin", train)
np.save(PREFIX + "_val.bin", val)
print(f"instruct tokens: train={len(train):,} val={len(val):,}")
