"""Dataset builder: tokenize corpus files -> train.bin / val.bin (uint16 ids).

Corpus sections are wrapped in domain tags so the model learns domain
switching, e.g.:
  <|text|> ... <|code:python|> ... <|write:email|> ...
Tags are plain text tokens (part of vocab via corpus).
The final 2% of each source is reserved for validation (temporal-ish split
within file, no doc leakage across train/val from the same contiguous block).
"""
import os
import numpy as np

from .tokenizer import NexusTokenizer

DOMAIN_TAGS = {
    ".txt": "<|text|>",
    ".md": "<|text|>",
    ".py": "<|code:python|>",
    ".js": "<|code:js|>",
    ".ts": "<|code:ts|>",
    ".html": "<|code:html|>",
    ".css": "<|code:css|>",
    ".sql": "<|code:sql|>",
    ".sh": "<|code:shell|>",
    ".c": "<|code:c|>",
    ".cpp": "<|code:cpp|>",
    ".java": "<|code:java|>",
    ".kt": "<|code:kotlin|>",
    ".jsonl": "<|instruct|>",
}


def build_bin(corpus_dir: str, tok: NexusTokenizer, out_prefix: str, val_frac: float = 0.02):
    train_ids, val_ids = [], []
    manifest = []
    for root, _, files in os.walk(corpus_dir):
        for fn in sorted(files):
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, corpus_dir)
            ext = os.path.splitext(fn)[1].lower()
            tag = DOMAIN_TAGS.get(ext, "<|text|>")
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            if not text.strip():
                continue
            ids = tok.encode(f"\n{tag}\n" + text + "\n<|end|>\n")
            n_val = max(1, int(len(ids) * val_frac))
            # reserve a contiguous tail block of each doc for validation
            val_ids.extend(ids[len(ids) - n_val:])
            train_ids.extend(ids[:len(ids) - n_val])
            manifest.append({"file": rel, "chars": len(text), "tokens": len(ids), "tag": tag})
    train = np.array(train_ids, dtype=np.uint16)
    val = np.array(val_ids, dtype=np.uint16)
    np.save(out_prefix + "_train.bin", train)
    np.save(out_prefix + "_val.bin", val)
    return {"train_tokens": len(train), "val_tokens": len(val), "files": manifest}


def get_batch(split: str, prefix: str, block_size: int, batch_size: int):
    data = np.load(prefix + f"_{split}.bin.npy")
    ix = np.random.randint(0, max(1, len(data) - block_size - 1), size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + block_size + 1] for i in ix])
    return (torch_from_numpy(x), torch_from_numpy(y))


def torch_from_numpy(a):
    import torch
    return torch.from_numpy(a.astype(np.int64))
