"""Evaluation harness: perplexity + coding checks + writing heuristics.

Honesty rules baked in: results are always tagged with model scale and the
number of evaluated items; tiny demo models are never presented as capable.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig
from .model import TransformerLM
from .tokenizer import BPETokenizer
from .train import PackedBins, evaluate_loss, load_checkpoint

CODING_PROMPTS = [
    "### Instruction:\nWrite a Python function named add that returns the sum of two numbers.\n### Response:\n",
    "### Instruction:\nWrite a Python function named is_even that returns True when a number is even.\n### Response:\n",
    "### Instruction:\nWrite a Python function named greet that returns a string Hello, World.\n### Response:\n",
    "### Instruction:\nWrite a Python for loop that prints the numbers from 1 to 3.\n### Response:\n",
    "### Instruction:\nWrite a Kotlin val declaration that stores the number 42.\n### Response:\n",
]

WRITING_PROMPTS = [
    "### Instruction:\nWrite a one-paragraph welcome email to a new team member.\n### Response:\n",
    "### Instruction:\nExplain in one short paragraph why backing up files is important.\n### Response:\n",
    "### Instruction:\nWrite a two-sentence summary of what a calendar reminder is.\n### Response:\n",
    "### Instruction:\nWrite a friendly one-paragraph note reminding a colleague about tomorrow's meeting.\n### Response:\n",
    "### Instruction:\nDescribe in one paragraph what an offline AI assistant is.\n### Response:\n",
]


def load_model(ckpt_path: str | Path) -> tuple:
    ck = load_checkpoint(ckpt_path)
    cfg = ModelConfig.from_dict(ck["config"])
    model = TransformerLM(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck


def extract_code(text: str) -> str:
    lines, in_code, buf = [], False, []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("```"):
            if in_code:
                break
            in_code = True
            continue
        if in_code:
            buf.append(ln)
    if buf:
        return "\n".join(buf)
    return text


@torch.no_grad()
def gen_text(model, tok, prompt: str, max_new: int = 48) -> str:
    from .generate import generate
    ids = tok.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    new = generate(model, x, max_new, temperature=0.7, top_k=40,
                   eos_id=tok.eos_token_id, rng=np.random.default_rng(3))
    return tok.decode(ids + new)


def check_python_parses(code: str) -> bool:
    if not code.strip():
        return False          # empty output is NOT valid generated code
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def writing_metrics(text: str) -> dict:
    words = text.split()
    if len(words) < 2:
        return {"words": len(words), "distinct_ratio": 0.0, "avg_word_len": 0.0}
    grams = list(zip(words[:-1], words[1:]))
    distinct = len(set(grams)) / len(grams)
    return {
        "words": len(words),
        "distinct_ratio": round(distinct, 3),           # low = repetitive babble
        "avg_word_len": round(sum(len(w) for w in words) / len(words), 2),
    }


def evaluate(ckpt: str | Path, tokenizer_path: str | Path, test_bins: str = "",
             label: str = "model", scale_note: str = "") -> dict:
    model, cfg, ck = load_model(ckpt)
    tok = BPETokenizer.load_json(tokenizer_path)

    report: dict = {
        "label": label,
        "config": cfg.to_dict(),
        "scale_note": scale_note,
        "params": model.num_parameters(),
        "checkpoint_step": ck.get("step"),
    }

    if test_bins and Path(test_bins).exists():
        ppl_bins = PackedBins(test_bins)
        if ppl_bins.n_windows > 0:
            loss = evaluate_loss(model, ppl_bins, batch_size=min(16, max(2, ppl_bins.n_windows)))
            report["heldout_loss"] = round(loss, 4)
            report["heldout_ppl"] = round(math.exp(min(loss, 100)), 3)

    coding_results = []
    for prompt in CODING_PROMPTS:
        full = gen_text(model, tok, prompt, max_new=48)
        continuation = full[len(prompt):]          # text the model added after the prompt
        code = extract_code(continuation)
        coding_results.append({"prompt": prompt[:60], "parses": check_python_parses(code),
                               "sample": continuation.strip()[:120]})
    report["coding_parse_rate"] = round(
        sum(1 for r in coding_results if r["parses"]) / len(coding_results), 3)
    report["coding_samples"] = coding_results

    writing_results = []
    for prompt in WRITING_PROMPTS:
        full = gen_text(model, tok, prompt, max_new=48)
        continuation = full[len(prompt):]
        writing_results.append({"prompt": prompt[:60], **writing_metrics(continuation),
                                "sample": continuation.strip()[:120]})
    report["writing_avg_words"] = round(
        float(np.mean([r["words"] for r in writing_results])), 1)
    report["writing_avg_distinct_ratio"] = round(
        float(np.mean([r["distinct_ratio"] for r in writing_results])), 3)
    report["writing_samples"] = writing_results
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="NEXUS evaluation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--test-bins", default="")
    p.add_argument("--label", default="model")
    p.add_argument("--scale-note", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rep = evaluate(a.checkpoint, a.tokenizer, a.test_bins, a.label, a.scale_note)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if not k.endswith("samples")}, indent=2))


if __name__ == "__main__":
    main()
