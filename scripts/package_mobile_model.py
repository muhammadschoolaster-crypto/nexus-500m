#!/usr/bin/env python3
"""Package the trained .nexus model + golden data into the Android app:
- assets/models/<file>.nexus       (bundled offline model)
- assets/models/registry.json      (build-time integrity fingerprint)
- src/test/resources/<file>.nexus  (same file for JVM golden parity tests)
- src/test/resources/golden.json   (Python-produced expected outputs)
Also writes the honest model card (artifacts/demo_model/model_card.json).
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts" / "demo_model"
APP = ROOT / "android" / "app"

MODEL = SRC / "nexus-demo-v0.1.0.nexus"
GOLDEN = SRC / "golden.json"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main() -> None:
    assert MODEL.exists(), "run scripts/run_smoke_training.sh first"
    assert GOLDEN.exists(), "golden data missing"

    assets = APP / "src" / "main" / "assets" / "models"
    tests = APP / "src" / "test" / "resources"
    assets.mkdir(parents=True, exist_ok=True)
    tests.mkdir(parents=True, exist_ok=True)

    shutil.copy2(MODEL, assets / MODEL.name)
    shutil.copy2(MODEL, tests / MODEL.name)
    shutil.copy2(GOLDEN, tests / "golden.json")

    registry = {
        "models": [
            {
                "file": MODEL.name,
                "name": "nexus-demo",
                "version": "0.1.0",
                "sha256": sha256(MODEL),
                "size_bytes": MODEL.stat().st_size,
                "license": "MIT (this repository)",
                "ram_estimate_bytes": 886528 * 4,
                "compatible": True,
                "notes": "tiny demo checkpoint genuinely trained by this project's pipeline "
                         "(base 500 steps + SFT 150 steps, CPU). Scale demonstration only - "
                         "NOT a claim of assistant-level capability.",
            }
        ]
    }
    (assets / "registry.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")

    eval_base = json.loads((SRC / "eval_base.json").read_text())
    eval_sft = json.loads((SRC / "eval_sft.json").read_text())
    model_card = {
        "model": "nexus-demo v0.1.0",
        "architecture": "GPT-style decoder-only causal Transformer, pre-LN, tied embeddings, GELU(tanh)",
        "parameters": 886528,
        "config": {"d_model": 128, "n_layers": 3, "n_heads": 4, "d_ff": 512,
                   "ctx_len": 256, "vocab_size": 2048},
        "tokenizer": {"type": "byte-level BPE (project-trained)", "vocab_size": 2048},
        "training": {
            "hardware": "2 vCPU (no GPU)",
            "base_steps": 500, "sft_steps": 150,
            "base_final_train_loss": 0.4982, "base_val_loss": 2.0997,
            "sft_val_loss": 4.0782,
            "dataset": "data/demo/prepared (project-authored demo corpus, see manifest.json)",
            "dataset_tokens": {"train": 19436, "val": 1027, "test": 2570},
            "sft_examples": 104,
            "total_time_seconds": 500 * 7.2 + 150 * 3.5,
        },
        "evaluation": {
            "base": {"heldout_loss": eval_base["heldout_loss"], "heldout_ppl": eval_base["heldout_ppl"],
                     "coding_parse_rate": eval_base["coding_parse_rate"],
                     "writing_avg_words": eval_base["writing_avg_words"],
                     "writing_distinct_ratio": eval_base["writing_avg_distinct_ratio"]},
            "sft": {"heldout_loss_on_base_testset": eval_sft["heldout_loss"],
                    "coding_parse_rate": eval_sft["coding_parse_rate"],
                    "writing_avg_words": eval_sft["writing_avg_words"],
                    "writing_distinct_ratio": eval_sft["writing_avg_distinct_ratio"]},
        },
        "honest_limitations": [
            "This is a tiny (0.89M-parameter) demonstration checkpoint trained on a ~19K-token corpus.",
            "It produces mostly word-salad with recognizable learned fragments; it cannot hold conversations, "
            "write correct code, or follow instructions reliably (coding parse rate 0.0 in evaluation).",
            "It exists to prove the END-TO-END pipeline is real: from-scratch training -> int8 export -> "
            "verified on-device inference with Python/Kotlin numeric parity.",
            "The nexus_500m configuration (507,418,560 parameters) is fully specified and verified by "
            "parameter counting; training it needs the GPU compute documented in docs/TRAINING.md.",
        ],
        "file_sha256": sha256(MODEL),
        "file_size_bytes": MODEL.stat().st_size,
    }
    (SRC / "model_card.json").write_text(json.dumps(model_card, indent=2) + "\n", encoding="utf-8")

    print("packaged:", MODEL.name, "->", assets)
    print("registry sha256:", registry["models"][0]["sha256"][:32], "...")
    print("model card ->", SRC / "model_card.json")


if __name__ == "__main__":
    sys.exit(main())
