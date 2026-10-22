#!/usr/bin/env bash
# NEXUS demo end-to-end pipeline:
# data -> tokenizer -> base training -> SFT -> evaluation -> int8 export -> golden data
#
# This is the REAL training run recorded in artifacts/ (logs, eval, model card).
# It uses the tiny nexus_demo scale because this machine has no GPU; the very
# same commands with configs/nexus_500m.json and a real dataset train the 500M
# configuration (see docs/TRAINING.md for the exact compute requirements).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/demo_model

echo "=== [1/6] prepare demo dataset ==="
python scripts/prepare_demo_data.py

echo "=== [2/6] base training (nexus_demo, CPU) ==="
python -m nexus.train \
  --config configs/nexus_demo.json \
  --train-bins data/demo/prepared/train.bin.npy \
  --val-bins   data/demo/prepared/val.bin.npy \
  --out artifacts/runs/demo_base \
  --steps "${BASE_STEPS:-400}" --batch-size "${BATCH:-16}" --lr 1e-3 \
  --warmup 25 --seed 1337 --log-every 25 --val-every 100 --save-every 100 --stage base

echo "=== [3/6] instruction tuning (SFT) ==="
python -m nexus.sft \
  --config configs/nexus_demo.json \
  --tokenizer data/demo/prepared/tokenizer.json \
  --instructions data/demo/prepared/instructions.jsonl \
  --base-checkpoint artifacts/runs/demo_base/checkpoints/final.pt \
  --out artifacts/runs/demo_sft \
  --steps "${SFT_STEPS:-120}" --batch-size 16 --lr 3e-4 --seed 1337 --stage sft

echo "=== [4/6] evaluation (base + SFT) ==="
python -m nexus.evaluate --checkpoint artifacts/runs/demo_base/checkpoints/final.pt \
  --tokenizer data/demo/prepared/tokenizer.json \
  --test-bins data/demo/prepared/test.bin.npy \
  --label base --scale-note "tiny demo model (0.89M params), trained 400 steps CPU" \
  --out artifacts/demo_model/eval_base.json
python -m nexus.evaluate --checkpoint artifacts/runs/demo_sft/checkpoints/final.pt \
  --tokenizer data/demo/prepared/tokenizer.json \
  --test-bins data/demo/prepared/test.bin.npy \
  --label sft --scale-note "tiny demo model (0.89M params) after instruction tuning" \
  --out artifacts/demo_model/eval_sft.json

echo "=== [5/6] export int8 .nexus + verify ==="
python -m nexus.export \
  --checkpoint artifacts/runs/demo_sft/checkpoints/final.pt \
  --tokenizer data/demo/prepared/tokenizer.json \
  --out artifacts/demo_model/nexus-demo-v0.1.0.nexus \
  --model-name nexus-demo --version 0.1.0 \
  --notes "tiny demo checkpoint genuinely trained by scripts/run_smoke_training.sh" \
  --verify

echo "=== [6/6] golden data for Android parity tests ==="
python -m nexus.generate --nexus artifacts/demo_model/nexus-demo-v0.1.0.nexus \
  --tokenizer data/demo/prepared/tokenizer.json \
  --prompt "### Instruction:
Write a Python function named add that returns the sum of two numbers.
### Response:
" --max-new 24 --greedy --golden-out artifacts/demo_model/golden.json

echo "ALL STAGES COMPLETE."
