#!/usr/bin/env bash
# NEXUS build helper
set -euo pipefail

MODEL_DIR="training/runs/poc"
OUT_DIR="android/NexusAssistant/app/src/main/assets"

echo "[1/3] exporting model for android..."
python3 -m nexus_lm.export_android --ckpt "$MODEL_DIR/nexus_ckpt.pt"   --tokenizer training/tokenizer.json --outdir "$OUT_DIR"

echo "[2/3] verifying export..."
python3 - <<'PY'
import json, os
cfg = json.load(open("assets/nexus_config.json"))
assert cfg["bin_bytes"] == os.path.getsize("assets/nexus_model.bin"), "size mismatch"
print("export verified:", cfg["params_total"], "params")
PY

echo "[3/3] done"
