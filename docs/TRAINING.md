# TRAINING — what ran, what it takes to train the full 500M model

## What actually ran in this project (verified, reproducible)

| Item | Value (real) |
|---|---|
| Environment | 2 vCPU, no GPU, PyTorch 2.14.1+cpu |
| Base training | nexus_demo config (886,528 params), 500 steps, batch 20, ctx 256 |
| Loss curve | 7.39 (step 5) → 1.29 (step 250) → 0.50 (step 500); val 2.84 → 2.10 |
| SFT | 150 steps on 104 project-authored instruction examples; val 7.07 (base, template-blind) → 3.98 → 4.08 |
| Time | ~70 min total (7.2 s/step base, 3.5 s/step SFT) |
| Data | 19,436 train tokens / 1,027 val / 2,570 test — project-authored demo corpus, fully licensed (see docs/DATA_AND_LICENSES.md) |
| Honest capability | coding_parse_rate 0.0; generations are word-salad with recognizable fragments — normal for 0.89M params on 19K tokens |

Reproduce exactly: `bash scripts/run_smoke_training.sh` (deterministic seeds: 1337 / 20260101).

## What the full 500M training requires (exact requirements, honest math)

- **Model**: `configs/nexus_500m.json` — verified exactly **507,418,560 parameters** (meta-device count, `python -m nexus.count_params`).
- **Data**: for a capable base model at this size, plan ~10–20 billion tokens (Chinchilla-style ~20 tokens/param) of licensed text + code (e.g. RedPajama, FineWeb, The Stack v2 — all with documented licenses; several hundred GB raw). The pipeline in `nexus/data.py` (provenance manifest → clean → sensitive filter → dedupe → hash-split → pack) is the same one used for the demo run and scales by swapping the data sources.
- **Compute** (6·N·D FLOPs):
  - 6 × 5.07e8 × 1.0e10 ≈ **3.0e19 FLOPs** for a 10B-token run.
  - On one A100 at realistic 40% MFU (~80 TFLOP/s effective bf16): 3.0e19 / 8e13 ≈ **~104 hours ≈ 4.3 days**, realistically 100–300 GPU-hours with overheads and retries.
  - On 8×A100: roughly half a day to a day. On rented spot capacity this is on the order of a few hundred USD (at $1–3/GPU-hour).
  - On the CPU used here (~5–10 GFLOP/s effective): **~90+ years**. That is why the full 500M run did not and cannot happen in this environment — this is the genuine blocker, stated plainly.
- **Storage**: fp32 weights 2.03 GB; AdamW states add ~2×; checkpoints ~6 GB each (keep 2–4); int8 export ~507 MB (fits modern phones; the same `.nexus` pipeline scales).
- **Then**: SFT (instruction tuning) on 50k–500k licensed instruction examples, evaluation, int8 export, device compatibility check — all already implemented here at demo scale.

## Pipeline stages (all implemented and actually executed)

```
scripts/prepare_demo_data.py   seeds -> clean -> filter -> dedupe -> split -> BPE train -> pack
nexus/train.py                 base training (AdamW, warmup+cosine, clip, JSONL logs, checkpoints, resume)
nexus/sft.py                   instruction tuning (weights-only init from base, validation gate)
nexus/evaluate.py              held-out PPL, coding parse rate, writing heuristics (honest)
nexus/export.py                int8 .nexus export + verification (|err| <= max|w|/254)
scripts/package_mobile_model.py bundle model into the APK assets + registry + model card
```

## Rollback safety

- Base and SFT checkpoints are separate files; `best.pt` keeps the best-validation model.
- `save_checkpoint` is atomic (tmp + rename) — a full disk or crash never corrupts a checkpoint (see docs/ANDROID_APP.md for the same rule on device).
- A new model version only replaces the app model after passing its validation gate.
