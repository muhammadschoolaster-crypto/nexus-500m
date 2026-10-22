# NEXUS-500M

An **offline-first Android AI assistant** powered by an independently-trained
Transformer language model (the "NEXUS" model family), with:

- **own model** — trained from random initialization; no third-party LLM, no API wrapper, no cloud
- **coding specialization** — Python, Kotlin, JS, HTML/CSS, SQL, shell (domain-tagged corpus)
- **writing specialization** — books + instruction-tuned response formats
- **persistent memory** — on-device SQLite, user-approved, exportable, deletable
- **provably offline** — the Android manifest declares **zero network permissions**

> **STATUS — read this first (honest engineering statement):**
>
> - The full training pipeline, tokenizer, model, export and on-device inference
>   are **real, implemented, and verified end-to-end** (engine equivalence
>   proven to 5e-6 max logit difference vs PyTorch).
> - The **~500M target config** (`nexus_lm/configs/nexus_500m.json` =
>   **496,458,240 params**, exact count verified by formula and instantiation)
>   is **defined but NOT trained**: the build machine has **2 CPU cores, no
>   GPU**, and 500M-param training requires ~10B tokens and hundreds of
>   A100-GPU-hours. Fabricating results is prohibited by this project's rules.
> - What WAS genuinely trained: **NEXUS-PoC (12,567,680 params)** — 3,500
>   pretraining steps (7.2M tokens, real loss 8.41 → 3.44, best val loss 3.54)
>   + 176 instruction-tuning steps (val 1.79 → 0.65). That model is what
>   ships inside the APK and runs on your phone.

## Repository layout

```
nexus-500m/
├── nexus_lm/                 # ML pipeline (pure PyTorch + stdlib)
│   ├── model.py              # GPT-style decoder-only transformer (exact param count)
│   ├── tokenizer.py          # from-scratch BPE (pure stdlib, Kotlin-portable)
│   ├── data.py               # corpus → uint16 token bins (domain-tagged)
│   ├── train.py              # AdamW + cosine LR + warmup, ckpt/resume, CSV logs
│   ├── sample.py             # generation from checkpoints
│   ├── export_android.py    # fp32 export per documented tensor plan
│   └── configs/
│       ├── nexus_poc.json    # 12.57M params — the ACTUALLY-TRAINED PoC
│       └── nexus_500m.json   # 496.5M params — the verified TARGET config
├── training/
│   ├── data/corpus/          # licensing-clean corpus (+ corpus_manifest.json)
│   ├── tokenizer.json
│   ├── runs/poc/             # real checkpoints + train_log.csv + val_log.csv
│   └── runs/instruct/        # instruction-tuned checkpoints + logs
├── scripts/                  # corpus build, tokenizer training, verification, plots
├── android/NexusAssistant/   # Android app (Kotlin, pure-on-device inference)
├── docs/                     # TRAINING_REPORT.md, HANDOVER.md, curves PNG
└── samples/                  # real model samples (verbatim)
```

## Quick start (reproduce the training)

```bash
pip install torch numpy
python scripts/build_corpus.py                     # or use training/data as-is
python scripts/train_tokenizer_build_bins.py       # BPE 4096 → bins
python -m nexus_lm.train --config nexus_lm/configs/nexus_poc.json \
    --prefix training/data/nexus_poc --out training/runs/poc \
    --steps 3500 --batch 8 --lr 1e-3 --warmup 100   # ~50 min on 2 CPU cores
python scripts/build_instruct_bins.py              # instruction bins
# (copy nexus_ckpt.pt into training/runs/instruct/, then resume with --lr 1.5e-4)
python -m nexus_lm.sample --ckpt training/runs/poc/nexus_best.pt \
    --tokenizer training/tokenizer.json --prompt "..." 
python -m nexus_lm.export_android --ckpt training/runs/instruct/nexus_best.pt \
    --tokenizer training/tokenizer.json --outdir <assets-dir>
python scripts/verify_kotlin_equivalence.py        # on-device math proof
```

To train the **496M target**: same pipeline, swap in
`nexus_lm/configs/nexus_500m.json`, scale data to ~10B licensed tokens and run
on a GPU cluster (est. 8×A100 × ~1 week at 4096-token batches; budget
roughly 400–700 A100-hours). The pipeline requires no code changes.

## Android app

- Kotlin + Material 3 (classic Views for maximum build reliability), minSdk 26
- **Chat** / **Code** / **Write** modes (domain-tagged prompting)
- **Pure-Kotlin transformer inference** — fp32, K/V-cached, no TFLite, no native libs
- Persistent memory: type `remember: …` in chat; view/delete/export via toolbar
- Conversation search, new chat, model info, dark mode
- **No INTERNET permission** in the manifest — offline is provable

Build: open `android/NexusAssistant` in Android Studio, or
`gradle :app:assembleDebug` with `local.properties` pointing at your SDK.

## License

MIT for this repository's code. Corpus keeps its own provenance
(`training/data/corpus_manifest.json`): Project Gutenberg (public domain),
CPython stdlib (PSF), authored samples (MIT).

## Security

See `SECURITY.md`. No secrets are committed; the build machine never stored the
GitHub PAT used in the originating chat session (it was never re-entered here,
and it should be revoked regardless).
