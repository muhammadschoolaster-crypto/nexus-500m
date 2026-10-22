# NEXUS-500M Architecture

## 1. System overview

```
┌─────────────────────────── Training side (Python) ───────────────────────────┐
│  data pipeline  →  tokenizer (byte-level BPE)  →  base training (AdamW)       │
│        │                                    │                                 │
│        │                              checkpoints                            │
│        ▼                                    ▼                                 │
│  provenance/filter/dedupe/split   instruction tuning (SFT)                   │
│        │                                    ▼                                 │
│        └──────────────────────►  evaluation harness ──► int8 export (.nexus) │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼ (small .nexus checkpoint bundled)
┌─────────────────────────── Android side (Kotlin) ───────────────────────────┐
│  ModelManager ─► BpeTokenizer ─► InferenceEngine (pure-Kotlin Transformer)    │
│        ▲                                    ▲                                 │
│  Chat UI (Compose)  │  Memory store  │  TaskEngine (tools, permissions)      │
│  ConversationStore │  Storage mgr   │  WorkManager (background/recovery)    │
└─────────────────────────────────────────────────────────────────────────────┘
```

Core AI is **self-contained**: tokenizer, weights, tokenizer merges and inference are all our own artifacts. No cloud AI service is required for any offline feature.

## 2. Model

- GPT-style decoder-only causal Transformer:
  - byte-level BPE tokenizer (GPT-2-compatible byte-to-unicode mapping), versioned vocab files;
  - token + learned positional embeddings;
  - `n_layers` blocks of: pre-LayerNorm → multi-head causal self-attention → residual, pre-LayerNorm → GELU (tanh approx) MLP with 4× expansion → residual;
  - final LayerNorm; tied embedding/unembedding matrix;
  - random weight initialization (normal 0.02, scaled residual projections) — no pretrained weights anywhere.
- Presets:
  - `nexus_500m`: d_model 1760, 12 layers, 16 heads, ctx 2048, vocab 32,768 → **exact count printed by `nexus.count_params` (~507M)**
  - `nexus_demo`: d_model 128, 3 layers, 4 heads, ctx 256 — the scale actually trained in this CPU-only environment and shipped in the app.
- Parameter counting builds the model on PyTorch's `meta` device (no RAM allocation), so even the 500M configuration can be *verified exactly* without holding 2 GB of weights in memory.

## 3. Training pipeline

`prepare data → base train → validation → SFT → evaluation → int8 export → Android inference`

- Reproducible: fixed seeds, config files recorded with every checkpoint, JSONL training logs.
- Checkpoints store model + optimizer state + step + config hash; resume is tested.
- Stages keep separate checkpoints; a working model is never overwritten until the new one passes validation (see `docs/TRAINING.md`).

## 4. Android application

- **Chat**: Compose UI, streaming token-by-token generation, copy/stop/regenerate, Markdown-lite + code-block rendering, conversation history with search/rename/archive/delete.
- **Offline model**: `ModelManager` verifies SHA-256 integrity of the bundled (or downloaded) `.nexus` file, loads it into a pure-Kotlin inference engine, and runs generation fully offline.
- **Memory**: separate stores for conversation history, user-approved memories, session summaries; keyword-ranked retrieval injected as context. This is retrieval augmentation — *not* silent retraining — and is transparent and toggleable.
- **Agent**: `TaskEngine` runs interpret → plan → step → observe → recover → verify → report, with confirmation gates for risky actions, a global stop, time limits, and persisted task history.
- **Background**: WorkManager for deferrable/resumable work; foreground service only where Android policy allows; task checkpoints recover after process death. No impossible claims: no execution while the phone is off.

## 5. Extensibility

The inference engine implements the `.nexus` format (JSON manifest + int8 weight blobs inside a zip). Larger future NEXUS models only need: a new config, training with the same pipeline, export, and a device compatibility check (RAM/headroom) — the app's `InferenceEngine` is dimension-driven, so no redesign is needed.
