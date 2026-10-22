# NEXUS Training Report (real numbers only)

## Machine

| Item | Value |
|---|---|
| CPU | 2 cores |
| RAM | 4.1 GB |
| GPU | **none** |
| Disk | 9.9 GB total |
| Software | Python 3.12.15, PyTorch 2.14.1+cpu, NumPy |

Every number below comes from runs executed on this machine. Logs:
`training/runs/poc/train_log.csv`, `training/runs/poc/val_log.csv`,
`training/runs/instruct/*.csv`, chart: `docs/training_curves.png`.

## Tokenizer (from-scratch BPE)

- Type: character-level BPE, pure Python stdlib, designed to mirror 1:1 in Kotlin
- Vocab: **4,096** entries (1 `<unk>` + 187 base chars + 3,908 learned merges)
- Trained on 9.72M chars in 60.6 s; compression **2.54 chars/token**
- Lossless round-trip verified programmatically

## Corpus (licensing-clean, manifest included)

| Source | Files | License |
|---|---|---|
| Project Gutenberg books | 10 (Alice, Sherlock, Frankenstein, Moby Dick, Dorian Gray, Jane Eyre, Treasure Island, War of the Worlds, Time Machine, Art of War) | public domain |
| CPython 3.12 stdlib sources | 190 real `.py` files | PSF |
| Authored domain samples | 5 (HTML/CSS, SQL, shell, Kotlin, JS) | MIT (this repo) |
| Synthetic instruction examples | 1 set (~10.4K tokens, 45 examples) | MIT (this repo), labeled synthetic |
| **Total** | 206 files, 9.82 MB | |

Tokenized totals: **3,752,313 train / 76,477 val** tokens (uint16 bins).
Every document wrapped in domain tags: `<|text|>`, `<|code:python|>`, …,
`<|instruct|>`, terminated by `<|end|>`.

## Stage 1 — pretraining (genuinely executed)

- Config: **NEXUS-PoC** = 12,567,680 params (d_model 320, 8 layers, 5 heads,
  d_ff 1280, ctx 256, vocab 4096) — exact count matches the closed-form formula
- AdamW (β 0.9/0.95, wd 0.1), peak LR 1e-3, warmup 100, cosine to 3e-5,
  batch 8 × 256 ctx = 2,048 tokens/step, grad-clip 1.0
- **3,500 steps = 7,168,000 tokens (~1.9 epochs)**, ~50 min wall clock,
  ~2,070 tok/s sustained on 2 cores

| Metric | Value |
|---|---|
| Train loss | **8.41 → 3.44** (best running 3.29) |
| Val loss | **4.58 (step 250) → 3.54 (best, step 3250)** |
| Val perplexity | 97.1 → **34.5** |

## Stage 2 — instruction tuning (genuinely executed)

- Resumed from the pretrained checkpoint; 176 steps on the labeled-synthetic
  instruction set (10,388 train tokens) at peak LR 1.5e-4
- Instruct val loss: **1.79 → 0.65** (the model memorizes the small format set
  — honest interpretation: **format demonstration**, not deep alignment)

## Samples (verbatim, temperature 0.7 / 0.6 — see `samples/`)

Base model, story continuation:
```
The little robot booted its offline language model and the second string and the Python
# for the file object of a file class the string.
```
Instruct model, instruction format:
```
### Instruction:
### Input:
Generate a unit for a Python adder.
```
**Honest quality statement:** at 12.5M params / 50 CPU-minutes the model
learns vocabulary, syntax tokens, domain tags and response format skeletons,
but NOT coherent semantics. No claim of conversational competence is made.

## The ~500M target — exact config, verified, not trained

`nexus_lm/configs/nexus_500m.json`: vocab 32,000 · d_model 1,280 · **21 layers** ·
20 heads (d_head 64) · d_ff 5,120 · ctx 1,024 → **496,458,240 params**
(verified twice: closed-form formula and actual module instantiation).

**Why not trained here (the exact blocker):** fp32 weights alone are 1.99 GB;
with AdamW m/v states the optimizer footprint is ~6 GB, exceeding the machine's
4.1 GB RAM — training would not even allocate, let alone finish. Compute-wise,
Chinchilla-style training of a ~500M model needs ~10B tokens ≈ 400–700
A100-hours, versus ~5 GFLOPs effective throughput on this CPU box (weeks of
wall-clock, power-throttled, still RAM-bound). **Fabricating a "500M model"
would violate the project's anti-fabrication rules.** The complete runnable
pipeline ships in this repo: swapping the config file onto a GPU rig is the
only step required.

## Export & on-device verification (genuinely executed)

- `export_android.py`: 101 tensors, 12,567,680 fp32 values = **50,270,720 bytes**,
  written in a documented tensor plan + `export_manifest.json`
- `scripts/verify_kotlin_equivalence.py` replicates the **exact Kotlin engine
  algorithm** (same tensor plan, same A&S-erf approximation, same K/V-cache
  flow) in NumPy and compares against PyTorch on the trained weights:
  **max |logit diff| = 5e-6, top-5 next-token sets identical** → the on-device
  engine math is verified equivalent (pending on-phone smoke test, which
  requires a physical device).

## Reproducibility

Seeds: torch default init order only; sampling uses runtime RNG. Checkpoints
(`training/runs/poc/nexus_best.pt`, `nexus_ckpt.pt` + instruct equivalents) and
CSV logs are included. Re-running `train.py` reproduces the loss trajectory
within CPU-threading nondeterminism.
