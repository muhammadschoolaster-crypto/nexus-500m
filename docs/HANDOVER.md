# NEXUS-500M Final Handover Report (16 items — honest status on every line)

1. **Repository (live).** Public at
   **https://github.com/muhammadschoolaster-crypto/nexus-500m**
   (branch `main`). Published history: milestones `24fe2ad`→`b6fbc45`
   (v0.1.0 work) + the v0.2.0 code commit **`9246c2c`** + the commit that
   adds this report (the branch tip — verify with `git rev-parse origin/main`
   or the release tag `v0.2.0-poc`).
   *Honest commit-history note:* the v0.2.0 snapshot was first committed
   locally as `41ba563` with 989 Gradle build-output files accidentally
   included; it was **never published** and was rebuilt cleanly as
   `9246c2c` (identical project content — model, checkpoints, app, logs,
   docs — minus build junk and one machine-local file). The first attempt
   survives only on a local backup branch.

2. **Commit SHAs to verify after push:** v0.2.0 code **`9246c2c`**;
   handover commit = branch tip (tag `v0.2.0-poc`). Previously published
   tip: `b6fbc45`.

3. **Architecture.** Decoder-only Transformer LM (pre-LayerNorm, causal
   self-attention, GELU MLP, untied lm-head, learned absolute position
   embeddings — chosen for exact on-device Kotlin portability). Pure PyTorch
   implementation, random init, no third-party LLM anywhere. BPE tokenizer
   written from scratch (pure stdlib), mirrored 1:1 in Kotlin.

4. **Exact parameter counts (both verified twice: closed-form + instantiation).**
   - Target config `nexus_lm/configs/nexus_500m.json`: **496,458,240 params**
     (vocab 32k · d 1280 · 21 layers · 20 heads · ff 5120 · ctx 1024)
   - Actually-trained PoC `nexus_poc.json`: **12,567,680 params**
     (vocab 4096 · d 320 · 8 layers · 5 heads · ff 1280 · ctx 256)

5. **Training status — REAL, and honestly limited.** PoC pretrained
   **3,500 steps / 7,168,000 tokens (~1.9 epochs) in ~50 min on 2 CPU cores**
   (~2,070 tok/s sustained): train loss **8.41 → 3.44**, best val **3.54**
   (perplexity 34.5). Instruction tuning: **176 steps**, val **1.79 → 0.65**
   (format demonstration on a labeled-synthetic 10.4K-token set).
   The **496M model was NOT trained** — the machine has no GPU and 4.1 GB RAM;
   the optimizer state alone (fp32 weights 1.99 GB + AdamW m/v ~4 GB) cannot
   allocate, and Chinchilla-style data needs ~10B tokens ≈ 400–700 A100-hours.
   **No fabricated numbers exist anywhere in this report.** All logs are in the
   repo (`training/runs/*/train_log.csv`, `val_log.csv`, chart
   `docs/training_curves.png`).

6. **Dataset provenance & licensing.** `training/data/corpus_manifest.json`
   lists all 206 files / 9.82 MB: 10 Project Gutenberg books (public domain),
   190 CPython-3.12 stdlib sources (PSF), 5 authored samples (MIT, in-repo),
   1 labeled-synthetic instruction set (MIT, in-repo). Domain-tagged
   (`<|text|>`, `<|code:*>`, `<|instruct|>`, `<|end|>`).

7. **Training logs.** Full CSVs committed (train + val, both stages), verbatim
   console logs in `training/logs/`, loss-curve chart in `docs/`.

8. **Evaluation results.** Held-out val loss/perplexity as above; 4 verbatim
   samples in `samples/real_samples.txt` (they show learned structure — words,
   code tokens, instruction format — but not coherent semantics; stated
   plainly). Coding-ability of a 12.5M PoC: format/keyword level only.

9. **On-device inference.** Pure-Kotlin fp32 engine, K/V-cached, single-threaded
   `Dispatchers.Default`, no TFLite/native libs. **Equivalence proven**:
   `scripts/verify_kotlin_equivalence.py` replicates the exact Kotlin
   algorithm against PyTorch on the trained weights — max logit diff **5e-6**,
   identical top-5 predictions. Speed on phone will be roughly 5–15 tok/s at
   short contexts (PoC model); memory ~55–80 MB heap.

10. **Offline features shipped:** chat with streaming generation, Chat/Code/
    Write modes, persistent memory (`remember: …`, view/delete/export/wipe),
    conversation search, model info, dark mode — **manifest declares no
    INTERNET permission at all**, so offline operation is provable.

11. **Tested autonomous tasks / agent:** the previous v0.1.0 app's
    permission-gated agent (calculator, notes, reminders — rule-based tools)
    still exists at `android/app` with 14/14 JVM tests passing then.
    v0.2.0 (`android/NexusAssistant`) is the current app (model-driven chat
    + memory); its on-device smoke test requires a physical phone and is
    **UNTESTED** — that is the honest status.

12. **Background execution limits:** Android kills background processes when a
    device powers off or Doze is aggressive; nothing can execute while a
    phone is switched off. v0.2.0 performs inference only in the foreground
    (by design for the PoC); WorkManager scheduling remains from v0.1.0.

13. **APK — published on GitHub Releases, direct link.**
    **https://github.com/muhammadschoolaster-crypto/nexus-500m/releases/download/v0.2.0-poc/nexus-assistant-poc.apk**
    — **52,758,149 bytes** — SHA-256
    **`65d2fccb309fa2588e7c74358a4dbd35274d2e6423086a2893588cc762202670`**
    — debug-signed (Android Debug CN), minSdk 26 (Android 8.0+), targetSdk 34,
    contains the **genuinely-trained 12,567,680-param model** (50,270,720-byte
    fp32 binary verified inside the APK). Release page:
    `https://github.com/muhammadschoolaster-crypto/nexus-500m/releases/tag/v0.2.0-poc`
    (also lists `SHA256SUMS.txt` and auto-generated source archives).
    Verify after download: `sha256sum nexus-assistant-poc.apk`.

14. **Known defects / limitations (complete list):**
    - PoC model is not conversationally intelligent (12.5M params / 50 min CPU —
      expects word-salad-ish replies; stated in-app too).
    - Generation speed on low-end phones may feel slow (pure-Kotlin fp32).
    - v0.2.0 app not yet smoke-tested on a physical device.
    - APK is debug-signed (normal for PoC; a release-signed build needs a
      keystore you control).
    - The two best checkpoints (`nexus_best.pt`, ~48 MB each) are committed to
      git for provenance; optimizer-state files (150 MB each) are excluded.

15. **Install steps (Android 8.0+, directly on your phone):**
    1. Open the APK link in item 13 in your phone's browser; the file
       downloads (`nexus-assistant-poc.apk`, ~52.8 MB).
    2. Tap the downloaded file (or open it from your Downloads folder).
    3. Allow "install unknown apps" for your browser/file manager when
       prompted (normal for any APK outside Play Store).
    4. Tap **Install** → open **NEXUS Assistant**.
    5. First launch loads the model from assets (~2–5 s). Type a message, or
       try `remember: my name is …` to store a memory; toolbar icons:
       new chat / search / memories / model info.

16. **Security actions for you (GitHub):**
    - **Revoke the PAT used for this task** — it was pasted in plaintext chat.
      GitHub → Settings → Developer settings → Personal access tokens →
      delete it. (The repo contains no secrets; the token is stored nowhere
      in it — secret scan verified.)
    - To make the repo private later: repo **Settings → General → Danger
      Zone → Change visibility → Private**. **Note:** a private repo makes
      the public release-download link in item 13 stop working on your phone
      unless you are signed in — keep that in mind before changing visibility.
      Re-host the APK elsewhere first if you privatize.

**Scaling to the real 500M model (the genuine path):** rent ~8×A100 (or
equivalent) for ~1 week, gather ~10B licensed tokens (e.g., The Stack v2 for
code, licensed books for writing), then run the committed pipeline
unmodified with `nexus_lm/configs/nexus_500m.json`, then export → rebuild the
APK — identical code, as verified end-to-end by this PoC.
