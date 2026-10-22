# Android application guide

## What this app is

A native Android (Kotlin + Jetpack Compose) personal AI assistant whose core intelligence is the
project's own from-scratch NEXUS model, running **fully offline** in a pure-Kotlin inference
engine. No cloud AI API is used anywhere in the core experience.

## Honest capability note (read this first)

The bundled model `nexus-demo v0.1.0` is the **tiny demo checkpoint genuinely produced by this
project's pipeline** (886,528 parameters, trained 650 total steps on a 19K-token demo corpus).
It generates mostly word-salad with recognizable learned fragments. Its purpose is to prove the
complete chain — training → int8 export → verified on-device inference — not to be a capable
assistant. The Settings screen states this in the app itself.

## What works offline (no internet needed)

- Chat with the local model (streaming token by token, stop + regenerate + copy).
- Conversation history: create, search, rename, archive, delete.
- Personal memory: add/edit/delete/export/disable; keyword retrieval injected as context
  (retrieval augmentation, NOT model retraining — and the app says so).
- Autonomous task engine with local tools: calculator (real parser, no eval), notes
  (app workspace only, create/read/delete with confirmation), memory search,
  reminders via WorkManager, report writing.
- Storage manager: per-category usage, safe cache cleanup, backup export.
- Everything verified by 14 JVM unit tests, including a **golden parity test**: the Kotlin
  engine reproduces the Python training side's token ids, logits and greedy continuation for
  the same quantized checkpoint.

## What needs optional, explicit consent

- **Web research tool**: disabled by default; when enabled it queries only the public
  Wikipedia REST API, and treats retrieved text as data, never as instructions.
- **Notifications** (Android 13+) are requested at runtime for reminders.

## Safety design (all implemented in code)

- Risky steps (delete note, etc.) require an in-app confirmation; refusing cancels the task.
- Global stop button cancels the running task; 90-second hard timeout per task; step limits.
- Model-generated text is never executed; every tool is a fixed, audited implementation.
- Secrets are refused as memories (`Tools.looksensitive`); memory is user-controlled.
- Model files are SHA-256 verified against the build-time registry before loading; a
  corrupted model is rejected (unit-tested).
- All data lives in the app sandbox; writes are atomic (tmp + rename) so a full disk
  cannot corrupt a store.

## Background execution — honest limits

- WorkManager is used for reminders: deferrable, resumable, survives restarts; Android may
  delay background work per battery policy — the app never claims otherwise.
- No foreground service is kept alive; no execution happens while the phone is powered off.
- Task history + conversations persist across restarts (tested logic, JVM-level).

## Building

```bash
cd android
./gradlew :app:assembleDebug        # APK at app/build/outputs/apk/debug/
./gradlew :app:testDebugUnitTest    # 14 JVM tests incl. the Python-parity golden test
```
Requirements: JDK 17+, Android SDK (platform 35, build-tools 35).

## Testing a future larger NEXUS model

The `.nexus` format and `InferenceEngine` are dimension-driven: a new model (e.g. a future
500M export) needs only a config + weights in the same format, a RAM/headroom check, and a
registry entry — no app redesign. For real 500M-scale inference, a native runtime (e.g.
llama.cpp-style GGUF) is the recommended next step; `ModelManager` isolates the engine choice.

## Known limitations (untested on physical devices)

- No instrumented/UI-automation tests and no physical-device run were possible in this build
  environment: on-device tokens/sec, notification delivery on real hardware, and battery
  behavior are **UNTESTED**. All JVM-testable logic passes; the app compiles and builds an
  installable APK.
- Generation recomputes the full window each token (no KV cache) — fine at demo scale,
  listed as future work for large models.
