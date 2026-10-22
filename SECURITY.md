# Security Policy

## Secrets
- No secrets, API keys, tokens or keystores are committed to this repository.
- The GitHub PAT mentioned in the originating chat session was **never written
  to any file in this repo or on this machine**, and the chat session it was
  pasted into should be considered public. **Revoke it** and issue a fresh one
  through a private channel before doing any repository work.

## Mobile app
- The Android APK is signed with a **debug key** (its fingerprint is listed in
  `docs/HANDOVER.md`). For a real distribution you must generate a private
  keystore (never commit it) and sign a release build.
- The app declares **no INTERNET permission** and contains no analytics, ads
  or remote-logging code. Model + memory stay on-device.
- Memories are stored in a local SQLite DB (`nexus_memory.db`) excluded from
  Android cloud backup and device-transfer by `data_extraction_rules.xml` /
  `backup_rules.xml`, and can be wiped in-app.
- App runs in a normal sandbox; model inference is pure Kotlin inside the app
  process (no native code execution surface added beyond the default app).

## Supply chain
- Training data provenance: `training/data/corpus_manifest.json` (public-domain
  books, PSF-licensed CPython sources, MIT-labeled authored files).
- Dependencies pinned: PyTorch 2.14.1+cpu, NumPy; Android: AGP 8.7.3,
  Kotlin 2.0.21, AndroidX/Material via pinned versions in Gradle files.
- No model weights are downloaded from third parties anywhere in the pipeline —
  every weight in every artifact shipped here was produced by `nexus_lm/train.py`
  on this machine (or, for the untrained 496M target, does not exist).
