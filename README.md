# NEXUS-500M

**An independent, offline-first personal AI assistant with its own language model, trained from randomly initialized weights, running fully on-device on Android.**

- **Core AI**: a from-scratch GPT-style causal Transformer language model (no third-party chatbot APIs, no pretrained LLM wrappers).
- **Platform**: native Android application written in Kotlin + Jetpack Compose, with a pure-Kotlin on-device inference engine.
- **Primary skills**: coding intelligence and professional/creative writing, instruction following, persistent personal memory, and a permission-controlled autonomous task engine.
- **Offline first**: core chat, memory, and local tools work with no internet connection. Web research is optional and consent-gated.

## Honest status overview

| Area | Status (evidence-based, see `docs/MILESTONES.md`) |
|---|---|
| 500M-parameter architecture | Implemented and **exactly counted** (~507M parameters for the `nexus_500m` config) |
| Training pipeline (base + SFT + eval + export) | Implemented, unit-tested, runs end to end |
| Real training executed | **Small-scale demonstration run** (CPU-only environment; full 500M run requires GPU compute — see `docs/TRAINING.md`) |
| Android app (chat, memory, agent, storage) | Implemented in Kotlin/Compose with bundled NEXUS demo checkpoint for fully offline inference |
| CI | GitHub Actions runs the Python test suite on every push |

> **Integrity policy**: nothing in this repository pretends a model was trained when it was not. The 500M configuration is fully specified and verified by parameter counting; the *demonstration* checkpoint shipped with the app is a genuinely trained tiny-scale NEXUS model, clearly labeled as such (`nexus-demo`), produced by the exact same pipeline that would train the 500M configuration given adequate compute.

## Repository layout

```
configs/          Model configuration presets (nexus_500m, nexus_demo, nexus_smoke)
nexus/            Python package: tokenizer, Transformer model, data, training, SFT, evaluation, export
scripts/          Demo data preparation, smoke training, secret scanning
tests/            pytest unit + integration tests (real runs recorded in CI)
artifacts/        Small demo training artifacts: logs, eval reports, mobile model
data/             Data pipeline rules + small provenance-clean demo corpus
docs/             Architecture, training/compute plan, data licensing, app guide, milestone status
android/          Native Android application (Kotlin, Jetpack Compose, offline inference engine)
.github/          CI workflows
```

## Quick start (Python side)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q                     # run the full test suite
python -m nexus.count_params         # verify the exact 500M parameter count
scripts/run_smoke_training.sh        # reproduce the real tiny-scale training run
```

## Quick start (Android side)

See `docs/ANDROID_APP.md`. The app bundles a real trained demo model (`nexus-demo-v0.1`) so offline chat works on first launch with zero downloads and zero internet.

## License

- This project's code and documentation: **MIT** (see `LICENSE`).
- Third-party dependencies and their licenses are disclosed in `docs/DATA_AND_LICENSES.md`.
- Demo training data provenance and licenses: `docs/DATA_AND_LICENSES.md` and `data/demo/manifest.json`.

## Security

See `SECURITY.md`. Summary of hard rules: no API keys, tokens, keystores, passwords, private chats or confidential data are ever committed; secret scanning runs before every push; the personal APK and signing keys are kept out of public releases.
