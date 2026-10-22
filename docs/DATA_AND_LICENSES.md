# Data provenance and licenses

## Training data

- **Demo corpus** (`data/demo/seeds/`): 100% written by this project (code examples with explanations in Python/Kotlin/Java/JS/TS/SQL/shell, professional emails and letters, technical knowledge paragraphs, short stories, and 104 instruction/response pairs). License: **MIT**, same as the code.
- **Synthetic expansion** (documented, labeled): `scripts/prepare_demo_data.py` creates deterministic paragraph-order shuffled variants (`origin: synthetic_expansion`), each tracing back to its seed document. No third-party data is hidden inside the expansion.
- **No web scraping, no third-party datasets, no copyrighted material** were used at any point.
- Every prepared dataset records a manifest (`data/demo/prepared/manifest.json`) with per-document source, license, origin, and the exact pipeline steps applied. Sensitive-data filtering (emails, keys, phone-like numbers, password assignments) is enforced before packing and covered by unit tests (`tests/test_data.py`).

## For the future full 500M run

Only datasets with machine-readable licenses that permit model training should be used, for example: RedPajama (Apache-2.0 / ODC-BY components), FineWeb (ODC-BY), The Stack v2 (per-repo licenses, opt-out respected), plus licensed or public-domain books. The data pipeline requires `source` and `license` per document and refuses to pack unprovenanced data; each dataset version must be recorded with the manifest hash in the model card.

## Software dependencies (training side)

| Dependency | License |
|---|---|
| Python 3.12 | PSF license |
| PyTorch 2.14.1 (CPU) | BSD-3-Clause |
| NumPy | BSD-3-Clause |
| pytest | MIT |

## Android dependencies

| Dependency | License |
|---|---|
| AndroidX (core, activity-compose, lifecycle, work-runtime) | Apache-2.0 |
| Jetpack Compose (ui, material3, icons-extended) | Apache-2.0 |
| Kotlin stdlib / coroutines | Apache-2.0 |
| org.json (JVM test artifact; on-device provided by Android) | Public domain / JSON.org license |

## Model files

`nexus-demo-v0.1.0.nexus` is a trained artifact of this repository's own pipeline on the project-authored corpus; it is distributed under the repository MIT license with no third-party weights inside.
