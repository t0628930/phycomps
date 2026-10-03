# PhyCoMPS: Auditable Bounded Priors for Hydroponic Time-Series Generation

This repository contains the implementation and post hoc validation materials associated with the revision of **“Auditable Bounded Priors for Hydroponic Time-Series Generation”**, submitted to the *International Journal of Electrical and Computer Engineering* (IJECE; manuscript 42922).

**Authors:** Tarik Idrissi, Abdessamad El Rharras, and Rachid Saadane.

## Release status and scope

This repository is an **unpublished release candidate**, prepared in a private repository on 3 October 2026. Authorization to prepare the code and results was obtained from the authors. A noncommercial-use policy, with commercial use subject to separate permission, is envisaged; however, the precise license has not been approved. The license, public accessibility, and publication of an immutable release therefore remain pending. This candidate must not be described as a published open-source release. No DOI, acceptance date, or historical commit identifier is claimed.

## Repository contents

The repository provides:

- Archived scripts for data cleaning and transformation, model implementation, training, and ablation analysis.
- Forty archived configurations, modified only to use relative input and output locations.
- The canonical evaluation implementation and numerical results tables.
- Additional utilities for temporal reconstruction, operational plausibility index (OPI) sensitivity analysis, and checkpoint timing.
- A separate compatibility overlay for the MPS interface, accompanied by value- and gradient-consistency tests.
- Twelve data-independent regression tests and an independent verifier for reported numerical tables.

The directory name `Frozen_Reproducibility` is retained for compatibility with existing imports. This **public-subset candidate does not constitute the complete private frozen snapshot**: source data are excluded, configuration paths have been adapted, and selected path metadata have been redacted. The relevant transformations are documented in `docs/PROVENANCE.md`. Unmodified source scripts preserve their original SHA-256 hashes.

## Computational environment and verification

The existing CPU-based validation was conducted using Python 3.12. For an isolated CPU environment, install the dependencies listed in `validation_20261001/requirements-validated-cpu.txt`. The recorded environment included NumPy 2.2.6, pandas 2.3.3, and PyTorch 2.7.1+cpu. A clean installation of these dependencies has not been independently validated; the pinned requirements cover direct dependencies only. The CPU-only requirements should not be substituted for the original Colab CUDA environment.

From the repository root, execute:

```text
python validation_20261001/test_validation.py
python tools/verify_result_tables.py
python tools/test_result_tables.py
python tools/test_release_gates.py
python tools/check_release.py
python validation_20261001/runtime_entry.py train --help
python validation_20261001/runtime_entry.py ablation --help
```

The results-table verifier uses only the Python standard library. It independently recalculates all 660 OPI configurations from the reported component summaries and checks 700 timing samples against the checkpoint-level and overall means. **These checks establish arithmetic consistency; they do not independently reproduce the results from the original sensor trajectories.**

## Data availability and limits of reproducibility

Neither raw nor processed sensor observations, real or synthetic trajectory arrays, nor trained model checkpoints are distributed in this repository. The input artifacts required for data-dependent analyses are identified in `docs/DATA_ACCESS.md`.

The provided scripts support rerunning temporal reconstruction, trajectory-based metrics, and checkpoint timing only when the corresponding external artifacts are available. Passing the code-only tests is therefore insufficient to establish full training reproducibility. No additional model was trained specifically for this revision.

## Interpretation and outstanding validation

The OPI weights and the 30 g/m³ threshold are diagnostic heuristics, rather than independently calibrated agronomic limits. The newly measured CPU execution times must not be interpreted as validation of the original 2.738 ms host measurement.

Several questions remain unresolved, including full-training computational and memory overhead, the historical selection of training-only envelopes and models, responses to active causal interventions, and validation across independent cultivation sites. Detailed evidence and limitations are reported in `docs/VALIDATION_REPORT.md`.

## Citation, permissions, and third-party rights

The author and title metadata are provided in `CITATION.cff`. A release DOI should be added only after publication and verification. No software license has yet been approved. Any applicable third-party notices and rights must remain intact and must not be superseded by a subsequent author-selected license.
