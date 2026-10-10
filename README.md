# PhyCoMPS: Auditable Bounded Priors for Hydroponic Time-Series Generation

Research code and post-hoc validation results for **Auditable Bounded Priors for Hydroponic Time-Series Generation**.

**Authors:** Tarik Idrissi, Abdessamad El Rharras, Rachid Saadane.

**Repository:** https://github.com/t0628930/phycomps

## Research overview
PhyCoMPS combines context-conditioned matrix-product-state (MPS) Born priors, exact auditing of discrete latent admissibility, recurrent decoding and smooth bounded outputs for hydroponic multivariate time series. Its experiments distinguish latent prior probabilities from output containment, and compare structured and IID configurations as well as direct and residual decoders.

## Repository status
This public research subset provides archived code, aggregate results and supplementary validation utilities. It does not contain all private training artifacts. Author-owned software is licensed under GNU GPL version 3 only (GPL-3.0-only), which permits commercial use subject to its terms. No DOI or journal acceptance is asserted.

## Included resources
- Archived cleaning/transformation scripts, model and training/ablation modules.
- Forty archived configurations; only their input/output locations are made relative.
- The canonical evaluator and numerical result tables.
- Temporal, OPI-sensitivity and checkpoint timing utilities.
- An MPS interface compatibility overlay, with value/gradient tests.
- Twelve data-free regression tests and an independent result-table verifier.

`Frozen_Reproducibility` retains its directory name for import compatibility. This PUBLIC SUBSET is not the complete private frozen snapshot: data are excluded, configuration locations are changed and path metadata are redacted. See provenance. Unchanged source scripts retain their original SHA-256 values.

## Environment and tests
Python 3.12 was used for the recorded CPU validation. To prepare an isolated CPU installation, use `validation_20261001/requirements-validated-cpu.txt` with pip. The recorded environment has NumPy 2.2.6, pandas 2.3.3 and PyTorch 2.7.1+cpu. Fresh dependency installation has not been tested; pins cover direct dependencies. Do not replace a Colab CUDA environment with these CPU-only requirements.

Run from the repository root:

```bash
python validation_20261001/test_validation.py
python tools/verify_result_tables.py
python tools/test_result_tables.py
python tools/test_release_gates.py
python tools/check_release.py
python validation_20261001/runtime_entry.py train --help
python validation_20261001/runtime_entry.py ablation --help
```

The result-table verifier uses Python's standard library. It recalculates all 660 OPI settings from component summaries and checks 700 timing samples against checkpoint/overall means. This is arithmetic validation, not independent reproduction from raw sensor trajectories.

## Data-dependent analyses
Raw and processed sensor rows, real/synthetic trajectory arrays and checkpoints are NOT distributed here. `docs/DATA_ACCESS.md` lists their required locations. Supplied validation scripts require those external artifacts to rerun temporal reconstruction, trajectory metrics and checkpoint timing. Code-only tests do not reproduce full training. No fresh model was trained for the supplementary post-hoc analyses.

## Interpretation and limitations
OPI weights and the 30 g/m³ cutoff are project-specific diagnostic heuristics. New CPU timings do not identify the host of the historical 2.738 ms measurement. Full-training and memory overhead, train-only historical envelope/model selection, active causal response, long-horizon stability and cross-site validation remain unproven. See `docs/VALIDATION_REPORT.md`.

## Citation and rights
Please cite PhyCoMPS and the associated unpublished research manuscript when using this software or its results. `CITATION.cff` and `CITATION.bib` provide software and manuscript citation metadata. No DOI or journal acceptance is asserted.

This scientific citation request is not an additional condition of the GPL and does not restrict private use. Preserve copyright and license notices as required by the GPL. See `LICENSE` and `COPYRIGHT.md`. Third-party notices and licenses remain applicable. The software license does not automatically license external datasets, the manuscript or result tables.

`docs/VALIDATION_REPORT.md` preserves historical local-validation notes, which may reflect an earlier licensing status. Refer to `LICENSE` and current repository metadata for present software terms.

## Timestamp characteristics
The prepared observations have irregular sampling: 5 s and 10 s increments are frequent, with a 10 s median within segments. The 62 segment spans total 110.10 h over 30.44 calendar days; causes of interruptions are not inferred. Thresholds remain per observation, not per second. See `docs/TIMESTAMP_PROFILE.md`.
