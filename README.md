# PhyCoMPS: auditable bounded priors

Code and post-hoc validation results associated with the IJECE revision of
**Auditable Bounded Priors for Hydroponic Time-Series Generation** (manuscript 42922).
Authors: Tarik Idrissi, Abdessamad El Rharras, Rachid Saadane.

## Release status
This is an unpublished release candidate prepared for the private repository
https://github.com/t0628930/phycomps-ijece-42922 (3 October 2026). Author authorization for code and
results was supplied during preparation. A noncommercial policy with separate commercial permission is intended; the exact
license remains unapproved. License choice, public accessibility and an immutable public release remain pending. Do not describe this candidate as a published open-source release.
No DOI, acceptance date or historical commit identity is asserted.

## Included
- Archived cleaning/transformation scripts, model and training/ablation modules.
- Forty archived configurations; only their input/output locations are made relative.
- The canonical evaluator and numerical result tables.
- New temporal, OPI-sensitivity and checkpoint timing utilities.
- A separate MPS interface compatibility overlay, with value/gradient tests.
- Twelve data-free regression tests and an independent result-table verifier.

`Frozen_Reproducibility` retains its original directory name for import compatibility.
This PUBLIC SUBSET is not the complete private frozen snapshot: data are excluded,
configuration locations are changed and path metadata are redacted. See provenance.
Unchanged source scripts retain their original SHA-256 values.

## Environment and tests
Python 3.12 was used for the existing CPU validation. To prepare an isolated CPU
installation, use `validation_20261001/requirements-validated-cpu.txt` with pip.
The recorded environment has NumPy 2.2.6, pandas 2.3.3 and PyTorch 2.7.1+cpu.
Fresh dependency installation has not been tested; pins cover direct dependencies.
Do not replace a Colab CUDA environment with these CPU-only requirements.

Run from this repository root:

```text
python validation_20261001/test_validation.py
python tools/verify_result_tables.py
python tools/test_result_tables.py
python tools/test_release_gates.py
python tools/check_release.py
python validation_20261001/runtime_entry.py train --help
python validation_20261001/runtime_entry.py ablation --help
```

The result-table verifier needs only Python's standard library. It recalculates
all 660 OPI settings from component summaries and checks the 700 timing samples
against checkpoint/overall means. This is arithmetic validation, not independent
reproduction from raw sensor trajectories.

## Data-dependent analyses
Raw and processed sensor rows, real/synthetic trajectory arrays and checkpoints
are NOT distributed here. `docs/DATA_ACCESS.md` lists their required locations.
The validation scripts are supplied but require those external artifacts to rerun
temporal reconstruction, trajectory metrics or checkpoint timing. Code-only tests
are not full training reproduction. No fresh model was trained for this revision.

## Interpretation
OPI weights and the 30 g/m3 cutoff are diagnostic heuristics. The new CPU timings
are not evidence of the original 2.738 ms host. Full-training/memory overhead,
train-only historical envelope/model selection, active causal response and
cross-site validation remain unproven. See `docs/VALIDATION_REPORT.md`.

## Citation and rights
Use the supplied author/title metadata in `CITATION.cff`; add the verified release
DOI only after publication. No software license has been selected yet. Third-party
notices, if applicable, must not be removed or overridden by the authors' license.
