# PhyCoMPS: auditable bounded priors

Code and post-hoc validation results associated with the IJECE revision of
**Auditable Bounded Priors for Hydroponic Time-Series Generation** (manuscript 42922).
Authors: Tarik Idrissi, Abdessamad El Rharras, Rachid Saadane.

## Release status
The code and aggregate results are publicly accessible at
https://github.com/t0628930/phycomps-ijece-42922, verified without authentication
on 4 October 2026. The authors authorized publication and confirmed the
administrative declarations. Author-owned software is licensed under GNU GPL
version 3 only (GPL-3.0-only); commercial use is permitted under that license.
A fixed archival release is being prepared. No DOI or journal acceptance is asserted.

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
Please cite PhyCoMPS and its associated manuscript in publications, theses and
reports that use this software or its results. `CITATION.cff` supplies the verified
software author/title metadata; `CITATION.bib` provides software and unpublished
manuscript references. No DOI or journal acceptance is asserted.

This scientific citation request is not an additional condition of the GPL and
does not restrict private use. Preserve copyright and license notices when
redistributing code as required by the GPL. See `LICENSE` and `COPYRIGHT.md`.
Third-party notices and licenses remain applicable. The software license does not
automatically license external datasets, the manuscript or result tables.

`docs/VALIDATION_REPORT.md` preserves the earlier local-validation report, including
its then-pending licensing status. Current licensing and publication status are
recorded here and in `RELEASE_STATUS.json`; scientific results are unchanged.

## Timestamp clarification (4 October 2026)
The prepared observations have irregular sampling: 5 s and 10 s increments are
frequent, with a 10 s median within segments. The 62 segment spans total 110.10 h
over 30.44 calendar days; causes of interruptions are not inferred. Thresholds
remain per observation, not per second. See `docs/TIMESTAMP_PROFILE.md`.
