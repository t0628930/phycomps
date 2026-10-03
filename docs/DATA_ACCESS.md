# External data and optional artifacts

The manuscript identifies Hydroponic IoT Sensor and Actuator Logs with DOI
https://doi.org/10.34740/kaggle/ds/8223904. The DOI was not retrievable through the
preparation browser on this run, so current download access and redistribution
terms were not independently confirmed. Consult the original dataset record and
its license; this candidate does not grant rights to that dataset.

No rows of that dataset, transformed rows, trajectory arrays or model checkpoints
are included in this code/results candidate. Only aggregate tables and checksums
are provided. Their redistribution authorization is separate from source-data rights.

## Required inputs
- Temporal reconstruction: supply the original CSV through `validate_temporal.py --raw`;
  archived `hydro_dataset_pcwgan_gap_clean.csv` and `real_unit.npy` are expected
  in `Frozen_Reproducibility/data/`.
- OPI trajectory rerun: requires the archived prepared CSV and
  `Frozen_Reproducibility/data/{variant}/seed_{seed}/fake_unit.npy` as in the private kit.
- Checkpoint timing: pass `benchmark_checkpoint.py --runs` pointing to the ten
  direct/Full checkpoint directories, including their associated arrays/configs;
  the historical prepared CSV and matching direct/Full `fake_unit.npy` files
  in the frozen data subtree are also required by the supplied loader.

The private local kit records these artifacts and their hashes. It has not been
published. No public download of missing checkpoints or trajectory arrays is
claimed. A code-only publication therefore does not deliver full result reproduction.

`EXTERNAL_ARTIFACTS.csv` lists excluded artifacts and their recorded checksums.
Checksums establish file identity only, not permission to redistribute them.
