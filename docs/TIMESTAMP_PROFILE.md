# Timestamp cadence and OPI rounding recheck

Rechecked on 4 October 2026 without changing source data, training or metric policy.
The existing prepared CSV contains 50,238 rows, 62 segments and 61 gaps longer
than 60 seconds. Within segments the median increment is 10 s; 19,939 increments
are 5 s and 22,993 are 10 s. This is irregular sampling, not a uniform 10 s grid.
No resampling is used by the historical pipeline.

The calendar span is 30.443611 days. Summing last-minus-first timestamps within
segments gives 110.101111 hours, or 1.775824 hours per segment on average. These
are recorded segment spans, not independently measured equipment uptime. There
are 13,118 observations at recorded hours 00:00-05:59; no timezone or daytime-only
operating interpretation is inferred. Interruption causes remain unknown.
Training dynamic thresholds are per-observation differences, not per-second
rates; their interpretation is specific to this cadence mixture.

The paired mean OPI difference from unrounded seed values is approximately
0.0109492805669276, which correctly rounds to +0.0109. Subtracting displayed,
already-rounded means is not the calculation used for the paired difference.

## Recheck with the external prepared CSV

```text
python tools/verify_temporal_profile.py --prepared PATH_TO_PREPARED_CSV --metrics Frozen_Reproducibility/results/canonical_metrics_by_seed.csv --output temporal_profile.json
```

The prepared data are excluded from this public repository; see DATA_ACCESS.md.
The recorded output is `validation_20261001/results/temporal/temporal_profile_20261004.json`.
File hashes and complete interval counts are included. This does not constitute
fresh training or independent validation on another installation.
