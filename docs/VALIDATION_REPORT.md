# Supplementary Validation Report

1 October 2026

New post-hoc checks and local CPU measurements. No new model training or public release.

## S1. Raw data and temporal separation

This is a new post-hoc validation, not a new training experiment. The archived cleaner reconstructs 50,238 prepared records from 50,570 raw records. Sensor differences are below absolute tolerance 1e-7; no timestamps are unmatched. There are no missing raw sensor values, so bidirectional interpolation has no numerical effect on these observations.

The chronological window audit yields 1323 training and 234 holdout windows with 3 purged windows and 0 shared original raw IDs. The last training observation is 2023-12-26T01:17:21; the first holdout observation is 2023-12-26T01:17:31. The purge excludes windows spanning the boundary; it does not imply a multi-minute gap between the retained raw observations.

Partitioning raw observations before cleaning reproduces the same float32 training and holdout windows. Altering only future sensor values leaves training data unchanged. A synthetic missing-value boundary case makes the global cleaner leak, while the split-first procedure prevents the influence. Synthetic test data are only software tests, not added experimental evidence.

Remaining limits: the historical operating envelope and model selection are not proven train-only. Reconstructing the present files does not prove that every historical training session used byte-identical source files. Offline interpolation within a partition is not causal online imputation. No prospective-site or leakage-free retraining claim is made.

## S2. OPI sensitivity

Stored outputs from direct, residual Full and residual IID configurations, five seeds each, are decomposed into range compliance, humidity-deficit compliance and smoothness. The decomposition matches the frozen canonical OPI for all 15 runs within 1e-12. Original experimental scores are retained.

The diagnostic grid contains 66 weight triples on the simplex with increments of 0.1; cutoffs 5, 10, 15, 20 and 30 g/m3; and either the historical fixed jump thresholds or training-q95 thresholds. These choices are exploratory diagnostics, not new agronomic recommendations, parameter optimization, or independent experiments.

Across 660 settings, the direct mean never falls below Full or IID. Each pair has 654 strict advantages and six ties, where only the thermodynamic component is weighted at cutoffs 15, 20 or 30 under either jump-threshold definition. This is robustness of the tested means on a finite grid, not a significance result or universal ordering.

At 30 g/m3, all 15 core outputs have thermodynamic compliance equal to one. The threshold contributes a constant term in these comparisons. Sensitivity does not establish an agronomic justification for 30, and direct versus IID still changes the decoder as well as the prior configuration.

| Configuration | Jump thresholds | Mean OPI |

|---|---|---:|

| Direct | Fixed | 0.982199 |

| Direct | Train q95 | 0.957491 |

| Full | Fixed | 0.971250 |

| Full | Train q95 | 0.944369 |

| IID | Fixed | 0.853458 |

| IID | Train q95 | 0.847104 |

## S3. Hardware-documented inference timings

New local measurements: Intel Core i5-7200U CPU, nominal 2.50 GHz, x86-64, two cores/four logical processors; Windows; Python 3.12.3; PyTorch 2.7.1+cpu. One intra-op and one inter-op thread; CPU execution, float32, evaluation and inference mode.

The ten completed checkpoints comprise direct and residual Full, seeds 42-46. Checkpoint state dictionaries are loaded strictly using weights_only=True, associated generated arrays are compared with frozen outputs, and file hashes are retained. The workload uses the first 1 or 16 historical holdout conditions, which are passive.

Each operation has three warm-up calls and seven measured calls per checkpoint/batch. Operation order rotates between rounds. perf_counter_ns measures elapsed wall time; a CUDA synchronization path is included for future GPU runs, but no GPU measurements are reported here. Fixed, preallocated noise and resident input tensors exclude I/O, loading and transfers. Fresh prior samples are drawn only for runtime measurement.

Reported numbers are means of five checkpoint means. Batch-16 per-window values are amortized throughput, not batch-one latency. Technical repetitions are not independent training seeds. Standalone exact box auditing recomputes the gate, normalizers and allowed-mass contractions. It excludes sampling and marginal/transition audits.

Incremental forward overhead compares prior NLL alone with NLL plus the exact box barrier using shared caches. It is not training/backward/optimizer or memory overhead. Desktop scheduling and thermal conditions are not controlled as on a dedicated benchmark server; raw timing variability is retained. Model families were measured sequentially, so their timing differences are descriptive, not controlled architecture speedup estimates. These measurements do not recover the CPU or protocol of the historical 2.738 ms result.

| Model | Batch | Operation | ms/batch | ms/window* |

|---|---:|---|---:|---:|

| Full | 1 | Decoder | 12.031 | 12.031 |

| Full | 1 | Sample + decoder | 1465.006 | 1465.006 |

| Full | 1 | Exact box audit | 480.909 | - |

| Full | 16 | Decoder | 36.073 | 2.255 |

| Full | 16 | Sample + decoder | 4621.267 | 288.829 |

| Full | 16 | Exact box audit | 475.016 | - |

| Direct | 1 | Decoder | 14.198 | 14.198 |

| Direct | 1 | Sample + decoder | 1729.670 | 1729.670 |

| Direct | 1 | Exact box audit | 552.592 | - |

| Direct | 16 | Decoder | 39.590 | 2.474 |

| Direct | 16 | Sample + decoder | 5140.806 | 321.300 |

| Direct | 16 | Exact box audit | 523.568 | - |

* Batch 16 is amortized; audit timings are per complete batch.

Full, batch 1: mean paired incremental prior-objective forward cost 243.488 ms (58.3% of the NLL-only forward call). This is not total-training overhead.

Full, batch 16: mean paired incremental prior-objective forward cost 275.720 ms (65.3% of the NLL-only forward call). This is not total-training overhead.

Direct, batch 1: mean paired incremental prior-objective forward cost 310.331 ms (61.5% of the NLL-only forward call). This is not total-training overhead.

Direct, batch 16: mean paired incremental prior-objective forward cost 264.222 ms (55.1% of the NLL-only forward call). This is not total-training overhead.

## S4. Executable compatibility and reproducibility

An integration failure was reproduced: the frozen mixture calls GenerativeMPS.log_prob(..., log_norm=...), but the frozen MPS class does not accept this argument. The original archive is unchanged. A separate runtime copy adds optional log_norm arguments to log_prob and allowed_log_barrier only; the patch is supplied.

The correction reuses the same contractions and does not change architecture, weights, sampling or default uncached formulas. Twelve regression tests pass, including values and gradients against the original MPS, cached-versus-uncached mixture objectives, train/test boundary negative controls, out-of-range scaling, discrete clamping and bounded decoding.

The missing pc_mps_train.py dependency was copied from the existing project into the overlay. Training and ablation entry points import and expose --help successfully. This is an import/interface check, not a full fresh-install or end-to-end training reproduction. All 80 frozen source/evidence hashes remain unchanged.

The local kit contains scripts, source hashes, recorded dependency versions, raw timing tables and validation outputs. Imported module versions, rather than potentially stale installer metadata, define the recorded requirements; both are retained in runtime_version_details.json. A public repository/immutable release is still required. Code/data publication rights and licenses must be confirmed; no upload, public sharing or journal submission has occurred.

## Evidence

All source hashes, complete measured samples, grid results and test logs are retained in validation_20261001/results and runtime. Frozen_Reproducibility remains a distinct, unchanged historical-evidence snapshot.
