"""Audit the historical inputs and test a split-first preprocessing alternative."""
import argparse
import contextlib
import io

import numpy as np
import pandas as pd

from validation_common import (
    FEATURES, SENSORS, FROZEN, historical_split, load_cleaner, sha256,
    strict_prepare, window_indices, write_json,
)


def compare(left, right):
    left, right = left.copy(), right.copy()
    for frame in (left, right):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        if frame.timestamp.duplicated().any():
            raise ValueError("Duplicate timestamps make comparison ambiguous")
    joined = left.merge(right, on="timestamp", how="outer", suffixes=("_a", "_b"), indicator=True)
    paired = joined[joined["_merge"] == "both"]
    delta = np.abs(paired[[f + "_a" for f in FEATURES]].to_numpy() -
                   paired[[f + "_b" for f in FEATURES]].to_numpy())
    return {
        "left_rows": len(left), "right_rows": len(right),
        "unmatched_timestamps": int((joined["_merge"] != "both").sum()),
        "changed_feature_values_atol_1e_7": int((delta > 1e-7).sum()),
        "max_absolute_feature_change": float(delta.max()) if delta.size else None,
    }


def run(raw_path, out):
    cleaner = load_cleaner()
    raw = pd.read_csv(raw_path)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    frozen = pd.read_csv(FROZEN / "data/hydro_dataset_pcwgan_gap_clean.csv")
    frozen["timestamp"] = pd.to_datetime(frozen.timestamp)
    train_idx, test_idx, all_idx = historical_split(frozen)
    train_rows, test_rows = np.unique(train_idx), np.unique(test_idx)
    train_end = frozen.loc[train_rows, "timestamp"].max()
    holdout_start = frozen.loc[test_rows, "timestamp"].min()
    if train_end >= holdout_start:
        raise AssertionError("Historical split is not chronologically disjoint")
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        rebuilt = cleaner.to_pipeline_format(cleaner.clean(raw))
        strict = strict_prepare(raw, train_end, holdout_start, cleaner)
        modified = raw.copy()
        future = modified.timestamp > train_end
        for column in SENSORS:
            lo, hi = cleaner.PLAUSIBLE[column]
            modified.loc[future, column] = lo + .75 * (hi - lo)
        changed_global = cleaner.to_pipeline_format(cleaner.clean(modified))
        strict_changed = strict_prepare(modified, train_end, holdout_start, cleaner)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cleaning_log.txt").write_text(log.getvalue(), encoding="utf-8")
    rebuilt["timestamp"] = pd.to_datetime(rebuilt.timestamp)
    changed_global["timestamp"] = pd.to_datetime(changed_global.timestamp)
    before = rebuilt[rebuilt.timestamp <= train_end]
    after = changed_global[changed_global.timestamp <= train_end]
    raw_reconstruction = compare(frozen, rebuilt)
    observed = compare(before, after)
    strict_invariance = compare(strict["train"], strict_changed["train"])
    if strict_invariance["unmatched_timestamps"] or strict_invariance["changed_feature_values_atol_1e_7"]:
        raise AssertionError("Split-first pipeline changed after future-only perturbation")
    reconstructed_train = frozen.loc[train_rows].copy()
    for name, frame in strict.items():
        frame.to_csv(out / f"strict_{name}.csv", index=False)
    used_strict = strict["train"].copy()
    used_strict["timestamp"] = pd.to_datetime(used_strict.timestamp)
    used_strict = used_strict[used_strict.timestamp.isin(reconstructed_train.timestamp)]
    strict_train_idx = window_indices(strict["train"])
    strict_holdout_idx = window_indices(strict["holdout"])
    strict_train_values = strict["train"][FEATURES].to_numpy(np.float32)[strict_train_idx]
    strict_holdout_values = strict["holdout"][FEATURES].to_numpy(np.float32)[strict_holdout_idx]
    raw_lookup = raw.set_index("timestamp")
    train_raw_ids = raw_lookup.loc[frozen.loc[train_rows, "timestamp"], "id"].to_numpy()
    test_raw_ids = raw_lookup.loc[frozen.loc[test_rows, "timestamp"], "id"].to_numpy()
    bounds_rows = []
    for name in FEATURES:
        for label, frame in (("all_prepared_rows", frozen), ("all_prepared_training_rows", strict["train"]), ("unique_training_rows", reconstructed_train),
                             ("unique_holdout_rows", frozen.loc[test_rows])):
            low, high = frame[name].quantile([.01, .99])
            bounds_rows.append(dict(feature=name, population=label, q01=low, q99=high,
                                   min=float(frame[name].min()), max=float(frame[name].max())))
    pd.DataFrame(bounds_rows).to_csv(out / "envelope_quantiles.csv", index=False)
    train_values = frozen.loc[:, FEATURES].to_numpy(np.float32)[train_idx]
    holdout_values = frozen.loc[:, FEATURES].to_numpy(np.float32)[test_idx]
    stored_holdout = np.load(FROZEN / "data/real_unit.npy", allow_pickle=False)
    extrema = []
    for i, feature in enumerate(FEATURES):
        lo, hi = train_values[..., i].min(), train_values[..., i].max()
        values = holdout_values[..., i]
        extrema.append(dict(feature=feature, training_min=float(lo), training_max=float(hi),
                            holdout_min=float(values.min()), holdout_max=float(values.max()),
                            fraction_outside_training_extrema=float(((values < lo) | (values > hi)).mean())))
    pd.DataFrame(extrema).to_csv(out / "holdout_extrema.csv", index=False)
    report = {
        "kind": "new_posthoc_validation_no_retraining",
        "raw_sha256": sha256(raw_path),
        "prepared_sha256": sha256(FROZEN / "data/hydro_dataset_pcwgan_gap_clean.csv"),
        "cleaner_sha256": sha256(FROZEN / "scripts/prepare_hydroponic_dataset_v4_gap_clean.py"),
        "raw_rows": len(raw), "raw_duplicate_timestamps": int(raw.timestamp.duplicated().sum()),
        "raw_missing_sensor_values": {c: int(pd.to_numeric(raw[c], errors="coerce").isna().sum()) for c in SENSORS},
        "training_windows": len(train_idx), "holdout_windows": len(test_idx),
        "purged_windows": len(all_idx) - len(train_idx) - len(test_idx),
        "shared_prepared_rows": int(len(np.intersect1d(train_rows, test_rows))),
        "shared_original_raw_ids": int(len(np.intersect1d(train_raw_ids, test_raw_ids))),
        "strict_training_windows": len(strict_train_idx),
        "strict_holdout_windows": len(strict_holdout_idx),
        "strict_training_windows_match_historical": bool(
            strict_train_values.shape == train_values.shape and
            np.array_equal(strict_train_values, train_values)),
        "strict_holdout_windows_match_historical": bool(
            strict_holdout_values.shape == holdout_values.shape and
            np.array_equal(strict_holdout_values, holdout_values)),
        "train_last_timestamp": train_end.isoformat(), "holdout_first_timestamp": holdout_start.isoformat(),
        "separation_seconds": (holdout_start - train_end).total_seconds(),
        "stored_holdout_matches_reconstruction": bool(
            stored_holdout.shape == holdout_values.shape and np.allclose(stored_holdout, holdout_values, atol=1e-7, rtol=1e-6)),
        "current_cleaner_vs_historical_prepared": raw_reconstruction,
        "historical_train_vs_split_first_common_timestamps": compare(reconstructed_train, used_strict),
        "global_cleaning_future_perturbation": observed,
        "split_first_future_perturbation": strict_invariance,
        "strict_partition_rows": {name: len(frame) for name, frame in strict.items()},
        "limitations": [
            "No training or generator evaluation on the split-first inputs has been performed.",
            "A successful perturbation check on these observations is not a general proof of no leakage.",
            "Historical envelope selection and model selection provenance remain unverified.",
            "Bidirectional interpolation remains within each partition; this is offline validation, not causal online imputation.",
            "Any mismatch between current cleaner and historical prepared CSV prevents claiming exact cleaning reproduction.",
        ],
    }
    write_json(out / "temporal_validation.json", report)
    print(__import__("json").dumps(report, indent=2))


if __name__ == "__main__":
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    run(args.raw, args.out)
