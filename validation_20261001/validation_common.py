"""Shared helpers for the new revision validation, not historical training code."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT.parent / "Frozen_Reproducibility"
sys.path.insert(0, str(FROZEN))
import canonical_metrics as metrics

FEATURES = list(metrics.CANONICAL_FEATURES)
SENSORS = ["pH", "TDS", "water_temp", "DHT_temp", "DHT_humidity"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_cleaner():
    path = FROZEN / "scripts/prepare_hydroponic_dataset_v4_gap_clean.py"
    spec = importlib.util.spec_from_file_location("frozen_cleaner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def window_indices(frame, length=120, stride=30):
    rows = []
    for _, group in frame.groupby("segment_id", sort=False):
        group = group.sort_values("timestamp")
        times = pd.to_datetime(group.timestamp)
        for start in range(0, len(group) - length + 1, stride):
            if times.iloc[start:start + length].diff().dt.total_seconds().max() > 60:
                continue
            rows.append(group.index[start:start + length].to_numpy())
    if not rows:
        raise ValueError("No eligible windows")
    return np.stack(rows)


def historical_split(frame, length=120, stride=30, holdout_ratio=.15):
    indices = window_indices(frame, length, stride)
    n_test = int(len(indices) * holdout_ratio)
    purge = int(np.ceil(length / stride)) - 1
    if n_test < 1 or len(indices) - n_test - purge < 1:
        raise ValueError("Insufficient windows for purged split")
    return indices[:-n_test - purge], indices[-n_test:], indices


def strict_prepare(raw, train_end, holdout_start, cleaner):
    """Partition raw timestamps BEFORE interpolation, with a separate embargo."""
    raw = raw.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    if pd.Timestamp(train_end) >= pd.Timestamp(holdout_start):
        raise ValueError("train_end must precede holdout_start")
    masks = {
        "train": raw.timestamp <= pd.Timestamp(train_end),
        "embargo": (raw.timestamp > pd.Timestamp(train_end)) & (raw.timestamp < pd.Timestamp(holdout_start)),
        "holdout": raw.timestamp >= pd.Timestamp(holdout_start),
    }
    result = {}
    for name, mask in masks.items():
        part = raw.loc[mask].copy()
        if part.empty:
            result[name] = pd.DataFrame()
        else:
            result[name] = cleaner.to_pipeline_format(cleaner.clean(part))
    return result


def opi_components(array, hd_limit=30., smooth_thresholds=None):
    array = metrics.validate_sequences(array, FEATURES, "sequences")
    thresholds = smooth_thresholds or metrics.SMOOTH_THRESHOLDS
    bounds = np.array([metrics.OPERATIONAL_BOUNDS[name] for name in FEATURES])
    compliance = ((array >= bounds[:, 0]) & (array <= bounds[:, 1])).mean()
    temp, rh = array[..., 3], array[..., 4]
    saturation = .6108 * np.exp(np.clip(17.27 * temp / np.maximum(temp + 237.3, .001), -50, 50))
    deficit = 2166.79 * saturation * (1 - rh / 100) / np.maximum(temp + 273.15, 1)
    thermo = (np.isfinite(deficit) & (deficit >= 0) & (deficit <= hd_limit)).mean()
    jumps = np.abs(np.diff(array, axis=1))
    smooth = 1 - (jumps > np.array([thresholds[name] for name in FEATURES])).mean()
    return np.array([compliance, thermo, smooth]), deficit


def weight_grid(step_count=10):
    return np.array([(a, b, step_count - a - b)
                     for a in range(step_count + 1)
                     for b in range(step_count + 1 - a)], dtype=float) / step_count
