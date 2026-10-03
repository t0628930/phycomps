#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical post-hoc metrics for the PhyCoMPS-RWGAN comparison.

This module is deliberately independent from every training loop.  It accepts
real-unit arrays, applies one explicitly supplied training-only z-score, and
computes the same metric definitions for every model.  It contains no torch
import, no optimizer and no training entry point.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


CANONICAL_FEATURES: Tuple[str, ...] = (
    "ph",
    "ec",
    "water_temp",
    "air_temp",
    "rh",
)

# Historical, prescribed operating envelope used by the project.  These are
# not universal agronomic limits and must not be described as such.
OPERATIONAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "ph": (5.38, 6.98),
    "ec": (1.11, 3.21),
    "water_temp": (18.07, 24.93),
    "air_temp": (20.0, 26.2),
    "rh": (52.8, 82.8),
}

SMOOTH_THRESHOLDS: Dict[str, float] = {
    "ph": 0.15,
    "ec": 0.20,
    "water_temp": 3.0,
    "air_temp": 1.0,
    "rh": 4.0,
}

_FEATURE_ALIASES: Dict[str, str] = {
    "ph": "ph",
    "p_h": "ph",
    "ec": "ec",
    "electrical_conductivity": "ec",
    "water_temp": "water_temp",
    "water_temperature": "water_temp",
    "air_temp": "air_temp",
    "air_temperature": "air_temp",
    "rh": "rh",
    "relative_humidity": "rh",
}


def canonical_feature_name(name: Any) -> str:
    """Resolve an exact known alias; never use ambiguous substring matching."""
    token = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in token:
        token = token.replace("__", "_")
    if token not in _FEATURE_ALIASES:
        raise ValueError(
            f"Unknown feature {name!r}. Expected exact aliases for: "
            + ", ".join(CANONICAL_FEATURES)
        )
    return _FEATURE_ALIASES[token]


def canonicalize_feature_names(names: Sequence[Any]) -> Tuple[str, ...]:
    out = tuple(canonical_feature_name(name) for name in names)
    if len(set(out)) != len(out):
        raise ValueError(f"Duplicate canonical feature names: {out}")
    if set(out) != set(CANONICAL_FEATURES) or len(out) != len(CANONICAL_FEATURES):
        raise ValueError(
            f"Expected exactly {CANONICAL_FEATURES}; received canonical order {out}"
        )
    return out


def validate_sequences(array: np.ndarray, feature_names: Sequence[Any], label: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 3:
        raise ValueError(f"{label} must have shape [N,T,F], got {arr.shape}")
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError(f"{label} must contain at least two sequences and two time steps")
    if arr.shape[2] != len(feature_names):
        raise ValueError(
            f"{label} feature dimension {arr.shape[2]} != {len(feature_names)} names"
        )
    if not np.isfinite(arr).all():
        bad = int(np.size(arr) - np.isfinite(arr).sum())
        raise ValueError(f"{label} contains {bad} non-finite values")
    return np.asarray(arr, dtype=np.float64)


@dataclass(frozen=True)
class CanonicalScaler:
    mean: np.ndarray
    std: np.ndarray
    feature_names: Tuple[str, ...] = CANONICAL_FEATURES

    @classmethod
    def fit(cls, train_unit: np.ndarray, feature_names: Sequence[Any]) -> "CanonicalScaler":
        names = canonicalize_feature_names(feature_names)
        arr = validate_sequences(train_unit, names, "train_unit")
        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))
        std = np.where(std > 1e-12, std, 1.0)
        return cls(mean=mean.astype(np.float64), std=std.astype(np.float64), feature_names=names)

    def transform(self, unit: np.ndarray) -> np.ndarray:
        arr = validate_sequences(unit, self.feature_names, "unit_array")
        return (arr - self.mean.reshape(1, 1, -1)) / self.std.reshape(1, 1, -1)

    def inverse_transform(self, norm: np.ndarray) -> np.ndarray:
        arr = validate_sequences(norm, self.feature_names, "normalised_array")
        return arr * self.std.reshape(1, 1, -1) + self.mean.reshape(1, 1, -1)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "provenance": "training_windows_only",
        }


def _covariance(features: np.ndarray, shrinkage: float = 0.05) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Expected a two-dimensional feature matrix")
    if x.shape[0] <= 1:
        return np.eye(x.shape[1], dtype=np.float64)
    cov = np.atleast_2d(np.cov(x, rowvar=False)).astype(np.float64)
    cov = 0.5 * (cov + cov.T)
    if shrinkage > 0:
        p = cov.shape[0]
        target = np.eye(p, dtype=np.float64) * (np.trace(cov) / max(p, 1))
        cov = (1.0 - shrinkage) * cov + shrinkage * target
    return 0.5 * (cov + cov.T)


def sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    mat = 0.5 * (mat + mat.T)
    values, vectors = np.linalg.eigh(mat)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.T


def temporal_statistical_embeddings(x: np.ndarray, max_lag: int = 12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Expected [N,T,F], got {arr.shape}")
    n, t, f = arr.shape
    feats = [
        arr.mean(axis=1),
        arr.std(axis=1),
        arr.min(axis=1),
        arr.max(axis=1),
        np.median(arr, axis=1),
        np.quantile(arr, 0.25, axis=1),
        np.quantile(arr, 0.75, axis=1),
    ]
    time_axis = np.arange(t, dtype=np.float64)
    time_axis = (time_axis - time_axis.mean()) / (time_axis.std() + 1e-12)
    feats.append((arr * time_axis[None, :, None]).mean(axis=1))
    delta = np.diff(arr, axis=1)
    feats.extend([
        delta.mean(axis=1),
        delta.std(axis=1),
        np.abs(delta).max(axis=1),
    ])
    centered = arr - arr.mean(axis=1, keepdims=True)
    variance = np.mean(centered ** 2, axis=1) + 1e-12
    lag_features = []
    for lag in range(1, min(int(max_lag), t - 1) + 1):
        cov = np.mean(centered[:, :-lag, :] * centered[:, lag:, :], axis=1)
        lag_features.append(cov / variance)
    if lag_features:
        feats.append(np.concatenate(lag_features, axis=1))
    embedded = np.concatenate([part.reshape(n, -1) for part in feats], axis=1)
    return np.nan_to_num(embedded, nan=0.0, posinf=1e6, neginf=-1e6)


def stat_fid(real_norm: np.ndarray, fake_norm: np.ndarray, max_lag: int = 12) -> float:
    """Fréchet distance with the PSD-safe covariance sandwich formula."""
    real_emb = temporal_statistical_embeddings(real_norm, max_lag=max_lag)
    fake_emb = temporal_statistical_embeddings(fake_norm, max_lag=max_lag)
    mu_real = real_emb.mean(axis=0)
    mu_fake = fake_emb.mean(axis=0)
    cov_real = _covariance(real_emb) + np.eye(real_emb.shape[1]) * 1e-6
    cov_fake = _covariance(fake_emb) + np.eye(fake_emb.shape[1]) * 1e-6
    sqrt_real = sqrt_psd(cov_real)
    middle = sqrt_real @ cov_fake @ sqrt_real
    cov_mean = sqrt_psd(middle)
    value = float(
        np.dot(mu_real - mu_fake, mu_real - mu_fake)
        + np.trace(cov_real + cov_fake - 2.0 * cov_mean)
    )
    if value < 0 and abs(value) < 1e-8:
        value = 0.0
    return max(0.0, value)


def mmd_rbf(real_norm: np.ndarray, fake_norm: np.ndarray, max_samples: int = 512) -> float:
    n = min(len(real_norm), len(fake_norm), int(max_samples))
    if n < 2:
        return float("nan")
    x = np.asarray(real_norm[:n], dtype=np.float64).reshape(n, -1)
    y = np.asarray(fake_norm[:n], dtype=np.float64).reshape(n, -1)
    joined = np.concatenate([x, y], axis=0)
    sample_n = min(len(joined), 256)
    indices = np.linspace(0, len(joined) - 1, sample_n, dtype=int)
    sample = joined[indices]
    distances = ((sample[:, None, :] - sample[None, :, :]) ** 2).sum(axis=-1)
    positive = distances[distances > 0]
    median = float(np.median(positive)) if positive.size else 1.0
    base_gamma = 1.0 / (median + 1e-12)

    def kernel(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
        return np.exp(-gamma * d2)

    values = []
    for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
        gamma = base_gamma * scale
        values.append(
            kernel(x, x, gamma).mean()
            + kernel(y, y, gamma).mean()
            - 2.0 * kernel(x, y, gamma).mean()
        )
    return float(np.mean(values))


def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    centered = arr - arr.mean(axis=1, keepdims=True)
    variance = np.mean(centered ** 2, axis=1) + 1e-12
    values = []
    for lag in range(1, min(int(max_lag), arr.shape[1] - 1) + 1):
        covariance = np.mean(centered[:, :-lag, :] * centered[:, lag:, :], axis=1)
        values.append(np.mean(covariance / variance, axis=0))
    if not values:
        return np.zeros((0, arr.shape[-1]), dtype=np.float64)
    return np.stack(values, axis=0)


def acf_error(real_norm: np.ndarray, fake_norm: np.ndarray, max_lag: int = 12) -> float:
    a = _acf(real_norm, max_lag)
    b = _acf(fake_norm, max_lag)
    return float(np.mean((a - b) ** 2))


def std_ratio_error(real_norm: np.ndarray, fake_norm: np.ndarray) -> float:
    real_std = np.std(real_norm, axis=(0, 1)) + 1e-12
    fake_std = np.std(fake_norm, axis=(0, 1))
    return float(np.mean(np.abs(fake_std / real_std - 1.0)))


def delta_std_ratio_error(real_norm: np.ndarray, fake_norm: np.ndarray) -> float:
    real_delta = np.diff(real_norm, axis=1)
    fake_delta = np.diff(fake_norm, axis=1)
    real_std = np.std(real_delta, axis=(0, 1)) + 1e-12
    fake_std = np.std(fake_delta, axis=(0, 1))
    return float(np.mean(np.abs(fake_std / real_std - 1.0)))


def violation_score(
    fake_unit: np.ndarray,
    feature_names: Sequence[Any],
    bounds: Mapping[str, Tuple[float, float]] = OPERATIONAL_BOUNDS,
) -> float:
    names = canonicalize_feature_names(feature_names)
    arr = validate_sequences(fake_unit, names, "fake_unit")
    values = []
    for index, name in enumerate(names):
        if name not in bounds:
            raise ValueError(f"No operational bounds configured for {name!r}")
        lower, upper = bounds[name]
        values.append(np.mean((arr[:, :, index] < lower) | (arr[:, :, index] > upper)))
    return float(np.mean(values))


def operational_plausibility_index(
    unit: np.ndarray,
    feature_names: Sequence[Any],
    bounds: Mapping[str, Tuple[float, float]] = OPERATIONAL_BOUNDS,
    smooth_thresholds: Mapping[str, float] = SMOOTH_THRESHOLDS,
) -> float:
    """Project-specific plausibility index; not a universal agronomic score."""
    names = canonicalize_feature_names(feature_names)
    arr = validate_sequences(unit, names, "unit")
    index = {name: i for i, name in enumerate(names)}
    compliance = []
    for name in names:
        lower, upper = bounds[name]
        values = arr[:, :, index[name]]
        compliance.append(np.mean((values >= lower) & (values <= upper)))
    range_compliance = float(np.mean(compliance))

    temp = arr[:, :, index["air_temp"]]
    humidity = arr[:, :, index["rh"]]
    denominator = np.maximum(temp + 237.3, 1e-3)
    psat = 0.6108 * np.exp(np.clip(17.27 * temp / denominator, -50.0, 50.0))
    pv = (humidity / 100.0) * psat
    hd = 2.16679 * (psat - pv) / np.maximum(temp + 273.15, 1.0) * 1000.0
    thermo = float(np.mean(np.isfinite(hd) & (hd >= 0.0) & (hd <= 30.0)))

    jump_violations = []
    for name in names:
        threshold = float(smooth_thresholds[name])
        delta = np.abs(np.diff(arr[:, :, index[name]], axis=1))
        jump_violations.append(np.mean(delta > threshold))
    smoothness = 1.0 - float(np.mean(jump_violations))
    return float(np.clip(0.5 * range_compliance + 0.3 * thermo + 0.2 * smoothness, 0.0, 1.0))


def array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("ascii"))
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def compute_canonical_metrics(
    real_unit: np.ndarray,
    fake_unit: np.ndarray,
    scaler: CanonicalScaler,
    feature_names: Sequence[Any] = CANONICAL_FEATURES,
    max_lag: int = 12,
) -> Dict[str, Any]:
    names = canonicalize_feature_names(feature_names)
    real = validate_sequences(real_unit, names, "real_unit")
    fake = validate_sequences(fake_unit, names, "fake_unit")
    if real.shape[1:] != fake.shape[1:]:
        raise ValueError(f"real/fake trailing shapes differ: {real.shape} vs {fake.shape}")
    n = min(len(real), len(fake))
    real = real[:n]
    fake = fake[:n]
    real_norm = scaler.transform(real)
    fake_norm = scaler.transform(fake)
    metrics: Dict[str, Any] = {
        "n_sequences": int(n),
        "seq_len": int(real.shape[1]),
        "n_features": int(real.shape[2]),
        "stat_fid": stat_fid(real_norm, fake_norm, max_lag=max_lag),
        "mmd_rbf": mmd_rbf(real_norm, fake_norm),
        "acf_error": acf_error(real_norm, fake_norm, max_lag=max_lag),
        "std_ratio_error": std_ratio_error(real_norm, fake_norm),
        "delta_std_ratio_error": delta_std_ratio_error(real_norm, fake_norm),
        "violation_score": violation_score(fake, names),
        "operational_plausibility_index": operational_plausibility_index(fake, names),
        "opi_real_reference": operational_plausibility_index(real, names),
        "real_sha256": array_sha256(real),
        "fake_sha256": array_sha256(fake),
        "metric_contract": "canonical_posthoc_v1",
    }
    for key, value in list(metrics.items()):
        if isinstance(value, float) and not math.isfinite(value):
            metrics[key] = None
    return metrics


_T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
}


def student_t_critical_975(df: int) -> float:
    if int(df) < 1:
        raise ValueError("Student degrees of freedom must be >= 1")
    try:
        from scipy.stats import t as student_t

        return float(student_t.ppf(0.975, int(df)))
    except Exception:
        return _T_CRITICAL_975.get(int(df), 1.96)


def mean_student_ci95(values: Iterable[Any]) -> Tuple[float, float, float, int]:
    clean = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean.append(number)
    n = len(clean)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(clean))
    if n == 1:
        return mean, float("nan"), 0.0, 1
    std = float(np.std(clean, ddof=1))
    half_width = student_t_critical_975(n - 1) * std / math.sqrt(n)
    return mean, float(half_width), std, n


PUBLICATION_METRICS: Tuple[str, ...] = (
    "stat_fid",
    "mmd_rbf",
    "acf_error",
    "std_ratio_error",
    "delta_std_ratio_error",
    "violation_score",
    "operational_plausibility_index",
)


__all__ = [
    "CANONICAL_FEATURES",
    "OPERATIONAL_BOUNDS",
    "CanonicalScaler",
    "PUBLICATION_METRICS",
    "array_sha256",
    "canonical_feature_name",
    "canonicalize_feature_names",
    "compute_canonical_metrics",
    "mean_student_ci95",
    "operational_plausibility_index",
    "stat_fid",
    "student_t_critical_975",
    "violation_score",
]
