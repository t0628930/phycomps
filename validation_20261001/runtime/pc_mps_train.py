#!/usr/bin/env python3
"""Train the exploratory PC-MPS baseline on the hydroponic dataset.

This runner deliberately keeps the MPS as a separate explicit-density model.
It shares data windows, temporal purging, physical ranges, metrics and output
artefacts with the PC-WGAN-GP pipeline.

Current scope: one MPS per sensor channel. This is the tractable option (a)
from the supplied formalisation. It does not model cross-channel dependence or
condition on actuator channels, so the output metadata states that limitation.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from pc_mps import Discretizer, GenerativeMPS
from pcwgan_train import (
    Config as MetricConfig,
    TimeSeriesScaler,
    compute_metrics,
    ensure_dir,
    get_device,
    make_windows,
    save_json,
    set_seed,
    update_physics_thresholds_from_real_data,
    window_purge_count,
)


DEFAULT_FEATURES = ["ph", "ec", "water_temp", "air_temp", "rh"]


@dataclass
class MPSConfig:
    data_path: str
    output_dir: str
    feature_cols: list[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))
    condition_cols: list[str] = field(default_factory=list)
    time_col: Optional[str] = None
    seq_len: int = 120
    stride: int = 15
    max_gap_seconds: float = 60.0
    holdout_test_ratio: float = 0.15
    bins: int = 12
    bond_dim: int = 8
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.01
    lambda_bound: float = 2.0
    lambda_dyn: float = 1.0
    dynamic_quantile: float = 0.99
    domain_margin: float = 0.10
    grad_clip_norm: float = 5.0
    device: str = "auto"
    seed: int = 42


def validate_config(cfg: MPSConfig) -> None:
    positive = ("seq_len", "stride", "bins", "bond_dim", "epochs", "batch_size")
    invalid = [name for name in positive if int(getattr(cfg, name)) <= 0]
    if invalid:
        raise ValueError(f"Configuration values must be positive: {invalid}")
    if cfg.bins < 2:
        raise ValueError("bins must be at least 2")
    if not 0.0 < cfg.holdout_test_ratio < 1.0:
        raise ValueError("holdout_test_ratio must be in (0, 1)")
    if not 0.5 <= cfg.dynamic_quantile < 1.0:
        raise ValueError("dynamic_quantile must be in [0.5, 1)")
    if cfg.lambda_bound < 0.0 or cfg.lambda_dyn < 0.0:
        raise ValueError("physical loss weights must be non-negative")


def _feature_bounds(name: str, cfg: MetricConfig) -> tuple[float, float]:
    lower = name.lower()
    if "water" in lower and "temp" in lower:
        return float(cfg.water_temp_min), float(cfg.water_temp_max)
    if "ph" in lower:
        return float(cfg.ph_min), float(cfg.ph_max)
    if "ec" in lower or "tds" in lower:
        return float(cfg.ec_min), float(cfg.ec_max)
    if "rh" in lower or "humid" in lower:
        return float(cfg.rh_min), float(cfg.rh_max)
    if "temp" in lower:
        return float(cfg.temp_min), float(cfg.temp_max)
    values = getattr(cfg, "feature_ranges", {}).get(name)
    if values is not None:
        return float(values[0]), float(values[1])
    raise ValueError(f"No physical bounds configured for feature {name!r}")


def _dynamic_threshold(name: str, cfg: MetricConfig, train_values: np.ndarray, quantile: float) -> float:
    lower = name.lower()
    key = None
    if "water" in lower and "temp" in lower:
        key = "water_temp"
    elif "ph" in lower:
        key = "ph"
    elif "ec" in lower or "tds" in lower:
        key = "ec"
    elif "rh" in lower or "humid" in lower:
        key = "rh"
    elif "temp" in lower:
        key = "air_temp"
    configured = float(getattr(cfg, f"smooth_threshold_{key}", 0.0)) if key else 0.0
    differences = np.abs(np.diff(train_values, axis=1)).reshape(-1)
    differences = differences[np.isfinite(differences)]
    empirical = float(np.quantile(differences, quantile)) if differences.size else 0.0
    return max(configured, empirical, 1e-8)


def _build_discretizer(values: np.ndarray, bounds: tuple[float, float], bins: int, margin: float) -> Discretizer:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot discretize a feature containing no finite values")
    physical_low, physical_high = bounds
    base_low = min(float(finite.min()), physical_low)
    base_high = max(float(finite.max()), physical_high)
    span = max(base_high - base_low, abs(base_low) * 1e-6, 1e-6)
    padding = max(0.0, float(margin)) * span
    return Discretizer(base_low - padding, base_high + padding, int(bins))


def _load_windows(cfg: MPSConfig) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    path = Path(cfg.data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    missing_features = [name for name in cfg.feature_cols if name not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    missing_conditions = [name for name in cfg.condition_cols if name not in frame.columns]
    if missing_conditions:
        raise ValueError(f"Missing condition columns: {missing_conditions}")
    condition_cols = list(cfg.condition_cols)
    if not condition_cols:
        frame = frame.copy()
        frame["condition_constant"] = 1.0
        condition_cols = ["condition_constant"]
    windows, conditions = make_windows(
        frame,
        feature_cols=cfg.feature_cols,
        condition_cols=condition_cols,
        seq_len=cfg.seq_len,
        stride=cfg.stride,
        time_col=cfg.time_col,
        max_gap_seconds=cfg.max_gap_seconds,
    )
    return frame, windows, conditions


def purged_holdout_split(
    windows: np.ndarray,
    conditions: np.ndarray,
    test_ratio: float,
    purge_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(windows) != len(conditions):
        raise ValueError("windows and conditions must have the same length")
    n_test = max(1, int(len(windows) * float(test_ratio)))
    test_start = len(windows) - n_test
    train_end = test_start - max(0, int(purge_windows))
    if train_end < 2:
        raise ValueError("Not enough windows for a purged train/hold-out split")
    return windows[:train_end], conditions[:train_end], windows[test_start:], conditions[test_start:]


def _prepare_models(
    cfg: MPSConfig,
    train: np.ndarray,
    metric_cfg: MetricConfig,
    device: torch.device,
) -> tuple[list[GenerativeMPS], list[Discretizer], list[torch.Tensor], list[torch.Tensor], list[float]]:
    models: list[GenerativeMPS] = []
    discretizers: list[Discretizer] = []
    allowed_sites: list[torch.Tensor] = []
    transition_masks: list[torch.Tensor] = []
    dynamic_thresholds: list[float] = []
    for feature_index, feature_name in enumerate(cfg.feature_cols):
        bounds = _feature_bounds(feature_name, metric_cfg)
        values = train[:, :, feature_index]
        discretizer = _build_discretizer(values, bounds, cfg.bins, cfg.domain_margin)
        threshold = _dynamic_threshold(feature_name, metric_cfg, values, cfg.dynamic_quantile)
        model = GenerativeMPS(
            length=cfg.seq_len,
            bins=cfg.bins,
            bond_dim=cfg.bond_dim,
            seed=cfg.seed + feature_index,
        ).to(device)
        allowed = discretizer.allowed_indices(*bounds, device=device)
        models.append(model)
        discretizers.append(discretizer)
        allowed_sites.append(allowed)
        transition_masks.append(discretizer.transition_mask(threshold, device=device))
        dynamic_thresholds.append(threshold)
    return models, discretizers, allowed_sites, transition_masks, dynamic_thresholds


def _encode(train: np.ndarray, discretizers: Sequence[Discretizer]) -> torch.Tensor:
    encoded = [
        discretizer.to_index(torch.from_numpy(train[:, :, index]))
        for index, discretizer in enumerate(discretizers)
    ]
    return torch.stack(encoded, dim=-1)


def train_models(
    cfg: MPSConfig,
    train: np.ndarray,
    models: Sequence[GenerativeMPS],
    discretizers: Sequence[Discretizer],
    allowed_bins: Sequence[torch.Tensor],
    transition_masks: Sequence[torch.Tensor],
    output_dir: Path,
) -> list[dict[str, float]]:
    device = next(models[0].parameters()).device
    encoded = _encode(train, discretizers)
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(
        TensorDataset(encoded),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    parameters = [parameter for model in models for parameter in model.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=cfg.learning_rate)
    history: list[dict[str, float]] = []
    feature_count = len(models)

    for epoch in range(1, cfg.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        nll_total = 0.0
        sample_total = 0
        batch_count = max(1, len(loader))
        for (batch_cpu,) in loader:
            batch = batch_cpu.to(device)
            batch_nll = torch.stack([
                model.nll(batch[:, :, feature_index])
                for feature_index, model in enumerate(models)
            ]).mean()
            (batch_nll / batch_count).backward()
            nll_total += float(batch_nll.detach().cpu()) * len(batch)
            sample_total += len(batch)

        bound_loss = torch.stack([
            model.allowed_log_barrier([allowed] * cfg.seq_len)
            for model, allowed in zip(models, allowed_bins)
        ]).mean()
        dynamic_loss = torch.stack([
            model.forbidden_mass_dynamics(mask)
            for model, mask in zip(models, transition_masks)
        ]).mean()
        physical_loss = cfg.lambda_bound * bound_loss + cfg.lambda_dyn * dynamic_loss
        physical_loss.backward()
        if cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip_norm)
        optimizer.step()
        for model in models:
            model.normalize_cores_()

        row = {
            "epoch": float(epoch),
            "nll": nll_total / max(1, sample_total),
            "nll_per_site": nll_total / max(1, sample_total * cfg.seq_len),
            "bound_log_barrier_per_site": float(bound_loss.detach().cpu()),
            "dynamic_mass": float(dynamic_loss.detach().cpu()),
            "physical_loss": float(physical_loss.detach().cpu()),
        }
        history.append(row)
        if epoch == 1 or epoch == cfg.epochs or epoch % max(1, cfg.epochs // 10) == 0:
            print(
                f"[MPS] epoch={epoch:04d} nll/site={row['nll_per_site']:.5f} "
                f"bound_barrier={row['bound_log_barrier_per_site']:.5f} "
                f"dyn={row['dynamic_mass']:.5f}"
            )
        if epoch % max(1, cfg.epochs // 5) == 0 or epoch == cfg.epochs:
            torch.save(
                {"epoch": epoch, "models": [model.state_dict() for model in models]},
                output_dir / "checkpoint_latest.pt",
            )
    return history


@torch.no_grad()
def generate_real_unit(
    models: Sequence[GenerativeMPS],
    discretizers: Sequence[Discretizer],
    count: int,
) -> np.ndarray:
    channels = []
    for model, discretizer in zip(models, discretizers):
        indices = model.sample(count)
        channels.append(discretizer.to_value(indices).cpu().numpy())
    return np.stack(channels, axis=-1).astype(np.float32)


@torch.no_grad()
def evaluate_exact(
    test: np.ndarray,
    models: Sequence[GenerativeMPS],
    discretizers: Sequence[Discretizer],
    allowed_bins: Sequence[torch.Tensor],
    transition_masks: Sequence[torch.Tensor],
    batch_size: int,
) -> dict[str, Any]:
    device = next(models[0].parameters()).device
    encoded = _encode(test, discretizers)
    loader = DataLoader(TensorDataset(encoded), batch_size=batch_size, shuffle=False)
    log_likelihood_sum = np.zeros(len(models), dtype=np.float64)
    count = 0
    for (batch_cpu,) in loader:
        batch = batch_cpu.to(device)
        for feature_index, model in enumerate(models):
            log_likelihood_sum[feature_index] += float(model.log_prob(batch[:, :, feature_index]).sum().cpu())
        count += len(batch)
    per_feature = (log_likelihood_sum / max(1, count)).tolist()
    bound = [
        float(model.forbidden_mass_bounds([allowed] * model.length).cpu())
        for model, allowed in zip(models, allowed_bins)
    ]
    dynamic = [
        float(model.forbidden_mass_dynamics(mask).cpu())
        for model, mask in zip(models, transition_masks)
    ]
    entropy = [model.entanglement_entropy(model.length // 2) for model in models]
    return {
        "exact_log_likelihood_per_feature": per_feature,
        "exact_log_likelihood_mean": float(np.mean(per_feature)),
        "exact_log_likelihood_per_site": float(np.mean(per_feature) / test.shape[1]),
        "exact_bound_mass_per_feature": bound,
        "exact_bound_mass_mean": float(np.mean(bound)),
        "exact_dynamic_mass_per_feature": dynamic,
        "exact_dynamic_mass_mean": float(np.mean(dynamic)),
        "midpoint_entanglement_entropy_per_feature": entropy,
    }


def run_pipeline(cfg: MPSConfig) -> dict[str, Any]:
    validate_config(cfg)
    set_seed(cfg.seed)
    output_dir = ensure_dir(cfg.output_dir)
    frame, windows, conditions = _load_windows(cfg)
    purge = window_purge_count(cfg.seq_len, cfg.stride)
    train, train_conditions, test, test_conditions = purged_holdout_split(
        windows,
        conditions,
        test_ratio=cfg.holdout_test_ratio,
        purge_windows=purge,
    )
    del train_conditions, test_conditions
    print(f"[MPS] windows={len(windows)} train={len(train)} holdout={len(test)} purge={purge}")

    metric_cfg = MetricConfig(seq_len=cfg.seq_len, stride=cfg.stride, seed=cfg.seed)
    update_physics_thresholds_from_real_data(metric_cfg, train, cfg.feature_cols)
    device = get_device(cfg.device)
    models, discretizers, allowed_bins, transition_masks, dynamic_thresholds = _prepare_models(
        cfg, train, metric_cfg, device
    )
    history = train_models(
        cfg,
        train,
        models,
        discretizers,
        allowed_bins,
        transition_masks,
        output_dir,
    )
    fake_real = generate_real_unit(models, discretizers, len(test))
    scaler = TimeSeriesScaler("zscore").fit(train)
    real_norm = scaler.transform(test).astype(np.float32)
    fake_norm = scaler.transform(fake_real).astype(np.float32)
    np.save(output_dir / "real_sequences.npy", real_norm)
    np.save(output_dir / "fake_sequences.npy", fake_norm)
    np.save(output_dir / "real_sequences_real_unit.npy", test.astype(np.float32))
    np.save(output_dir / "fake_sequences_real_unit.npy", fake_real)
    np.save(output_dir / "zscore_mu.npy", np.asarray(scaler.mean).reshape(-1))
    np.save(output_dir / "zscore_sigma.npy", np.asarray(scaler.std).reshape(-1))

    exact = evaluate_exact(
        test,
        models,
        discretizers,
        allowed_bins,
        transition_masks,
        batch_size=cfg.batch_size,
    )
    metrics = compute_metrics(real_norm, fake_norm, cfg.feature_cols, metric_cfg, scaler=scaler)
    metadata: dict[str, Any] = {
        "status": "ok",
        "model": "pc_mps_independent_channels",
        "conditional": False,
        "channel_factorization": "one_mps_per_feature",
        "comparability": "exploratory_unconditional_baseline",
        "limitations": [
            "Sensor channels are generated independently.",
            "Actuator condition columns are used for aligned windowing but not as model inputs.",
        ],
        "n_training_sequences": int(len(train)),
        "n_holdout_sequences": int(len(test)),
        "feature_cols": list(cfg.feature_cols),
        "condition_cols": list(cfg.condition_cols),
        "purge_windows": int(purge),
        **metrics,
        **exact,
    }
    save_json(metadata, output_dir / "metrics.json")
    save_json(asdict(cfg), output_dir / "config.json")
    save_json(
        {
            "discretizers": [item.as_dict() for item in discretizers],
            "physical_bounds": [list(_feature_bounds(name, metric_cfg)) for name in cfg.feature_cols],
            "dynamic_thresholds": dynamic_thresholds,
        },
        output_dir / "mps_physics_config.json",
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    frame.to_csv(output_dir / "input_dataset_snapshot.csv", index=False)
    torch.save(
        {
            "models": [model.state_dict() for model in models],
            "feature_cols": cfg.feature_cols,
            "discretizers": [item.as_dict() for item in discretizers],
            "config": asdict(cfg),
        },
        output_dir / "pc_mps_final.pt",
    )
    print(f"[MPS] completed; outputs saved to {output_dir.resolve()}")
    return metadata


def parse_args() -> MPSConfig:
    parser = argparse.ArgumentParser(description="PC-MPS hydroponic baseline")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_cols", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--condition_cols", default="")
    parser.add_argument("--time_col", default="")
    parser.add_argument("--seq_len", type=int, default=120)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--max_gap_seconds", type=float, default=60.0)
    parser.add_argument("--holdout_test_ratio", type=float, default=0.15)
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--bond_dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--lambda_bound", type=float, default=2.0)
    parser.add_argument("--lambda_dyn", type=float, default=1.0)
    parser.add_argument("--dynamic_quantile", type=float, default=0.99)
    parser.add_argument("--domain_margin", type=float, default=0.10)
    parser.add_argument("--grad_clip_norm", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return MPSConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        feature_cols=[item.strip() for item in args.feature_cols.split(",") if item.strip()],
        condition_cols=[item.strip() for item in args.condition_cols.split(",") if item.strip()],
        time_col=args.time_col or None,
        seq_len=args.seq_len,
        stride=args.stride,
        max_gap_seconds=args.max_gap_seconds,
        holdout_test_ratio=args.holdout_test_ratio,
        bins=args.bins,
        bond_dim=args.bond_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lambda_bound=args.lambda_bound,
        lambda_dyn=args.lambda_dyn,
        dynamic_quantile=args.dynamic_quantile,
        domain_margin=args.domain_margin,
        grad_clip_norm=args.grad_clip_norm,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    run_pipeline(parse_args())
