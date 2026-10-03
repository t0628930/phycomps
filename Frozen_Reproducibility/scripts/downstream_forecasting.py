from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from results_tools.common import (
    SEEDS,
    bootstrap_mean_ci,
    format_number,
    parse_seeds,
    read_json,
    safe_float,
    write_csv,
    write_json,
)


FEATURES = ["ph", "ec", "water_temp", "air_temp", "rh"]
CONDITIONS = (
    "Real only",
    "PhyCoMPS synthetic only",
    "Real + PhyCoMPS synthetic",
    "Real + PC-cWGAN-GP synthetic",
)


def selected_rows(manifest_path: Path) -> list[dict[str, Any]]:
    payload = read_json(manifest_path)
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    return [
        row for row in rows
        if str(row.get("selected_as_canonical", "")).lower() == "true"
        and str(row.get("status")) == "complete"
    ]


def row_for(rows: Iterable[dict[str, Any]], config: str, seed: int) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("config") == config and int(row.get("seed", -1)) == seed), None)


def array_paths(row: dict[str, Any]) -> tuple[Path, Path]:
    values = [Path(value) for value in str(row.get("arrays_path", "")).split(";") if value]
    if len(values) != 2:
        raise ValueError(f"Expected real/fake array pair for {row.get('config')} seed {row.get('seed')}")
    return values[0], values[1]


def make_gap_safe_windows(
    data_path: Path,
    seq_len: int,
    stride: int,
    max_gap_seconds: float,
) -> np.ndarray:
    frame = pd.read_csv(data_path)
    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset misses forecasting features: {missing}")
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        frame = frame.sort_values("timestamp").reset_index(drop=True)
    if "segment_id" not in frame.columns:
        if "timestamp" in frame.columns:
            delta = frame["timestamp"].diff().dt.total_seconds()
            median = float(delta.dropna().median()) if delta.notna().any() else 10.0
            frame["segment_id"] = (delta.fillna(median) > max_gap_seconds).cumsum().astype(int)
        else:
            frame["segment_id"] = 0
    windows: list[np.ndarray] = []
    for _, group in frame.groupby("segment_id", sort=True):
        values = group[FEATURES].to_numpy(dtype=np.float32)
        for start in range(0, len(values) - seq_len + 1, stride):
            windows.append(values[start : start + seq_len])
    if not windows:
        raise ValueError("No gap-safe windows could be constructed")
    return np.stack(windows)


def purged_split(windows: np.ndarray, ratio: float, purge: int) -> tuple[np.ndarray, np.ndarray]:
    n_test = max(1, int(len(windows) * ratio))
    test_start = len(windows) - n_test
    train_end = test_start - max(0, purge)
    if train_end < 2:
        raise ValueError("Not enough windows for the purged downstream split")
    return windows[:train_end], windows[test_start:]


def supervised_pairs(sequences: np.ndarray, context: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(sequences, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] <= context:
        raise ValueError(f"Expected [N,T,F] arrays with T>{context}; got {values.shape}")
    x_parts = [values[:, step - context : step, :].reshape(len(values), -1) for step in range(context, values.shape[1])]
    y_parts = [values[:, step, :] for step in range(context, values.shape[1])]
    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    mask = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    return x[mask], y[mask]


def capped(x: np.ndarray, y: np.ndarray, maximum: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= maximum:
        return x, y
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(x), size=maximum, replace=False))
    return x[indices], y[indices]


def convert_to_physical(fake: np.ndarray, row: dict[str, Any], full_row: dict[str, Any]) -> np.ndarray:
    if str(row.get("arrays_units")) == "physical":
        return np.asarray(fake, dtype=np.float64)
    full_run = Path(str(full_row["run_dir"]))
    mean_path = full_run / "zscore_mu.npy"
    sigma_path = full_run / "zscore_sigma.npy"
    if not (mean_path.is_file() and sigma_path.is_file()):
        raise ValueError("Cannot verify inverse transformation for standardized synthetic arrays")
    mean = np.load(mean_path, allow_pickle=False).reshape(1, 1, -1)
    sigma = np.load(sigma_path, allow_pickle=False).reshape(1, 1, -1)
    if fake.shape[-1] != mean.shape[-1]:
        raise ValueError("Scaler dimension does not match synthetic arrays")
    return np.asarray(fake, dtype=np.float64) * sigma + mean


def fit_and_score(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", tol=1e-6)
    model.fit((train_x - x_mean) / x_scale, (train_y - y_mean) / y_scale)
    pred_scaled = model.predict((test_x - x_mean) / x_scale)
    prediction = pred_scaled * y_scale + y_mean
    error = prediction - test_y
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(np.square(error))))


def bootstrap_summary(raw_rows: list[dict[str, Any]], condition: str, metric: str) -> dict[str, Any]:
    values = [row[metric] for row in raw_rows if row["training_data"] == condition]
    return bootstrap_mean_ci(values, random_state=20_260_711 + (0 if metric == "mae" else 1))


def write_latex(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table*}[!t]",
        r"\caption{Frozen one-step forecasting evaluation on the identical real hold-out. Positive gain denotes lower error than real-only training.}",
        r"\label{tab:downstream}",
        r"\centering",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Training data & $n$ & MAE & RMSE & MAE gain vs. real-only \\",
        r"\midrule",
    ]
    for row in summary_rows:
        condition = str(row["training_data"]).replace("PhyCoMPS", r"PhyCoMPS")
        mae = format_number(row["mae_mean"], "mae")
        rmse = format_number(row["rmse_mean"], "rmse")
        if int(row["n"]) > 1:
            mae += r" $\pm$ " + format_number(row["mae_ci_half_width"], "mae")
            rmse += r" $\pm$ " + format_number(row["rmse_ci_half_width"], "rmse")
        gain = format_number(row["gain_vs_real_only_mean"], "mae")
        lines.append(f"{condition} & {row['n']} & {mae} & {rmse} & {gain} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a frozen, matched-holdout forecasting utility experiment.")
    parser.add_argument("--project-root", type=Path, default=Path("PhyCoMPS_FINAL_RUN"))
    parser.add_argument("--manifest", type=Path, default=Path("ARTICLE_TRACER_OUTPUTS/results_manifest.json"))
    parser.add_argument("--data-csv", type=Path, default=Path("PhyCoMPS_FINAL_RUN/hydro_dataset_pcwgan_gap_clean.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("ARTICLE_TRACER_OUTPUTS/tables"))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--context", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-train-pairs", type=int, default=50_000)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    rows = selected_rows(args.manifest.resolve())
    seeds = parse_seeds(args.seeds)
    full_rows = {seed: row_for(rows, "full", seed) for seed in seeds}
    pcwgan_rows = {seed: row_for(rows, "pc_cwgan_gp", seed) for seed in seeds}
    if any(row is None for row in full_rows.values()) or any(row is None for row in pcwgan_rows.values()):
        print("Downstream inputs are incomplete in the canonical manifest", file=sys.stderr)
        return 2

    config = read_json(Path(str(full_rows[seeds[0]]["config_path"])))
    seq_len = int(config.get("seq_len", 120))
    stride = int(config.get("stride", 30))
    ratio = float(config.get("holdout_test_ratio", 0.15))
    max_gap = float(config.get("max_gap_seconds", 60.0))
    purge = max(0, math.ceil(seq_len / stride) - 1)
    windows = make_gap_safe_windows(args.data_csv.resolve(), seq_len, stride, max_gap)
    train_real, reconstructed_holdout = purged_split(windows, ratio, purge)

    first_real_path, _ = array_paths(full_rows[seeds[0]])
    holdout = np.load(first_real_path, allow_pickle=False).astype(np.float64)
    if str(full_rows[seeds[0]].get("arrays_units")) != "physical":
        raise ValueError("Canonical PhyCoMPS hold-out is not in verified physical units")
    if holdout.shape != reconstructed_holdout.shape or not np.allclose(holdout, reconstructed_holdout, rtol=1e-5, atol=1e-5):
        raise ValueError("Reconstructed real hold-out does not match the canonical exported hold-out")
    for seed in seeds[1:]:
        real_path, _ = array_paths(full_rows[seed])
        candidate = np.load(real_path, allow_pickle=False)
        if candidate.shape != holdout.shape or not np.allclose(candidate, holdout, rtol=1e-5, atol=1e-5):
            raise ValueError(f"Real hold-out differs for seed {seed}")

    real_x, real_y = supervised_pairs(train_real, args.context)
    test_x, test_y = supervised_pairs(holdout, args.context)
    real_x, real_y = capped(real_x, real_y, args.max_train_pairs, seed=20_260_711)
    x_mean = real_x.mean(axis=0)
    x_scale = real_x.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    y_mean = real_y.mean(axis=0)
    y_scale = real_y.std(axis=0)
    y_scale[y_scale < 1e-8] = 1.0

    baseline_mae, baseline_rmse = fit_and_score(
        real_x, real_y, test_x, test_y, x_mean, x_scale, y_mean, y_scale, args.alpha
    )
    raw_rows: list[dict[str, Any]] = []
    for seed in seeds:
        _, phy_fake_path = array_paths(full_rows[seed])
        _, pcwgan_fake_path = array_paths(pcwgan_rows[seed])
        phy_fake = convert_to_physical(
            np.load(phy_fake_path, allow_pickle=False), full_rows[seed], full_rows[seed]
        )
        pcwgan_fake = convert_to_physical(
            np.load(pcwgan_fake_path, allow_pickle=False), pcwgan_rows[seed], full_rows[seed]
        )
        phy_x, phy_y = supervised_pairs(phy_fake, args.context)
        pc_x, pc_y = supervised_pairs(pcwgan_fake, args.context)
        phy_x, phy_y = capped(phy_x, phy_y, args.max_train_pairs, seed)
        pc_x, pc_y = capped(pc_x, pc_y, args.max_train_pairs, seed)
        datasets = {
            "Real only": (real_x, real_y),
            "PhyCoMPS synthetic only": (phy_x, phy_y),
            "Real + PhyCoMPS synthetic": (np.concatenate([real_x, phy_x]), np.concatenate([real_y, phy_y])),
            "Real + PC-cWGAN-GP synthetic": (np.concatenate([real_x, pc_x]), np.concatenate([real_y, pc_y])),
        }
        for condition, (train_x, train_y) in datasets.items():
            if condition == "Real only":
                mae, rmse = baseline_mae, baseline_rmse
            else:
                mae, rmse = fit_and_score(
                    train_x, train_y, test_x, test_y,
                    x_mean, x_scale, y_mean, y_scale, args.alpha,
                )
            raw_rows.append(
                {
                    "training_data": condition,
                    "seed": seed,
                    "n_train_pairs": len(train_x),
                    "n_test_pairs": len(test_x),
                    "mae": mae,
                    "rmse": rmse,
                    "gain_vs_real_only": baseline_mae - mae,
                    "holdout_sha256": hashlib.sha256(holdout.tobytes(order="C")).hexdigest(),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    baseline_by_seed = {row["seed"]: row["mae"] for row in raw_rows if row["training_data"] == "Real only"}
    from results_tools.common import paired_bootstrap_advantage
    for condition in CONDITIONS:
        mae = bootstrap_summary(raw_rows, condition, "mae")
        rmse = bootstrap_summary(raw_rows, condition, "rmse")
        current = {row["seed"]: row["mae"] for row in raw_rows if row["training_data"] == condition}
        gain = paired_bootstrap_advantage(baseline_by_seed, current, "mae")
        summary_rows.append(
            {
                "training_data": condition,
                "n": mae["n"],
                "mae_mean": mae["mean"],
                "mae_ci_low": mae["ci_low"],
                "mae_ci_high": mae["ci_high"],
                "mae_ci_half_width": mae["ci_half_width"],
                "rmse_mean": rmse["mean"],
                "rmse_ci_low": rmse["ci_low"],
                "rmse_ci_high": rmse["ci_high"],
                "rmse_ci_half_width": rmse["ci_half_width"],
                "gain_vs_real_only_mean": gain["mean"],
                "paired_ci_low": gain["ci_low"],
                "paired_ci_high": gain["ci_high"],
                "matched_seeds": ";".join(str(seed) for seed in gain["matched_seeds"]),
            }
        )

    outdir = args.outdir.resolve()
    write_csv(outdir / "downstream_forecasting_raw.csv", raw_rows)
    write_csv(outdir / "downstream_forecasting_summary.csv", summary_rows)
    write_latex(outdir / "downstream_forecasting_summary.tex", summary_rows)
    write_json(
        outdir / "downstream_forecasting_metadata.json",
        {
            "status": "complete",
            "protocol": "frozen one-step multi-output ridge forecasting",
            "features": FEATURES,
            "context_steps": args.context,
            "ridge_alpha": args.alpha,
            "predictor_selection": "none; predictor and alpha fixed before inspecting downstream scores",
            "real_holdout_identical_across_conditions": True,
            "holdout_sha256": raw_rows[0]["holdout_sha256"],
            "holdout_windows": int(len(holdout)),
            "purge_windows": purge,
            "synthetic_export_limitation": (
                "The available synthetic arrays were generated with evaluation-condition sequences. "
                "They were not used for model or hyperparameter selection; downstream estimates remain exploratory."
            ),
            "bootstrap_replications": 10_000,
            "random_state": 20_260_711,
        },
    )
    print(f"Wrote {len(raw_rows)} downstream rows and {len(summary_rows)} summaries to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

