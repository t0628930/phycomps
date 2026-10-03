#!/usr/bin/env python3
"""Multi-seed ablation harness for PhyCoMPS-RWGAN.

Run from the My_Scripts root with::

    python -m pc_cmps_rwgan.ablation --data_path data.csv --output_root outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pc_mps_train import MPSConfig, run_pipeline as run_pc_mps
from pcwgan_train import Config as PCWGANConfig
from pcwgan_train import run_pipeline as run_pcwgan

from .train import HybridConfig, run_pipeline as run_hybrid


@dataclass(frozen=True)
class AblationSpec:
    key: str
    question: str
    runner: str = "hybrid"
    overrides: tuple[tuple[str, Any], ...] = ()

    def override_dict(self) -> dict[str, Any]:
        return dict(self.overrides)


ABLATIONS = [
    AblationSpec("full", "Modele complet de reference."),
    AblationSpec(
        "K1",
        "Le melange conditionnel apporte-t-il un gain par rapport a un seul MPS ?",
        overrides=(("mps_experts", 1), ("lambda_gate_balance", 0.0), ("lambda_gate_entropy", 0.0)),
    ),
    AblationSpec(
        "gate_constant",
        "Les actionneurs sont-ils utiles au gate ? Le melange global reste appris.",
        overrides=(("gate_constant", True),),
    ),
    AblationSpec(
        "prior_iid_uniform",
        "Le MPS conditionnel bat-il un prior categoriel IID uniforme ?",
        overrides=(
            ("prior_iid_uniform", True),
            ("mps_pretrain_epochs", 0),
            ("lambda_mps_bound", 0.0),
            ("lambda_gate_balance", 0.0),
            ("lambda_gate_entropy", 0.0),
        ),
    ),
    AblationSpec(
        "no_code",
        "La coherence de code empeche-t-elle le decodeur d'ignorer le prior ?",
        overrides=(("lambda_code", 0.0),),
    ),
    AblationSpec(
        "no_bound_exact",
        "La masse physique exacte du MPS reduit-elle les violations ?",
        overrides=(("lambda_mps_bound", 0.0),),
    ),
    AblationSpec(
        "no_phys_continuous",
        "La loss physique continue apporte-t-elle un gain au-dela du bornage architectural ?",
        overrides=(("lambda_physics", 0.0),),
    ),
    AblationSpec(
        "non_residual_decoder",
        "L'ancrage residuel aux centres des bins est-il utile ?",
        overrides=(("non_residual_decoder", True),),
    ),
    AblationSpec(
        "pc_mps_only",
        "Le raffinement conditionnel adversarial est-il utile face au PC-MPS autonome ?",
        runner="pc_mps",
    ),
    AblationSpec(
        "pc_cwgan_gp",
        "L'hybride apporte-t-il un gain face au PC-cWGAN-GP canonique ?",
        runner="pcwgan",
    ),
]


PREFERRED_METRICS = [
    "stat_fid",
    "fid_temporal",
    "mmd_rbf",
    "dtw_approx",
    "acf_error",
    "std_ratio_error",
    "delta_std_ratio_error",
    "aqi",
    "violation_score",
    "composite_score",
    "exact_conditional_log_likelihood_per_site",
    "exact_conditional_bound_log_barrier_per_site",
    "exact_log_likelihood_per_site",
    "exact_bound_mass_mean",
    "code_recovery_accuracy",
    "decoder_coarse_mse",
]

HIGHER_IS_BETTER = {
    "aqi",
    "aqi_real_ref",
    "code_recovery_accuracy",
    "exact_conditional_log_likelihood_per_site",
    "exact_log_likelihood_per_site",
    "iid_uniform_log_likelihood_per_site",
}


def _spec_by_key(key: str) -> AblationSpec:
    return next(spec for spec in ABLATIONS if spec.key == key)


def _selected_specs(only: Optional[list[str]]) -> list[AblationSpec]:
    if not only:
        return list(ABLATIONS)
    known = {spec.key for spec in ABLATIONS}
    unknown = set(only) - known
    if unknown:
        raise ValueError(f"Ablations inconnues: {sorted(unknown)}")
    return [spec for spec in ABLATIONS if spec.key in set(only)]


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    """Flatten scalar and vector diagnostics from metrics.json."""
    output: dict[str, float] = {}
    if _finite_number(value):
        output[prefix] = float(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten_numeric(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            output.update(flatten_numeric(item, child))
    return output


def _hybrid_kwargs(args: argparse.Namespace, output_dir: Path, seed: int) -> dict[str, Any]:
    return {
        "data_path": args.data_path,
        "output_dir": str(output_dir),
        "feature_cols": args.feature_cols,
        "condition_cols": args.condition_cols,
        "time_col": args.time_col,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "max_gap_seconds": args.max_gap_seconds,
        "holdout_test_ratio": args.holdout_test_ratio,
        "bins": args.bins,
        "mps_experts": args.mps_experts,
        "mps_bond_dim": args.mps_bond_dim,
        "batch_size": args.batch_size,
        "mps_pretrain_epochs": args.mps_pretrain_epochs,
        "decoder_pretrain_epochs": args.decoder_pretrain_epochs,
        "adversarial_epochs": args.adversarial_epochs,
        "critic_steps": args.critic_steps,
        "device": args.device,
        "seed": seed,
        "auto_resume": True,
    }


def _run_hybrid(spec: AblationSpec, args: argparse.Namespace, output_dir: Path, seed: int) -> dict[str, Any]:
    kwargs = _hybrid_kwargs(args, output_dir, seed)
    kwargs.update(spec.override_dict())
    return run_hybrid(HybridConfig(**kwargs))


def _run_pc_mps(args: argparse.Namespace, output_dir: Path, seed: int) -> dict[str, Any]:
    cfg = MPSConfig(
        data_path=args.data_path,
        output_dir=str(output_dir),
        feature_cols=list(args.feature_cols),
        condition_cols=list(args.condition_cols),
        time_col=args.time_col,
        seq_len=args.seq_len,
        stride=args.stride,
        max_gap_seconds=args.max_gap_seconds,
        holdout_test_ratio=args.holdout_test_ratio,
        bins=args.bins,
        bond_dim=args.mps_bond_dim,
        epochs=args.pc_mps_epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=seed,
    )
    return run_pc_mps(cfg)


def _normalize_pcwgan_metrics(aggregate: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {"model": "pc_cwgan_gp", "status": "ok"}
    for metric in PREFERRED_METRICS:
        for candidate in (
            f"final_test_{metric}_mean",
            f"final_val_{metric}_mean",
            f"{metric}_mean",
        ):
            if candidate in aggregate:
                normalized[metric] = aggregate[candidate]
                break
    normalized["pcwgan_aggregate_metrics"] = aggregate
    return normalized


def _run_pcwgan(args: argparse.Namespace, output_dir: Path, seed: int) -> dict[str, Any]:
    experiment_name = "pc_cwgan_gp"
    cfg = PCWGANConfig(
        data_path=args.data_path,
        output_dir=str(output_dir),
        experiment_name=experiment_name,
        feature_cols=list(args.feature_cols),
        condition_cols=list(args.condition_cols) or None,
        time_col=args.time_col,
        seq_len=args.seq_len,
        stride=args.stride,
        max_gap_seconds=args.max_gap_seconds,
        holdout_test_ratio=args.holdout_test_ratio,
        epochs=args.adversarial_epochs,
        batch_size=args.batch_size,
        critic_steps=args.critic_steps,
        n_splits=1,
        device=args.device,
        seed=seed,
        auto_resume=True,
        skip_completed_folds=True,
        disable_early_stopping=True,
    )
    run_pcwgan(cfg)
    aggregate_path = output_dir / experiment_name / "aggregate_results.json"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"Missing PC-cWGAN aggregate metrics: {aggregate_path}")
    return _normalize_pcwgan_metrics(json.loads(aggregate_path.read_text(encoding="utf-8-sig")))


def run_one(
    spec: AblationSpec,
    args: argparse.Namespace,
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    output_dir = output_root / "runs" / spec.key / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
        return {
            "ablation": spec.key,
            "seed": seed,
            "status": "cached",
            "wall_seconds": 0.0,
            "output_dir": str(output_dir),
            "metrics": metrics,
        }

    started = time.time()
    record: dict[str, Any] = {
        "ablation": spec.key,
        "seed": seed,
        "output_dir": str(output_dir),
    }
    try:
        if spec.runner == "hybrid":
            metrics = _run_hybrid(spec, args, output_dir, seed)
        elif spec.runner == "pc_mps":
            metrics = _run_pc_mps(args, output_dir, seed)
        elif spec.runner == "pcwgan":
            metrics = _run_pcwgan(args, output_dir, seed)
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        else:
            raise ValueError(f"Unknown runner: {spec.runner}")
        record["status"] = "ok"
        record["metrics"] = metrics
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        record["metrics"] = {}
    record["wall_seconds"] = round(time.time() - started, 3)
    return record


def mean_ci95(values: list[float]) -> tuple[float, float, float, int]:
    clean = [float(value) for value in values if _finite_number(value)]
    n = len(clean)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(clean))
    if n == 1:
        return mean, float("nan"), 0.0, 1
    std = float(np.std(clean, ddof=1))
    t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
               6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    critical = t_table.get(n - 1, 1.96)
    return mean, critical * std / math.sqrt(n), std, n


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for spec in ABLATIONS:
        rows = [
            row for row in records
            if row["ablation"] == spec.key and row["status"] in {"ok", "cached"}
        ]
        if not rows:
            continue
        flattened = [flatten_numeric(row.get("metrics", {})) for row in rows]
        keys = sorted(set().union(*(item.keys() for item in flattened)))
        summary: dict[str, Any] = {
            "ablation": spec.key,
            "n_seeds_ok": len(rows),
            "wall_seconds_mean": float(np.mean([row["wall_seconds"] for row in rows])),
        }
        for key in keys:
            mean, half, std, n = mean_ci95([item.get(key) for item in flattened])
            if n:
                summary[f"{key}_mean"] = mean
                summary[f"{key}_ci95"] = half
                summary[f"{key}_std"] = std
                summary[f"{key}_n"] = n
        output.append(summary)
    return output


def paired_deltas(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full = {
        row["seed"]: flatten_numeric(row.get("metrics", {}))
        for row in records
        if row["ablation"] == "full" and row["status"] in {"ok", "cached"}
    }
    output = []
    for row in records:
        if row["ablation"] == "full" or row["status"] not in {"ok", "cached"}:
            continue
        reference = full.get(row["seed"])
        if reference is None:
            continue
        metrics = flatten_numeric(row.get("metrics", {}))
        for key in sorted(set(reference) & set(metrics)):
            delta = metrics[key] - reference[key]
            improvement = delta if key in HIGHER_IS_BETTER else -delta
            output.append({
                "ablation": row["ablation"],
                "seed": row["seed"],
                "metric": key,
                "ablation_value": metrics[key],
                "full_value": reference[key],
                "delta_ablation_minus_full": delta,
                "oriented_improvement": improvement,
            })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if columns:
            writer.writeheader()
            writer.writerows(rows)


def write_markdown(path: Path, summary: list[dict[str, Any]]) -> None:
    present = [
        metric for metric in PREFERRED_METRICS
        if any(f"{metric}_mean" in row for row in summary)
    ]
    columns = ["Ablation", "n"] + present
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in summary:
        cells = [str(row["ablation"]), str(row["n_seeds_ok"])]
        for metric in present:
            mean = row.get(f"{metric}_mean")
            half = row.get(f"{metric}_ci95")
            if mean is None:
                cells.append("-")
            elif not _finite_number(half):
                cells.append(f"{mean:.5g}")
            else:
                cells.append(f"{mean:.5g} +/- {half:.2g}")
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PhyCoMPS-RWGAN ablation harness")
    parser.add_argument("--data_path")
    parser.add_argument("--output_root")
    parser.add_argument("--feature_cols", default="ph,ec,water_temp,air_temp,rh")
    parser.add_argument("--condition_cols", default="")
    parser.add_argument("--time_col", default="")
    parser.add_argument("--seq_len", type=int, default=120)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--max_gap_seconds", type=float, default=60.0)
    parser.add_argument("--holdout_test_ratio", type=float, default=0.15)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--mps_experts", type=int, default=3)
    parser.add_argument("--mps_bond_dim", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mps_pretrain_epochs", type=int, default=20)
    parser.add_argument("--decoder_pretrain_epochs", type=int, default=10)
    parser.add_argument("--adversarial_epochs", type=int, default=100)
    parser.add_argument("--pc_mps_epochs", type=int, default=100)
    parser.add_argument("--critic_steps", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.feature_cols = [item.strip() for item in args.feature_cols.split(",") if item.strip()]
    args.condition_cols = [item.strip() for item in args.condition_cols.split(",") if item.strip()]
    args.time_col = args.time_col or None
    return args


def main() -> None:
    args = parse_args()
    specs = _selected_specs(args.only)
    if args.list:
        for spec in ABLATIONS:
            print(f"{spec.key:24s} [{spec.runner:7s}] {spec.question}")
        return
    if not args.data_path or not args.output_root:
        raise SystemExit("--data_path and --output_root are required")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    plan = [
        {
            "key": spec.key,
            "runner": spec.runner,
            "question": spec.question,
            "overrides": spec.override_dict(),
            "seeds": args.seeds,
        }
        for spec in specs
    ]
    (output_root / "ablation_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )
    for item in plan:
        print(f"[READY] {item['key']:24s} runner={item['runner']} seeds={item['seeds']}")
    if args.dry_run:
        print("Dry-run complete; no training was started.")
        return

    records = []
    for spec in specs:
        for seed in args.seeds:
            print(f"\n>>> {spec.key} seed={seed}")
            record = run_one(spec, args, seed, output_root)
            records.append(record)
            print(f"status={record['status']} wall={record['wall_seconds']}s")
            if record["status"] == "error":
                print(record["error"])

    raw_rows = []
    for record in records:
        raw_rows.append({
            "ablation": record["ablation"],
            "seed": record["seed"],
            "status": record["status"],
            "wall_seconds": record["wall_seconds"],
            **{f"metric.{key}": value for key, value in flatten_numeric(record.get("metrics", {})).items()},
        })
    summary = aggregate(records)
    deltas = paired_deltas(records)
    write_csv(output_root / "ablation_results_raw.csv", raw_rows)
    write_csv(output_root / "ablation_summary.csv", summary)
    write_csv(output_root / "paired_deltas_vs_full.csv", deltas)
    write_markdown(output_root / "ablation_summary.md", summary)
    (output_root / "ablation_report.json").write_text(
        json.dumps({"plan": plan, "records": records, "summary": summary, "paired_deltas": deltas}, indent=2),
        encoding="utf-8",
    )
    print(f"\nAblation outputs saved to {output_root.resolve()}")


if __name__ == "__main__":
    main()
