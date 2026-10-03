"""Post-hoc OPI sensitivity. Never tune historical weights to favor a model."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from validation_common import FROZEN, FEATURES, historical_split, metrics, opi_components, weight_grid, write_json


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(FROZEN / "data/hydro_dataset_pcwgan_gap_clean.csv")
    train_idx, _, _ = historical_split(frame)
    train = frame[FEATURES].to_numpy(np.float32)[train_idx]
    q95 = np.maximum(np.quantile(abs(np.diff(train, axis=1)), .95, axis=(0, 1)), 1e-6)
    variants = ["non_residual_decoder", "full", "prior_iid_uniform"]
    rows = []
    for variant in variants:
        for seed in range(42, 47):
            array = np.load(FROZEN / f"data/{variant}/seed_{seed}/fake_unit.npy", allow_pickle=False)
            for mode, thresholds in (("historical_fixed", None), ("training_q95", dict(zip(FEATURES, q95)))):
                for hd_limit in (5., 10., 15., 20., 30.):
                    parts, deficit = opi_components(array, hd_limit, thresholds)
                    score = float(parts @ np.array([.5, .3, .2]))
                    if mode == "historical_fixed" and hd_limit == 30:
                        expected = metrics.operational_plausibility_index(array, FEATURES)
                        if not np.isclose(score, expected, rtol=0, atol=1e-12):
                            raise AssertionError("Component decomposition differs from frozen canonical OPI")
                    rows.append(dict(variant=variant, seed=seed, thresholds=mode, hd_limit=hd_limit,
                                     range_compliance=parts[0], thermo_compliance=parts[1],
                                     smoothness=parts[2], opi_original_weights=score,
                                     observed_hd_min=float(deficit.min()), observed_hd_max=float(deficit.max())))
    table = pd.DataFrame(rows)
    table.to_csv(out / "opi_components_by_seed.csv", index=False)
    summary = table.groupby(["variant", "thresholds", "hd_limit"], sort=False).mean(numeric_only=True).drop(columns="seed").reset_index()
    summary.to_csv(out / "opi_sensitivity_means.csv", index=False)
    grid = weight_grid()
    records = []
    for mode in ("historical_fixed", "training_q95"):
        for hd_limit in (5., 10., 15., 20., 30.):
            selected = summary[(summary.thresholds == mode) & (summary.hd_limit == hd_limit)].set_index("variant")
            parts = selected.loc[variants, ["range_compliance", "thermo_compliance", "smoothness"]].to_numpy()
            scores = parts @ grid.T
            for k, weights in enumerate(grid):
                records.append(dict(thresholds=mode, hd_limit=hd_limit,
                                    w_range=weights[0], w_thermo=weights[1], w_smooth=weights[2],
                                    direct=scores[0, k], full=scores[1, k], iid=scores[2, k],
                                    direct_minus_full=scores[0, k]-scores[1, k],
                                    direct_minus_iid=scores[0, k]-scores[2, k]))
    grid_table = pd.DataFrame(records)
    grid_table.to_csv(out / "opi_weight_grid.csv", index=False)
    outcomes = []
    for (mode, hd_limit), group in grid_table.groupby(["thresholds", "hd_limit"]):
        for comparator in ("full", "iid"):
            delta = group["direct_minus_" + comparator].to_numpy()
            outcomes.append(dict(thresholds=mode, hd_limit=float(hd_limit), comparator=comparator,
                                 grid_points=len(group), direct_higher=int((delta > 1e-12).sum()),
                                 tied=int((abs(delta) <= 1e-12).sum()), direct_lower=int((delta < -1e-12).sum()),
                                 minimum_difference=float(delta.min()), maximum_difference=float(delta.max())))
    report = {
        "kind": "new_posthoc_sensitivity_not_agronomic_validation",
        "variants": variants, "seeds": list(range(42, 47)), "weight_grid_points": len(grid),
        "hd_limits_g_m3": [5, 10, 15, 20, 30],
        "training_q95": dict(zip(FEATURES, q95.tolist())),
        "canonical_opi_decomposition_verified_all_15_runs": True,
        "ranking_outcomes": outcomes,
        "limitations": [
            "All grids are diagnostic choices, not validated crop-specific thresholds or optimized weights.",
            "No significance claim: outcomes compare means of five stored seeds.",
            "Grid frequencies are not probabilities or independent experiments.",
            "Direct versus residual IID is not a prior-only ablation.",
            "Results remain conditional on the historical preprocessing and operating envelope.",
        ],
    }
    write_json(out / "opi_sensitivity.json", report)
    print(summary[summary.hd_limit == 30].to_string(index=False))
    print(__import__("json").dumps(outcomes[:4], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.out)
