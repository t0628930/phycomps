#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_hydroponic_dataset_v4_gap_clean.py

Convertit le dataset réel "Hydroponic IoT Sensor & Actuator Logs" au format
attendu par le PC-WGAN-GP hydroponique, avec une correction importante pour la
continuité temporelle.

Améliorations V4 :
- conservation du pas brut de 10 s, sans resampling ;
- coercition numérique + interpolation bornée des NaN capteurs ;
- filtrage des aberrations capteurs ;
- conversion TDS -> EC ;
- conversion des actionneurs ON/OFF en conditions binaires ;
- ajout d'un `segment_id` basé sur les interruptions temporelles ;
- le fichier généré permet ensuite au modèle d'exclure automatiquement les
  fenêtres qui traversent un gap temporel.

Usage recommandé :
    python prepare_hydroponic_dataset_v4_gap_clean.py ^
      --in "IoTData --Raw--.csv" ^
      --out hydro_dataset_pcwgan_gap_clean.csv ^
      --max_gap_seconds 60
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# Plages larges de plausibilité physique : elles retirent uniquement les
# aberrations manifestes sans imposer artificiellement les bornes du modèle.
PLAUSIBLE = {
    "pH": (4.0, 9.0),
    "TDS": (1.0, 5000.0),          # ppm, > 0
    "DHT_temp": (10.0, 45.0),
    "DHT_humidity": (20.0, 100.0),
    "water_temp": (5.0, 35.0),
}


FEATURE_COLS = ["ph", "ec", "water_temp", "air_temp", "rh"]
CONDITION_COLS = [
    "condition_ph_reducer",
    "condition_add_water",
    "condition_nutrients_adder",
    "condition_humidifier",
    "condition_ex_fan",
]


def clean(df: pd.DataFrame, max_gap_seconds: float = 60.0) -> pd.DataFrame:
    """Sort, filter sensor outliers, and attach temporal segment identifiers."""
    df = df.copy()
    if "timestamp" not in df.columns:
        raise ValueError("La colonne `timestamp` est obligatoire pour corriger les gaps temporels.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Identify acquisition gaps before any interpolation so missing values are
    # never filled using observations from the other side of a long outage.
    raw_dt = df["timestamp"].diff().dt.total_seconds()
    raw_median_dt = float(raw_dt.dropna().median()) if raw_dt.notna().any() else 10.0
    raw_dt = raw_dt.fillna(raw_median_dt)
    df["_raw_segment_id"] = (raw_dt > float(max_gap_seconds)).cumsum().astype(int)

    sensor_cols = ["pH", "TDS", "DHT_temp", "DHT_humidity", "water_temp"]
    missing_sensors = [c for c in sensor_cols if c not in df.columns]
    if missing_sensors:
        raise ValueError(f"Colonnes capteurs absentes du CSV brut : {missing_sensors}")

    # Robust numeric conversion + limited interpolation for isolated holes.
    # The raw file is kept intact; only the prepared modelling CSV is repaired.
    for col in sensor_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n_nan_before = int(df[sensor_cols].isna().sum().sum())
    df[sensor_cols] = df.groupby("_raw_segment_id", sort=False)[sensor_cols].transform(
        lambda series: series.interpolate(
            method="linear", limit=3, limit_direction="both"
        ).ffill(limit=3).bfill(limit=3)
    )
    n_nan_after = int(df[sensor_cols].isna().sum().sum())

    mask = (
        df["pH"].between(*PLAUSIBLE["pH"])
        & df["TDS"].between(*PLAUSIBLE["TDS"])
        & df["DHT_temp"].between(*PLAUSIBLE["DHT_temp"])
        & df["DHT_humidity"].between(*PLAUSIBLE["DHT_humidity"])
        & df["water_temp"].between(*PLAUSIBLE["water_temp"])
    )
    mask = mask & df[sensor_cols].notna().all(axis=1)
    removed = int((~mask).sum())
    df = df[mask].reset_index(drop=True)

    # Après le filtrage, un point supprimé peut créer un gap apparent. On calcule
    # donc les segments sur le flux effectivement utilisé par le modèle.
    dt = df["timestamp"].diff().dt.total_seconds()
    median_dt = float(dt.dropna().median()) if dt.notna().any() else 10.0
    dt = dt.fillna(median_dt)
    df["gap_seconds"] = dt.astype(float)
    df["segment_id"] = (df["gap_seconds"] > float(max_gap_seconds)).cumsum().astype(int)
    df = df.drop(columns=["_raw_segment_id"], errors="ignore")

    n_segments = int(df["segment_id"].nunique()) if len(df) else 0
    n_large_gaps = int((df["gap_seconds"] > float(max_gap_seconds)).sum())
    print(f"[clean] NaN capteurs avant/après interpolation limitée : {n_nan_before}/{n_nan_after}")
    print(f"[clean] lignes retirées (aberrations ou NaN persistants) : {removed} | conservées : {len(df)}")
    print(
        f"[gaps] max_gap_seconds={max_gap_seconds:g} | "
        f"pas médian={median_dt:.2f}s | gaps détectés={n_large_gaps} | segments={n_segments}"
    )
    return df


def to_pipeline_format(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["timestamp"] = df["timestamp"]
    out["segment_id"] = df["segment_id"].astype(int)
    out["gap_seconds"] = df["gap_seconds"].astype(float)

    # Features en unités réelles.
    out["ph"] = df["pH"].astype(float)
    out["ec"] = df["TDS"].astype(float) / 500.0   # approximation TDS ppm -> EC dS/m
    out["water_temp"] = df["water_temp"].astype(float)
    out["air_temp"] = df["DHT_temp"].astype(float)
    out["rh"] = df["DHT_humidity"].astype(float)

    # Conditions : actionneurs ON/OFF -> 1/0 ; NaN -> OFF par défaut.
    actuators = {
        "pH_reducer": "condition_ph_reducer",
        "add_water": "condition_add_water",
        "nutrients_adder": "condition_nutrients_adder",
        "humidifier": "condition_humidifier",
        "ex_fan": "condition_ex_fan",
    }
    missing = [src for src in actuators if src not in df.columns]
    if missing:
        raise ValueError(f"Colonnes actionneurs absentes du CSV brut : {missing}")

    for src, dst in actuators.items():
        out[dst] = (df[src].astype(str).str.upper() == "ON").astype(float)

    return out


def report_bounds(out: pd.DataFrame) -> None:
    print("\n[bounds] valeurs recommandées pour le Config (p1 / p99) :")
    mapping = {
        "ph": ("ph_min", "ph_max"),
        "ec": ("ec_min", "ec_max"),
        "water_temp": ("water_temp_min", "water_temp_max"),
        "air_temp": ("temp_min", "temp_max"),
        "rh": ("rh_min", "rh_max"),
    }
    for col, (lo, hi) in mapping.items():
        p1, p99 = out[col].quantile(0.01), out[col].quantile(0.99)
        print(f"  {lo:16s}= {p1:6.2f}   {hi:16s}= {p99:6.2f}")
    print("\n  -> Conserver le CSV brut intact et entraîner sur le CSV gap-clean préparé.")


def report_segments(out: pd.DataFrame) -> None:
    if "segment_id" not in out.columns:
        return
    sizes = out.groupby("segment_id").size().sort_values(ascending=False)
    print("\n[segments] plus grands segments continus :")
    for seg, size in sizes.head(10).items():
        print(f"  segment {int(seg):4d} : {int(size):6d} points")


def main() -> None:
    ap = argparse.ArgumentParser(description="Préparation gap-clean du dataset hydroponique réel.")
    root = Path(__file__).resolve().parent
    ap.add_argument(
        "--in", dest="inp", default=str(root / "data" / "IoTData --Raw--.csv")
    )
    ap.add_argument(
        "--out",
        dest="out",
        default=str(root / "data" / "hydro_dataset_pcwgan_gap_clean.csv"),
    )
    ap.add_argument("--max_gap_seconds", type=float, default=60.0)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    print(f"[load] {len(df)} lignes, colonnes : {list(df.columns)}")
    df = clean(df, max_gap_seconds=args.max_gap_seconds)
    out = to_pipeline_format(df)
    Path(args.out).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[save] {args.out} : {out.shape[0]} lignes, {out.shape[1]} colonnes")
    print(f"        features : {FEATURE_COLS}")
    print(f"        conditions temporelles : {CONDITION_COLS}")
    report_segments(out)
    report_bounds(out)


if __name__ == "__main__":
    main()
