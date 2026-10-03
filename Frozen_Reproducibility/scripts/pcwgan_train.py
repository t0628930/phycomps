#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pcwgan_train.py

Script canonique consolidé. Cette version améliore la compatibilité avec le dataset hydroponique réel et corrige la qualité temporelle générative.
Elle intègre les modifications suivantes :

- Les bornes agronomiques (`ph_min`, `ec_min`, etc.) sont recalées sur les
  quantiles p1/p99 observés dans le dataset réel Hydroponic IoT
  Sensor & Actuator Logs (voir `prepare_hydroponic_dataset.py`).
- Les variables CO₂ et lumière sont absentes du dataset, donc les poids
  associés (`w_co2_balance`, `w_light_thermo`) sont mis à zéro par défaut.
- Les longueurs de fenêtre et strides par défaut sont adaptées au pas de
  10 secondes (recommandé : fenêtre de 240 pas = 40 minutes).
- La validation croisée se fait par défaut avec trois plis (`n_splits=3`)
  et un test hold‑out après la zone train+val (`holdout_test_ratio=0.15`).
- De nouveaux arguments CLI permettent d’override les bornes agronomiques
  sans modifier le code (par exemple `--ph_min 5.3 --ph_max 7.0`).

Cette version attend la version `agronomic_physics_v4_fixed_smooth.py` ou un module `agronomic_physics.py` équivalent, avec smoothness fixe : les termes CO₂ et lumière produisent automatiquement zéro
si les variables correspondantes sont absentes, et les poids nuls les
désactivent complètement dans la loss.

Avertissement : cette version suppose que le fichier de données est déjà
préparé via `prepare_hydroponic_dataset.py` et contient les colonnes :
`timestamp`, `ph`, `ec`, `water_temp`, `air_temp`, `rh`, suivies des
`condition_*`.

Auteur : Tarik Idrissi — version adaptée pour dataset hydroponique réel.

"""

from __future__ import annotations

import os
import json
import math
import time
import random
import argparse
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

try:
    # Drop-in physics constraints and Agronomic Quality Index.
    from agronomic_physics import AgronomicConstraintLoss, agronomic_quality_index
except ImportError as exc:
    raise ImportError(
        "Missing agronomic_physics.py. Place it in the same directory as this script, "
        "or install it as a module."
    ) from exc

warnings.filterwarnings("ignore")

# MKLDNN can make second-order gradients slow for gradient penalty
try:
    torch.backends.mkldnn.enabled = False
except Exception:
    pass

# Keep CuDNN globally enabled for speed. It is disabled locally only inside
# gradient_penalty(...), where WGAN-GP needs double backward through the RNN critic.
try:
    torch.backends.cudnn.enabled = True
except Exception:
    pass


###############################################################################
# 1. CONFIGURATION
###############################################################################


@dataclass
class Config:
    # -----------------------------
    # Paths
    # -----------------------------
    data_path: str = ""
    output_dir: str = "outputs_pc_wgan_gp"
    experiment_name: str = "vertical_farming_pc_wgan_gp"

    # -----------------------------
    # Data
    # -----------------------------
    time_col: Optional[str] = None
    condition_cols: Optional[List[str]] = None
    feature_cols: Optional[List[str]] = None
    seq_len: int = 120         # essai prioritaire : 20 minutes au pas 10 s
    stride: int = 15           # chevauchement de 15 pas (~2.5 minutes)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    max_gap_seconds: float = 60.0  # fenêtres interdites à travers les interruptions temporelles

    # -----------------------------
    # Model
    # -----------------------------
    latent_dim: int = 64
    condition_dim: int = 5
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.15
    rnn_type: str = "lstm"         # "lstm" ou "gru"
    use_skip: bool = True
    # Injecting time-varying noise avoids repeating the same latent vector at every time step.
    use_step_noise: bool = True
    step_noise_dim: int = 32

    # -----------------------------
    # Training
    # -----------------------------
    epochs: int = 300
    batch_size: int = 64
    critic_steps: int = 3
    lr_g: float = 2e-4
    lr_c: float = 5e-5
    beta1: float = 0.0
    beta2: float = 0.9
    weight_decay: float = 0.0
    gradient_penalty_lambda: float = 10.0
    grad_clip_norm: float = 1.0      # <=0 disables clipping; stabilizes RNN GAN training

    # -----------------------------
    # Loss weights
    # -----------------------------
    lambda_physics: float = 1.0
    lambda_temporal: float = 2.0
    lambda_acf: float = 1.5
    lambda_smooth: float = 0.03
    lambda_soft_dtw: float = 0.0  # mettre 0.01 pour les essais finaux avec Soft-DTW
    lambda_var: float = 1.0       # évite le collapse de variance
    lambda_delta: float = 1.0     # force les variations locales réalistes

    # -----------------------------
    # Scheduler and early stopping
    # -----------------------------
    scheduler_patience: int = 20
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 80          # was 50 — increased for WGAN-GP convergence dynamics
    min_delta: float = 5e-4                    # was 1e-4 — composite_score oscillates ~0.05; tighten kills runs
    warmup_epochs: int = 30                    # no early stopping before this epoch

    # -----------------------------
    # Cross-validation
    # -----------------------------
    n_splits: int = 3           # par défaut, validation croisée 3 plis
    holdout_test_ratio: float = 0.15
    max_gap_seconds: float = 60.0  # fenêtres interdites à travers les interruptions temporelles

    # -----------------------------
    # Device / reproducibility
    # -----------------------------
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0              # 0 is safest on Colab; override with --num_workers for CPU/GPU pipelines

    # -----------------------------
    # Normalization
    # -----------------------------
    normalize: bool = True
    normalization_method: str = "zscore"  # "zscore" ou "minmax"

    # -----------------------------
    # Temporal-FID settings (Statistical-FID for time series)
    # -----------------------------
    fid_max_samples: int = 2048
    fid_feature_lags: int = 12
    fid_cov_epsilon: float = 1e-6

    # -----------------------------
    # Physics proxy thresholds and weights (hydroponic real dataset)
    # -----------------------------
    hd_max: float = 30.0
    co2_max_rate: float = 120.0
    ec_max_curvature: float = 0.3
    light_scale: float = 1000.0
    # Les contraintes CO2 et lumière sont désactivées (poids=0)
    w_humidity_deficit: float = 1.0
    w_co2_balance: float = 0.0
    w_light_thermo: float = 0.0
    w_irrigation: float = 0.5
    w_range: float = 1.0
    w_smooth_phys: float = 0.2

    # -----------------------------
    # Data-derived physics thresholds
    # -----------------------------
    adaptive_physics_thresholds: bool = True
    smooth_quantile: float = 0.95
    curvature_quantile: float = 0.99
    smooth_threshold_ph: float = 0.15
    smooth_threshold_ec: float = 0.20
    smooth_threshold_water_temp: float = 3.0
    smooth_threshold_air_temp: float = 1.0
    smooth_threshold_rh: float = 4.0

    # -----------------------------
    # Agronomic ranges in real units (calibrées sur dataset hydroponique)
    # -----------------------------
    temp_min: float = 20.0
    temp_max: float = 26.2
    rh_min: float = 52.80
    rh_max: float = 82.80
    co2_min: float = 350.0
    co2_max: float = 1200.0
    ph_min: float = 5.38
    ph_max: float = 6.98
    ec_min: float = 1.11
    ec_max: float = 3.21
    water_temp_min: float = 18.07
    water_temp_max: float = 24.93

    # -----------------------------
    # Logging
    # -----------------------------
    save_every: int = 25
    plot_every: int = 25
    max_plot_samples: int = 4

    # -----------------------------
    # Robust Colab resume / checkpointing
    # -----------------------------
    auto_resume: bool = True          # reprendre automatiquement si resume_latest.pt existe
    resume_checkpoint: str = "resume_latest.pt"
    save_last_every: int = 5          # sauvegarde de reprise toutes les N epochs
    skip_completed_folds: bool = True # évite de recalculer un fold terminé
    disable_early_stopping: bool = False



###############################################################################
# 2. UTILS
###############################################################################


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def validate_config(cfg: Config) -> None:
    positive_ints = ("seq_len", "stride", "epochs", "batch_size", "critic_steps", "n_splits")
    invalid = [name for name in positive_ints if int(getattr(cfg, name)) <= 0]
    if invalid:
        raise ValueError(f"Configuration values must be positive: {invalid}")
    if not 0.0 <= float(cfg.holdout_test_ratio) < 1.0:
        raise ValueError("holdout_test_ratio must be in [0, 1)")
    if cfg.save_every <= 0 or cfg.save_last_every <= 0:
        raise ValueError("Checkpoint intervals must be positive")


def get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _feature_index(feature_names: List[str], *tokens: str) -> Optional[int]:
    lower = [str(n).lower() for n in feature_names]
    for idx, name in enumerate(lower):
        if any(tok in name for tok in tokens):
            return idx
    return None


def update_physics_thresholds_from_real_data(cfg: Config, x_real: np.ndarray, feature_names: List[str]) -> None:
    """Calibrate single-step smoothness and EC-curvature thresholds from real training data.

    The thresholds are computed on REAL-UNIT training windows before normalization:
    - smooth thresholds: q-th percentile of |x[t+1]-x[t]| per feature.
    - EC curvature: q-th percentile of |EC[t+2]-2EC[t+1]+EC[t]|.

    This replaces undocumented fixed constants by data-derived values while keeping
    CLI overrides available through Config fields.
    """
    if not getattr(cfg, "adaptive_physics_thresholds", True):
        return
    if x_real is None or len(x_real) == 0 or x_real.shape[1] < 2:
        return

    mapping = {
        "ph": ("ph",),
        "ec": ("ec", "tds"),
        "water_temp": ("water_temp", "water_temperature", "temp_water"),
        "air_temp": ("air_temp", "temp_air", "t_air"),
        "rh": ("rh", "humidity", "humid"),
    }
    q = float(getattr(cfg, "smooth_quantile", 0.95))
    q = min(max(q, 0.50), 0.999)
    calibrated = {}
    for key, tokens in mapping.items():
        idx = _feature_index(feature_names, *tokens)
        if idx is None:
            continue
        jumps = np.abs(np.diff(x_real[:, :, idx], axis=1)).reshape(-1)
        jumps = jumps[np.isfinite(jumps)]
        if jumps.size == 0:
            continue
        val = float(np.quantile(jumps, q))
        val = max(val, 1e-6)
        setattr(cfg, f"smooth_threshold_{key}", val)
        calibrated[f"smooth_threshold_{key}"] = val

    ec_idx = _feature_index(feature_names, "ec", "tds")
    if ec_idx is not None and x_real.shape[1] >= 3:
        curv = np.abs(x_real[:, 2:, ec_idx] - 2.0 * x_real[:, 1:-1, ec_idx] + x_real[:, :-2, ec_idx]).reshape(-1)
        curv = curv[np.isfinite(curv)]
        if curv.size > 0:
            cq = float(getattr(cfg, "curvature_quantile", 0.99))
            cq = min(max(cq, 0.50), 0.999)
            cfg.ec_max_curvature = max(float(np.quantile(curv, cq)), 1e-6)
            calibrated["ec_max_curvature"] = cfg.ec_max_curvature

    if calibrated:
        msg = " | ".join(f"{k}={v:.6g}" for k, v in calibrated.items())
        print(f"[INFO] Data-derived physics thresholds: {msg}")


def init_pcwgan_weights(module: nn.Module, leaky_relu_slope: float = 0.2) -> None:
    """
    Custom initialization for the temporal PC/WC-WGAN-GP architecture.

    - Linear layers: Xavier uniform
    - LSTM/GRU input-to-hidden weights (weight_ih_*): Xavier uniform, gate by gate
    - LSTM/GRU hidden-to-hidden recurrent weights (weight_hh_*): Orthogonal, gate by gate
    - Biases: zero
    - LSTM forget-gate bias: one
    """
    linear_gain = nn.init.calculate_gain("leaky_relu", leaky_relu_slope)

    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=linear_gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return

    if isinstance(module, (nn.LSTM, nn.GRU)):
        is_lstm = isinstance(module, nn.LSTM)
        gate_count = 4 if is_lstm else 3

        for name, param in module.named_parameters():
            if "weight_ih" in name:
                for gate_param in param.chunk(gate_count, dim=0):
                    nn.init.xavier_uniform_(gate_param, gain=linear_gain)
            elif "weight_hh" in name:
                for gate_param in param.chunk(gate_count, dim=0):
                    nn.init.orthogonal_(gate_param)
            elif "bias" in name:
                nn.init.zeros_(param)
                if is_lstm:
                    hidden_size = param.shape[0] // 4
                    # PyTorch LSTM gate order: input | forget | cell | output
                    param.data[hidden_size:2 * hidden_size].fill_(1.0)


###############################################################################
# 3. SCALER
###############################################################################


class TimeSeriesScaler:
    def __init__(self, method: str = "zscore"):
        self.method = method
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

    def fit(self, x: np.ndarray) -> "TimeSeriesScaler":
        axes = tuple(range(x.ndim - 1))
        if self.method == "zscore":
            self.mean = np.mean(x, axis=axes, keepdims=True)
            self.std = np.std(x, axis=axes, keepdims=True) + 1e-8
        elif self.method == "minmax":
            self.min = np.min(x, axis=axes, keepdims=True)
            self.max = np.max(x, axis=axes, keepdims=True)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.method == "zscore":
            return (x - self.mean) / self.std
        return 2.0 * (x - self.min) / (self.max - self.min + 1e-8) - 1.0

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.method == "zscore":
            return x * self.std + self.mean
        return (x + 1.0) * 0.5 * (self.max - self.min + 1e-8) + self.min

    def state_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
            "min": None if self.min is None else self.min.tolist(),
            "max": None if self.max is None else self.max.tolist(),
        }

    @staticmethod
    def from_state_dict(state: Dict[str, Any]) -> "TimeSeriesScaler":
        s = TimeSeriesScaler(state["method"])
        for k in ["mean", "std", "min", "max"]:
            v = state.get(k, None)
            setattr(s, k, None if v is None else np.array(v, dtype=np.float32))
        return s


###############################################################################
# 4. DATASET
###############################################################################


class FarmingSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, conditions: np.ndarray):
        assert len(sequences) == len(conditions)
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.conditions = torch.tensor(conditions, dtype=torch.float32)

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.conditions[idx]


def generate_demo_dataset(n_steps: int = 5000, seed: int = 42) -> pd.DataFrame:
    # Demo synthetic dataset as before
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    day_phase = (np.sin(2 * np.pi * t / 96.0) + 1) / 2
    irrigation = ((t % 144) < 12).astype(float)
    stress = ((t // 700) % 2).astype(float) * 0.5
    fault = (rng.random(n_steps) < 0.01).astype(float)

    air_temp = 20 + 5 * day_phase + 0.8 * rng.normal(size=n_steps) + stress * 2
    rh = 75 - 15 * day_phase + 2.0 * rng.normal(size=n_steps) - stress * 5
    co2 = 450 + 180 * (1 - day_phase) + 40 * rng.normal(size=n_steps)
    ph = 6.1 + 0.1 * np.sin(2 * np.pi * t / 300) + 0.03 * rng.normal(size=n_steps)
    ec = 1.5 + 0.15 * irrigation - 0.05 * day_phase + 0.03 * rng.normal(size=n_steps)
    water_temp = 19 + 3 * day_phase + 0.4 * rng.normal(size=n_steps)
    light_lux = 15000 * day_phase + 500 * rng.normal(size=n_steps)
    light_lux = np.clip(light_lux, 0, None)
    air_temp[fault == 1] += rng.normal(0, 8, size=int(fault.sum()))
    rh[fault == 1] += rng.normal(0, 20, size=int(fault.sum()))
    return pd.DataFrame({
        "air_temp": air_temp,
        "rh": rh,
        "co2": co2,
        "ph": ph,
        "ec": ec,
        "water_temp": water_temp,
        "light_lux": light_lux,
        "condition_day_phase": day_phase,
        "condition_irrigation": irrigation,
        "condition_stress": stress,
        "condition_fault": fault,
    })


def _attach_segment_id_if_needed(df: pd.DataFrame, time_col: Optional[str], max_gap_seconds: float) -> pd.DataFrame:
    """Return a sorted dataframe with segment_id, without modifying the caller's object."""
    out = df.copy()
    if "segment_id" in out.columns:
        return out
    if time_col and time_col in out.columns:
        out[time_col] = pd.to_datetime(out[time_col])
        out = out.sort_values(time_col).reset_index(drop=True)
        dt = out[time_col].diff().dt.total_seconds()
        median_dt = float(dt.dropna().median()) if dt.notna().any() else 10.0
        dt = dt.fillna(median_dt)
        out["segment_id"] = (dt > float(max_gap_seconds)).cumsum().astype(int)
    else:
        out["segment_id"] = 0
    return out


def make_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    condition_cols: List[str],
    seq_len: int,
    stride: int,
    time_col: Optional[str] = None,
    max_gap_seconds: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create gap-safe windows with temporal actuator conditions.

    Returns
    -------
    sequences: np.ndarray
        Shape [N, T, F].
    conditions: np.ndarray
        Shape [N, T, C], not [N, C]. This preserves the timing of actuator
        events inside each window.
    """
    df = _attach_segment_id_if_needed(df, time_col=time_col, max_gap_seconds=max_gap_seconds)
    sequences: List[np.ndarray] = []
    conditions: List[np.ndarray] = []

    for _, g in df.groupby("segment_id", sort=True):
        if len(g) < seq_len:
            continue
        x = g[feature_cols].values.astype(np.float32)
        c_all = g[condition_cols].values.astype(np.float32)
        for start in range(0, len(g) - seq_len + 1, stride):
            end = start + seq_len
            sequences.append(x[start:end])
            conditions.append(c_all[start:end])

    if not sequences:
        raise ValueError(
            f"No valid windows created. seq_len={seq_len}, stride={stride}. "
            "Check segment lengths or increase --max_gap_seconds."
        )
    return np.stack(sequences), np.stack(conditions)


def load_or_generate_data(cfg: Config) -> Tuple[pd.DataFrame, List[str], List[str]]:
    # If data_path is provided and exists, load it.
    if cfg.data_path and Path(cfg.data_path).exists():
        df = pd.read_csv(cfg.data_path)
        if cfg.time_col and cfg.time_col in df.columns:
            df = df.sort_values(cfg.time_col).reset_index(drop=True)
    else:
        # Otherwise use synthetic demo dataset
        print("[INFO] No dataset found. Using synthetic demo dataset.")
        df = generate_demo_dataset(seed=cfg.seed)

    # Determine conditions and features if not specified
    if cfg.condition_cols is None:
        default_conditions = [c for c in df.columns if c.startswith("condition_")]
        condition_cols = default_conditions if default_conditions else []
    else:
        condition_cols = cfg.condition_cols

    if cfg.feature_cols is None:
        excluded = set(condition_cols)
        if cfg.time_col:
            excluded.add(cfg.time_col)
        feature_cols = [c for c in df.columns if c not in excluded]
    else:
        feature_cols = cfg.feature_cols

    if not condition_cols:
        df["condition_constant"] = 1.0
        condition_cols = ["condition_constant"]

    return df, feature_cols, condition_cols


def temporal_split(x: np.ndarray, c: np.ndarray, train_ratio: float, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    x_train = x[:n_train]
    c_train = c[:n_train]
    x_val = x[n_train:n_train + n_val]
    c_val = c[n_train:n_train + n_val]
    x_test = x[n_train + n_val:]
    c_test = c[n_train + n_val:]
    return x_train, c_train, x_val, c_val, x_test, c_test


def window_purge_count(seq_len: int, stride: int) -> int:
    """Number of adjacent window indices to drop to prevent raw-sample overlap."""
    if seq_len <= 0 or stride <= 0:
        raise ValueError("seq_len and stride must be positive")
    return max(0, math.ceil(seq_len / stride) - 1)


def temporal_cv_splits(n: int, n_splits: int, purge_windows: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window CV with an embargo between train and validation windows."""
    if n < 2:
        return []
    purge_windows = max(0, int(purge_windows))
    if n_splits <= 1:
        train_end = int(0.8 * n)
        val_start = min(n, train_end + purge_windows)
        return [(np.arange(0, train_end), np.arange(val_start, n))] if val_start < n else []
    folds = []
    fold_size = n // (n_splits + 1)
    for k in range(1, n_splits + 1):
        train_end = fold_size * k
        val_end = fold_size * (k + 1)
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(min(val_end, train_end + purge_windows), val_end)
        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx))
    return folds


###############################################################################
# 5. MODEL
###############################################################################


class ConditionalTemporalGenerator(nn.Module):
    def __init__(self, latent_dim: int, condition_dim: int, output_dim: int, seq_len: int, hidden_dim: int, num_layers: int, dropout: float, rnn_type: str = "lstm", use_skip: bool = True, use_step_noise: bool = True, step_noise_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.use_skip = use_skip
        self.use_step_noise = use_step_noise
        self.step_noise_dim = step_noise_dim if use_step_noise else 0
        self.rnn_type = rnn_type.lower()

        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        rnn_input_dim = hidden_dim + condition_dim + self.step_noise_dim
        if self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=rnn_input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        else:
            self.rnn = nn.LSTM(input_size=rnn_input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

        self.skip_proj = nn.Linear(hidden_dim + condition_dim, hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        if cond.dim() == 3:
            cond_seq = cond
            cond_summary = cond_seq.mean(dim=1)
        elif cond.dim() == 2:
            cond_summary = cond
            cond_seq = cond.unsqueeze(1).repeat(1, self.seq_len, 1)
        else:
            raise ValueError(f"Expected condition tensor [B,C] or [B,T,C], got {tuple(cond.shape)}")
        base = torch.cat([z, cond_summary], dim=-1)
        h = self.input_proj(base)
        h_seq = h.unsqueeze(1).repeat(1, self.seq_len, 1)
        pieces = [h_seq, cond_seq]
        if self.use_step_noise and self.step_noise_dim > 0:
            zt = torch.randn(b, self.seq_len, self.step_noise_dim, device=z.device, dtype=z.dtype)
            pieces.append(zt)
        rnn_input = torch.cat(pieces, dim=-1)
        rnn_out, _ = self.rnn(rnn_input)
        if self.use_skip:
            skip = self.skip_proj(torch.cat([h_seq, cond_seq], dim=-1))
            rnn_out = rnn_out + skip
        x = self.out(rnn_out)
        return x


class ConditionalTemporalCritic(nn.Module):
    def __init__(self, input_dim: int, condition_dim: int, seq_len: int, hidden_dim: int, num_layers: int, dropout: float, rnn_type: str = "lstm"):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.seq_len = seq_len
        self.rnn_type = rnn_type.lower()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        if self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
        else:
            self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.dim() == 3:
            cond_seq = cond
        elif cond.dim() == 2:
            cond_seq = cond.unsqueeze(1).repeat(1, x.shape[1], 1)
        else:
            raise ValueError(f"Expected condition tensor [B,C] or [B,T,C], got {tuple(cond.shape)}")
        if cond_seq.shape[1] != x.shape[1]:
            raise ValueError(f"Condition length {cond_seq.shape[1]} does not match sequence length {x.shape[1]}")
        h = torch.cat([x, cond_seq], dim=-1)
        h = self.input_proj(h)
        rnn_out, _ = self.rnn(h)
        pooled = rnn_out.mean(dim=1)
        return self.head(pooled)


###############################################################################
# 6. LOSSES AND CONSTRAINTS
###############################################################################


def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor, cond: torch.Tensor, device: torch.device) -> torch.Tensor:
    """WGAN-GP gradient penalty compatible with CUDA + LSTM/GRU critics.

    PyTorch/Colab uses CuDNN kernels for RNNs on CUDA. CuDNN RNNs do not
    support the double backward required by WGAN-GP, because the gradient
    penalty is differentiated once by autograd.grad(..., create_graph=True)
    and a second time by c_loss.backward().

    We therefore disable CuDNN only for the critic forward used inside the
    gradient-penalty graph. The rest of the training can still use CuDNN.
    """
    b = real.shape[0]
    alpha = torch.rand(b, 1, 1, device=device)
    interpolated = alpha * real + (1.0 - alpha) * fake
    interpolated.requires_grad_(True)

    # Critical fix for Colab/CUDA with LSTM/GRU critic:
    # disable CuDNN only for the forward pass involved in the GP graph.
    with torch.backends.cudnn.flags(enabled=False):
        scores = critic(interpolated, cond)

    gradients = autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.reshape(b, -1)
    gp = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp


def smoothness_loss(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] < 2:
        return torch.tensor(0.0, device=x.device)
    dx = x[:, 1:, :] - x[:, :-1, :]
    return torch.mean(dx ** 2)


def acf_torch(x: torch.Tensor, max_lag: int = 12) -> torch.Tensor:
    x = x - x.mean(dim=1, keepdim=True)
    var = torch.mean(x ** 2, dim=1) + 1e-8
    acfs = []
    t = x.shape[1]
    for lag in range(1, min(max_lag, t - 1) + 1):
        cov = torch.mean(x[:, :-lag, :] * x[:, lag:, :], dim=1)
        acfs.append((cov / var).mean(dim=0))
    if not acfs:
        return torch.zeros(1, x.shape[-1], device=x.device)
    return torch.stack(acfs, dim=0)


def acf_loss(real: torch.Tensor, fake: torch.Tensor, max_lag: int = 12) -> torch.Tensor:
    return torch.mean((acf_torch(real, max_lag) - acf_torch(fake, max_lag)) ** 2)


def temporal_distribution_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    real_mean = real.mean(dim=(0, 1))
    fake_mean = fake.mean(dim=(0, 1))
    real_std = real.std(dim=(0, 1))
    fake_std = fake.std(dim=(0, 1))
    return torch.mean((real_mean - fake_mean) ** 2) + torch.mean((real_std - fake_std) ** 2)


def variance_matching_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Penalise variance collapse in generated sequences."""
    real_std = real.std(dim=(0, 1))
    fake_std = fake.std(dim=(0, 1))
    return torch.mean((fake_std - real_std) ** 2)


def delta_distribution_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Match the distribution of local temporal increments."""
    if real.shape[1] < 2 or fake.shape[1] < 2:
        return real.new_tensor(0.0)
    dr = real[:, 1:, :] - real[:, :-1, :]
    df = fake[:, 1:, :] - fake[:, :-1, :]
    mean_loss = torch.mean((dr.mean(dim=(0, 1)) - df.mean(dim=(0, 1))) ** 2)
    std_loss = torch.mean((dr.std(dim=(0, 1)) - df.std(dim=(0, 1))) ** 2)
    return mean_loss + std_loss


def pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = (x ** 2).sum(dim=1).view(-1, 1)
    y_norm = (y ** 2).sum(dim=1).view(1, -1)
    return x_norm + y_norm - 2.0 * torch.mm(x, y.t())


def mmd_rbf_loss(real: torch.Tensor, fake: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    r = real.reshape(real.shape[0], -1)
    f = fake.reshape(fake.shape[0], -1)
    k_rr = torch.exp(-gamma * pairwise_distances(r, r)).mean()
    k_ff = torch.exp(-gamma * pairwise_distances(f, f)).mean()
    k_rf = torch.exp(-gamma * pairwise_distances(r, f)).mean()
    return k_rr + k_ff - 2 * k_rf


def soft_dtw_approx(x: torch.Tensor, y: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    b, n, _ = x.shape
    m = y.shape[1]
    dist = torch.cdist(x, y, p=2) ** 2
    inf = 1e6
    r = torch.ones((b, n + 1, m + 1), device=x.device) * inf
    r[:, 0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            r0 = -r[:, i - 1, j - 1] / gamma
            r1 = -r[:, i - 1, j] / gamma
            r2 = -r[:, i, j - 1] / gamma
            rmax = torch.max(torch.max(r0, r1), r2)
            rsum = torch.exp(r0 - rmax) + torch.exp(r1 - rmax) + torch.exp(r2 - rmax)
            softmin = -gamma * (torch.log(rsum) + rmax)
            r[:, i, j] = dist[:, i - 1, j - 1] + softmin
    return r[:, n, m].mean()


###############################################################################
# 7. METRICS
###############################################################################


def rmse_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def acf_np(x: np.ndarray, max_lag: int = 12) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    var = np.mean(x ** 2, axis=1) + 1e-8
    acfs = []
    for lag in range(1, min(max_lag, x.shape[1] - 1) + 1):
        cov = np.mean(x[:, :-lag, :] * x[:, lag:, :], axis=1)
        acfs.append(np.mean(cov / var, axis=0))
    return np.stack(acfs, axis=0) if acfs else np.zeros((1, x.shape[-1]))


def acf_error_np(real: np.ndarray, fake: np.ndarray, max_lag: int = 12) -> float:
    return float(np.mean((acf_np(real, max_lag) - acf_np(fake, max_lag)) ** 2))


def std_ratio_error_np(real: np.ndarray, fake: np.ndarray) -> float:
    real_std = real.std(axis=(0, 1)) + 1e-8
    fake_std = fake.std(axis=(0, 1))
    return float(np.mean(np.abs(fake_std / real_std - 1.0)))


def delta_std_ratio_error_np(real: np.ndarray, fake: np.ndarray) -> float:
    if real.shape[1] < 2 or fake.shape[1] < 2:
        return 0.0
    dr = np.diff(real, axis=1)
    df = np.diff(fake, axis=1)
    real_std = dr.std(axis=(0, 1)) + 1e-8
    fake_std = df.std(axis=(0, 1))
    return float(np.mean(np.abs(fake_std / real_std - 1.0)))


def dtw_distance_approx(a: np.ndarray, b: np.ndarray, max_samples: int = 64) -> float:
    n = min(len(a), len(b), max_samples)
    if n == 0:
        return float("nan")
    total = 0.0
    for k in range(n):
        x = a[k]
        y = b[k]
        t1, t2 = x.shape[0], y.shape[0]
        dp = np.ones((t1 + 1, t2 + 1)) * 1e9
        dp[0, 0] = 0
        for i in range(1, t1 + 1):
            for j in range(1, t2 + 1):
                cost = np.linalg.norm(x[i - 1] - y[j - 1])
                dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
        total += dp[t1, t2] / (t1 + t2)
    return float(total / n)


def mmd_rbf_np(a: np.ndarray, b: np.ndarray, gamma: Optional[float] = None, max_samples: int = 512) -> float:
    """Deterministic multi-scale RBF MMD.

    The previous implementation estimated gamma from a random subset, which made
    repeated evaluations non-deterministic. Here the median distance is estimated
    from deterministic evenly-spaced samples and averaged over multiple scales.
    """
    n = min(len(a), len(b), max_samples)
    if n <= 1:
        return float("nan")
    a = a[:n].reshape(n, -1).astype(np.float64)
    b = b[:n].reshape(n, -1).astype(np.float64)
    if gamma is None:
        z = np.concatenate([a, b], axis=0)
        sample_n = min(len(z), 256)
        idx = np.linspace(0, len(z) - 1, sample_n, dtype=int)
        zz = z[idx]
        dists = ((zz[:, None, :] - zz[None, :, :]) ** 2).sum(axis=-1)
        positive = dists[dists > 0]
        med = float(np.median(positive)) if positive.size else 1.0
        base_gamma = 1.0 / (med + 1e-8)
    else:
        base_gamma = float(gamma)
    gammas = [base_gamma * s for s in (0.25, 0.5, 1.0, 2.0, 4.0)]

    def kernel(x, y, g):
        d = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=-1)
        return np.exp(-g * d)

    vals = []
    for g in gammas:
        vals.append(kernel(a, a, g).mean() + kernel(b, b, g).mean() - 2 * kernel(a, b, g).mean())
    return float(np.mean(vals))


def _covariance_np(x: np.ndarray, shrinkage: float = 0.05) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("Expected 2D feature matrix")
    if x.shape[0] <= 1:
        return np.eye(x.shape[1], dtype=np.float64)
    cov = np.cov(x, rowvar=False).astype(np.float64)
    cov = 0.5 * (cov + cov.T)
    if shrinkage > 0.0:
        p = cov.shape[0]
        target = (np.trace(cov) / max(p, 1)) * np.eye(p, dtype=np.float64)
        cov = (1.0 - shrinkage) * cov + shrinkage * target
    return cov


def _matrix_sqrt_psd(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    mat = 0.5 * (mat + mat.T)
    try:
        from scipy.linalg import sqrtm
        out = sqrtm(mat)
        if np.iscomplexobj(out):
            out = out.real
        return np.asarray(out, dtype=np.float64)
    except Exception:
        vals, vecs = np.linalg.eigh(mat)
        vals = np.clip(vals, 0.0, None)
        return (vecs * np.sqrt(vals)) @ vecs.T


def frechet_distance_np(emb_real: np.ndarray, emb_fake: np.ndarray, eps: float = 1e-6) -> float:
    emb_real = np.asarray(emb_real, dtype=np.float64)
    emb_fake = np.asarray(emb_fake, dtype=np.float64)
    if emb_real.ndim != 2 or emb_fake.ndim != 2 or emb_real.shape[1] != emb_fake.shape[1]:
        raise ValueError("Embeddings must be 2D with matching dimensions")
    mu_r = np.mean(emb_real, axis=0)
    mu_f = np.mean(emb_fake, axis=0)
    cov_r = _covariance_np(emb_real) + eps * np.eye(emb_real.shape[1])
    cov_f = _covariance_np(emb_fake) + eps * np.eye(emb_fake.shape[1])
    diff = mu_r - mu_f
    cov_prod_sqrt = _matrix_sqrt_psd(cov_r @ cov_f)
    fid = diff.dot(diff) + np.trace(cov_r + cov_f - 2.0 * cov_prod_sqrt)
    return float(max(fid, 0.0))


def temporal_statistical_embeddings(x: np.ndarray, max_lag: int = 12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("Expected x with shape [N, T, F]")
    n, t, f = x.shape
    feats = []
    feats.extend([x.mean(axis=1), x.std(axis=1), x.min(axis=1), x.max(axis=1), np.median(x, axis=1), np.quantile(x, 0.25, axis=1), np.quantile(x, 0.75, axis=1)])
    time_axis = np.arange(t, dtype=np.float64)
    time_axis = (time_axis - time_axis.mean()) / (time_axis.std() + 1e-8)
    slope = (x * time_axis[None, :, None]).mean(axis=1)
    feats.append(slope)
    if t >= 2:
        dx = np.diff(x, axis=1)
        feats.extend([dx.mean(axis=1), dx.std(axis=1), np.abs(dx).max(axis=1)])
    else:
        feats.extend([np.zeros((n, f)), np.zeros((n, f)), np.zeros((n, f))])
    xc = x - x.mean(axis=1, keepdims=True)
    var = np.mean(xc ** 2, axis=1) + 1e-8
    lag_feats = []
    usable_lags = min(max_lag, max(t - 1, 0))
    for lag in range(1, usable_lags + 1):
        cov = np.mean(xc[:, :-lag, :] * xc[:, lag:, :], axis=1)
        lag_feats.append(cov / var)
    if lag_feats:
        feats.append(np.concatenate(lag_feats, axis=1))
    emb = np.concatenate([v.reshape(n, -1) for v in feats], axis=1)
    emb = np.nan_to_num(emb, nan=0.0, posinf=1e6, neginf=-1e6)
    return emb.astype(np.float64)


def fid_temporal_np(real: np.ndarray, fake: np.ndarray, max_samples: int = 2048, max_lag: int = 12, eps: float = 1e-6) -> float:
    n = min(len(real), len(fake), max_samples)
    if n < 2:
        return float("nan")
    real = real[:n]
    fake = fake[:n]
    emb_real = temporal_statistical_embeddings(real, max_lag=max_lag)
    emb_fake = temporal_statistical_embeddings(fake, max_lag=max_lag)
    return frechet_distance_np(emb_real, emb_fake, eps=eps)


def violation_score_np(x_real_unit: np.ndarray, feature_names: List[str], cfg: Config) -> float:
    ranges = {
        "air_temp": (cfg.temp_min, cfg.temp_max),
        "temperature": (cfg.temp_min, cfg.temp_max),
        "temp": (cfg.temp_min, cfg.temp_max),
        "rh": (cfg.rh_min, cfg.rh_max),
        "humidity": (cfg.rh_min, cfg.rh_max),
        "co2": (cfg.co2_min, cfg.co2_max),
        "ph": (cfg.ph_min, cfg.ph_max),
        "ec": (cfg.ec_min, cfg.ec_max),
        "water_temp": (cfg.water_temp_min, cfg.water_temp_max),
    }
    violations = []
    lower_names = [n.lower() for n in feature_names]
    for idx, name in enumerate(lower_names):
        for key, (low, high) in ranges.items():
            if key in name:
                v = x_real_unit[:, :, idx]
                violations.append(np.mean((v < low) | (v > high)))
                break
    if not violations:
        return 0.0
    return float(np.mean(violations))


def compute_metrics(real_norm: np.ndarray, fake_norm: np.ndarray, feature_names: List[str], cfg: Config, scaler: Optional[TimeSeriesScaler] = None) -> Dict[str, float]:
    """Compute distribution, temporal and agronomic metrics.

    Naming convention:
    - real_norm/fake_norm are the arrays used for statistical metrics.
    - real_unit/fake_unit are explicitly real-unit arrays used for AQI and violations.
      If scaler is None, the input arrays are assumed to already be in real units.
    """
    n = min(len(real_norm), len(fake_norm))
    real_norm = real_norm[:n]
    fake_norm = fake_norm[:n]
    metrics = {
        "rmse_norm_unpaired": rmse_np(real_norm, fake_norm),
        "mae_norm_unpaired": mae_np(real_norm, fake_norm),
        "acf_error": acf_error_np(real_norm, fake_norm),
        "dtw_approx": dtw_distance_approx(real_norm, fake_norm),
        "mmd_rbf": mmd_rbf_np(real_norm, fake_norm),
        "fid_temporal": fid_temporal_np(real_norm, fake_norm, max_samples=cfg.fid_max_samples, max_lag=cfg.fid_feature_lags, eps=cfg.fid_cov_epsilon),
        "std_ratio_error": std_ratio_error_np(real_norm, fake_norm),
        "delta_std_ratio_error": delta_std_ratio_error_np(real_norm, fake_norm),
    }
    metrics["stat_fid"] = metrics["fid_temporal"]  # manuscript alias: Statistical-FID

    if scaler is not None:
        real_unit = scaler.inverse_transform(real_norm)
        fake_unit = scaler.inverse_transform(fake_norm)
    else:
        real_unit = real_norm
        fake_unit = fake_norm

    metrics["rmse_real_unit_unpaired"] = rmse_np(real_unit, fake_unit)
    metrics["mae_real_unit_unpaired"] = mae_np(real_unit, fake_unit)
    metrics["violation_score"] = violation_score_np(fake_unit, feature_names, cfg)
    metrics["aqi"] = agronomic_quality_index(fake_unit, feature_names, cfg)
    metrics["aqi_real_ref"] = agronomic_quality_index(real_unit, feature_names, cfg)

    fid_term = 0.1 * math.log1p(metrics["fid_temporal"]) if np.isfinite(metrics["fid_temporal"]) else 0.0
    metrics["composite_score"] = (
        metrics["mmd_rbf"]
        + 0.5 * metrics["acf_error"]
        + 0.2 * metrics["dtw_approx"]
        + fid_term
        + 1.0 * metrics["std_ratio_error"]
        + 1.0 * metrics["delta_std_ratio_error"]
        + 2.0 * max(0.0, 0.98 - metrics["aqi"])
        + 5.0 * max(0.0, metrics["violation_score"] - 0.02)
    )
    return metrics


###############################################################################
# 8. PLOTTING
###############################################################################


def plot_samples(real: np.ndarray, fake: np.ndarray, feature_names: List[str], out_path: Path, max_samples: int = 4) -> None:
    n_samples = min(max_samples, len(real), len(fake))
    n_features = min(len(feature_names), real.shape[-1])
    for s in range(n_samples):
        plt.figure(figsize=(12, 3 * n_features))
        for f in range(n_features):
            ax = plt.subplot(n_features, 1, f + 1)
            ax.plot(real[s, :, f], label="Real")
            ax.plot(fake[s, :, f], label="Generated", linestyle="--")
            ax.set_title(feature_names[f])
            ax.grid(True, alpha=0.3)
            if f == 0:
                ax.legend()
        plt.tight_layout()
        plt.savefig(out_path / f"sample_{s}.png", dpi=180)
        plt.close()


def plot_training_history(history: List[Dict[str, float]], out_path: Path) -> None:
    if not history:
        return
    df = pd.DataFrame(history)
    csv_path = out_path / "training_history.csv"
    df.to_csv(csv_path, index=False)
    for col in df.columns:
        if col == "epoch":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            plt.figure(figsize=(8, 4))
            plt.plot(df["epoch"], df[col])
            plt.title(col)
            plt.xlabel("Epoch")
            plt.ylabel(col)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_path / f"history_{col}.png", dpi=160)
            plt.close()


###############################################################################
# 9. TRAINER
###############################################################################


class PCWGANTrainer:
    def __init__(self, cfg: Config, feature_names: List[str], condition_names: List[str], scaler: Optional[TimeSeriesScaler], output_dir: Path):
        self.cfg = cfg
        self.feature_names = feature_names
        self.condition_names = condition_names
        self.scaler = scaler
        self.output_dir = ensure_dir(output_dir)
        self.device = get_device(cfg.device)
        self.feature_dim = len(feature_names)
        self.condition_dim = len(condition_names)
        self.generator = ConditionalTemporalGenerator(
            latent_dim=cfg.latent_dim,
            condition_dim=self.condition_dim,
            output_dim=self.feature_dim,
            seq_len=cfg.seq_len,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            rnn_type=cfg.rnn_type,
            use_skip=cfg.use_skip,
            use_step_noise=cfg.use_step_noise,
            step_noise_dim=cfg.step_noise_dim,
        ).to(self.device)
        self.critic = ConditionalTemporalCritic(
            input_dim=self.feature_dim,
            condition_dim=self.condition_dim,
            seq_len=cfg.seq_len,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            rnn_type=cfg.rnn_type,
        ).to(self.device)
        # Controlled initialization for new experiments.
        # If a checkpoint is loaded later, its saved weights overwrite this initialization.
        self.generator.apply(init_pcwgan_weights)
        self.critic.apply(init_pcwgan_weights)
        print("[INFO] Weight initialization: Linear=Xavier uniform | RNN weight_ih=Xavier uniform | RNN weight_hh=Orthogonal")
        self.physics_loss = AgronomicConstraintLoss(
            cfg=cfg,
            feature_names=feature_names,
            scaler=scaler,
            condition_names=condition_names,
        ).to(self.device)
        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_c, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
        self.scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_g, mode="min", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience)
        self.scheduler_c = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_c, mode="min", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience)
        self.best_score = float("inf")
        self.best_epoch = -1
        self.no_improve = 0
        self.history: List[Dict[str, float]] = []

    def sample_noise(self, b: int) -> torch.Tensor:
        return torch.randn(b, self.cfg.latent_dim, device=self.device)

    def generate(self, cond: torch.Tensor) -> torch.Tensor:
        z = self.sample_noise(cond.shape[0])
        return self.generator(z, cond)

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.generator.train()
        self.critic.train()
        c_losses = []
        g_losses = []
        gp_losses = []
        physics_losses = []
        acf_losses = []
        temporal_losses = []
        var_losses = []
        delta_losses = []
        for real, cond in loader:
            real = real.to(self.device)
            cond = cond.to(self.device)
            b = real.shape[0]
            for _ in range(self.cfg.critic_steps):
                z = self.sample_noise(b)
                fake = self.generator(z, cond).detach()
                c_real = self.critic(real, cond).mean()
                c_fake = self.critic(fake, cond).mean()
                gp = gradient_penalty(self.critic, real, fake, cond, self.device)
                c_loss = c_fake - c_real + self.cfg.gradient_penalty_lambda * gp
                self.opt_c.zero_grad(set_to_none=True)
                c_loss.backward()
                if self.cfg.grad_clip_norm and self.cfg.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip_norm)
                self.opt_c.step()
            z = self.sample_noise(b)
            fake = self.generator(z, cond)
            adv_loss = -self.critic(fake, cond).mean()
            phy_loss = self.physics_loss(fake, cond)
            temp_loss = temporal_distribution_loss(real, fake)
            smooth_loss = smoothness_loss(fake)
            acf_l = acf_loss(real, fake)
            var_loss = variance_matching_loss(real, fake)
            delta_loss = delta_distribution_loss(real, fake)
            g_loss = (
                adv_loss
                + self.cfg.lambda_physics * phy_loss
                + self.cfg.lambda_temporal * temp_loss
                + self.cfg.lambda_acf * acf_l
                + self.cfg.lambda_smooth * smooth_loss
                + self.cfg.lambda_var * var_loss
                + self.cfg.lambda_delta * delta_loss
            )
            if self.cfg.lambda_soft_dtw > 0:
                g_loss = g_loss + self.cfg.lambda_soft_dtw * soft_dtw_approx(real, fake)
            self.opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            if self.cfg.grad_clip_norm and self.cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.generator.parameters(), self.cfg.grad_clip_norm)
            self.opt_g.step()
            c_losses.append(float(c_loss.detach().cpu()))
            g_losses.append(float(g_loss.detach().cpu()))
            gp_losses.append(float(gp.detach().cpu()))
            physics_losses.append(float(phy_loss.detach().cpu()))
            acf_losses.append(float(acf_l.detach().cpu()))
            temporal_losses.append(float(temp_loss.detach().cpu()))
            var_losses.append(float(var_loss.detach().cpu()))
            delta_losses.append(float(delta_loss.detach().cpu()))
        return {
            "critic_loss": float(np.mean(c_losses)),
            "generator_loss": float(np.mean(g_losses)),
            "gradient_penalty": float(np.mean(gp_losses)),
            "physics_loss": float(np.mean(physics_losses)),
            "acf_loss": float(np.mean(acf_losses)),
            "temporal_loss": float(np.mean(temporal_losses)),
            "variance_loss": float(np.mean(var_losses)),
            "delta_loss": float(np.mean(delta_losses)),
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, max_batches: Optional[int] = None) -> Dict[str, float]:
        self.generator.eval()
        real_list: List[np.ndarray] = []
        fake_list: List[np.ndarray] = []
        cuda_devices: List[int] = []
        if self.device.type == "cuda":
            cuda_devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        # Validation must compare epochs against the same latent/step noise and
        # must not advance the training RNG state.
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            eval_seed = int(self.cfg.seed) + 100_003
            torch.manual_seed(eval_seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(eval_seed)
            for i, (real, cond) in enumerate(loader):
                real = real.to(self.device)
                cond = cond.to(self.device)
                fake = self.generate(cond)
                real_list.append(to_numpy(real))
                fake_list.append(to_numpy(fake))
                if max_batches is not None and i + 1 >= max_batches:
                    break
        if not real_list:
            raise ValueError("Cannot evaluate an empty DataLoader")
        real_np = np.concatenate(real_list, axis=0)
        fake_np = np.concatenate(fake_list, axis=0)
        return compute_metrics(real_norm=real_np, fake_norm=fake_np, feature_names=self.feature_names, cfg=self.cfg, scaler=self.scaler)

    @torch.no_grad()
    def generate_numpy(self, conditions: np.ndarray, batch_size: int = 256) -> np.ndarray:
        self.generator.eval()
        out = []
        for start in range(0, len(conditions), batch_size):
            cond = torch.tensor(conditions[start:start + batch_size], dtype=torch.float32, device=self.device)
            fake = self.generate(cond)
            out.append(to_numpy(fake))
        return np.concatenate(out, axis=0)

    def _atomic_torch_save(self, ckpt: Dict[str, Any], path: Path) -> None:
        """Save checkpoint atomically to avoid corrupted files on Colab/Drive interruption."""
        path = Path(path)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, path)

    def _checkpoint_payload(self, epoch: Optional[int] = None, metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        return {
            "config": asdict(self.cfg),
            "feature_names": self.feature_names,
            "condition_names": self.condition_names,
            "epoch": int(epoch or 0),
            "generator_state_dict": self.generator.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "opt_g_state_dict": self.opt_g.state_dict(),
            "opt_c_state_dict": self.opt_c.state_dict(),
            "scheduler_g_state_dict": self.scheduler_g.state_dict(),
            "scheduler_c_state_dict": self.scheduler_c.state_dict(),
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "no_improve": self.no_improve,
            "history": self.history,
            "metrics": metrics or {},
            "scaler": None if self.scaler is None else self.scaler.state_dict(),
            "rng_state_torch": torch.get_rng_state(),
            "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "rng_state_numpy": np.random.get_state(),
            "rng_state_python": random.getstate(),
        }

    def save_checkpoint(self, name: str, metrics: Optional[Dict[str, float]] = None, epoch: Optional[int] = None) -> None:
        ckpt = self._checkpoint_payload(epoch=epoch, metrics=metrics)
        self._atomic_torch_save(ckpt, self.output_dir / name)

    def _validate_checkpoint(self, ckpt: Dict[str, Any], path: Path) -> None:
        saved_features = ckpt.get("feature_names")
        saved_conditions = ckpt.get("condition_names")
        if saved_features is not None and list(saved_features) != list(self.feature_names):
            raise ValueError(f"Checkpoint feature order mismatch: {path}")
        if saved_conditions is not None and list(saved_conditions) != list(self.condition_names):
            raise ValueError(f"Checkpoint condition order mismatch: {path}")
        saved_cfg = ckpt.get("config") or {}
        architecture_keys = (
            "seq_len", "latent_dim", "hidden_dim", "num_layers", "rnn_type",
            "use_step_noise", "step_noise_dim",
        )
        mismatches = [
            key for key in architecture_keys
            if key in saved_cfg and saved_cfg[key] != getattr(self.cfg, key)
        ]
        if mismatches:
            details = ", ".join(
                f"{key}: saved={saved_cfg[key]!r}, current={getattr(self.cfg, key)!r}"
                for key in mismatches
            )
            raise ValueError(f"Incompatible checkpoint {path}: {details}")

    @staticmethod
    def _restore_rng_state(ckpt: Dict[str, Any]) -> None:
        if ckpt.get("rng_state_torch") is not None:
            torch.set_rng_state(ckpt["rng_state_torch"].cpu())
        if torch.cuda.is_available() and ckpt.get("rng_state_cuda") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in ckpt["rng_state_cuda"]])
        if ckpt.get("rng_state_numpy") is not None:
            np.random.set_state(ckpt["rng_state_numpy"])
        if ckpt.get("rng_state_python") is not None:
            random.setstate(ckpt["rng_state_python"])

    def load_checkpoint(self, name: str = "best_model.pt", load_optimizers: bool = False) -> bool:
        path = self.output_dir / name
        if not path.exists():
            return False
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._validate_checkpoint(ckpt, path)
        self.generator.load_state_dict(ckpt["generator_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        if load_optimizers:
            if "opt_g_state_dict" in ckpt:
                self.opt_g.load_state_dict(ckpt["opt_g_state_dict"])
            if "opt_c_state_dict" in ckpt:
                self.opt_c.load_state_dict(ckpt["opt_c_state_dict"])
            if "scheduler_g_state_dict" in ckpt:
                self.scheduler_g.load_state_dict(ckpt["scheduler_g_state_dict"])
            if "scheduler_c_state_dict" in ckpt:
                self.scheduler_c.load_state_dict(ckpt["scheduler_c_state_dict"])
        self.best_score = float(ckpt.get("best_score", self.best_score))
        self.best_epoch = int(ckpt.get("best_epoch", self.best_epoch))
        self.no_improve = int(ckpt.get("no_improve", self.no_improve))
        if isinstance(ckpt.get("history", None), list):
            self.history = ckpt["history"]
        return True

    def _infer_epoch_from_existing_files(self, checkpoint_name: str) -> int:
        hist_path = self.output_dir / "training_history.csv"
        if hist_path.exists():
            try:
                hist_df = pd.read_csv(hist_path)
                if "epoch" in hist_df.columns and len(hist_df) > 0:
                    self.history = hist_df.to_dict("records")
                    return int(hist_df["epoch"].max())
            except Exception:
                pass
        import re
        m = re.search(r"checkpoint_epoch_(\d+)\.pt", checkpoint_name)
        if m:
            return int(m.group(1))
        return 0

    def load_resume_checkpoint(self) -> int:
        """Load resume checkpoint and return the next epoch to run."""
        candidates = [self.cfg.resume_checkpoint, "last_model.pt"]
        numbered = sorted(
            self.output_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else -1,
            reverse=True,
        )
        candidates += [p.name for p in numbered]
        for name in candidates:
            path = self.output_dir / name
            if not path.exists():
                continue
            print(f"[INFO] Resume checkpoint found: {path}")
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self._validate_checkpoint(ckpt, path)
            self.generator.load_state_dict(ckpt["generator_state_dict"])
            self.critic.load_state_dict(ckpt["critic_state_dict"])
            if "opt_g_state_dict" in ckpt:
                self.opt_g.load_state_dict(ckpt["opt_g_state_dict"])
            if "opt_c_state_dict" in ckpt:
                self.opt_c.load_state_dict(ckpt["opt_c_state_dict"])
            if "scheduler_g_state_dict" in ckpt:
                self.scheduler_g.load_state_dict(ckpt["scheduler_g_state_dict"])
            if "scheduler_c_state_dict" in ckpt:
                self.scheduler_c.load_state_dict(ckpt["scheduler_c_state_dict"])
            self.best_score = float(ckpt.get("best_score", self.best_score))
            self.best_epoch = int(ckpt.get("best_epoch", self.best_epoch))
            self.no_improve = int(ckpt.get("no_improve", self.no_improve))
            if isinstance(ckpt.get("history", None), list):
                self.history = ckpt["history"]
            self._restore_rng_state(ckpt)
            last_epoch = int(ckpt.get("epoch", 0))
            if last_epoch <= 0:
                last_epoch = self._infer_epoch_from_existing_files(name)
            print(f"[INFO] Resuming from epoch {last_epoch + 1}/{self.cfg.epochs} | Best={self.best_score:.6f}@{self.best_epoch}")
            return last_epoch + 1
        return 1

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        print(f"[INFO] Training on device: {self.device}")
        start_time = time.time()
        start_epoch = 1
        if self.cfg.auto_resume:
            start_epoch = self.load_resume_checkpoint()
            if start_epoch > self.cfg.epochs:
                print(f"[INFO] This fold already reached epoch {self.cfg.epochs}. Nothing to train.")
                plot_training_history(self.history, self.output_dir)
                summary = {
                    "best_score": self.best_score,
                    "best_epoch": self.best_epoch,
                    "elapsed_seconds": 0.0,
                    "final_epoch": self.history[-1]["epoch"] if self.history else self.cfg.epochs,
                    "resumed_already_complete": True,
                }
                save_json(summary, self.output_dir / "training_summary.json")
                return summary
        try:
            for epoch in range(start_epoch, self.cfg.epochs + 1):
                train_stats = self.train_epoch(train_loader)
                val_metrics = self.evaluate(val_loader, max_batches=8)
                score = val_metrics["composite_score"]
                self.scheduler_g.step(score)
                self.scheduler_c.step(score)
                row = {
                    "epoch": epoch,
                    **train_stats,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    "lr_g": self.opt_g.param_groups[0]["lr"],
                    "lr_c": self.opt_c.param_groups[0]["lr"],
                }
                self.history.append(row)
                improved = score < self.best_score - self.cfg.min_delta
                if improved:
                    self.best_score = score
                    self.best_epoch = epoch
                    self.no_improve = 0
                    self.save_checkpoint("best_model.pt", val_metrics, epoch=epoch)
                else:
                    self.no_improve += 1
                if epoch % 5 == 0 or epoch == 1 or epoch == start_epoch:
                    print(
                        f"[Epoch {epoch:04d}] "
                        f"G={train_stats['generator_loss']:.4f} "
                        f"C={train_stats['critic_loss']:.4f} "
                        f"GP={train_stats['gradient_penalty']:.4f} "
                        f"Score={score:.6f} "
                        f"Best={self.best_score:.6f}@{self.best_epoch}"
                    )
                if epoch % self.cfg.save_every == 0:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt", val_metrics, epoch=epoch)
                if epoch % self.cfg.save_last_every == 0 or epoch == 1:
                    self.save_checkpoint(self.cfg.resume_checkpoint, val_metrics, epoch=epoch)
                    plot_training_history(self.history, self.output_dir)
                    save_json({
                        "best_score": self.best_score,
                        "best_epoch": self.best_epoch,
                        "final_epoch": epoch,
                        "resume_checkpoint": self.cfg.resume_checkpoint,
                        "status": "running",
                    }, self.output_dir / "training_summary.json")
                warmup = getattr(self.cfg, "warmup_epochs", 30)
                if (not self.cfg.disable_early_stopping) and epoch > warmup and self.no_improve >= self.cfg.early_stopping_patience:
                    print(f"[INFO] Early stopping at epoch {epoch}. Best epoch: {self.best_epoch}")
                    break
        except KeyboardInterrupt:
            last_epoch = self.history[-1]["epoch"] if self.history else max(0, start_epoch - 1)
            print(f"\n[WARN] KeyboardInterrupt at epoch {last_epoch}. Saving resume checkpoint...")
            self.save_checkpoint(self.cfg.resume_checkpoint, epoch=last_epoch)
            self.save_checkpoint("interrupted_model.pt", epoch=last_epoch)
            plot_training_history(self.history, self.output_dir)
            raise
        last_epoch = self.history[-1]["epoch"] if self.history else max(0, start_epoch - 1)
        self.save_checkpoint(self.cfg.resume_checkpoint, epoch=last_epoch)
        self.save_checkpoint("last_model.pt", epoch=last_epoch)
        plot_training_history(self.history, self.output_dir)
        elapsed = time.time() - start_time
        summary = {
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "elapsed_seconds": elapsed,
            "final_epoch": last_epoch,
        }
        save_json(summary, self.output_dir / "training_summary.json")
        return summary


###############################################################################
# 10. MAIN PIPELINE
###############################################################################


def prepare_data_for_fold(x: np.ndarray, c: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[TimeSeriesScaler]]:
    x_train = x[train_idx]
    c_train = c[train_idx]
    x_val = x[val_idx]
    c_val = c[val_idx]
    scaler = None
    if cfg.normalize:
        scaler = TimeSeriesScaler(cfg.normalization_method)
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_val = scaler.transform(x_val)
    return x_train, c_train, x_val, c_val, scaler


def run_single_fold(cfg: Config, x: np.ndarray, c: np.ndarray, feature_names: List[str], condition_names: List[str], train_idx: np.ndarray, val_idx: np.ndarray, fold_id: int, root_out: Path, x_test_raw: Optional[np.ndarray] = None, c_test_raw: Optional[np.ndarray] = None) -> Dict[str, Any]:
    fold_out = ensure_dir(root_out / f"fold_{fold_id}")
    fold_result_path = fold_out / "fold_result.json"
    if cfg.skip_completed_folds and fold_result_path.exists():
        try:
            previous = load_json(fold_result_path)
            final_epoch = int(previous.get("final_epoch", 0))
            if final_epoch >= cfg.epochs and (fold_out / "final_validation_metrics.json").exists():
                print(f"[INFO] Fold {fold_id}: already complete at epoch {final_epoch}; skipping.")
                return previous
        except Exception:
            pass
    if getattr(cfg, "adaptive_physics_thresholds", True):
        update_physics_thresholds_from_real_data(cfg, x[train_idx], feature_names)
        save_json(asdict(cfg), fold_out / "config_effective_fold.json")
    x_train, c_train, x_val, c_val, scaler = prepare_data_for_fold(x, c, train_idx, val_idx, cfg)
    train_ds = FarmingSequenceDataset(x_train, c_train)
    val_ds = FarmingSequenceDataset(x_val, c_val)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=len(train_ds) >= cfg.batch_size,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=False)
    trainer = PCWGANTrainer(cfg=cfg, feature_names=feature_names, condition_names=condition_names, scaler=scaler, output_dir=fold_out)
    summary = trainer.fit(train_loader, val_loader)
    if trainer.load_checkpoint("best_model.pt"):
        print(f"[INFO] Fold {fold_id}: best_model.pt reloaded before final metrics and plots")
    final_metrics = trainer.evaluate(val_loader)
    save_json(final_metrics, fold_out / "final_validation_metrics.json")
    output_row: Dict[str, Any] = {"fold": fold_id, **summary, **{f"final_val_{k}": v for k, v in final_metrics.items()}}
    # Hold-out test set evaluation
    if x_test_raw is not None and c_test_raw is not None and len(x_test_raw) > 0:
        if scaler is not None:
            x_test = scaler.transform(x_test_raw)
        else:
            x_test = x_test_raw
        c_test = c_test_raw
        test_ds = FarmingSequenceDataset(x_test, c_test)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=False)
        test_metrics = trainer.evaluate(test_loader)
        save_json(test_metrics, fold_out / "final_holdout_test_metrics.json")
        output_row.update({f"final_test_{k}": v for k, v in test_metrics.items()})
    fake_val = trainer.generate_numpy(c_val[: min(len(c_val), 256)])
    real_val = x_val[: len(fake_val)]
    if scaler is not None:
        fake_plot = scaler.inverse_transform(fake_val)
        real_plot = scaler.inverse_transform(real_val)
    else:
        fake_plot = fake_val
        real_plot = real_val
    plot_samples(real_plot, fake_plot, feature_names, fold_out, max_samples=cfg.max_plot_samples)
    np.save(fold_out / "generated_val_sequences.npy", fake_plot)
    np.save(fold_out / "real_val_sequences.npy", real_plot)
    save_json(output_row, fold_out / "fold_result.json")
    return output_row


def run_pipeline(cfg: Config) -> None:
    validate_config(cfg)
    set_seed(cfg.seed)
    print("[INFO] Training script: pcwgan_train.py (canonical consolidated version)")
    root_out = ensure_dir(Path(cfg.output_dir) / cfg.experiment_name)
    df, feature_cols, condition_cols = load_or_generate_data(cfg)
    # Update condition_dim and save config
    cfg.condition_dim = len(condition_cols)
    save_json(asdict(cfg), root_out / "config.json")
    print("[INFO] Features:", feature_cols)
    print("[INFO] Conditions:", condition_cols)
    df.to_csv(root_out / "input_dataset_snapshot.csv", index=False)
    x, c = make_windows(df=df, feature_cols=feature_cols, condition_cols=condition_cols, seq_len=cfg.seq_len, stride=cfg.stride, time_col=cfg.time_col, max_gap_seconds=cfg.max_gap_seconds)
    print(f"[INFO] Windowed dataset: sequences={x.shape}, conditions={c.shape}")
    purge_windows = window_purge_count(cfg.seq_len, cfg.stride)
    if cfg.holdout_test_ratio > 0 and len(x) >= 10:
        n_test = max(1, int(len(x) * cfg.holdout_test_ratio))
        test_start = len(x) - n_test
        n_trainval = max(0, test_start - purge_windows)
        if n_trainval < 2 or test_start >= len(x):
            n_trainval = len(x)
            n_test = 0
            test_start = len(x)
    else:
        n_trainval = len(x)
        n_test = 0
        test_start = len(x)
    x_trainval, c_trainval = x[:n_trainval], c[:n_trainval]
    x_test_raw = x[test_start:] if n_test > 0 else None
    c_test_raw = c[test_start:] if n_test > 0 else None
    if n_test > 0:
        print(
            f"[INFO] Hold-out test set: {n_test} sequences; "
            f"purged {test_start - n_trainval} overlapping boundary windows"
        )
    else:
        print("[INFO] No hold-out test set created")
    folds = temporal_cv_splits(len(x_trainval), cfg.n_splits, purge_windows=purge_windows)
    if not folds:
        raise ValueError(
            "No valid temporal CV folds after applying the overlap purge. "
            "Reduce seq_len, increase stride, or provide more data."
        )
    all_results = []
    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
        print(f"\n[INFO] Fold {fold_id}/{len(folds)} | train={len(train_idx)} | val={len(val_idx)}")
        result = run_single_fold(cfg=cfg, x=x_trainval, c=c_trainval, feature_names=feature_cols, condition_names=condition_cols, train_idx=train_idx, val_idx=val_idx, fold_id=fold_id, root_out=root_out, x_test_raw=x_test_raw, c_test_raw=c_test_raw)
        all_results.append(result)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(root_out / "cross_validation_results.csv", index=False)
    aggregate = {}
    for col in results_df.columns:
        if col == "fold":
            continue
        if pd.api.types.is_numeric_dtype(results_df[col]):
            aggregate[col + "_mean"] = float(results_df[col].mean())
            aggregate[col + "_std"] = float(results_df[col].std(ddof=0))
    save_json(aggregate, root_out / "aggregate_results.json")
    print("\n[INFO] Training complete.")
    print("[INFO] Results saved to:", root_out.resolve())
    print(results_df.to_string(index=False))


###############################################################################
# 11. CLI
###############################################################################


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="PC-WGAN-GP for hydroponic dataset")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs_pc_wgan_gp")
    parser.add_argument("--experiment_name", type=str, default="vertical_farming_pc_wgan_gp")
    parser.add_argument("--seq_len", type=int, default=120)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--critic_steps", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--rnn_type", type=str, default="lstm", choices=["lstm", "gru"])
    parser.add_argument("--use_step_noise", action="store_true", default=True)
    parser.add_argument("--no_step_noise", action="store_false", dest="use_step_noise")
    parser.add_argument("--step_noise_dim", type=int, default=32)
    parser.add_argument("--lr_g", type=float, default=2e-4)
    parser.add_argument("--lr_c", type=float, default=5e-5)
    parser.add_argument("--lambda_physics", type=float, default=1.0)
    parser.add_argument("--lambda_temporal", type=float, default=2.0)
    parser.add_argument("--lambda_acf", type=float, default=1.5)
    parser.add_argument("--lambda_smooth", type=float, default=0.03)
    parser.add_argument("--lambda_soft_dtw", type=float, default=0.0)
    parser.add_argument("--lambda_var", type=float, default=1.0)
    parser.add_argument("--lambda_delta", type=float, default=1.0)
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--holdout_test_ratio", type=float, default=0.15)
    parser.add_argument("--fid_max_samples", type=int, default=2048)
    parser.add_argument("--fid_feature_lags", type=int, default=12)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hd_max", type=float, default=30.0)
    parser.add_argument("--co2_max_rate", type=float, default=120.0)
    parser.add_argument("--ec_max_curvature", type=float, default=0.3)
    parser.add_argument("--feature_cols", type=str, default="")
    parser.add_argument("--condition_cols", type=str, default="")
    parser.add_argument("--time_col", type=str, default="")
    parser.add_argument("--max_gap_seconds", type=float, default=60.0)
    # Robust resume/checkpointing options for Colab
    parser.add_argument("--auto_resume", action="store_true", default=True, help="Resume automatically from resume_latest.pt if found")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume", help="Disable automatic resume")
    parser.add_argument("--resume_checkpoint", type=str, default="resume_latest.pt")
    parser.add_argument("--save_last_every", type=int, default=5, help="Save resumable checkpoint every N epochs")
    parser.add_argument("--skip_completed_folds", action="store_true", default=True)
    parser.add_argument("--no_skip_completed_folds", action="store_false", dest="skip_completed_folds")
    parser.add_argument("--disable_early_stopping", action="store_true", default=False)
    parser.add_argument("--early_stopping_patience", type=int, default=80)  # aligned with Config default
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--adaptive_physics_thresholds", action="store_true", default=True)
    parser.add_argument("--fixed_physics_thresholds", action="store_false", dest="adaptive_physics_thresholds")
    parser.add_argument("--smooth_quantile", type=float, default=0.95)
    parser.add_argument("--curvature_quantile", type=float, default=0.99)
    # Boundary overrides
    parser.add_argument("--ph_min", type=float)
    parser.add_argument("--ph_max", type=float)
    parser.add_argument("--ec_min", type=float)
    parser.add_argument("--ec_max", type=float)
    parser.add_argument("--water_temp_min", type=float)
    parser.add_argument("--water_temp_max", type=float)
    parser.add_argument("--temp_min", type=float)
    parser.add_argument("--temp_max", type=float)
    parser.add_argument("--rh_min", type=float)
    parser.add_argument("--rh_max", type=float)
    args = parser.parse_args()
    cfg = Config()
    for k, v in vars(args).items():
        if k in ["feature_cols", "condition_cols"]:
            continue
        if v is not None:
            setattr(cfg, k, v)
    cfg.time_col = args.time_col if args.time_col else None
    cfg.feature_cols = [s.strip() for s in args.feature_cols.split(",") if s.strip()] if args.feature_cols else None
    cfg.condition_cols = [s.strip() for s in args.condition_cols.split(",") if s.strip()] if args.condition_cols else None
    # Override boundaries if provided
    for bound in ["ph_min", "ph_max", "ec_min", "ec_max", "water_temp_min", "water_temp_max", "temp_min", "temp_max", "rh_min", "rh_max"]:
        val = getattr(args, bound)
        if val is not None:
            setattr(cfg, bound, val)
    return cfg


if __name__ == "__main__":
    config = parse_args()
    run_pipeline(config)
