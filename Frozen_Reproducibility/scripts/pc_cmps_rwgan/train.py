#!/usr/bin/env python3
"""Train PC-cMoMPS-RWGAN on prepared hydroponic time series."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from agronomic_physics import AgronomicConstraintLoss
from .model import (
    ConditionalMPSMixture,
    ResidualTrajectoryDecoder,
    count_trainable_parameters,
)
from pc_mps import Discretizer
from pcwgan_train import (
    Config as BaseConfig,
    ConditionalTemporalCritic,
    TimeSeriesScaler,
    acf_loss,
    compute_metrics,
    delta_distribution_loss,
    ensure_dir,
    get_device,
    gradient_penalty,
    init_pcwgan_weights,
    make_windows,
    save_json,
    set_seed,
    smoothness_loss,
    temporal_distribution_loss,
    update_physics_thresholds_from_real_data,
    variance_matching_loss,
    window_purge_count,
)


DEFAULT_FEATURES = ["ph", "ec", "water_temp", "air_temp", "rh"]
CHECKPOINT_PHASE_ORDER = {
    "prior_pretrain": 0,
    "decoder_pretrain": 1,
    "adversarial": 2,
    "complete": 3,
}


def load_checkpoint_payload(path: Path, map_location: Any = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def find_latest_valid_checkpoint(
    output_dir: Path,
) -> tuple[Optional[Path], list[tuple[Path, str]]]:
    candidates = [output_dir / "checkpoint_latest.pt"]
    candidates.extend(sorted(output_dir.glob("checkpoint_snapshot_*.pt")))
    valid: list[tuple[tuple[int, int, float], Path]] = []
    invalid: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            checkpoint = load_checkpoint_payload(path, map_location="cpu")
            phase = str(checkpoint.get("phase", "adversarial"))
            epoch = int(checkpoint.get("epoch", 0))
            rank = CHECKPOINT_PHASE_ORDER.get(phase, -1)
            valid.append(((rank, epoch, path.stat().st_mtime), path))
        except Exception as exc:
            invalid.append((path, str(exc)))
    if not valid:
        return None, invalid
    return max(valid, key=lambda item: item[0])[1], invalid


@dataclass
class HybridConfig:
    data_path: str
    output_dir: str
    feature_cols: list[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))
    condition_cols: list[str] = field(default_factory=list)
    time_col: Optional[str] = None
    seq_len: int = 120
    stride: int = 15
    max_gap_seconds: float = 60.0
    holdout_test_ratio: float = 0.15
    bins: int = 8
    mps_experts: int = 3
    mps_bond_dim: int = 4
    gate_hidden_dim: int = 32
    decoder_hidden_dim: int = 128
    critic_hidden_dim: int = 128
    num_layers: int = 2
    embedding_dim: int = 8
    noise_dim: int = 8
    residual_scale: float = 0.75
    batch_size: int = 64
    mps_pretrain_epochs: int = 20
    decoder_pretrain_epochs: int = 10
    adversarial_epochs: int = 100
    critic_steps: int = 3
    lr_mps: float = 0.005
    lr_decoder: float = 0.0002
    lr_critic: float = 0.00005
    gradient_penalty_lambda: float = 10.0
    lambda_mps_bound: float = 1.0
    lambda_gate_balance: float = 0.05
    lambda_gate_entropy: float = 0.005
    lambda_physics: float = 1.0
    lambda_code: float = 1.0
    lambda_reconstruction: float = 2.0
    lambda_temporal: float = 1.0
    lambda_acf: float = 0.5
    lambda_variance: float = 0.5
    lambda_delta: float = 0.5
    lambda_smooth: float = 0.02
    code_temperature: float = 0.25
    domain_margin: float = 0.10
    grad_clip_norm: float = 5.0
    gate_constant: bool = False
    prior_iid_uniform: bool = False
    non_residual_decoder: bool = False
    device: str = "auto"
    seed: int = 42
    auto_resume: bool = False
    resume_checkpoint: Optional[str] = None
    pretrain_save_every: int = 1
    save_every: int = 5
    checkpoint_keep: int = 3


def validate_config(cfg: HybridConfig) -> None:
    positive = (
        "seq_len", "stride", "bins", "mps_experts", "mps_bond_dim",
        "batch_size", "critic_steps", "pretrain_save_every", "save_every",
        "checkpoint_keep",
    )
    invalid = [name for name in positive if int(getattr(cfg, name)) <= 0]
    if invalid:
        raise ValueError(f"Configuration values must be positive: {invalid}")
    if cfg.bins < 2:
        raise ValueError("bins must be at least 2")
    if not 0.0 < cfg.holdout_test_ratio < 1.0:
        raise ValueError("holdout_test_ratio must be in (0, 1)")
    if cfg.mps_pretrain_epochs < 0 or cfg.decoder_pretrain_epochs < 0:
        raise ValueError("pretraining epochs cannot be negative")
    if cfg.adversarial_epochs < 0:
        raise ValueError("adversarial_epochs cannot be negative")


def feature_bounds(name: str, cfg: BaseConfig) -> tuple[float, float]:
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
    raise ValueError(f"No agronomic bounds are configured for feature {name!r}")


def build_discretizers(
    train_real: np.ndarray,
    feature_names: Sequence[str],
    metric_cfg: BaseConfig,
    bins: int,
    margin: float,
) -> list[Discretizer]:
    discretizers = []
    for index, name in enumerate(feature_names):
        values = train_real[:, :, index]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f"Feature {name!r} has no finite training values")
        physical_low, physical_high = feature_bounds(name, metric_cfg)
        low = min(float(finite.min()), physical_low)
        high = max(float(finite.max()), physical_high)
        span = max(high - low, 1e-6)
        padding = max(0.0, float(margin)) * span
        discretizers.append(Discretizer(low - padding, high + padding, bins))
    return discretizers


def encode_joint(values: np.ndarray, discretizers: Sequence[Discretizer]) -> torch.Tensor:
    channels = [
        discretizer.to_index(torch.from_numpy(values[:, :, feature]))
        for feature, discretizer in enumerate(discretizers)
    ]
    return torch.stack(channels, dim=-1).reshape(len(values), -1)


def decoder_geometry(
    discretizers: Sequence[Discretizer],
    scaler: TimeSeriesScaler,
    feature_names: Sequence[str],
    metric_cfg: BaseConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    mean = np.asarray(scaler.mean, dtype=np.float32).reshape(-1)
    std = np.asarray(scaler.std, dtype=np.float32).reshape(-1)
    centers = []
    lower = []
    upper = []
    allowed = []
    for feature, (name, discretizer) in enumerate(zip(feature_names, discretizers)):
        feature_centers = discretizer.centers().numpy()
        centers.append((feature_centers - mean[feature]) / std[feature])
        low, high = feature_bounds(name, metric_cfg)
        lower.append((low - mean[feature]) / std[feature])
        upper.append((high - mean[feature]) / std[feature])
        allowed.append(discretizer.allowed_indices(low, high))
    return (
        torch.tensor(np.stack(centers), dtype=torch.float32),
        torch.tensor(lower, dtype=torch.float32),
        torch.tensor(upper, dtype=torch.float32),
        allowed,
    )


def load_windows(cfg: HybridConfig) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    path = Path(cfg.data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    missing = [name for name in cfg.feature_cols + cfg.condition_cols if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")
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
    return frame, windows, conditions, condition_cols


def purged_holdout(
    windows: np.ndarray,
    conditions: np.ndarray,
    test_ratio: float,
    purge_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_test = max(1, int(len(windows) * test_ratio))
    test_start = len(windows) - n_test
    train_end = test_start - max(0, purge_windows)
    if train_end < 2:
        raise ValueError("Not enough windows for a purged hold-out")
    return windows[:train_end], conditions[:train_end], windows[test_start:], conditions[test_start:]


class HybridTrainer:
    def __init__(
        self,
        cfg: HybridConfig,
        metric_cfg: BaseConfig,
        feature_names: list[str],
        condition_names: list[str],
        scaler: TimeSeriesScaler,
        discretizers: list[Discretizer],
        output_dir: Path,
    ) -> None:
        self.cfg = cfg
        self.metric_cfg = metric_cfg
        self.feature_names = feature_names
        self.condition_names = condition_names
        self.scaler = scaler
        self.discretizers = discretizers
        self.output_dir = output_dir
        self.device = get_device(cfg.device)
        centers, lower, upper, allowed = decoder_geometry(
            discretizers, scaler, feature_names, metric_cfg
        )
        self.allowed_per_site = [
            allowed[feature].to(self.device)
            for _time in range(cfg.seq_len)
            for feature in range(len(feature_names))
        ]
        length = cfg.seq_len * len(feature_names)
        self.prior = ConditionalMPSMixture(
            length=length,
            bins=cfg.bins,
            condition_dim=len(condition_names),
            experts=cfg.mps_experts,
            bond_dim=cfg.mps_bond_dim,
            gate_hidden_dim=cfg.gate_hidden_dim,
            gate_constant=cfg.gate_constant,
            seed=cfg.seed,
        ).to(self.device)
        self.decoder = ResidualTrajectoryDecoder(
            seq_len=cfg.seq_len,
            feature_dim=len(feature_names),
            condition_dim=len(condition_names),
            bins=cfg.bins,
            centers_normalized=centers,
            lower_bounds_normalized=lower,
            upper_bounds_normalized=upper,
            embedding_dim=cfg.embedding_dim,
            hidden_dim=cfg.decoder_hidden_dim,
            num_layers=cfg.num_layers,
            noise_dim=cfg.noise_dim,
            residual_scale=cfg.residual_scale,
            non_residual=cfg.non_residual_decoder,
        ).to(self.device)
        self.critic = ConditionalTemporalCritic(
            input_dim=len(feature_names),
            condition_dim=len(condition_names),
            seq_len=cfg.seq_len,
            hidden_dim=cfg.critic_hidden_dim,
            num_layers=cfg.num_layers,
            dropout=0.1,
            rnn_type="gru",
        ).to(self.device)
        self.decoder.apply(init_pcwgan_weights)
        self.critic.apply(init_pcwgan_weights)
        self.physics_loss = AgronomicConstraintLoss(
            metric_cfg,
            feature_names,
            scaler=scaler,
            condition_names=condition_names,
        ).to(self.device)
        self.opt_prior = torch.optim.Adam(self.prior.parameters(), lr=cfg.lr_mps)
        self.opt_decoder = torch.optim.Adam(self.decoder.parameters(), lr=cfg.lr_decoder, betas=(0.0, 0.9))
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic, betas=(0.0, 0.9))
        self.history: list[dict[str, Any]] = []
        # Reused zero-noise buffer: avoids reallocating [B,T,noise_dim] tensors in
        # decoder pretraining and the adversarial reconstruction branch. The buffer
        # is intentionally not registered in state_dict because it has no learned
        # state and is rebuilt lazily after resume/device changes.
        self._zero_noise: Optional[torch.Tensor] = None
        self.live_status_path = self.output_dir / "live_status.json"
        self.live_history_path = self.output_dir / "training_history_live.csv"

    def _get_zero_noise(self, batch_size: int) -> torch.Tensor:
        if (
            self._zero_noise is None
            or self._zero_noise.device != self.device
            or self._zero_noise.dtype != torch.float32
            or self._zero_noise.shape[0] < int(batch_size)
            or self._zero_noise.shape[1] != self.cfg.seq_len
            or self._zero_noise.shape[2] != self.cfg.noise_dim
        ):
            self._zero_noise = torch.zeros(
                int(batch_size),
                self.cfg.seq_len,
                self.cfg.noise_dim,
                device=self.device,
            )
        return self._zero_noise[:int(batch_size)]

    def _device_is_cuda(self) -> bool:
        return getattr(self.device, "type", str(self.device)) == "cuda"

    def _sync_if_cuda(self) -> None:
        if self._device_is_cuda() and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 0 or not np.isfinite(seconds):
            return "?"
        seconds_int = int(round(seconds))
        hours, remainder = divmod(seconds_int, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:d}h{minutes:02d}m{secs:02d}s"
        if minutes:
            return f"{minutes:d}m{secs:02d}s"
        return f"{secs:d}s"

    def _gpu_memory_status(self) -> dict[str, float]:
        if not self._device_is_cuda() or not torch.cuda.is_available():
            return {}
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        return {
            "gpu_memory_allocated_gb": float(torch.cuda.memory_allocated(index) / 1024 ** 3),
            "gpu_memory_reserved_gb": float(torch.cuda.memory_reserved(index) / 1024 ** 3),
            "gpu_memory_max_allocated_gb": float(torch.cuda.max_memory_allocated(index) / 1024 ** 3),
        }

    @staticmethod
    def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
        return {
            key: float(np.nanmean([row[key] for row in rows]))
            for key in rows[0]
        }

    def _write_live_status(self, payload: dict[str, Any]) -> None:
        payload = {
            "updated_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            **payload,
            **self._gpu_memory_status(),
        }
        try:
            save_json(payload, self.live_status_path)
        except Exception as exc:
            print(f"[Hybrid status] could not write {self.live_status_path}: {exc}", flush=True)

    def _write_live_history(self) -> None:
        try:
            pd.DataFrame(self.history).to_csv(self.live_history_path, index=False)
        except Exception as exc:
            print(f"[Hybrid status] could not write {self.live_history_path}: {exc}", flush=True)

    def _prior_objective(self, codes: torch.Tensor, conditions: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        nll, bound = self.prior.nll_and_allowed_log_barrier(
            codes, self.allowed_per_site, conditions
        )
        balance, entropy = self.prior.gate_regularization(conditions)
        loss = (
            nll
            + self.cfg.lambda_mps_bound * bound
            + self.cfg.lambda_gate_balance * balance
            + self.cfg.lambda_gate_entropy * entropy
        )
        return loss, {
            "prior_nll_per_site": float(nll.detach().cpu()),
            "prior_bound_barrier_per_site": float(bound.detach().cpu()),
            "gate_balance_kl": float(balance.detach().cpu()),
            "gate_entropy": float(entropy.detach().cpu()),
        }

    @torch.no_grad()
    def _sample_codes(self, conditions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.prior_iid_uniform:
            codes = torch.randint(
                0,
                self.cfg.bins,
                (len(conditions), self.prior.length),
                device=self.device,
            )
            expert_ids = torch.full(
                (len(conditions),), -1, dtype=torch.long, device=self.device
            )
            return codes, expert_ids
        return self.prior.sample(conditions)

    def pretrain_prior(self, loader: DataLoader, start_epoch: int = 1) -> None:
        if self.cfg.prior_iid_uniform:
            print("[Hybrid prior] skipped: IID-uniform categorical prior ablation", flush=True)
            return
        print(
            f"[Hybrid prior] starting epochs {start_epoch}-{self.cfg.mps_pretrain_epochs} "
            f"with {len(loader)} batches/epoch",
            flush=True,
        )
        self.prior.train()
        progress_every = max(1, len(loader) // 4)
        for epoch in range(start_epoch, self.cfg.mps_pretrain_epochs + 1):
            epoch_started = time.perf_counter()
            rows = []
            for batch_index, (_, conditions, codes) in enumerate(loader, start=1):
                conditions = conditions.to(self.device)
                codes = codes.to(self.device)
                loss, metrics = self._prior_objective(codes, conditions)
                self.opt_prior.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.prior.parameters(), self.cfg.grad_clip_norm)
                self.opt_prior.step()
                self.prior.normalize_cores_()
                rows.append(metrics)
                if batch_index == 1 or batch_index % progress_every == 0 or batch_index == len(loader):
                    print(
                        f"[Hybrid prior] epoch={epoch:04d} "
                        f"batch={batch_index:04d}/{len(loader):04d}",
                        flush=True,
                    )
            summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
            self.history.append({"phase": "prior_pretrain", "epoch": epoch, **summary})
            if epoch == 1 or epoch == self.cfg.mps_pretrain_epochs or epoch % max(1, self.cfg.mps_pretrain_epochs // 5) == 0:
                print(
                    f"[Hybrid prior] epoch={epoch:04d} "
                    f"nll/site={summary['prior_nll_per_site']:.5f} "
                    f"seconds={time.perf_counter() - epoch_started:.1f}",
                    flush=True,
                )
            if epoch % self.cfg.pretrain_save_every == 0 or epoch == self.cfg.mps_pretrain_epochs:
                self.save_checkpoint(epoch, "checkpoint_latest.pt", phase="prior_pretrain")

    def pretrain_decoder(self, loader: DataLoader, start_epoch: int = 1) -> None:
        print(
            f"[Hybrid decoder] starting epochs {start_epoch}-{self.cfg.decoder_pretrain_epochs} "
            f"with {len(loader)} batches/epoch",
            flush=True,
        )
        self.decoder.train()
        progress_every = max(1, len(loader) // 4)
        for epoch in range(start_epoch, self.cfg.decoder_pretrain_epochs + 1):
            epoch_started = time.perf_counter()
            losses = []
            for batch_index, (real, conditions, codes) in enumerate(loader, start=1):
                real = real.to(self.device)
                conditions = conditions.to(self.device)
                codes = codes.to(self.device)
                reconstructed = self.decoder(codes, conditions, self._get_zero_noise(len(real)))
                reconstruction = torch.mean((reconstructed - real) ** 2)
                code = self.decoder.code_consistency_loss(
                    reconstructed, codes, self.cfg.code_temperature
                )
                loss = reconstruction + self.cfg.lambda_code * code
                self.opt_decoder.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.decoder.parameters(), self.cfg.grad_clip_norm)
                self.opt_decoder.step()
                losses.append((float(reconstruction.detach().cpu()), float(code.detach().cpu())))
                if batch_index == 1 or batch_index % progress_every == 0 or batch_index == len(loader):
                    print(
                        f"[Hybrid decoder] epoch={epoch:04d} "
                        f"batch={batch_index:04d}/{len(loader):04d}",
                        flush=True,
                    )
            reconstruction_mean = float(np.mean([item[0] for item in losses]))
            code_mean = float(np.mean([item[1] for item in losses]))
            self.history.append({
                "phase": "decoder_pretrain", "epoch": epoch,
                "reconstruction_loss": reconstruction_mean,
                "code_consistency_loss": code_mean,
            })
            if epoch == 1 or epoch == self.cfg.decoder_pretrain_epochs or epoch % max(1, self.cfg.decoder_pretrain_epochs // 5) == 0:
                print(
                    f"[Hybrid decoder] epoch={epoch:04d} reconstruction={reconstruction_mean:.5f} "
                    f"code={code_mean:.5f} seconds={time.perf_counter() - epoch_started:.1f}",
                    flush=True,
                )
            if epoch % self.cfg.pretrain_save_every == 0 or epoch == self.cfg.decoder_pretrain_epochs:
                self.save_checkpoint(epoch, "checkpoint_latest.pt", phase="decoder_pretrain")

    def train_adversarial(self, loader: DataLoader, start_epoch: int = 1) -> None:
        print(
            f"[Hybrid adversarial] starting epochs {start_epoch}-{self.cfg.adversarial_epochs} "
            f"with {len(loader)} batches/epoch and {self.cfg.critic_steps} critic steps",
            flush=True,
        )
        progress_every = max(1, len(loader) // 4)
        total_epochs = self.cfg.adversarial_epochs
        for epoch in range(start_epoch, self.cfg.adversarial_epochs + 1):
            epoch_started = time.perf_counter()
            batch_rows = []
            for batch_index, (real, conditions, real_codes) in enumerate(loader, start=1):
                real = real.to(self.device)
                conditions = conditions.to(self.device)
                real_codes = real_codes.to(self.device)

                if self.cfg.prior_iid_uniform:
                    prior_metrics = {
                        "prior_nll_per_site": float(np.log(self.cfg.bins)),
                        "prior_bound_barrier_per_site": float("nan"),
                        "gate_balance_kl": float("nan"),
                        "gate_entropy": float("nan"),
                    }
                else:
                    prior_loss, prior_metrics = self._prior_objective(real_codes, conditions)
                    self.opt_prior.zero_grad(set_to_none=True)
                    prior_loss.backward()
                    nn.utils.clip_grad_norm_(self.prior.parameters(), self.cfg.grad_clip_norm)
                    self.opt_prior.step()
                    self.prior.normalize_cores_()

                for _ in range(self.cfg.critic_steps):
                    with torch.no_grad():
                        sampled_codes, _ = self._sample_codes(conditions)
                        fake = self.decoder(sampled_codes, conditions)
                    real_score = self.critic(real, conditions).mean()
                    fake_score = self.critic(fake, conditions).mean()
                    gp = gradient_penalty(self.critic, real, fake, conditions, self.device)
                    critic_loss = fake_score - real_score + self.cfg.gradient_penalty_lambda * gp
                    self.opt_critic.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip_norm)
                    self.opt_critic.step()

                with torch.no_grad():
                    sampled_codes, expert_ids = self._sample_codes(conditions)
                fake = self.decoder(sampled_codes, conditions)
                adversarial = -self.critic(fake, conditions).mean()
                physics = self.physics_loss(fake, conditions)
                code = self.decoder.code_consistency_loss(fake, sampled_codes, self.cfg.code_temperature)
                reconstructed = self.decoder(
                    real_codes, conditions, self._get_zero_noise(len(real))
                )
                reconstruction = torch.mean((reconstructed - real) ** 2)
                temporal = temporal_distribution_loss(real, fake)
                acf = acf_loss(real, fake)
                variance = variance_matching_loss(real, fake)
                delta = delta_distribution_loss(real, fake)
                smooth = smoothness_loss(fake)
                decoder_loss = (
                    adversarial
                    + self.cfg.lambda_physics * physics
                    + self.cfg.lambda_code * code
                    + self.cfg.lambda_reconstruction * reconstruction
                    + self.cfg.lambda_temporal * temporal
                    + self.cfg.lambda_acf * acf
                    + self.cfg.lambda_variance * variance
                    + self.cfg.lambda_delta * delta
                    + self.cfg.lambda_smooth * smooth
                )
                self.opt_decoder.zero_grad(set_to_none=True)
                decoder_loss.backward()
                nn.utils.clip_grad_norm_(self.decoder.parameters(), self.cfg.grad_clip_norm)
                self.opt_decoder.step()
                if self.cfg.prior_iid_uniform:
                    usage = torch.zeros(self.cfg.mps_experts, device=self.device)
                else:
                    usage = torch.bincount(expert_ids, minlength=self.cfg.mps_experts).float()
                    usage = usage / usage.sum().clamp_min(1.0)
                row = {
                    **prior_metrics,
                    "critic_loss": float(critic_loss.detach().cpu()),
                    "gradient_penalty": float(gp.detach().cpu()),
                    "decoder_loss": float(decoder_loss.detach().cpu()),
                    "adversarial_loss": float(adversarial.detach().cpu()),
                    "physics_loss": float(physics.detach().cpu()),
                    "code_consistency_loss": float(code.detach().cpu()),
                    "reconstruction_loss": float(reconstruction.detach().cpu()),
                    "temporal_loss": float(temporal.detach().cpu()),
                    "acf_loss": float(acf.detach().cpu()),
                    "variance_loss": float(variance.detach().cpu()),
                    "delta_loss": float(delta.detach().cpu()),
                    **{f"expert_usage_{index}": float(usage[index].cpu()) for index in range(self.cfg.mps_experts)},
                }
                batch_rows.append(row)
                if batch_index == 1 or batch_index % progress_every == 0 or batch_index == len(loader):
                    self._sync_if_cuda()
                    elapsed = time.perf_counter() - epoch_started
                    seconds_per_batch = elapsed / max(1, batch_index)
                    remaining_epoch_batches = max(0, len(loader) - batch_index)
                    remaining_total_batches = (
                        remaining_epoch_batches
                        + max(0, total_epochs - epoch) * len(loader)
                    )
                    eta_epoch = remaining_epoch_batches * seconds_per_batch
                    eta_total = remaining_total_batches * seconds_per_batch
                    rolling = self._mean_metrics(batch_rows[-min(5, len(batch_rows)):])
                    memory = self._gpu_memory_status()
                    memory_text = ""
                    if memory:
                        memory_text = (
                            f" mem={memory['gpu_memory_allocated_gb']:.2f}GB"
                            f"/{memory['gpu_memory_reserved_gb']:.2f}GB"
                        )
                    pct = 100.0 * batch_index / max(1, len(loader))
                    print(
                        f"[Hybrid adversarial] epoch={epoch:04d}/{total_epochs:04d} "
                        f"batch={batch_index:04d}/{len(loader):04d} "
                        f"{pct:5.1f}% elapsed={self._format_duration(elapsed)} "
                        f"eta_epoch={self._format_duration(eta_epoch)} "
                        f"eta_total~={self._format_duration(eta_total)} "
                        f"D={rolling['critic_loss']:.4f} G={rolling['decoder_loss']:.4f} "
                        f"gp={rolling['gradient_penalty']:.4f} "
                        f"phys={rolling['physics_loss']:.4f} "
                        f"code={rolling['code_consistency_loss']:.4f} "
                        f"rec={rolling['reconstruction_loss']:.4f}"
                        f"{memory_text}",
                        flush=True,
                    )
                    self._write_live_status({
                        "phase": "adversarial",
                        "event": "batch_progress",
                        "epoch": epoch,
                        "epoch_total": total_epochs,
                        "batch": batch_index,
                        "batches_per_epoch": len(loader),
                        "epoch_progress_pct": pct,
                        "elapsed_epoch_seconds": float(elapsed),
                        "eta_epoch_seconds": float(eta_epoch),
                        "eta_total_seconds": float(eta_total),
                        "seconds_per_batch": float(seconds_per_batch),
                        "latest_batch_metrics": row,
                        "rolling_last_5_batch_metrics": rolling,
                    })
            self._sync_if_cuda()
            summary = self._mean_metrics(batch_rows)
            self.history.append({"phase": "adversarial", "epoch": epoch, **summary})
            epoch_seconds = time.perf_counter() - epoch_started
            self._write_live_history()
            print(
                f"[Hybrid epoch] epoch={epoch:04d}/{total_epochs:04d} "
                f"D={summary['critic_loss']:.4f} G={summary['decoder_loss']:.4f} "
                f"adv={summary['adversarial_loss']:.4f} gp={summary['gradient_penalty']:.4f} "
                f"phys={summary['physics_loss']:.4f} code={summary['code_consistency_loss']:.4f} "
                f"rec={summary['reconstruction_loss']:.4f} temp={summary['temporal_loss']:.4f} "
                f"acf={summary['acf_loss']:.4f} var={summary['variance_loss']:.4f} "
                f"delta={summary['delta_loss']:.4f} "
                f"prior_nll/site={summary['prior_nll_per_site']:.4f} "
                f"seconds={epoch_seconds:.1f} avg_batch={epoch_seconds / max(1, len(loader)):.2f}s",
                flush=True,
            )
            self._write_live_status({
                "phase": "adversarial",
                "event": "epoch_summary",
                "epoch": epoch,
                "epoch_total": total_epochs,
                "batches_per_epoch": len(loader),
                "epoch_seconds": float(epoch_seconds),
                "average_seconds_per_batch": float(epoch_seconds / max(1, len(loader))),
                "checkpoint_due": bool(epoch % self.cfg.save_every == 0 or epoch == self.cfg.adversarial_epochs),
                "epoch_mean_metrics": summary,
                "live_history_csv": str(self.live_history_path),
                "latest_checkpoint": str(self.output_dir / "checkpoint_latest.pt"),
            })
            if epoch % self.cfg.save_every == 0 or epoch == self.cfg.adversarial_epochs:
                self.save_checkpoint(epoch, "checkpoint_latest.pt", phase="adversarial")

    @torch.no_grad()
    def generate(
        self,
        conditions: np.ndarray,
        batch_size: int = 256,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.prior.eval()
        self.decoder.eval()
        generated = []
        experts = []
        sampled_codes = []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.cfg.seed + 100_003)
            for start in range(0, len(conditions), batch_size):
                batch = torch.tensor(conditions[start:start + batch_size], dtype=torch.float32, device=self.device)
                codes, expert_ids = self._sample_codes(batch)
                generated.append(self.decoder(codes, batch).cpu().numpy())
                experts.append(expert_ids.cpu().numpy())
                sampled_codes.append(codes.cpu().numpy())
        return (
            np.concatenate(generated),
            np.concatenate(experts),
            np.concatenate(sampled_codes),
        )

    @torch.no_grad()
    def prior_metrics(self, codes: torch.Tensor, conditions: torch.Tensor) -> dict[str, Any]:
        if self.cfg.prior_iid_uniform:
            allowed_fraction = [
                float(len(allowed)) / float(self.cfg.bins)
                for allowed in self.allowed_per_site
            ]
            return {
                "iid_uniform_log_likelihood_per_site": -float(np.log(self.cfg.bins)),
                "iid_uniform_bound_log_barrier_per_site": -float(
                    np.mean(np.log(np.maximum(allowed_fraction, 1e-30)))
                ),
                "mean_gate_probabilities": [],
            }
        self.prior.eval()
        codes = codes.to(self.device)
        conditions = conditions.to(self.device)
        gate_log_probs = self.prior.gate_log_probs(conditions)
        log_norms = self.prior.expert_log_norms()
        log_prob = self.prior.log_prob(
            codes, conditions, log_norms=log_norms, gate_log_probs=gate_log_probs
        )
        gate_prob = gate_log_probs.exp().mean(dim=0)
        bound = self.prior.allowed_log_barrier(
            self.allowed_per_site, conditions, log_norms=log_norms, gate_log_probs=gate_log_probs
        )
        return {
            "exact_conditional_log_likelihood_mean": float(log_prob.mean().cpu()),
            "exact_conditional_log_likelihood_per_site": float((log_prob.mean() / self.prior.length).cpu()),
            "exact_conditional_bound_log_barrier_per_site": float(bound.cpu()),
            "mean_gate_probabilities": [float(value) for value in gate_prob.cpu()],
        }

    def save_checkpoint(self, epoch: int, name: str, phase: str = "adversarial") -> None:
        payload = {
            "epoch": epoch,
            "phase": phase,
            "prior": self.prior.state_dict(),
            "decoder": self.decoder.state_dict(),
            "critic": self.critic.state_dict(),
            "opt_prior": self.opt_prior.state_dict(),
            "opt_decoder": self.opt_decoder.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "history": self.history,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "config": asdict(self.cfg),
            "feature_names": self.feature_names,
            "condition_names": self.condition_names,
            "discretizers": [item.as_dict() for item in self.discretizers],
        }
        target = self.output_dir / name
        fd, temporary_name = tempfile.mkstemp(prefix="pc_cmps_rwgan_", suffix=".pt")
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            torch.save(payload, temporary)
            load_checkpoint_payload(temporary, map_location="cpu")
            if name == "checkpoint_latest.pt":
                snapshot = self.output_dir / f"checkpoint_snapshot_{phase}_{epoch:04d}.pt"
                shutil.copyfile(temporary, snapshot)
                load_checkpoint_payload(snapshot, map_location="cpu")
                shutil.copyfile(temporary, target)
                load_checkpoint_payload(target, map_location="cpu")
                snapshots = sorted(
                    self.output_dir.glob(f"checkpoint_snapshot_{phase}_*.pt"),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
                for obsolete in snapshots[self.cfg.checkpoint_keep:]:
                    obsolete.unlink(missing_ok=True)
            else:
                shutil.copyfile(temporary, target)
                load_checkpoint_payload(target, map_location="cpu")
        finally:
            temporary.unlink(missing_ok=True)
        print(
            f"[Hybrid checkpoint] phase={phase} epoch={epoch} "
            f"path={target}",
            flush=True,
        )

    def load_checkpoint(self, checkpoint_path: Path) -> tuple[str, int]:
        checkpoint = load_checkpoint_payload(checkpoint_path, map_location=self.device)

        checkpoint_config = checkpoint.get("config", {})
        structural_keys = (
            "feature_cols", "condition_cols", "seq_len", "bins", "mps_experts",
            "mps_bond_dim", "gate_hidden_dim", "decoder_hidden_dim",
            "critic_hidden_dim", "num_layers", "embedding_dim", "noise_dim",
            "gate_constant", "prior_iid_uniform", "non_residual_decoder",
            "batch_size", "critic_steps", "mps_pretrain_epochs",
            "decoder_pretrain_epochs", "adversarial_epochs", "lr_mps",
            "lr_decoder", "lr_critic", "lambda_mps_bound", "lambda_physics",
            "lambda_code", "lambda_reconstruction",
        )
        mismatches = [
            key for key in structural_keys
            if key in checkpoint_config
            and checkpoint_config[key] != asdict(self.cfg).get(key)
        ]
        if mismatches:
            raise ValueError(
                "Checkpoint incompatible with the current architecture: "
                + ", ".join(mismatches)
            )
        if checkpoint.get("feature_names") != self.feature_names:
            raise ValueError("Checkpoint feature columns do not match the current dataset")
        if checkpoint.get("condition_names") != self.condition_names:
            raise ValueError("Checkpoint condition columns do not match the current dataset")

        self.prior.load_state_dict(checkpoint["prior"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.critic.load_state_dict(checkpoint["critic"])
        for key, optimizer in (
            ("opt_prior", self.opt_prior),
            ("opt_decoder", self.opt_decoder),
            ("opt_critic", self.opt_critic),
        ):
            if key in checkpoint:
                optimizer.load_state_dict(checkpoint[key])
        self.history = list(checkpoint.get("history", []))
        if "torch_rng_state" in checkpoint:
            cpu_rng_state = checkpoint["torch_rng_state"].detach().to(
                device="cpu", dtype=torch.uint8
            )
            torch.set_rng_state(cpu_rng_state)
        cuda_state = checkpoint.get("cuda_rng_state_all")
        if cuda_state is not None and torch.cuda.is_available():
            try:
                states = [cuda_state] if torch.is_tensor(cuda_state) else list(cuda_state)
                normalized_states = [
                    state.detach().to(device="cpu", dtype=torch.uint8)
                    for state in states[:torch.cuda.device_count()]
                ]
                torch.cuda.set_rng_state_all(normalized_states)
            except (TypeError, ValueError, RuntimeError, AttributeError) as exc:
                print(
                    f"[Hybrid warning] CUDA RNG state was not restored: {exc}",
                    flush=True,
                )
        phase = str(checkpoint.get("phase", "adversarial"))
        epoch = int(checkpoint.get("epoch", 0))
        print(f"[Hybrid] resumed phase={phase} epoch={epoch} from {checkpoint_path}", flush=True)
        return phase, epoch

    def maybe_resume(self) -> tuple[Optional[str], int]:
        if self.cfg.resume_checkpoint:
            checkpoint_path = Path(self.cfg.resume_checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = self.output_dir / checkpoint_path
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
            return self.load_checkpoint(checkpoint_path)
        if self.cfg.auto_resume:
            checkpoint_path, invalid = find_latest_valid_checkpoint(self.output_dir)
            for invalid_path, reason in invalid:
                print(
                    f"[Hybrid warning] ignoring invalid checkpoint {invalid_path}: {reason}",
                    flush=True,
                )
            if checkpoint_path is not None:
                return self.load_checkpoint(checkpoint_path)
            print(
                f"[Hybrid] no valid checkpoint found in {self.output_dir}; starting fresh",
                flush=True,
            )
        return None, 0


def run_pipeline(cfg: HybridConfig) -> dict[str, Any]:
    validate_config(cfg)
    set_seed(cfg.seed)
    output_dir = ensure_dir(cfg.output_dir)
    frame, windows, conditions, condition_names = load_windows(cfg)
    purge = window_purge_count(cfg.seq_len, cfg.stride)
    train_real, train_conditions, test_real, test_conditions = purged_holdout(
        windows, conditions, cfg.holdout_test_ratio, purge
    )
    scaler = TimeSeriesScaler("zscore").fit(train_real)
    train_norm = scaler.transform(train_real).astype(np.float32)
    test_norm = scaler.transform(test_real).astype(np.float32)

    metric_cfg = BaseConfig(seq_len=cfg.seq_len, stride=cfg.stride, seed=cfg.seed)
    update_physics_thresholds_from_real_data(metric_cfg, train_real, cfg.feature_cols)
    discretizers = build_discretizers(
        train_real, cfg.feature_cols, metric_cfg, cfg.bins, cfg.domain_margin
    )
    train_codes = encode_joint(train_real, discretizers)
    test_codes = encode_joint(test_real, discretizers)
    loader_generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(train_norm),
            torch.tensor(train_conditions, dtype=torch.float32),
            train_codes,
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=loader_generator,
        drop_last=False,
    )
    trainer = HybridTrainer(
        cfg, metric_cfg, cfg.feature_cols, condition_names,
        scaler, discretizers, output_dir,
    )
    print(
        f"[Hybrid] windows={len(windows)} train={len(train_real)} holdout={len(test_real)} "
        f"purge={purge} device={trainer.device}",
        flush=True,
    )
    print(
        f"[Hybrid] parameters prior={count_trainable_parameters(trainer.prior)} "
        f"decoder={count_trainable_parameters(trainer.decoder)} "
        f"critic={count_trainable_parameters(trainer.critic)}",
        flush=True,
    )
    resumed_phase, resumed_epoch = trainer.maybe_resume()
    if resumed_phase is None:
        if cfg.mps_pretrain_epochs:
            trainer.pretrain_prior(loader)
        if cfg.decoder_pretrain_epochs:
            trainer.pretrain_decoder(loader)
        adversarial_start = 1
    elif resumed_phase == "prior_pretrain":
        if resumed_epoch < cfg.mps_pretrain_epochs:
            trainer.pretrain_prior(loader, start_epoch=resumed_epoch + 1)
        if cfg.decoder_pretrain_epochs:
            trainer.pretrain_decoder(loader)
        adversarial_start = 1
    elif resumed_phase == "decoder_pretrain":
        if resumed_epoch < cfg.decoder_pretrain_epochs:
            trainer.pretrain_decoder(loader, start_epoch=resumed_epoch + 1)
        adversarial_start = 1
    elif resumed_phase in {"adversarial", "complete"}:
        adversarial_start = resumed_epoch + 1
    else:
        raise ValueError(f"Unknown checkpoint phase: {resumed_phase}")
    trainer.train_adversarial(loader, start_epoch=adversarial_start)
    fake_norm, sampled_experts, sampled_codes = trainer.generate(test_conditions)
    fake_real = scaler.inverse_transform(fake_norm)

    np.save(output_dir / "real_sequences.npy", test_norm)
    np.save(output_dir / "fake_sequences.npy", fake_norm.astype(np.float32))
    np.save(output_dir / "real_sequences_real_unit.npy", test_real.astype(np.float32))
    np.save(output_dir / "fake_sequences_real_unit.npy", fake_real.astype(np.float32))
    np.save(output_dir / "sampled_expert_ids.npy", sampled_experts)
    np.save(output_dir / "sampled_mps_codes.npy", sampled_codes)
    np.save(output_dir / "zscore_mu.npy", np.asarray(scaler.mean).reshape(-1))
    np.save(output_dir / "zscore_sigma.npy", np.asarray(scaler.std).reshape(-1))

    standard_metrics = compute_metrics(
        test_norm, fake_norm, cfg.feature_cols, metric_cfg, scaler=scaler
    )
    exact_metrics = trainer.prior_metrics(
        test_codes,
        torch.tensor(test_conditions, dtype=torch.float32),
    )
    with torch.no_grad():
        fake_tensor = torch.tensor(fake_norm, dtype=torch.float32, device=trainer.device)
        code_tensor = torch.tensor(sampled_codes, dtype=torch.long, device=trainer.device)
        recovered = trainer.decoder.code_logits(
            fake_tensor, cfg.code_temperature
        ).argmax(dim=-1)
        structured_codes = trainer.decoder.reshape_tokens(code_tensor)
        coarse = trainer.decoder.tokens_to_centers(code_tensor)
        code_recovery_accuracy = float(
            (recovered == structured_codes).float().mean().cpu()
        )
        decoder_coarse_mse = float(torch.mean((fake_tensor - coarse) ** 2).cpu())
    metrics: dict[str, Any] = {
        "status": "ok",
        "model": "pc_cmps_rwgan",
        "research_status": "novelty_candidate_not_global_novelty_proof",
        "conditional": True,
        "joint_multivariate_mps": True,
        "prior_type": "iid_uniform_categorical" if cfg.prior_iid_uniform else "conditional_mps_mixture",
        "gate_type": "global_learned" if cfg.gate_constant else "actuator_conditioned",
        "decoder_type": "direct_bounded" if cfg.non_residual_decoder else "residual_bounded",
        "discrete_gradient_policy": "alternating_exact_nll_and_continuous_adversarial_optimization",
        "n_training_sequences": int(len(train_real)),
        "n_holdout_sequences": int(len(test_real)),
        "purge_windows": int(purge),
        "sampled_expert_usage": [] if cfg.prior_iid_uniform else [
            float(np.mean(sampled_experts == index)) for index in range(cfg.mps_experts)
        ],
        "code_recovery_accuracy": code_recovery_accuracy,
        "decoder_coarse_mse": decoder_coarse_mse,
        **standard_metrics,
        **exact_metrics,
    }
    save_json(metrics, output_dir / "metrics.json")
    save_json(asdict(cfg), output_dir / "config.json")
    save_json({
        "feature_cols": cfg.feature_cols,
        "condition_cols": condition_names,
        "discretizers": [item.as_dict() for item in discretizers],
        "physical_bounds": [list(feature_bounds(name, metric_cfg)) for name in cfg.feature_cols],
        "scaler": scaler.state_dict(),
    }, output_dir / "hybrid_geometry.json")
    pd.DataFrame(trainer.history).to_csv(output_dir / "training_history.csv", index=False)
    frame.to_csv(output_dir / "input_dataset_snapshot.csv", index=False)
    trainer.save_checkpoint(
        cfg.adversarial_epochs, "pc_cmps_rwgan_final.pt", phase="complete"
    )
    print(f"[Hybrid] completed; outputs saved to {output_dir.resolve()}", flush=True)
    return metrics


def parse_args() -> HybridConfig:
    parser = argparse.ArgumentParser(description="PC-cMoMPS-RWGAN hydroponic runner")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_cols", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--condition_cols", default="")
    parser.add_argument("--time_col", default="")
    parser.add_argument("--seq_len", type=int, default=120)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--max_gap_seconds", type=float, default=60.0)
    parser.add_argument("--holdout_test_ratio", type=float, default=0.15)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--mps_experts", type=int, default=3)
    parser.add_argument("--mps_bond_dim", type=int, default=4)
    parser.add_argument("--gate_hidden_dim", type=int, default=32)
    parser.add_argument("--decoder_hidden_dim", type=int, default=128)
    parser.add_argument("--critic_hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=8)
    parser.add_argument("--noise_dim", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mps_pretrain_epochs", type=int, default=20)
    parser.add_argument("--decoder_pretrain_epochs", type=int, default=10)
    parser.add_argument("--adversarial_epochs", type=int, default=100)
    parser.add_argument("--critic_steps", type=int, default=3)
    parser.add_argument("--lr_mps", type=float, default=0.005)
    parser.add_argument("--lr_decoder", type=float, default=0.0002)
    parser.add_argument("--lr_critic", type=float, default=0.00005)
    parser.add_argument("--lambda_mps_bound", type=float, default=1.0)
    parser.add_argument("--lambda_physics", type=float, default=1.0)
    parser.add_argument("--lambda_code", type=float, default=1.0)
    parser.add_argument("--lambda_reconstruction", type=float, default=2.0)
    parser.add_argument("--gate_constant", action="store_true")
    parser.add_argument("--prior_iid_uniform", action="store_true")
    parser.add_argument("--non_residual_decoder", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--pretrain_save_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--checkpoint_keep", type=int, default=3)
    args = parser.parse_args()
    return HybridConfig(
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
        mps_experts=args.mps_experts,
        mps_bond_dim=args.mps_bond_dim,
        gate_hidden_dim=args.gate_hidden_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        critic_hidden_dim=args.critic_hidden_dim,
        num_layers=args.num_layers,
        embedding_dim=args.embedding_dim,
        noise_dim=args.noise_dim,
        batch_size=args.batch_size,
        mps_pretrain_epochs=args.mps_pretrain_epochs,
        decoder_pretrain_epochs=args.decoder_pretrain_epochs,
        adversarial_epochs=args.adversarial_epochs,
        critic_steps=args.critic_steps,
        lr_mps=args.lr_mps,
        lr_decoder=args.lr_decoder,
        lr_critic=args.lr_critic,
        lambda_mps_bound=args.lambda_mps_bound,
        lambda_physics=args.lambda_physics,
        lambda_code=args.lambda_code,
        lambda_reconstruction=args.lambda_reconstruction,
        gate_constant=args.gate_constant,
        prior_iid_uniform=args.prior_iid_uniform,
        non_residual_decoder=args.non_residual_decoder,
        device=args.device,
        seed=args.seed,
        auto_resume=args.auto_resume,
        resume_checkpoint=args.resume_checkpoint or None,
        pretrain_save_every=args.pretrain_save_every,
        save_every=args.save_every,
        checkpoint_keep=args.checkpoint_keep,
    )


if __name__ == "__main__":
    run_pipeline(parse_args())
