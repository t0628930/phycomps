#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agronomic_physics.py

Drop-in replacement for the physics constraints and a new Agronomic Quality
Index (AQI) for `final_agri_pc_wgan_gp.py`.

This module makes the code MATCH the manuscript: instead of only penalising
min/max bounds, it implements *differentiable* agronomic relations:

  phi_1  Humidity deficit (Tetens saturation vapour pressure + ideal-gas HD)
  phi_2  CO2 mass balance (discrete continuity: change ~ assimilation/ventilation)
  phi_3  Lighting thermodynamics (radiative load vs. air-temperature response)
  phi_4  Irrigation dynamics (EC/water-temp response to irrigation events)
  phi_5  Range plausibility (kept from original, now one term among several)
  phi_6  Smoothness / no implausible single-step jumps

All terms operate on REAL units (the model output is de-normalised first using
the same scaler statistics as the rest of the pipeline), so the penalties are
physically meaningful and comparable across variables.

AQI
---
The Agronomic Quality Index summarises biophysical plausibility on a 0..1 scale
(1 = fully plausible). It is computed on REAL-UNIT data and combines:
  - range compliance (fraction of timesteps within agronomic bounds),
  - thermodynamic consistency (humidity-deficit non-negativity & plausibility),
  - temporal smoothness (absence of implausible jumps).
It is reported as a numpy float and is meant to replace "AQI" in your tables.

INTEGRATION (3 steps)
---------------------
1) In `final_agri_pc_wgan_gp.py`, delete the body of class
   `AgronomicConstraintLoss` and import this one instead:
       from agronomic_physics import AgronomicConstraintLoss, agronomic_quality_index
   The constructor signature is unchanged: (cfg, feature_names, scaler).

2) In `compute_metrics(...)`, after you build `real_unit`/`fake_unit`, add:
       metrics["aqi"] = agronomic_quality_index(fake_unit, feature_names, cfg)
       metrics["aqi_real_ref"] = agronomic_quality_index(real_unit, feature_names, cfg)
   (the second gives a reference baseline = AQI of the real data).

3) (Recommended) make the composite score reward high AQI:
       metrics["composite_score"] = (
           metrics["mmd_rbf"]
           + 0.5 * metrics["acf_error"]
           + 0.2 * metrics["dtw_approx"]
           + 2.0 * (1.0 - metrics["aqi"])      # lower is better -> 1-AQI
       )

Notes
- Feature mapping is by substring on lower-cased names, kept compatible with
  your existing naming (air_temp, rh, co2, ph, ec, water_temp, light_lux).
- Every term is guarded: if the variables it needs are absent, it contributes 0,
  so the loss degrades gracefully on arbitrary datasets.
- Constants follow the manuscript: Tetens (eq. 2) and HD (eq. 3).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# Default physically interpretable single-step thresholds in real units.
# At training time, pcwgan_train.py can override them from real-data quantiles
# through cfg.smooth_threshold_<feature> fields.
SMOOTH_THRESHOLDS = {
    "ph": 0.15,
    "ec": 0.20,
    "water_temp": 3.0,
    "air_temp": 1.0,
    "rh": 4.0,
}


# =============================================================================
# Feature resolution helper
# =============================================================================

def _resolve_indices(feature_names: List[str]) -> Dict[str, Optional[int]]:
    """
    Map canonical agronomic variables to column indices by substring match.
    Returns a dict with keys: air_temp, rh, co2, ph, ec, water_temp, light.
    Missing variables map to None.
    """
    lower = [n.lower() for n in feature_names]

    def find(*keys: str) -> Optional[int]:
        for idx, name in enumerate(lower):
            if any(k in name for k in keys):
                return idx
        return None

    return {
        "air_temp": find("air_temp", "temperature", "temp_air", "t_air"),
        "rh":       find("rh", "humidity", "humid"),
        "co2":      find("co2", "co_2"),
        "ph":       find("ph"),
        "ec":       find("ec", "tds"),
        "water_temp": find("water_temp", "water_temperature", "temp_water"),
        "light":    find("light", "lux", "ppfd", "par", "dli"),
    }


# =============================================================================
# Differentiable physics helpers (operate on REAL units, torch tensors)
# =============================================================================

def tetens_saturation_vapor_pressure(temp_c: torch.Tensor) -> torch.Tensor:
    """
    Tetens approximation (manuscript eq. 2). Input T in degrees C, output kPa.
    Psat = 0.6108 * exp(17.27 T / (T + 237.3))
    Fully differentiable.
    """
    denominator = torch.clamp(temp_c + 237.3, min=1e-3)
    exponent = torch.clamp(17.27 * temp_c / denominator, min=-50.0, max=50.0)
    return 0.6108 * torch.exp(exponent)


def humidity_deficit(temp_c: torch.Tensor, rh_percent: torch.Tensor) -> torch.Tensor:
    """
    Humidity deficit HD in g/m^3 (manuscript eq. 3), derived from psychrometry.
    Pv = (RH/100) * Psat ; HD = 2.16679 * (Psat - Pv) / (T + 273.15) * 1000

    RH is intentionally not hard-clamped here. Out-of-range RH values are handled
    by the range penalty, while this term still provides a useful gradient when
    the generator predicts RH > 100% and causes negative humidity deficit.
    """
    psat = tetens_saturation_vapor_pressure(temp_c)
    rh_frac = rh_percent / 100.0
    pv = rh_frac * psat
    absolute_temp = torch.clamp(temp_c + 273.15, min=1.0)
    hd = 2.16679 * (psat - pv) / absolute_temp * 1000.0
    return hd


def _safe_index(x: torch.Tensor, idx: Optional[int]) -> Optional[torch.Tensor]:
    """Return x[:, :, idx] (shape [B, T]) or None if idx is None."""
    if idx is None:
        return None
    return x[:, :, idx]


# =============================================================================
# Agronomic constraint loss (differentiable, multi-domain)
# =============================================================================

class AgronomicConstraintLoss(nn.Module):
    """
    Multi-domain differentiable agronomic penalty for the generator.

    Signature is kept identical to the original so it is a drop-in replacement:
        AgronomicConstraintLoss(cfg, feature_names, scaler)

    forward(x) expects x in NORMALISED space, shape [B, T, F]; it de-normalises
    internally using the scaler (same convention as the original code).
    Returns a scalar tensor (sum of weighted physics terms).
    """

    def __init__(self, cfg, feature_names: List[str], scaler=None, condition_names: Optional[List[str]] = None):
        super().__init__()
        self.cfg = cfg
        self.feature_names = feature_names
        self.scaler = scaler
        self.idx = _resolve_indices(feature_names)
        self.condition_names = list(condition_names or [])
        irrigation_tokens = ("irrig", "add_water", "nutrient", "dosing", "ph_reducer")
        self.irrigation_condition_indices = [
            i for i, name in enumerate(self.condition_names)
            if any(token in name.lower() for token in irrigation_tokens)
        ]

        # Per-term weights (override via cfg attributes if present).
        self.w_hd     = float(getattr(cfg, "w_humidity_deficit", 1.0))
        self.w_co2    = float(getattr(cfg, "w_co2_balance", 1.0))
        self.w_light  = float(getattr(cfg, "w_light_thermo", 0.5))
        self.w_irrig  = float(getattr(cfg, "w_irrigation", 0.5))
        self.w_range  = float(getattr(cfg, "w_range", 1.0))
        self.w_smooth = float(getattr(cfg, "w_smooth_phys", 0.2))

        # Agronomic ranges (real units), pulled from cfg.
        self.ranges = {
            "air_temp":  (cfg.temp_min, cfg.temp_max),
            "rh":        (cfg.rh_min, cfg.rh_max),
            "co2":       (cfg.co2_min, cfg.co2_max),
            "ph":        (cfg.ph_min, cfg.ph_max),
            "ec":        (cfg.ec_min, cfg.ec_max),
            "water_temp": (cfg.water_temp_min, cfg.water_temp_max),
        }

    # ---- de-normalisation (matches original _denormalize_torch) ------------
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.scaler is None:
            return x
        if getattr(self.scaler, "method", "zscore") == "zscore":
            mean = torch.as_tensor(self.scaler.mean, dtype=x.dtype, device=x.device)
            std = torch.as_tensor(self.scaler.std, dtype=x.dtype, device=x.device)
            return x * std + mean
        minv = torch.as_tensor(self.scaler.min, dtype=x.dtype, device=x.device)
        maxv = torch.as_tensor(self.scaler.max, dtype=x.dtype, device=x.device)
        return (x + 1.0) * 0.5 * (maxv - minv + 1e-8) + minv

    @staticmethod
    def _range_penalty(v: torch.Tensor, low: float, high: float) -> torch.Tensor:
        return torch.relu(low - v).mean() + torch.relu(v - high).mean()

    def _smooth_threshold(self, key: str) -> float:
        """Return data-calibrated threshold if present, otherwise the default."""
        return float(getattr(self.cfg, f"smooth_threshold_{key}", SMOOTH_THRESHOLDS[key]))

    # ---- individual physics terms ------------------------------------------
    def _phi_humidity_deficit(self, xr: torch.Tensor) -> torch.Tensor:
        """phi_1: thermodynamic consistency via humidity deficit (eq. 2-3)."""
        t = _safe_index(xr, self.idx["air_temp"])
        rh = _safe_index(xr, self.idx["rh"])
        if t is None or rh is None:
            return xr.new_tensor(0.0)
        hd = humidity_deficit(t, rh)                 # [B, T]
        # Physical: HD must be >= 0 (Pv <= Psat). Penalise negative HD.
        neg = torch.relu(-hd).mean()
        # Plausibility band for greenhouse/CEA HD (g/m^3), soft upper bound.
        hd_max = float(getattr(self.cfg, "hd_max", 30.0))
        over = torch.relu(hd - hd_max).mean()
        return neg + over

    def _phi_co2_balance(self, xr: torch.Tensor) -> torch.Tensor:
        """
        phi_2: CO2 mass balance as discrete continuity.
        In a closed CEA volume, dCO2/dt is bounded by assimilation + ventilation;
        we penalise implausibly large single-step CO2 changes relative to the
        batch-typical change (a differentiable proxy for mass-balance limits).
        """
        co2 = _safe_index(xr, self.idx["co2"])
        if co2 is None or co2.shape[1] < 2:
            return xr.new_tensor(0.0)
        dco2 = co2[:, 1:] - co2[:, :-1]              # [B, T-1] ppm/step
        max_rate = float(getattr(self.cfg, "co2_max_rate", 120.0))  # ppm/step
        return torch.relu(torch.abs(dco2) - max_rate).mean()

    def _phi_light_thermo(self, xr: torch.Tensor) -> torch.Tensor:
        """
        phi_3: lighting thermodynamics. Radiative input raises sensible heat,
        so air temperature should not DECREASE while light strongly INCREASES
        within the same step (sign-consistency of the radiative coupling).
        Differentiable hinge on the product of (-dT) and (+dLight).
        """
        t = _safe_index(xr, self.idx["air_temp"])
        light = _safe_index(xr, self.idx["light"])
        if t is None or light is None or t.shape[1] < 2:
            return xr.new_tensor(0.0)
        dt = t[:, 1:] - t[:, :-1]
        dl = light[:, 1:] - light[:, :-1]
        # Normalise light delta to a unit-ish scale to keep the term balanced.
        scale = float(getattr(self.cfg, "light_scale", 1000.0))
        coupling = torch.relu(dl / scale) * torch.relu(-dt)  # >0 only if light up & temp down
        return coupling.mean()

    def _phi_irrigation(self, xr: torch.Tensor, conditions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        phi_4: irrigation dynamics. EC and water temperature should change
        smoothly between irrigation events; penalise abrupt EC reversals that
        are physically implausible without a dosing/irrigation action.
        Proxy: second difference (curvature) of EC kept bounded.
        """
        ec = _safe_index(xr, self.idx["ec"])
        if ec is None or ec.shape[1] < 3:
            return xr.new_tensor(0.0)
        d2 = ec[:, 2:] - 2.0 * ec[:, 1:-1] + ec[:, :-2]   # discrete curvature
        max_curv = float(getattr(self.cfg, "ec_max_curvature", 0.3))
        violation = torch.relu(torch.abs(d2) - max_curv)
        if conditions is not None and self.irrigation_condition_indices:
            if conditions.dim() == 2:
                conditions = conditions.unsqueeze(1).expand(-1, xr.shape[1], -1)
            if conditions.dim() != 3 or conditions.shape[1] != xr.shape[1]:
                raise ValueError("conditions must have shape [B,C] or [B,T,C] aligned with x")
            active = conditions[:, :, self.irrigation_condition_indices].amax(dim=-1)
            event_nearby = torch.maximum(active[:, :-2], torch.maximum(active[:, 1:-1], active[:, 2:]))
            violation = violation * (1.0 - event_nearby.clamp(0.0, 1.0))
        return violation.mean()

    def _phi_range(self, xr: torch.Tensor) -> torch.Tensor:
        """phi_5: agronomic range plausibility (kept from original)."""
        loss = xr.new_tensor(0.0)
        count = 0
        for key, (low, high) in self.ranges.items():
            j = self.idx.get(key)
            if j is None:
                continue
            loss = loss + self._range_penalty(xr[:, :, j], low, high)
            count += 1
        if count == 0:
            return xr.new_tensor(0.0)
        return loss / count

    def _phi_smooth(self, xr: torch.Tensor) -> torch.Tensor:
        """phi_6: penalise implausibly large single-step jumps using cfg-calibrated thresholds."""
        if xr.shape[1] < 2:
            return xr.new_tensor(0.0)
        loss = xr.new_tensor(0.0)
        count = 0
        for key in SMOOTH_THRESHOLDS:
            j = self.idx.get(key)
            if j is None:
                continue
            thr = self._smooth_threshold(key)
            jumps = torch.abs(xr[:, 1:, j] - xr[:, :-1, j])
            loss = loss + torch.relu(jumps - float(thr)).mean()
            count += 1
        if count == 0:
            return xr.new_tensor(0.0)
        return loss / count

    # ---- forward ------------------------------------------------------------
    def forward(self, x: torch.Tensor, conditions: Optional[torch.Tensor] = None) -> torch.Tensor:
        xr = self._denormalize(x)
        total = (
            self.w_hd     * self._phi_humidity_deficit(xr)
            + self.w_co2  * self._phi_co2_balance(xr)
            + self.w_light * self._phi_light_thermo(xr)
            + self.w_irrig * self._phi_irrigation(xr, conditions)
            + self.w_range * self._phi_range(xr)
            + self.w_smooth * self._phi_smooth(xr)
        )
        return total

    # ---- optional: per-term breakdown for logging / ablation ---------------
    @torch.no_grad()
    def term_breakdown(self, x: torch.Tensor, conditions: Optional[torch.Tensor] = None) -> Dict[str, float]:
        xr = self._denormalize(x)
        return {
            "phi_humidity_deficit": float(self._phi_humidity_deficit(xr)),
            "phi_co2_balance": float(self._phi_co2_balance(xr)),
            "phi_light_thermo": float(self._phi_light_thermo(xr)),
            "phi_irrigation": float(self._phi_irrigation(xr, conditions)),
            "phi_range": float(self._phi_range(xr)),
            "phi_smooth": float(self._phi_smooth(xr)),
        }


# =============================================================================
# Agronomic Quality Index (AQI) — numpy, on REAL units, 0..1 (higher better)
# =============================================================================

def agronomic_quality_index(
    x_real_unit: np.ndarray,
    feature_names: List[str],
    cfg,
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> float:
    """
    Compute the AQI on REAL-UNIT sequences.

    x_real_unit : [N, T, F] in physical units (already inverse-transformed).
    Returns a float in [0, 1]; 1 = fully plausible.

    AQI = w1 * range_compliance + w2 * thermo_consistency + w3 * smoothness
    """
    idx = _resolve_indices(feature_names)
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("AQI weights must be non-negative and have a positive sum")
    weight_sum = float(sum(weights))
    w_range, w_thermo, w_smooth = (float(w) / weight_sum for w in weights)

    ranges = {
        "air_temp":  (cfg.temp_min, cfg.temp_max),
        "rh":        (cfg.rh_min, cfg.rh_max),
        "co2":       (cfg.co2_min, cfg.co2_max),
        "ph":        (cfg.ph_min, cfg.ph_max),
        "ec":        (cfg.ec_min, cfg.ec_max),
        "water_temp": (cfg.water_temp_min, cfg.water_temp_max),
    }

    # --- (1) range compliance: fraction of in-range timesteps ----------------
    compliances = []
    for key, (low, high) in ranges.items():
        j = idx.get(key)
        if j is None:
            continue
        v = x_real_unit[:, :, j]
        inside = np.mean((v >= low) & (v <= high))
        compliances.append(float(inside))
    range_compliance = float(np.mean(compliances)) if compliances else 1.0

    # --- (2) thermodynamic consistency: finite HD inside a plausible band ----
    ti, ri = idx["air_temp"], idx["rh"]
    if ti is not None and ri is not None:
        t = x_real_unit[:, :, ti]
        rh = x_real_unit[:, :, ri]
        denominator = np.maximum(t + 237.3, 1e-3)
        exponent = np.clip(17.27 * t / denominator, -50.0, 50.0)
        psat = 0.6108 * np.exp(exponent)
        pv = (rh / 100.0) * psat
        hd = 2.16679 * (psat - pv) / np.maximum(t + 273.15, 1.0) * 1000.0
        hd_max = float(getattr(cfg, "hd_max", 30.0))
        thermo_consistency = float(np.mean(np.isfinite(hd) & (hd >= 0.0) & (hd <= hd_max)))
    else:
        thermo_consistency = 1.0

    # --- (3) smoothness: 1 - fraction of implausible jumps -------------------
    # Uses the same fixed physical thresholds as the differentiable loss.
    if x_real_unit.shape[1] >= 2:
        violations = []
        for key, default_thr in SMOOTH_THRESHOLDS.items():
            j = idx.get(key)
            if j is None:
                continue
            thr = float(getattr(cfg, f"smooth_threshold_{key}", default_thr))
            jumps = np.abs(np.diff(x_real_unit[:, :, j], axis=1))
            violations.append(float(np.mean(jumps > float(thr))))
        implausible = float(np.mean(violations)) if violations else 0.0
        smoothness = float(1.0 - implausible)
    else:
        smoothness = 1.0

    aqi = w_range * range_compliance + w_thermo * thermo_consistency + w_smooth * smoothness
    return float(np.clip(aqi, 0.0, 1.0))


# =============================================================================
# Self-test (synthetic) — run `python agronomic_physics.py`
# =============================================================================

if __name__ == "__main__":
    class _Cfg:
        temp_min, temp_max = 16.0, 30.0
        rh_min, rh_max = 45.0, 85.0
        co2_min, co2_max = 350.0, 1200.0
        ph_min, ph_max = 5.5, 6.8
        ec_min, ec_max = 0.8, 2.4
        water_temp_min, water_temp_max = 15.0, 28.0

    cfg = _Cfg()
    names = ["air_temp", "rh", "co2", "ph", "ec", "water_temp", "light_lux"]

    rng = np.random.default_rng(0)
    # Plausible sample
    good = np.stack([
        np.clip(22 + rng.normal(0, 1, (8, 48)), 16, 30),
        np.clip(65 + rng.normal(0, 3, (8, 48)), 45, 85),
        np.clip(600 + rng.normal(0, 20, (8, 48)), 350, 1200),
        np.clip(6.1 + rng.normal(0, 0.05, (8, 48)), 5.5, 6.8),
        np.clip(1.5 + rng.normal(0, 0.05, (8, 48)), 0.8, 2.4),
        np.clip(20 + rng.normal(0, 1, (8, 48)), 15, 28),
        np.clip(12000 + rng.normal(0, 300, (8, 48)), 0, None),
    ], axis=-1)
    print("AQI (plausible):", round(agronomic_quality_index(good, names, cfg), 4))

    # Implausible sample (out of range + noisy)
    bad = good + rng.normal(0, 25, good.shape)
    print("AQI (implausible):", round(agronomic_quality_index(bad, names, cfg), 4))

    # Loss smoke test
    loss = AgronomicConstraintLoss(cfg, names, scaler=None)
    xt = torch.tensor(good, dtype=torch.float32, requires_grad=True)
    val = loss(xt)
    val.backward()
    print("Constraint loss (real-unit input):", float(val.detach()))
    print("Grad is finite:", bool(torch.isfinite(xt.grad).all()))
    print("Breakdown:", {k: round(v, 4) for k, v in loss.term_breakdown(xt).items()})
