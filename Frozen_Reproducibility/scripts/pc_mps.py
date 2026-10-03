"""Numerically stable generative Matrix Product State primitives.

The model is classical PyTorch code. It uses the Born-rule parameterisation
P(x) = |Psi(x)|^2 / Z, but does not require quantum hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Discretizer:
    """Uniform mapping between a continuous physical range and MPS bins."""

    vmin: float
    vmax: float
    bins: int

    def __post_init__(self) -> None:
        if self.bins < 2:
            raise ValueError("bins must be at least 2")
        if not self.vmax > self.vmin:
            raise ValueError("vmax must be greater than vmin")

    @property
    def bin_width(self) -> float:
        return (float(self.vmax) - float(self.vmin)) / int(self.bins)

    def centers(self, device: Optional[torch.device] = None) -> torch.Tensor:
        edges = torch.linspace(
            float(self.vmin),
            float(self.vmax),
            int(self.bins) + 1,
            device=device,
        )
        return 0.5 * (edges[:-1] + edges[1:])

    def to_index(self, values: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(values, dtype=torch.float32)
        scaled = (values - float(self.vmin)) / (float(self.vmax) - float(self.vmin))
        return torch.floor(scaled * int(self.bins)).long().clamp(0, int(self.bins) - 1)

    def to_value(self, indices: torch.Tensor) -> torch.Tensor:
        indices = torch.as_tensor(indices, dtype=torch.long)
        return self.centers(indices.device)[indices]

    def allowed_indices(self, low: float, high: float, device: Optional[torch.device] = None) -> torch.Tensor:
        if high < low:
            raise ValueError("high must be greater than or equal to low")
        centers = self.centers(device)
        allowed = torch.where((centers >= float(low)) & (centers <= float(high)))[0]
        if allowed.numel() == 0:
            midpoint = 0.5 * (float(low) + float(high))
            allowed = torch.argmin((centers - midpoint).abs()).reshape(1)
        return allowed.long()

    def transition_mask(self, max_delta: float, device: Optional[torch.device] = None) -> torch.Tensor:
        centers = self.centers(device)
        return (centers[:, None] - centers[None, :]).abs() > float(max_delta)

    def as_dict(self) -> dict[str, float | int]:
        return {"vmin": float(self.vmin), "vmax": float(self.vmax), "bins": int(self.bins)}


class GenerativeMPS(nn.Module):
    """Open-boundary real MPS with exact density and exact sampling."""

    def __init__(self, length: int, bins: int, bond_dim: int = 8, seed: int = 0) -> None:
        super().__init__()
        if length < 1:
            raise ValueError("length must be positive")
        if bins < 2:
            raise ValueError("bins must be at least 2")
        if bond_dim < 1:
            raise ValueError("bond_dim must be positive")
        self.length = int(length)
        self.bins = int(bins)
        self.bond_dim = int(bond_dim)
        generator = torch.Generator().manual_seed(int(seed))
        cores = []
        for site in range(self.length):
            left = 1 if site == 0 else self.bond_dim
            right = 1 if site == self.length - 1 else self.bond_dim
            scale = 1.0 / max(1.0, float(left * self.bins)) ** 0.5
            cores.append(nn.Parameter(torch.randn(left, self.bins, right, generator=generator) * scale))
        self.cores = nn.ParameterList(cores)

    def _transfer(self, site: int, allowed: Optional[torch.Tensor] = None) -> torch.Tensor:
        core = self.cores[site]
        if allowed is not None:
            allowed = torch.as_tensor(allowed, dtype=torch.long, device=core.device)
            core = core.index_select(1, allowed)
        transfer = torch.einsum("asr,bst->abrt", core, core)
        left, right = core.shape[0], core.shape[2]
        return transfer.reshape(left * left, right * right)

    @staticmethod
    def _rescale(vector: torch.Tensor) -> torch.Tensor:
        scale = vector.detach().abs().amax().clamp_min(1e-30)
        return vector / scale

    def _contract_log_norm(self, allowed_per_site: Optional[Sequence[Optional[torch.Tensor]]] = None) -> torch.Tensor:
        if allowed_per_site is not None and len(allowed_per_site) != self.length:
            raise ValueError("allowed_per_site must have one entry per MPS site")
        allowed0 = None if allowed_per_site is None else allowed_per_site[0]
        vector = self._transfer(0, allowed0)
        log_scale = torch.zeros((), device=vector.device, dtype=vector.dtype)
        scale = vector.detach().abs().amax().clamp_min(1e-30)
        vector = vector / scale
        log_scale = log_scale + scale.log()
        for site in range(1, self.length):
            allowed = None if allowed_per_site is None else allowed_per_site[site]
            vector = vector @ self._transfer(site, allowed)
            scale = vector.detach().abs().amax().clamp_min(1e-30)
            vector = vector / scale
            log_scale = log_scale + scale.log()
        scalar = vector.reshape(-1).sum().clamp_min(1e-30)
        return scalar.log() + log_scale

    def log_norm(self) -> torch.Tensor:
        return self._contract_log_norm()

    @torch.no_grad()
    def log_allowed_mass(
        self,
        allowed_per_site: Sequence[torch.Tensor],
        log_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return log P(all sites lie in their prescribed allowed-bin sets)."""
        denominator = self.log_norm() if log_norm is None else torch.as_tensor(
            log_norm, device=self.cores[0].device, dtype=self.cores[0].dtype
        )
        return (self._contract_log_norm(list(allowed_per_site)) - denominator).clamp(max=0.0)

    @torch.no_grad()
    def site_marginals(self) -> torch.Tensor:
        """Return exact normalized one-site marginals with shape [length, bins]."""
        left = self._left_environments()
        right = self._right_environments()
        rows = []
        for site in range(self.length):
            weights = torch.stack([
                left[site] @ self._transfer(
                    site,
                    torch.tensor([value], device=self.cores[0].device),
                ) @ right[site + 1]
                for value in range(self.bins)
            ]).reshape(self.bins).clamp_min(0.0)
            rows.append(weights / weights.sum().clamp_min(1e-30))
        return torch.stack(rows)

    def _log_abs_amplitude(self, indices: torch.Tensor) -> torch.Tensor:
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.cores[0].device)
        if indices.ndim != 2 or indices.shape[1] != self.length:
            raise ValueError(f"indices must have shape [batch, {self.length}]")
        vector = torch.ones(indices.shape[0], 1, device=indices.device, dtype=self.cores[0].dtype)
        log_scale = torch.zeros(indices.shape[0], device=indices.device, dtype=vector.dtype)
        for site, core in enumerate(self.cores):
            selected = core[:, indices[:, site], :].permute(1, 0, 2)
            vector = torch.bmm(vector.unsqueeze(1), selected).squeeze(1)
            scale = vector.detach().abs().amax(dim=1).clamp_min(1e-30)
            vector = vector / scale[:, None]
            log_scale = log_scale + scale.log()
        return vector.squeeze(-1).abs().clamp_min(1e-30).log() + log_scale

    def amplitude(self, indices: torch.Tensor) -> torch.Tensor:
        """Return amplitudes; use log_prob for long sequences."""
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.cores[0].device)
        vector = torch.ones(indices.shape[0], 1, device=indices.device, dtype=self.cores[0].dtype)
        for site, core in enumerate(self.cores):
            selected = core[:, indices[:, site], :].permute(1, 0, 2)
            vector = torch.bmm(vector.unsqueeze(1), selected).squeeze(1)
        return vector.squeeze(-1)

    def log_prob(self, indices: torch.Tensor) -> torch.Tensor:
        return 2.0 * self._log_abs_amplitude(indices) - self.log_norm()

    def nll(self, indices: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(indices).mean()

    def forbidden_mass_bounds(self, allowed_per_site: Sequence[torch.Tensor]) -> torch.Tensor:
        """Exact probability mass outside the Cartesian product of allowed bins."""
        log_allowed = self._contract_log_norm(list(allowed_per_site))
        log_ratio = (log_allowed - self.log_norm()).clamp(max=0.0)
        return (1.0 - torch.exp(log_ratio)).clamp(0.0, 1.0)

    def allowed_log_barrier(self, allowed_per_site: Sequence[torch.Tensor], per_site: bool = True) -> torch.Tensor:
        """Stable monotonic penalty -log(P(all sites allowed)).

        The raw forbidden mass approaches one exponentially with sequence length,
        which can make its gradient vanish. This exact log-domain transform keeps
        the same optimum while remaining trainable on long windows.
        """
        log_allowed = self._contract_log_norm(list(allowed_per_site))
        penalty = (self.log_norm() - log_allowed).clamp_min(0.0)
        return penalty / self.length if per_site else penalty

    def _left_environments(self) -> list[torch.Tensor]:
        env = [torch.ones(1, device=self.cores[0].device, dtype=self.cores[0].dtype)]
        for site in range(self.length):
            env.append(self._rescale(env[-1] @ self._transfer(site)))
        return env

    def _right_environments(self) -> list[torch.Tensor]:
        env: list[Optional[torch.Tensor]] = [None] * (self.length + 1)
        env[self.length] = torch.ones(1, device=self.cores[0].device, dtype=self.cores[0].dtype)
        for site in range(self.length - 1, -1, -1):
            env[site] = self._rescale(self._transfer(site) @ env[site + 1])
        return [item for item in env if item is not None]

    def _pair_marginal_from_env(
        self,
        site: int,
        left_environments: Sequence[torch.Tensor],
        right_environments: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        left = left_environments[site]
        right = right_environments[site + 2]
        first = self.cores[site]
        second = self.cores[site + 1]
        left_matrix = left.reshape(first.shape[0], first.shape[0])
        right_matrix = right.reshape(second.shape[2], second.shape[2])
        after_first = torch.einsum("uv,uar,vat->art", left_matrix, first, first)
        before_right = torch.einsum("rbs,tbu,su->brt", second, second, right_matrix)
        weights = torch.einsum("art,brt->ab", after_first, before_right).clamp_min(0.0)
        return weights / weights.sum().clamp_min(1e-30)

    def pair_marginal(self, site: int) -> torch.Tensor:
        """Exact normalized marginal P(x_site, x_site+1)."""
        if not 0 <= site < self.length - 1:
            raise ValueError("site must identify an adjacent pair")
        return self._pair_marginal_from_env(
            site,
            self._left_environments(),
            self._right_environments(),
        )

    def forbidden_mass_dynamics(
        self,
        forbidden_mask: torch.Tensor,
        sites: Optional[Iterable[int]] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Exact mass of forbidden adjacent transitions."""
        mask = torch.as_tensor(forbidden_mask, dtype=torch.bool, device=self.cores[0].device)
        if mask.shape != (self.bins, self.bins):
            raise ValueError(f"forbidden_mask must have shape {(self.bins, self.bins)}")
        selected_sites = list(range(self.length - 1) if sites is None else sites)
        if not selected_sites:
            return torch.zeros((), device=self.cores[0].device, dtype=self.cores[0].dtype)
        if any(not 0 <= int(site) < self.length - 1 for site in selected_sites):
            raise ValueError("all transition sites must identify adjacent pairs")
        left = self._left_environments()
        right = self._right_environments()
        masses = torch.stack([
            self._pair_marginal_from_env(int(site), left, right)[mask].sum()
            for site in selected_sites
        ])
        if reduction == "sum":
            return masses.sum()
        if reduction != "mean":
            raise ValueError("reduction must be 'mean' or 'sum'")
        return masses.mean()

    @torch.no_grad()
    def sample(self, count: int) -> torch.Tensor:
        """Draw exact samples with sequential conditional probabilities."""
        if count < 1:
            raise ValueError("count must be positive")
        device = self.cores[0].device
        right: list[Optional[torch.Tensor]] = [None] * (self.length + 1)
        right[self.length] = torch.ones(1, device=device, dtype=self.cores[0].dtype)
        for site in range(self.length - 1, -1, -1):
            right[site] = self._rescale(self._transfer(site) @ right[site + 1])

        result = torch.zeros(count, self.length, dtype=torch.long, device=device)
        left = torch.ones(count, 1, device=device, dtype=self.cores[0].dtype)
        for site, core in enumerate(self.cores):
            right_matrix = right[site + 1].reshape(core.shape[2], core.shape[2])
            probabilities = []
            for value in range(self.bins):
                partial = left @ core[:, value, :]
                probability = torch.einsum("nr,rs,ns->n", partial, right_matrix, partial)
                probabilities.append(probability.clamp_min(0.0))
            probs = torch.stack(probabilities, dim=1)
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-30)
            drawn = torch.multinomial(probs, 1).squeeze(1)
            result[:, site] = drawn
            selected = core[:, drawn, :].permute(1, 0, 2)
            left = torch.bmm(left.unsqueeze(1), selected).squeeze(1)
            left = left / left.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
        return result

    @torch.no_grad()
    def normalize_cores_(self) -> None:
        """Gauge rescaling that leaves the represented probability unchanged."""
        for core in self.cores:
            core.div_(core.norm().clamp_min(1e-12))

    @torch.no_grad()
    def entanglement_entropy(self, cut: int) -> float:
        """Exact von Neumann entropy across a bond, in nats."""
        if not 1 <= cut < self.length:
            raise ValueError("cut must lie strictly inside the MPS")
        left_vector = self._left_environments()[cut]
        right_vector = self._right_environments()[cut]
        bond = self.cores[cut].shape[0]
        left = left_vector.reshape(bond, bond)
        right = right_vector.reshape(bond, bond)
        left = 0.5 * (left + left.T)
        right = 0.5 * (right + right.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(left)
        eigenvalues = eigenvalues.clamp_min(0.0)
        sqrt_left = (eigenvectors * eigenvalues.sqrt().unsqueeze(0)) @ eigenvectors.T
        density = sqrt_left @ right @ sqrt_left
        density = 0.5 * (density + density.T)
        probabilities = torch.linalg.eigvalsh(density).clamp_min(0.0)
        probabilities = probabilities / probabilities.sum().clamp_min(1e-30)
        probabilities = probabilities[probabilities > 1e-12]
        return float(-(probabilities * probabilities.log()).sum().cpu())
