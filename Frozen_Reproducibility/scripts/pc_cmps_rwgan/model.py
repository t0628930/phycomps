"""Core modules for the PC-cMoMPS-RWGAN research prototype.

PC-cMoMPS-RWGAN stands for Physics-Constrained conditional Mixture-of-MPS
Residual Wasserstein GAN. The model is fully classical:

1. a condition gate selects a mixture of exact-density MPS experts;
2. the selected MPS generates a joint discrete trajectory over time/features;
3. a conditional residual decoder maps bin centres to continuous trajectories;
4. code consistency prevents the decoder from ignoring the MPS trajectory.

Sampling from the MPS is discrete, so the MPS and decoder objectives are
optimized alternately rather than pretending that adversarial gradients pass
through the sampled tokens.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from pc_mps import GenerativeMPS


class ConditionGate(nn.Module):
    """Encode an actuator trajectory into mixture weights."""

    def __init__(
        self,
        condition_dim: int,
        experts: int,
        hidden_dim: int = 32,
        gate_constant: bool = False,
    ) -> None:
        super().__init__()
        if condition_dim < 1 or experts < 1:
            raise ValueError("condition_dim and experts must be positive")
        self.condition_dim = int(condition_dim)
        self.experts = int(experts)
        self.gate_constant = bool(gate_constant)
        if self.gate_constant:
            # Learned global mixture: experts remain available, but actuator
            # trajectories cannot affect their probabilities.
            self.constant_logits = nn.Parameter(torch.zeros(experts))
            self.encoder = None
            self.head = None
        else:
            self.register_parameter("constant_logits", None)
            self.encoder = nn.GRU(condition_dim, hidden_dim, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, experts),
            )

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        if conditions.ndim == 2:
            conditions = conditions.unsqueeze(1)
        if conditions.ndim != 3 or conditions.shape[-1] != self.condition_dim:
            raise ValueError(
                f"conditions must have shape [B,T,{self.condition_dim}] or [B,{self.condition_dim}]"
            )
        if self.gate_constant:
            return self.constant_logits.unsqueeze(0).expand(conditions.shape[0], -1)
        encoded, _ = self.encoder(conditions)
        return self.head(encoded[:, -1])


class ConditionalMPSMixture(nn.Module):
    """Conditional mixture of Born-rule MPS experts with exact token density."""

    def __init__(
        self,
        length: int,
        bins: int,
        condition_dim: int,
        experts: int = 3,
        bond_dim: int = 4,
        gate_hidden_dim: int = 32,
        gate_constant: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if experts < 1:
            raise ValueError("experts must be positive")
        self.length = int(length)
        self.bins = int(bins)
        self.expert_count = int(experts)
        self.gate = ConditionGate(
            condition_dim,
            experts,
            gate_hidden_dim,
            gate_constant=gate_constant,
        )
        self.experts = nn.ModuleList([
            GenerativeMPS(length, bins, bond_dim=bond_dim, seed=seed + index)
            for index in range(experts)
        ])

    def gate_log_probs(self, conditions: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.gate(conditions), dim=-1)

    def expert_log_norms(self) -> torch.Tensor:
        """Exact per-expert log normalizers, reusable within one objective."""
        return torch.stack([expert.log_norm() for expert in self.experts], dim=0)

    def expert_log_probs(
        self,
        indices: torch.Tensor,
        log_norms: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if indices.ndim != 2 or indices.shape[1] != self.length:
            raise ValueError(f"indices must have shape [B,{self.length}]")
        if log_norms is None:
            return torch.stack([expert.log_prob(indices) for expert in self.experts], dim=-1)
        return torch.stack([
            expert.log_prob(indices, log_norm=log_norms[index])
            for index, expert in enumerate(self.experts)
        ], dim=-1)

    def log_prob(
        self,
        indices: torch.Tensor,
        conditions: torch.Tensor,
        log_norms: Optional[torch.Tensor] = None,
        gate_log_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if gate_log_probs is None:
            gate_log_probs = self.gate_log_probs(conditions)
        return torch.logsumexp(
            gate_log_probs + self.expert_log_probs(indices, log_norms=log_norms),
            dim=-1,
        )

    def nll(
        self,
        indices: torch.Tensor,
        conditions: torch.Tensor,
        per_site: bool = True,
        log_norms: Optional[torch.Tensor] = None,
        gate_log_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        value = -self.log_prob(
            indices, conditions, log_norms=log_norms, gate_log_probs=gate_log_probs
        ).mean()
        return value / self.length if per_site else value

    def _expert_log_allowed(
        self,
        allowed_per_site: Sequence[torch.Tensor],
        log_norms: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if log_norms is None:
            return torch.stack([
                -expert.allowed_log_barrier(allowed_per_site, per_site=False)
                for expert in self.experts
            ])
        return torch.stack([
            -expert.allowed_log_barrier(
                allowed_per_site, per_site=False, log_norm=log_norms[index]
            )
            for index, expert in enumerate(self.experts)
        ])

    def allowed_log_barrier(
        self,
        allowed_per_site: Sequence[torch.Tensor],
        conditions: torch.Tensor,
        per_site: bool = True,
        log_norms: Optional[torch.Tensor] = None,
        gate_log_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Exact -log P(all sites allowed | condition)."""
        if gate_log_probs is None:
            gate_log_probs = self.gate_log_probs(conditions)
        expert_log_allowed = self._expert_log_allowed(
            allowed_per_site, log_norms=log_norms
        )
        log_probability = torch.logsumexp(
            gate_log_probs + expert_log_allowed.unsqueeze(0),
            dim=-1,
        )
        barrier = -log_probability.mean()
        return barrier / self.length if per_site else barrier

    def nll_and_allowed_log_barrier(
        self,
        indices: torch.Tensor,
        allowed_per_site: Sequence[torch.Tensor],
        conditions: torch.Tensor,
        per_site: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute prior NLL and exact box barrier with shared caches.

        This is the fast path used by training. It contracts each expert norm
        once, reuses the gate log-probabilities once, and leaves the public
        ``nll`` and ``allowed_log_barrier`` methods backwards compatible.
        """
        gate_log_probs = self.gate_log_probs(conditions)
        log_norms = self.expert_log_norms()
        nll = self.nll(
            indices,
            conditions,
            per_site=per_site,
            log_norms=log_norms,
            gate_log_probs=gate_log_probs,
        )
        barrier = self.allowed_log_barrier(
            allowed_per_site,
            conditions,
            per_site=per_site,
            log_norms=log_norms,
            gate_log_probs=gate_log_probs,
        )
        return nll, barrier

    def gate_regularization(self, conditions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return batch-balance KL and per-sample gate entropy."""
        probabilities = self.gate_log_probs(conditions).exp()
        average = probabilities.mean(dim=0).clamp_min(1e-12)
        uniform_log = -math.log(self.expert_count)
        balance_kl = torch.sum(average * (average.log() - uniform_log))
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1).mean()
        return balance_kl, entropy

    @torch.no_grad()
    def sample(
        self,
        conditions: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one expert per condition and then sample that expert exactly."""
        probabilities = self.gate_log_probs(conditions).exp()
        expert_ids = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
        output = torch.empty(
            len(conditions), self.length, dtype=torch.long, device=conditions.device
        )
        for expert_index, expert in enumerate(self.experts):
            selected = torch.where(expert_ids == expert_index)[0]
            if selected.numel() == 0:
                continue
            output[selected] = expert.sample(int(selected.numel()))
        return output, expert_ids

    @torch.no_grad()
    def normalize_cores_(self) -> None:
        for expert in self.experts:
            expert.normalize_cores_()


class ResidualTrajectoryDecoder(nn.Module):
    """Decode joint MPS tokens into bounded continuous sensor trajectories."""

    def __init__(
        self,
        seq_len: int,
        feature_dim: int,
        condition_dim: int,
        bins: int,
        centers_normalized: torch.Tensor,
        lower_bounds_normalized: torch.Tensor,
        upper_bounds_normalized: torch.Tensor,
        embedding_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
        noise_dim: int = 8,
        dropout: float = 0.1,
        residual_scale: float = 0.75,
        clip_beta: float = 6.0,
        non_residual: bool = False,
    ) -> None:
        super().__init__()
        if centers_normalized.shape != (feature_dim, bins):
            raise ValueError(f"centers_normalized must have shape {(feature_dim, bins)}")
        if lower_bounds_normalized.shape != (feature_dim,):
            raise ValueError("lower_bounds_normalized has the wrong shape")
        if upper_bounds_normalized.shape != (feature_dim,):
            raise ValueError("upper_bounds_normalized has the wrong shape")
        if torch.any(upper_bounds_normalized <= lower_bounds_normalized):
            raise ValueError("every upper bound must be greater than its lower bound")

        self.seq_len = int(seq_len)
        self.feature_dim = int(feature_dim)
        self.condition_dim = int(condition_dim)
        self.bins = int(bins)
        self.noise_dim = int(noise_dim)
        self.residual_scale = float(residual_scale)
        self.clip_beta = float(clip_beta)
        self.non_residual = bool(non_residual)
        self.embeddings = nn.ModuleList([
            nn.Embedding(bins, embedding_dim) for _ in range(feature_dim)
        ])
        input_dim = feature_dim * embedding_dim + feature_dim + condition_dim + noise_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.rnn = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.register_buffer("centers_normalized", centers_normalized.float())
        self.register_buffer("lower_bounds_normalized", lower_bounds_normalized.float())
        self.register_buffer("upper_bounds_normalized", upper_bounds_normalized.float())

    def reshape_tokens(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim == 2:
            expected = self.seq_len * self.feature_dim
            if indices.shape[1] != expected:
                raise ValueError(f"flat indices must have length {expected}")
            return indices.reshape(indices.shape[0], self.seq_len, self.feature_dim)
        if indices.ndim == 3 and indices.shape[1:] == (self.seq_len, self.feature_dim):
            return indices
        raise ValueError("indices must be flat [B,T*F] or structured [B,T,F]")

    def tokens_to_centers(self, indices: torch.Tensor) -> torch.Tensor:
        structured = self.reshape_tokens(indices)
        channels = [
            self.centers_normalized[feature].index_select(0, structured[:, :, feature].reshape(-1))
            .reshape(structured.shape[0], self.seq_len)
            for feature in range(self.feature_dim)
        ]
        return torch.stack(channels, dim=-1)

    def _smooth_bound(self, raw: torch.Tensor) -> torch.Tensor:
        lower = self.lower_bounds_normalized.view(1, 1, -1)
        upper = self.upper_bounds_normalized.view(1, 1, -1)
        beta = self.clip_beta
        return lower + F.softplus(raw - lower, beta=beta) - F.softplus(raw - upper, beta=beta)

    def forward(
        self,
        indices: torch.Tensor,
        conditions: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        structured = self.reshape_tokens(indices)
        if conditions.ndim == 2:
            conditions = conditions.unsqueeze(1).expand(-1, self.seq_len, -1)
        if conditions.shape[1:] != (self.seq_len, self.condition_dim):
            raise ValueError(
                f"conditions must have shape [B,{self.seq_len},{self.condition_dim}]"
            )
        embeddings = torch.cat([
            embedding(structured[:, :, feature])
            for feature, embedding in enumerate(self.embeddings)
        ], dim=-1)
        coarse = self.tokens_to_centers(structured)
        if noise is None:
            noise = torch.randn(
                structured.shape[0], self.seq_len, self.noise_dim,
                device=structured.device, dtype=coarse.dtype,
            )
        if noise.shape != (structured.shape[0], self.seq_len, self.noise_dim):
            raise ValueError("noise has the wrong shape")
        hidden = self.input_projection(torch.cat([embeddings, coarse, conditions, noise], dim=-1))
        hidden, _ = self.rnn(hidden)
        decoded = self.output(hidden)
        if self.non_residual:
            return self._smooth_bound(decoded)
        residual = self.residual_scale * torch.tanh(decoded)
        return self._smooth_bound(coarse + residual)

    def code_logits(self, generated: torch.Tensor, temperature: float = 0.25) -> torch.Tensor:
        if generated.shape[1:] != (self.seq_len, self.feature_dim):
            raise ValueError("generated has the wrong shape")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        distance = generated.unsqueeze(-1) - self.centers_normalized.view(
            1, 1, self.feature_dim, self.bins
        )
        return -(distance / float(temperature)).square()

    def code_consistency_loss(
        self,
        generated: torch.Tensor,
        source_indices: torch.Tensor,
        temperature: float = 0.25,
    ) -> torch.Tensor:
        targets = self.reshape_tokens(source_indices)
        logits = self.code_logits(generated, temperature)
        return F.cross_entropy(logits.reshape(-1, self.bins), targets.reshape(-1))


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
