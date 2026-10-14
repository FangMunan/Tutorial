#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SNNTransformBlock-v1 — standalone engineering interface.

This file is intentionally self-contained.  It depends only on PyTorch and the
Python standard library; it does not import any experiment/U-stage scripts.

Canonical v1 operators
----------------------
* event-cached Q/K/V token projections
* positive-RF normalized linear attention with persistent H/g state
* optional fixed per-head Izhikevich membrane-trajectory feature interface
* structurally sparse output projection
* structurally sparse FFN with exact-zero thresholded hidden activity
* deterministic state carry/reset

Mamba-like input-dependent selective retention is NOT part of v1.  The block
uses fixed retention_alpha=1 by default, while allowing a caller-supplied fixed
retention scalar/tensor for controlled experiments.

Minimal API
-----------
    block = SNNTransformBlock(...)
    state = None
    y, state = block(x, state)          # x: [B, N, D]
    y, state = block(x_next, state)     # persistent execution
    y, state = block(x_new, state, reset=True)

Load the frozen U5 checkpoint
-----------------------------
    block, checkpoint = load_frozen_v1("SNNTransformBlock_v1.pt")
    y, state = block(x, None)

The checkpoint loader installs the saved per-head Izh interface before loading
weights, so downstream applications do not need to reconstruct calibration.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
Retention = Union[float, Tensor, None]


@dataclass(frozen=True)
class BlockConfig:
    d: int = 32
    heads: int = 2
    features: int = 32
    ff: int = 64
    out_density: float = 0.25
    ff1_density: float = 0.25
    ff2_density: float = 0.25
    theta_init: float = 0.0
    retention_alpha: float = 1.0
    seed: int = 17

    def validate(self) -> None:
        if self.d <= 0 or self.heads <= 0 or self.features <= 0 or self.ff <= 0:
            raise ValueError("d, heads, features and ff must be positive")
        if self.d % self.heads != 0:
            raise ValueError("d must be divisible by heads")
        for name in ("out_density", "ff1_density", "ff2_density"):
            value = float(getattr(self, name))
            if not (0.0 < value <= 1.0):
                raise ValueError(f"{name} must be in (0, 1]")


@dataclass
class SNNTransformState:
    """Persistent state carried between calls.

    H, g       : normalized linear-attention associative state
    pk, v      : cached K feature and V per token
    weight     : retained contribution weight per token/head
    pq         : cached Q feature per token
    x_cache    : previous raw input, used only when changed-mask is inferred
    """

    H: Tensor
    g: Tensor
    pk: Tensor
    v: Tensor
    weight: Tensor
    pq: Tensor
    x_cache: Tensor

    def detach(self) -> "SNNTransformState":
        return SNNTransformState(
            H=self.H.detach(),
            g=self.g.detach(),
            pk=self.pk.detach(),
            v=self.v.detach(),
            weight=self.weight.detach(),
            pq=self.pq.detach(),
            x_cache=self.x_cache.detach(),
        )

    def to(self, *args, **kwargs) -> "SNNTransformState":
        return SNNTransformState(
            H=self.H.to(*args, **kwargs),
            g=self.g.to(*args, **kwargs),
            pk=self.pk.to(*args, **kwargs),
            v=self.v.to(*args, **kwargs),
            weight=self.weight.to(*args, **kwargs),
            pq=self.pq.to(*args, **kwargs),
            x_cache=self.x_cache.to(*args, **kwargs),
        )


class PositiveRF(nn.Module):
    """Positive random-feature map used as the exact/reference interface."""

    def __init__(self, heads: int, head_dim: int, features: int, seed: int = 17):
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.features = int(features)
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        omega = torch.randn(self.heads, self.features, self.head_dim, generator=g)
        omega = omega / omega.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        omega = omega * math.sqrt(self.head_dim)
        self.register_buffer("omega", omega)

    def stable_a(self, x: Tensor, is_query: bool) -> Tensor:
        xx = x * (self.head_dim ** -0.25)
        dash = torch.einsum("bhnd,hmd->bhnm", xx, self.omega)
        diag = 0.5 * (xx * xx).sum(-1, keepdim=True)
        a = dash - diag
        if is_query:
            return (a - a.amax(-1, keepdim=True).detach()).clamp(-30.0, 0.0)
        return a.clamp(-30.0, 30.0)

    def forward(self, x: Tensor, *, is_query: bool) -> Tensor:
        return torch.exp(self.stable_a(x, is_query)) * (self.features ** -0.5) + 1e-6


@torch.no_grad()
def izh_voltage(a: Tensor, bias: float, gain: float, window_ms: int) -> Tensor:
    """Izhikevich RS membrane trajectory used by the frozen neuronal interface."""

    x = a.float()
    v = torch.full_like(x, -65.0)
    u = 0.2 * v
    current = float(bias) + float(gain) * x
    trajectory = []

    for _ in range(int(window_ms)):
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current
        v = v + 0.5 * dv
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current
        v = v + 0.5 * dv
        u = u + 0.02 * (0.2 * v - u)
        fired = v >= 30.0
        v = torch.where(fired, torch.full_like(v, -65.0), v)
        u = torch.where(fired, u + 8.0, u)
        trajectory.append(v)

    return torch.stack(trajectory, dim=-1)


@torch.no_grad()
def trajectory_design(a: Tensor, bias: float, gain: float, window_ms: int) -> Tensor:
    v = izh_voltage(a, bias, gain, window_ms)
    one = torch.ones(*a.shape, 1, device=a.device, dtype=v.dtype)
    return torch.cat([one, v], dim=-1)


class PerHeadIzhPhi(nn.Module):
    """Fixed per-head Izh RS trajectory -> linear-decoder feature map."""

    def __init__(
        self,
        exact: PositiveRF,
        specs: Sequence[Dict[str, Any]],
        decoders: Sequence[Tensor],
    ):
        super().__init__()
        if len(specs) != exact.heads or len(decoders) != exact.heads:
            raise ValueError("one Izh spec and one decoder are required per attention head")

        self.head_dim = exact.head_dim
        self.features = exact.features
        self.heads = exact.heads
        self.register_buffer("omega", exact.omega.detach().clone())
        self.specs = [dict(s) for s in specs]
        for h, decoder in enumerate(decoders):
            self.register_buffer(f"decoder_{h}", decoder.detach().clone())
        self.last_clamp_fraction = 0.0

    def stable_a(self, x: Tensor, is_query: bool) -> Tensor:
        xx = x * (self.head_dim ** -0.25)
        dash = torch.einsum("bhnd,hmd->bhnm", xx, self.omega)
        diag = 0.5 * (xx * xx).sum(-1, keepdim=True)
        a = dash - diag
        if is_query:
            return (a - a.amax(-1, keepdim=True).detach()).clamp(-30.0, 0.0)
        return a.clamp(-30.0, 30.0)

    def forward(self, x: Tensor, *, is_query: bool) -> Tensor:
        a = self.stable_a(x, is_query)
        outputs = []
        clamp_sum = 0.0

        for h, spec in enumerate(self.specs):
            ah = a[:, h]
            design = trajectory_design(ah.reshape(-1), **spec)
            decoder = getattr(self, f"decoder_{h}")
            raw = design @ decoder.to(dtype=design.dtype, device=design.device)
            clamp_sum += float((raw <= 1e-8).float().mean().detach())
            outputs.append(raw.clamp_min(1e-8).reshape_as(ah))

        self.last_clamp_fraction = clamp_sum / max(1, self.heads)
        return torch.stack(outputs, dim=1)


class SparseLinear(nn.Module):
    """Linear layer with a persistent structural connectivity mask."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        density: float,
        *,
        bias: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None

        k = max(1, min(self.in_features, int(round(float(density) * self.in_features))))
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        mask = torch.zeros(self.out_features, self.in_features)
        for o in range(self.out_features):
            idx = torch.randperm(self.in_features, generator=g)[:k]
            mask[o, idx] = 1.0
        self.register_buffer("mask", mask)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5.0))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)

    @property
    def nnz(self) -> int:
        return int(self.mask.sum().item())

    @property
    def density(self) -> float:
        return float(self.mask.mean().item())


class EventCachedLinearAttention(nn.Module):
    """Persistent normalized H/g linear attention with token-level Q/K/V cache."""

    def __init__(
        self,
        d: int,
        heads: int,
        features: int,
        out_density: float,
        seed: int,
    ):
        super().__init__()
        if d % heads != 0:
            raise ValueError("d must be divisible by heads")

        self.d = int(d)
        self.heads = int(heads)
        self.head_dim = self.d // self.heads
        self.features = int(features)

        self.q = nn.Linear(self.d, self.d)
        self.k = nn.Linear(self.d, self.d)
        self.v = nn.Linear(self.d, self.d)
        self.out = SparseLinear(self.d, self.d, out_density, bias=True, seed=seed + 101)
        self.phi_q: nn.Module = PositiveRF(self.heads, self.head_dim, self.features, seed)
        self.phi_k: nn.Module = PositiveRF(self.heads, self.head_dim, self.features, seed)

    def _split(self, x: Tensor) -> Tensor:
        B, N, D = x.shape
        return x.reshape(B, N, self.heads, self.head_dim).transpose(1, 2)

    def _project_q_all(self, x: Tensor) -> Tensor:
        return self.phi_q(self._split(self.q(x)), is_query=True)

    def _project_kv_all(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        k = self._split(self.k(x))
        v = self._split(self.v(x))
        return self.phi_k(k, is_query=False), v

    @staticmethod
    def _retention_tensor(retention: Retention, B: int, H: int, device, dtype) -> Tensor:
        if retention is None:
            return torch.ones(B, H, device=device, dtype=dtype)
        if torch.is_tensor(retention):
            alpha = retention.to(device=device, dtype=dtype)
            if alpha.ndim == 0:
                return alpha.expand(B, H)
            if alpha.shape == (H,):
                return alpha[None].expand(B, H)
            if alpha.shape == (B, H):
                return alpha
            raise ValueError("retention tensor must be scalar, [H], or [B,H]")
        return torch.full((B, H), float(retention), device=device, dtype=dtype)

    def step(
        self,
        x: Tensor,
        state: Optional[SNNTransformState],
        changed: Tensor,
        *,
        retention: Retention = None,
        force_project_all: bool = False,
        raw_x_for_cache: Optional[Tensor] = None,
    ) -> Tuple[Tensor, SNNTransformState, Dict[str, Any]]:
        B, N, _ = x.shape
        alpha = self._retention_tensor(retention, B, self.heads, x.device, x.dtype)
        q_evals = k_evals = v_evals = 0

        if state is None:
            pq = self._project_q_all(x)
            pk, value = self._project_kv_all(x)
            H = torch.einsum("bhnm,bhnd->bhmd", pk, value)
            g = pk.sum(dim=2)
            weight = torch.ones(B, self.heads, N, device=x.device, dtype=x.dtype)
            q_evals = k_evals = v_evals = B * N
        else:
            H = state.H * alpha[:, :, None, None]
            g = state.g * alpha[:, :, None]
            weight = state.weight * alpha[:, :, None]
            pq = state.pq.clone()
            pk = state.pk.clone()
            value = state.v.clone()

            if force_project_all:
                pq_full = self._project_q_all(x)
                pk_full, value_full = self._project_kv_all(x)
                q_evals = k_evals = v_evals = B * N
                pq = pq_full
            else:
                pq_full = pk_full = value_full = None

            for b in range(B):
                idx = torch.nonzero(changed[b], as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                n_changed = int(idx.numel())

                if force_project_all:
                    new_pq = pq_full[b:b + 1, :, idx, :]
                    new_pk = pk_full[b:b + 1, :, idx, :]
                    new_v = value_full[b:b + 1, :, idx, :]
                else:
                    xb = x[b:b + 1, idx]
                    new_pq = self.phi_q(self._split(self.q(xb)), is_query=True)
                    new_k = self._split(self.k(xb))
                    new_v = self._split(self.v(xb))
                    new_pk = self.phi_k(new_k, is_query=False)
                    q_evals += n_changed
                    k_evals += n_changed
                    v_evals += n_changed

                old_pk = state.pk[b:b + 1, :, idx, :]
                old_v = state.v[b:b + 1, :, idx, :]
                old_weight = weight[b:b + 1, :, idx]

                H = H - torch.einsum("bhn,bhnm,bhnd->bhmd", old_weight, old_pk, old_v)
                H = H + torch.einsum("bhnm,bhnd->bhmd", new_pk, new_v)
                g = g - (old_pk * old_weight[:, :, :, None]).sum(dim=2) + new_pk.sum(dim=2)

                pq[b, :, idx, :] = new_pq[0]
                pk[b, :, idx, :] = new_pk[0]
                value[b, :, idx, :] = new_v[0]
                weight[b, :, idx] = 1.0

            if force_project_all:
                pk = pk_full
                value = value_full

        numerator = torch.einsum("bhnm,bhmd->bhnd", pq, H)
        denominator = torch.einsum("bhnm,bhm->bhn", pq, g).unsqueeze(-1).clamp_min(1e-8)
        attended = numerator / denominator
        output = self.out(attended.transpose(1, 2).reshape(B, N, self.d))

        cache = raw_x_for_cache if raw_x_for_cache is not None else x
        new_state = SNNTransformState(
            H=H,
            g=g,
            pk=pk,
            v=value,
            weight=weight,
            pq=pq,
            x_cache=cache.detach().clone(),
        )
        stats = {
            "q_token_evals": q_evals,
            "k_token_evals": k_evals,
            "v_token_evals": v_evals,
            "changed_tokens": int(changed.sum().item()),
            "total_tokens": B * N,
            "retention_mean": float(alpha.mean().detach()),
        }
        return output, new_state, stats


class SNNTransformBlock(nn.Module):
    """Canonical reusable SNN-Transform block v1.

    Input shape:  [B, N, D]
    Output shape: [B, N, D]

    Main call:
        y, state = block(x, state)

    If ``changed`` is omitted, changed tokens are inferred by exact comparison
    against the previous raw input cache.  If an upstream event source already
    knows which tokens changed, passing ``changed=[B,N]`` is preferable.
    """

    VERSION = "SNNTransformBlock-v1"

    def __init__(
        self,
        d: int = 32,
        heads: int = 2,
        features: int = 32,
        ff: int = 64,
        out_density: float = 0.25,
        ff1_density: float = 0.25,
        ff2_density: float = 0.25,
        theta_init: float = 0.0,
        retention_alpha: float = 1.0,
        seed: int = 17,
    ):
        super().__init__()
        self.config = BlockConfig(
            d=d,
            heads=heads,
            features=features,
            ff=ff,
            out_density=out_density,
            ff1_density=ff1_density,
            ff2_density=ff2_density,
            theta_init=theta_init,
            retention_alpha=retention_alpha,
            seed=seed,
        )
        self.config.validate()

        self.d = d
        self.heads = heads
        self.ff = ff
        self.retention_alpha = float(retention_alpha)
        self.n1 = nn.LayerNorm(d)
        self.attn = EventCachedLinearAttention(d, heads, features, out_density, seed)
        self.n2 = nn.LayerNorm(d)
        self.f1 = SparseLinear(d, ff, ff1_density, bias=True, seed=seed + 201)
        self.f2 = SparseLinear(ff, d, ff2_density, bias=True, seed=seed + 301)
        self.theta = nn.Parameter(torch.full((ff,), float(theta_init)))
        self.last_stats: Dict[str, Any] = {}

    def set_izh_interface(
        self,
        q_specs: Sequence[Dict[str, Any]],
        q_decoders: Sequence[Tensor],
        k_specs: Sequence[Dict[str, Any]],
        k_decoders: Sequence[Tensor],
    ) -> None:
        """Replace exact PositiveRF Q/K interfaces with fixed per-head Izh interfaces."""

        q_exact = self.attn.phi_q
        k_exact = self.attn.phi_k
        if not isinstance(q_exact, PositiveRF) or not isinstance(k_exact, PositiveRF):
            raise TypeError("Izh interface is already installed or exact PositiveRF is unavailable")
        self.attn.phi_q = PerHeadIzhPhi(q_exact, q_specs, q_decoders)
        self.attn.phi_k = PerHeadIzhPhi(k_exact, k_specs, k_decoders)

    def infer_changed(self, x: Tensor, state: Optional[SNNTransformState]) -> Tensor:
        if state is None:
            return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if state.x_cache.shape != x.shape:
            raise ValueError("state/input shape mismatch; reset state when batch/token shape changes")
        return torch.any(x != state.x_cache, dim=-1)

    def step(
        self,
        x: Tensor,
        state: Optional[SNNTransformState] = None,
        changed: Optional[Tensor] = None,
        *,
        reset: bool = False,
        retention: Retention = None,
        force_project_all: bool = False,
    ) -> Tuple[Tensor, SNNTransformState, Dict[str, Any]]:
        if x.ndim != 3 or x.shape[-1] != self.d:
            raise ValueError(f"x must have shape [B,N,{self.d}]")
        if reset:
            state = None

        if changed is None:
            changed = self.infer_changed(x, state)
        else:
            changed = changed.to(device=x.device, dtype=torch.bool)
            if changed.shape != x.shape[:2]:
                raise ValueError("changed must have shape [B,N]")

        if retention is None:
            retention = self.retention_alpha

        attention, new_state, stats = self.attn.step(
            self.n1(x),
            state,
            changed,
            retention=retention,
            force_project_all=force_project_all,
            raw_x_for_cache=x,
        )
        z = x + attention
        pre = self.f1(self.n2(z))
        hidden = F.relu(pre - self.theta)
        y = z + self.f2(hidden)

        hidden_activity = float((hidden > 0).float().mean().detach())
        stats.update(
            {
                "hidden_activity": hidden_activity,
                "output_density": self.attn.out.density,
                "ff1_density": self.f1.density,
                "ff2_density": self.f2.density,
            }
        )
        self.last_stats = stats
        return y, new_state, stats

    def forward(
        self,
        x: Tensor,
        state: Optional[SNNTransformState] = None,
        changed: Optional[Tensor] = None,
        *,
        reset: bool = False,
        retention: Retention = None,
    ) -> Tuple[Tensor, SNNTransformState]:
        y, new_state, stats = self.step(
            x,
            state,
            changed,
            reset=reset,
            retention=retention,
        )
        self.last_stats = stats
        return y, new_state

    @staticmethod
    def reset_state() -> None:
        """The canonical reset state is simply ``None``."""
        return None

    def synapse_account(self, hidden_activity: Optional[float] = None) -> Dict[str, float]:
        if hidden_activity is None:
            hidden_activity = float(self.last_stats.get("hidden_activity", 1.0))
        dense = self.d * self.d + self.d * self.ff + self.ff * self.d
        static = self.attn.out.nnz + self.f1.nnz + self.f2.nnz
        effective = self.attn.out.nnz + self.f1.nnz + float(hidden_activity) * self.f2.nnz
        return {
            "dense_tail_synapses": float(dense),
            "static_sparse_synapses": float(static),
            "static_fraction": float(static / dense),
            "effective_synops_proxy": float(effective),
            "effective_fraction": float(effective / dense),
            "effective_reduction": float(1.0 - effective / dense),
        }

    def config_dict(self) -> Dict[str, Any]:
        return asdict(self.config)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        *,
        map_location: Union[str, torch.device] = "cpu",
        eval_mode: bool = True,
    ) -> Tuple["SNNTransformBlock", Dict[str, Any]]:
        """Load the frozen U5 ``SNNTransformBlock_v1.pt`` checkpoint.

        Returns ``(block, checkpoint_dict)`` so a downstream application can
        also inspect scientific metadata/readout weights if desired.
        """

        path = Path(checkpoint_path)
        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:  # compatibility with older PyTorch
            checkpoint = torch.load(path, map_location=map_location)

        if checkpoint.get("format") != cls.VERSION:
            raise ValueError(
                f"expected checkpoint format {cls.VERSION!r}, got {checkpoint.get('format')!r}"
            )

        config = dict(checkpoint["config"])
        block = cls(**config)
        block.set_izh_interface(
            checkpoint["q_specs"],
            checkpoint["q_decoders"],
            checkpoint["k_specs"],
            checkpoint["k_decoders"],
        )
        block.load_state_dict(checkpoint["model_state_dict"], strict=True)
        block.to(map_location)
        if eval_mode:
            block.eval()
        return block, checkpoint


def load_frozen_v1(
    checkpoint_path: Union[str, Path],
    *,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[SNNTransformBlock, Dict[str, Any]]:
    """Convenience loader for the canonical frozen U5 checkpoint."""
    return SNNTransformBlock.from_checkpoint(checkpoint_path, map_location=device, eval_mode=True)


def _smoke_test() -> None:
    torch.manual_seed(1)
    block = SNNTransformBlock(d=32, heads=2, features=32, ff=64)
    x0 = torch.randn(2, 16, 32)
    y0, state = block(x0, None)

    x1 = x0.clone()
    x1[:, 0] += 0.1
    y1, state = block(x1, state)

    print("SNNTransformBlock-v1 standalone smoke test PASS")
    print("y0:", tuple(y0.shape), "y1:", tuple(y1.shape))
    print("changed tokens:", block.last_stats["changed_tokens"])
    print("tail accounting:", block.synapse_account())


if __name__ == "__main__":
    _smoke_test()
