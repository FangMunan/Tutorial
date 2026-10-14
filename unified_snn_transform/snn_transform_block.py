#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical lightweight SNN-Transform block v1.

Accepted operators assembled behind one reusable API:
  * event-cached Q/K/V projection execution (including Q),
  * positive-RF normalized linear attention with persistent H/g state,
  * optional fixed per-head Izhikevich trajectory feature interface for Q/K,
  * 25%-style structural sparse output projection and FFN,
  * exact-zero thresholded FFN hidden population,
  * deterministic state carry/reset.

Mamba-like selective retention is intentionally NOT part of v1.  A caller may
supply a fixed retention scalar/tensor to step(); input-dependent selective
retention remains a parked future module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositiveRF(nn.Module):
    def __init__(self, heads: int, head_dim: int, features: int, seed: int = 17):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.features = features
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        omega = torch.randn(heads, features, head_dim, generator=g)
        omega = omega / omega.norm(dim=-1, keepdim=True).clamp_min(1e-8) * math.sqrt(head_dim)
        self.register_buffer("omega", omega)

    def stable_a(self, x: torch.Tensor, is_query: bool) -> torch.Tensor:
        xx = x * (self.head_dim ** -0.25)
        dash = torch.einsum("bhnd,hmd->bhnm", xx, self.omega)
        diag = 0.5 * (xx * xx).sum(-1, keepdim=True)
        a = dash - diag
        if is_query:
            return (a - a.amax(-1, keepdim=True).detach()).clamp(-30.0, 0.0)
        return a.clamp(-30.0, 30.0)

    def forward(self, x: torch.Tensor, *, is_query: bool) -> torch.Tensor:
        a = self.stable_a(x, is_query)
        return torch.exp(a) * (self.features ** -0.5) + 1e-6


@torch.no_grad()
def izh_voltage(a: torch.Tensor, bias: float, gain: float, window_ms: int) -> torch.Tensor:
    x = a.float()
    v = torch.full_like(x, -65.0)
    u = 0.2 * v
    I = float(bias) + float(gain) * x
    out = []
    for _ in range(int(window_ms)):
        dv = 0.04 * v * v + 5 * v + 140 - u + I
        v = v + 0.5 * dv
        dv = 0.04 * v * v + 5 * v + 140 - u + I
        v = v + 0.5 * dv
        u = u + 0.02 * (0.2 * v - u)
        fired = v >= 30.0
        v = torch.where(fired, torch.full_like(v, -65.0), v)
        u = torch.where(fired, u + 8.0, u)
        out.append(v)
    return torch.stack(out, -1)


@torch.no_grad()
def traj_design(a: torch.Tensor, bias: float, gain: float, window_ms: int) -> torch.Tensor:
    v = izh_voltage(a, bias, gain, window_ms)
    one = torch.ones(*a.shape, 1, device=a.device, dtype=v.dtype)
    return torch.cat([one, v], -1)


class PerHeadIzhPhi(nn.Module):
    """Fixed per-head Izh RS trajectory -> linear decoder feature interface."""

    def __init__(
        self,
        exact: PositiveRF,
        specs: Sequence[dict],
        decoders: Sequence[torch.Tensor],
    ):
        super().__init__()
        if len(specs) != exact.heads or len(decoders) != exact.heads:
            raise ValueError("one Izh spec and decoder are required per attention head")
        self.head_dim = exact.head_dim
        self.features = exact.features
        self.heads = exact.heads
        self.register_buffer("omega", exact.omega.detach().clone())
        self.specs = [dict(s) for s in specs]
        for h, dec in enumerate(decoders):
            self.register_buffer(f"decoder_{h}", dec.detach().clone())
        self.last_clamp = 0.0

    def stable_a(self, x: torch.Tensor, is_query: bool) -> torch.Tensor:
        xx = x * (self.head_dim ** -0.25)
        dash = torch.einsum("bhnd,hmd->bhnm", xx, self.omega)
        diag = 0.5 * (xx * xx).sum(-1, keepdim=True)
        a = dash - diag
        if is_query:
            return (a - a.amax(-1, keepdim=True).detach()).clamp(-30.0, 0.0)
        return a.clamp(-30.0, 30.0)

    def forward(self, x: torch.Tensor, *, is_query: bool) -> torch.Tensor:
        a = self.stable_a(x, is_query)
        outs = []
        clamps = []
        for h, spec in enumerate(self.specs):
            ah = a[:, h]
            X = traj_design(ah.reshape(-1), **spec)
            dec = getattr(self, f"decoder_{h}")
            raw = X @ dec.to(X.dtype)
            clamps.append(float((raw <= 1e-8).float().mean()))
            outs.append(raw.clamp_min(1e-8).reshape_as(ah))
        self.last_clamp = float(np.mean(clamps))
        return torch.stack(outs, 1)


class SparseLinear(nn.Module):
    """Structural sparse Linear; mask is a persistent buffer and maps to synapses."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        density: float,
        bias: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        k = max(1, min(in_features, int(round(float(density) * in_features))))
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        mask = torch.zeros(out_features, in_features)
        for o in range(out_features):
            idx = torch.randperm(in_features, generator=g)[:k]
            mask[o, idx] = 1.0
        self.register_buffer("mask", mask)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)

    @property
    def nnz(self) -> int:
        return int(self.mask.sum().item())

    @property
    def density(self) -> float:
        return float(self.mask.mean().item())


@dataclass
class SNNTransformState:
    H: torch.Tensor
    g: torch.Tensor
    pk: torch.Tensor
    v: torch.Tensor
    weight: torch.Tensor
    pq: torch.Tensor
    x_cache: torch.Tensor

    def detach(self) -> "SNNTransformState":
        return SNNTransformState(
            H=self.H.detach(), g=self.g.detach(), pk=self.pk.detach(), v=self.v.detach(),
            weight=self.weight.detach(), pq=self.pq.detach(), x_cache=self.x_cache.detach()
        )


class EventCachedLinearAttention(nn.Module):
    def __init__(
        self,
        d: int = 32,
        heads: int = 2,
        features: int = 32,
        out_density: float = 0.25,
        seed: int = 17,
    ):
        super().__init__()
        if d % heads:
            raise ValueError("d must be divisible by heads")
        self.d = d
        self.heads = heads
        self.dh = d // heads
        self.m = features
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = SparseLinear(d, d, out_density, bias=True, seed=seed + 101)
        self.phi_q: nn.Module = PositiveRF(heads, self.dh, features, seed)
        self.phi_k: nn.Module = PositiveRF(heads, self.dh, features, seed)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        return x.reshape(B, N, self.heads, self.dh).transpose(1, 2)

    def _q_all(self, x: torch.Tensor) -> torch.Tensor:
        return self.phi_q(self._split(self.q(x)), is_query=True)

    def _kv_all(self, x: torch.Tensor):
        k = self._split(self.k(x))
        v = self._split(self.v(x))
        return self.phi_k(k, is_query=False), v

    @staticmethod
    def _alpha_tensor(retention, B: int, H: int, device, dtype) -> torch.Tensor:
        if retention is None:
            return torch.ones(B, H, device=device, dtype=dtype)
        if torch.is_tensor(retention):
            a = retention.to(device=device, dtype=dtype)
            if a.ndim == 0:
                return a.expand(B, H)
            if a.shape == (H,):
                return a[None].expand(B, H)
            if a.shape == (B, H):
                return a
            raise ValueError("retention tensor must be scalar, [H], or [B,H]")
        return torch.full((B, H), float(retention), device=device, dtype=dtype)

    def step(
        self,
        x: torch.Tensor,
        state: Optional[SNNTransformState],
        changed: torch.Tensor,
        *,
        retention=None,
        force_project_all: bool = False,
        raw_x_for_cache: Optional[torch.Tensor] = None,
    ):
        B, N, _ = x.shape
        alpha = self._alpha_tensor(retention, B, self.heads, x.device, x.dtype)
        q_evals = k_evals = v_evals = 0

        if state is None:
            pq = self._q_all(x)
            pk, v = self._kv_all(x)
            H = torch.einsum("bhnm,bhnd->bhmd", pk, v)
            g = pk.sum(2)
            weight = torch.ones(B, self.heads, N, device=x.device, dtype=x.dtype)
            q_evals = k_evals = v_evals = B * N
        else:
            a4 = alpha[:, :, None, None]
            a3 = alpha[:, :, None]
            H = state.H * a4
            g = state.g * a3
            weight = state.weight * alpha[:, :, None]

            pq = state.pq.clone()
            pk = state.pk.clone()
            v = state.v.clone()

            if force_project_all:
                pq_full = self._q_all(x)
                pk_full, v_full = self._kv_all(x)
                q_evals = k_evals = v_evals = B * N
                pq = pq_full
                pk_new_source = pk_full
                v_new_source = v_full
            else:
                pk_new_source = None
                v_new_source = None

            for b in range(B):
                idx = torch.nonzero(changed[b], as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                nidx = int(idx.numel())

                if force_project_all:
                    ppq = pq_full[b:b + 1, :, idx, :]
                    ppk = pk_new_source[b:b + 1, :, idx, :]
                    vvk = v_new_source[b:b + 1, :, idx, :]
                else:
                    xb = x[b:b + 1, idx]
                    ppq = self.phi_q(self._split(self.q(xb)), is_query=True)
                    kk = self._split(self.k(xb))
                    vvk = self._split(self.v(xb))
                    ppk = self.phi_k(kk, is_query=False)
                    q_evals += nidx
                    k_evals += nidx
                    v_evals += nidx

                oldpk = state.pk[b:b + 1, :, idx, :]
                oldv = state.v[b:b + 1, :, idx, :]
                ow = weight[b:b + 1, :, idx]
                H = H - torch.einsum("bhn,bhnm,bhnd->bhmd", ow, oldpk, oldv)
                H = H + torch.einsum("bhnm,bhnd->bhmd", ppk, vvk)
                g = g - (oldpk * ow[:, :, :, None]).sum(2) + ppk.sum(2)

                pq[b, :, idx, :] = ppq[0]
                pk[b, :, idx, :] = ppk[0]
                v[b, :, idx, :] = vvk[0]
                weight[b, :, idx] = 1.0

            if force_project_all:
                # Recomputed unchanged entries are mathematically identical; keeping them
                # makes the dense-projection reference explicit without changing state semantics.
                pk = pk_new_source
                v = v_new_source

        num = torch.einsum("bhnm,bhmd->bhnd", pq, H)
        den = torch.einsum("bhnm,bhm->bhn", pq, g).unsqueeze(-1).clamp_min(1e-8)
        y = num / den
        out = self.out(y.transpose(1, 2).reshape(B, N, self.d))
        cache = raw_x_for_cache if raw_x_for_cache is not None else x
        new_state = SNNTransformState(H=H, g=g, pk=pk, v=v, weight=weight, pq=pq,
                                      x_cache=cache.detach().clone())
        stats = {
            "q_token_evals": q_evals,
            "k_token_evals": k_evals,
            "v_token_evals": v_evals,
            "changed_tokens": int(changed.sum().item()),
            "total_tokens": B * N,
            "retention_mean": float(alpha.mean().detach()),
        }
        return out, new_state, stats


class SNNTransformBlock(nn.Module):
    """Reusable canonical block.

    Main API:
        y, state = block(x, state)

    If `changed` is omitted, changed tokens are inferred exactly from the cached
    previous input.  Supplying `changed` avoids the comparison and is preferred
    when an upstream event source already provides the event mask.
    """

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
        self.last_stats = {}

    def set_izh_interface(
        self,
        q_specs: Sequence[dict],
        q_decoders: Sequence[torch.Tensor],
        k_specs: Sequence[dict],
        k_decoders: Sequence[torch.Tensor],
    ) -> None:
        q_exact = self.attn.phi_q
        k_exact = self.attn.phi_k
        if not isinstance(q_exact, PositiveRF) or not isinstance(k_exact, PositiveRF):
            raise TypeError("set_izh_interface must be called while exact PositiveRF modules are installed")
        self.attn.phi_q = PerHeadIzhPhi(q_exact, q_specs, q_decoders)
        self.attn.phi_k = PerHeadIzhPhi(k_exact, k_specs, k_decoders)

    def infer_changed(self, x: torch.Tensor, state: Optional[SNNTransformState]) -> torch.Tensor:
        if state is None:
            return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if state.x_cache.shape != x.shape:
            raise ValueError("state/input shape mismatch; reset state when batch or token shape changes")
        return torch.any(x != state.x_cache, dim=-1)

    def step(
        self,
        x: torch.Tensor,
        state: Optional[SNNTransformState] = None,
        changed: Optional[torch.Tensor] = None,
        *,
        reset: bool = False,
        retention=None,
        force_project_all: bool = False,
    ):
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

        a, new_state, stats = self.attn.step(
            self.n1(x), state, changed, retention=retention,
            force_project_all=force_project_all, raw_x_for_cache=x,
        )
        z = x + a
        pre = self.f1(self.n2(z))
        h = F.relu(pre - self.theta)
        y = z + self.f2(h)
        hidden_activity = float((h > 0).float().mean().detach())
        stats.update({
            "hidden_activity": hidden_activity,
            "output_density": self.attn.out.density,
            "ff1_density": self.f1.density,
            "ff2_density": self.f2.density,
        })
        self.last_stats = stats
        return y, new_state, stats

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[SNNTransformState] = None,
        changed: Optional[torch.Tensor] = None,
        *,
        reset: bool = False,
        retention=None,
    ):
        y, state, stats = self.step(x, state, changed, reset=reset, retention=retention)
        self.last_stats = stats
        return y, state

    def synapse_account(self, hidden_activity: float) -> dict:
        d, ff = self.d, self.ff
        dense = d * d + d * ff + ff * d
        static = self.attn.out.nnz + self.f1.nnz + self.f2.nnz
        effective = self.attn.out.nnz + self.f1.nnz + float(hidden_activity) * self.f2.nnz
        return {
            "dense_tail_synapses": dense,
            "static_sparse_synapses": static,
            "static_fraction": static / dense,
            "effective_synops_proxy": effective,
            "effective_fraction": effective / dense,
            "effective_reduction": 1.0 - effective / dense,
        }
