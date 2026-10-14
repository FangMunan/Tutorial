#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SNNTransformBlock-v1.1 — cleaned canonical engineering core.

This version preserves the accepted v1 computation while removing execution
residue identified by the mathematical audit:

* fixed-alpha retention state/plumbing removed;
* redundant FFN threshold absorbed into the first FFN bias;
* raw-input cache and automatic event inference removed from the core;
* validation/profiling paths removed from the hot path;
* Izhikevich trajectory decoding performed online without materializing the
  full membrane trajectory tensor.

Core contract
-------------
    y, state = block(x, state=None, changed=None)

For a fresh state, all tokens are projected.  For persistent execution,
``changed`` must be supplied explicitly as a Boolean [B,N] event mask.

Persistent state is minimal for mutable-token execution:
    S = (H, g, phi_q_cache, phi_k_cache, value_cache)

The reference PyTorch SparseLinear still stores a dense tensor plus a structural
mask so that v1 checkpoint migration remains exact.  Hardware/scalable backends
should store only extant synapses; that is an execution backend change, not a
change to this mathematical operator.
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


@dataclass(frozen=True)
class BlockConfig:
    d: int = 32
    heads: int = 2
    features: int = 32
    ff: int = 64
    out_density: float = 0.25
    ff1_density: float = 0.25
    ff2_density: float = 0.25
    seed: int = 17

    def validate(self) -> None:
        if min(self.d, self.heads, self.features, self.ff) <= 0:
            raise ValueError("d, heads, features and ff must be positive")
        if self.d % self.heads != 0:
            raise ValueError("d must be divisible by heads")
        for name in ("out_density", "ff1_density", "ff2_density"):
            value = float(getattr(self, name))
            if not (0.0 < value <= 1.0):
                raise ValueError(f"{name} must be in (0,1]")


@dataclass
class SNNTransformState:
    """Minimal persistent state for mutable-token event execution."""

    H: Tensor
    g: Tensor
    pq: Tensor
    pk: Tensor
    v: Tensor

    def detach(self) -> "SNNTransformState":
        return SNNTransformState(
            H=self.H.detach(),
            g=self.g.detach(),
            pq=self.pq.detach(),
            pk=self.pk.detach(),
            v=self.v.detach(),
        )

    def to(self, *args, **kwargs) -> "SNNTransformState":
        return SNNTransformState(
            H=self.H.to(*args, **kwargs),
            g=self.g.to(*args, **kwargs),
            pq=self.pq.to(*args, **kwargs),
            pk=self.pk.to(*args, **kwargs),
            v=self.v.to(*args, **kwargs),
        )


def _stable_a(x: Tensor, omega: Tensor, head_dim: int, is_query: bool) -> Tensor:
    xx = x * (head_dim ** -0.25)
    dash = torch.einsum("bhnd,hmd->bhnm", xx, omega)
    diag = 0.5 * (xx * xx).sum(dim=-1, keepdim=True)
    a = dash - diag
    if is_query:
        return (a - a.amax(dim=-1, keepdim=True).detach()).clamp(-30.0, 0.0)
    return a.clamp(-30.0, 30.0)


class PositiveRF(nn.Module):
    """Exact/reference positive random-feature interface."""

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

    def forward(self, x: Tensor, *, is_query: bool) -> Tensor:
        a = _stable_a(x, self.omega, self.head_dim, is_query)
        return torch.exp(a) * (self.features ** -0.5) + 1e-6


@torch.no_grad()
def _stream_izh_decode(a: Tensor, decoder: Tensor, *, bias: float, gain: float, window_ms: int) -> Tensor:
    """Decode an Izh RS membrane trajectory online.

    The historical interface computes [1,v1,...,vT] @ decoder.  This routine
    evaluates the same linear functional incrementally and never stores the full
    trajectory.  ``decoder`` is one vector of length T+1, shared across scalar
    random features exactly as in the frozen v1 calibration.
    """

    dec = decoder.to(device=a.device, dtype=torch.float32).flatten()
    T = int(window_ms)
    if dec.numel() != T + 1:
        raise ValueError(f"decoder length must be window_ms+1={T+1}, got {dec.numel()}")

    x = a.float()
    v = torch.full_like(x, -65.0)
    u = 0.2 * v
    current = float(bias) + float(gain) * x
    out = torch.full_like(x, float(dec[0]))

    for t in range(T):
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current
        v = v + 0.5 * dv
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current
        v = v + 0.5 * dv
        u = u + 0.02 * (0.2 * v - u)
        fired = v >= 30.0
        v = torch.where(fired, torch.full_like(v, -65.0), v)
        u = torch.where(fired, u + 8.0, u)
        out = out + v * dec[t + 1]

    return out


class PerHeadIzhPhi(nn.Module):
    """Frozen per-head Izh RS membrane-dynamics feature interface."""

    def __init__(
        self,
        exact: PositiveRF,
        specs: Sequence[Dict[str, Any]],
        decoders: Sequence[Tensor],
    ):
        super().__init__()
        if len(specs) != exact.heads or len(decoders) != exact.heads:
            raise ValueError("one Izh spec and one decoder are required per head")
        self.heads = exact.heads
        self.head_dim = exact.head_dim
        self.features = exact.features
        self.register_buffer("omega", exact.omega.detach().clone())
        self.specs = [dict(spec) for spec in specs]
        for h, decoder in enumerate(decoders):
            self.register_buffer(f"decoder_{h}", decoder.detach().clone())

    def forward(self, x: Tensor, *, is_query: bool) -> Tensor:
        a = _stable_a(x, self.omega, self.head_dim, is_query)
        outputs = []
        for h, spec in enumerate(self.specs):
            raw = _stream_izh_decode(
                a[:, h],
                getattr(self, f"decoder_{h}"),
                bias=float(spec["bias"]),
                gain=float(spec["gain"]),
                window_ms=int(spec["window_ms"]),
            )
            outputs.append(raw.clamp_min(1e-8))
        return torch.stack(outputs, dim=1)


class SparseLinear(nn.Module):
    """Reference structural sparse linear operator.

    The graph is sparse; this portable PyTorch reference keeps dense storage for
    checkpoint compatibility.  Efficient backends should compile mask!=0 entries
    into compact synapse lists without changing the operator.
    """

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
    """Normalized H/g linear attention with explicit token events."""

    def __init__(self, d: int, heads: int, features: int, out_density: float, seed: int):
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
        B, N, _ = x.shape
        return x.reshape(B, N, self.heads, self.head_dim).transpose(1, 2)

    def _q_all(self, x: Tensor) -> Tensor:
        return self.phi_q(self._split(self.q(x)), is_query=True)

    def _kv_all(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        k = self._split(self.k(x))
        v = self._split(self.v(x))
        return self.phi_k(k, is_query=False), v

    def forward(
        self,
        x: Tensor,
        state: Optional[SNNTransformState] = None,
        changed: Optional[Tensor] = None,
    ) -> Tuple[Tensor, SNNTransformState]:
        B, N, _ = x.shape

        if state is None:
            pq = self._q_all(x)
            pk, value = self._kv_all(x)
            H = torch.einsum("bhnm,bhnd->bhmd", pk, value)
            g = pk.sum(dim=2)
        else:
            if changed is None:
                raise ValueError("changed [B,N] is required when persistent state is supplied")
            changed = changed.to(device=x.device, dtype=torch.bool)
            if changed.shape != (B, N):
                raise ValueError(f"changed must have shape {(B,N)}")

            H = state.H.clone()
            g = state.g.clone()
            pq = state.pq.clone()
            pk = state.pk.clone()
            value = state.v.clone()

            # Reference implementation: simple batch loop.  Event-native backends
            # should consume the same changed-token set directly.  H/g deltas are
            # batch-local by contract; changing sample b must never mutate another
            # sample's resident state.
            for b in range(B):
                idx = torch.nonzero(changed[b], as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                xb = x[b:b + 1, idx]
                new_pq = self.phi_q(self._split(self.q(xb)), is_query=True)
                new_k = self._split(self.k(xb))
                new_v = self._split(self.v(xb))
                new_pk = self.phi_k(new_k, is_query=False)

                old_pk = state.pk[b:b + 1, :, idx, :]
                old_v = state.v[b:b + 1, :, idx, :]
                H[b:b + 1] -= torch.einsum("bhnm,bhnd->bhmd", old_pk, old_v)
                H[b:b + 1] += torch.einsum("bhnm,bhnd->bhmd", new_pk, new_v)
                g[b:b + 1] -= old_pk.sum(dim=2)
                g[b:b + 1] += new_pk.sum(dim=2)

                pq[b, :, idx, :] = new_pq[0]
                pk[b, :, idx, :] = new_pk[0]
                value[b, :, idx, :] = new_v[0]

        numerator = torch.einsum("bhnm,bhmd->bhnd", pq, H)
        denominator = torch.einsum("bhnm,bhm->bhn", pq, g).unsqueeze(-1).clamp_min(1e-8)
        attended = numerator / denominator
        output = self.out(attended.transpose(1, 2).reshape(B, N, self.d))
        new_state = SNNTransformState(H=H, g=g, pq=pq, pk=pk, v=value)
        return output, new_state


class SNNTransformBlock(nn.Module):
    """Clean canonical SNN-Transform block v1.1."""

    VERSION = "SNNTransformBlock-v1.1"

    def __init__(
        self,
        d: int = 32,
        heads: int = 2,
        features: int = 32,
        ff: int = 64,
        out_density: float = 0.25,
        ff1_density: float = 0.25,
        ff2_density: float = 0.25,
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
            seed=seed,
        )
        self.config.validate()
        self.d = int(d)
        self.heads = int(heads)
        self.ff = int(ff)

        self.n1 = nn.LayerNorm(d)
        self.attn = EventCachedLinearAttention(d, heads, features, out_density, seed)
        self.n2 = nn.LayerNorm(d)
        self.f1 = SparseLinear(d, ff, ff1_density, bias=True, seed=seed + 201)
        self.f2 = SparseLinear(ff, d, ff2_density, bias=True, seed=seed + 301)

    def set_izh_interface(
        self,
        q_specs: Sequence[Dict[str, Any]],
        q_decoders: Sequence[Tensor],
        k_specs: Sequence[Dict[str, Any]],
        k_decoders: Sequence[Tensor],
    ) -> None:
        q_exact = self.attn.phi_q
        k_exact = self.attn.phi_k
        if not isinstance(q_exact, PositiveRF) or not isinstance(k_exact, PositiveRF):
            raise TypeError("Izh interface already installed or PositiveRF reference unavailable")
        self.attn.phi_q = PerHeadIzhPhi(q_exact, q_specs, q_decoders)
        self.attn.phi_k = PerHeadIzhPhi(k_exact, k_specs, k_decoders)

    def forward(
        self,
        x: Tensor,
        state: Optional[SNNTransformState] = None,
        changed: Optional[Tensor] = None,
    ) -> Tuple[Tensor, SNNTransformState]:
        if x.ndim != 3 or x.shape[-1] != self.d:
            raise ValueError(f"x must have shape [B,N,{self.d}]")
        a, new_state = self.attn(self.n1(x), state, changed)
        z = x + a
        hidden = F.relu(self.f1(self.n2(z)))
        y = z + self.f2(hidden)
        return y, new_state

    @staticmethod
    def reset_state() -> None:
        return None

    def config_dict(self) -> Dict[str, Any]:
        return asdict(self.config)


def _torch_load(path: Union[str, Path], map_location="cpu") -> Dict[str, Any]:
    try:
        return torch.load(Path(path), map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(Path(path), map_location=map_location)


def _install_checkpoint_izh(block: SNNTransformBlock, checkpoint: Dict[str, Any]) -> None:
    if all(k in checkpoint for k in ("q_specs", "q_decoders", "k_specs", "k_decoders")):
        block.set_izh_interface(
            checkpoint["q_specs"], checkpoint["q_decoders"],
            checkpoint["k_specs"], checkpoint["k_decoders"],
        )


def load_v1_checkpoint_as_v1_1(
    checkpoint_path: Union[str, Path],
    *,
    device: Union[str, torch.device] = "cpu",
    eval_mode: bool = True,
) -> Tuple[SNNTransformBlock, Dict[str, Any]]:
    """Load a frozen v1 checkpoint into the mathematically equivalent v1.1 core.

    Exact migration performed:
        f1.bias_new = f1.bias_v1 - theta_v1
    The fixed-alpha retention configuration is intentionally discarded because
    v1.1 represents only the frozen alpha=1 architecture.
    """

    checkpoint = _torch_load(checkpoint_path, map_location=device)
    if checkpoint.get("format") != "SNNTransformBlock-v1":
        raise ValueError("expected a SNNTransformBlock-v1 checkpoint")

    old_cfg = dict(checkpoint["config"])
    cfg = {
        k: old_cfg[k]
        for k in ("d", "heads", "features", "ff", "out_density", "ff1_density", "ff2_density", "seed")
    }
    block = SNNTransformBlock(**cfg)
    _install_checkpoint_izh(block, checkpoint)

    state = dict(checkpoint["model_state_dict"])
    theta = state.pop("theta", None)
    if theta is not None:
        if "f1.bias" not in state:
            raise ValueError("v1 checkpoint has theta but no f1.bias")
        state["f1.bias"] = state["f1.bias"] - theta.to(state["f1.bias"])

    incompatible = block.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError({
            "missing": incompatible.missing_keys,
            "unexpected": incompatible.unexpected_keys,
        })
    block.to(device)
    if eval_mode:
        block.eval()
    return block, checkpoint


def checkpoint_dict(block: SNNTransformBlock, *, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "format": block.VERSION,
        "config": block.config_dict(),
        "model_state_dict": block.state_dict(),
        "metadata": {} if metadata is None else dict(metadata),
    }
    if isinstance(block.attn.phi_q, PerHeadIzhPhi) and isinstance(block.attn.phi_k, PerHeadIzhPhi):
        payload.update({
            "q_specs": [dict(x) for x in block.attn.phi_q.specs],
            "q_decoders": [getattr(block.attn.phi_q, f"decoder_{h}").detach().cpu() for h in range(block.heads)],
            "k_specs": [dict(x) for x in block.attn.phi_k.specs],
            "k_decoders": [getattr(block.attn.phi_k, f"decoder_{h}").detach().cpu() for h in range(block.heads)],
        })
    return payload


def save_v1_1(block: SNNTransformBlock, path: Union[str, Path], *, metadata: Optional[Dict[str, Any]] = None) -> None:
    torch.save(checkpoint_dict(block, metadata=metadata), Path(path))


def load_v1_1(
    path: Union[str, Path],
    *,
    device: Union[str, torch.device] = "cpu",
    eval_mode: bool = True,
) -> Tuple[SNNTransformBlock, Dict[str, Any]]:
    checkpoint = _torch_load(path, map_location=device)
    if checkpoint.get("format") != SNNTransformBlock.VERSION:
        raise ValueError(f"expected format {SNNTransformBlock.VERSION!r}")
    block = SNNTransformBlock(**checkpoint["config"])
    _install_checkpoint_izh(block, checkpoint)
    block.load_state_dict(checkpoint["model_state_dict"], strict=True)
    block.to(device)
    if eval_mode:
        block.eval()
    return block, checkpoint