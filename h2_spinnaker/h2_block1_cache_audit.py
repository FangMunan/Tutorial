#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H2 pre-mapping audit: resident block-1 projection / dynamical-feature cache.

This optimization is deliberately narrower than caching the whole attention read.
It removes repeated Q/K/V projections and repeated Izh feature evaluations that are
provably redundant in the persistent first block, while recomputing the transient
H_m/g_m reduction from cached token contributions each round.  Recomputing that
reduction preserves the current H1 numerical summation semantics much more closely
than repeatedly subtracting old contributions from H_m/g_m.

Contract
--------
* generation starts fully masked;
* reveal is monotonic (no committed token is re-masked);
* block-1 input is token + position + class embedding only;
* therefore a position changes at most once: MASK -> committed token.

No learned function is changed.  The optimization is eligible for H2 hardware only
if the calibrated resident path passes the FP32 equivalence gate against the frozen
H1 sparse path.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "h1_cloud"))
import h0_base as h0
import h1_sparse_fusion as h1
import h1b2_error_localization as h1b2
import h1b21_qk_layer_scan as b21


@dataclass
class CachedState:
    Hp: torch.Tensor
    gp: torch.Tensor
    pq_cache: torch.Tensor
    pk_mask: torch.Tensor
    v_mask: torch.Tensor
    known: torch.Tensor
    total_q_tokens: int = 0
    total_kv_tokens: int = 0


def _split_projected(linear, x: torch.Tensor, heads: int, dh: int) -> torch.Tensor:
    z = linear(x)
    B, N, D = z.shape
    return z.reshape(B, N, heads, dh).transpose(1, 2)


def persistent_cached(
    self: h0.LinearAttention,
    x: torch.Tensor,
    known: torch.Tensor,
    state: Optional[CachedState],
    alpha: torch.Tensor,
) -> Tuple[torch.Tensor, CachedState, Dict[str, float]]:
    B, N, _ = x.shape
    H, M, dh = self.heads, self.m, self.dh
    initialized_now = state is None

    if state is None:
        if bool(known.any()):
            raise RuntimeError(
                "Resident block1 cache requires the first generation canvas to be fully masked"
            )
        q0 = self._split(self.q(x))
        k0 = self._split(self.k(x))
        v0 = self._split(self.v(x))
        pq0 = self.phi(q0, is_query=True)
        pk0 = self.phi(k0, is_query=False)
        state = CachedState(
            Hp=torch.zeros(B, H, M, dh, device=x.device, dtype=x.dtype),
            gp=torch.zeros(B, H, M, device=x.device, dtype=x.dtype),
            pq_cache=pq0.clone(),
            pk_mask=pk0.clone(),
            v_mask=v0.clone(),
            known=torch.zeros(B, N, device=x.device, dtype=torch.bool),
            total_q_tokens=B * N,
            total_kv_tokens=B * N,
        )

    prev_known = state.known
    remasked = prev_known & (~known)
    if bool(remasked.any()):
        raise RuntimeError("Resident block1 cache assumes monotonic reveal; remasking detected")

    # Synchronized selective retention applies only to committed historical state.
    Hp = state.Hp * alpha.reshape(B, H, 1, 1)
    gp = state.gp * alpha.reshape(B, H, 1)
    pq_cache = state.pq_cache.clone()

    new_known = known & (~prev_known)
    n_new_total = 0

    # A newly revealed token is projected / dynamically mapped exactly once.
    for bi in range(B):
        idx = torch.nonzero(new_known[bi], as_tuple=False).flatten()
        n_new_total += int(idx.numel())
        if idx.numel() == 0:
            continue
        xa = x[bi:bi + 1, idx, :]
        qa = _split_projected(self.q, xa, H, dh)
        ka = _split_projected(self.k, xa, H, dh)
        va = _split_projected(self.v, xa, H, dh)
        pqa = self.phi(qa, is_query=True)
        pka = self.phi(ka, is_query=False)
        pq_cache[bi:bi + 1, :, idx, :] = pqa
        Hp[bi:bi + 1] += torch.einsum("bhnm,bhnd->bhmd", pka, va)
        gp[bi:bi + 1] += pka.sum(dim=2)

    # Rebuild transient Hm/gm from cached MASK features.  This costs the outer-product
    # reduction but *not* Q/K/V projection nor the 6-ms Izh feature dynamics.  Using
    # the same sorted masked-token subset each round preserves the H1 reduction order.
    Hm = torch.zeros_like(Hp)
    gm = torch.zeros_like(gp)
    for bi in range(B):
        midx = torch.nonzero(~known[bi], as_tuple=False).flatten()
        if midx.numel() == 0:
            continue
        pkm = state.pk_mask[bi:bi + 1, :, midx, :]
        vm = state.v_mask[bi:bi + 1, :, midx, :]
        Hm[bi:bi + 1] = torch.einsum("bhnm,bhnd->bhmd", pkm, vm)
        gm[bi:bi + 1] = pkm.sum(dim=2)

    Htot = Hp + Hm
    gtot = gp + gm
    num = torch.einsum("bhnm,bhmd->bhnd", pq_cache, Htot)
    den = torch.einsum("bhnm,bhm->bhn", pq_cache, gtot).unsqueeze(-1).clamp_min(1e-8)
    y = num / den
    out = self.out(y.transpose(1, 2).reshape(x.shape))

    ns = CachedState(
        Hp=Hp,
        gp=gp,
        pq_cache=pq_cache,
        pk_mask=state.pk_mask,
        v_mask=state.v_mask,
        known=known.clone(),
        total_q_tokens=state.total_q_tokens + n_new_total,
        total_kv_tokens=state.total_kv_tokens + n_new_total,
    )
    stats = {
        "new_known_mean": float(new_known.sum(1).float().mean().detach().cpu()),
        "masked_mean": float((~known).sum(1).float().mean().detach().cpu()),
        "state_H_rms": float(Hp.square().mean().sqrt().detach().cpu()),
        "state_g_min": float(gp.min().detach().cpu()),
        "q_tokens_evaluated_this_step": float(B * N if initialized_now else n_new_total),
        "kv_tokens_evaluated_this_step": float(B * N if initialized_now else n_new_total),
        "resident_q_tokens_total": float(ns.total_q_tokens),
        "resident_kv_tokens_total": float(ns.total_kv_tokens),
    }
    return out, ns, stats


def enable_cached_block1(model: h0.MaskGIT) -> h0.MaskGIT:
    import types
    if model.kind != "linear":
        raise ValueError("resident cache is defined for the linear backbone")
    model.blocks[0].attn.persistent = types.MethodType(persistent_cached, model.blocks[0].attn)
    return model


def load_selected(path: Path, device):
    d = json.loads(path.read_text())
    out = {}
    for key, val in d["selected"].items():
        layer = int(key.split("_")[0][1:])
        side = key.split("_")[1]
        out[(layer, side)] = {
            "spec": val["spec"],
            "decoder": torch.tensor(val["decoder"], dtype=torch.float32, device=device),
            "row": val.get("probe", {}),
        }
    return out


@torch.no_grad()
def audit(current, cached, device, rounds=10, batch=4, seed=991):
    labels, canvases = h1.make_reveal_canvases(batch, rounds, device, seed)
    sc = sr = None
    rows = []
    max_abs = 0.0
    max_rel = 0.0
    current_q_total = current_kv_total = 0.0
    cached_q_total = cached_kv_total = 0.0

    for ri, cur in enumerate(canvases):
        yc, sc, sts_c = current.forward_persistent(cur, labels, sc, "selective")
        yr, sr, sts_r = cached.forward_persistent(cur, labels, sr, "selective")
        diff = yr - yc
        ae = float(diff.abs().max().cpu())
        re = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(yc).clamp_min(1e-12))
        max_abs = max(max_abs, ae)
        max_rel = max(max_rel, re)

        current_q_total += batch * h0.SEQ
        current_kv_total += batch * float(sts_c["active_kv_tokens_mean"])
        cached_q_total += float(sts_r["q_tokens_evaluated_this_step"])
        cached_kv_total += float(sts_r["kv_tokens_evaluated_this_step"])

        rows.append({
            "round": ri,
            "known": int(cur[0].ne(h0.MASK_ID).sum()),
            "max_abs": ae,
            "rel_l2": re,
            "current_active_kv_mean": float(sts_c["active_kv_tokens_mean"]),
            "cached_new_known_mean": float(sts_r["new_known_mean"]),
        })

    return {
        "max_abs": max_abs,
        "max_rel_l2": max_rel,
        "pass_fp32": bool(max_abs <= 1e-4 and max_rel <= 1e-5),
        "rounds": rows,
        "operation_accounting": {
            "current_q_token_evals": current_q_total,
            "current_kv_token_evals": current_kv_total,
            "cached_q_token_evals": cached_q_total,
            "cached_kv_token_evals": cached_kv_total,
            "q_reduction_fraction": 1.0 - cached_q_total / current_q_total,
            "kv_reduction_fraction": 1.0 - cached_kv_total / current_kv_total,
            "combined_q_plus_kv_reduction_fraction": 1.0 - (cached_q_total + cached_kv_total) / (current_q_total + current_kv_total),
            "note": "Transient Hm/gm outer-product reduction is still executed each round; these counts cover projection + dynamic-feature evaluations only."
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1b", type=Path, required=True)
    ap.add_argument("--b21", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("H2_CACHE_AUDIT"))
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=512)
    ap.add_argument("--features", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    args.reuse = args.h1b
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, dense_sel, _, _ = h1b2.load_reused_models(args, device)
    selected = load_selected(args.b21 / "H1B21_RESULTS.json", device)

    current = copy.deepcopy(dense_sel)
    h1.enable_sparse_block1(current)
    b21.install_all_calibrated(current, selected)
    current.eval()

    cached = copy.deepcopy(dense_sel)
    enable_cached_block1(cached)
    b21.install_all_calibrated(cached, selected)
    cached.eval()

    result = audit(current, cached, device, rounds=10, batch=4, seed=args.seed + 70001)
    result["stage"] = "H2_BLOCK1_RESIDENT_CACHE_AUDIT"
    result["architecture"] = {
        "seq": h0.SEQ, "d": args.d, "layers": args.layers,
        "heads": args.heads, "features": args.features
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "H2_CACHE_AUDIT.json").write_text(json.dumps(result, indent=2))
    print("H2_CACHE_AUDIT_DONE", json.dumps(result, indent=2), flush=True)
    if not result["pass_fp32"]:
        raise SystemExit("resident cache failed exactness gate")


if __name__ == "__main__":
    main()
