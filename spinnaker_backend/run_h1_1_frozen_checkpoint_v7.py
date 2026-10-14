#!/usr/bin/env python3
"""H1.1 v7 — hardware-aware Q operating-point search using projective feature error.

The normalized H/g read is invariant to a scalar rescaling of each query feature
vector: c*phi(q) cancels between numerator and denominator.  Consequently raw
scalar phi relative-L2 is not the right calibration objective for Q.  v7 keeps
the frozen feature-map target but searches the fixed-point Izh operating point
(bias/gain/window) and linear trajectory decoder using a label-free, projective
query-feature objective.

No task label, task loss, backpropagation, or architecture change is used.
The K side keeps the v6 hardware-aware decoder because it is already <3e-4
calibration rel-L2.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import torch

# Use the safer v4 numerical contract.
import spinnaker_backend.h1_fixed_point_emulator as h1
h1.F_ACT = 20
h1.S_ACT = 1 << h1.F_ACT
h1.F_MEM = 22
h1.F_DEC = 20

import spinnaker_backend.run_h1_1_frozen_checkpoint as base
base.F_ACT = 20
base.F_DEC = 20
base.F_PHI = 16

import spinnaker_backend.run_h1_1_frozen_checkpoint_v2 as v2
v2.F_ACT = 20
v2.F_MEM = 22
v2.F_DEC = 20
v2.F_PQ = 24
v2.F_PK = 16
v2.F_H = 20

import spinnaker_backend.run_h1_1_frozen_checkpoint_v3 as v3
import spinnaker_backend.run_h1_1_frozen_checkpoint_v6 as v6

REPORT = Path("spinnaker_backend/run/H1_1/H1_1_V7_PROJECTIVE_Q_SEARCH.json")


def projective_rel(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    ps = np.maximum(pred.sum(axis=-1, keepdims=True), 1e-18)
    ts = np.maximum(target.sum(axis=-1, keepdims=True), 1e-18)
    pn = pred / ps
    tn = target / ts
    return base.rel_np(pn, tn)


def collect_tokens(block, hir, *, batches: int, batch: int, seed: int):
    gen = base.AssocGenerator(tokens=16, d=block.d, value_dim=8)
    g = torch.Generator(); g.manual_seed(seed)
    aq = [[] for _ in range(block.heads)]
    ak = [[] for _ in range(block.heads)]
    qtarget = [[] for _ in range(block.heads)]
    ktarget = [[] for _ in range(block.heads)]
    with torch.no_grad():
        for _ in range(batches):
            x, _ = gen.sample(batch, g)
            xq = h1.qround(x.numpy(), v2.F_ACT)
            xnq = h1.layernorm_q(xq, hir.ln1_g, hir.ln1_b)
            qiq = h1.split_heads(h1.dense_linear_q(xnq, hir.Wq, hir.bq), block.heads)
            kiq = h1.split_heads(h1.dense_linear_q(xnq, hir.Wk, hir.bk), block.heads)
            aqi = h1.stable_a_q(qiq, hir.omega, True)
            aki = h1.stable_a_q(kiq, hir.omega, False)

            xnf = block.n1(x)
            qf = block.attn._split(block.attn.q(xnf))
            kf = block.attn._split(block.attn.k(xnf))
            qt = block.attn.phi_q(qf, is_query=True).cpu().numpy()
            kt = block.attn.phi_k(kf, is_query=False).cpu().numpy()
            for h in range(block.heads):
                # token-major [tokens, features]
                aq[h].append(aqi[:, h].reshape(-1, hir.features).copy())
                ak[h].append(aki[:, h].reshape(-1, hir.features).copy())
                qtarget[h].append(qt[:, h].reshape(-1, hir.features).astype(np.float64))
                ktarget[h].append(kt[:, h].reshape(-1, hir.features).astype(np.float64))
    return (
        [np.concatenate(x, axis=0) for x in aq],
        [np.concatenate(x, axis=0) for x in qtarget],
        [np.concatenate(x, axis=0) for x in ak],
        [np.concatenate(x, axis=0) for x in ktarget],
    )


def fit_decoder_from_tokens(a_tokens, y_tokens, spec, max_scalar=16384):
    af = a_tokens.reshape(-1)
    yf = y_tokens.reshape(-1)
    if af.size > max_scalar:
        # Deterministic stride sample spans the whole empirical distribution.
        idx = np.linspace(0, af.size - 1, max_scalar, dtype=np.int64)
        af = af[idx]; yf = yf[idx]
    X = v6.fixed_izh_design(af, spec)
    d = v6.ridge_fit(X, yf)
    dq = h1.qround(d, v2.F_DEC)
    return dq, d


def predict_tokens(a_tokens, spec, decoder_q, out_frac):
    flat = a_tokens.reshape(-1)
    X = v6.fixed_izh_design(flat, spec)
    d = decoder_q.astype(np.float64) / (1 << v2.F_DEC)
    floor = 1.0 / (1 << out_frac)
    return np.maximum(X @ d, floor).reshape(a_tokens.shape)


def search_q_specs(block, hir):
    fit_a, fit_y, _, _ = collect_tokens(block, hir, batches=8, batch=8, seed=424242)
    val_a, val_y, _, _ = collect_tokens(block, hir, batches=4, batch=8, seed=515151)
    biases = [2.0, 3.0, 4.0, 5.0, 6.0]
    gains = [0.25, 0.35, 0.5, 0.75, 1.0, 1.4]
    windows = [12, 16, 20, 24, 28, 32]
    rows_all: List[dict] = []
    selected_specs = []
    selected_decoders = []

    for head in range(block.heads):
        best = None
        for bias in biases:
            for gain in gains:
                for window in windows:
                    spec = (bias, gain, window)
                    try:
                        dq, df = fit_decoder_from_tokens(fit_a[head], fit_y[head], spec)
                        # Reject coefficients that leave insufficient int64 headroom.
                        dmax = float(np.max(np.abs(df)))
                        if not np.isfinite(dmax) or dmax > 128.0:
                            continue
                        pred = predict_tokens(val_a[head], spec, dq, v2.F_PQ)
                        raw_rel = base.rel_np(pred, val_y[head])
                        proj_rel = projective_rel(pred, val_y[head])
                        # Projective error is primary; raw error breaks near ties.
                        score = proj_rel + 0.02 * raw_rel
                        row = {
                            "head": head, "bias": bias, "gain": gain, "window_ms": window,
                            "raw_rel_l2": raw_rel, "projective_rel_l2": proj_rel,
                            "decoder_max_abs": dmax, "score": score,
                        }
                        rows_all.append(row)
                        if best is None or score < best[0]:
                            best = (score, spec, row)
                    except (np.linalg.LinAlgError, FloatingPointError, OverflowError):
                        continue
        if best is None:
            raise RuntimeError(f"no valid Q operating point for head {head}")
        spec = best[1]
        # Refit selected point on all fit tokens, not the small search subset.
        dq, df = fit_decoder_from_tokens(fit_a[head], fit_y[head], spec, max_scalar=fit_a[head].size)
        selected_specs.append(spec)
        selected_decoders.append(dq)

    report = {
        "method": "label-free fixed-point Izh operating-point search; objective=projective query-feature error",
        "grid": {"bias": biases, "gain": gains, "window_ms": windows},
        "selected": [],
    }
    for h, (spec, dq) in enumerate(zip(selected_specs, selected_decoders)):
        pred = predict_tokens(val_a[h], spec, dq, v2.F_PQ)
        report["selected"].append({
            "head": h, "bias": spec[0], "gain": spec[1], "window_ms": spec[2],
            "heldout_raw_rel_l2": base.rel_np(pred, val_y[h]),
            "heldout_projective_rel_l2": projective_rel(pred, val_y[h]),
            "decoder_max_abs": float(np.max(np.abs(dq.astype(np.float64)/(1<<v2.F_DEC)))),
        })
    report["top10"] = sorted(rows_all, key=lambda r: r["score"])[:10]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({"H1_1_v7_projective_q_search": report}, indent=2), flush=True)
    return selected_specs, selected_decoders


def calibrate_k_decoders(block, hir):
    # Keep the original accepted K operating points; only compensate the fixed-
    # point membrane trajectory exactly as v6 did.
    _, _, ka, ky = collect_tokens(block, hir, batches=12, batch=8, seed=424242)
    kdec = []
    report = []
    for h in range(block.heads):
        spec = hir.k_specs[h]
        dq, df = fit_decoder_from_tokens(ka[h], ky[h], spec, max_scalar=ka[h].size)
        pred = predict_tokens(ka[h], spec, dq, v2.F_PK)
        kdec.append(dq)
        report.append({"head": h, "rel_l2": base.rel_np(pred, ky[h])})
    return kdec, report


_original_export = base.export_hir

def projective_calibrated_export(block):
    hir = _original_export(block)
    qspec, qdec = search_q_specs(block, hir)
    kdec, kreport = calibrate_k_decoders(block, hir)
    hir.q_specs = qspec
    hir.q_decoders = qdec
    hir.k_decoders = kdec
    print(json.dumps({"H1_1_v7_k_calibration": kreport}, indent=2), flush=True)
    return hir


if __name__ == "__main__":
    v2.export_hir = projective_calibrated_export
    v2.ActualCheckpointRuntime = v3.IndexCorrectRuntime
    v2.main()
