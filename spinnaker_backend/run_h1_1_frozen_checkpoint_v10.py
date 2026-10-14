#!/usr/bin/env python3
"""H1.1 v10 — sensitivity-weighted Q trajectory-decoder calibration.

v9 localized the remaining strict-parity error to the normalized H/g read.  A
single shared per-head Izh trajectory decoder is retained exactly as in the
frozen block; only its coefficients are calibrated with the correct local
conditioning of the normalized read.

For A(q)=q^T H/(q^T g), the sensitivity of output dimension d to query feature
m is

    dA_d/dq_m = (H[m,d] - A[d] g[m]) / (q^T g).

We therefore fit the SAME linear trajectory decoder with weighted least squares,
using the squared norm of this derivative as the weight of each scalar query
feature sample.  This uses no task labels and no downstream readout.  It changes
neither the Izh equations, the feature-map topology, nor the number of decoder
parameters.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

# Safe actual-checkpoint numerical contract.
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
import spinnaker_backend.run_h1_1_frozen_checkpoint_v7 as v7
import spinnaker_backend.run_h1_1_frozen_checkpoint_v9 as v9

REPORT = Path("spinnaker_backend/run/H1_1/H1_1_V10_SENSITIVITY_WEIGHTED_Q.json")


def weighted_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float = 1e-7):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    # Normalize weights to mean one for numerical conditioning; clip extreme
    # outliers so a tiny set of near-zero denominators cannot dominate the fit.
    pos = w[w > 0]
    if pos.size == 0:
        return v6.ridge_fit(X, y, ridge=ridge)
    cap = np.quantile(pos, 0.995)
    w = np.clip(w, 0.0, cap)
    w = w / max(float(w.mean()), 1e-18)
    sw = np.sqrt(w)[:, None]
    Xw = X * sw
    yw = y * sw[:, 0]
    p = X.shape[1]
    R = np.eye(p, dtype=np.float64) * ridge
    R[0, 0] = 0.0
    return np.linalg.solve(Xw.T @ Xw + R, Xw.T @ yw)


def sensitivity_weights(pack: dict) -> np.ndarray:
    q = pack["q"]          # [T,M]
    H = pack["H"]          # [T,M,Dh]
    g = pack["g"]          # [T,M]
    A = pack["att"]        # [T,Dh]
    den = np.einsum("tm,tm->t", q, g)
    den = np.maximum(np.abs(den), 1e-10)
    # derivative [T,M,Dh]
    deriv = (H - g[:, :, None] * A[:, None, :]) / den[:, None, None]
    sens2 = np.sum(deriv * deriv, axis=-1)
    # sqrt(sens2) is already an error-amplitude weight; WLS expects squared
    # residual weights, so use sens2 directly.
    return sens2


def fit_weighted_decoder(a_tokens, y_tokens, weights, spec, max_scalar=32768):
    af = a_tokens.reshape(-1)
    yf = y_tokens.reshape(-1)
    wf = weights.reshape(-1)
    if af.size > max_scalar:
        # Preserve high-sensitivity examples by ranking half the budget by w,
        # and span the remainder uniformly across the empirical distribution.
        half = max_scalar // 2
        hi = np.argpartition(wf, -half)[-half:]
        uni = np.linspace(0, af.size - 1, max_scalar - half, dtype=np.int64)
        idx = np.unique(np.concatenate([hi, uni]))
        af, yf, wf = af[idx], yf[idx], wf[idx]
    X = v6.fixed_izh_design(af, spec)
    d = weighted_ridge(X, yf, wf)
    dq = h1.qround(d, v2.F_DEC)
    return dq, d


def search_weighted_q(block, hir):
    fit = v9.collect_attention_data(block, hir, batches=10, batch=8, seed=424242)
    val = v9.collect_attention_data(block, hir, batches=5, batch=8, seed=515151)
    fit_w = [sensitivity_weights(p) for p in fit]

    biases = [2.0, 3.0, 4.0, 5.0, 6.0]
    gains = [0.25, 0.35, 0.5, 0.75, 1.0, 1.4]
    windows = [12, 16, 20, 24, 28, 32]
    rows: List[dict] = []
    selected_specs, selected_decoders = [], []

    for head in range(block.heads):
        best = None
        for bias in biases:
            for gain in gains:
                for window in windows:
                    spec = (bias, gain, window)
                    try:
                        dq, df = fit_weighted_decoder(
                            fit[head]["a"], fit[head]["q"], fit_w[head], spec
                        )
                        dmax = float(np.max(np.abs(df)))
                        if not np.isfinite(dmax) or dmax > 128.0:
                            continue
                        qp = v7.predict_tokens(val[head]["a"], spec, dq, v2.F_PQ)
                        ap = v9.attention_from_q(qp, val[head]["H"], val[head]["g"])
                        att_rel = base.rel_np(ap, val[head]["att"])
                        q_rel = base.rel_np(qp, val[head]["q"])
                        score = att_rel + 0.002 * q_rel
                        row = {
                            "head": head, "bias": bias, "gain": gain,
                            "window_ms": window, "attention_rel_l2": att_rel,
                            "q_raw_rel_l2": q_rel, "decoder_max_abs": dmax,
                            "score": score,
                        }
                        rows.append(row)
                        if best is None or score < best[0]:
                            best = (score, spec, dq.copy(), row)
                    except (np.linalg.LinAlgError, FloatingPointError, OverflowError):
                        continue
        if best is None:
            raise RuntimeError(f"no valid weighted Q operating point head {head}")
        selected_specs.append(best[1]); selected_decoders.append(best[2])

    report = {
        "method": "label-free sensitivity-weighted trajectory fit; held-out normalized-H/g attention selection",
        "selected": [min((r for r in rows if r["head"] == h), key=lambda r: r["score"]) for h in range(block.heads)],
        "top10": sorted(rows, key=lambda r: r["score"])[:10],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({"H1_1_v10_sensitivity_weighted_q": report}, indent=2), flush=True)
    return selected_specs, selected_decoders


_original_export = base.export_hir

def export_v10(block):
    hir = _original_export(block)
    qspec, qdec = search_weighted_q(block, hir)
    kdec, kreport = v7.calibrate_k_decoders(block, hir)
    hir.q_specs = qspec; hir.q_decoders = qdec; hir.k_decoders = kdec
    print(json.dumps({"H1_1_v10_k_calibration": kreport}, indent=2), flush=True)
    return hir


if __name__ == "__main__":
    v2.export_hir = export_v10
    v2.ActualCheckpointRuntime = v3.IndexCorrectRuntime
    v2.main()
