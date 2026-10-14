#!/usr/bin/env python3
"""H1.1 v6 — hardware-aware calibration of the frozen Izh trajectory decoder.

The v4/v5 precision sweeps showed that simply adding fractional bits does not
remove the remaining error: the fixed-point Izh trajectory is slightly different
from the float reference trajectory, and the normalized H/g read amplifies that
small feature-interface mismatch.

This stage does NOT retrain the task model and does NOT use task labels.  It
re-fits only the already-existing per-head linear trajectory decoder so that the
SpiNNaker-style fixed-point Izh trajectory realizes the same frozen feature map
on the trained input distribution.  This is the hardware analogue of the U4G
interface calibration that was already accepted scientifically.

The operator remains:
    stable_a -> Izh dynamics -> fixed linear trajectory decoder -> positive phi
Only the decoder coefficients are calibrated to the fixed-point implementation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

# Use the safer v4 numerical range contract rather than the near-limit v5 one.
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


CALIBRATION_REPORT = Path("spinnaker_backend/run/H1_1/H1_1_DECODER_CALIBRATION.json")


def fixed_izh_design(aq: np.ndarray, spec):
    """Return [1,v1,...,vT] for the exact v6 fixed-point Izh implementation."""
    bias, gain, T = spec
    T = int(T)
    aq = np.asarray(aq, dtype=np.int64).reshape(-1)
    a_mem = aq << (v2.F_MEM - v2.F_ACT)
    S = 1 << v2.F_MEM
    v = np.full_like(aq, -65 * S)
    c02v = int(round(0.2 * S))
    u = h1.rshift_round(v * c02v, v2.F_MEM)
    current = int(round(float(bias) * S)) + h1.rshift_round(
        a_mem * int(round(float(gain) * S)), v2.F_MEM
    )
    c004 = int(round(0.04 * S))
    c02 = int(round(0.02 * S))
    c140 = 140 * S
    cols = [np.ones(aq.size, dtype=np.float64)]
    for _ in range(T):
        v2q = h1.rshift_round(v * v, v2.F_MEM)
        dv = h1.rshift_round(v2q * c004, v2.F_MEM) + 5 * v + c140 - u + current
        v = v + h1.rshift_round(dv, 1)
        v2q = h1.rshift_round(v * v, v2.F_MEM)
        dv = h1.rshift_round(v2q * c004, v2.F_MEM) + 5 * v + c140 - u + current
        v = v + h1.rshift_round(dv, 1)
        du = h1.rshift_round(
            (h1.rshift_round(v * c02v, v2.F_MEM) - u) * c02,
            v2.F_MEM,
        )
        u = u + du
        fired = v >= 30 * S
        v = np.where(fired, -65 * S, v)
        u = np.where(fired, u + 8 * S, u)
        cols.append(v.astype(np.float64) / S)
    return np.stack(cols, axis=1)


def ridge_fit(X: np.ndarray, y: np.ndarray, ridge: float = 1e-7):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = X.shape[1]
    R = np.eye(p, dtype=np.float64) * ridge
    R[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + R, X.T @ y)


def collect_calibration(block, n_batches=16, batch=8, seed=424242):
    gen = base.AssocGenerator(tokens=16, d=block.d, value_dim=8)
    g = torch.Generator(); g.manual_seed(seed)
    # Store per-head scalar a and the corresponding frozen float-Izh phi target.
    q_a = [[] for _ in range(block.heads)]
    k_a = [[] for _ in range(block.heads)]
    q_y = [[] for _ in range(block.heads)]
    k_y = [[] for _ in range(block.heads)]

    with torch.no_grad():
        for _ in range(n_batches):
            x, _ = gen.sample(batch, g)
            # Hardware-side pre-feature path, including fixed-point LN and GEMV.
            xq = h1.qround(x.numpy(), v2.F_ACT)
            xnq = h1.layernorm_q(xq, _CAL_HIR.ln1_g, _CAL_HIR.ln1_b)
            qiq = h1.split_heads(h1.dense_linear_q(xnq, _CAL_HIR.Wq, _CAL_HIR.bq), block.heads)
            kiq = h1.split_heads(h1.dense_linear_q(xnq, _CAL_HIR.Wk, _CAL_HIR.bk), block.heads)
            aq_int = h1.stable_a_q(qiq, _CAL_HIR.omega, True)
            ak_int = h1.stable_a_q(kiq, _CAL_HIR.omega, False)

            # Frozen float target interface; no task labels are used.
            xnf = block.n1(x)
            qf = block.attn._split(block.attn.q(xnf))
            kf = block.attn._split(block.attn.k(xnf))
            qtarget = block.attn.phi_q(qf, is_query=True)
            ktarget = block.attn.phi_k(kf, is_query=False)

            for h in range(block.heads):
                q_a[h].append(aq_int[:, h].reshape(-1).copy())
                k_a[h].append(ak_int[:, h].reshape(-1).copy())
                q_y[h].append(qtarget[:, h].reshape(-1).cpu().numpy().astype(np.float64))
                k_y[h].append(ktarget[:, h].reshape(-1).cpu().numpy().astype(np.float64))

    return (
        [np.concatenate(x) for x in q_a], [np.concatenate(x) for x in q_y],
        [np.concatenate(x) for x in k_a], [np.concatenate(x) for x in k_y],
    )


def fit_hardware_decoders(block, hir):
    global _CAL_HIR
    _CAL_HIR = hir
    qa, qy, ka, ky = collect_calibration(block)
    report = {"method": "label-free ridge fit: fixed-point Izh trajectory -> frozen float Izh feature", "heads": []}
    qdec, kdec = [], []
    for h in range(block.heads):
        qX = fixed_izh_design(qa[h], hir.q_specs[h])
        kX = fixed_izh_design(ka[h], hir.k_specs[h])
        qd = ridge_fit(qX, qy[h])
        kd = ridge_fit(kX, ky[h])
        qdq = h1.qround(qd, v2.F_DEC)
        kdq = h1.qround(kd, v2.F_DEC)
        qpred = np.maximum(qX @ (qdq.astype(np.float64) / (1 << v2.F_DEC)), 1.0 / (1 << v2.F_PQ))
        kpred = np.maximum(kX @ (kdq.astype(np.float64) / (1 << v2.F_DEC)), 1.0 / (1 << v2.F_PK))
        qerr = base.rel_np(qpred, qy[h])
        kerr = base.rel_np(kpred, ky[h])
        qdec.append(qdq); kdec.append(kdq)
        report["heads"].append({
            "head": h,
            "n_scalar_samples": int(qy[h].size),
            "q_calibration_rel_l2": qerr,
            "k_calibration_rel_l2": kerr,
            "q_decoder_max_abs": float(np.max(np.abs(qd))),
            "k_decoder_max_abs": float(np.max(np.abs(kd))),
        })
    hir.q_decoders = qdec
    hir.k_decoders = kdec
    report["max_q_calibration_rel_l2"] = max(x["q_calibration_rel_l2"] for x in report["heads"])
    report["max_k_calibration_rel_l2"] = max(x["k_calibration_rel_l2"] for x in report["heads"])
    CALIBRATION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({"H1_1_hardware_decoder_calibration": report}, indent=2), flush=True)
    return hir


# Replace only the offline compiler hook.  The validation/gates and integer
# runtime are unchanged from v2/v3.
_original_export = base.export_hir

def calibrated_export(block):
    hir = _original_export(block)
    return fit_hardware_decoders(block, hir)


if __name__ == "__main__":
    v2.export_hir = calibrated_export
    v2.ActualCheckpointRuntime = v3.IndexCorrectRuntime
    v2.main()
