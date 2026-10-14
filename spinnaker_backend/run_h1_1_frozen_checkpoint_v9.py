#!/usr/bin/env python3
"""H1.1 v9 — calibrate the Q neuronal interface against normalized H/g read.

Why this objective is required
------------------------------
The trained K feature state is highly anisotropic (individual K features can be
hundreds while Q features are <=~0.18).  Therefore a very small unweighted Q
feature error can occur in exactly the feature coordinate that dominates
phi(q)^T g or phi(q)^T H.  v8 demonstrated this: raw/projective Q errors below
3e-4 on its selection set did not imply accurate normalized attention.

v9 searches the same fixed-point Izh operating points and linear trajectory
decoder, but scores each candidate by the *internal attention operator*:

    A(q;H,g) = (phi(q)^T H) / (phi(q)^T g)

on held-out, label-free inputs.  H and g are the frozen float K/V state during
selection, so the objective contains no task label and no downstream readout.
This is an interface calibration, not task training and not an architecture
change.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import torch

# Safe actual-checkpoint contract from v4/v6.
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

REPORT = Path("spinnaker_backend/run/H1_1/H1_1_V9_ATTENTION_AWARE_Q_SEARCH.json")


def collect_attention_data(block, hir, *, batches: int, batch: int, seed: int):
    gen = base.AssocGenerator(tokens=16, d=block.d, value_dim=8)
    rng = torch.Generator(); rng.manual_seed(seed)
    aq = [[] for _ in range(block.heads)]
    qtarget = [[] for _ in range(block.heads)]
    Htok = [[] for _ in range(block.heads)]
    gtok = [[] for _ in range(block.heads)]
    atarget = [[] for _ in range(block.heads)]

    with torch.no_grad():
        for _ in range(batches):
            x, _ = gen.sample(batch, rng)
            # Hardware-side current a(q), including fixed-point LN/Q/omega path.
            xq = h1.qround(x.numpy(), v2.F_ACT)
            xnq = h1.layernorm_q(xq, hir.ln1_g, hir.ln1_b)
            qiq = h1.split_heads(h1.dense_linear_q(xnq, hir.Wq, hir.bq), block.heads)
            aqi = h1.stable_a_q(qiq, hir.omega, True)

            # Frozen float target H/g and query interface.
            xnf = block.n1(x)
            qf = block.attn._split(block.attn.q(xnf))
            kf = block.attn._split(block.attn.k(xnf))
            vf = block.attn._split(block.attn.v(xnf))
            qt = block.attn.phi_q(qf, is_query=True)
            kt = block.attn.phi_k(kf, is_query=False)
            H = torch.einsum("bhnm,bhnd->bhmd", kt, vf)
            g = kt.sum(dim=2)
            num = torch.einsum("bhnm,bhmd->bhnd", qt, H)
            den = torch.einsum("bhnm,bhm->bhn", qt, g).unsqueeze(-1).clamp_min(1e-8)
            att = num / den

            B, _, N, M = qt.shape
            dh = att.shape[-1]
            for h in range(block.heads):
                aq[h].append(aqi[:, h].reshape(B*N, M).copy())
                qtarget[h].append(qt[:, h].reshape(B*N, M).cpu().numpy().astype(np.float64))
                # H/g are shared by N queries in each sample; expand to token rows.
                Hh = H[:, h].cpu().numpy().astype(np.float64)
                gh = g[:, h].cpu().numpy().astype(np.float64)
                Htok[h].append(np.repeat(Hh[:, None, :, :], N, axis=1).reshape(B*N, M, dh))
                gtok[h].append(np.repeat(gh[:, None, :], N, axis=1).reshape(B*N, M))
                atarget[h].append(att[:, h].reshape(B*N, dh).cpu().numpy().astype(np.float64))

    pack = []
    for h in range(block.heads):
        pack.append({
            "a": np.concatenate(aq[h], 0),
            "q": np.concatenate(qtarget[h], 0),
            "H": np.concatenate(Htok[h], 0),
            "g": np.concatenate(gtok[h], 0),
            "att": np.concatenate(atarget[h], 0),
        })
    return pack


def attention_from_q(q: np.ndarray, H: np.ndarray, g: np.ndarray) -> np.ndarray:
    num = np.einsum("tm,tmd->td", q, H)
    den = np.einsum("tm,tm->t", q, g)[:, None]
    den = np.maximum(den, 1e-12)
    return num / den


def search_attention_aware_q(block, hir):
    fit = collect_attention_data(block, hir, batches=8, batch=8, seed=424242)
    val = collect_attention_data(block, hir, batches=4, batch=8, seed=515151)
    biases = [2.0, 3.0, 4.0, 5.0, 6.0]
    gains = [0.25, 0.35, 0.5, 0.75, 1.0, 1.4]
    windows = [12, 16, 20, 24, 28, 32]
    selected_specs = []
    selected_decoders = []
    rows: List[dict] = []

    for head in range(block.heads):
        best = None
        for bias in biases:
            for gain in gains:
                for window in windows:
                    spec = (bias, gain, window)
                    try:
                        dq, df = v7.fit_decoder_from_tokens(fit[head]["a"], fit[head]["q"], spec)
                        dmax = float(np.max(np.abs(df)))
                        if not np.isfinite(dmax) or dmax > 128.0:
                            continue
                        qp = v7.predict_tokens(val[head]["a"], spec, dq, v2.F_PQ)
                        ap = attention_from_q(qp, val[head]["H"], val[head]["g"])
                        att_rel = base.rel_np(ap, val[head]["att"])
                        q_rel = base.rel_np(qp, val[head]["q"])
                        score = att_rel + 0.005 * q_rel
                        row = {
                            "head": head, "bias": bias, "gain": gain, "window_ms": window,
                            "attention_rel_l2": att_rel, "q_raw_rel_l2": q_rel,
                            "decoder_max_abs": dmax, "score": score,
                        }
                        rows.append(row)
                        if best is None or score < best[0]:
                            best = (score, spec, dq.copy(), row)
                    except (np.linalg.LinAlgError, FloatingPointError, OverflowError):
                        continue
        if best is None:
            raise RuntimeError(f"no valid Q operating point head {head}")
        selected_specs.append(best[1]); selected_decoders.append(best[2])

    report = {
        "method": "label-free held-out normalized-H/g attention error",
        "grid": {"bias": biases, "gain": gains, "window_ms": windows},
        "selected": [min((r for r in rows if r["head"]==h), key=lambda r:r["score"]) for h in range(block.heads)],
        "top10": sorted(rows, key=lambda r:r["score"])[:10],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({"H1_1_v9_attention_aware_q_search": report}, indent=2), flush=True)
    return selected_specs, selected_decoders


_original_export = base.export_hir

def export_v9(block):
    hir = _original_export(block)
    qspec, qdec = search_attention_aware_q(block, hir)
    kdec, kreport = v7.calibrate_k_decoders(block, hir)
    hir.q_specs = qspec; hir.q_decoders = qdec; hir.k_decoders = kdec
    print(json.dumps({"H1_1_v9_k_calibration": kreport}, indent=2), flush=True)
    return hir


if __name__ == "__main__":
    v2.export_hir = export_v9
    v2.ActualCheckpointRuntime = v3.IndexCorrectRuntime
    v2.main()
