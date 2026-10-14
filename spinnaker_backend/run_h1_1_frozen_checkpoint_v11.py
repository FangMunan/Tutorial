#!/usr/bin/env python3
"""H1.1 v11 — direct label-free calibration of the existing Q trajectory decoder.

This keeps exactly one linear Izh-trajectory decoder per attention head, as in
the frozen block.  The decoder coefficients are optimized offline against the
internal normalized H/g read on calibration inputs.  No task labels/readout are
used, and the integer runtime is unchanged.

Motivation: v9 showed the remaining error is conditioning of A(q)=q^T H/q^T g,
not state quantization.  Linear least-squares gives excellent unweighted phi
error but does not minimize this rational operator.  Because the decoder itself
is already a calibrated interface parameter, optimizing its coefficients for
that interface's actual downstream invariant does not alter the scientific
operator or add parameters.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

# Safe v4/v9 numerical contract.
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

REPORT = Path("spinnaker_backend/run/H1_1/H1_1_V11_DIRECT_ATTENTION_DECODER.json")


def make_design_tokens(a_tokens: np.ndarray, spec):
    flat = a_tokens.reshape(-1)
    X = v6.fixed_izh_design(flat, spec)
    return X.reshape(a_tokens.shape[0], a_tokens.shape[1], X.shape[-1])


def initial_decoder(a_tokens, q_tokens, spec):
    dq, _ = v7.fit_decoder_from_tokens(a_tokens, q_tokens, spec, max_scalar=32768)
    return dq.astype(np.float64) / (1 << v2.F_DEC)


def optimize_decoder(pack, spec, d0, *, steps=350, lr=2e-3, raw_reg=2e-3):
    # Deterministic token subset bounds CI cost while preserving the full feature axis.
    ntok = pack["a"].shape[0]
    nuse = min(ntok, 640)
    idx = np.linspace(0, ntok - 1, nuse, dtype=np.int64)
    a = pack["a"][idx]
    qt = pack["q"][idx]
    H = pack["H"][idx]
    g = pack["g"][idx]
    at = pack["att"][idx]
    X = make_design_tokens(a, spec)

    Xt = torch.tensor(X, dtype=torch.float64)
    qtt = torch.tensor(qt, dtype=torch.float64)
    Ht = torch.tensor(H, dtype=torch.float64)
    gt = torch.tensor(g, dtype=torch.float64)
    att_t = torch.tensor(at, dtype=torch.float64)
    d = torch.nn.Parameter(torch.tensor(d0, dtype=torch.float64))
    opt = torch.optim.Adam([d], lr=lr)
    floor = 1.0 / (1 << v2.F_PQ)
    att_norm2 = torch.sum(att_t * att_t).clamp_min(1e-18)
    q_norm2 = torch.sum(qtt * qtt).clamp_min(1e-18)

    best = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        q = torch.clamp(torch.einsum("tmp,p->tm", Xt, d), min=floor)
        num = torch.einsum("tm,tmd->td", q, Ht)
        den = torch.einsum("tm,tm->t", q, gt).unsqueeze(-1).clamp_min(1e-12)
        att = num / den
        loss_att = torch.sum((att - att_t) ** 2) / att_norm2
        loss_q = torch.sum((q - qtt) ** 2) / q_norm2
        loss = loss_att + raw_reg * loss_q
        loss.backward()
        torch.nn.utils.clip_grad_norm_([d], 10.0)
        opt.step()
        if best is None or float(loss.detach()) < best[0]:
            best = (float(loss.detach()), d.detach().clone())
        if step in (150, 250):
            for group in opt.param_groups:
                group["lr"] *= 0.3
    return best[1].cpu().numpy()


def validate_decoder(pack, spec, dq):
    qp = v7.predict_tokens(pack["a"], spec, dq, v2.F_PQ)
    ap = v9.attention_from_q(qp, pack["H"], pack["g"])
    return {
        "attention_rel_l2": base.rel_np(ap, pack["att"]),
        "q_raw_rel_l2": base.rel_np(qp, pack["q"]),
    }


def calibrate_q(block, hir):
    fit = v9.collect_attention_data(block, hir, batches=10, batch=8, seed=424242)
    val = v9.collect_attention_data(block, hir, batches=5, batch=8, seed=515151)
    # Start from the best robust v9 operating points; also test nearby alternatives.
    candidates = {
        0: [(5.0,0.35,28),(5.0,0.25,20),(4.0,0.25,16),(4.0,0.35,28)],
        1: [(4.0,1.0,12),(3.0,1.4,28),(4.0,0.5,32),(2.0,0.75,12)],
    }
    selected_specs=[]; selected_dec=[]; rows=[]
    for head in range(block.heads):
        best=None
        for spec in candidates[head]:
            d0=initial_decoder(fit[head]["a"],fit[head]["q"],spec)
            dopt=optimize_decoder(fit[head],spec,d0)
            if not np.all(np.isfinite(dopt)) or float(np.max(np.abs(dopt)))>128.0:
                continue
            dq=h1.qround(dopt,v2.F_DEC)
            met=validate_decoder(val[head],spec,dq)
            row={"head":head,"bias":spec[0],"gain":spec[1],"window_ms":spec[2],
                 **met,"decoder_max_abs":float(np.max(np.abs(dopt)))}
            rows.append(row)
            if best is None or met["attention_rel_l2"]<best[0]:
                best=(met["attention_rel_l2"],spec,dq.copy(),row)
        if best is None: raise RuntimeError(f"no valid direct-attention decoder head {head}")
        selected_specs.append(best[1]); selected_dec.append(best[2])
    report={"method":"label-free Adam on existing linear trajectory decoder; objective=normalized H/g attention + small raw-phi regularizer",
            "selected":[min((r for r in rows if r["head"]==h),key=lambda r:r["attention_rel_l2"]) for h in range(block.heads)],
            "all_candidates":rows}
    REPORT.parent.mkdir(parents=True,exist_ok=True); REPORT.write_text(json.dumps(report,indent=2))
    print(json.dumps({"H1_1_v11_direct_attention_decoder":report},indent=2),flush=True)
    return selected_specs,selected_dec


_original_export=base.export_hir

def export_v11(block):
    hir=_original_export(block)
    qs,qd=calibrate_q(block,hir)
    kd,krep=v7.calibrate_k_decoders(block,hir)
    hir.q_specs=qs; hir.q_decoders=qd; hir.k_decoders=kd
    print(json.dumps({"H1_1_v11_k_calibration":krep},indent=2),flush=True)
    return hir


if __name__=="__main__":
    v2.export_hir=export_v11
    v2.ActualCheckpointRuntime=v3.IndexCorrectRuntime
    v2.main()
