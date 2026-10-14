#!/usr/bin/env python3
"""H1.1 v2 — actual frozen checkpoint with range-aware fixed-point formats.

Why v2 exists
-------------
The first H1.1 attempt exposed two issues that representative H1 parameters did
not reveal:

1. The trained Q/K feature ranges are strongly asymmetric: Q phi <= ~0.18 while
   K phi can exceed 300.  A single Q*.12 feature format therefore under-resolves
   small Q features even though K is well represented.
2. The frozen PyTorch event-update loop updates H/g without a batch slice; for
   B>1 this broadcasts one sample's delta into every batch item.  That violates
   the intended per-sample linear-attention equations.  H1.1 v2 uses the stated
   mathematical invariant H=sum(pk*v), g=sum(pk) as the float reference and
   records this software residue explicitly.  No scientific operator is changed.

Backend contract refined here:
- activation / dense weights: Q*.14
- Q feature cache: Q*.20
- K feature cache and g: Q*.12
- H resident state: Q*.18 (requantized outer products; int32-safe target)
- Izh membrane: inherited Q*.20
- decoder coefficients: Q*.18

The different Q/K feature scales cancel exactly in the normalized H/g read; H
may also use a different storage scale provided the quotient compensates for it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from SNNTransformBlock_v1_1.SNNTransformBlock_v1_1 import (
    PerHeadIzhPhi,
    _stable_a,
    load_v1_checkpoint_as_v1_1,
)
from spinnaker_backend.h1_fixed_point_emulator import (
    F_ACT,
    F_DEC,
    F_MEM,
    HIR,
    SparseRowsQ,
    dense_linear_q,
    div_round,
    layernorm_q,
    qround,
    rshift_round,
    sparse_linear_q,
    split_heads,
    stable_a_q,
)
from spinnaker_backend.run_h1_1_frozen_checkpoint import (
    AssocGenerator,
    export_hir,
    make_mutable_sequence,
    rel_np,
    sha256_file,
)

F_PQ = 20
F_PK = 12
F_H = 18


@dataclass
class IntState:
    H: np.ndarray   # Q(F_H)
    g: np.ndarray   # Q(F_PK)
    pq: np.ndarray  # Q(F_PQ)
    pk: np.ndarray  # Q(F_PK)
    v: np.ndarray   # Q(F_ACT)


@dataclass
class FloatState:
    H: torch.Tensor
    g: torch.Tensor
    pq: torch.Tensor
    pk: torch.Tensor
    v: torch.Tensor


def izh_decode_scaled(aq: np.ndarray, spec: Tuple[float, float, int], decoder_q: np.ndarray, out_frac: int) -> np.ndarray:
    """H1 fused Izh decoder with selectable output fractional precision."""
    bias, gain, T = spec
    T = int(T)
    if decoder_q.size != T + 1:
        raise ValueError("decoder length != window+1")
    aq = np.asarray(aq, dtype=np.int64)
    a_mem = aq << (F_MEM - F_ACT)
    S_MEM = 1 << F_MEM
    v = np.full_like(aq, -65 * S_MEM)
    c02v = int(round(0.2 * S_MEM))
    u = rshift_round(v * c02v, F_MEM)
    current = int(round(float(bias) * S_MEM)) + rshift_round(
        a_mem * int(round(float(gain) * S_MEM)), F_MEM
    )
    c004 = int(round(0.04 * S_MEM))
    c02 = int(round(0.02 * S_MEM))
    c140 = 140 * S_MEM
    acc = np.full_like(aq, int(decoder_q[0]) << F_MEM, dtype=np.int64)
    for t in range(T):
        v2 = rshift_round(v * v, F_MEM)
        dv = rshift_round(v2 * c004, F_MEM) + 5 * v + c140 - u + current
        v = v + rshift_round(dv, 1)
        v2 = rshift_round(v * v, F_MEM)
        dv = rshift_round(v2 * c004, F_MEM) + 5 * v + c140 - u + current
        v = v + rshift_round(dv, 1)
        du = rshift_round((rshift_round(v * c02v, F_MEM) - u) * c02, F_MEM)
        u = u + du
        fired = v >= 30 * S_MEM
        v = np.where(fired, -65 * S_MEM, v)
        u = np.where(fired, u + 8 * S_MEM, u)
        acc = acc + v * int(decoder_q[t + 1])
    raw = rshift_round(acc, F_MEM + F_DEC - out_frac)
    return np.maximum(raw, 1)


def phi_scaled(xq, omegaq, specs, decoders, is_query: bool, out_frac: int):
    a = stable_a_q(xq, omegaq, is_query)
    out = np.empty_like(a)
    for h in range(a.shape[1]):
        out[:, h] = izh_decode_scaled(a[:, h], specs[h], decoders[h], out_frac)
    return out


def outer_to_H(pk: np.ndarray, v: np.ndarray) -> np.ndarray:
    raw = np.einsum("bhnm,bhnd->bhmd", pk, v, dtype=np.int64)
    return rshift_round(raw, F_PK + F_ACT - F_H)


class ActualCheckpointRuntime:
    def __init__(self, hir: HIR):
        self.p = hir
        self.q_token_evals = 0
        self.k_token_evals = 0
        self.v_token_evals = 0

    def step(self, xq: np.ndarray, state: Optional[IntState], changed: Optional[np.ndarray]):
        p = self.p
        xq = np.asarray(xq, dtype=np.int64)
        B, N, D = xq.shape
        xn = layernorm_q(xq, p.ln1_g, p.ln1_b)
        if state is None:
            q = split_heads(dense_linear_q(xn, p.Wq, p.bq), p.heads)
            k = split_heads(dense_linear_q(xn, p.Wk, p.bk), p.heads)
            v = split_heads(dense_linear_q(xn, p.Wv, p.bv), p.heads)
            pq = phi_scaled(q, p.omega, p.q_specs, p.q_decoders, True, F_PQ)
            pk = phi_scaled(k, p.omega, p.k_specs, p.k_decoders, False, F_PK)
            H = outer_to_H(pk, v)
            g = pk.sum(axis=2, dtype=np.int64)
            self.q_token_evals += B * N; self.k_token_evals += B * N; self.v_token_evals += B * N
        else:
            if changed is None:
                raise ValueError("changed required")
            H, g = state.H.copy(), state.g.copy()
            pq, pk, v = state.pq.copy(), state.pk.copy(), state.v.copy()
            for b in range(B):
                idx = np.nonzero(np.asarray(changed[b], dtype=bool))[0]
                if not idx.size:
                    continue
                xb = xn[b:b+1, idx]
                nq = split_heads(dense_linear_q(xb, p.Wq, p.bq), p.heads)
                nk = split_heads(dense_linear_q(xb, p.Wk, p.bk), p.heads)
                nv = split_heads(dense_linear_q(xb, p.Wv, p.bv), p.heads)
                npq = phi_scaled(nq, p.omega, p.q_specs, p.q_decoders, True, F_PQ)
                npk = phi_scaled(nk, p.omega, p.k_specs, p.k_decoders, False, F_PK)
                oldpk = state.pk[b:b+1, :, idx, :]
                oldv = state.v[b:b+1, :, idx, :]
                H[b:b+1] -= outer_to_H(oldpk, oldv)
                H[b:b+1] += outer_to_H(npk, nv)
                g[b:b+1] -= oldpk.sum(axis=2, dtype=np.int64)
                g[b:b+1] += npk.sum(axis=2, dtype=np.int64)
                pq[b, :, idx, :] = npq[0]
                pk[b, :, idx, :] = npk[0]
                v[b, :, idx, :] = nv[0]
                n = int(idx.size)
                self.q_token_evals += n; self.k_token_evals += n; self.v_token_evals += n

        dh = D // p.heads
        attended = np.empty((B, p.heads, N, dh), dtype=np.int64)
        quotient_shift = F_ACT + F_PK - F_H
        if quotient_shift < 0:
            raise ValueError("chosen H scale would require pre-division right shift")
        for b in range(B):
            for h in range(p.heads):
                for n in range(N):
                    den = sum(int(pq[b,h,n,m]) * int(g[b,h,m]) for m in range(p.features))
                    for d in range(dh):
                        num = sum(int(pq[b,h,n,m]) * int(H[b,h,m,d]) for m in range(p.features))
                        attended[b,h,n,d] = div_round(num << quotient_shift, den)
        attended = attended.transpose(0,2,1,3).reshape(B,N,D)
        a = sparse_linear_q(attended, p.Wo)
        z = xq + a
        zn = layernorm_q(z, p.ln2_g, p.ln2_b)
        hidden = np.maximum(sparse_linear_q(zn, p.W1), 0)
        y = z + sparse_linear_q(hidden, p.W2)
        return y, IntState(H=H,g=g,pq=pq,pk=pk,v=v)


@torch.no_grad()
def correct_float_step(block, x: torch.Tensor, state: Optional[FloatState], changed: Optional[torch.Tensor]):
    """The intended independent-batch v1.1 equations."""
    xn = block.n1(x)
    B,N,_ = x.shape
    if state is None:
        q = block.attn._split(block.attn.q(xn)); k = block.attn._split(block.attn.k(xn)); v = block.attn._split(block.attn.v(xn))
        pq = block.attn.phi_q(q, is_query=True); pk = block.attn.phi_k(k, is_query=False)
        H = torch.einsum("bhnm,bhnd->bhmd", pk, v); g = pk.sum(2)
    else:
        H=state.H.clone(); g=state.g.clone(); pq=state.pq.clone(); pk=state.pk.clone(); v=state.v.clone()
        for b in range(B):
            idx=torch.nonzero(changed[b],as_tuple=False).flatten()
            if idx.numel()==0: continue
            xb=xn[b:b+1,idx]
            nq=block.attn._split(block.attn.q(xb)); nk=block.attn._split(block.attn.k(xb)); nv=block.attn._split(block.attn.v(xb))
            npq=block.attn.phi_q(nq,is_query=True); npk=block.attn.phi_k(nk,is_query=False)
            oldpk=state.pk[b:b+1,:,idx,:]; oldv=state.v[b:b+1,:,idx,:]
            H[b:b+1] -= torch.einsum("bhnm,bhnd->bhmd",oldpk,oldv)
            H[b:b+1] += torch.einsum("bhnm,bhnd->bhmd",npk,nv)
            g[b:b+1] -= oldpk.sum(2); g[b:b+1] += npk.sum(2)
            pq[b,:,idx,:]=npq[0]; pk[b,:,idx,:]=npk[0]; v[b,:,idx,:]=nv[0]
    num=torch.einsum("bhnm,bhmd->bhnd",pq,H)
    den=torch.einsum("bhnm,bhm->bhn",pq,g).unsqueeze(-1).clamp_min(1e-8)
    attended=num/den
    a=block.attn.out(attended.transpose(1,2).reshape(B,N,block.d))
    z=x+a; hidden=F.relu(block.f1(block.n2(z))); y=z+block.f2(hidden)
    return y, FloatState(H=H,g=g,pq=pq,pk=pk,v=v)


@torch.no_grad()
def detect_batch_broadcast_residue(block) -> Dict[str,float]:
    """Compare frozen implementation with intended math on one persistent B=2 update."""
    g=torch.Generator(); g.manual_seed(9091)
    x0=torch.randn(2,16,block.d,generator=g)*0.25
    x1=x0.clone(); x1[0,0]+=0.5
    mask=torch.zeros(2,16,dtype=torch.bool); mask[0,0]=True
    _,s0=block(x0,None,None)
    y_bug,s_bug=block(x1,s0,mask)
    y_ok,s_ok=correct_float_step(block,x1,FloatState(s0.H,s0.g,s0.pq,s0.pk,s0.v),mask)
    return {
        "output_rel_l2": rel_np(y_bug.numpy(),y_ok.numpy()),
        "H_rel_l2": rel_np(s_bug.H.numpy(),s_ok.H.numpy()),
        "g_rel_l2": rel_np(s_bug.g.numpy(),s_ok.g.numpy()),
    }


def save_hir_v2(h: HIR, path: Path, source_sha: str):
    arrays={
        "ln1_g":h.ln1_g,"ln1_b":h.ln1_b,"ln2_g":h.ln2_g,"ln2_b":h.ln2_b,
        "Wq":h.Wq,"bq":h.bq,"Wk":h.Wk,"bk":h.bk,"Wv":h.Wv,"bv":h.bv,"omega":h.omega,
    }
    for name,r in (("Wo",h.Wo),("W1",h.W1),("W2",h.W2)):
        arrays[name+"_row_ptr"]=r.row_ptr; arrays[name+"_col_idx"]=r.col_idx; arrays[name+"_values"]=r.values; arrays[name+"_bias"]=r.bias
    for i,d in enumerate(h.q_decoders): arrays[f"q_decoder_{i}"]=d
    for i,d in enumerate(h.k_decoders): arrays[f"k_decoder_{i}"]=d
    meta={"format":"SNNTransformBlock-v1.1-HIR-H1.1-v2","source_checkpoint_sha256":source_sha,
          "d":h.d,"heads":h.heads,"features":h.features,"ff":h.ff,
          "F_ACT":F_ACT,"F_PQ":F_PQ,"F_PK":F_PK,"F_H":F_H,"F_DEC":F_DEC,
          "q_specs":h.q_specs,"k_specs":h.k_specs,"Wo_nnz":h.Wo.nnz,"W1_nnz":h.W1.nnz,"W2_nnz":h.W2.nnz}
    arrays["metadata_json_utf8"]=np.frombuffer(json.dumps(meta,sort_keys=True).encode(),dtype=np.uint8)
    np.savez_compressed(path,**arrays)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",type=Path,required=True); ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--batch",type=int,default=8); ap.add_argument("--steps",type=int,default=6); ap.add_argument("--change-frac",type=float,default=.25); ap.add_argument("--seed",type=int,default=17171)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4)
    block,ck=load_v1_checkpoint_as_v1_1(args.checkpoint,device="cpu",eval_mode=True)
    hir=export_hir(block); source_sha=sha256_file(args.checkpoint); save_hir_v2(hir,args.out/"SNNTransformBlock_v1_1_HIR_H1_1_v2.npz",source_sha)
    residue=detect_batch_broadcast_residue(block)

    gen=AssocGenerator(16,block.d,8); xs,masks,targets=make_mutable_sequence(gen,batch=args.batch,steps=args.steps,change_frac=args.change_frac,seed=args.seed)
    rt=ActualCheckpointRuntime(hir); fs=None; qs=None; rows=[]; imax={k:0 for k in ("H","g","pq","pk","v")}; float_ranges={k:0.0 for k in ("H","g","pq","pk","v","y")}
    readout=torch.nn.Linear(block.d,8); readout.load_state_dict(ck["readout_state_dict"]); readout.eval()
    pf0=pi0=t0=None
    with torch.no_grad():
        for t,(x,m) in enumerate(zip(xs,masks)):
            yf,fs=correct_float_step(block,x,fs,None if fs is None else m)
            yiq,qs=rt.step(qround(x.numpy(),F_ACT),qs,None if qs is None else m.numpy()); yi=yiq.astype(np.float64)/(1<<F_ACT)
            sf={"H":fs.H.numpy(),"g":fs.g.numpy(),"pq":fs.pq.numpy(),"pk":fs.pk.numpy(),"v":fs.v.numpy()}
            si={"H":qs.H.astype(np.float64)/(1<<F_H),"g":qs.g.astype(np.float64)/(1<<F_PK),"pq":qs.pq.astype(np.float64)/(1<<F_PQ),"pk":qs.pk.astype(np.float64)/(1<<F_PK),"v":qs.v.astype(np.float64)/(1<<F_ACT)}
            row={"step":t,"output_rel_l2":rel_np(yi,yf.numpy()),"H_rel_l2":rel_np(si["H"],sf["H"]),"g_rel_l2":rel_np(si["g"],sf["g"]),"pq_rel_l2":rel_np(si["pq"],sf["pq"]),"pk_rel_l2":rel_np(si["pk"],sf["pk"]),"v_rel_l2":rel_np(si["v"],sf["v"]),"changed_tokens":int(m.sum())}; rows.append(row)
            for k in imax: imax[k]=max(imax[k],int(np.max(np.abs(getattr(qs,k)))))
            for k in ("H","g","pq","pk","v"): float_ranges[k]=max(float_ranges[k],float(np.max(np.abs(sf[k]))))
            float_ranges["y"]=max(float_ranges["y"],float(yf.abs().max()))
            if t==0:
                pf0=readout(yf[:,-1]); pi0=readout(torch.from_numpy(yi).float()[:,-1]); t0=targets[0]
    mx={k:max(r[f"{k}_rel_l2"] for r in rows) for k in ("output","H","g","pq","pk","v")}
    pred_rel=rel_np(pi0.numpy(),pf0.numpy()); fsign=float(((pf0>=0)==(t0>=0)).float().mean()); isign=float(((pi0>=0)==(t0>=0)).float().mean())
    dense=args.batch*16*args.steps; expected=args.batch*16+(args.steps-1)*args.batch*max(1,round(16*args.change_frac))
    int32=(1<<31)-1
    gates={
      "actual_checkpoint":ck.get("format")=="SNNTransformBlock-v1",
      "output_le_1pct":mx["output"]<=.01,"H_le_1pct":mx["H"]<=.01,"g_le_1pct":mx["g"]<=.01,
      "pq_le_1pct":mx["pq"]<=.01,"pk_le_1pct":mx["pk"]<=.01,"v_le_0p5pct":mx["v"]<=.005,
      "prediction_le_1pct":pred_rel<=.01,"sign_drop_le_0p02":isign>=fsign-.02,
      "event_count_exact":rt.q_token_evals==rt.k_token_evals==rt.v_token_evals==expected,
      "event_reduction_ge_0p60":1-rt.q_token_evals/dense>=.60,
      "tail_nnz_1280":hir.Wo.nnz+hir.W1.nnz+hir.W2.nnz==1280,
      "H_int32_safe":imax["H"]<=int32,"g_int32_safe":imax["g"]<=int32,"pq_int32_safe":imax["pq"]<=int32,"pk_int32_safe":imax["pk"]<=int32,"v_int32_safe":imax["v"]<=int32,
      "batch_residue_detected":residue["H_rel_l2"]>1e-6,
    }
    result={"stage":"H1_1_V2_FROZEN_CHECKPOINT_RANGE_AWARE_HIR","PASS":all(gates.values()),
      "source":{"checkpoint_sha256":source_sha,"checkpoint_format":ck.get("format"),"artifact_run_id":38538624735},
      "scientific_scope":{"operator":"intended frozen v1.1 equations unchanged","numerical_change":"asymmetric Q/K feature scales plus independent H storage scale","board_claim":"none"},
      "discovered_python_batch_broadcast_residue":residue,
      "formats":{"F_ACT":F_ACT,"F_PQ":F_PQ,"F_PK":F_PK,"F_H":F_H,"F_DEC":F_DEC},
      "max_rel_l2":mx,"trajectory":rows,
      "task_probe_step0":{"prediction_rel_l2":pred_rel,"float_sign_accuracy":fsign,"integer_sign_accuracy":isign},
      "real_float_max_abs":float_ranges,"resident_integer_max_abs":imax,
      "event_execution":{"q":rt.q_token_evals,"k":rt.k_token_evals,"v":rt.v_token_evals,"dense_per_projection":dense,"expected_event_per_projection":expected,"reduction":1-rt.q_token_evals/dense},
      "sparse_tail":{"total_nnz":hir.Wo.nnz+hir.W1.nnz+hir.W2.nnz,"static_fraction":.25},"gates":gates,
      "decision":"H1.1 PASS: freeze range-aware actual-checkpoint HIR; fix Python batch indexing and proceed H2" if all(gates.values()) else "H1.1 v2 FAIL: continue numerical localization before H2"}
    (args.out/"H1_1_V2_RESULTS.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True)
    if not result["PASS"]: raise SystemExit(2)


if __name__=="__main__": main()
