#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H1-B: Izhikevich dynamical feature-map fusion on top of H1-A.

Frozen H1-A ingredients retained:
  * normalized positive random-feature Linear Attention (H/g algebra),
  * persistent H/g state across MaskGIT generation rounds,
  * learned selective retention alpha_t,
  * event-sparse K/V execution in persistent block 1.

H1-B changes exactly one interface:

    stable random-feature preactivation a
        -> exp(a)/sqrt(M)                       [exact H1-A]

becomes

    stable random-feature preactivation a
        -> constant current I = bias + gain*a
        -> standard Izhikevich RS dynamics
        -> short membrane trajectory [v(1),...,v(T)]
        -> fixed linear decoder + positivity clamp
        -> phi_IZH(a)                           [H1-B]

The same scalar dynamical map is used in every head and every Linear-Attention
block; each block retains its own frozen random projection omega.  Decoder fitting
uses only calibration preactivations collected from the training split. Candidate
Izh operating points are selected on a held-out masked-token batch by end-to-end
fresh-logit error, not by generation labels.

This is a SOFTWARE dynamical gate.  It does not claim SpiNNaker equivalence yet.
A later H1-C gate must replay the selected interface on SpiNNaker and replace the
software Izh trajectories by hardware-generated trajectories.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import h0_base as h0
import h1_sparse_fusion as h1


# -----------------------------------------------------------------------------
# Izhikevich scalar dynamics
# -----------------------------------------------------------------------------

@torch.no_grad()
def izh_rs_voltage(a: torch.Tensor, bias: float, gain: float, window_ms: int) -> torch.Tensor:
    """Standard RS Izhikevich trajectory, 1-ms step with two half-steps for v.

    a is an arbitrary-shape tensor of scalar feature preactivations. Output shape is
    a.shape + [window_ms]. Initial state is reset for each feature evaluation, so this
    module is a dynamical nonlinear FEATURE MAP, not the inter-generation memory.
    Persistent sequence memory remains H/g.
    """
    x = a.float()
    v = torch.full_like(x, -65.0)
    u = 0.2 * v
    I = float(bias) + float(gain) * x
    out = []
    for _ in range(int(window_ms)):
        dv = 0.04*v*v + 5.0*v + 140.0 - u + I
        v = v + 0.5*dv
        dv = 0.04*v*v + 5.0*v + 140.0 - u + I
        v = v + 0.5*dv
        u = u + 0.02*(0.2*v - u)
        fired = v >= 30.0
        # Record the reset state, matching an exposed post-update membrane state.
        v = torch.where(fired, torch.full_like(v, -65.0), v)
        u = torch.where(fired, u + 8.0, u)
        out.append(v)
    return torch.stack(out, dim=-1)


@torch.no_grad()
def trajectory_design(a: torch.Tensor, bias: float, gain: float, window_ms: int) -> torch.Tensor:
    v = izh_rs_voltage(a, bias, gain, window_ms)
    one = torch.ones(*a.shape, 1, device=a.device, dtype=v.dtype)
    return torch.cat([one, v], dim=-1)


@torch.no_grad()
def fit_decoder(a_train: torch.Tensor, target_train: torch.Tensor,
                bias: float, gain: float, window_ms: int, ridge: float=1e-5) -> torch.Tensor:
    X = trajectory_design(a_train, bias, gain, window_ms).double()
    y = target_train.double()
    p = X.shape[-1]
    eye = torch.eye(p, dtype=torch.float64, device=X.device)
    eye[0,0] = 0.0  # do not regularize intercept
    w = torch.linalg.solve(X.T @ X + float(ridge)*eye, X.T @ y)
    return w.float()


class IzhDynamicFeatures(nn.Module):
    """Drop-in dynamical replacement for h0.PositiveRandomFeatures."""
    def __init__(self, exact_phi: h0.PositiveRandomFeatures, *, bias: float, gain: float,
                 window_ms: int, decoder: torch.Tensor):
        super().__init__()
        self.register_buffer("omega", exact_phi.omega.detach().clone())
        self.register_buffer("decoder", decoder.detach().float().clone())
        self.features = int(exact_phi.features)
        self.head_dim = int(exact_phi.head_dim)
        self.bias = float(bias)
        self.gain = float(gain)
        self.window_ms = int(window_ms)
        self._last_clamp_fraction = 0.0

    def stable_preactivation(self, x: torch.Tensor, *, is_query: bool) -> torch.Tensor:
        xx = x * (self.head_dim ** -0.25)
        dash = torch.einsum("bhnd,hmd->bhnm", xx, self.omega)
        diag = 0.5 * (xx*xx).sum(dim=-1, keepdim=True)
        logits = dash - diag
        if is_query:
            shift = logits.amax(dim=-1, keepdim=True).detach()
            return (logits - shift).clamp(min=-30.0, max=0.0)
        return logits.clamp(min=-30.0, max=30.0)

    def forward(self, x: torch.Tensor, *, is_query: bool) -> torch.Tensor:
        a = self.stable_preactivation(x, is_query=is_query)
        shp = a.shape
        flat = a.reshape(-1)
        # Chunking caps peak memory without changing dynamics.
        chunks = []
        clamp_n = 0
        total_n = 0
        chunk = 350000
        for i in range(0, flat.numel(), chunk):
            aa = flat[i:i+chunk]
            X = trajectory_design(aa, self.bias, self.gain, self.window_ms)
            y = X @ self.decoder.to(X.dtype)
            clamp_n += int((y <= 1e-8).sum().item())
            total_n += int(y.numel())
            chunks.append(y.clamp_min(1e-8))
        self._last_clamp_fraction = clamp_n / max(1, total_n)
        return torch.cat(chunks, dim=0).reshape(shp)


# -----------------------------------------------------------------------------
# Exact preactivation collection
# -----------------------------------------------------------------------------

@torch.no_grad()
def exact_stable(attn: h0.LinearAttention, x: torch.Tensor, is_query: bool) -> torch.Tensor:
    z = attn._split(attn.q(x) if is_query else attn.k(x))
    phi = attn.phi
    xx = z * (phi.head_dim ** -0.25)
    dash = torch.einsum("bhnd,hmd->bhnm", xx, phi.omega)
    diag = 0.5*(xx*xx).sum(dim=-1, keepdim=True)
    logits = dash - diag
    if is_query:
        shift = logits.amax(dim=-1, keepdim=True).detach()
        return (logits-shift).clamp(-30.0,0.0)
    return logits.clamp(-30.0,30.0)


@torch.no_grad()
def collect_empirical_a(model: h0.MaskGIT, tokens: np.ndarray, labels: np.ndarray, device,
                        seed: int, batches: int=8, batch_size: int=24,
                        samples_per_layer_side: int=2500) -> Tuple[torch.Tensor, Dict]:
    """Pool preactivations from all attention blocks and both Q/K sides."""
    model.eval()
    rng = np.random.default_rng(seed)
    vals: List[torch.Tensor] = []
    by_side = {"q":[],"k":[]}
    by_layer = {str(i):[] for i in range(model.layers)}
    for bi in range(batches):
        idx = rng.choice(len(tokens), size=min(batch_size,len(tokens)), replace=False)
        tok = torch.from_numpy(tokens[idx]).to(device)
        y = torch.from_numpy(labels[idx]).to(device)
        # deterministic random mask for calibration coverage
        g = torch.Generator(device=device); g.manual_seed(seed + 1009*bi)
        ratio = torch.rand(len(tok), generator=g, device=device) * 0.95 + 0.02
        mask = torch.rand(tok.shape, generator=g, device=device) < ratio[:,None]
        inp = tok.clone(); inp[mask] = h0.MASK_ID
        z = model.embed(inp,y)
        for li,b in enumerate(model.blocks):
            for side,isq in (("q",True),("k",False)):
                aa = exact_stable(b.attn,z,isq).reshape(-1)
                n = min(samples_per_layer_side,aa.numel())
                gi = torch.Generator(device=device); gi.manual_seed(seed+bi*10007+li*101+(1 if isq else 2))
                take = torch.randint(0,aa.numel(),(n,),generator=gi,device=device)
                s = aa[take].detach().cpu()
                vals.append(s); by_side[side].append(s); by_layer[str(li)].append(s)
            z = b(z)
    allv = torch.cat(vals).float()
    def qstats(v):
        v=torch.cat(v).float()
        qs=torch.quantile(v,torch.tensor([0,0.001,0.01,0.1,0.5,0.9,0.99,0.999,1.0]))
        return {"n":int(v.numel()),"quantiles":qs.tolist(),"mean":float(v.mean()),"std":float(v.std())}
    stats={"all":qstats([allv]),"q":qstats(by_side["q"]),"k":qstats(by_side["k"]),
           "layers":{k:qstats(v) for k,v in by_layer.items()}}
    return allv,stats


def target_phi(a: torch.Tensor, features: int) -> torch.Tensor:
    return torch.exp(a.clamp(-30.0,30.0)) * (features ** -0.5) + 1e-6


@torch.no_grad()
def scalar_metrics(a: torch.Tensor, target: torch.Tensor, decoder: torch.Tensor,
                   bias: float, gain: float, window_ms: int) -> Dict[str,float]:
    X=trajectory_design(a,bias,gain,window_ms)
    raw=X@decoder.to(X.dtype)
    pred=raw.clamp_min(1e-8)
    rel=float(torch.linalg.vector_norm(pred-target)/torch.linalg.vector_norm(target).clamp_min(1e-12))
    mask=target>max(1e-5,float(torch.quantile(target,0.10)))
    point=((pred-target).abs()/target.clamp_min(1e-8))[mask]
    return {
        "scalar_rel_l2":rel,
        "median_point_rel":float(point.median()) if point.numel() else float("nan"),
        "p95_point_rel":float(torch.quantile(point,0.95)) if point.numel() else float("nan"),
        "clamp_fraction":float((raw<=1e-8).float().mean()),
    }


def replace_all_phi(model: h0.MaskGIT, spec: Dict, decoder: torch.Tensor) -> h0.MaskGIT:
    for b in model.blocks:
        old=b.attn.phi
        b.attn.phi=IzhDynamicFeatures(old,bias=spec["bias"],gain=spec["gain"],
                                      window_ms=spec["window_ms"],decoder=decoder).to(next(model.parameters()).device)
    return model


@torch.no_grad()
def masked_batch(tokens,labels,device,seed,n=8):
    rng=np.random.default_rng(seed)
    idx=rng.choice(len(tokens),size=min(n,len(tokens)),replace=False)
    tok=torch.from_numpy(tokens[idx]).to(device); y=torch.from_numpy(labels[idx]).to(device)
    g=torch.Generator(device=device);g.manual_seed(seed+17)
    mask=torch.rand(tok.shape,generator=g,device=device)<0.65
    inp=tok.clone();inp[mask]=h0.MASK_ID
    return inp,y


@torch.no_grad()
def fresh_logit_error(exact: h0.MaskGIT, candidate: h0.MaskGIT,
                      tokens,labels,device,seed,n=8) -> Dict[str,float]:
    inp,y=masked_batch(tokens,labels,device,seed,n)
    ze=exact(inp,y); zi=candidate(inp,y)
    d=zi-ze
    return {
        "max_abs":float(d.abs().max()),
        "rel_l2":float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(ze).clamp_min(1e-12)),
        "argmax_agreement":float((zi.argmax(-1)==ze.argmax(-1)).float().mean()),
    }


@torch.no_grad()
def persistent_logit_error(exact: h0.MaskGIT, candidate: h0.MaskGIT,
                           tokens,labels,device,seed,steps=6,n=8,mode="selective") -> Dict:
    rng=np.random.default_rng(seed)
    idx=rng.choice(len(tokens),size=min(n,len(tokens)),replace=False)
    tok=torch.from_numpy(tokens[idx]).to(device); y=torch.from_numpy(labels[idx]).to(device)
    gen=torch.Generator(device=device);gen.manual_seed(seed+33)
    canv=h0.teacher_mask_sequence(tok,steps,gen)
    se=si=None;rows=[]
    for t,cur in enumerate(canv):
        ze,se,_=exact.forward_persistent(cur,y,se,mode)
        zi,si,_=candidate.forward_persistent(cur,y,si,mode)
        d=zi-ze
        rows.append({"step":t,"rel_l2":float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(ze).clamp_min(1e-12)),
                     "argmax_agreement":float((zi.argmax(-1)==ze.argmax(-1)).float().mean()),
                     "max_abs":float(d.abs().max())})
    return {"max_rel_l2":max(r["rel_l2"] for r in rows),
            "min_argmax_agreement":min(r["argmax_agreement"] for r in rows),"rows":rows}


# -----------------------------------------------------------------------------
# Main H1-B gate
# -----------------------------------------------------------------------------

def build_linear(args,device,gate=False):
    return h0.MaskGIT(kind="linear",d=args.d,layers=args.layers,heads=args.heads,ff=args.ff,
                      features=args.features,nclass=3,feature_seed=args.seed+101,
                      selective_gate=gate).to(device)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,default=None)
    ap.add_argument("--out",type=Path,default=Path("H1B_IZH"))
    ap.add_argument("--n-per-class",type=int,default=2000)
    ap.add_argument("--epochs-linear",type=int,default=18)
    ap.add_argument("--gate-epochs",type=int,default=2)
    ap.add_argument("--batch",type=int,default=256)
    ap.add_argument("--d",type=int,default=128)
    ap.add_argument("--layers",type=int,default=4)
    ap.add_argument("--heads",type=int,default=4)
    ap.add_argument("--ff",type=int,default=512)
    ap.add_argument("--features",type=int,default=128)
    ap.add_argument("--steps",type=int,default=10)
    ap.add_argument("--temp",type=float,default=.25)
    ap.add_argument("--eval-n",type=int,default=30)
    ap.add_argument("--ref-epochs",type=int,default=3)
    ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--threads",type=int,default=4)
    args=ap.parse_args()

    h0.seed_all(args.seed);torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data is not None: xall,yall,xte,yte=h0.load_mnist_npz(args.data)
    else: xall,yall,xte,yte=h0.load_mnist_keras()
    x012,y012=h0.filter_classes(xall,yall)
    x012,y012=h0.subset_per_class(x012,y012,args.n_per_class,args.seed)
    xtr,ytr,xva,yva=h0.stratified_split(x012,y012,.1,args.seed)
    ytr=h0.map_labels_012(ytr);yva=h0.map_labels_012(yva)
    cent=h0.fit_codebook(xtr,args.out/"codebook.npy",args.seed)
    tr=h0.encode_tokens(xtr,cent);va=h0.encode_tokens(xva,cent)

    ref=h0.train_ref_classifier(xall,yall,device,args.ref_epochs)
    ref_acc=float((h0.classify(ref,xte,device).argmax(1)==yte).mean())

    # Rebuild/freeze the exact H1-A model under the same protocol.
    base=build_linear(args,device,False)
    train_info=h0.train_backbone(base,tr,ytr,va,yva,args.out/"linear_base",args.epochs_linear,args.batch,device)
    dense_sel=build_linear(args,device,True)
    cur=dense_sel.state_dict();bsd=base.state_dict()
    for k,v in bsd.items():
        if k in cur and cur[k].shape==v.shape:cur[k]=v
    dense_sel.load_state_dict(cur)
    h0.train_selective_gate(dense_sel,tr,ytr,va,yva,args.out,device,args.gate_epochs,
                            min(args.batch,64),steps=min(args.steps,6),seed=args.seed)
    exact_sparse=copy.deepcopy(dense_sel);h1.enable_sparse_block1(exact_sparse)

    # Collect empirical a from exact trained model; split without using test/generation labels.
    avec,astats=collect_empirical_a(base,tr,ytr,device,args.seed+4001)
    g=torch.Generator();g.manual_seed(args.seed+5001)
    perm=torch.randperm(avec.numel(),generator=g)
    nfit=int(0.75*len(perm)); ai=avec[perm[:nfit]].to(device); av=avec[perm[nfit:]].to(device)
    # Add a deterministic calibration grid over the robust empirical range.
    lo=float(torch.quantile(avec,0.001)); hi=float(torch.quantile(avec,0.999))
    grid=torch.linspace(lo,hi,4096,device=device)
    ai_fit=torch.cat([ai,grid])
    yi=target_phi(ai_fit,args.features); yv=target_phi(av,args.features)

    candidates=[
        {"bias":4.0,"gain":2.0,"window_ms":6},
        {"bias":6.0,"gain":0.5,"window_ms":12},
        {"bias":6.0,"gain":1.0,"window_ms":8},
        {"bias":8.0,"gain":0.5,"window_ms":12},
        {"bias":8.0,"gain":1.0,"window_ms":8},
        {"bias":10.0,"gain":0.5,"window_ms":12},
    ]
    scan=[]
    best=None
    for ci,spec in enumerate(candidates):
        dec=fit_decoder(ai_fit,yi,**spec,ridge=1e-5)
        sm=scalar_metrics(av,yv,dec,**spec)
        cand=copy.deepcopy(base);replace_all_phi(cand,spec,dec)
        fm=fresh_logit_error(base,cand,va,yva,device,args.seed+6000+ci,n=8)
        row={**spec,**sm,"fresh_logit_probe":fm,"decoder":dec.detach().cpu().tolist()}
        scan.append(row)
        print("IZH_CANDIDATE",json.dumps({k:v for k,v in row.items() if k!="decoder"}),flush=True)
        score=fm["rel_l2"]
        if best is None or score<best[0]:best=(score,spec,dec,row)

    _,selected,decoder,selected_row=best
    torch.save({"spec":selected,"decoder":decoder.detach().cpu(),"empirical_stats":astats},args.out/"izh_feature_map.pt")

    # Exact sparse and Izh sparse differ ONLY at phi.
    izh_sparse=copy.deepcopy(dense_sel);h1.enable_sparse_block1(izh_sparse);replace_all_phi(izh_sparse,selected,decoder)
    fresh_izh=copy.deepcopy(base);replace_all_phi(fresh_izh,selected,decoder)

    fresh_audit=fresh_logit_error(base,fresh_izh,va,yva,device,args.seed+7001,n=16)
    pers_audit=persistent_logit_error(exact_sparse,izh_sparse,va,yva,device,args.seed+7002,steps=6,n=8,mode="selective")

    # Small end-to-end generation gate. Same labels and random sampling seed.
    labs=(np.arange(args.eval_n)%3).astype(np.int64);gen_seed=args.seed+1000
    exact_gen=h1.eval_sparse_generation("h1a_exact_sparse_selective",exact_sparse,labs,cent,ref,device,args.out,
                                        args.steps,args.temp,gen_seed,"selective",1.0)
    izh_gen=h1.eval_sparse_generation("h1b_izh_sparse_selective",izh_sparse,labs,cent,ref,device,args.out,
                                      args.steps,args.temp,gen_seed,"selective",1.0)

    cond_drop=exact_gen["condition_match"]-izh_gen["condition_match"]
    valid_drop=exact_gen["valid_012_rate"]-izh_gen["valid_012_rate"]
    gate={
        # First software fusion gate: tight enough to detect a broken feature interface,
        # intentionally looser than the eventual SpiNNaker publication gate.
        "fresh_logit_rel_l2_le_0p15":fresh_audit["rel_l2"]<=0.15,
        "persistent_logit_rel_l2_le_0p20":pers_audit["max_rel_l2"]<=0.20,
        "generation_condition_drop_le_0p10":cond_drop<=0.10,
        "generation_valid_drop_le_0p10":valid_drop<=0.10,
    }
    gate["PASS"]=all(gate.values())

    results={
        "stage":"H1B_IZH_DYNAMIC_FEATURE_FUSION",
        "scope":"all four Linear-Attention phi maps replaced by software Izh RS trajectory decoders; H1-A selective H/g + sparse K/V retained",
        "device":str(device),"reference_classifier_test_acc":ref_acc,
        "model":{"parameters":sum(p.numel() for p in base.parameters()),"best_epoch":train_info["best_epoch"],"best_val_ce":train_info["best_val_ce"]},
        "empirical_preactivation_stats":astats,
        "candidate_scan":scan,
        "selected":{"spec":selected,"scalar":{k:v for k,v in selected_row.items() if k not in ("decoder","fresh_logit_probe")},
                    "selection_probe":selected_row["fresh_logit_probe"],"decoder":decoder.detach().cpu().tolist()},
        "audits":{"fresh_logits":fresh_audit,"persistent_selective_logits":pers_audit},
        "generation":{"h1a_exact":exact_gen,"h1b_izh":izh_gen,
                      "condition_match_drop":cond_drop,"valid_012_drop":valid_drop},
        "gate":gate,
        "claims":{
            "linear_attention_Hg":True,"selective_temporal_state":True,"event_sparse_kv":True,
            "izh_dynamic_phi_all_attention_blocks":True,"global_qk_matrix":False,
            "spinnaker_hardware":False,"all_on_chip":False,
            "important":"Izh state resets for each phi evaluation; persistent generation memory is H/g, not Izh v/u."
        }
    }
    h0.json_dump(args.out/"H1B_RESULTS.json",results)
    print("H1B_DONE",json.dumps({
        "selected":results["selected"],"audits":results["audits"],
        "generation":{k:{kk:vv for kk,vv in v.items() if kk in ("condition_match","valid_012_rate","kv_token_reduction_fraction")} if isinstance(v,dict) else v for k,v in results["generation"].items()},
        "gate":gate
    },indent=2),flush=True)
    if not gate["PASS"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
