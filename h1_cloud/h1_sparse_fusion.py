#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H1 fusion gate: Linear Attention + selective temporal state + event-sparse SNN-style execution.

This stage deliberately changes EXECUTION before changing the learned function.
The frozen H0 model already has the algebraic state

    H_t = alpha_t H_{t-1} + new writes
    g_t = alpha_t g_{t-1} + new writes

across MaskGIT generation rounds.  H1 replaces the dense block-1 K/V evaluation
with an event-driven path:

    active_t = newly_revealed_t UNION currently_masked_t

Previously committed tokens are represented only by the persistent H/g state and
are not projected through K/V again.  Q is still evaluated for all tokens because
all block-1 outputs feed deeper blocks.  No N x N attention matrix is introduced.

Scientific purpose
------------------
1. Preserve the H0 Linear-Attention / H-g computation.
2. Preserve the Mamba-inspired selective retention alpha_t.
3. Make the temporal sparsity computationally explicit: only active K/V token events
   are evaluated/written, rather than evaluating K/V for all N tokens and masking later.
4. First require numerical equivalence to the dense H0 persistent path; only after this
   control passes should a neural/Izh feature-map approximation be inserted.

This is therefore the first functional fusion gate, not yet the final all-on-chip SNN.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
import types
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

import h0_base as h0


def _split_projected(linear, x: torch.Tensor, heads: int, dh: int) -> torch.Tensor:
    """Linear projection for a [1,n,d] active-token packet -> [1,H,n,Dh]."""
    z = linear(x)
    B, N, D = z.shape
    return z.reshape(B, N, heads, dh).transpose(1, 2)


def persistent_sparse(
    self: h0.LinearAttention,
    x: torch.Tensor,
    known: torch.Tensor,
    state: Optional[h0.PersistentState],
    alpha: torch.Tensor,
) -> Tuple[torch.Tensor, h0.PersistentState, Dict[str, float]]:
    """Exact event-sparse counterpart of H0 LinearAttention.persistent.

    Assumption: the MaskGIT reveal trajectory is monotonic within one generation run.
    If a previously committed token becomes masked again, this implementation raises;
    replacement/delta semantics are a separate gate and must not be silently mixed in.

    The key difference from H0 is that K/V and phi(K) are evaluated only for
    tokens that are not already cached in the persistent state.
    """
    B, N, D = x.shape
    H = self.heads
    M = self.m

    # Q is needed for every token because deeper blocks consume every block-1 output.
    q = self._split(self.q(x))
    pq = self.phi(q, is_query=True)

    if state is None:
        Hp = torch.zeros(B, H, M, self.dh, device=x.device, dtype=x.dtype)
        gp = torch.zeros(B, H, M, device=x.device, dtype=x.dtype)
        prev_known = torch.zeros(B, N, device=x.device, dtype=torch.bool)
    else:
        Hp, gp, prev_known = state.H, state.g, state.known

    remasked = prev_known & (~known)
    if bool(remasked.any()):
        raise RuntimeError(
            "H1 sparse fusion assumes monotonic reveal. A previously known token was re-masked; "
            "use an explicit replacement/delta-state gate instead."
        )

    # Mamba-like retention remains synchronized for H and g.
    Hp = Hp * alpha.reshape(B, H, 1, 1)
    gp = gp * alpha.reshape(B, H, 1)

    new_known = known & (~prev_known)
    masked = ~known
    active = new_known | masked  # == ~prev_known under monotonic reveal

    # Transient masked-token contribution for this round.
    Hm = torch.zeros_like(Hp)
    gm = torch.zeros_like(gp)

    active_counts = []
    new_counts = []
    masked_counts = []

    # Batch loop is intentional: it is the cloud reference for an event packet.
    # There is no receiver-sender N^2 loop and no dense K/V projection on cached tokens.
    for bi in range(B):
        idx = torch.nonzero(active[bi], as_tuple=False).flatten()
        active_counts.append(int(idx.numel()))
        new_counts.append(int(new_known[bi].sum().item()))
        masked_counts.append(int(masked[bi].sum().item()))
        if idx.numel() == 0:
            continue

        xa = x[bi:bi+1, idx, :]
        ka = _split_projected(self.k, xa, self.heads, self.dh)
        va = _split_projected(self.v, xa, self.heads, self.dh)
        pka = self.phi(ka, is_query=False)

        is_new = known[bi, idx]          # active + known => newly committed
        is_mask = ~known[bi, idx]

        if bool(is_new.any()):
            pkn = pka[:, :, is_new, :]
            vn = va[:, :, is_new, :]
            Hp_b = torch.einsum("bhnm,bhnd->bhmd", pkn, vn)
            gp_b = pkn.sum(dim=2)
            Hp[bi:bi+1] = Hp[bi:bi+1] + Hp_b
            gp[bi:bi+1] = gp[bi:bi+1] + gp_b

        if bool(is_mask.any()):
            pkm = pka[:, :, is_mask, :]
            vm = va[:, :, is_mask, :]
            Hm[bi:bi+1] = torch.einsum("bhnm,bhnd->bhmd", pkm, vm)
            gm[bi:bi+1] = pkm.sum(dim=2)

    Htot = Hp + Hm
    gtot = gp + gm
    num = torch.einsum("bhnm,bhmd->bhnd", pq, Htot)
    den = torch.einsum("bhnm,bhm->bhn", pq, gtot).unsqueeze(-1).clamp_min(1e-8)
    y = num / den
    out = self.out(y.transpose(1, 2).reshape(x.shape))

    ns = h0.PersistentState(H=Hp, g=gp, known=known.clone())
    n_active = float(np.mean(active_counts)) if active_counts else 0.0
    n_new = float(np.mean(new_counts)) if new_counts else 0.0
    n_mask = float(np.mean(masked_counts)) if masked_counts else 0.0
    stats = {
        "new_known_mean": n_new,
        "masked_mean": n_mask,
        "state_H_rms": float(Hp.square().mean().sqrt().detach().cpu()),
        "state_g_min": float(gp.min().detach().cpu()),
        "active_kv_tokens_mean": n_active,
        "kv_token_fraction": n_active / float(N),
        "cached_known_mean": float(prev_known.sum(1).float().mean().detach().cpu()),
        "dense_kv_tokens_equivalent": float(N),
    }
    self._last_sparse_stats = stats
    return out, ns, stats


def enable_sparse_block1(model: h0.MaskGIT) -> h0.MaskGIT:
    if model.kind != "linear":
        raise ValueError("H1 sparse fusion is defined for the linear backbone")
    attn = model.blocks[0].attn
    attn.persistent = types.MethodType(persistent_sparse, attn)
    attn._h1_sparse_enabled = True
    return model


def make_reveal_canvases(batch: int, rounds: int, device, seed: int):
    """Deterministic monotonic token reveal sequence used only for equivalence audit."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    target = torch.randint(0, h0.CODEBOOK, (batch, h0.SEQ), generator=g, device=device)
    labels = torch.randint(0, 3, (batch,), generator=g, device=device)
    order = torch.rand(batch, h0.SEQ, generator=g, device=device).argsort(dim=1)
    rank = torch.empty_like(order)
    ar = torch.arange(h0.SEQ, device=device)[None].expand(batch, -1)
    rank.scatter_(1, order, ar)
    canvases = []
    for r in range(rounds):
        reveal = int(round(h0.SEQ * r / max(1, rounds - 1)))
        known = rank < reveal
        cur = torch.full_like(target, h0.MASK_ID)
        cur[known] = target[known]
        canvases.append(cur)
    return labels, canvases


@torch.no_grad()
def dense_sparse_equivalence_audit(
    dense: h0.MaskGIT,
    sparse: h0.MaskGIT,
    device,
    mode: str,
    fixed_alpha: float = 1.0,
    batch: int = 4,
    rounds: int = 8,
    seed: int = 991,
):
    dense.eval(); sparse.eval()
    labels, canvases = make_reveal_canvases(batch, rounds, device, seed)
    sd = ss = None
    rows = []
    max_abs = 0.0
    max_rel = 0.0
    for ri, cur in enumerate(canvases):
        yd, sd, _ = dense.forward_persistent(cur, labels, sd, mode, fixed_alpha)
        ys, ss, sts = sparse.forward_persistent(cur, labels, ss, mode, fixed_alpha)
        diff = yd - ys
        abs_e = float(diff.abs().max().cpu())
        num = torch.linalg.vector_norm(diff.reshape(batch, -1), dim=1)
        den = torch.linalg.vector_norm(yd.reshape(batch, -1), dim=1).clamp_min(1e-12)
        rel = float((num / den).max().cpu())
        max_abs = max(max_abs, abs_e)
        max_rel = max(max_rel, rel)
        rows.append({
            "round": ri,
            "known": int(cur[0].ne(h0.MASK_ID).sum().item()),
            "max_abs": abs_e,
            "rel_l2": rel,
            "active_kv_tokens_mean": sts["active_kv_tokens_mean"],
            "kv_token_fraction": sts["kv_token_fraction"],
            "alpha_mean": sts.get("alpha_mean"),
        })
    return {
        "mode": mode,
        "fixed_alpha": float(fixed_alpha),
        "max_abs": max_abs,
        "max_rel_l2": max_rel,
        "pass_fp32": bool(max_abs <= 1e-4 and max_rel <= 1e-5),
        "rounds": rows,
    }


@torch.no_grad()
def generate_sparse(model: h0.MaskGIT, labels: torch.Tensor, device, steps=10, temp0=0.25,
                    seed=1007, mode="selective", fixed_alpha=1.0):
    model.eval(); labels=labels.to(device); B=len(labels)
    cur=torch.full((B,h0.SEQ),h0.MASK_ID,dtype=torch.long,device=device)
    initial=torch.full((B,),h0.SEQ,dtype=torch.long,device=device)
    gen=torch.Generator(device=device); gen.manual_seed(seed)
    state=None
    rec={k:[] for k in [
        "mask_before","alpha_mean","state_H_rms","active_kv_tokens_mean",
        "kv_token_fraction","cached_known_mean","new_known_mean","masked_mean"
    ]}
    for st in range(steps):
        unk=cur.eq(h0.MASK_ID)
        rec["mask_before"].append(unk.sum(1).detach().cpu().numpy())
        logits,state,stats=model.forward_persistent(cur,labels,state,mode,fixed_alpha)
        p=logits.softmax(-1)
        sampled=torch.multinomial(p.reshape(-1,h0.CODEBOOK),1,generator=gen).reshape(B,h0.SEQ)
        prop=torch.where(unk,sampled,cur)
        sp=p.gather(-1,prop[...,None]).squeeze(-1)
        if st==steps-1:
            cur=prop
        else:
            score_base=torch.where(unk,sp,torch.full_like(sp,float("inf")))
            frac=(st+1)/steps
            desired=torch.floor(initial.float()*math.cos(0.5*math.pi*frac)).long()
            desired=torch.minimum(desired,torch.clamp(unk.sum(1)-1,min=0))
            u=torch.rand(score_base.shape,device=device,generator=gen).clamp_(1e-6,1-1e-6)
            gum=-torch.log(-torch.log(u))
            score=torch.log(score_base.clamp_min(1e-12))+temp0*(1-frac)*gum
            score=torch.where(torch.isinf(score_base),torch.full_like(score,float("inf")),score)
            new=prop.clone()
            for bi in range(B):
                k=int(desired[bi])
                if k:
                    new[bi,torch.topk(score[bi],k,largest=False).indices]=h0.MASK_ID
            cur=new
        for k in rec:
            if k == "mask_before":
                continue
            rec[k].append(float(stats[k]))
    rec["mask_before"]=np.stack(rec["mask_before"])
    for k in rec:
        if k!="mask_before": rec[k]=np.asarray(rec[k],dtype=np.float32)
    return cur.cpu().numpy().astype(np.int16), rec


def eval_sparse_generation(name, model, labels_np, centroids, ref, device, outdir,
                           steps, temp, seed, mode="selective", fixed_alpha=1.0):
    labels=torch.from_numpy(labels_np.astype(np.int64))
    tokens,rec=generate_sparse(model,labels,device,steps,temp,seed,mode,fixed_alpha)
    images=h0.decode_tokens(tokens,centroids)
    probs=h0.classify(ref,images,device)
    pred=probs.argmax(1)
    counts=np.bincount(pred,minlength=10)
    q=counts[:3]/max(1,counts[:3].sum())
    tvd=.5*float(np.abs(q-np.ones(3)/3).sum())
    total_sparse=float(rec["active_kv_tokens_mean"].sum())
    dense_equiv=float(h0.SEQ*steps)
    met={
        "name":name,"mode":mode,"fixed_alpha":float(fixed_alpha),"n":len(labels_np),
        "condition_match":float((pred==labels_np).mean()),
        "valid_012_rate":float(np.isin(pred,[0,1,2]).mean()),
        "mean_classifier_confidence":float(probs.max(1).mean()),
        "confidence_ge_095":float((probs.max(1)>=.95).mean()),
        "TVD_to_balanced_012_among_valid":tvd,
        "class_counts":counts.tolist(),
        "unique_token_fraction":float(len(np.unique(tokens,axis=0))/len(tokens)),
        "morphology":h0.morphology(images),
        "mask_counts_first_sample":[int(x) for x in rec["mask_before"][:,0]],
        "alpha_mean_by_step":rec["alpha_mean"].tolist(),
        "state_H_rms_by_step":rec["state_H_rms"].tolist(),
        "active_kv_tokens_mean_by_step":rec["active_kv_tokens_mean"].tolist(),
        "kv_token_fraction_by_step":rec["kv_token_fraction"].tolist(),
        "cached_known_mean_by_step":rec["cached_known_mean"].tolist(),
        "sparse_kv_token_evaluations_per_sample":total_sparse,
        "dense_kv_token_evaluations_per_sample":dense_equiv,
        "kv_token_reduction_fraction":1.0-total_sparse/dense_equiv,
    }
    np.savez_compressed(outdir/f"{name}_samples.npz",labels=labels_np,tokens=tokens,images=images,pred=pred,probs=probs)
    h0.json_dump(outdir/f"{name}_metrics.json",met)
    return met


def build_linear(args, device, gate=False):
    return h0.MaskGIT(kind="linear",d=args.d,layers=args.layers,heads=args.heads,ff=args.ff,
                      features=args.features,nclass=3,feature_seed=args.seed+101,
                      selective_gate=gate).to(device)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,default=None)
    ap.add_argument("--out",type=Path,default=Path("H1_FUSION"))
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
    ap.add_argument("--eval-n",type=int,default=300)
    ap.add_argument("--ref-epochs",type=int,default=3)
    ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--threads",type=int,default=4)
    args=ap.parse_args()

    h0.seed_all(args.seed); torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data is not None:
        xall,yall,xte,yte=h0.load_mnist_npz(args.data)
    else:
        xall,yall,xte,yte=h0.load_mnist_keras()

    x012,y012=h0.filter_classes(xall,yall)
    x012,y012=h0.subset_per_class(x012,y012,args.n_per_class,args.seed)
    xtr,ytr,xva,yva=h0.stratified_split(x012,y012,.1,args.seed)
    ytr=h0.map_labels_012(ytr); yva=h0.map_labels_012(yva)

    cent=h0.fit_codebook(xtr,args.out/"codebook.npy",args.seed)
    tr=h0.encode_tokens(xtr,cent); va=h0.encode_tokens(xva,cent)
    np.savez_compressed(args.out/"tokenized_012.npz",train_tokens=tr,train_labels=ytr,val_tokens=va,val_labels=yva)

    ref=h0.train_ref_classifier(xall,yall,device,args.ref_epochs)
    ref_test=h0.classify(ref,xte,device)
    ref_acc=float((ref_test.argmax(1)==yte).mean())
    torch.save(ref.state_dict(),args.out/"reference_classifier.pt")

    # Train exactly the H0 linear backbone first; fusion happens after this frozen point.
    base=build_linear(args,device,gate=False)
    train_info=h0.train_backbone(base,tr,ytr,va,yva,args.out/"linear_base",args.epochs_linear,args.batch,device)

    # Selective temporal gate: backbone frozen, same H0 training protocol.
    dense_sel=build_linear(args,device,gate=True)
    cur=dense_sel.state_dict(); bsd=base.state_dict()
    for k,v in bsd.items():
        if k in cur and cur[k].shape==v.shape: cur[k]=v
    dense_sel.load_state_dict(cur)
    h0.train_selective_gate(dense_sel,tr,ytr,va,yva,args.out,device,args.gate_epochs,
                            min(args.batch,64),steps=min(args.steps,6),seed=args.seed)

    # Sparse execution clone: identical learned weights and gate, only execution changes.
    sparse_sel=copy.deepcopy(dense_sel)
    enable_sparse_block1(sparse_sel)

    # Identity/fixed/selective equivalence gates.
    dense_identity=copy.deepcopy(base)
    sparse_identity=copy.deepcopy(base); enable_sparse_block1(sparse_identity)
    audits={
        "identity":dense_sparse_equivalence_audit(dense_identity,sparse_identity,device,"identity",1.0,seed=args.seed+801),
        "fixed_0p98":dense_sparse_equivalence_audit(dense_identity,sparse_identity,device,"fixed",0.98,seed=args.seed+802),
        "selective":dense_sparse_equivalence_audit(dense_sel,sparse_sel,device,"selective",1.0,seed=args.seed+803),
    }
    if not all(v["pass_fp32"] for v in audits.values()):
        h0.json_dump(args.out/"H1_AUDIT_FAIL.json",audits)
        raise RuntimeError(f"H1 dense/sparse equivalence failed: {audits}")

    labs=(np.arange(args.eval_n)%3).astype(np.int64)
    seed_gen=args.seed+1000
    generation={}
    generation["linear_fresh"]=h0.eval_generation("linear_fresh",base,labs,cent,ref,device,args.out,args.steps,args.temp,seed_gen,"fresh")
    generation["dense_selective"]=h0.eval_generation("dense_selective",dense_sel,labs,cent,ref,device,args.out,args.steps,args.temp,seed_gen,"selective")
    generation["sparse_identity"]=eval_sparse_generation("sparse_identity",sparse_identity,labs,cent,ref,device,args.out,args.steps,args.temp,seed_gen,"identity",1.0)
    generation["sparse_fixed_0p98"]=eval_sparse_generation("sparse_fixed_0p98",sparse_identity,labs,cent,ref,device,args.out,args.steps,args.temp,seed_gen,"fixed",0.98)
    generation["sparse_selective"]=eval_sparse_generation("sparse_selective",sparse_sel,labs,cent,ref,device,args.out,args.steps,args.temp,seed_gen,"selective",1.0)

    result={
        "stage":"H1_FULL_DYNAMICS_FUSION_GATE_A",
        "scope":"Linear Attention H/g + Mamba-inspired selective retention + event-sparse K/V execution",
        "device":str(device),
        "reference_classifier_test_acc":ref_acc,
        "model":{"parameters":sum(p.numel() for p in base.parameters()),
                 "best_epoch":train_info["best_epoch"],"best_val_ce":train_info["best_val_ce"]},
        "audits":audits,
        "selective_gate":{
            "theta0":dense_sel.gate.theta0.detach().cpu().tolist(),
            "theta_mask":dense_sel.gate.theta_mask.detach().cpu().tolist(),
            "theta_new":dense_sel.gate.theta_new.detach().cpu().tolist(),
        },
        "generation":generation,
        "claims":{
            "linear_attention_present":True,
            "selective_temporal_state_present":True,
            "event_sparse_kv_execution_present":True,
            "global_qk_matrix_present":False,
            "all_on_chip_snn":False,
            "izh_feature_map_present":False,
            "note":"Izh/LIF phi replacement is intentionally deferred until the three-way algebra/execution fusion passes exactly."
        }
    }
    h0.json_dump(args.out/"H1_RESULTS.json",result)
    print("H1_DONE",json.dumps({
        "audits":audits,
        "reference_classifier_test_acc":ref_acc,
        "generation":{k:{
            "condition_match":v["condition_match"],
            "valid_012_rate":v["valid_012_rate"],
            "kv_token_reduction_fraction":v.get("kv_token_reduction_fraction")
        } for k,v in generation.items()}
    },indent=2),flush=True)


if __name__=="__main__":
    main()
