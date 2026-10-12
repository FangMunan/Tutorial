#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""H6: 784-pixel SNN_Transform MaskGIT with frozen PGI ionic-mask protocol.

Scientific target
-----------------
Move from H4's 196 lossless 2x2 patch tokens to the final physical interface:
    28x28 = 784 pixel tokens, token alphabet {0,1,MASK}.
The generator is unconditional on MNIST digits {0,1,2}.

The SNN_Transform core is unchanged in kind:
  * positive random-feature normalized linear attention (H/g)
  * block-1 persistent H/g across generation rounds
  * per-head selective retention gate
  * blocks 2..L fresh each generation round
  * event-sparse semantics for new/changed K/V retained for later hardware mapping

Controllers
-----------
standard:
  standard cosine exact-count MaskGIT corruption during training and confidence-based
  monotonic demasking during generation.

ionic:
  frozen measured PGI binary-event scanner replaces MASK LOCATIONS only. The canonical
  ionic NPZ is never re-thresholded, ranked, shuffled, normalized, concatenated across
  physical sequence boundaries, or spatially transformed.

Frozen PGI scanner contract (from the accepted prior protocol)
---------------------------------------------------------------
For one image the pixel cursor scans 0,1,...,783,0,1,... . Ionic bits are consumed
continuously from the selected physical sequence:
  bit 0 -> no transition
  bit 1 -> transition iff current pixel is eligible
A bit=1 on an ineligible pixel is consumed but does not count toward the successful
transition quota. Stop when successful transitions == quota. Pixel cursor resets for
each image; ionic cursor persists within each physical sequence and wraps ONLY inside
that same sequence. Physical split:
  train: ionseq_001..003; validation: ionseq_004; generation: ionseq_005.

H6-A starts with the standard controller to prove that the 784-pixel SNN_Transform can
learn high-quality generation. H6-B uses the same code/model family with --controller ionic
once the canonical measured PGI_ionic_binary.npz is supplied.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "h3_maskgit28"))
sys.path.insert(0, str(ROOT / "h1_cloud"))
import train_snn_transform_maskgit28 as base
import train_snn_transform_maskgit28_v2 as v2
import h0_base as h0

SIDE = 28
NPIX = 784
DATA_VOCAB = 2
MASK_ID = 2
VOCAB = 3
TRAIN_IDS = ("ionseq_001", "ionseq_002", "ionseq_003")
VAL_IDS = ("ionseq_004",)
TEST_IDS = ("ionseq_005",)


def configure_pixel_globals():
    h0.SIDE = SIDE
    h0.PATCH = 1
    h0.GRID = SIDE
    h0.SEQ = NPIX
    h0.CODEBOOK = DATA_VOCAB
    h0.MASK_ID = MASK_ID
    h0.VOCAB = VOCAB
    h0.GEN_CLASSES = (0, 1, 2)
    base.PATCH = 1
    base.GRID = SIDE
    base.SEQ = NPIX
    base.CODEBOOK = DATA_VOCAB
    base.MASK_ID = MASK_ID
    base.VOCAB = VOCAB


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def json_dump(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def flatten_pixels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint8)
    if x.ndim != 3 or x.shape[1:] != (28, 28):
        raise ValueError(f"expected [N,28,28], got {x.shape}")
    if not np.all(np.isin(np.unique(x), [0, 1])):
        raise ValueError("pixel tokenizer requires binarized MNIST")
    return x.reshape(len(x), NPIX).astype(np.int64)


def unflatten_pixels(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t)
    if t.ndim != 2 or t.shape[1] != NPIX:
        raise ValueError(f"expected [N,784], got {t.shape}")
    if np.any((t < 0) | (t > 1)):
        raise ValueError("final pixel tokens must be 0/1")
    return t.reshape(-1, 28, 28).astype(np.uint8)


def load_canonical_ionic(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"canonical ionic NPZ not found: {path}")
    d = np.load(path, allow_pickle=False)
    required = {"bits_concat", "sequence_ids", "sequence_offsets"}
    missing = required - set(d.files)
    if missing:
        raise KeyError(f"canonical ionic NPZ missing keys: {sorted(missing)}")
    bits = np.asarray(d["bits_concat"], dtype=np.uint8)
    ids = np.asarray(d["sequence_ids"]).astype(str)
    offsets = np.asarray(d["sequence_offsets"], dtype=np.int64)
    if bits.ndim != 1 or not np.all(np.isin(np.unique(bits), [0, 1])):
        raise ValueError("bits_concat must be a 1-D binary array")
    if len(offsets) != len(ids) + 1 or offsets[0] != 0 or offsets[-1] != len(bits):
        raise ValueError("invalid sequence_offsets")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("every measured ionic sequence must be non-empty")
    by_id = {}
    for i, sid in enumerate(ids.tolist()):
        if sid in by_id:
            raise ValueError(f"duplicate ionic sequence ID: {sid}")
        by_id[sid] = bits[offsets[i]:offsets[i + 1]].copy()
    return by_id


class IonicForwardPool:
    """Persistent independent physical streams for exact-count forward corruption."""
    def __init__(self, by_id: dict[str, np.ndarray], selected_ids):
        self.ids = tuple(selected_ids)
        for sid in self.ids:
            if sid not in by_id:
                raise KeyError(f"missing ionic sequence {sid}; available={sorted(by_id)}")
        self.bits = [np.asarray(by_id[sid], dtype=np.uint8) for sid in self.ids]
        self.cursors = np.zeros(len(self.ids), dtype=np.int64)
        self.wraps = np.zeros(len(self.ids), dtype=np.int64)
        self.sample_counter = 0
        self.total_reads = self.total_ones = self.total_zeros = 0

    def mask_batch(self, quotas: np.ndarray):
        quotas = np.asarray(quotas, dtype=np.int64)
        out = np.zeros((len(quotas), NPIX), dtype=np.bool_)
        for j, quota in enumerate(quotas.tolist()):
            if not (0 <= quota <= NPIX):
                raise ValueError(f"invalid quota {quota}")
            seq = (self.sample_counter + j) % len(self.ids)
            bits = self.bits[seq]
            cur = int(self.cursors[seq])
            pixel = 0
            success = 0
            guard = 0
            while success < quota:
                bit = int(bits[cur]); cur += 1
                if cur >= len(bits):
                    cur = 0; self.wraps[seq] += 1
                p = pixel; pixel = (pixel + 1) % NPIX
                self.total_reads += 1; guard += 1
                if bit == 1:
                    self.total_ones += 1
                    if not out[j, p]:
                        out[j, p] = True; success += 1
                else:
                    self.total_zeros += 1
                if guard > 10_000_000:
                    raise RuntimeError("ionic forward scanner guard exceeded")
            self.cursors[seq] = cur
        self.sample_counter += len(quotas)
        return out

    def stats(self):
        return {
            "sequence_ids": list(self.ids), "cursors": self.cursors.astype(int).tolist(),
            "wraps": self.wraps.astype(int).tolist(), "sample_counter": int(self.sample_counter),
            "total_reads": int(self.total_reads), "total_ones": int(self.total_ones),
            "total_zeros": int(self.total_zeros),
        }


def build_maskgit_reveal_quotas(steps: int):
    current = NPIX
    rows = []
    for st in range(steps):
        if st == steps - 1:
            target = 0
        else:
            ratio = float(st + 1) / float(steps)
            target = int(math.floor(NPIX * math.cos(0.5 * math.pi * ratio)))
            target = max(1, min(current - 1, target)) if current > 1 else 0
        quota = current - target
        rows.append({"step": st + 1, "mask_before": current, "mask_after": target,
                     "reveal_quota": quota})
        current = target
    if sum(r["reveal_quota"] for r in rows) != NPIX:
        raise RuntimeError("MaskGIT reveal quotas do not sum to 784")
    return rows


def build_ionic_reveal_maps(bits: np.ndarray, steps: int, n_images: int):
    schedule = build_maskgit_reveal_quotas(steps)
    quotas = [r["reveal_quota"] for r in schedule]
    reveal = np.zeros((n_images, NPIX), dtype=np.uint8)
    cur = 0
    total_wraps = 0
    image_reads = np.zeros(n_images, dtype=np.int64)
    step_reads = np.zeros((n_images, steps), dtype=np.int64)
    for im in range(n_images):
        eligible = np.ones(NPIX, dtype=np.bool_)
        pixel = 0
        for st, quota in enumerate(quotas):
            success = 0; guard = 0
            while success < quota:
                bit = int(bits[cur]); cur += 1
                if cur >= len(bits):
                    cur = 0; total_wraps += 1
                p = pixel; pixel = (pixel + 1) % NPIX
                image_reads[im] += 1; step_reads[im, st] += 1; guard += 1
                if bit == 1 and eligible[p]:
                    eligible[p] = False
                    reveal[im, p] = st + 1
                    success += 1
                if guard > 10_000_000:
                    raise RuntimeError("ionic reverse scanner guard exceeded")
    if np.any(reveal == 0):
        raise RuntimeError("ionic reveal map left residual MASK pixels")
    diag = {
        "schedule": schedule, "final_cursor": int(cur), "total_wraps": int(total_wraps),
        "mean_reads_per_image": float(image_reads.mean()), "median_reads_per_image": float(np.median(image_reads)),
        "image_reads": image_reads.tolist(), "step_reads": step_reads.tolist(),
    }
    return reveal, diag


def synthetic_ionic_for_smoke(seed=123):
    rng = np.random.default_rng(seed)
    return {f"ionseq_{i:03d}": (rng.random(8192) < (0.35 + 0.03*i)).astype(np.uint8)
            for i in range(1, 6)}


def build_model(args, device):
    m = h0.MaskGIT(kind="linear", d=args.d, layers=args.layers, heads=args.heads,
                   ff=args.ff, features=args.features, nclass=1,
                   feature_seed=args.seed + 101, selective_gate=True).to(device)
    v2.init_maskgit_style(m)
    return m


def standard_training_mask(tok: torch.Tensor, min_mask_rate: float):
    B, N = tok.shape
    r = torch.rand(B, device=tok.device)
    ratio = torch.cos(0.5 * math.pi * r).clamp(min=min_mask_rate, max=1.0)
    nmask = torch.round(ratio * N).long().clamp(1, N)
    score = torch.rand(B, N, device=tok.device)
    rank = score.argsort(dim=1).argsort(dim=1)
    mask = rank < nmask[:, None]
    inp = tok.clone(); inp[mask] = MASK_ID
    return inp, mask, ratio


def ionic_training_mask(tok: torch.Tensor, min_mask_rate: float, pool: IonicForwardPool):
    B, N = tok.shape
    r = torch.rand(B, device=tok.device)
    ratio = torch.cos(0.5 * math.pi * r).clamp(min=min_mask_rate, max=1.0)
    quotas = torch.round(ratio * N).long().clamp(1, N).cpu().numpy()
    mask_np = pool.mask_batch(quotas)
    mask = torch.from_numpy(mask_np).to(tok.device)
    inp = tok.clone(); inp[mask] = MASK_ID
    return inp, mask, ratio


def run_epoch(model, loader, device, args, optimizer=None, controller="standard",
              ionic_pool: Optional[IonicForwardPool] = None, global_step=0, total_steps=1,
              deterministic_seed=None):
    train = optimizer is not None
    model.train(train)
    saved = torch.random.get_rng_state() if deterministic_seed is not None else None
    if deterministic_seed is not None:
        torch.manual_seed(deterministic_seed)
    ls = ac = mr_sum = n = 0.0
    try:
        for (tok,) in loader:
            tok = tok.to(device)
            labels = torch.zeros(len(tok), dtype=torch.long, device=device)
            if train:
                lr = base.lr_for_step(global_step, total_steps, args.warmup_steps, args.lr, args.min_lr)
                for pg in optimizer.param_groups: pg["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
            if controller == "standard":
                inp, mask, ratio = standard_training_mask(tok, args.min_mask_rate)
            elif controller == "ionic":
                if ionic_pool is None: raise RuntimeError("ionic controller requires IonicForwardPool")
                inp, mask, ratio = ionic_training_mask(tok, args.min_mask_rate, ionic_pool)
            else:
                raise ValueError(controller)
            logits = model(inp, labels)
            loss = F.cross_entropy(logits[mask], tok[mask], label_smoothing=args.label_smoothing)
            acc = (logits[mask].argmax(-1) == tok[mask]).float().mean()
            if train:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); global_step += 1
            bs = len(tok); ls += loss.item()*bs; ac += acc.item()*bs; mr_sum += ratio.mean().item()*bs; n += bs
    finally:
        if saved is not None: torch.random.set_rng_state(saved)
    return {"loss":ls/n, "masked_token_acc":ac/n, "mean_mask_ratio":mr_sum/n}, global_step


def train_backbone(model, tr_tok, va_tok, out, device, args, controller, by_id=None):
    tr = DataLoader(TensorDataset(torch.from_numpy(tr_tok)), batch_size=args.batch, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(va_tok)), batch_size=args.batch, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.96), weight_decay=args.weight_decay)
    total = args.epochs * len(tr); step=0; best=float("inf"); best_ep=0; hist=[]
    train_pool = IonicForwardPool(by_id, TRAIN_IDS) if controller == "ionic" else None
    for ep in range(1, args.epochs+1):
        t0=time.time()
        a,step=run_epoch(model,tr,device,args,opt,controller,train_pool,step,total)
        val_pool=IonicForwardPool(by_id,VAL_IDS) if controller=="ionic" else None
        b,_=run_epoch(model,va,device,args,None,controller,val_pool,0,1,args.seed+50000)
        rec={"epoch":ep,"train_ce":a["loss"],"train_mask_acc":a["masked_token_acc"],
             "train_mask_ratio":a["mean_mask_ratio"],"val_ce":b["loss"],
             "val_mask_acc":b["masked_token_acc"],"val_mask_ratio":b["mean_mask_ratio"],
             "lr":opt.param_groups[0]["lr"],"seconds":time.time()-t0,"global_step":step}
        if train_pool is not None: rec["ionic_train_reads"] = train_pool.total_reads
        hist.append(rec); print("EPOCH",json.dumps(rec),flush=True)
        if b["loss"] < best:
            best=b["loss"]; best_ep=ep
            torch.save({"model":model.state_dict(),"epoch":ep,"best":best,"args":vars(args)},out/"best.pt")
        torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"epoch":ep,"best":best,"args":vars(args)},out/"latest.pt")
        with (out/"history.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
    ck=torch.load(out/"best.pt",map_location=device,weights_only=False); model.load_state_dict(ck["model"])
    return {"best_epoch":best_ep,"best_val_ce":best,"optimizer_updates":step,
            "ionic_train_stats":None if train_pool is None else train_pool.stats(),"history":hist}


@torch.no_grad()
def identity_audit(model, device, batch=2, rounds=5, seed=123):
    model.eval(); g=torch.Generator(device=device); g.manual_seed(seed)
    truth=torch.randint(0,DATA_VOCAB,(batch,NPIX),device=device,generator=g)
    labels=torch.zeros(batch,dtype=torch.long,device=device)
    perm=torch.argsort(torch.rand(batch,NPIX,device=device,generator=g),dim=1)
    rank=torch.empty_like(perm); rank.scatter_(1,perm,torch.arange(NPIX,device=device)[None].expand(batch,-1))
    state=None; max_abs=max_rel=0.0; rows=[]
    for r in range(rounds):
        reveal=int(round(NPIX*r/max(1,rounds-1)))
        known=rank<reveal; cur=torch.full_like(truth,MASK_ID); cur[known]=truth[known]
        fresh=model(cur,labels); pers,state,stats=model.forward_persistent(cur,labels,state,"identity",1.0)
        d=fresh-pers; ma=float(d.abs().max().cpu()); rel=float((d.norm()/fresh.norm().clamp_min(1e-12)).cpu())
        max_abs=max(max_abs,ma); max_rel=max(max_rel,rel)
        rows.append({"round":r,"reveal":reveal,"max_abs":ma,"rel_l2":rel,"state_H_rms":stats["state_H_rms"]})
    return {"max_abs":max_abs,"max_rel_l2":max_rel,"pass_fp32":bool(max_abs<1e-4 and max_rel<1e-6),"rounds":rows}


@torch.no_grad()
def generate_standard(model,n,device,args,mode,seed):
    labels=torch.zeros(n,dtype=torch.long)
    tok,rec=h0.generate(model,labels,device,steps=args.decode_steps,temp0=args.choice_temperature,
                        seed=seed,mode=mode,record=True)
    return tok.astype(np.int64),rec


@torch.no_grad()
def generate_ionic_exact(model,n,device,args,by_id,mode,seed):
    if TEST_IDS[0] not in by_id: raise KeyError(f"missing {TEST_IDS[0]}")
    reveal,diag=build_ionic_reveal_maps(np.asarray(by_id[TEST_IDS[0]],dtype=np.uint8),args.decode_steps,n)
    model.eval(); gen=torch.Generator(device=device); gen.manual_seed(seed)
    outs=[]; alpha_all=[]; h_all=[]; traj_first=[]
    for b0 in range(0,n,args.gen_batch):
        b=min(args.gen_batch,n-b0)
        stepmap=torch.from_numpy(reveal[b0:b0+b]).to(device=device,dtype=torch.long)
        cur=torch.full((b,NPIX),MASK_ID,dtype=torch.long,device=device)
        labels=torch.zeros(b,dtype=torch.long,device=device); state=None
        alpha=[]; hr=[]; first=[]
        for st in range(args.decode_steps):
            if mode=="fresh":
                logits=model(cur,labels); stats={"alpha_mean":float("nan"),"state_H_rms":float("nan")}
            else:
                logits,state,stats=model.forward_persistent(cur,labels,state,mode)
            p=logits.softmax(-1)
            select=stepmap.eq(st+1)
            if select.any():
                vals=torch.multinomial(p[select],1,generator=gen).squeeze(1)
                cur[select]=vals
            alpha.append(stats["alpha_mean"]); hr.append(stats["state_H_rms"])
            if b0==0: first.append(cur[0].detach().cpu().numpy().astype(np.int16))
        if cur.eq(MASK_ID).any(): raise RuntimeError("ionic generation ended with residual MASK")
        outs.append(cur.cpu().numpy().astype(np.int64)); alpha_all.append(alpha); h_all.append(hr)
        if b0==0: traj_first=np.stack(first)
    rec={"alpha_mean":np.asarray(alpha_all[0],np.float32),"state_H_rms":np.asarray(h_all[0],np.float32),
         "first_post":traj_first,"reveal_step_first":reveal[0].reshape(28,28),"scanner":diag}
    return np.concatenate(outs,axis=0),rec


def save_grid(images,path,title):
    base.save_grid(images,path,100,title)


def save_pixel_trajectory(arr,path,title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arr=np.asarray(arr); T=len(arr); cols=6; rows=math.ceil(T/cols)
    fig,ax=plt.subplots(rows,cols,figsize=(cols*2,rows*2)); ax=np.asarray(ax).reshape(-1)
    for i,a in enumerate(ax):
        a.axis("off")
        if i<T:
            t=arr[i]; im=np.where(t.reshape(28,28)==MASK_ID,0.5,t.reshape(28,28)).astype(float)
            a.imshow(im,cmap="gray",vmin=0,vmax=1,interpolation="nearest"); a.set_title(f"step {i+1}")
    fig.suptitle(title); plt.tight_layout(); fig.savefig(path,dpi=160,bbox_inches="tight"); plt.close(fig)


def train_gate_small(model,tr_tok,out,device,args):
    if args.gate_epochs <= 0: return []
    n=min(len(tr_tok),args.gate_samples)
    tok=tr_tok[:n]; lab=np.zeros(n,dtype=np.int64)
    return h0.train_selective_gate(model,tok,lab,tok[:min(256,n)],lab[:min(256,n)],out,device,
                                   epochs=args.gate_epochs,batch=args.gate_batch,steps=args.gate_steps,
                                   lr=args.gate_lr,seed=args.seed)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=Path("H6_PIXEL784"))
    ap.add_argument("--controller",choices=["standard","ionic"],default="standard")
    ap.add_argument("--ionic",type=Path,default=None,help="canonical PGI_ionic_binary.npz for ionic controller")
    ap.add_argument("--epochs",type=int,default=20); ap.add_argument("--batch",type=int,default=32)
    ap.add_argument("--n-per-class",type=int,default=2000)
    ap.add_argument("--d",type=int,default=128); ap.add_argument("--layers",type=int,default=4); ap.add_argument("--heads",type=int,default=4)
    ap.add_argument("--ff",type=int,default=512); ap.add_argument("--features",type=int,default=128)
    ap.add_argument("--lr",type=float,default=3e-4); ap.add_argument("--min-lr",type=float,default=1e-5); ap.add_argument("--warmup-steps",type=int,default=300)
    ap.add_argument("--weight-decay",type=float,default=0.05); ap.add_argument("--label-smoothing",type=float,default=0.1); ap.add_argument("--min-mask-rate",type=float,default=0.5)
    ap.add_argument("--gate-epochs",type=int,default=1); ap.add_argument("--gate-samples",type=int,default=1024); ap.add_argument("--gate-batch",type=int,default=16); ap.add_argument("--gate-steps",type=int,default=4); ap.add_argument("--gate-lr",type=float,default=5e-3)
    ap.add_argument("--decode-steps",type=int,default=24); ap.add_argument("--choice-temperature",type=float,default=4.5); ap.add_argument("--ngen",type=int,default=300); ap.add_argument("--gen-batch",type=int,default=32)
    ap.add_argument("--seed",type=int,default=7); ap.add_argument("--threads",type=int,default=4); ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()

    configure_pixel_globals(); seed_all(args.seed); args.out.mkdir(parents=True,exist_ok=True)
    if args.smoke:
        args.epochs=1; args.n_per_class=64; args.d=64; args.layers=2; args.ff=256; args.features=32; args.batch=8
        args.gate_epochs=1; args.gate_samples=64; args.gate_batch=8; args.gate_steps=3; args.ngen=12; args.gen_batch=6
    torch.set_num_threads(args.threads or min(8,os.cpu_count() or 1)); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device",device,"threads",torch.get_num_threads(),flush=True)

    xtr_all,ytr_all,xte_all,yte_all=base.load_mnist(); x012,y012=base.subset012(xtr_all,ytr_all); xte012,_=base.subset012(xte_all,yte_all)
    xtr,ytr,xva,yva=base.stratified_train_val(x012,y012,0.10,args.seed); xtr,ytr=base.subset_per_class(xtr,ytr,args.n_per_class,args.seed)
    tr_tok=flatten_pixels(xtr); va_tok=flatten_pixels(xva)
    by_id=None
    if args.controller=="ionic":
        by_id=synthetic_ionic_for_smoke(args.seed+77) if args.smoke and args.ionic is None else load_canonical_ionic(args.ionic)
    elif args.ionic is not None:
        by_id=load_canonical_ionic(args.ionic)
    print("data",xtr.shape,xva.shape,xte012.shape,"tokens",tr_tok.shape,"controller",args.controller,flush=True)

    model=build_model(args,device); nparam=sum(p.numel() for p in model.parameters()); print("parameters",nparam,flush=True)
    train_summary=train_backbone(model,tr_tok,va_tok,args.out,device,args,args.controller,by_id)
    audit=identity_audit(model,device,batch=2,rounds=5,seed=args.seed+1234); json_dump(args.out/"identity_audit.json",audit); print("IDENTITY",json.dumps(audit),flush=True)
    if not audit["pass_fp32"]: raise RuntimeError(f"persistent identity gate failed: {audit}")

    fresh_tok,fresh_rec=generate_standard(model,args.ngen,device,args,"fresh",args.seed+10000)
    fresh_img=unflatten_pixels(fresh_tok); np.savez_compressed(args.out/"generation_standard_fresh.npz",samples=fresh_img,tokens=fresh_tok)
    save_grid(fresh_img,args.out/"generation_standard_fresh_grid.png","H6 784-pixel SNN_Transform: standard MaskGIT")
    save_pixel_trajectory(fresh_rec["post"][:,0],args.out/"trajectory_standard_fresh.png","H6 standard trajectory")

    gate_hist=train_gate_small(model,tr_tok,args.out,device,args)
    sel_tok,sel_rec=generate_standard(model,args.ngen,device,args,"selective",args.seed+10000)
    sel_img=unflatten_pixels(sel_tok); np.savez_compressed(args.out/"generation_standard_selective.npz",samples=sel_img,tokens=sel_tok,alpha_mean=sel_rec["alpha_mean"],state_H_rms=sel_rec["state_H_rms"])
    save_grid(sel_img,args.out/"generation_standard_selective_grid.png","H6 784-pixel SNN_Transform: selective standard MaskGIT")

    ionic_tok=ionic_rec=ionic_img=None
    if by_id is not None and TEST_IDS[0] in by_id:
        ionic_tok,ionic_rec=generate_ionic_exact(model,args.ngen,device,args,by_id,"selective",args.seed+10000)
        ionic_img=unflatten_pixels(ionic_tok); np.savez_compressed(args.out/"generation_ionic_selective.npz",samples=ionic_img,tokens=ionic_tok,alpha_mean=ionic_rec["alpha_mean"],state_H_rms=ionic_rec["state_H_rms"])
        save_grid(ionic_img,args.out/"generation_ionic_selective_grid.png","H6 784-pixel SNN_Transform: frozen ionic mask")
        save_pixel_trajectory(ionic_rec["first_post"],args.out/"trajectory_ionic_selective.png","H6 ionic trajectory")
        json_dump(args.out/"ionic_scanner_diagnostics.json",ionic_rec["scanner"])

    print("training shared MNIST evaluator",flush=True); seed_all(args.seed+30000)
    ref,refhist=base.train_ref10(xtr_all,ytr_all,xte_all,yte_all,device,epochs=4,batch=256)
    with (args.out/"ref_classifier10_history.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(refhist[0])); w.writeheader(); w.writerows(refhist)
    fresh_met=base.evaluate_generated("h6_standard_fresh",fresh_img,ref,xte012,x012,device)
    sel_met=base.evaluate_generated("h6_standard_selective",sel_img,ref,xte012,x012,device)
    ionic_met=None if ionic_img is None else base.evaluate_generated("h6_ionic_selective",ionic_img,ref,xte012,x012,device)
    train_prior=np.bincount(y012,minlength=3).astype(float); train_prior/=train_prior.sum()
    for met in [fresh_met,sel_met,ionic_met]:
        if met is None: continue
        p=np.asarray(met["class_prob_0_to_9"],dtype=float); met["train012_empirical_prior"]=train_prior.tolist()
        met["TVD_to_train012_prior"]=float(0.5*(np.abs(p[:3]-train_prior).sum()+p[3:].sum()))

    summary={
      "status":"H6 784-pixel unconditional SNN_Transform MaskGIT",
      "controller_training":args.controller,
      "task":"unconditional MNIST-012","tokens":"784 individual binary pixels","input_vocab":"0/1/MASK","output_vocab":"0/1",
      "architecture":{"d":args.d,"layers":args.layers,"heads":args.heads,"ff":args.ff,"features_per_head":args.features,"parameters":nparam,
                      "attention":"positive-RF normalized linear H/g","persistent_block":0,"selective_gate":"per-head retention"},
      "training":{"epochs":args.epochs,"n_per_class":args.n_per_class,"best_epoch":train_summary["best_epoch"],"best_val_ce":train_summary["best_val_ce"],"optimizer_updates":train_summary["optimizer_updates"],"ionic_train_stats":train_summary["ionic_train_stats"]},
      "identity_audit":audit,"standard_fresh":fresh_met,"standard_selective":sel_met,"ionic_selective":ionic_met,
      "paired_selective_vs_fresh":{"token_hamming":float(np.mean(fresh_tok!=sel_tok)),"pixel_hamming":float(np.mean(fresh_img!=sel_img))},
      "selective_alpha_mean_by_step":[float(x) for x in sel_rec["alpha_mean"]],"selective_state_H_rms_by_step":[float(x) for x in sel_rec["state_H_rms"]],
      "gate_history":gate_hist,
      "ionic_contract":"canonical measured binary stream; pixel scanner 0..783; bit0 skip; bit1 transition iff eligible; success quota; pixel cursor reset/image; ionic cursor persistent/sequence; same-sequence wrap only",
      "scientific_boundary":"software positive-feature H6; no Izh/fixed-point/SpiNNaker claim yet"
    }
    json_dump(args.out/"FINAL_METRICS.json",summary)
    torch.save({"model":model.state_dict(),"args":vars(args),"summary":summary},args.out/"snn_transform_h6_pixel784_final.pt")
    print("FINAL",json.dumps(summary),flush=True)

if __name__=="__main__": main()
