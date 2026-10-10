#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SNN_Transform MaskGIT-28 cloud experiment.

Frozen task shell from the user's prior MaskGIT-28 experiment:
  * unconditional MNIST digits {0,1,2}
  * exact lossless 2x2 binary patch tokenizer: 28x28 -> 14x14=196 tokens
  * vocab=16 visual codes + [MASK]
  * random cosine masked-token training (minimum mask rate 0.5)
  * 24-step fully-masked parallel MaskGIT decoding with annealed Gumbel confidence

Only the backbone is replaced by the project's SNN_Transform reference:
  * positive random-feature normalized linear attention
  * persistent H/g in block 1 across generation rounds
  * per-head selective retention gate (Mamba-inspired state transition)
  * blocks 2..L recomputed fresh each generation round

Training is intentionally split into:
  A) full backbone masked-token learning using fresh linear attention;
  B) trajectory training of the selective retention gate on monotonic reveal paths.
This isolates whether the linear-transform backbone can learn image generation before
asking the recurrent state transition to improve/retain it.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.linalg import sqrtm
except Exception:
    sqrtm = None

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "h1_cloud"))
import h0_base as h0

# -----------------------------------------------------------------------------
# Frozen MaskGIT-28 task contract
# -----------------------------------------------------------------------------
PATCH = 2
GRID = 14
SEQ = 196
CODEBOOK = 16
MASK_ID = 16
VOCAB = 17
GEN_CLASSES = (0, 1, 2)


def configure_h0_globals():
    """Reuse the already-audited SNN_Transform implementation at a 196-token grid."""
    h0.SIDE = 28
    h0.PATCH = PATCH
    h0.GRID = GRID
    h0.SEQ = SEQ
    h0.CODEBOOK = CODEBOOK
    h0.MASK_ID = MASK_ID
    h0.VOCAB = VOCAB
    h0.GEN_CLASSES = GEN_CLASSES


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def json_dump(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Data / tokenizer (faithful to prior local MaskGIT-28 harness)
# -----------------------------------------------------------------------------
def load_mnist():
    xtr, ytr, xte, yte = h0.load_mnist_keras()
    # Prior harness binarized at 0.5/128.
    xtr = (xtr >= 0.5).astype(np.uint8)
    xte = (xte >= 0.5).astype(np.uint8)
    return xtr, ytr.astype(np.int64), xte, yte.astype(np.int64)


def subset012(x, y):
    m = np.isin(y, [0, 1, 2])
    return x[m], y[m]


def stratified_train_val(x, y, val_fraction=0.10, seed=7):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in (0, 1, 2):
        ids = np.where(y == c)[0]
        rng.shuffle(ids)
        nv = max(1, int(round(len(ids) * val_fraction)))
        va.append(ids[:nv]); tr.append(ids[nv:])
    tr = np.concatenate(tr); va = np.concatenate(va)
    rng.shuffle(tr); rng.shuffle(va)
    return x[tr], y[tr], x[va], y[va]


def subset_per_class(x, y, n_per_class, seed):
    if n_per_class <= 0:
        return x, y
    rng = np.random.default_rng(seed)
    ids=[]
    for c in (0,1,2):
        z=np.where(y==c)[0]; rng.shuffle(z); ids.append(z[:min(n_per_class,len(z))])
    ids=np.concatenate(ids); rng.shuffle(ids)
    return x[ids], y[ids]


def encode_2x2_np(images):
    x=np.asarray(images,dtype=np.uint8); b=len(x)
    p=x.reshape(b,GRID,PATCH,GRID,PATCH).transpose(0,1,3,2,4)
    bits=p.reshape(b,SEQ,4)
    w=np.array([8,4,2,1],dtype=np.uint8)
    return (bits*w[None,None,:]).sum(-1).astype(np.int64)


def decode_2x2_np(tokens):
    t=np.asarray(tokens,dtype=np.int64)
    bits=np.stack([(t>>3)&1,(t>>2)&1,(t>>1)&1,t&1],axis=-1)
    p=bits.reshape(-1,GRID,GRID,PATCH,PATCH).transpose(0,1,3,2,4)
    return p.reshape(-1,28,28).astype(np.uint8)


def tokenizer_self_test():
    rng=np.random.default_rng(123)
    x=rng.integers(0,2,size=(8,28,28),dtype=np.uint8)
    assert np.array_equal(x, decode_2x2_np(encode_2x2_np(x)))


# -----------------------------------------------------------------------------
# Matched MaskGIT training shell
# -----------------------------------------------------------------------------
def cosine_mask_ratio(r):
    return torch.cos(0.5*math.pi*r)


def make_training_mask(tokens, min_mask_rate=0.5):
    b,l=tokens.shape
    r=torch.rand(b,device=tokens.device)
    ratio=cosine_mask_ratio(r).clamp(min=min_mask_rate,max=1.0)
    nmask=torch.round(ratio*l).long().clamp(1,l)
    scores=torch.rand(b,l,device=tokens.device)
    ranks=scores.argsort(dim=1).argsort(dim=1)
    mask=ranks<nmask[:,None]
    x=tokens.clone(); x[mask]=MASK_ID
    return x,mask,ratio


def lr_for_step(step,total,warmup,max_lr,min_lr):
    if step<warmup:
        return max_lr*float(step+1)/max(1,warmup)
    p=(step-warmup)/max(1,total-warmup); p=min(max(p,0.0),1.0)
    return min_lr+0.5*(max_lr-min_lr)*(1+math.cos(math.pi*p))


def run_epoch(model,loader,device,optimizer=None,label_smoothing=0.1,min_mask_rate=0.5,
              global_step=0,total_steps=1,warmup=500,max_lr=3e-4,min_lr=1e-5,
              deterministic_seed=None):
    train=optimizer is not None
    model.train(train)
    if deterministic_seed is not None:
        cpu_state=torch.random.get_rng_state(); torch.manual_seed(deterministic_seed)
    else:
        cpu_state=None
    loss_sum=acc_sum=mask_sum=n=0.0
    try:
        for tokens in loader:
            tok=tokens[0].to(device)
            labels=torch.zeros(len(tok),dtype=torch.long,device=device)  # unconditional
            if train:
                lr=lr_for_step(global_step,total_steps,warmup,max_lr,min_lr)
                for pg in optimizer.param_groups: pg["lr"]=lr
                optimizer.zero_grad(set_to_none=True)
            inp,mask,mr=make_training_mask(tok,min_mask_rate)
            logits=model(inp,labels)
            loss=F.cross_entropy(logits[mask],tok[mask],label_smoothing=label_smoothing)
            acc=(logits[mask].argmax(-1)==tok[mask]).float().mean()
            if train:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); global_step+=1
            bs=len(tok); loss_sum+=loss.item()*bs; acc_sum+=acc.item()*bs; mask_sum+=mr.mean().item()*bs; n+=bs
    finally:
        if cpu_state is not None: torch.random.set_rng_state(cpu_state)
    return {"loss":loss_sum/n,"masked_token_acc":acc_sum/n,"mean_mask_ratio":mask_sum/n},global_step


def train_backbone(model,tr_tok,va_tok,out,device,args):
    train_loader=DataLoader(TensorDataset(torch.from_numpy(tr_tok)),batch_size=args.batch,shuffle=True)
    val_loader=DataLoader(TensorDataset(torch.from_numpy(va_tok)),batch_size=args.batch,shuffle=False)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(0.9,0.96),weight_decay=args.weight_decay)
    total=args.epochs*len(train_loader); step=0; best=float("inf"); best_epoch=0; hist=[]
    for ep in range(1,args.epochs+1):
        t0=time.time()
        tr,step=run_epoch(model,train_loader,device,opt,args.label_smoothing,args.min_mask_rate,
                          step,total,args.warmup_steps,args.lr,args.min_lr)
        va,_=run_epoch(model,val_loader,device,None,0.0,args.min_mask_rate,
                       deterministic_seed=args.seed+50000)
        rec={"epoch":ep,"train_ce":tr["loss"],"train_mask_acc":tr["masked_token_acc"],
             "train_mask_ratio":tr["mean_mask_ratio"],"val_ce":va["loss"],
             "val_mask_acc":va["masked_token_acc"],"val_mask_ratio":va["mean_mask_ratio"],
             "lr":opt.param_groups[0]["lr"],"seconds":time.time()-t0,"global_step":step}
        hist.append(rec); print(json.dumps(rec),flush=True)
        if va["loss"]<best:
            best=va["loss"]; best_epoch=ep
            torch.save({"model":model.state_dict(),"epoch":ep,"best":best,"args":vars(args)},out/"best.pt")
        torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"epoch":ep,"best":best,"args":vars(args)},out/"latest.pt")
        with (out/"train_history.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
    ck=torch.load(out/"best.pt",map_location=device,weights_only=False); model.load_state_dict(ck["model"])
    return {"best_epoch":best_epoch,"best_val_ce":best,"optimizer_updates":step,"history":hist}


# -----------------------------------------------------------------------------
# Gate trajectory training (Mamba-like temporal state)
# -----------------------------------------------------------------------------
def train_gate(model,tr_tok,out,device,args):
    # Reuse the project's audited teacher-forced persistent-state training routine.
    labels=np.zeros(len(tr_tok),dtype=np.int64)
    hist=h0.train_selective_gate(model,tr_tok,labels,tr_tok[:min(512,len(tr_tok))],labels[:min(512,len(labels))],
                                 out,device,epochs=args.gate_epochs,batch=args.gate_batch,
                                 steps=args.gate_steps,lr=args.gate_lr,seed=args.seed)
    return hist


# -----------------------------------------------------------------------------
# Evaluation utilities preserved from prior MaskGIT-28 harness
# -----------------------------------------------------------------------------
class RefCNN(nn.Module):
    def __init__(self):
        super().__init__(); self.c1=nn.Conv2d(1,32,3,padding=1); self.c2=nn.Conv2d(32,64,3,padding=1)
        self.fc1=nn.Linear(64*7*7,128); self.fc2=nn.Linear(128,10)
    def forward(self,x,return_features=False):
        x=F.relu(self.c1(x)); x=F.max_pool2d(x,2); x=F.relu(self.c2(x)); x=F.max_pool2d(x,2)
        x=x.flatten(1); f=F.relu(self.fc1(x)); y=self.fc2(f); return (y,f) if return_features else y


def train_ref10(xtr,ytr,xte,yte,device,epochs=4,batch=256):
    m=RefCNN().to(device); opt=torch.optim.Adam(m.parameters(),1e-3)
    ld=DataLoader(TensorDataset(torch.from_numpy(xtr[:,None].astype(np.float32)),torch.from_numpy(ytr)),batch_size=batch,shuffle=True)
    te_x=torch.from_numpy(xte[:,None].astype(np.float32)); te_y=torch.from_numpy(yte)
    hist=[]
    for ep in range(1,epochs+1):
        m.train(); ls=cor=n=0
        for xb,yb in ld:
            xb=xb.to(device); yb=yb.to(device); z=m(xb); loss=F.cross_entropy(z,yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            ls+=loss.item()*len(xb); cor+=(z.argmax(1)==yb).sum().item(); n+=len(xb)
        m.eval(); c=nn_=0
        with torch.no_grad():
            for i in range(0,len(te_x),512):
                xb=te_x[i:i+512].to(device); yb=te_y[i:i+512].to(device); pr=m(xb).argmax(1); c+=(pr==yb).sum().item(); nn_+=len(yb)
        rec={"epoch":ep,"train_loss":ls/n,"train_acc":cor/n,"test_acc":c/nn_}; hist.append(rec); print("REF10",json.dumps(rec),flush=True)
    return m,hist

@torch.no_grad()
def classifier_eval(model,images,device,batch=256):
    model.eval(); probs=[]; feats=[]; x=torch.from_numpy(images[:,None].astype(np.float32))
    for i in range(0,len(x),batch):
        z,f=model(x[i:i+batch].to(device),return_features=True); probs.append(z.softmax(-1).cpu().numpy()); feats.append(f.cpu().numpy())
    return np.concatenate(probs),np.concatenate(feats)


def stable_frechet(f1,f2):
    m1,m2=f1.mean(0),f2.mean(0); c1=np.cov(f1,rowvar=False)+np.eye(f1.shape[1])*1e-6; c2=np.cov(f2,rowvar=False)+np.eye(f2.shape[1])*1e-6
    diff=m1-m2
    if sqrtm is not None:
        cm=sqrtm(c1@c2); cm=cm.real if np.iscomplexobj(cm) else cm
        return float(diff@diff+np.trace(c1+c2-2*cm))
    w,v=np.linalg.eigh(c1); s1=(v*np.sqrt(np.clip(w,0,None)))@v.T; mid=s1@c2@s1
    w2,v2=np.linalg.eigh((mid+mid.T)/2); sm=(v2*np.sqrt(np.clip(w2,0,None)))@v2.T
    return float(diff@diff+np.trace(c1+c2-2*sm))


def nearest_hamming(gen,train,chunk=25):
    gp=np.packbits(gen.reshape(len(gen),-1),axis=1); tp=np.packbits(train.reshape(len(train),-1),axis=1)
    lut=np.array([bin(i).count("1") for i in range(256)],dtype=np.uint8); out=[]
    for i in range(0,len(gp),chunk):
        xor=np.bitwise_xor(gp[i:i+chunk,None,:],tp[None,:,:]); d=lut[xor].sum(2)/784.0; out.extend(d.min(1).tolist())
    return np.asarray(out)


def morphology(images):
    from scipy import ndimage
    comps=[]; largest=[]; isolated=[]
    for im in images.astype(np.uint8):
        lab,n=ndimage.label(im); comps.append(n); fg=int(im.sum())
        counts=np.bincount(lab.ravel())[1:]
        largest.append(float(counts.max()/fg) if fg and len(counts) else 0.0); isolated.append(int((counts==1).sum()))
    return {"mean_connected_components":float(np.mean(comps)),
            "mean_largest_component_fraction_of_foreground":float(np.mean(largest)),
            "mean_isolated_single_pixel_components":float(np.mean(isolated))}


def evaluate_generated(name,images,ref,real012,train012,device):
    p,fg=classifier_eval(ref,images,device); _,fr=classifier_eval(ref,real012,device); pred=p.argmax(1); conf=p.max(1)
    counts=np.bincount(pred,minlength=10); valid=np.isin(pred,[0,1,2]); q=counts[:3].astype(float)/len(images)
    nh=nearest_hamming(images,train012); uniq=np.unique(np.packbits(images.reshape(len(images),-1),axis=1),axis=0).shape[0]
    prior=np.ones(3)/3; tvd=.5*(np.abs(q-prior).sum()+counts[3:].sum()/len(images))
    return {"name":name,"n":len(images),"class_counts_0_to_9":counts.tolist(),"class_prob_0_to_9":(counts/len(images)).tolist(),
            "valid_012_rate":float(valid.mean()),"out_of_set_3_to_9_rate":float(1-valid.mean()),
            "mean_classifier_confidence":float(conf.mean()),"confidence_ge_095":float((conf>=.95).mean()),
            "distribution_TVD_to_balanced_012":float(tvd),"feature_frechet_vs_real012":stable_frechet(fg,fr),
            "unique_samples":int(uniq),"unique_fraction":float(uniq/len(images)),
            "mean_nearest_train_hamming":float(nh.mean()),"median_nearest_train_hamming":float(np.median(nh)),
            "morphology":morphology(images)}


def save_grid(images,path,n=100,title=None):
    n=min(n,len(images)); cols=10; rows=math.ceil(n/cols); fig,ax=plt.subplots(rows,cols,figsize=(cols,rows)); ax=np.asarray(ax).reshape(-1)
    for i,a in enumerate(ax):
        a.axis("off")
        if i<n: a.imshow(images[i],cmap="gray",vmin=0,vmax=1,interpolation="nearest")
    if title: fig.suptitle(title)
    plt.tight_layout(); fig.savefig(path,dpi=160,bbox_inches="tight"); plt.close(fig)


def save_trajectory(rec,path,title):
    # first sample, post-update canvas at every step; masked patches rendered gray.
    post=rec["post"][:,0]  # [T,196]
    imgs=[]
    for tok in post:
        known=tok!=MASK_ID; safe=tok.copy(); safe[~known]=0
        im=decode_2x2_np(safe[None])[0].astype(np.float32)
        km=np.repeat(np.repeat(known.reshape(GRID,GRID),2,0),2,1)
        im[~km]=0.5; imgs.append(im)
    cols=6; rows=math.ceil(len(imgs)/cols); fig,ax=plt.subplots(rows,cols,figsize=(cols*2,rows*2)); ax=np.asarray(ax).reshape(-1)
    for i,a in enumerate(ax):
        a.axis("off")
        if i<len(imgs): a.imshow(imgs[i],cmap="gray",vmin=0,vmax=1,interpolation="nearest"); a.set_title(f"step {i+1}")
    fig.suptitle(title); plt.tight_layout(); fig.savefig(path,dpi=160,bbox_inches="tight"); plt.close(fig)


# -----------------------------------------------------------------------------
# Generation wrapper
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate_mode(model,n,device,args,mode,seed):
    labels=torch.zeros(n,dtype=torch.long)
    tok,rec=h0.generate(model,labels,device,steps=args.decode_steps,temp0=args.choice_temperature,
                        seed=seed,mode=mode,record=True)
    return tok,rec


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=Path("H3_MASKGIT28_SNN"))
    ap.add_argument("--epochs",type=int,default=36)
    ap.add_argument("--batch",type=int,default=96)
    ap.add_argument("--n-per-class",type=int,default=0,help="0=all MNIST-012 training data")
    ap.add_argument("--d",type=int,default=128); ap.add_argument("--layers",type=int,default=4); ap.add_argument("--heads",type=int,default=4)
    ap.add_argument("--ff",type=int,default=512); ap.add_argument("--features",type=int,default=128)
    ap.add_argument("--lr",type=float,default=3e-4); ap.add_argument("--min-lr",type=float,default=1e-5); ap.add_argument("--warmup-steps",type=int,default=500)
    ap.add_argument("--weight-decay",type=float,default=0.05); ap.add_argument("--label-smoothing",type=float,default=0.1); ap.add_argument("--min-mask-rate",type=float,default=0.5)
    ap.add_argument("--gate-epochs",type=int,default=2); ap.add_argument("--gate-batch",type=int,default=64); ap.add_argument("--gate-steps",type=int,default=6); ap.add_argument("--gate-lr",type=float,default=1e-2)
    ap.add_argument("--decode-steps",type=int,default=24); ap.add_argument("--choice-temperature",type=float,default=4.5); ap.add_argument("--ngen",type=int,default=600)
    ap.add_argument("--seed",type=int,default=7); ap.add_argument("--threads",type=int,default=0)
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()

    configure_h0_globals(); seed_all(args.seed); tokenizer_self_test(); args.out.mkdir(parents=True,exist_ok=True)
    if args.smoke:
        args.epochs=2; args.gate_epochs=1; args.gate_steps=3; args.n_per_class=128; args.ngen=24; args.d=64; args.layers=2; args.ff=256; args.features=32; args.batch=64
    torch.set_num_threads(args.threads or min(8,os.cpu_count() or 1)); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device",device,"threads",torch.get_num_threads(),flush=True)

    xtr_all,ytr_all,xte_all,yte_all=load_mnist(); x012,y012=subset012(xtr_all,ytr_all); xte012,_=subset012(xte_all,yte_all)
    xtr,ytr,xva,yva=stratified_train_val(x012,y012,0.10,args.seed); xtr,ytr=subset_per_class(xtr,ytr,args.n_per_class,args.seed)
    tr_tok=encode_2x2_np(xtr); va_tok=encode_2x2_np(xva)
    print("data",xtr.shape,xva.shape,xte012.shape,"tokens",tr_tok.shape,flush=True)

    model=h0.MaskGIT(kind="linear",d=args.d,layers=args.layers,heads=args.heads,ff=args.ff,features=args.features,
                     nclass=3,feature_seed=args.seed+101,selective_gate=True).to(device)
    nparam=sum(p.numel() for p in model.parameters())
    print("parameters",nparam,flush=True)

    train_summary=train_backbone(model,tr_tok,va_tok,args.out,device,args)
    identity=h0.identity_audit(model,device,batch=4,rounds=6,seed=args.seed+1234)
    json_dump(args.out/"identity_audit.json",identity)
    print("IDENTITY",json.dumps(identity),flush=True)

    # Generate the learned fresh backbone before gate fitting.
    fresh_tok,fresh_rec=generate_mode(model,args.ngen,device,args,"fresh",args.seed+10000)
    fresh_img=decode_2x2_np(fresh_tok); np.savez_compressed(args.out/"generation_fresh.npz",samples=fresh_img,tokens=fresh_tok)
    save_grid(fresh_img,args.out/"generation_fresh_grid.png",100,"SNN_Transform MaskGIT-28 fresh linear attention")
    save_trajectory(fresh_rec,args.out/"trajectory_fresh.png","Fresh SNN_Transform MaskGIT trajectory")

    gate_hist=train_gate(model,tr_tok,args.out,device,args)
    selective_tok,selective_rec=generate_mode(model,args.ngen,device,args,"selective",args.seed+10000)
    selective_img=decode_2x2_np(selective_tok); np.savez_compressed(args.out/"generation_selective.npz",samples=selective_img,tokens=selective_tok,
        alpha_mean=selective_rec["alpha_mean"],state_H_rms=selective_rec["state_H_rms"])
    save_grid(selective_img,args.out/"generation_selective_grid.png",100,"SNN_Transform MaskGIT-28 selective persistent H/g")
    save_trajectory(selective_rec,args.out/"trajectory_selective.png","Selective persistent SNN_Transform MaskGIT trajectory")

    print("training shared MNIST evaluator",flush=True); seed_all(args.seed+30000)
    ref,refhist=train_ref10(xtr_all,ytr_all,xte_all,yte_all,device,epochs=4,batch=256)
    with (args.out/"ref_classifier10_history.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(refhist[0])); w.writeheader(); w.writerows(refhist)
    fresh_met=evaluate_generated("snn_transform_fresh",fresh_img,ref,xte012,x012,device)
    sel_met=evaluate_generated("snn_transform_selective",selective_img,ref,xte012,x012,device)
    paired_token=float(np.mean(fresh_tok!=selective_tok)); paired_pixel=float(np.mean(fresh_img!=selective_img))

    summary={
        "status":"H3 MaskGIT-28 SNN_Transform software training and generation",
        "task":"unconditional MNIST-012",
        "tokenizer":"lossless 2x2 binary patches","token_grid":"14x14 (196)","vocab":16,
        "architecture":{"d":args.d,"layers":args.layers,"heads":args.heads,"ff":args.ff,"features_per_head":args.features,"parameters":nparam,
                        "attention":"positive-RF normalized linear H/g","persistent_block":0,"selective_gate":"per-head Mamba-inspired retention"},
        "training":{"epochs":args.epochs,"min_mask_rate":args.min_mask_rate,"best_epoch":train_summary["best_epoch"],"best_val_ce":train_summary["best_val_ce"],
                    "optimizer_updates":train_summary["optimizer_updates"],"gate_epochs":args.gate_epochs,"gate_steps":args.gate_steps},
        "generation":{"steps":args.decode_steps,"choice_temperature":args.choice_temperature,"n":args.ngen},
        "identity_audit":identity,
        "fresh":fresh_met,"selective":sel_met,
        "selective_alpha_mean_by_step":[float(x) for x in selective_rec["alpha_mean"]],
        "selective_state_H_rms_by_step":[float(x) for x in selective_rec["state_H_rms"]],
        "paired_selective_vs_fresh":{"token_hamming":paired_token,"pixel_hamming":paired_pixel},
        "gate_history":gate_hist,
        "scientific_boundary":"Software exact positive-feature model. Izh calibration and SpiNNaker mapping are subsequent stages."
    }
    json_dump(args.out/"FINAL_METRICS.json",summary)
    torch.save({"model":model.state_dict(),"args":vars(args),"summary":summary},args.out/"snn_transform_maskgit28_final.pt")
    print("FINAL",json.dumps(summary),flush=True)

if __name__=="__main__":
    main()
