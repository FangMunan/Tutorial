#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""H6-B: frozen ionic-mask substitution and non-inferiority test.

This stage changes ONLY the MaskGIT mask/reveal controller. The 784-pixel
SNN_Transform architecture, positive-RF H/g linear attention, persistent state,
selective retention and model dimensions are frozen.

Matched experiments
-------------------
A) A2 checkpoint + standard confidence/Gumbel MaskGIT controller
B) same A2 checkpoint + held-out measured ionic controller (controller-only swap)
C) one ionic-aware fine-tune from A2 using ionseq_001..003, validation ionseq_004,
   then held-out ionic generation with ionseq_005
D) same C checkpoint + standard controller (diagnostic only)

The canonical ionic NPZ is never re-thresholded, ranked, normalized, shuffled or
spatially transformed. Physical sequence boundaries are preserved.

Final gate is a pre-registered practical non-inferiority gate relative to matched A:
  valid012 drop <= 5 percentage points, with paired-bootstrap 95% lower CI >= -5 pp
  Frechet <= 1.20 x A
  connected components <= 1.25 x A
  isolated pixels <= 1.25 x A
  TVD <= A + 0.05

If C passes, H6 is ready to freeze for SpiNNaker transfer. This script does not
perform architecture search or further quality optimization.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import train_h6_pixel784 as h6
base = h6.base
h0 = h6.h0

TRAIN_IDS = h6.TRAIN_IDS
VAL_IDS = h6.VAL_IDS
TEST_IDS = h6.TEST_IDS

MARGINS = {
    "valid_drop_max": 0.05,
    "frechet_ratio_max": 1.20,
    "components_ratio_max": 1.25,
    "isolated_ratio_max": 1.25,
    "tvd_add_max": 0.05,
}


def dump(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_one(root: Path, name: str):
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one {name} under {root}; found {hits}")
    return hits[0]


def load_model(standard_dir: Path, bounded_dir: Path, device):
    final_ck = find_one(standard_dir, "snn_transform_h6_pixel784_final.pt")
    a2_ck = find_one(bounded_dir, "A2_best.pt")
    base_ck = torch.load(final_ck, map_location=device, weights_only=False)
    ca = base_ck["args"]
    ma = SimpleNamespace(
        d=int(ca["d"]), layers=int(ca["layers"]), heads=int(ca["heads"]),
        ff=int(ca["ff"]), features=int(ca["features"]), seed=int(ca["seed"])
    )
    model = h6.build_model(ma, device)
    a2 = torch.load(a2_ck, map_location=device, weights_only=False)
    model.load_state_dict(a2["model"])
    return model, ma, ca, final_ck, a2_ck


def load_data(seed: int, n_per_class: int):
    xtr_all,ytr_all,xte_all,yte_all = base.load_mnist()
    x012,y012 = base.subset012(xtr_all,ytr_all)
    xte012,_ = base.subset012(xte_all,yte_all)
    xtr,ytr,xva,yva = base.stratified_train_val(x012,y012,0.10,seed)
    xtr,ytr = base.subset_per_class(xtr,ytr,n_per_class,seed)
    return xtr,ytr,xva,yva,x012,xte012,xtr_all,ytr_all,xte_all,yte_all


@torch.no_grad()
def generate_standard(model, n, device, steps, temp, seed):
    labels = torch.zeros(n, dtype=torch.long)
    tok,_ = h0.generate(model, labels, device, steps=steps, temp0=temp,
                        seed=seed, mode="selective", record=False)
    return tok.astype(np.int64)


@torch.no_grad()
def generate_ionic(model, n, device, steps, seed, by_id, gen_batch):
    args = SimpleNamespace(decode_steps=steps, gen_batch=gen_batch)
    tok, rec = h6.generate_ionic_exact(model, n, device, args, by_id,
                                       "selective", seed)
    return tok.astype(np.int64), rec


def per_image_components(images):
    from scipy import ndimage
    comps=[]; isolated=[]; largest=[]
    for im in images.astype(np.uint8):
        lab,n = ndimage.label(im)
        comps.append(float(n))
        counts=np.bincount(lab.ravel())[1:]
        fg=int(im.sum())
        isolated.append(float((counts==1).sum()))
        largest.append(float(counts.max()/fg) if fg and len(counts) else 0.0)
    return np.asarray(comps), np.asarray(isolated), np.asarray(largest)


def full_eval(tag, tok, ref, xte012, x012, device, out: Path):
    img=h6.unflatten_pixels(tok)
    met=base.evaluate_generated(tag,img,ref,xte012,x012,device)
    probs, feats=base.classifier_eval(ref,img,device)
    pred=probs.argmax(1)
    valid=np.isin(pred,[0,1,2]).astype(np.float64)
    comps,isolated,largest=per_image_components(img)
    np.savez_compressed(out/f"{tag}.npz", samples=img, tokens=tok,
                        pred=pred.astype(np.int16), confidence=probs.max(1).astype(np.float32),
                        valid=valid.astype(np.uint8), components=comps.astype(np.float32),
                        isolated=isolated.astype(np.float32), largest=largest.astype(np.float32),
                        features=feats.astype(np.float32))
    h6.save_grid(img,out/f"{tag}_grid.png",tag)
    dump(out/f"{tag}.json",met)
    return met,{"valid":valid,"components":comps,"isolated":isolated}


def metric_core(m):
    mo=m["morphology"]
    return {
        "valid":float(m["valid_012_rate"]),
        "frechet":float(m["feature_frechet_vs_real012"]),
        "components":float(mo["mean_connected_components"]),
        "isolated":float(mo["mean_isolated_single_pixel_components"]),
        "tvd":float(m["distribution_TVD_to_balanced_012"]),
        "confidence":float(m["mean_classifier_confidence"]),
    }


def paired_bootstrap_valid(a_valid,b_valid,seed=20261012,nboot=4000):
    a=np.asarray(a_valid,dtype=np.float64); b=np.asarray(b_valid,dtype=np.float64)
    if len(a)!=len(b): raise ValueError("paired bootstrap requires equal n")
    d=b-a; rng=np.random.default_rng(seed); vals=np.empty(nboot,dtype=np.float64)
    n=len(d)
    for i in range(nboot):
        ix=rng.integers(0,n,size=n); vals[i]=d[ix].mean()
    lo,hi=np.quantile(vals,[0.025,0.975])
    return {"difference":float(d.mean()),"ci95_low":float(lo),"ci95_high":float(hi),
            "n":int(n),"n_boot":int(nboot)}


def noninferiority(candidate, baseline, boot):
    c=metric_core(candidate); a=metric_core(baseline)
    checks={
        "valid_point": c["valid"] >= a["valid"]-MARGINS["valid_drop_max"],
        "valid_bootstrap_ci": boot["ci95_low"] >= -MARGINS["valid_drop_max"],
        "frechet": c["frechet"] <= a["frechet"]*MARGINS["frechet_ratio_max"],
        "components": c["components"] <= a["components"]*MARGINS["components_ratio_max"],
        "isolated": c["isolated"] <= a["isolated"]*MARGINS["isolated_ratio_max"],
        "tvd": c["tvd"] <= a["tvd"]+MARGINS["tvd_add_max"],
    }
    return {"pass":bool(all(checks.values())),"checks":checks,
            "baseline":a,"candidate":c,"valid_bootstrap":boot,
            "margins":MARGINS}


def ionic_finetune(model,tr_tok,va_tok,by_id,device,out:Path,seed:int,
                    batch=32,epochs=5,max_updates=1000,lr=5e-5,min_mask_rate=0.05):
    tr=DataLoader(TensorDataset(torch.from_numpy(tr_tok)),batch_size=batch,shuffle=True)
    va=DataLoader(TensorDataset(torch.from_numpy(va_tok)),batch_size=batch,shuffle=False)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=(0.9,0.96),weight_decay=0.02)
    train_pool=h6.IonicForwardPool(by_id,TRAIN_IDS)
    best=float("inf"); best_ep=0; updates=0; hist=[]
    for ep in range(1,epochs+1):
        model.train(); ls=ac=mr=n=0.0; t0=time.time()
        for (tok,) in tr:
            if updates>=max_updates: break
            tok=tok.to(device); labels=torch.zeros(len(tok),dtype=torch.long,device=device)
            inp,mask,ratio=h6.ionic_training_mask(tok,min_mask_rate,train_pool)
            logits=model(inp,labels)
            loss=F.cross_entropy(logits[mask],tok[mask],label_smoothing=0.05)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            bs=len(tok); ls+=loss.item()*bs; ac+=(logits[mask].argmax(-1)==tok[mask]).float().mean().item()*bs; mr+=ratio.mean().item()*bs; n+=bs; updates+=1
        # Deterministic held-out ionic validation: reset ionseq_004 each epoch.
        saved=torch.random.get_rng_state(); torch.manual_seed(seed+50000)
        val_pool=h6.IonicForwardPool(by_id,VAL_IDS)
        model.eval(); vls=vac=vmr=vn=0.0
        with torch.no_grad():
            for (tok,) in va:
                tok=tok.to(device); labels=torch.zeros(len(tok),dtype=torch.long,device=device)
                inp,mask,ratio=h6.ionic_training_mask(tok,min_mask_rate,val_pool)
                logits=model(inp,labels); loss=F.cross_entropy(logits[mask],tok[mask])
                bs=len(tok); vls+=loss.item()*bs; vac+=(logits[mask].argmax(-1)==tok[mask]).float().mean().item()*bs; vmr+=ratio.mean().item()*bs; vn+=bs
        torch.random.set_rng_state(saved)
        rec={"epoch":ep,"updates":updates,"train_ce":ls/max(n,1),"train_mask_acc":ac/max(n,1),
             "train_mask_ratio":mr/max(n,1),"val_ce":vls/vn,"val_mask_acc":vac/vn,
             "val_mask_ratio":vmr/vn,"seconds":time.time()-t0,
             "train_ionic_stats":train_pool.stats(),"val_ionic_stats":val_pool.stats()}
        hist.append(rec); print("IONIC_FT",json.dumps(rec),flush=True)
        if rec["val_ce"]<best:
            best=rec["val_ce"]; best_ep=ep
            torch.save({"model":model.state_dict(),"epoch":ep,"best":best,"updates":updates,
                        "train_ionic_stats":train_pool.stats()},out/"H6B_ionic_best.pt")
        if updates>=max_updates: break
    ck=torch.load(out/"H6B_ionic_best.pt",map_location=device,weights_only=False)
    model.load_state_dict(ck["model"])
    with (out/"H6B_ionic_finetune_history.csv").open("w",newline="") as f:
        keys=[k for k in hist[0].keys() if not k.endswith("_stats")]
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows([{k:r[k] for k in keys} for r in hist])
    dump(out/"H6B_ionic_finetune_history.json",hist)
    return {"best_epoch":best_ep,"best_val_ce":best,"updates":updates,"history":hist}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--standard-dir",type=Path,required=True)
    ap.add_argument("--bounded-dir",type=Path,required=True)
    ap.add_argument("--ionic",type=Path,required=True)
    ap.add_argument("--out",type=Path,default=Path("H6B_IONIC_MASK"))
    ap.add_argument("--n",type=int,default=1000)
    ap.add_argument("--steps",type=int,default=96)
    ap.add_argument("--temperature",type=float,default=3.5)
    ap.add_argument("--gen-batch",type=int,default=32)
    ap.add_argument("--threads",type=int,default=4)
    ap.add_argument("--seed",type=int,default=7)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)

    h6.configure_pixel_globals(); h6.seed_all(args.seed); torch.set_num_threads(args.threads)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.ionic.is_file():
        raise FileNotFoundError(
            f"Canonical measured ionic file is required and was not found: {args.ionic}\n"
            "Do not substitute synthetic data or re-threshold another trace."
        )
    by_id,_raw=h6.load_canonical_ionic(args.ionic)
    required=set(TRAIN_IDS+VAL_IDS+TEST_IDS)
    missing=required-set(by_id)
    if missing: raise KeyError(f"canonical ionic file lacks required physical sequences: {sorted(missing)}")
    ionic_meta={"path":str(args.ionic),"sha256":sha256_file(args.ionic),
                "sequence_lengths":{k:int(len(by_id[k])) for k in sorted(required)},
                "train_ids":list(TRAIN_IDS),"val_ids":list(VAL_IDS),"generation_ids":list(TEST_IDS)}
    dump(args.out/"IONIC_INPUT_AUDIT.json",ionic_meta)
    print("IONIC_INPUT",json.dumps(ionic_meta),flush=True)

    model,ma,ca,final_ck,a2_ck=load_model(args.standard_dir,args.bounded_dir,device)
    xtr,ytr,xva,yva,x012,xte012,xtr_all,ytr_all,xte_all,yte_all=load_data(args.seed,int(ca.get("n_per_class",2000)))
    tr_tok=h6.flatten_pixels(xtr); va_tok=h6.flatten_pixels(xva)

    print("training matched evaluator",flush=True); h6.seed_all(args.seed+30000)
    ref,refhist=base.train_ref10(xtr_all,ytr_all,xte_all,yte_all,device,epochs=4,batch=256)
    dump(args.out/"ref_history.json",refhist)

    seed_gen=args.seed+10000
    t0=time.time(); tokA=generate_standard(model,args.n,device,args.steps,args.temperature,seed_gen)
    metA,rawA=full_eval("A_standard_A2",tokA,ref,xte012,x012,device,args.out)
    print("A_STANDARD",json.dumps({**metric_core(metA),"seconds":time.time()-t0}),flush=True)

    t0=time.time(); tokB,recB=generate_ionic(model,args.n,device,args.steps,seed_gen,by_id,args.gen_batch)
    metB,rawB=full_eval("B_A2_ionic_controller",tokB,ref,xte012,x012,device,args.out)
    dump(args.out/"B_ionic_scanner.json",recB["scanner"])
    bootB=paired_bootstrap_valid(rawA["valid"],rawB["valid"],seed=args.seed+7001)
    gateB=noninferiority(metB,metA,bootB)
    print("B_IONIC_CONTROLLER",json.dumps({**metric_core(metB),"gate":gateB,"seconds":time.time()-t0}),flush=True)

    # Final system candidate: ionic-aware training from the same frozen A2 start.
    # No architecture changes and no second optimization loop.
    modelC,_,_,_,_=load_model(args.standard_dir,args.bounded_dir,device)
    ft=ionic_finetune(modelC,tr_tok,va_tok,by_id,device,args.out,args.seed,
                      batch=32,epochs=5,max_updates=1000,lr=5e-5,min_mask_rate=0.05)

    t0=time.time(); tokC,recC=generate_ionic(modelC,args.n,device,args.steps,seed_gen,by_id,args.gen_batch)
    metC,rawC=full_eval("C_ionic_trained_ionic_controller",tokC,ref,xte012,x012,device,args.out)
    dump(args.out/"C_ionic_scanner.json",recC["scanner"])
    bootC=paired_bootstrap_valid(rawA["valid"],rawC["valid"],seed=args.seed+7002)
    gateC=noninferiority(metC,metA,bootC)
    print("C_FINAL_IONIC",json.dumps({**metric_core(metC),"gate":gateC,"seconds":time.time()-t0}),flush=True)

    # Diagnostic: isolate weight adaptation from physical controller substitution.
    t0=time.time(); tokD=generate_standard(modelC,args.n,device,args.steps,args.temperature,seed_gen)
    metD,rawD=full_eval("D_ionic_trained_standard_controller",tokD,ref,xte012,x012,device,args.out)
    print("D_STANDARD_AFTER_IONIC_TRAIN",json.dumps({**metric_core(metD),"seconds":time.time()-t0}),flush=True)

    status="H6B_PASS_FREEZE_FOR_SPINNAKER" if gateC["pass"] else "H6B_NONINFERIORITY_FAIL_REVIEW_BEFORE_HARDWARE"
    report={
        "status":status,
        "scientific_question":"Does replacing the digital mask/reveal controller with the frozen measured ionic protocol preserve H6 generation quality?",
        "frozen_architecture":"784-pixel SNN_Transform; positive-RF normalized H/g linear attention; persistent block-1 state; selective retention",
        "ionic_input":ionic_meta,
        "generation":{"n":args.n,"steps":args.steps,"standard_choice_temperature":args.temperature,"seed":seed_gen},
        "noninferiority_margins":MARGINS,
        "A_standard_A2":metA,
        "B_controller_only_ionic":metB,
        "B_gate":gateB,
        "C_ionic_training":ft,
        "C_final_ionic":metC,
        "C_gate":gateC,
        "D_ionic_trained_standard":metD,
        "checkpoint_sources":{"standard_final":str(final_ck),"A2_start":str(a2_ck),"final_ionic":"H6B_ionic_best.pt"},
        "next_step":"freeze code/checkpoint/protocol/reference outputs and begin SpiNNaker transfer" if gateC["pass"] else "do not tune architecture automatically; inspect failure against pre-registered margins first",
    }
    dump(args.out/"H6B_FINAL.json",report)
    print("FINAL",json.dumps({"status":status,"B_pass":gateB["pass"],"C_pass":gateC["pass"],
                              "A":metric_core(metA),"B":metric_core(metB),"C":metric_core(metC)}),flush=True)

if __name__=="__main__":
    main()
