#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Bounded H6 optimization.

Budget is frozen:
  A1) decoder-only sweep: steps={48,64,96} at temp=4.5, then temp={3.5,4.5,5.5}
      only at the best step count. No training.
  A2) only if A1 fails the quality gate: one short masked-token fine-tune from the
      existing H6 checkpoint with min_mask_rate=0.05, max 5 epochs / 1000 updates.
      No architecture changes, no auxiliary losses, no second fine-tune.

After A2, stop regardless of outcome. If the gate is still not reached, freeze the best
candidate and proceed to the ionic stage rather than continuing MNIST optimization.
"""

import argparse, csv, json, math, os, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import train_h6_pixel784 as h6
base = h6.base
h0 = h6.h0

GATE = {
    "valid012_min": 0.95,
    "components_max": 8.0,
    "isolated_max": 5.0,
}
STRETCH = {"frechet_max": 160.0, "tvd_max": 0.20}


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def quality_fields(m):
    mo = m["morphology"]
    return {
        "valid": float(m["valid_012_rate"]),
        "components": float(mo["mean_connected_components"]),
        "isolated": float(mo["mean_isolated_single_pixel_components"]),
        "frechet": float(m["feature_frechet_vs_real012"]),
        "tvd": float(m["distribution_TVD_to_balanced_012"]),
        "confidence": float(m["mean_classifier_confidence"]),
    }


def deficit_score(m):
    q = quality_fields(m)
    # Lower is better. This is only a deterministic candidate selector, not a scientific metric.
    return float(
        max(0.0, (GATE["valid012_min"] - q["valid"]) / 0.05)
        + max(0.0, (q["components"] - GATE["components_max"]) / GATE["components_max"])
        + max(0.0, (q["isolated"] - GATE["isolated_max"]) / GATE["isolated_max"])
        + 0.25 * max(0.0, (q["frechet"] - STRETCH["frechet_max"]) / STRETCH["frechet_max"])
        + 0.25 * max(0.0, (q["tvd"] - STRETCH["tvd_max"]) / STRETCH["tvd_max"])
    )


def pass_gate(m, matched_baseline):
    q = quality_fields(m); b = quality_fields(matched_baseline)
    primary = (
        q["valid"] >= GATE["valid012_min"] and
        q["components"] <= GATE["components_max"] and
        q["isolated"] <= GATE["isolated_max"]
    )
    nondegrade = q["frechet"] <= b["frechet"] and q["tvd"] <= b["tvd"]
    return bool(primary and nondegrade)


@torch.no_grad()
def generate(model, n, device, steps, temp, seed, mode="selective"):
    labels = torch.zeros(n, dtype=torch.long)
    tok, _ = h0.generate(model, labels, device, steps=steps, temp0=temp,
                         seed=seed, mode=mode, record=False)
    return tok.astype(np.int64)


def evaluate_cfg(model, ref, xte012, x012, device, out, n, steps, temp, seed, tag):
    t0 = time.time()
    tok = generate(model, n, device, steps, temp, seed, "selective")
    img = h6.unflatten_pixels(tok)
    met = base.evaluate_generated(tag, img, ref, xte012, x012, device)
    met["decode_steps"] = int(steps); met["choice_temperature"] = float(temp)
    met["deficit_score"] = deficit_score(met); met["seconds"] = time.time() - t0
    np.savez_compressed(out / f"{tag}.npz", samples=img, tokens=tok)
    h6.save_grid(img, out / f"{tag}_grid.png", tag)
    dump(out / f"{tag}.json", met)
    print("EVAL", json.dumps({"tag":tag, **quality_fields(met), "score":met["deficit_score"], "seconds":met["seconds"]}), flush=True)
    return met


def load_data(seed, n_per_class):
    xtr_all,ytr_all,xte_all,yte_all = base.load_mnist()
    x012,y012 = base.subset012(xtr_all,ytr_all); xte012,_ = base.subset012(xte_all,yte_all)
    xtr,ytr,xva,yva = base.stratified_train_val(x012,y012,0.10,seed)
    xtr,ytr = base.subset_per_class(xtr,ytr,n_per_class,seed)
    return xtr,ytr,xva,yva,x012,xte012,xtr_all,ytr_all,xte_all,yte_all


def short_finetune(model, tr_tok, va_tok, device, out, seed, batch=32, epochs=5,
                   max_updates=1000, lr=5e-5, min_mask_rate=0.05):
    tr = DataLoader(TensorDataset(torch.from_numpy(tr_tok)), batch_size=batch, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(va_tok)), batch_size=batch, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.96), weight_decay=0.02)
    a = SimpleNamespace(min_mask_rate=min_mask_rate, label_smoothing=0.05,
                        warmup_steps=0, lr=lr, min_lr=lr)
    best = float("inf"); best_ep = 0; updates = 0; hist=[]
    for ep in range(1, epochs+1):
        model.train(); ls=ac=mr=n=0.0; t0=time.time()
        for (tok,) in tr:
            if updates >= max_updates: break
            tok=tok.to(device); labels=torch.zeros(len(tok),dtype=torch.long,device=device)
            inp,mask,ratio=h6.standard_training_mask(tok,min_mask_rate)
            logits=model(inp,labels)
            loss=torch.nn.functional.cross_entropy(logits[mask],tok[mask],label_smoothing=0.05)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            bs=len(tok); ls+=loss.item()*bs; ac+=(logits[mask].argmax(-1)==tok[mask]).float().mean().item()*bs; mr+=ratio.mean().item()*bs; n+=bs; updates+=1
        # deterministic validation with the same expanded mask range
        saved=torch.random.get_rng_state(); torch.manual_seed(seed+50000)
        model.eval(); vls=vac=vmr=vn=0.0
        with torch.no_grad():
            for (tok,) in va:
                tok=tok.to(device); labels=torch.zeros(len(tok),dtype=torch.long,device=device)
                inp,mask,ratio=h6.standard_training_mask(tok,min_mask_rate); logits=model(inp,labels)
                loss=torch.nn.functional.cross_entropy(logits[mask],tok[mask])
                bs=len(tok); vls+=loss.item()*bs; vac+=(logits[mask].argmax(-1)==tok[mask]).float().mean().item()*bs; vmr+=ratio.mean().item()*bs; vn+=bs
        torch.random.set_rng_state(saved)
        rec={"epoch":ep,"updates":updates,"train_ce":ls/max(n,1),"train_mask_acc":ac/max(n,1),"train_mask_ratio":mr/max(n,1),
             "val_ce":vls/vn,"val_mask_acc":vac/vn,"val_mask_ratio":vmr/vn,"seconds":time.time()-t0}
        hist.append(rec); print("A2",json.dumps(rec),flush=True)
        if rec["val_ce"] < best:
            best=rec["val_ce"]; best_ep=ep
            torch.save({"model":model.state_dict(),"epoch":ep,"best":best,"updates":updates},out/"A2_best.pt")
        if updates >= max_updates: break
    ck=torch.load(out/"A2_best.pt",map_location=device,weights_only=False); model.load_state_dict(ck["model"])
    with (out/"A2_history.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
    return {"best_epoch":best_ep,"best_val_ce":best,"updates":updates,"history":hist}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,default=Path("H6_BOUNDED_OPT"))
    ap.add_argument("--ngen",type=int,default=300); ap.add_argument("--threads",type=int,default=4); ap.add_argument("--seed",type=int,default=7)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    h6.configure_pixel_globals(); h6.seed_all(args.seed); torch.set_num_threads(args.threads)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    final_ck=args.baseline_dir/"snn_transform_h6_pixel784_final.pt"
    if not final_ck.is_file(): raise FileNotFoundError(final_ck)
    ck=torch.load(final_ck,map_location=device,weights_only=False); ca=ck["args"]
    model_args=SimpleNamespace(d=int(ca["d"]),layers=int(ca["layers"]),heads=int(ca["heads"]),ff=int(ca["ff"]),features=int(ca["features"]),seed=int(ca["seed"]))
    model=h6.build_model(model_args,device); model.load_state_dict(ck["model"])

    xtr,ytr,xva,yva,x012,xte012,xtr_all,ytr_all,xte_all,yte_all=load_data(args.seed,int(ca.get("n_per_class",2000)))
    tr_tok=h6.flatten_pixels(xtr); va_tok=h6.flatten_pixels(xva)
    print("training matched evaluator",flush=True); h6.seed_all(args.seed+30000)
    ref,refhist=base.train_ref10(xtr_all,ytr_all,xte_all,yte_all,device,epochs=4,batch=256)
    dump(args.out/"ref_history.json",refhist)

    # Matched re-evaluation of the current selective baseline under this evaluator.
    baseline=evaluate_cfg(model,ref,xte012,x012,device,args.out,args.ngen,24,4.5,args.seed+10000,"A0_baseline_T24_temp4p5")

    # A1a: step sweep, no training.
    step_results={}
    for steps in (48,64,96):
        m=evaluate_cfg(model,ref,xte012,x012,device,args.out,args.ngen,steps,4.5,args.seed+10000,f"A1_T{steps}_temp4p5")
        step_results[str(steps)]=m
    best_step=int(min(step_results,key=lambda k:step_results[k]["deficit_score"]))

    # A1b: only the best step gets a small temperature sweep. Reuse temp=4.5 result.
    a1={f"T{best_step}_temp4p5":step_results[str(best_step)]}
    for temp in (3.5,5.5):
        tag=f"A1_T{best_step}_temp{str(temp).replace('.','p')}"
        a1[f"T{best_step}_temp{temp}"]=evaluate_cfg(model,ref,xte012,x012,device,args.out,args.ngen,best_step,temp,args.seed+10000,tag)
    best_name=min(a1,key=lambda k:a1[k]["deficit_score"]); best_a1=a1[best_name]
    a1_pass=pass_gate(best_a1,baseline)

    report={"budget":"1 decoder sweep + at most 1 short fine-tune","gate":GATE,"stretch_targets":STRETCH,
            "matched_baseline":baseline,"A1_step_sweep":step_results,"A1_best_step":best_step,"A1_temperature_sweep":a1,
            "A1_best":best_name,"A1_pass":a1_pass}

    # A2 is conditional and is the final allowed optimization action.
    if a1_pass:
        report["final_status"]="OPTIMIZATION_PASS_A1"
        report["recommended_checkpoint"]="original H6 final checkpoint"
        report["recommended_decode_steps"]=int(best_a1["decode_steps"]); report["recommended_temperature"]=float(best_a1["choice_temperature"])
    else:
        # Restore original checkpoint before the one allowed fine-tune.
        model.load_state_dict(ck["model"])
        a2train=short_finetune(model,tr_tok,va_tok,device,args.out,args.seed,batch=32,epochs=5,max_updates=1000,lr=5e-5,min_mask_rate=0.05)
        a2=evaluate_cfg(model,ref,xte012,x012,device,args.out,args.ngen,int(best_a1["decode_steps"]),float(best_a1["choice_temperature"]),args.seed+10000,"A2_short_finetune_best_decoder")
        a2_pass=pass_gate(a2,baseline)
        report["A2_training"]=a2train; report["A2_result"]=a2; report["A2_pass"]=a2_pass
        # Select the better candidate even if neither passes, then STOP optimization.
        if a2["deficit_score"] < best_a1["deficit_score"]:
            report["recommended_checkpoint"]="A2_best.pt"; recommended=a2
        else:
            report["recommended_checkpoint"]="original H6 final checkpoint"; recommended=best_a1
        report["recommended_decode_steps"]=int(recommended["decode_steps"]); report["recommended_temperature"]=float(recommended["choice_temperature"])
        report["final_status"]="OPTIMIZATION_PASS_A2" if a2_pass else "STOP_OPTIMIZATION_AND_PROCEED_TO_IONIC"

    dump(args.out/"BOUNDED_OPT_FINAL.json",report)
    print("FINAL",json.dumps({"status":report["final_status"],"A1_best":report["A1_best"],
                              "recommended_checkpoint":report["recommended_checkpoint"],
                              "steps":report["recommended_decode_steps"],"temperature":report["recommended_temperature"]}),flush=True)

if __name__=="__main__": main()
