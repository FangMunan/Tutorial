#!/usr/bin/env python3
from __future__ import annotations

"""Formal H3 MaskGIT-28 SNN_Transform run.

This entrypoint reuses the v1 experiment harness but applies the initialization used by
our previous MaskGIT-28 baseline (small truncated-normal weights) and a compute-practical
formal architecture for cloud CPU training: d=64, 4 blocks, 4 heads, FF=256, 64 positive
random features/head.  The task itself stays at the full 196-token 28x28 MaskGIT problem.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import train_snn_transform_maskgit28 as base
h0 = base.h0


def init_maskgit_style(model):
    # Match the prior standard MaskGIT-28 initialization rather than PyTorch Embedding defaults.
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02, a=-0.04, b=0.04)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02, a=-0.04, b=0.04)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(model.pos, std=0.02, a=-0.04, b=0.04)
        nn.init.zeros_(model.bias)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=Path('H3_MASKGIT28_SNN_V2'))
    ap.add_argument('--epochs',type=int,default=60)
    ap.add_argument('--batch',type=int,default=96)
    ap.add_argument('--n-per-class',type=int,default=2000)
    ap.add_argument('--d',type=int,default=64); ap.add_argument('--layers',type=int,default=4); ap.add_argument('--heads',type=int,default=4)
    ap.add_argument('--ff',type=int,default=256); ap.add_argument('--features',type=int,default=64)
    ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--min-lr',type=float,default=1e-5); ap.add_argument('--warmup-steps',type=int,default=300)
    ap.add_argument('--weight-decay',type=float,default=0.05); ap.add_argument('--label-smoothing',type=float,default=0.1); ap.add_argument('--min-mask-rate',type=float,default=0.5)
    ap.add_argument('--gate-epochs',type=int,default=3); ap.add_argument('--gate-batch',type=int,default=64); ap.add_argument('--gate-steps',type=int,default=6); ap.add_argument('--gate-lr',type=float,default=1e-2)
    ap.add_argument('--decode-steps',type=int,default=24); ap.add_argument('--choice-temperature',type=float,default=4.5); ap.add_argument('--ngen',type=int,default=128)
    ap.add_argument('--seed',type=int,default=7); ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()

    base.configure_h0_globals(); base.seed_all(args.seed); base.tokenizer_self_test(); args.out.mkdir(parents=True,exist_ok=True)
    if args.smoke:
        args.epochs=4; args.gate_epochs=1; args.gate_steps=4; args.n_per_class=256; args.ngen=32; args.features=32
    torch.set_num_threads(args.threads or min(8,os.cpu_count() or 1)); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device',device,'threads',torch.get_num_threads(),flush=True)

    xtr_all,ytr_all,xte_all,yte_all=base.load_mnist(); x012,y012=base.subset012(xtr_all,ytr_all); xte012,_=base.subset012(xte_all,yte_all)
    xtr,ytr,xva,yva=base.stratified_train_val(x012,y012,0.10,args.seed); xtr,ytr=base.subset_per_class(xtr,ytr,args.n_per_class,args.seed)
    tr_tok=base.encode_2x2_np(xtr); va_tok=base.encode_2x2_np(xva)
    print('data',xtr.shape,xva.shape,xte012.shape,'tokens',tr_tok.shape,flush=True)

    model=h0.MaskGIT(kind='linear',d=args.d,layers=args.layers,heads=args.heads,ff=args.ff,features=args.features,
                     nclass=3,feature_seed=args.seed+101,selective_gate=True).to(device)
    init_maskgit_style(model)
    nparam=sum(p.numel() for p in model.parameters())
    print('parameters',nparam,flush=True)

    train_summary=base.train_backbone(model,tr_tok,va_tok,args.out,device,args)
    identity=h0.identity_audit(model,device,batch=4,rounds=6,seed=args.seed+1234)
    base.json_dump(args.out/'identity_audit.json',identity); print('IDENTITY',json.dumps(identity),flush=True)

    fresh_tok,fresh_rec=base.generate_mode(model,args.ngen,device,args,'fresh',args.seed+10000)
    fresh_img=base.decode_2x2_np(fresh_tok); np.savez_compressed(args.out/'generation_fresh.npz',samples=fresh_img,tokens=fresh_tok)
    base.save_grid(fresh_img,args.out/'generation_fresh_grid.png',100,'SNN_Transform MaskGIT-28 fresh linear attention')
    base.save_trajectory(fresh_rec,args.out/'trajectory_fresh.png','Fresh SNN_Transform MaskGIT trajectory')

    gate_hist=base.train_gate(model,tr_tok,args.out,device,args)
    selective_tok,selective_rec=base.generate_mode(model,args.ngen,device,args,'selective',args.seed+10000)
    selective_img=base.decode_2x2_np(selective_tok); np.savez_compressed(args.out/'generation_selective.npz',samples=selective_img,tokens=selective_tok,
        alpha_mean=selective_rec['alpha_mean'],state_H_rms=selective_rec['state_H_rms'])
    base.save_grid(selective_img,args.out/'generation_selective_grid.png',100,'SNN_Transform MaskGIT-28 selective persistent H/g')
    base.save_trajectory(selective_rec,args.out/'trajectory_selective.png','Selective persistent SNN_Transform MaskGIT trajectory')

    print('training shared MNIST evaluator',flush=True); base.seed_all(args.seed+30000)
    ref,refhist=base.train_ref10(xtr_all,ytr_all,xte_all,yte_all,device,epochs=4,batch=256)
    with (args.out/'ref_classifier10_history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(refhist[0])); w.writeheader(); w.writerows(refhist)
    fresh_met=base.evaluate_generated('snn_transform_fresh',fresh_img,ref,xte012,x012,device)
    sel_met=base.evaluate_generated('snn_transform_selective',selective_img,ref,xte012,x012,device)
    summary={
        'status':'H3 MaskGIT-28 SNN_Transform software training and generation V2',
        'task':'unconditional MNIST-012','tokenizer':'lossless 2x2 binary patches','token_grid':'14x14 (196)','vocab':16,
        'architecture':{'d':args.d,'layers':args.layers,'heads':args.heads,'ff':args.ff,'features_per_head':args.features,'parameters':nparam,
                        'attention':'positive-RF normalized linear H/g','persistent_block':0,'selective_gate':'per-head Mamba-inspired retention'},
        'training':{'epochs':args.epochs,'n_per_class':args.n_per_class,'best_epoch':train_summary['best_epoch'],'best_val_ce':train_summary['best_val_ce'],
                    'optimizer_updates':train_summary['optimizer_updates'],'gate_epochs':args.gate_epochs,'gate_steps':args.gate_steps},
        'generation':{'steps':args.decode_steps,'choice_temperature':args.choice_temperature,'n':args.ngen},
        'identity_audit':identity,'fresh':fresh_met,'selective':sel_met,
        'selective_alpha_mean_by_step':[float(x) for x in selective_rec['alpha_mean']],
        'selective_state_H_rms_by_step':[float(x) for x in selective_rec['state_H_rms']],
        'paired_selective_vs_fresh':{'token_hamming':float(np.mean(fresh_tok!=selective_tok)),'pixel_hamming':float(np.mean(fresh_img!=selective_img))},
        'gate_history':gate_hist,
        'scientific_boundary':'Software positive-feature SNN_Transform; Izh calibration and SpiNNaker mapping are subsequent.'
    }
    base.json_dump(args.out/'FINAL_METRICS.json',summary)
    torch.save({'model':model.state_dict(),'args':vars(args),'summary':summary},args.out/'snn_transform_maskgit28_final.pt')
    print('FINAL',json.dumps(summary),flush=True)

if __name__=='__main__': main()
