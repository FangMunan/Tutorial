#!/usr/bin/env python3
from __future__ import annotations

"""H4: quality-focused SNN_Transform MaskGIT-28.

Purpose
-------
Make the software reference a strong image generator before another SpiNNaker mapping.
The experimental shell stays the user's 28x28 MaskGIT task (lossless 2x2 binary patches,
196 tokens, 16 visual codes, 24-step parallel decoding), while correcting two limitations
of H3-V2:

1. Restore the class-conditioned 0/1/2 task used by the strong prior MaskGIT-28 control.
2. Match the prior model scale (d=128, L=4, H=4, FF=512) and jointly fine-tune the
   full linear-attention + selective persistent H/g model through monotonic MaskGIT
   trajectories with BPTT, rather than fitting only 12 gate parameters.

This remains the software positive-feature reference. Izh calibration/fixed-point/SpiNNaker
mapping are deliberately downstream of a satisfactory image-quality checkpoint.
"""

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import train_snn_transform_maskgit28 as base
import train_snn_transform_maskgit28_v2 as v2

h0 = base.h0


def conditioned_epoch(model, loader, device, optimizer, args, global_step, total_steps,
                      deterministic_seed=None):
    train = optimizer is not None
    model.train(train)
    saved_rng = None
    if deterministic_seed is not None:
        saved_rng = torch.random.get_rng_state()
        torch.manual_seed(deterministic_seed)
    ls = ac = mr_sum = n = 0.0
    try:
        for tok, lab in loader:
            tok = tok.to(device); lab = lab.to(device)
            if train:
                lr = base.lr_for_step(global_step, total_steps, args.warmup_steps,
                                      args.lr, args.min_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
            inp, mask, mr = base.make_training_mask(tok, args.min_mask_rate)
            logits = model(inp, lab)
            loss = F.cross_entropy(logits[mask], tok[mask], label_smoothing=args.label_smoothing)
            acc = (logits[mask].argmax(-1) == tok[mask]).float().mean()
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); global_step += 1
            bs = len(tok)
            ls += loss.item() * bs; ac += acc.item() * bs; mr_sum += mr.mean().item() * bs; n += bs
    finally:
        if saved_rng is not None:
            torch.random.set_rng_state(saved_rng)
    return {'loss': ls/n, 'masked_token_acc': ac/n, 'mean_mask_ratio': mr_sum/n}, global_step


def train_conditioned_backbone(model, tr_tok, tr_y, va_tok, va_y, out, device, args):
    tr = DataLoader(TensorDataset(torch.from_numpy(tr_tok), torch.from_numpy(tr_y)),
                    batch_size=args.batch, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(va_tok), torch.from_numpy(va_y)),
                    batch_size=args.batch, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.96), weight_decay=args.weight_decay)
    total = args.epochs * len(tr); step = 0; best = float('inf'); best_ep = 0; hist = []
    for ep in range(1, args.epochs+1):
        t0 = time.time()
        a, step = conditioned_epoch(model,tr,device,opt,args,step,total)
        b, _ = conditioned_epoch(model,va,device,None,args,0,1,deterministic_seed=args.seed+50000)
        rec = {'epoch':ep,'train_ce':a['loss'],'train_mask_acc':a['masked_token_acc'],
               'train_mask_ratio':a['mean_mask_ratio'],'val_ce':b['loss'],
               'val_mask_acc':b['masked_token_acc'],'val_mask_ratio':b['mean_mask_ratio'],
               'lr':opt.param_groups[0]['lr'],'seconds':time.time()-t0,'global_step':step}
        hist.append(rec); print('BACKBONE',json.dumps(rec),flush=True)
        if b['loss'] < best:
            best=b['loss']; best_ep=ep
            torch.save({'model':model.state_dict(),'epoch':ep,'best':best,'args':vars(args)},out/'best_conditioned.pt')
        with (out/'backbone_history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
    ck=torch.load(out/'best_conditioned.pt',map_location=device,weights_only=False); model.load_state_dict(ck['model'])
    return {'best_epoch':best_ep,'best_val_ce':best,'history':hist,'updates':step}


def teacher_trajectory(tokens, steps, generator):
    return h0.teacher_mask_sequence(tokens, steps, generator)


def trajectory_validation(model, tok_np, y_np, device, args, seed):
    model.eval(); ld=DataLoader(TensorDataset(torch.from_numpy(tok_np),torch.from_numpy(y_np)),
                               batch_size=args.joint_batch,shuffle=False)
    total=n=0.0
    with torch.no_grad():
        for bi,(tok,lab) in enumerate(ld):
            tok=tok.to(device); lab=lab.to(device)
            gen=torch.Generator(device=device); gen.manual_seed(seed+bi)
            canv=teacher_trajectory(tok,args.joint_steps,gen)
            state=None; loss=0.0; terms=0
            for cur in canv:
                mask=cur.eq(base.MASK_ID)
                if not mask.any(): continue
                logits,state,_=model.forward_persistent(cur,lab,state,'selective')
                loss=loss+F.cross_entropy(logits[mask],tok[mask]); terms+=1
            loss=loss/max(1,terms); total+=loss.item()*len(tok); n+=len(tok)
    return total/n


def joint_trajectory_finetune(model,tr_tok,tr_y,va_tok,va_y,out,device,args):
    # Full BPTT through the persistent state. This is deliberately not gate-only training.
    for p in model.parameters(): p.requires_grad_(True)
    opt=torch.optim.AdamW(model.parameters(),lr=args.joint_lr,betas=(0.9,0.96),weight_decay=args.joint_weight_decay)
    ld=DataLoader(TensorDataset(torch.from_numpy(tr_tok),torch.from_numpy(tr_y)),
                  batch_size=args.joint_batch,shuffle=True)
    best=float('inf'); best_ep=0; hist=[]
    for ep in range(1,args.joint_epochs+1):
        model.train(); tot=n=0.0; t0=time.time()
        for bi,(tok,lab) in enumerate(ld):
            tok=tok.to(device); lab=lab.to(device)
            gen=torch.Generator(device=device); gen.manual_seed(args.seed+900001*ep+bi)
            canv=teacher_trajectory(tok,args.joint_steps,gen)
            state=None; loss=0.0; terms=0
            for cur in canv:
                mask=cur.eq(base.MASK_ID)
                if not mask.any(): continue
                logits,state,_=model.forward_persistent(cur,lab,state,'selective')
                loss=loss+F.cross_entropy(logits[mask],tok[mask],label_smoothing=args.joint_label_smoothing); terms+=1
            loss=loss/max(1,terms)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tot+=loss.item()*len(tok); n+=len(tok)
        va=trajectory_validation(model,va_tok,va_y,device,args,args.seed+700000)
        rec={'epoch':ep,'train_traj_ce':tot/n,'val_traj_ce':va,'seconds':time.time()-t0,
             'alpha_theta0':model.gate.theta0.detach().cpu().tolist(),
             'alpha_theta_mask':model.gate.theta_mask.detach().cpu().tolist(),
             'alpha_theta_new':model.gate.theta_new.detach().cpu().tolist()}
        hist.append(rec); print('JOINT',json.dumps(rec),flush=True)
        if va<best:
            best=va; best_ep=ep; torch.save({'model':model.state_dict(),'epoch':ep,'best':best,'args':vars(args)},out/'best_joint.pt')
        with (out/'joint_history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
    ck=torch.load(out/'best_joint.pt',map_location=device,weights_only=False); model.load_state_dict(ck['model'])
    return {'best_epoch':best_ep,'best_val_traj_ce':best,'history':hist}


@torch.no_grad()
def generate_balanced(model,n,device,args,mode,seed):
    # Balanced requested labels make condition-match directly interpretable.
    labels=(torch.arange(n,dtype=torch.long)%3)
    tok,rec=h0.generate(model,labels,device,steps=args.decode_steps,temp0=args.choice_temperature,
                        seed=seed,mode=mode,record=True)
    return tok,labels.numpy(),rec


def eval_conditional(name,images,requested,ref,real012,train012,device):
    m=base.evaluate_generated(name,images,ref,real012,train012,device)
    p,_=base.classifier_eval(ref,images,device); pred=p.argmax(1)
    m['condition_match_rate']=float((pred==requested).mean())
    m['requested_counts_012']=np.bincount(requested,minlength=3).tolist()
    m['predicted_given_requested_confusion_012']=np.asarray([
        [int(((requested==r)&(pred==c)).sum()) for c in range(3)] for r in range(3)
    ]).tolist()
    return m


def save_stage(model,out,prefix,n,device,args,ref,real012,train012,seed,mode):
    tok,req,rec=generate_balanced(model,n,device,args,mode,seed)
    img=base.decode_2x2_np(tok)
    np.savez_compressed(out/f'{prefix}.npz',samples=img,tokens=tok,requested=req,
                        alpha_mean=rec['alpha_mean'],state_H_rms=rec['state_H_rms'])
    base.save_grid(img,out/f'{prefix}_grid.png',100,prefix)
    base.save_trajectory(rec,out/f'{prefix}_trajectory.png',prefix+' trajectory')
    return eval_conditional(prefix,img,req,ref,real012,train012,device),rec


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=Path('H4_MASKGIT28_QUALITY'))
    ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--batch',type=int,default=64)
    ap.add_argument('--n-per-class',type=int,default=2000)
    ap.add_argument('--d',type=int,default=128); ap.add_argument('--layers',type=int,default=4); ap.add_argument('--heads',type=int,default=4)
    ap.add_argument('--ff',type=int,default=512); ap.add_argument('--features',type=int,default=128)
    ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--min-lr',type=float,default=1e-5); ap.add_argument('--warmup-steps',type=int,default=300)
    ap.add_argument('--weight-decay',type=float,default=0.05); ap.add_argument('--label-smoothing',type=float,default=0.1); ap.add_argument('--min-mask-rate',type=float,default=0.5)
    ap.add_argument('--joint-epochs',type=int,default=4); ap.add_argument('--joint-batch',type=int,default=16); ap.add_argument('--joint-steps',type=int,default=6)
    ap.add_argument('--joint-lr',type=float,default=5e-5); ap.add_argument('--joint-weight-decay',type=float,default=0.02); ap.add_argument('--joint-label-smoothing',type=float,default=0.05)
    ap.add_argument('--decode-steps',type=int,default=24); ap.add_argument('--choice-temperature',type=float,default=4.5); ap.add_argument('--ngen',type=int,default=300)
    ap.add_argument('--seed',type=int,default=7); ap.add_argument('--threads',type=int,default=4); ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()

    base.configure_h0_globals(); base.seed_all(args.seed); base.tokenizer_self_test(); args.out.mkdir(parents=True,exist_ok=True)
    if args.smoke:
        args.epochs=2; args.n_per_class=128; args.d=64; args.layers=2; args.ff=256; args.features=32; args.batch=32
        args.joint_epochs=1; args.joint_batch=8; args.joint_steps=4; args.ngen=24
    torch.set_num_threads(args.threads or min(8,os.cpu_count() or 1)); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device',device,'threads',torch.get_num_threads(),flush=True)

    xtr_all,ytr_all,xte_all,yte_all=base.load_mnist(); x012,y012=base.subset012(xtr_all,ytr_all); xte012,yte012=base.subset012(xte_all,yte_all)
    xtr,ytr,xva,yva=base.stratified_train_val(x012,y012,0.10,args.seed); xtr,ytr=base.subset_per_class(xtr,ytr,args.n_per_class,args.seed)
    tr_tok=base.encode_2x2_np(xtr); va_tok=base.encode_2x2_np(xva)
    print('data',xtr.shape,xva.shape,xte012.shape,'tokens',tr_tok.shape,flush=True)

    model=h0.MaskGIT(kind='linear',d=args.d,layers=args.layers,heads=args.heads,ff=args.ff,features=args.features,
                     nclass=3,feature_seed=args.seed+101,selective_gate=True).to(device)
    v2.init_maskgit_style(model)
    nparam=sum(p.numel() for p in model.parameters()); print('parameters',nparam,flush=True)

    back=train_conditioned_backbone(model,tr_tok,ytr,va_tok,yva,args.out,device,args)
    identity_pre=h0.identity_audit(model,device,batch=4,rounds=6,seed=args.seed+1234)

    # Shared evaluator trained once; use it for both pre- and post-joint checkpoints.
    print('training shared MNIST evaluator',flush=True); base.seed_all(args.seed+30000)
    ref,refhist=base.train_ref10(xtr_all,ytr_all,xte_all,yte_all,device,epochs=4,batch=256)
    with (args.out/'ref_classifier10_history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(refhist[0])); w.writeheader(); w.writerows(refhist)

    fresh_met,_=save_stage(model,args.out,'capacity_fresh',args.ngen,device,args,ref,xte012,x012,args.seed+10000,'fresh')

    joint=joint_trajectory_finetune(model,tr_tok,ytr,va_tok,yva,args.out,device,args)
    identity_post=h0.identity_audit(model,device,batch=4,rounds=6,seed=args.seed+2234)
    hybrid_met,hybrid_rec=save_stage(model,args.out,'joint_selective',args.ngen,device,args,ref,xte012,x012,args.seed+10000,'selective')
    # Same post-joint weights with state disabled: separates weight improvement from recurrent-state contribution.
    postfresh_met,_=save_stage(model,args.out,'joint_weights_fresh_read',args.ngen,device,args,ref,xte012,x012,args.seed+10000,'fresh')

    summary={
      'status':'H4 quality-focused class-conditioned SNN_Transform MaskGIT-28',
      'task':'class-conditioned MNIST-012','tokenizer':'lossless 2x2 binary patches','token_grid':'14x14 (196)','vocab':16,
      'architecture':{'d':args.d,'layers':args.layers,'heads':args.heads,'ff':args.ff,'features_per_head':args.features,
                      'parameters':nparam,'attention':'positive-RF normalized linear H/g','persistent_block':0,
                      'selective_gate':'per-head Mamba-inspired retention'},
      'backbone_training':back,'joint_training':joint,'identity_pre':identity_pre,'identity_post':identity_post,
      'capacity_fresh':fresh_met,'joint_selective':hybrid_met,'joint_weights_fresh_read':postfresh_met,
      'selective_alpha_mean_by_step':[float(x) for x in hybrid_rec['alpha_mean']],
      'selective_state_H_rms_by_step':[float(x) for x in hybrid_rec['state_H_rms']],
      'quality_target':{'reference_prior_maskgit_valid_rate':0.97,'reference_prior_maskgit_condition_match':0.9667},
      'scientific_boundary':'Software positive-feature reference; no Izh/fixed-point/SpiNNaker claim in H4.'
    }
    # Avoid huge duplicated histories inside final checkpoint metadata but retain full CSVs.
    summary['backbone_training'].pop('history',None); summary['joint_training'].pop('history',None)
    base.json_dump(args.out/'FINAL_METRICS.json',summary)
    torch.save({'model':model.state_dict(),'args':vars(args),'summary':summary},args.out/'snn_transform_maskgit28_h4_final.pt')
    print('FINAL',json.dumps(summary),flush=True)

if __name__=='__main__':
    main()
