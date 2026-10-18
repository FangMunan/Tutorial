#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R6B.3 S1 — minimal no-scaffold bootstrap smoke test.

Purpose
-------
Test only whether the exact P3/MSDL local dynamics can self-start from WR=0
without the startup readout scaffold, under the final label-free MaskGIT input
and masking protocol.

This is NOT a training-quality or generation experiment.  It runs only a small
number of complete MaskGIT mask batches and records the causal startup sequence:

    WR movement -> BR movement -> internal forward-weight movement

BO/B2 are allowed to move before BR because their reciprocal mirrors depend on
already-nonzero WO/W2 rather than WR.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'snn_transform_p3_image'))
sys.path.insert(0,str(HERE))

import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
from maskgit_input_representation import build_input_embeddings,context_from_arr_unconditional
from maskgit_msdl_training import MaskGITP3System,sample_mask_set

INTERNAL=('WQ','WK','WV','WO','W1','W2')


def fro_norm(x):
    return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))


def rel_move(cur,ref):
    return float(np.linalg.norm(cur-ref)/(np.linalg.norm(ref)+1e-12))


def first_batch_over(rows,key,thr):
    for r in rows:
        if float(r[key])>thr:
            return int(r['batch'])
    return None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--batches',type=int,default=8)
    ap.add_argument('--D',type=int,default=96)
    ap.add_argument('--heads',type=int,default=8)
    ap.add_argument('--features',type=int,default=24)
    ap.add_argument('--ff',type=int,default=192)
    ap.add_argument('--R',type=int,default=7)
    ap.add_argument('--internal-lr-scale',type=float,default=.5)
    ap.add_argument('--encode-batch',type=int,default=256)
    ap.add_argument('--seed',type=int,default=7311)
    ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args()

    if a.batches<2:
        raise ValueError('S1 needs at least two mask batches to test WR -> BR bootstrap')

    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cpu')

    # Historical 0/1/2 subset selection is outside the neural path; labels are
    # never passed to the model, learning rule, context constructor, or generation.
    xall,yall,_,_=core.load(a.data)
    x012,y012=core.subset012(xall,yall)
    xtr,ytr,_,_=core.split(x012,y012,.1,7)

    tokenizer,tok_meta=base.load_frozen_tokenizer(a.tokenizer,device)
    tr,_=base.encode_snn(tokenizer,xtr,device,a.encode_batch)

    cfg=core.Config(D=a.D,H=a.heads,M=a.features,FF=a.ff,R=a.R)
    model=core.init_model(cfg,a.seed)
    for k in ('lrq','lrk','lrv','lro','lr1','lr2'):
        model[k]*=a.internal_lr_scale

    # S1 invariant: absolutely no startup readout scaffold.
    if np.any(model['WR']!=0) or np.any(model['BR']!=0):
        raise AssertionError('S1 must start from exact WR=0 and BR=0')

    tok,pos,input_audit=build_input_embeddings(tokenizer,cfg.D,grid=7)
    system=MaskGITP3System(cfg,model,a.seed+303)
    ones=np.ones(cfg.V,np.float64)
    rng=np.random.default_rng(a.seed+404)

    init={k:np.asarray(getattr(system,k),dtype=np.float64).copy() for k in INTERNAL}
    rows=[]

    for bi in range(1,a.batches+1):
        ii=int(rng.integers(len(tr)))
        sm=sample_mask_set(cfg.N,rng)
        truth=np.asarray(tr[ii],dtype=np.int64)
        arr=truth.copy();arr[sm.masked_positions]=core.MASK
        xn,xraw=context_from_arr_unconditional(arr,tok,pos)

        before={k:np.asarray(getattr(system,k),dtype=np.float64).copy()
                for k in ('WR','BR','BO','B2')+INTERNAL}
        rec=system.learn_mask_batch(xn,xraw,sm.masked_positions,truth,ones,rng)

        row={
            'batch':bi,
            'masked_targets':int(rec['masked_targets']),
            'sampled_r':float(sm.ratio_r),
            'mask_ratio':float(sm.mask_ratio),
            'online_nll':float(system.obs_nll/max(1,system.obs_count)),
            'WR_norm':fro_norm(system.WR),
            'WR_batch_delta':fro_norm(system.WR-before['WR']),
            'BR_mean_abs':float(np.mean(np.abs(system.BR))),
            'BR_batch_delta':fro_norm(system.BR-before['BR']),
            'BO_mean_abs':float(np.mean(np.abs(system.BO))),
            'BO_batch_delta':fro_norm(system.BO-before['BO']),
            'B2_mean_abs':float(np.mean(np.abs(system.B2))),
            'B2_batch_delta':fro_norm(system.B2-before['B2']),
        }
        for k in INTERNAL:
            row[f'{k}_rel_from_init']=rel_move(np.asarray(getattr(system,k)),init[k])
            row[f'{k}_batch_delta']=fro_norm(np.asarray(getattr(system,k))-before[k])
        row['internal_max_rel_from_init']=max(row[f'{k}_rel_from_init'] for k in INTERNAL)
        row['internal_max_batch_delta']=max(row[f'{k}_batch_delta'] for k in INTERNAL)
        rows.append(row)
        print('S1_BATCH',json.dumps(row),flush=True)

    # Thresholds distinguish exact/near-exact zero from true numerical movement.
    zero_tol=1e-14
    move_tol=1e-12
    first_wr=first_batch_over(rows,'WR_norm',move_tol)
    first_br=first_batch_over(rows,'BR_mean_abs',move_tol)
    first_internal=first_batch_over(rows,'internal_max_rel_from_init',move_tol)

    gates={
        'no_startup_scaffold':True,
        'labels_enter_neural_path':False,
        'initial_WR_exact_zero':True,
        'initial_BR_exact_zero':True,
        'WR_moves_by_first_batch':bool(rows[0]['WR_norm']>move_tol),
        'BR_stays_zero_through_first_batch':bool(rows[0]['BR_mean_abs']<=zero_tol),
        'internal_stays_zero_through_first_batch':bool(rows[0]['internal_max_rel_from_init']<=zero_tol),
        'BR_moves_after_WR':bool(first_br is not None and first_wr is not None and first_br>first_wr),
        'internal_moves_after_BR_available':bool(first_internal is not None and first_br is not None and first_internal>=first_br),
        'bootstrap_chain_observed':bool(first_wr is not None and first_br is not None and first_internal is not None and first_wr<first_br<=first_internal),
    }

    result={
        'stage':'R6B.3 S1 no-scaffold bootstrap smoke test',
        'scope':'minimal local-dynamics startup test only; no generation and no quality claim',
        'batches':a.batches,
        'seed':a.seed,
        'tokenizer_metadata':tok_meta,
        'input_representation':input_audit,
        'initialization':{
            'WR_norm':0.0,'BR_mean_abs':0.0,
            'startup_scaffold_used':False,
            'class_conditioning_used':False,
        },
        'expected_order':'WR first; BR only after WR is nonzero; internal forward weights no earlier than useful BR; BO/B2 may move immediately',
        'observed_first_movement_batch':{
            'WR':first_wr,'BR':first_br,'internal':first_internal,
        },
        'thresholds':{'zero_tol':zero_tol,'move_tol':move_tol},
        'gates':gates,
        'pass':bool(gates['WR_moves_by_first_batch'] and gates['BR_stays_zero_through_first_batch'] and gates['internal_stays_zero_through_first_batch'] and gates['bootstrap_chain_observed']),
        'trace':rows,
    }
    (a.out/'s1_bootstrap.json').write_text(json.dumps(result,indent=2))
    print('S1_FINAL',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
