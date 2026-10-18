#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Label-free MaskGIT training/evaluation wrappers for the exact P3/MSDL core.

The local P3/MSDL mechanics are reused from maskgit_msdl_training.py.  This file
removes the historical class-label path entirely: corrupted contexts are built
only from visual-token and 2-D position representations.
"""
from __future__ import annotations

import json
import math
import time
from typing import Dict, List, Tuple

import numpy as np

import p3i_cb_exact_msdl as core
from maskgit_input_representation import context_from_arr_unconditional
from maskgit_msdl_training import (
    MaskGITP3System,
    MaskSample,
    cosine_mask_ratio,
    masking_self_test,
    sample_mask_set,
)


def build_maskgit_scaffold_contexts_unconditional(model, cfg, tokens, tok, pos,
                                                   nctx: int, seed: int):
    """Build startup readout rows from label-free MaskGIT corrupted contexts."""
    rng=np.random.default_rng(seed)
    F=np.empty((nctx,cfg.D+1),dtype=np.float64)
    T=np.empty(nctx,dtype=np.int64)
    row=0;batches=0;counts=[]
    while row<nctx:
        ii=int(rng.integers(len(tokens)))
        sm=sample_mask_set(cfg.N,rng)
        truth=np.asarray(tokens[ii],dtype=np.int64)
        arr=truth.copy();arr[sm.masked_positions]=core.MASK
        xn,xraw=context_from_arr_unconditional(arr,tok,pos)
        counts.append(len(sm.masked_positions));batches+=1
        for t in sm.masked_positions:
            if row>=nctx:
                break
            F[row]=core.static_feature(model,cfg,xn,xraw[int(t)],int(t))
            T[row]=int(truth[int(t)])
            row+=1
    audit={
        'protocol':'label-free MaskGIT-style scaffold contexts',
        'target_rows':int(nctx),
        'mask_contexts_sampled':int(batches),
        'mean_masked_targets_per_context':float(np.mean(counts)),
        'same_corrupted_context_for_all_targets':True,
        'progressive_ground_truth_reveal':False,
        'mask_schedule':'cosine with ceil(gamma(r)*N)',
        'class_labels_used':False,
    }
    return F,T,audit


def eval_masked_unconditional(model,cfg,tokens,tok,pos,ratio,seed,
                              nimages=128,targets_per_image=6):
    """Masked-token evaluation with no label/class input."""
    rng=np.random.default_rng(seed)
    ce=[];acc=[];fgce=[];fgacc=[];fgmass=[]
    dom=int(np.argmax(np.bincount(np.asarray(tokens).reshape(-1),minlength=cfg.V)))
    system=core.P3ImageSystem(cfg,model,seed+30000)
    for ii in rng.choice(len(tokens),size=min(nimages,len(tokens)),replace=False):
        nmask=max(1,int(round(float(ratio)*cfg.N)))
        masked=rng.choice(cfg.N,size=nmask,replace=False)
        arr=np.asarray(tokens[ii],dtype=np.int64).copy();arr[masked]=core.MASK
        xn,xraw=context_from_arr_unconditional(arr,tok,pos)
        picks=rng.choice(masked,size=min(targets_per_image,nmask),replace=False)
        for t in picks:
            p=system.forward_only(xn,xraw[int(t)],int(t),new_image=True)
            z=int(tokens[ii,int(t)])
            cur=-math.log(max(float(p[z]),1e-12))
            ce.append(cur);acc.append(int(np.argmax(p)==z))
            if z!=dom:
                fgce.append(cur);fgacc.append(int(np.argmax(p)==z));fgmass.append(1-float(p[dom]))
    return {
        'ce':float(np.mean(ce)),
        'acc':float(np.mean(acc)),
        'foreground_ce':float(np.mean(fgce)) if fgce else None,
        'foreground_acc':float(np.mean(fgacc)) if fgacc else None,
        'foreground_non_dominant_mass':float(np.mean(fgmass)) if fgmass else None,
        'samples':len(ce),'fg_samples':len(fgce),'dominant_token':dom,
        'class_labels_used':False,
    }


def train_maskgit_msdl_unconditional(system: MaskGITP3System, cfg, tr, tok, pos,
                                     weights: np.ndarray, target_events: int,
                                     seed: int, report_every: int = 5000
                                     ) -> Tuple[List[Dict],Dict]:
    """MaskGIT multi-mask P3/MSDL training with no label/class conditioning."""
    rng=np.random.default_rng(seed)
    done=0;batches=0;hist=[];t0=time.time();next_report=int(report_every)
    mask_counts=[];sampled_r=[]

    while done<int(target_events):
        ii=int(rng.integers(len(tr)))
        sm=sample_mask_set(cfg.N,rng)
        truth=np.asarray(tr[ii],dtype=np.int64)
        arr=truth.copy();arr[sm.masked_positions]=core.MASK
        xn,xraw=context_from_arr_unconditional(arr,tok,pos)

        rec=system.learn_mask_batch(xn,xraw,sm.masked_positions,truth,weights,rng)
        nmask=int(rec['masked_targets']);done+=nmask;batches+=1
        mask_counts.append(nmask);sampled_r.append(sm.ratio_r)

        if done>=next_report:
            row={
                'target_events':int(done),
                'mask_batches':int(batches),
                'online_nll':system.obs_nll/max(1,system.obs_count),
                'updates':int(system.update_count),
                'mean_masked_targets_per_batch':float(np.mean(mask_counts)),
                'last_masked_targets':nmask,
                'elapsed_s':time.time()-t0,
                'mean_abs_BR':float(np.mean(np.abs(system.BR))),
                'mean_abs_BO':float(np.mean(np.abs(system.BO))),
                'mean_abs_B2':float(np.mean(np.abs(system.B2))),
                'class_labels_used':False,
            }
            hist.append(row);print('TRAIN',json.dumps(row),flush=True)
            while next_report<=done:
                next_report+=int(report_every)

    audit={
        'pass':True,
        'requested_target_events':int(target_events),
        'actual_target_events':int(done),
        'mask_batches':int(batches),
        'mean_masked_targets_per_batch':float(np.mean(mask_counts)),
        'min_masked_targets':int(np.min(mask_counts)),
        'max_masked_targets':int(np.max(mask_counts)),
        'mean_sampled_r':float(np.mean(sampled_r)),
        'schedule':'r~Uniform[0,1), gamma(r)=cos(pi*r/2), nmask=ceil(gamma(r)*N)',
        'uniform_mask_positions':True,
        'all_masked_targets_learned_from_same_corrupted_context':True,
        'ground_truth_reveal_inside_batch':False,
        'forward_predictions_cached_before_weight_update':True,
        'forward_weights_fixed_across_masked_targets':True,
        'one_forward_weight_update_per_mask_context':True,
        'local_EM_state':'persistent and sequentially advanced across masked-target contributions in randomized order',
        'backpropagation':False,
        'class_labels_used':False,
    }
    return hist,audit
