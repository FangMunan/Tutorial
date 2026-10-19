#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Short gate for R6B.4 complete forward-parameter plasticity.

Loads the exact R6B.3 2.0-epoch checkpoint, upgrades it only in memory, then
consumes a small number of fresh epoch-3 MaskGIT contexts.  This is a movement /
finite-state gate, not a generation-quality experiment and not the formal 2->3
continuation.
"""
from __future__ import annotations

import argparse
import json
import pickle
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
import r6b3_context_epoch_training as trainer
from maskgit_input_representation import build_input_embeddings
from maskgit_msdl_training import sample_mask_set
from r6b4_full_forward_plasticity import (
    FORWARD_PARAMETER_KEYS,
    movement_report,
    upgrade_r6b3_system,
)


def all_arrays_finite(obj):
    bad=[]
    for k,v in vars(obj).items():
        if isinstance(v,np.ndarray) and not np.all(np.isfinite(v)):
            bad.append(k)
    return {'pass':not bad,'nonfinite_arrays':bad}


def max_old_forward_diff(before,system):
    mx=0.0;bad=[]
    for k,a in before.items():
        b=np.asarray(getattr(system,k));d=float(np.max(np.abs(b-a))) if a.size else 0.0
        mx=max(mx,d)
        if d!=0.0:bad.append(k)
    return {'pass':not bad,'max_abs_diff':mx,'changed_fields':bad}


def zero_forward_learning_rates(system):
    for k in ('lrq','lrk','lrv','lro','lr1','lr2','lrr'):
        setattr(system,k,0.0)
    system.aux_lr_scale=0.0


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--contexts',type=int,default=12)
    ap.add_argument('--aux-lr-scale',type=float,default=0.10)
    ap.add_argument('--encode-batch',type=int,default=256)
    ap.add_argument('--seed',type=int,default=7311)
    ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args()
    if a.contexts<1:raise ValueError('--contexts must be positive')
    a.out.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(a.threads)
    device=torch.device('cpu')

    old_system,meta,train_rng,schedule=trainer.load_full_checkpoint(a.checkpoint)
    if abs(float(meta.get('context_epoch_equivalent',-1))-2.0)>1e-12:
        raise ValueError(f'expected exact 2.0-epoch checkpoint, got {meta.get("context_epoch_equivalent")}')
    if getattr(old_system,'accum',0)!=0:raise ValueError('checkpoint is not at a clean update boundary')

    xall,yall,_xte0,_yte0=core.load(a.data)
    x012,y012=core.subset012(xall,yall)
    xtr,_ytr,_xva,_yva=core.split(x012,y012,.1,7)
    tokenizer,tok_meta=base.load_frozen_tokenizer(a.tokenizer,device)
    tr,sp_tr=base.encode_snn(tokenizer,xtr,device,a.encode_batch)
    tok0,pos0,input_audit=build_input_embeddings(tokenizer,old_system.c.D,grid=7)

    old_forward={k:np.array(getattr(old_system,k),copy=True) for k in (
        'WQ','WK','WV','WO','W1','W2','WR','bq','bk','bv','bo','b1','b2',
        'gamma2','beta2','theta')}
    old_learning_state_names=set(vars(old_system))

    system=upgrade_r6b3_system(old_system,tok0,pos0,aux_lr_scale=a.aux_lr_scale)
    upgrade_preserves_old=max_old_forward_diff(old_forward,system)
    if not upgrade_preserves_old['pass']:
        raise RuntimeError(f'upgrade changed pre-existing forward parameters: {upgrade_preserves_old}')
    before=system.forward_parameter_snapshot()

    # Verify the restoration did not add learned feedback/inverse matrices.
    new_state_names=set(vars(system))-old_learning_state_names
    forbidden_new_feedback=[x for x in new_state_names if x.startswith('B')]
    no_new_feedback=(len(forbidden_new_feedback)==0)

    control=pickle.loads(pickle.dumps(system,pickle.HIGHEST_PROTOCOL))
    zero_forward_learning_rates(control)
    ctrl_rng=np.random.default_rng(99173)
    ctrl_truth=np.asarray(tr[0],dtype=np.int64)
    ctrl_sm=sample_mask_set(control.c.N,ctrl_rng)
    ctrl_arr=ctrl_truth.copy();ctrl_arr[ctrl_sm.masked_positions]=core.MASK
    core.seed_numba(171717)
    ctrl_before=control.forward_parameter_snapshot()
    control.learn_mask_batch_arr(ctrl_arr,ctrl_sm.masked_positions,ctrl_truth,
                                 np.ones(control.c.V),ctrl_rng)
    ctrl_move=movement_report(ctrl_before,control,atol=1e-14)
    m_off_pass=not any(v['moved'] for v in ctrl_move['groups'].values())

    indices=trainer.consume_indices(schedule,len(tr),a.contexts,train_rng)
    seen_visible=set();seen_targets=set();rows=[]
    ones=np.ones(system.c.V,dtype=np.float64)
    start_total=int(schedule['total_contexts'])-len(indices)
    for j,image_index in enumerate(indices,start=1):
        truth=np.asarray(tr[int(image_index)],dtype=np.int64)
        sm=sample_mask_set(system.c.N,train_rng)
        arr=truth.copy();arr[sm.masked_positions]=core.MASK
        visible=np.setdiff1d(np.arange(system.c.N),sm.masked_positions,assume_unique=False)
        seen_visible.update(int(x) for x in arr[visible] if int(x)<system.c.V)
        seen_targets.update(int(truth[t]) for t in sm.masked_positions)
        mirror_seed=int(train_rng.integers(1,2**31-1));core.seed_numba(mirror_seed)
        rec=system.learn_mask_batch_arr(arr,sm.masked_positions,truth,ones,train_rng)
        rows.append({'context':int(start_total+j),'image_index':int(image_index),
                     'masked_targets':int(rec['masked_targets']),
                     'mean_nll':float(rec['mean_nll']),'mirror_seed':mirror_seed})
        print('R6B4_SMOKE_CONTEXT',json.dumps(rows[-1]),flush=True)

    move=movement_report(before,system)
    visual_delta=np.max(np.abs(system.tok[:system.c.V]-before['tok'][:system.c.V]),axis=1)
    seen_visible_sorted=sorted(seen_visible)
    seen_target_sorted=sorted(seen_targets)
    seen_visible_moved=[z for z in seen_visible_sorted if visual_delta[z]>1e-12]
    seen_visible_coverage={
        'seen_visible_token_ids':seen_visible_sorted,
        'seen_visible_count':len(seen_visible_sorted),
        'seen_visible_rows_moved':seen_visible_moved,
        'seen_visible_rows_moved_count':len(seen_visible_moved),
        'all_seen_visible_rows_move':len(seen_visible_moved)==len(seen_visible_sorted),
        'masked_target_token_ids_seen':seen_target_sorted,
    }
    finite=all_arrays_finite(system)

    gates={
        'upgrade_preserves_all_preexisting_forward_values_exactly':upgrade_preserves_old['pass'],
        'no_new_learned_feedback_or_inverse_matrix':bool(no_new_feedback),
        'm_off_freezes_every_forward_parameter_group':bool(m_off_pass),
        'all_forward_parameter_groups_move':bool(move['all_forward_groups_move']),
        'all_forward_parameter_groups_finite':bool(move['all_forward_groups_finite'] and finite['pass']),
        'all_seen_visible_token_embedding_rows_move':bool(seen_visible_coverage['all_seen_visible_rows_move']),
        'all_position_embedding_rows_move':bool(move['position_rows_moved']==move['position_rows_total']),
        'labels_enter_neural_input':False,
        'backpropagation_used':False,
    }
    report={
        'stage':'R6B.4 complete forward-parameter plasticity smoke gate',
        'source_checkpoint_epoch':float(meta['context_epoch_equivalent']),
        'formal_training_continuation':False,
        'contexts_consumed_in_disposable_smoke':int(a.contexts),
        'aux_lr_scale':float(a.aux_lr_scale),
        'forward_parameter_keys':list(FORWARD_PARAMETER_KEYS),
        'structural_fixed_parameters':['theta','omega','maskO','mask1','mask2','eps2','learning-rate scalars','frozen SNN-VQ encoder/decoder'],
        'new_state_names':sorted(new_state_names),
        'forbidden_new_feedback_names':forbidden_new_feedback,
        'input_initialization_audit':input_audit,
        'tokenizer_metadata':tok_meta,
        'tokenizer_train_activity':sp_tr,
        'upgrade_preservation_audit':upgrade_preserves_old,
        'm_off_forward_movement':ctrl_move,
        'movement':move,
        'seen_token_movement':seen_visible_coverage,
        'all_system_arrays_finite':finite,
        'contexts':rows,
        'gates':gates,
        'pass':bool(all(gates.values())),
    }
    (a.out/'R6B4_FULL_FORWARD_PLASTICITY_SMOKE.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('R6B4_FULL_FORWARD_PLASTICITY_SMOKE_DONE',json.dumps({
        'pass':report['pass'],'gates':gates,
        'visual_rows_moved':move['visual_token_rows_moved'],
        'position_rows_moved':move['position_rows_moved'],
        'seen_visible':seen_visible_coverage,
    },indent=2),flush=True)
    if not report['pass']:raise SystemExit(2)


if __name__=='__main__':main()
