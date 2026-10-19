#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controlled effectiveness test for the newly restored R6B.4 forward plasticity.

Question tested
---------------
Does restoring plasticity of token/position embeddings, biases and normalization
parameters produce a measurable learning benefit beyond the original R6B.3
seven-matrix learner, and is aux_lr_scale=1.0 more effective than the conservative
0.1 smoke setting?

Three paired arms start from the exact same 2.0-epoch R6B.3 checkpoint and consume
exactly the same image order, masks, target order RNG stream and per-context Numba
mirror seeds:

  A) r6b3_frozen_new_params : original seven plastic matrices only;
  B) r6b4_aux_0p1          : restored parameters, auxiliary scale 0.1;
  C) r6b4_aux_1p0          : restored parameters, auxiliary scale 1.0.

No labels enter learning or generation. No BP/BPTT/autograd optimizer is used.
This is a short diagnostic continuation, not the formal third epoch.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import random
import sys
import time
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
import r6b3_decoder_operating_point_audit as coarse
import r6b3_decoder_spatial_refinement_audit as spatial
from maskgit_input_representation import build_input_embeddings, context_from_arr_unconditional
from maskgit_msdl_training import sample_mask_set
from r6b3_snn_encoder_maskgit_msdl import generation_tokens_maskgit, save_grid_unlabeled
from r6b4_full_forward_plasticity import movement_report, upgrade_r6b3_system


MASK_RATIOS=(.25,.5,.9,1.0)
GEN_SETTINGS=((3,8.0),(3,7.0))


def clone_rng(state):
    rng=np.random.default_rng()
    rng.bit_generator.state=copy.deepcopy(state)
    return rng


def all_arrays_finite(obj):
    bad=[]
    for k,v in vars(obj).items():
        if isinstance(v,np.ndarray) and not np.all(np.isfinite(v)):
            bad.append(k)
    return {'pass':not bad,'bad_arrays':bad}


def array_relative_move(before,after):
    a=np.asarray(before,dtype=np.float64);b=np.asarray(after,dtype=np.float64)
    d=b-a
    return {
        'max_abs_change':float(np.max(np.abs(d))) if d.size else 0.0,
        'delta_norm':float(np.linalg.norm(d)),
        'relative_change':float(np.linalg.norm(d)/(np.linalg.norm(a)+1e-12)),
        'finite':bool(np.all(np.isfinite(b))),
    }


def parameter_movement_for_baseline(before,system):
    out={}
    for k,a in before.items():
        out[k]=array_relative_move(a,getattr(system,k))
    return out


def paired_context(system,variant,truth,tok_fixed,pos_fixed,ones,rng):
    sm=sample_mask_set(system.c.N,rng)
    arr=np.asarray(truth,dtype=np.int64).copy()
    arr[sm.masked_positions]=core.MASK
    mirror_seed=int(rng.integers(1,2**31-1))
    core.seed_numba(mirror_seed)
    if variant=='baseline':
        xn,xraw=context_from_arr_unconditional(arr,tok_fixed,pos_fixed)
        rec=system.learn_mask_batch(xn,xraw,sm.masked_positions,truth,ones,rng)
    else:
        rec=system.learn_mask_batch_arr(arr,sm.masked_positions,truth,ones,rng)
    sig=hashlib.sha256()
    sig.update(np.asarray(sm.masked_positions,dtype=np.int64).tobytes())
    sig.update(np.asarray([mirror_seed],dtype=np.int64).tobytes())
    return dict(rec),sig.hexdigest(),int(len(sm.masked_positions)),mirror_seed


def train_arm(name,system,variant,tokens,indices,tok_fixed,pos_fixed,ones,rng_state,report_every):
    rng=clone_rng(rng_state)
    t0=time.time();nll_sum=0.0;targets=0;signatures=[];history=[]
    for j,image_index in enumerate(indices,start=1):
        rec,sig,nmask,mirror_seed=paired_context(
            system,variant,np.asarray(tokens[int(image_index)],dtype=np.int64),
            tok_fixed,pos_fixed,ones,rng)
        nll_sum+=float(rec['mean_nll'])*int(rec['masked_targets'])
        targets+=int(rec['masked_targets'])
        signatures.append(sig)
        if j==1 or j==len(indices) or (report_every>0 and j%report_every==0):
            row={'arm':name,'context_local':j,'image_index':int(image_index),
                 'masked_targets':int(rec['masked_targets']),
                 'segment_mean_nll':float(nll_sum/max(1,targets)),
                 'mirror_seed':mirror_seed,'elapsed_s':float(time.time()-t0)}
            history.append(row);print('EFFECTIVENESS_PROGRESS',json.dumps(row),flush=True)
    combined=hashlib.sha256(''.join(signatures).encode('ascii')).hexdigest()
    return {
        'mean_online_nll':float(nll_sum/max(1,targets)),
        'target_events':int(targets),
        'context_signature_sha256':combined,
        'rng_state_after':copy.deepcopy(rng.bit_generator.state),
        'history':history,
    }


def masked_eval(system,tokens,tok,pos,seed,eval_images):
    model=trainer.model_dict_from_system(system)
    return {str(r):trainer.eval_masked_unconditional(
        model,system.c,tokens,tok,pos,r,seed+int(round(r*1000)),eval_images)
        for r in MASK_RATIOS}


def generate_eval(label,system,tok,pos,tokenizer,device,real_p,real_pos_p,real_pair,
                  out,seed,samples,grid_samples):
    model=trainer.model_dict_from_system(system)
    rows={}
    for rounds,temp in GEN_SETTINGS:
        key=f'r{rounds}_t{temp:g}'
        ids,traces,audits=generation_tokens_maskgit(
            model,system.c,samples,tok,pos,seed,
            num_iter=rounds,choice_temperature=temp)
        global_m=coarse.distribution_metrics(ids,real_p,system.c.V)
        spatial_m=spatial.positionwise_metrics(ids,real_pos_p,system.c.V,system.c.N)
        pair_m=spatial.pair_metrics(ids,real_pair,system.c.V,system.c.N,7)
        imgs=base.decode_snn(tokenizer,np.asarray(ids[:grid_samples]),device)
        save_grid_unlabeled(imgs,out/f'{label}_{key}_grid.png')
        rows[key]={
            'global':global_m,
            'spatial':spatial_m,
            'pair':pair_m,
            'all_trace_audits_pass':bool(all(x['pass'] for x in audits)),
        }
        (out/f'{label}_{key}_traces.json').write_text(json.dumps(traces,indent=2),encoding='utf-8')
    return rows


def summarize_mask_delta(start,end):
    out={}
    for r in map(str,MASK_RATIOS):
        out[r]={
            'ce_start':float(start[r]['ce']),
            'ce_end':float(end[r]['ce']),
            'ce_delta':float(end[r]['ce']-start[r]['ce']),
            'acc_start':float(start[r]['acc']),
            'acc_end':float(end[r]['acc']),
            'acc_delta':float(end[r]['acc']-start[r]['acc']),
        }
    return out


def generation_delta(start,end):
    out={}
    for key in end:
        out[key]={
            'global_js_start':float(start[key]['global']['js_to_real_nats']),
            'global_js_end':float(end[key]['global']['js_to_real_nats']),
            'global_js_delta':float(end[key]['global']['js_to_real_nats']-start[key]['global']['js_to_real_nats']),
            'spatial_js_start':float(start[key]['spatial']['mean_js_nats']),
            'spatial_js_end':float(end[key]['spatial']['mean_js_nats']),
            'spatial_js_delta':float(end[key]['spatial']['mean_js_nats']-start[key]['spatial']['mean_js_nats']),
            'pair_js_start':float(start[key]['pair']['mean_orientation_js_nats']),
            'pair_js_end':float(end[key]['pair']['mean_orientation_js_nats']),
            'pair_js_delta':float(end[key]['pair']['mean_orientation_js_nats']-start[key]['pair']['mean_orientation_js_nats']),
        }
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--contexts',type=int,default=1024)
    ap.add_argument('--eval-images',type=int,default=512)
    ap.add_argument('--gen-samples',type=int,default=256)
    ap.add_argument('--grid-samples',type=int,default=36)
    ap.add_argument('--encode-batch',type=int,default=256)
    ap.add_argument('--seed',type=int,default=7311)
    ap.add_argument('--report-every',type=int,default=256)
    ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(a.threads)
    device=torch.device('cpu')

    source,meta,train_rng,schedule=trainer.load_full_checkpoint(a.checkpoint)
    if abs(float(meta.get('context_epoch_equivalent',-1))-2.0)>1e-12:
        raise ValueError('requires exact 2.0-epoch R6B.3 checkpoint')
    if int(schedule['offset'])+a.contexts>len(schedule['permutation']):
        raise ValueError('test contexts must stay within the saved epoch-3 permutation')
    indices=np.asarray(schedule['permutation'][int(schedule['offset']):int(schedule['offset'])+a.contexts],dtype=np.int64)
    rng_state=copy.deepcopy(train_rng.bit_generator.state)
    cfg=source.c

    xall,yall,_xte,_yte=core.load(a.data)
    x012,y012=core.subset012(xall,yall)
    xtr,_ytr,xva,_yva=core.split(x012,y012,.1,7)
    tokenizer,tok_meta=base.load_frozen_tokenizer(a.tokenizer,device)
    tr,sp_tr=base.encode_snn(tokenizer,xtr,device,a.encode_batch)
    va,sp_va=base.encode_snn(tokenizer,xva,device,a.encode_batch)
    tok0,pos0,input_audit=build_input_embeddings(tokenizer,cfg.D,grid=7)
    ones=np.ones(cfg.V,dtype=np.float64)

    # Label-free real token references for generation structure diagnostics.
    real_counts,real_p=coarse.probability_from_ids(va,cfg.V)
    real_pos_counts,real_pos_p=spatial.position_probabilities(va,cfg.V,cfg.N)
    real_pair=spatial.pair_reference(va,cfg.V,cfg.N,7)
    real_reference={
        'n_validation_images':int(len(va)),
        'global_active_codes':int(np.sum(real_counts>0)),
        'global_perplexity':float(np.exp(coarse.entropy_nats(real_p))),
        'global_top1_fraction':float(real_p.max()),
        'position_active_codes_mean':float(np.mean(np.sum(real_pos_counts>0,axis=1))),
    }

    # Common start metrics. Upgrading must not change the forward function before learning.
    start_mask=masked_eval(source,va,tok0,pos0,a.seed+10000,a.eval_images)
    start_gen=generate_eval('start_2epoch',source,tok0,pos0,tokenizer,device,
                            real_p,real_pos_p,real_pair,a.out,a.seed+20000,
                            a.gen_samples,a.grid_samples)

    baseline=pickle.loads(pickle.dumps(source,pickle.HIGHEST_PROTOCOL))
    full01=pickle.loads(pickle.dumps(source,pickle.HIGHEST_PROTOCOL))
    full10=pickle.loads(pickle.dumps(source,pickle.HIGHEST_PROTOCOL))
    upgrade_r6b3_system(full01,tok0,pos0,aux_lr_scale=.1)
    upgrade_r6b3_system(full10,tok0,pos0,aux_lr_scale=1.0)

    old_keys=('WQ','WK','WV','WO','W1','W2','WR')
    baseline_before={k:np.array(getattr(baseline,k),copy=True) for k in old_keys}
    full01_before=full01.forward_parameter_snapshot()
    full10_before=full10.forward_parameter_snapshot()

    train_results={}
    train_results['r6b3_frozen_new_params']=train_arm(
        'r6b3_frozen_new_params',baseline,'baseline',tr,indices,tok0,pos0,ones,rng_state,a.report_every)
    train_results['r6b4_aux_0p1']=train_arm(
        'r6b4_aux_0p1',full01,'full',tr,indices,tok0,pos0,ones,rng_state,a.report_every)
    train_results['r6b4_aux_1p0']=train_arm(
        'r6b4_aux_1p0',full10,'full',tr,indices,tok0,pos0,ones,rng_state,a.report_every)

    sigs={k:v['context_signature_sha256'] for k,v in train_results.items()}
    paired_stream_identical=(len(set(sigs.values()))==1)

    arms={
        'r6b3_frozen_new_params':(baseline,tok0,pos0),
        'r6b4_aux_0p1':(full01,full01.tok,full01.pos),
        'r6b4_aux_1p0':(full10,full10.tok,full10.pos),
    }
    end={};comparisons={}
    for i,(name,(system,tok,pos)) in enumerate(arms.items()):
        end_mask=masked_eval(system,va,tok,pos,a.seed+10000,a.eval_images)
        end_gen=generate_eval(name,system,tok,pos,tokenizer,device,real_p,real_pos_p,real_pair,
                              a.out,a.seed+20000,a.gen_samples,a.grid_samples)
        end[name]={'masked_eval':end_mask,'generation':end_gen,'finite':all_arrays_finite(system)}
        comparisons[name]={
            'masked_change_from_common_start':summarize_mask_delta(start_mask,end_mask),
            'generation_change_from_common_start':generation_delta(start_gen,end_gen),
        }

    movement={
        'r6b3_frozen_new_params':parameter_movement_for_baseline(baseline_before,baseline),
        'r6b4_aux_0p1':movement_report(full01_before,full01),
        'r6b4_aux_1p0':movement_report(full10_before,full10),
    }

    # Direct paired comparison: lower CE/JS is better. No arbitrary weighted score.
    direct={}
    for r in map(str,MASK_RATIOS):
        b=float(end['r6b3_frozen_new_params']['masked_eval'][r]['ce'])
        x01=float(end['r6b4_aux_0p1']['masked_eval'][r]['ce'])
        x10=float(end['r6b4_aux_1p0']['masked_eval'][r]['ce'])
        direct[f'mask_{r}']={'baseline_ce':b,'aux0p1_ce':x01,'aux1p0_ce':x10,
                             'aux0p1_minus_baseline':x01-b,'aux1p0_minus_baseline':x10-b}
    for key in start_gen:
        def trip(metric_path):
            def val(name):
                d=end[name]['generation'][key]
                for p in metric_path:d=d[p]
                return float(d)
            b=val('r6b3_frozen_new_params');x01=val('r6b4_aux_0p1');x10=val('r6b4_aux_1p0')
            return {'baseline':b,'aux0p1':x01,'aux1p0':x10,
                    'aux0p1_minus_baseline':x01-b,'aux1p0_minus_baseline':x10-b}
        direct[f'{key}_global_js']=trip(('global','js_to_real_nats'))
        direct[f'{key}_spatial_js']=trip(('spatial','mean_js_nats'))
        direct[f'{key}_pair_js']=trip(('pair','mean_orientation_js_nats'))

    report={
        'stage':'R6B4_FULL_FORWARD_EFFECTIVENESS_TEST',
        'scientific_question':'Do restored forward parameters add measurable benefit beyond R6B.3, and does aux scale 1.0 outperform the conservative 0.1 setting?',
        'formal_third_epoch':False,
        'source_context_epoch':float(meta['context_epoch_equivalent']),
        'contexts_per_arm':int(a.contexts),
        'context_epoch_fraction_per_arm':float(a.contexts/len(tr)),
        'paired_training_stream_identical':bool(paired_stream_identical),
        'context_signature_sha256_by_arm':sigs,
        'labels_used_in_learning_or_generation':False,
        'backpropagation_or_autograd_used':False,
        'tokenizer_metadata':tok_meta,
        'input_audit':input_audit,
        'train_tokenizer_activity':sp_tr,'validation_tokenizer_activity':sp_va,
        'real_reference':real_reference,
        'common_start':{'masked_eval':start_mask,'generation':start_gen},
        'training':train_results,
        'end':end,
        'movement':movement,
        'comparisons':comparisons,
        'direct_paired_comparison':direct,
    }
    (a.out/'R6B4_FULL_FORWARD_EFFECTIVENESS_RESULTS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('R6B4_EFFECTIVENESS_DONE',json.dumps({
        'paired_training_stream_identical':paired_stream_identical,
        'direct_paired_comparison':direct,
        'full_1p0_movement':movement['r6b4_aux_1p0'],
    },indent=2),flush=True)


if __name__=='__main__':main()
