#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R6B.3 formal context-epoch trainer.

Architecture is frozen.  This runner changes training infrastructure only:

1. exposure is counted in shuffled MaskGIT contexts/epochs rather than target events;
2. each image is visited at most once within an epoch, with a fresh cosine mask;
3. the complete live MaskGITP3System state is checkpointed, including persistent
   E/M and reciprocal states;
4. reciprocal Numba randomness is explicitly seeded once per mask context from the
   checkpointed training RNG, making continuation reproducible across processes;
5. fixed-mask validation and 8/16-round scheduled-parallel generation are evaluated
   before and after the training segment.

No P1-G neuronal Q/K reintegration is performed in this stage.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'snn_transform_p3_image'))
sys.path.insert(0, str(HERE))

import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
from maskgit_input_representation import (
    build_input_embeddings,
    context_from_arr_unconditional,
    representation_self_test,
)
from maskgit_msdl_training import sample_mask_set
from maskgit_msdl_training_unconditional import (
    MaskGITP3System,
    build_maskgit_scaffold_contexts_unconditional,
    eval_masked_unconditional,
    masking_self_test,
)
from maskgit_scheduled_parallel import controller_self_test
from r6b3_snn_encoder_maskgit_msdl import (
    generation_tokens_maskgit,
    posthoc_generation_metric,
    save_grid_unlabeled,
)

FORMAT = 'R6B3_FULL_DYNAMICAL_CHECKPOINT_V2'
MODEL_KEYS = (
    'WQ','WK','WV','WO','W1','W2','WR','BR','BO','B2',
    'maskO','mask1','mask2','omega','bq','bk','bv','bo','b1','b2',
    'theta','gamma2','beta2','eps2','lrq','lrk','lrv','lro','lr1','lr2',
    'lrr','recip_scale','sigq','sigk',
)
TRAINABLE_KEYS = ('WQ','WK','WV','WO','W1','W2','WR')


def model_dict_from_system(system):
    out = {}
    for k in MODEL_KEYS:
        if not hasattr(system, k):
            raise AttributeError(f'checkpoint/system is missing model field {k}')
        v = getattr(system, k)
        out[k] = np.array(v, copy=True) if isinstance(v, np.ndarray) else v
    return out


def system_inventory(system):
    inv = {}
    for name, value in vars(system).items():
        if isinstance(value, np.ndarray):
            inv[name] = {'type':'ndarray','shape':list(value.shape),'dtype':str(value.dtype)}
        elif isinstance(value, np.random.Generator):
            inv[name] = {'type':'numpy.Generator','bit_generator':type(value.bit_generator).__name__}
        elif isinstance(value, (int,float,bool,str,type(None))):
            inv[name] = {'type':type(value).__name__,'value':value}
        else:
            inv[name] = {'type':type(value).__name__}
    return inv


def save_full_checkpoint(path, *, system, meta, train_rng, schedule_state):
    payload = {
        'format': FORMAT,
        'meta': meta,
        'system': system,
        'train_rng_state': copy.deepcopy(train_rng.bit_generator.state),
        'schedule_state': copy.deepcopy(schedule_state),
        'python_random_state': random.getstate(),
        'torch_rng_state': torch.get_rng_state().cpu().numpy(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    inventory = {
        'format': FORMAT,
        'meta': meta,
        'schedule_state': {
            'epoch_index': int(schedule_state['epoch_index']),
            'offset': int(schedule_state['offset']),
            'permutation_length': int(len(schedule_state['permutation'])),
            'total_contexts': int(schedule_state['total_contexts']),
        },
        'system_state_inventory': system_inventory(system),
    }
    path.with_suffix('.inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')


def load_full_checkpoint(path):
    with path.open('rb') as f:
        payload = pickle.load(f)
    if payload.get('format') != FORMAT:
        raise ValueError(f'unsupported checkpoint format: {payload.get("format")}')
    train_rng = np.random.default_rng()
    train_rng.bit_generator.state = payload['train_rng_state']
    random.setstate(payload['python_random_state'])
    torch.set_rng_state(torch.from_numpy(payload['torch_rng_state']))
    return payload['system'], payload['meta'], train_rng, payload['schedule_state']


def new_schedule(n_train, rng, epoch_index=0, total_contexts=0):
    return {
        'epoch_index': int(epoch_index),
        'offset': 0,
        'permutation': rng.permutation(int(n_train)).astype(np.int64),
        'total_contexts': int(total_contexts),
    }


def consume_indices(schedule, n_train, count, rng):
    """Consume exactly count contexts, preserving one-visit-per-image epoch semantics."""
    pieces = []
    remaining = int(count)
    while remaining > 0:
        if len(schedule['permutation']) != int(n_train):
            raise ValueError('saved epoch permutation length does not match current training set')
        available = int(n_train) - int(schedule['offset'])
        take = min(remaining, available)
        lo = int(schedule['offset']); hi = lo + take
        pieces.append(schedule['permutation'][lo:hi])
        schedule['offset'] = hi
        schedule['total_contexts'] += take
        remaining -= take
        if schedule['offset'] == int(n_train):
            schedule['epoch_index'] += 1
            schedule['offset'] = 0
            schedule['permutation'] = rng.permutation(int(n_train)).astype(np.int64)
    return np.concatenate(pieces) if pieces else np.empty(0,dtype=np.int64)


def run_one_context(system, cfg, truth, tok, pos, ones, rng):
    sm = sample_mask_set(cfg.N, rng)
    arr = np.asarray(truth,dtype=np.int64).copy()
    arr[sm.masked_positions] = core.MASK
    xn,xraw = context_from_arr_unconditional(arr,tok,pos)

    # core.reciprocal_microsteps uses Numba's internal RNG.  Re-seeding it once
    # per context from the checkpointed Generator removes hidden RNG state from
    # continuation without changing the perturbation distribution or update rule.
    mirror_seed = int(rng.integers(1, 2**31-1))
    core.seed_numba(mirror_seed)
    rec = system.learn_mask_batch(xn,xraw,sm.masked_positions,truth,ones,rng)
    rec = dict(rec)
    rec['ratio_r'] = float(sm.ratio_r)
    rec['mask_ratio'] = float(sm.mask_ratio)
    rec['mirror_seed'] = mirror_seed
    return rec


def train_contexts(system, cfg, tokens, tok, pos, ones, indices, rng,
                   start_total_contexts, report_every=500):
    history=[]; target_events=0; t0=time.time()
    for local_i,image_index in enumerate(indices,start=1):
        rec = run_one_context(system,cfg,np.asarray(tokens[int(image_index)]),tok,pos,ones,rng)
        target_events += int(rec['masked_targets'])
        if local_i==1 or local_i==len(indices) or (report_every>0 and local_i%report_every==0):
            row={
                'context':int(start_total_contexts+local_i),
                'image_index':int(image_index),
                'masked_targets':int(rec['masked_targets']),
                'target_events_segment':int(target_events),
                'mask_ratio_sampled':float(rec['mask_ratio']),
                'batch_mean_nll':float(rec['mean_nll']),
                'cumulative_online_nll':float(system.obs_nll/max(1,system.obs_count)),
                'updates':int(system.update_count),
                'WR_norm':float(np.linalg.norm(system.WR)),
                'BR_mean_abs':float(np.mean(np.abs(system.BR))),
                'BO_mean_abs':float(np.mean(np.abs(system.BO))),
                'B2_mean_abs':float(np.mean(np.abs(system.B2))),
                'elapsed_s':float(time.time()-t0),
            }
            history.append(row)
            print('CONTEXT_PROGRESS',json.dumps(row),flush=True)
    return history,target_events


def compare_systems(a,b):
    keys = set(vars(a)) | set(vars(b))
    max_abs=0.0; mismatches=[]
    for k in sorted(keys):
        if not hasattr(a,k) or not hasattr(b,k):
            mismatches.append(k); continue
        x=getattr(a,k); y=getattr(b,k)
        if isinstance(x,np.ndarray) and isinstance(y,np.ndarray):
            if x.shape != y.shape:
                mismatches.append(k); continue
            if x.size:
                d=float(np.max(np.abs(x-y)))
                max_abs=max(max_abs,d)
                if not np.array_equal(x,y): mismatches.append(k)
        elif isinstance(x,np.random.Generator) and isinstance(y,np.random.Generator):
            if x.bit_generator.state != y.bit_generator.state: mismatches.append(k)
        elif isinstance(x,(int,float,bool,str,type(None))) and isinstance(y,(int,float,bool,str,type(None))):
            if x != y: mismatches.append(k)
    return {'pass':not mismatches,'max_array_abs_diff':max_abs,'mismatched_fields':mismatches}


def checkpoint_roundtrip_audit(path, *, reference_system, tokens, tok, pos, cfg, ones):
    loaded,meta,rng,schedule = load_full_checkpoint(path)
    immediate = compare_systems(reference_system,loaded)

    # One-step continuation probe on disposable clones.  Both clones start from
    # the exact saved state and consume the exact same next context/RNG stream.
    a = pickle.loads(pickle.dumps(reference_system,pickle.HIGHEST_PROTOCOL))
    b = pickle.loads(pickle.dumps(loaded,pickle.HIGHEST_PROTOCOL))
    rng_a=np.random.default_rng(); rng_a.bit_generator.state=copy.deepcopy(rng.bit_generator.state)
    rng_b=np.random.default_rng(); rng_b.bit_generator.state=copy.deepcopy(rng.bit_generator.state)
    sch_a=copy.deepcopy(schedule); sch_b=copy.deepcopy(schedule)
    ia=int(consume_indices(sch_a,len(tokens),1,rng_a)[0])
    ib=int(consume_indices(sch_b,len(tokens),1,rng_b)[0])
    if ia != ib:
        one_step={'pass':False,'reason':'next schedule index differs'}
    else:
        run_one_context(a,cfg,np.asarray(tokens[ia]),tok,pos,ones,rng_a)
        run_one_context(b,cfg,np.asarray(tokens[ib]),tok,pos,ones,rng_b)
        one_step=compare_systems(a,b)
        one_step['next_image_index']=ia
    return {
        'pass':bool(immediate['pass'] and one_step['pass']),
        'immediate_state_parity':immediate,
        'one_context_continuation_parity':one_step,
        'restored_meta':meta,
    }


def token_distribution_stats(ids,vocab_size):
    a=np.asarray(ids,dtype=np.int64).reshape(-1)
    counts=np.bincount(a,minlength=vocab_size).astype(np.float64)
    q=counts/max(1.0,counts.sum()); nz=q[q>0]
    return {
        'active_codes':int(np.sum(counts>0)),
        'perplexity':float(np.exp(-(nz*np.log(nz)).sum())),
        'top1_fraction':float(q.max()) if len(q) else 0.0,
    }


def evaluate_checkpoint(*,name,model,cfg,tokens,tok,pos,tokenizer,device,clf,
                        out,seed,eval_images,gen_samples,choice_temperature):
    masked={
        str(r):eval_masked_unconditional(model,cfg,tokens,tok,pos,r,
                                         seed+1000+int(100*r),eval_images)
        for r in (.25,.5,.9,1.0)
    }
    generations={}
    for iters in (8,16):
        ids,traces,audits=generation_tokens_maskgit(
            model,cfg,gen_samples,tok,pos,seed+2000,
            num_iter=iters,choice_temperature=choice_temperature)
        imgs=base.decode_snn(tokenizer,ids,device)
        save_grid_unlabeled(imgs,out/f'{name}_generation_{iters}rounds.png')
        generations[str(iters)]={
            'posthoc':posthoc_generation_metric(clf,imgs),
            'token_stats':token_distribution_stats(ids,cfg.V),
            'all_trace_audits_pass':bool(all(x['pass'] for x in audits)),
            'audits':audits,
        }
        (out/f'{name}_trace_{iters}rounds.json').write_text(json.dumps(traces,indent=2),encoding='utf-8')
    return {'masked_eval':masked,'generation':generations}


def selected_state_magnitudes(system):
    names=('WQ','WK','WV','WO','W1','W2','WR','BR','BO','B2',
           'ver','vmr','veq','vmq','vek','vmk','vev','vmv','veo','vmo','ve1','vm1','ve2','vm2')
    out={}
    for k in names:
        if hasattr(system,k):
            v=getattr(system,k)
            if isinstance(v,np.ndarray) and v.size:
                out[k]={'max_abs':float(np.max(np.abs(v))),'finite':bool(np.all(np.isfinite(v)))}
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--fractional-epoch',type=float,default=.25,
                    help='additional context-epoch equivalent to consume in this invocation')
    ap.add_argument('--resume',type=Path,default=None)
    ap.add_argument('--D',type=int,default=96);ap.add_argument('--heads',type=int,default=8)
    ap.add_argument('--features',type=int,default=24);ap.add_argument('--ff',type=int,default=192);ap.add_argument('--R',type=int,default=7)
    ap.add_argument('--scaffold-contexts',type=int,default=2000);ap.add_argument('--scaffold-paths',type=int,default=2500);ap.add_argument('--scaffold-lr',type=float,default=.6)
    ap.add_argument('--internal-lr-scale',type=float,default=.5)
    ap.add_argument('--eval-images',type=int,default=128);ap.add_argument('--gen-samples',type=int,default=36)
    ap.add_argument('--choice-temperature',type=float,default=4.5)
    ap.add_argument('--encode-batch',type=int,default=256)
    ap.add_argument('--report-every-contexts',type=int,default=500)
    ap.add_argument('--seed',type=int,default=7311);ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args()
    if not (0<a.fractional_epoch<=1.0): raise ValueError('--fractional-epoch must be in (0,1]')

    a.out.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(a.threads)
    device=torch.device('cpu')

    controller_audit=controller_self_test()
    masking_audit=masking_self_test(n_tokens=49,vocab_size=64,trials=128,seed=a.seed+17)
    if not controller_audit['pass'] or not masking_audit['pass']:
        raise RuntimeError('MaskGIT structural self-test failed')

    xall,yall,xte0,yte0=core.load(a.data)
    x012,y012=core.subset012(xall,yall);xte,yte=core.subset012(xte0,yte0)
    xtr,ytr,xva,yva=core.split(x012,y012,.1,7)

    tokenizer,tok_meta=base.load_frozen_tokenizer(a.tokenizer,device)
    tr,sp_tr=base.encode_snn(tokenizer,xtr,device,a.encode_batch)
    va,sp_va=base.encode_snn(tokenizer,xva,device,a.encode_batch)
    te,sp_te=base.encode_snn(tokenizer,xte,device,a.encode_batch)

    cfg=core.Config(D=a.D,H=a.heads,M=a.features,FF=a.ff,R=a.R)
    tok,pos,input_audit=build_input_embeddings(tokenizer,cfg.D,grid=7)
    rep_audit=representation_self_test(tokenizer,cfg.D,grid=7)
    if not rep_audit['pass']: raise RuntimeError('input representation self-test failed')
    ones=np.ones(cfg.V,np.float64)

    if a.resume is None:
        model=core.init_model(cfg,a.seed)
        for k in ('lrq','lrk','lrv','lro','lr1','lr2'): model[k]*=a.internal_lr_scale
        F,T,scaffold_audit=build_maskgit_scaffold_contexts_unconditional(
            model,cfg,tr,tok,pos,a.scaffold_contexts,a.seed+101)
        model['WR']=core.train_readout_msdl(F,T,ones,a.scaffold_paths,a.scaffold_lr,a.seed+202)
        system=MaskGITP3System(cfg,model,a.seed+303)
        train_rng=np.random.default_rng(a.seed+404)
        schedule=new_schedule(len(tr),train_rng,0,0)
        prior_meta=None
    else:
        system,prior_meta,train_rng,schedule=load_full_checkpoint(a.resume)
        scaffold_audit=prior_meta.get('scaffold_audit',{'restored':True})
        expected=(cfg.D,cfg.H,cfg.M,cfg.FF,cfg.R,cfg.N,cfg.V)
        got=(system.c.D,system.c.H,system.c.M,system.c.FF,system.c.R,system.c.N,system.c.V)
        if got!=expected: raise ValueError(f'resume architecture mismatch: {got} != {expected}')

    clf=LogisticRegression(max_iter=250,solver='lbfgs',random_state=7).fit(xtr.reshape(len(xtr),-1),ytr)
    baseline_model=model_dict_from_system(system)
    baseline_eval=evaluate_checkpoint(
        name='segment_start',model=baseline_model,cfg=cfg,tokens=va,tok=tok,pos=pos,
        tokenizer=tokenizer,device=device,clf=clf,out=a.out,seed=a.seed,
        eval_images=a.eval_images,gen_samples=a.gen_samples,choice_temperature=a.choice_temperature)

    before={k:np.array(getattr(system,k),copy=True) for k in TRAINABLE_KEYS}
    start_total=int(schedule['total_contexts'])
    n_contexts=max(1,int(math.ceil(a.fractional_epoch*len(tr))))
    indices=consume_indices(schedule,len(tr),n_contexts,train_rng)
    history,target_events=train_contexts(
        system,cfg,tr,tok,pos,ones,indices,train_rng,start_total,
        report_every=a.report_every_contexts)

    final_model=model_dict_from_system(system)
    actual_epoch_equiv=float(schedule['total_contexts']/len(tr))
    tag=f'{actual_epoch_equiv:.4f}'.replace('.','p')
    meta={
        'seed':a.seed,
        'contexts_completed_total':int(schedule['total_contexts']),
        'context_epoch_equivalent':actual_epoch_equiv,
        'completed_full_epochs':int(schedule['epoch_index']),
        'current_epoch_offset':int(schedule['offset']),
        'fractional_epoch_requested_this_run':float(a.fractional_epoch),
        'contexts_this_run':int(n_contexts),
        'target_events_this_run':int(target_events),
        'n_train_contexts_per_epoch':int(len(tr)),
        'architecture':{'D':cfg.D,'H':cfg.H,'M':cfg.M,'FF':cfg.FF,'R':cfg.R,'N':cfg.N,'V':cfg.V},
        'scaffold_audit':scaffold_audit,
        'tokenizer_metadata':tok_meta,
        'numba_reciprocal_rng':'explicit per-context seed drawn from checkpointed train_rng',
        'prior_checkpoint_meta':prior_meta,
    }
    ckpt=a.out/f'checkpoint_{tag}epoch.pkl'
    save_full_checkpoint(ckpt,system=system,meta=meta,train_rng=train_rng,schedule_state=schedule)
    checkpoint_audit=checkpoint_roundtrip_audit(
        ckpt,reference_system=system,tokens=tr,tok=tok,pos=pos,cfg=cfg,ones=ones)
    if not checkpoint_audit['pass']:
        raise RuntimeError(f'full dynamical checkpoint parity failed: {checkpoint_audit}')

    final_eval=evaluate_checkpoint(
        name=f'checkpoint_{tag}epoch',model=final_model,cfg=cfg,tokens=va,tok=tok,pos=pos,
        tokenizer=tokenizer,device=device,clf=clf,out=a.out,seed=a.seed,
        eval_images=a.eval_images,gen_samples=a.gen_samples,choice_temperature=a.choice_temperature)

    moves={
        k:float(np.linalg.norm(getattr(system,k)-before[k])/(np.linalg.norm(before[k])+1e-12))
        for k in TRAINABLE_KEYS
    }
    all_arrays_finite=bool(all(np.all(np.isfinite(v)) for v in vars(system).values() if isinstance(v,np.ndarray)))
    result={
        'stage':'R6B.3 G0 context-epoch training',
        'architecture_changed':False,
        'p1g_reintroduced':False,
        'labels_enter_neural_input':False,
        'labels_enter_msdl_learning':False,
        'labels_enter_generation':False,
        'labels_used_external_posthoc_only':True,
        'context_epoch_definition':'one shuffled visit per training image with one fresh MaskGIT cosine mask; permutation persisted across resume',
        'meta':meta,
        'controller_self_test':controller_audit,
        'masking_self_test':masking_audit,
        'input_representation_audit':rep_audit,
        'tokenizer_activity':{'train':sp_tr,'val':sp_va,'test':sp_te},
        'segment_start_evaluation':baseline_eval,
        'segment_final_evaluation':final_eval,
        'history':history,
        'relative_weight_move_this_run':moves,
        'mean_abs_reciprocal':{
            'BR':float(np.mean(np.abs(system.BR))),
            'BO':float(np.mean(np.abs(system.BO))),
            'B2':float(np.mean(np.abs(system.B2))),
        },
        'all_system_arrays_finite':all_arrays_finite,
        'selected_state_magnitudes':selected_state_magnitudes(system),
        'checkpoint':str(ckpt),
        'checkpoint_roundtrip_audit':checkpoint_audit,
        'primary_comparison':{
            'ce50_start':baseline_eval['masked_eval']['0.5']['ce'],
            'ce50_final':final_eval['masked_eval']['0.5']['ce'],
            'acc50_start':baseline_eval['masked_eval']['0.5']['acc'],
            'acc50_final':final_eval['masked_eval']['0.5']['acc'],
            'ce100_start':baseline_eval['masked_eval']['1.0']['ce'],
            'ce100_final':final_eval['masked_eval']['1.0']['ce'],
        },
    }
    (a.out/'G0_RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('G0_RESULT',json.dumps({
        'contexts_this_run':n_contexts,
        'context_epoch_equivalent':actual_epoch_equiv,
        'target_events_this_run':target_events,
        'all_system_arrays_finite':all_arrays_finite,
        'checkpoint_parity':checkpoint_audit['pass'],
        'ce50_start':result['primary_comparison']['ce50_start'],
        'ce50_final':result['primary_comparison']['ce50_final'],
        'ce100_start':result['primary_comparison']['ce100_start'],
        'ce100_final':result['primary_comparison']['ce100_final'],
        'checkpoint':str(ckpt),
    }),flush=True)


if __name__=='__main__':
    main()
