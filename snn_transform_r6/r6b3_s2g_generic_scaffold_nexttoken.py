#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R6B.3 S2G — minimal non-MaskGIT scaffold generality test.

Question
--------
Does the WR startup scaffold accelerate P3/MSDL only for MaskGIT, or does the
same WR -> BR bootstrap advantage appear in an ordinary next-token task?

Task
----
Use the same frozen R6B.1 SNN-VQ token sequences z[0:49], but remove MaskGIT
training entirely.  One online event samples an image and a target position
1 <= t < 49, presents only the causal prefix z[:t], keeps z[t:] at MASK, and
predicts z[t].  Thus there is:

  * no random multi-mask objective
  * no cosine masking schedule
  * no simultaneous masked reconstruction
  * no confidence remasking
  * no iterative generation
  * no class conditioning

Only the startup intervention differs:

  A: next-token WR scaffold, then full P3/MSDL
  B: exact native WR=0 start, then full P3/MSDL

Both online branches use the same initial internal P3 weights and the exact same
precomputed (image, target-position) schedule.  The MaskGITP3System class is
reused only as the already-audited exact local-update engine; every batch here
contains exactly one target and the dataflow is causal next-token prediction.

Predeclared minimal decision rule at 10k online target events:

  generic_bootstrap_support =
      (native_online_NLL - scaffold_online_NLL > 0.5)
      and (mean|BR|_scaffold / mean|BR|_native > 5)

This is deliberately a single-seed diagnostic, not a benchmark.
"""
from __future__ import annotations

import argparse,csv,hashlib,json,math,random,sys,time
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
from maskgit_msdl_training import MaskGITP3System

INTERNAL=('WQ','WK','WV','WO','W1','W2')


def clone_model(model):
    return {k:(np.array(v,copy=True) if isinstance(v,np.ndarray) else v) for k,v in model.items()}


def causal_context(truth,t,cfg,tok,pos):
    """Only z[:t] is visible; target z[t] and all future tokens are MASK."""
    t=int(t)
    if not (1 <= t < cfg.N):
        raise ValueError(f'next-token target must be in [1,{cfg.N-1}], got {t}')
    arr=np.full(cfg.N,core.MASK,dtype=np.int64)
    arr[:t]=np.asarray(truth,dtype=np.int64)[:t]
    if arr[t] != core.MASK or np.any(arr[t:] != core.MASK):
        raise AssertionError('target/future leakage in causal context')
    return context_from_arr_unconditional(arr,tok,pos)


def build_nexttoken_scaffold(model,cfg,tokens,tok,pos,nrows,seed):
    rng=np.random.default_rng(seed)
    F=np.empty((int(nrows),cfg.D+1),dtype=np.float64)
    T=np.empty(int(nrows),dtype=np.int64)
    positions=[]
    for i in range(int(nrows)):
        ii=int(rng.integers(len(tokens)))
        t=int(rng.integers(1,cfg.N))
        truth=np.asarray(tokens[ii],dtype=np.int64)
        xn,xraw=causal_context(truth,t,cfg,tok,pos)
        F[i]=core.static_feature(model,cfg,xn,xraw[t],t)
        T[i]=int(truth[t])
        positions.append(t)
    return F,T,{
        'protocol':'ordinary causal next-token scaffold',
        'target_rows':int(nrows),
        'target_position_min':int(np.min(positions)),
        'target_position_max':int(np.max(positions)),
        'target_position_mean':float(np.mean(positions)),
        'maskgit_training_used':False,
        'future_ground_truth_visible':False,
        'class_labels_used':False,
    }


def build_shared_schedule(n_images,cfg,events,seed):
    rng=np.random.default_rng(seed)
    out=[];h=hashlib.sha256()
    for _ in range(int(events)):
        ii=int(rng.integers(n_images))
        t=int(rng.integers(1,cfg.N))
        event_seed=int(rng.integers(0,2**63-1,dtype=np.int64))
        out.append((ii,t,event_seed))
        h.update(np.asarray([ii,t,event_seed],dtype=np.int64).tobytes())
    return out,h.hexdigest()


def snapshot(system,model0):
    return {
        'WR_norm':float(np.linalg.norm(system.WR)),
        'BR_mean_abs':float(np.mean(np.abs(system.BR))),
        'BO_mean_abs':float(np.mean(np.abs(system.BO))),
        'B2_mean_abs':float(np.mean(np.abs(system.B2))),
        'internal_max_rel_from_init':float(max(
            np.linalg.norm(np.asarray(getattr(system,k))-model0[k])/(np.linalg.norm(model0[k])+1e-12)
            for k in INTERNAL)),
    }


def train_branch(name,initial_model,model0,cfg,tokens,tok,pos,weights,schedule,report_every,system_seed):
    system=MaskGITP3System(cfg,initial_model,system_seed)
    history=[];done=0;next_report=int(report_every);t0=time.time()
    first={'WR':None,'BR':None,'internal':None};tol=1e-12
    onepos=np.empty(1,dtype=np.int64)

    for ii,t,event_seed in schedule:
        truth=np.asarray(tokens[int(ii)],dtype=np.int64)
        xn,xraw=causal_context(truth,int(t),cfg,tok,pos)
        onepos[0]=int(t)
        erng=np.random.default_rng(int(event_seed))
        rec=system.learn_mask_batch(xn,xraw,onepos,truth,weights,erng)
        if rec['masked_targets'] != 1:
            raise AssertionError('next-token experiment must have exactly one target per update')
        done+=1

        ss=snapshot(system,model0)
        if first['WR'] is None and ss['WR_norm']>tol:first['WR']=done
        if first['BR'] is None and ss['BR_mean_abs']>tol:first['BR']=done
        if first['internal'] is None and ss['internal_max_rel_from_init']>tol:first['internal']=done

        if done>=next_report or done==len(schedule):
            row={
                'branch':name,
                'target_events':int(done),
                'online_nll':float(system.obs_nll/max(1,system.obs_count)),
                'updates':int(system.update_count),
                'elapsed_s':float(time.time()-t0),
                **ss,
            }
            history.append(row)
            print('S2G_PROGRESS',json.dumps(row),flush=True)
            while next_report<=done:next_report+=int(report_every)

    final=clone_model(initial_model);final.update(system.weights())
    return system,final,history,first


def eval_nexttoken(model,cfg,tokens,tok,pos,seed,npairs=2048):
    rng=np.random.default_rng(seed)
    system=core.P3ImageSystem(cfg,model,seed+30000)
    ce=[];acc=[]
    for _ in range(int(npairs)):
        ii=int(rng.integers(len(tokens)))
        t=int(rng.integers(1,cfg.N))
        truth=np.asarray(tokens[ii],dtype=np.int64)
        xn,xraw=causal_context(truth,t,cfg,tok,pos)
        p=system.forward_only(xn,xraw[t],t,new_image=True)
        z=int(truth[t])
        ce.append(-math.log(max(float(p[z]),1e-12)))
        acc.append(int(np.argmax(p)==z))
    return {'ce':float(np.mean(ce)),'acc':float(np.mean(acc)),'pairs':int(npairs),'class_labels_used':False}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--D',type=int,default=96);ap.add_argument('--heads',type=int,default=8)
    ap.add_argument('--features',type=int,default=24);ap.add_argument('--ff',type=int,default=192);ap.add_argument('--R',type=int,default=7)
    ap.add_argument('--scaffold-contexts',type=int,default=2000);ap.add_argument('--scaffold-paths',type=int,default=2500);ap.add_argument('--scaffold-lr',type=float,default=.6)
    ap.add_argument('--events',type=int,default=10000);ap.add_argument('--eval-pairs',type=int,default=2048)
    ap.add_argument('--internal-lr-scale',type=float,default=.5);ap.add_argument('--encode-batch',type=int,default=256)
    ap.add_argument('--seed',type=int,default=7311);ap.add_argument('--report-every',type=int,default=2000);ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args()

    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True,exist_ok=True);device=torch.device('cpu')

    # Historical 0/1/2 subset selection is external to the neural task. Labels are
    # never passed to context construction, learning, evaluation, or the scaffold.
    xall,yall,xte0,yte0=core.load(a.data)
    x012,y012=core.subset012(xall,yall);xte,yte=core.subset012(xte0,yte0)
    xtr,ytr,xva,yva=core.split(x012,y012,.1,7)

    tokenizer,tok_meta=base.load_frozen_tokenizer(a.tokenizer,device)
    tr,sp_tr=base.encode_snn(tokenizer,xtr,device,a.encode_batch)
    te,_=base.encode_snn(tokenizer,xte,device,a.encode_batch)

    cfg=core.Config(D=a.D,H=a.heads,M=a.features,FF=a.ff,R=a.R)
    base_model=core.init_model(cfg,a.seed)
    for k in ('lrq','lrk','lrv','lro','lr1','lr2'):base_model[k]*=a.internal_lr_scale
    if np.any(base_model['WR']!=0) or np.any(base_model['BR']!=0):
        raise AssertionError('S2G base model must start from exact WR=BR=0')

    tok,pos,input_audit=build_input_embeddings(tokenizer,cfg.D,grid=7)
    ones=np.ones(cfg.V,np.float64)

    # A: same kind of WR-only warm start, but learned exclusively from causal
    # next-token contexts rather than MaskGIT multi-mask contexts.
    scaffold_model=clone_model(base_model)
    F,T,scaffold_audit=build_nexttoken_scaffold(
        scaffold_model,cfg,tr,tok,pos,a.scaffold_contexts,a.seed+101)
    ts=time.time()
    scaffold_model['WR']=core.train_readout_msdl(F,T,ones,a.scaffold_paths,a.scaffold_lr,a.seed+202)
    scaffold_runtime=float(time.time()-ts)
    scaffold_wr0=float(np.linalg.norm(scaffold_model['WR']))

    # B: exact native start.
    native_model=clone_model(base_model)
    if np.any(native_model['WR']!=0):raise AssertionError('native branch lost exact WR=0')

    schedule,schedule_hash=build_shared_schedule(len(tr),cfg,a.events,a.seed+404)
    schedule_summary={
        'task':'causal next-token prediction',
        'target_events':int(a.events),
        'sha256':schedule_hash,
        'same_schedule_for_both_branches':True,
        'one_target_per_update':True,
        'maskgit_training_used':False,
    }
    print('S2G_SCHEDULE',json.dumps(schedule_summary),flush=True)

    sys_a,final_a,hist_a,first_a=train_branch(
        'scaffold',scaffold_model,base_model,cfg,tr,tok,pos,ones,schedule,a.report_every,a.seed+303)
    sys_b,final_b,hist_b,first_b=train_branch(
        'native_no_scaffold',native_model,base_model,cfg,tr,tok,pos,ones,schedule,a.report_every,a.seed+303)

    eva=eval_nexttoken(final_a,cfg,te,tok,pos,a.seed+505,a.eval_pairs)
    evb=eval_nexttoken(final_b,cfg,te,tok,pos,a.seed+505,a.eval_pairs)

    last_a=hist_a[-1];last_b=hist_b[-1]
    nll_gap=float(last_b['online_nll']-last_a['online_nll'])
    br_ratio=float(last_a['BR_mean_abs']/max(last_b['BR_mean_abs'],1e-15))
    wr_ratio=float(last_a['WR_norm']/max(last_b['WR_norm'],1e-15))
    gates={
        'labels_enter_neural_path':False,
        'maskgit_training_used':False,
        'same_online_schedule':True,
        'native_started_WR_exact_zero':True,
        'native_bootstrap_WR':bool(first_b['WR'] is not None),
        'native_bootstrap_BR':bool(first_b['BR'] is not None and first_b['WR'] is not None and first_b['BR']>first_b['WR']),
        'native_internal_plasticity':bool(first_b['internal'] is not None),
        'nll_advantage_scaffold_gt_0p5':bool(nll_gap>0.5),
        'BR_ratio_scaffold_over_native_gt_5':bool(br_ratio>5.0),
    }
    generic=bool(gates['nll_advantage_scaffold_gt_0p5'] and gates['BR_ratio_scaffold_over_native_gt_5'])

    result={
        'stage':'R6B.3 S2G minimal non-MaskGIT scaffold generality test',
        'scope':'single-seed causal next-token diagnostic only; no MaskGIT training or generation',
        'seed':int(a.seed),
        'task':{
            'definition':'z[:t] -> z[t], with z[t:] represented as MASK in fixed 49-position P3 layout',
            'target_positions':'uniform integers 1..48',
            'future_ground_truth_visible':False,
            'maskgit_multi_mask':False,
            'maskgit_cosine_schedule':False,
            'maskgit_confidence_remasking':False,
            'generation_run':False,
        },
        'tokenizer_metadata':tok_meta,
        'tokenizer_activity_train':sp_tr,
        'input_representation':input_audit,
        'shared_schedule':schedule_summary,
        'predeclared_rule':{
            'nll_gap_native_minus_scaffold_gt':0.5,
            'BR_ratio_scaffold_over_native_gt':5.0,
            'interpretation':'if both pass, scaffold acceleration is supported as a generic P3/MSDL readout-bootstrap effect rather than a MaskGIT-only effect',
        },
        'scaffold':{
            'used':True,'target_rows':int(a.scaffold_contexts),'paths':int(a.scaffold_paths),'lr':float(a.scaffold_lr),
            'runtime_s':scaffold_runtime,'WR_norm_after_scaffold':scaffold_wr0,'audit':scaffold_audit,
            'first_movement_event_during_online':first_a,'online_history':hist_a,'final_eval':eva,
            'mean_abs_reciprocal':{'BR':float(np.mean(np.abs(sys_a.BR))),'BO':float(np.mean(np.abs(sys_a.BO))),'B2':float(np.mean(np.abs(sys_a.B2)))},
        },
        'native_no_scaffold':{
            'used':False,'WR_norm_at_start':0.0,'first_movement_event':first_b,'online_history':hist_b,'final_eval':evb,
            'mean_abs_reciprocal':{'BR':float(np.mean(np.abs(sys_b.BR))),'BO':float(np.mean(np.abs(sys_b.BO))),'B2':float(np.mean(np.abs(sys_b.B2)))},
        },
        'paired_final':{
            'online_nll_native_minus_scaffold':nll_gap,
            'BR_ratio_scaffold_over_native':br_ratio,
            'WR_ratio_scaffold_over_native':wr_ratio,
            'eval_ce_native_minus_scaffold':float(evb['ce']-eva['ce']),
            'eval_acc_scaffold_minus_native':float(eva['acc']-evb['acc']),
        },
        'gates':gates,
        'generic_bootstrap_support':generic,
        'scientific_boundary':'This minimal single-seed task-level control is intended only to decide whether to stop investigating the scaffold and return to the MaskGIT/continual-learning main line.',
    }

    (a.out/'s2g_nexttoken_generality.json').write_text(json.dumps(result,indent=2))
    with (a.out/'s2g_history.csv').open('w',newline='') as f:
        rows=hist_a+hist_b;w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    print('S2G_FINAL',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
