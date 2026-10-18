#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R6B.3 S2 — controlled startup-scaffold ablation.

Compare two branches under an explicitly shared training schedule:

  A: label-free MaskGIT WR startup scaffold + online P3/MSDL
  B: no scaffold, exact native WR=0 start + online P3/MSDL

Everything else is held fixed: frozen SNN-VQ tokenizer, structured token/position
representation, initial internal P3 weights, model seed, MaskGIT corrupted-image
sequence, masked positions, masked-target processing order, target-event budget,
evaluation masks, and MaskGIT generation controller/sampling seed.

This is an ablation of the *startup scaffold only*.  Labels never enter either
neural branch; they are retained solely for the historical 0/1/2 data subset and
an external post-hoc image classifier.
"""
from __future__ import annotations

import argparse,csv,hashlib,json,math,random,sys,time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'snn_transform_p3_image'))
sys.path.insert(0,str(HERE))

import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
from maskgit_input_representation import build_input_embeddings,context_from_arr_unconditional
from maskgit_msdl_training import MaskGITP3System,sample_mask_set
from maskgit_msdl_training_unconditional import (
    build_maskgit_scaffold_contexts_unconditional,
    eval_masked_unconditional,
)
from r6b3_snn_encoder_maskgit_msdl import (
    generation_tokens_maskgit,
    posthoc_generation_metric,
    save_grid_unlabeled,
)

INTERNAL=('WQ','WK','WV','WO','W1','W2')


def clone_model(model):
    return {k:(np.array(v,copy=True) if isinstance(v,np.ndarray) else v) for k,v in model.items()}


def token_stats(arr,vocab):
    a=np.asarray(arr,dtype=np.int64).reshape(-1)
    counts=np.bincount(a,minlength=vocab).astype(np.int64)
    q=counts/counts.sum()
    nz=q[q>0]
    return {
        'active_codes':int(np.sum(counts>0)),
        'perplexity':float(np.exp(-(nz*np.log(nz)).sum())),
        'top1_fraction':float(counts.max()/counts.sum()),
        'counts':counts.tolist(),
    }


def build_shared_schedule(n_images,cfg,target_events,seed):
    """Precompute image/mask/order seeds so both branches see identical training dataflow."""
    rng=np.random.default_rng(seed)
    schedule=[];done=0
    h=hashlib.sha256()
    while done<int(target_events):
        ii=int(rng.integers(n_images))
        sm=sample_mask_set(cfg.N,rng)
        order_seed=int(rng.integers(0,2**63-1,dtype=np.int64))
        pos=np.asarray(sm.masked_positions,dtype=np.int64).copy()
        rec={
            'image_index':ii,
            'ratio_r':float(sm.ratio_r),
            'mask_ratio':float(sm.mask_ratio),
            'masked_positions':pos,
            'order_seed':order_seed,
        }
        schedule.append(rec);done+=len(pos)
        h.update(np.asarray([ii,order_seed,len(pos)],dtype=np.int64).tobytes())
        h.update(np.asarray([sm.ratio_r,sm.mask_ratio],dtype=np.float64).tobytes())
        h.update(pos.tobytes())
    return schedule,int(done),h.hexdigest()


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


def train_on_schedule(name,initial_model,model0,cfg,tr,tok,pos,weights,schedule,
                      report_every,system_seed):
    system=MaskGITP3System(cfg,initial_model,system_seed)
    history=[];done=0;next_report=int(report_every);t0=time.time()
    first={'WR':None,'BR':None,'internal':None}
    tol=1e-12

    for bi,item in enumerate(schedule,1):
        ii=int(item['image_index'])
        masked=np.asarray(item['masked_positions'],dtype=np.int64)
        truth=np.asarray(tr[ii],dtype=np.int64)
        arr=truth.copy();arr[masked]=core.MASK
        xn,xraw=context_from_arr_unconditional(arr,tok,pos)
        batch_rng=np.random.default_rng(int(item['order_seed']))
        system.learn_mask_batch(xn,xraw,masked,truth,weights,batch_rng)
        done+=len(masked)

        ss=snapshot(system,model0)
        if first['WR'] is None and ss['WR_norm']>tol:first['WR']=bi
        if first['BR'] is None and ss['BR_mean_abs']>tol:first['BR']=bi
        if first['internal'] is None and ss['internal_max_rel_from_init']>tol:first['internal']=bi

        if done>=next_report or bi==len(schedule):
            row={
                'branch':name,
                'target_events':int(done),
                'mask_batches':int(bi),
                'online_nll':float(system.obs_nll/max(1,system.obs_count)),
                'updates':int(system.update_count),
                'elapsed_s':float(time.time()-t0),
                **ss,
            }
            history.append(row)
            print('S2_PROGRESS',json.dumps(row),flush=True)
            while next_report<=done:next_report+=int(report_every)

    final=clone_model(initial_model);final.update(system.weights())
    return system,final,history,first


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--D',type=int,default=96);ap.add_argument('--heads',type=int,default=8)
    ap.add_argument('--features',type=int,default=24);ap.add_argument('--ff',type=int,default=192);ap.add_argument('--R',type=int,default=7)
    ap.add_argument('--scaffold-contexts',type=int,default=2000);ap.add_argument('--scaffold-paths',type=int,default=2500);ap.add_argument('--scaffold-lr',type=float,default=.6)
    ap.add_argument('--events',type=int,default=30000)
    ap.add_argument('--eval-images',type=int,default=128);ap.add_argument('--gen-samples',type=int,default=36)
    ap.add_argument('--internal-lr-scale',type=float,default=.5);ap.add_argument('--encode-batch',type=int,default=256)
    ap.add_argument('--decode-iterations',type=int,default=8);ap.add_argument('--choice-temperature',type=float,default=4.5)
    ap.add_argument('--seed',type=int,default=7311);ap.add_argument('--report-every',type=int,default=5000);ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args()

    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True,exist_ok=True);device=torch.device('cpu')

    # Labels remain outside both neural branches.
    xall,yall,xte0,yte0=core.load(a.data)
    x012,y012=core.subset012(xall,yall);xte,yte=core.subset012(xte0,yte0)
    xtr,ytr,xva,yva=core.split(x012,y012,.1,7)

    tokenizer,tok_meta=base.load_frozen_tokenizer(a.tokenizer,device)
    tr,sp_tr=base.encode_snn(tokenizer,xtr,device,a.encode_batch)
    va,_=base.encode_snn(tokenizer,xva,device,a.encode_batch)
    te,_=base.encode_snn(tokenizer,xte,device,a.encode_batch)

    cfg=core.Config(D=a.D,H=a.heads,M=a.features,FF=a.ff,R=a.R)
    base_model=core.init_model(cfg,a.seed)
    for k in ('lrq','lrk','lrv','lro','lr1','lr2'):base_model[k]*=a.internal_lr_scale
    if np.any(base_model['WR']!=0) or np.any(base_model['BR']!=0):
        raise AssertionError('S2 base model must start from WR=BR=0')

    tok,pos,input_audit=build_input_embeddings(tokenizer,cfg.D,grid=7)
    ones=np.ones(cfg.V,np.float64)

    # A: only WR receives the historical startup scaffold. Contexts are label-free MaskGIT masks.
    scaffold_model=clone_model(base_model)
    F,T,scaffold_mask_audit=build_maskgit_scaffold_contexts_unconditional(
        scaffold_model,cfg,tr,tok,pos,a.scaffold_contexts,a.seed+101)
    ts=time.time()
    scaffold_model['WR']=core.train_readout_msdl(F,T,ones,a.scaffold_paths,a.scaffold_lr,a.seed+202)
    scaffold_runtime=float(time.time()-ts)
    scaffold_wr_norm=float(np.linalg.norm(scaffold_model['WR']))

    # B: no startup intervention at all.
    native_model=clone_model(base_model)
    if np.any(native_model['WR']!=0):raise AssertionError('native branch lost exact WR=0')

    # Hard paired schedule: same images, mask ratios/positions and masked-target order seeds.
    schedule,actual_events,schedule_hash=build_shared_schedule(len(tr),cfg,a.events,a.seed+404)
    schedule_summary={
        'requested_target_events':int(a.events),
        'actual_target_events':actual_events,
        'mask_batches':len(schedule),
        'sha256':schedule_hash,
        'same_schedule_for_both_branches':True,
        'same_masked_target_order_seeds':True,
    }
    print('S2_SCHEDULE',json.dumps(schedule_summary),flush=True)

    # Same system seed intentionally resets reciprocal/random-feature dynamics for each branch.
    sys_a,final_a,hist_a,first_a=train_on_schedule(
        'scaffold',scaffold_model,base_model,cfg,tr,tok,pos,ones,schedule,a.report_every,a.seed+303)
    sys_b,final_b,hist_b,first_b=train_on_schedule(
        'native_no_scaffold',native_model,base_model,cfg,tr,tok,pos,ones,schedule,a.report_every,a.seed+303)

    ratios=(.25,.5,.9,1.0)
    eval_a={str(r):eval_masked_unconditional(final_a,cfg,te,tok,pos,r,a.seed+500+int(100*r),a.eval_images) for r in ratios}
    eval_b={str(r):eval_masked_unconditional(final_b,cfg,te,tok,pos,r,a.seed+500+int(100*r),a.eval_images) for r in ratios}

    # One external post-hoc classifier for both branches; it never enters either generator.
    posthoc_clf=LogisticRegression(max_iter=250,solver='lbfgs',random_state=7).fit(xtr.reshape(len(xtr),-1),ytr)
    gtok_a,gtrace_a,gaudit_a=generation_tokens_maskgit(final_a,cfg,a.gen_samples,tok,pos,a.seed+606,a.decode_iterations,a.choice_temperature)
    gimg_a=base.decode_snn(tokenizer,gtok_a,device);save_grid_unlabeled(gimg_a,a.out/'generation_scaffold_final.png')
    gtok_b,gtrace_b,gaudit_b=generation_tokens_maskgit(final_b,cfg,a.gen_samples,tok,pos,a.seed+606,a.decode_iterations,a.choice_temperature)
    gimg_b=base.decode_snn(tokenizer,gtok_b,device);save_grid_unlabeled(gimg_b,a.out/'generation_native_final.png')
    gen_a={**posthoc_generation_metric(posthoc_clf,gimg_a),'token_stats':token_stats(gtok_a,cfg.V)}
    gen_b={**posthoc_generation_metric(posthoc_clf,gimg_b),'token_stats':token_stats(gtok_b,cfg.V)}

    e50a=eval_a['0.5'];e50b=eval_b['0.5']
    # Predeclared comparability bands for this paired S2 seed. They support, but do not by
    # themselves justify, permanent scaffold deletion across seeds/tasks.
    gates={
        'labels_enter_neural_path':False,
        'shared_training_schedule':True,
        'native_started_WR_exact_zero':True,
        'native_bootstrap_WR':bool(first_b['WR'] is not None),
        'native_bootstrap_BR':bool(first_b['BR'] is not None and first_b['WR'] is not None and first_b['BR']>first_b['WR']),
        'native_internal_plasticity':bool(first_b['internal'] is not None),
        'native_final_ce50_within_plus_0p10':bool(e50b['ce']<=e50a['ce']+0.10),
        'native_final_acc50_within_minus_0p05':bool(e50b['acc']>=e50a['acc']-0.05),
        'native_final_ce50_below_uniform':bool(e50b['ce']<math.log(cfg.V)),
        'all_generation_traces_maskgit_valid':bool(all(x['pass'] for x in gaudit_a+gaudit_b)),
    }
    paired_comparable=bool(gates['native_final_ce50_within_plus_0p10'] and gates['native_final_acc50_within_minus_0p05'])

    result={
        'stage':'R6B.3 S2 controlled startup-scaffold ablation',
        'scope':'paired single-seed scaffold-vs-native comparison; no class conditioning',
        'seed':a.seed,
        'tokenizer_metadata':tok_meta,
        'tokenizer_activity_train':sp_tr,
        'input_representation':input_audit,
        'shared_schedule':schedule_summary,
        'scaffold':{
            'used':True,
            'target_rows':int(a.scaffold_contexts),
            'paths':int(a.scaffold_paths),
            'lr':float(a.scaffold_lr),
            'runtime_s':scaffold_runtime,
            'WR_norm_after_scaffold':scaffold_wr_norm,
            'masking_audit':scaffold_mask_audit,
            'first_movement_batch_during_online':first_a,
            'online_history':hist_a,
            'final_eval':eval_a,
            'generation':gen_a,
            'mean_abs_reciprocal':{
                'BR':float(np.mean(np.abs(sys_a.BR))),'BO':float(np.mean(np.abs(sys_a.BO))),'B2':float(np.mean(np.abs(sys_a.B2))),
            },
        },
        'native_no_scaffold':{
            'used':False,
            'WR_norm_at_start':0.0,
            'first_movement_batch':first_b,
            'online_history':hist_b,
            'final_eval':eval_b,
            'generation':gen_b,
            'mean_abs_reciprocal':{
                'BR':float(np.mean(np.abs(sys_b.BR))),'BO':float(np.mean(np.abs(sys_b.BO))),'B2':float(np.mean(np.abs(sys_b.B2))),
            },
        },
        'paired_difference_native_minus_scaffold':{
            'ce25':float(eval_b['0.25']['ce']-eval_a['0.25']['ce']),
            'ce50':float(e50b['ce']-e50a['ce']),
            'ce90':float(eval_b['0.9']['ce']-eval_a['0.9']['ce']),
            'ce100':float(eval_b['1.0']['ce']-eval_a['1.0']['ce']),
            'acc25':float(eval_b['0.25']['acc']-eval_a['0.25']['acc']),
            'acc50':float(e50b['acc']-e50a['acc']),
            'acc90':float(eval_b['0.9']['acc']-eval_a['0.9']['acc']),
            'acc100':float(eval_b['1.0']['acc']-eval_a['1.0']['acc']),
        },
        'gates':gates,
        'paired_comparable_on_primary_50pct_gate':paired_comparable,
        'scientific_boundary':'A paired single-seed S2 can show whether scaffold materially helps this configuration; permanent removal should require interpreting the full curve/generation and preferably seed replication.',
    }

    (a.out/'s2_scaffold_ablation.json').write_text(json.dumps(result,indent=2))
    (a.out/'maskgit_trace_scaffold_final.json').write_text(json.dumps(gtrace_a,indent=2))
    (a.out/'maskgit_trace_native_final.json').write_text(json.dumps(gtrace_b,indent=2))
    np.savetxt(a.out/'generated_tokens_scaffold_final.csv',gtok_a,fmt='%d',delimiter=',')
    np.savetxt(a.out/'generated_tokens_native_final.csv',gtok_b,fmt='%d',delimiter=',')
    np.savez_compressed(a.out/'final_scaffold_state.npz',**{k:v for k,v in final_a.items() if isinstance(v,np.ndarray)})
    np.savez_compressed(a.out/'final_native_state.npz',**{k:v for k,v in final_b.items() if isinstance(v,np.ndarray)})
    rows=hist_a+hist_b
    if rows:
        with (a.out/'s2_history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    print('S2_FINAL',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
