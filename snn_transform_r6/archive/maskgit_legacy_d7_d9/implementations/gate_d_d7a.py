#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D7A — causal upper bound for preventing W2 re-adaptation.

Three synchronized trajectories from the exact D6 start state:
  F = F_EM safe/shadow trajectory
  B = D6A combined failure baseline (F_EM + WO-zero + exact bo oracle)
  C = B with W2 clamped after every context to the time-matched F trajectory

D7A is a privileged trajectory-level causal upper bound, not a candidate final
local/dynamical learning rule.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch

import gate_d_d3e_a0 as d3a0
import gate_d_d3e_runtime as d3r
import gate_d_d6a as d6a
import gate_d_d6a1 as d6a1
import gate_d_plasticity_audit as gd
import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
import r6b3_context_epoch_training as trainer
import r6b3_decoder_operating_point_audit as coarse
import r6b3_decoder_spatial_refinement_audit as spatial
from gate_d_d3e_eval import BASE_GEN_SEED, eval_masked, rollout_js
from maskgit_input_representation import build_input_embeddings
from r6b4_full_forward_plasticity import upgrade_r6b3_system

CHECKPOINTS=(64,128,256,512,768,1024)
F_LABEL='F_EM'
B_LABEL='F_EM_WO_zero_bo_oracle'
C_LABEL='B_W2_safe_clamp'
PRIMARY=('F61','D_non61')
ALL_METRICS=('F61','D_non61','C61','delta_P61','CE')
MASK_RATIOS=('0.25','0.5','0.9','1.0')
JS_KEYS=('global_js_nats','spatial_js_nats','pair_js_nats')
TOL_GATE=1e-7
TOL_MODEL=1e-8
EPS=1e-30


def clone(x):
    return pickle.loads(pickle.dumps(x,pickle.HIGHEST_PROTOCOL))


def locate(root:Path,name:str)->Path:
    return d3r.locate(root,name)


def write_csv(path:Path,rows):
    if not rows:
        path.write_text('',encoding='utf-8'); return
    keys=[]; seen=set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore'); w.writeheader(); w.writerows(rows)


def max_abs(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    if a.shape!=b.shape: return float('inf')
    if not np.array_equal(np.isnan(a),np.isnan(b)): return float('inf')
    if not np.array_equal(np.isposinf(a),np.isposinf(b)): return float('inf')
    if not np.array_equal(np.isneginf(a),np.isneginf(b)): return float('inf')
    good=np.isfinite(a)&np.isfinite(b)
    return 0.0 if not np.any(good) else float(np.max(np.abs(a[good]-b[good])))


def model_field_compare_tolerant(a,b,tol=TOL_MODEL):
    keys=list(d3a0.MODEL_KEYS)
    for k in ('tok','pos','aux_lr_scale'):
        if hasattr(a,k) or hasattr(b,k): keys.append(k)
    out={'pass':True,'tolerance':tol,'fields':{}}
    for k in keys:
        if not hasattr(a,k) or not hasattr(b,k):
            out['fields'][k]={'pass':False,'kind':'missing'}; out['pass']=False; continue
        x=getattr(a,k); y=getattr(b,k)
        if isinstance(x,np.ndarray) or isinstance(y,np.ndarray):
            shape_dtype=bool(isinstance(x,np.ndarray) and isinstance(y,np.ndarray)
                             and x.dtype==y.dtype and x.shape==y.shape)
            err=max_abs(x,y) if shape_dtype else float('inf')
            ok=bool(shape_dtype and err<tol)
            out['fields'][k]={'pass':ok,'shape_dtype_exact':shape_dtype,'max_abs':err}
        else:
            if isinstance(x,(float,np.floating)) or isinstance(y,(float,np.floating)):
                err=abs(float(x)-float(y)); ok=bool(err<tol)
                out['fields'][k]={'pass':ok,'max_abs':err}
            else:
                ok=bool(x==y); out['fields'][k]={'pass':ok,'a':repr(x),'b':repr(y)}
        out['pass']=out['pass'] and out['fields'][k]['pass']
    return out


def gate_calibration(ev,stored,prefix,tol=TOL_GATE):
    d={'self_metrics_max_abs':d6a.nanaware_max_abs(ev['self_metrics'],stored[f'{prefix}__self_metrics']),
       'teacher_metrics_max_abs':d6a.nanaware_max_abs(ev['teacher_metrics'],stored[f'{prefix}__teacher_metrics']),
       'self_proposal_exact':bool(np.array_equal(ev['self_proposal'],stored[f'{prefix}__self_proposal'])),
       'teacher_proposal_exact':bool(np.array_equal(ev['teacher_proposal'],stored[f'{prefix}__teacher_proposal'])),
       'effects':{}}
    for m in ALL_METRICS:
        d['effects'][m]=d6a.nanaware_max_abs(ev['effects'][m],stored[f'{prefix}__effect__{m}'])
    d['pass']=bool(d['self_metrics_max_abs']<tol and d['teacher_metrics_max_abs']<tol
                   and d['self_proposal_exact'] and d['teacher_proposal_exact']
                   and max(d['effects'].values())<tol)
    return d


def contrast(ev1,ev0,m,seed,nboot):
    x=np.asarray(ev1['effects'][m],dtype=np.float64)-np.asarray(ev0['effects'][m],dtype=np.float64)
    d=d6a.bootstrap_mean(x,seed,nboot)
    lo,hi=d['ci95']
    d['ci_entirely_negative']=bool(hi is not None and hi<0)
    d['ci_entirely_positive']=bool(lo is not None and lo>0)
    return d


def mean_effect(ev,m):
    return float(np.nanmean(np.asarray(ev['effects'][m],dtype=np.float64)))


def stream_equal(a,b):
    return bool(a==b)


def run_triplet(start,tokens,indices,rng_state):
    f=d6a.make_d6a(start,F_LABEL)
    b=d6a.make_d6a(start,B_LABEL)
    c=d6a.make_d6a(start,B_LABEL)
    rf=gd.clone_rng(rng_state); rb=gd.clone_rng(rng_state); rc=gd.clone_rng(rng_state)
    streamF=[]; streamB=[]; streamC=[]; trace=[]
    max_post_abs=0.0; cumulative_correction=0.0

    for j,image_index in enumerate(indices,start=1):
        # At context entry C must carry the previous time-matched safe W2 exactly.
        start_err=max_abs(c.W2,f.W2)
        if start_err!=0.0:
            raise RuntimeError({'D7A_precontext_W2_clamp_not_exact':{'context':j,'max_abs':start_err}})

        for s,r in ((f,rf),(b,rb),(c,rc)):
            before=d3r.jstate(r); d3r.reset_boundary(s,'F_EM')
            if d3r.jstate(r)!=before: raise RuntimeError('F_EM reset consumed RNG')

        truth=np.asarray(tokens[int(image_index)],dtype=np.int64)
        recF,evF=d6a.run_context(f,j,int(image_index),truth,rf)
        recB,evB=d6a.run_context(b,j,int(image_index),truth,rb)
        recC,evC=d6a.run_context(c,j,int(image_index),truth,rc)

        rowF=d6a.stream_row(j,int(image_index),evF)
        rowB=d6a.stream_row(j,int(image_index),evB)
        rowC=d6a.stream_row(j,int(image_index),evC)
        if not(stream_equal(rowF,rowB) and stream_equal(rowF,rowC)):
            raise RuntimeError({'D7A_stream_divergence':{'context':j,'F':rowF,'B':rowB,'C':rowC}})
        if not(d3r.jstate(rf)==d3r.jstate(rb)==d3r.jstate(rc)):
            raise RuntimeError({'D7A_external_rng_divergence':{'context':j}})
        streamF.append(rowF); streamB.append(rowB); streamC.append(rowC)

        # Privileged causal intervention: discard C's just-computed W2 displacement and
        # put it exactly on the time-matched safe F trajectory. All other C state remains.
        pre=np.asarray(c.W2,dtype=np.float64).copy()
        safe=np.asarray(f.W2,dtype=np.float64)
        correction=safe-pre
        correction_norm=float(np.linalg.norm(correction))
        cumulative_correction+=correction_norm
        pre_to_safe=correction_norm
        c.W2=np.array(f.W2,copy=True)
        post_err=max_abs(c.W2,f.W2)
        max_post_abs=max(max_post_abs,post_err)
        if post_err!=0.0:
            raise RuntimeError({'D7A_postcontext_W2_clamp_not_exact':{'context':j,'max_abs':post_err}})

        crow=c._d5b_rows[-1]
        crow['d7a_preclamp_W2_to_safe_norm']=pre_to_safe
        crow['d7a_postclamp_W2_max_abs_error']=post_err
        crow['d7a_cumulative_W2_clamp_correction']=cumulative_correction

        if j in CHECKPOINTS:
            tr={'context':int(j),
                'F_mean_nll':float(recF['mean_nll']),
                'B_mean_nll':float(recB['mean_nll']),
                'C_mean_nll':float(recC['mean_nll']),
                'B_minus_F_W2_norm':float(np.linalg.norm(np.asarray(b.W2,dtype=np.float64)-np.asarray(f.W2,dtype=np.float64))),
                'C_preclamp_minus_F_W2_norm':pre_to_safe,
                'C_postclamp_max_abs_error':post_err,
                'cumulative_clamp_correction':cumulative_correction}
            trace.append(tr)
            print('D7A_PROGRESS',json.dumps(tr),flush=True)

    return {'F':f,'B':b,'C':c,
            'streams':{'F':streamF,'B':streamB,'C':streamC},
            'rng_after':{'F':d3r.jstate(rf),'B':d3r.jstate(rb),'C':d3r.jstate(rc)},
            'trace':trace,'max_postclamp_W2_abs_error':max_post_abs,
            'cumulative_clamp_correction':cumulative_correction}


def guardrails(masked,rollout):
    mg={}; mg_pass=True
    for r in MASK_RATIOS:
        best=min(float(masked['F'][r]['ce']),float(masked['B'][r]['ce']))
        val=float(masked['C'][r]['ce']); lim=best*1.01+1e-8
        mg[r]={'C_ce':val,'best_F_or_B_ce':best,'limit':lim,'pass':bool(val<=lim)}
        mg_pass=mg_pass and mg[r]['pass']
    jg={}; jg_pass=True
    for k in JS_KEYS:
        best=min(float(rollout['F'][k]),float(rollout['B'][k]))
        val=float(rollout['C'][k]); lim=best*1.02+1e-6
        jg[k]={'C':val,'best_F_or_B':best,'limit':lim,'pass':bool(val<=lim)}
        jg_pass=jg_pass and jg[k]['pass']
    return {'masked_ce':mg,'masked_ce_all_pass':bool(mg_pass),
            'rollout_js':jg,'rollout_js_all_pass':bool(jg_pass)}


def save_terminal(path,s,label):
    with path.open('wb') as f:
        pickle.dump({'arm':label,'system':s},f,pickle.HIGHEST_PROTOCOL)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--d6a-artifact',type=Path,required=True)
    ap.add_argument('--d6a1-artifact',type=Path,required=True)
    ap.add_argument('--d6a2-artifact',type=Path,required=True)
    ap.add_argument('--d6a3-artifact',type=Path,required=True)
    ap.add_argument('--anchor-artifact',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--contexts',type=int,default=1024)
    ap.add_argument('--eval-images',type=int,default=256)
    ap.add_argument('--gen-samples',type=int,default=256)
    ap.add_argument('--encode-batch',type=int,default=256)
    ap.add_argument('--bootstrap',type=int,default=10000)
    ap.add_argument('--seed',type=int,default=7311)
    ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    if a.contexts!=1024 or a.eval_images!=256:
        raise ValueError('formal D7A requires 1024 contexts and exact 256-image Gate-A evaluation')

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads)
    device=torch.device('cpu')

    # Frozen D6 closure prerequisites.
    s6=json.loads(locate(a.d6a_artifact,'D6A_SUMMARY.json').read_text(encoding='utf-8'))
    s61=json.loads(locate(a.d6a1_artifact,'D6A1_SUMMARY.json').read_text(encoding='utf-8'))
    s62=json.loads(locate(a.d6a2_artifact,'D6A2_SUMMARY.json').read_text(encoding='utf-8'))
    s63=json.loads(locate(a.d6a3_artifact,'D6A3_SUMMARY.json').read_text(encoding='utf-8'))
    if s6.get('classification')!='R0_NO_REPAIR_SUFFICIENCY' or not s6.get('decision',{}).get('oracle_valid',False):
        raise RuntimeError('D6A prerequisite not frozen R0 with valid oracle')
    if s61.get('classification')!='I3_NO_INTRINSIC_BO_REPAIR_DEGRADATION':
        raise RuntimeError('D6A.1 prerequisite not frozen I3')
    if s62.get('classification')!='S1_DOMINANT_SINGLE_SUPPRESSOR':
        raise RuntimeError('D6A.2 prerequisite not frozen S1')
    if s63.get('classification')!='P4_MIXED_BACKGROUND_DEPENDENT':
        raise RuntimeError('D6A.3 prerequisite not frozen P4')

    stored=np.load(locate(a.d6a_artifact,'D6A_EFFECT_ARRAYS.npz'),allow_pickle=False)
    schedules=json.loads(locate(a.anchor_artifact,'HISTORICAL_R6B3_R3_T8_REFERENCE_TRACE.json').read_text(encoding='utf-8'))

    # Exact 2.0-epoch start.
    checkpoint_sha=d3r.sha_file(a.checkpoint)
    source,meta,train_rng,schedule=trainer.load_full_checkpoint(a.checkpoint)
    if checkpoint_sha!=d6a.EXPECTED_CHECKPOINT_SHA256 or abs(float(meta.get('context_epoch_equivalent',-1.0))-2.0)>1e-12:
        raise RuntimeError('D7A start checkpoint identity failed')
    offset=int(schedule['offset'])
    indices=np.asarray(schedule['permutation'],dtype=np.int64)[offset:offset+a.contexts]
    rng_state=copy.deepcopy(train_rng.bit_generator.state)
    if len(indices)!=1024: raise RuntimeError('checkpoint schedule lacks 1024 contexts')

    # Data/tokenization identical to D6A.
    xall,yall,_xte,_yte=core.load(a.data); x012,y012=core.subset012(xall,yall)
    xtr,_ytr,xva,_yva=core.split(x012,y012,.1,7)
    toknet,_=base.load_frozen_tokenizer(a.tokenizer,device)
    tr,_=base.encode_snn(toknet,xtr,device,a.encode_batch)
    va,_=base.encode_snn(toknet,xva,device,a.encode_batch)
    tok0,pos0,_=build_input_embeddings(toknet,source.c.D,grid=7)
    start=clone(source); upgrade_r6b3_system(start,tok0,pos0,aux_lr_scale=1.0)

    validation_ids=np.asarray(stored['validation_ids'],dtype=np.int64)
    gt=np.asarray(stored['gt_tokens'],dtype=np.int64)
    shared=np.asarray(stored['shared_uniform'],dtype=np.float64)
    if len(gt)!=256 or len(schedules)!=256: raise RuntimeError('frozen Gate-A coordinate system is not 256 images')
    if not np.array_equal(np.asarray(va[validation_ids],dtype=np.int64),gt):
        raise RuntimeError('D7A validation tokens mismatch frozen Gate-A ground truth')

    run=run_triplet(start,tr,indices,rng_state)
    F=run['F']; B=run['B']; C=run['C']
    if not(run['streams']['F']==run['streams']['B']==run['streams']['C']):
        raise RuntimeError('D7A all-arm stream exactness failed')
    if not(run['rng_after']['F']==run['rng_after']['B']==run['rng_after']['C']):
        raise RuntimeError('D7A all-arm external RNG exactness failed')
    if run['max_postclamp_W2_abs_error']!=0.0 or not np.array_equal(C.W2,F.W2):
        raise RuntimeError('D7A safe-W2 clamp identity failed')

    # Replayed F/B must reproduce the frozen successful D6A terminals.
    oldF=d6a1.load_d6a_terminal(locate(a.d6a_artifact,'D6A_F_EM_terminal.pkl'))
    oldB=d6a1.load_d6a_terminal(locate(a.d6a_artifact,'D6A_F_EM_WO_zero_bo_oracle_terminal.pkl'))
    model_replay={'F':model_field_compare_tolerant(F,oldF),'B':model_field_compare_tolerant(B,oldB)}
    if not all(x['pass'] for x in model_replay.values()):
        raise RuntimeError({'D7A_terminal_model_replay_failed':model_replay})

    # Update/oracle identity.
    localF={'contexts':len(F._d5b_rows),
            'mean_context_nll':float(np.mean([r['mean_nll'] for r in F._d5b_rows]))}
    localB=d6a.bo_oracle_summary(B._d5b_rows,1301000,a.bootstrap)
    localC=d6a.bo_oracle_summary(C._d5b_rows,1302000,a.bootstrap)
    for label,local in (('B',localB),('C',localC)):
        if local['max_bo_norm_match_abs_error']>2e-12:
            raise RuntimeError(f'D7A {label} bo norm match failed')
        if local['max_WO_zero_abs_error']>1e-15:
            raise RuntimeError(f'D7A {label} WO-zero identity failed')
    oracle_valid=bool(localB['finite_step_interpretability_pass'] and localC['finite_step_interpretability_pass'])

    # Gate-A primary deployment.
    gate={'F':d6a.gate_eval('D7A_F',F,schedules,gt,shared),
          'B':d6a.gate_eval('D7A_B',B,schedules,gt,shared),
          'C':d6a.gate_eval('D7A_C_W2_safe_clamp',C,schedules,gt,shared)}
    calibration={'F_to_D6A':gate_calibration(gate['F'],stored,F_LABEL),
                 'B_to_D6A':gate_calibration(gate['B'],stored,B_LABEL)}
    if not all(x['pass'] for x in calibration.values()):
        raise RuntimeError({'D7A_D6A_GateA_replay_failed':calibration})

    contrasts={'C_minus_B':{},'C_minus_F':{},'B_minus_F':{}}
    for mi,m in enumerate(ALL_METRICS):
        contrasts['C_minus_B'][m]=contrast(gate['C'],gate['B'],m,1310000+10*mi,a.bootstrap)
        contrasts['C_minus_F'][m]=contrast(gate['C'],gate['F'],m,1320000+10*mi,a.bootstrap)
        contrasts['B_minus_F'][m]=contrast(gate['B'],gate['F'],m,1330000+10*mi,a.bootstrap)

    recovery={}
    for m in ALL_METRICS:
        jf=mean_effect(gate['F'],m); jb=mean_effect(gate['B'],m); jc=mean_effect(gate['C'],m)
        den=jb-jf; num=jb-jc
        recovery[m]={'F_mean':jf,'B_mean':jb,'C_mean':jc,'B_minus_F':den,'B_minus_C':num,
                     'gap_closure_fraction':None if den<=EPS else float(num/den)}

    # Multi-objective guardrails under common deterministic seeds.
    mask_seed=a.seed+71000
    masked={k:eval_masked(s,va,mask_seed,a.eval_images) for k,s in {'F':F,'B':B,'C':C}.items()}
    _rc,real_p=coarse.probability_from_ids(va,source.c.V)
    _pc,real_pos_p=spatial.position_probabilities(va,source.c.V,source.c.N)
    real_pair=spatial.pair_reference(va,source.c.V,source.c.N,7)
    rollout={k:rollout_js(s,real_p,real_pos_p,real_pair,BASE_GEN_SEED,a.gen_samples)
             for k,s in {'F':F,'B':B,'C':C}.items()}
    if not all(v.get('all_trace_audits_pass',False) for v in rollout.values()):
        raise RuntimeError('D7A rollout trace audit failed')
    guards=guardrails(masked,rollout)

    primary_flags={m:bool(contrasts['C_minus_B'][m]['ci_entirely_negative']) for m in PRIMARY}
    primary_all=bool(all(primary_flags.values())); primary_any=bool(any(primary_flags.values()))
    if not oracle_valid:
        classification='TX_ORACLE_INTERPRETATION_INVALID'
    elif primary_all and guards['masked_ce_all_pass'] and guards['rollout_js_all_pass']:
        classification='T1_STRONG_W2_TRAJECTORY_RESCUE'
    elif primary_all:
        classification='T2_PRIMARY_RESCUE_WITH_GUARDRAIL_TENSION'
    elif primary_any:
        classification='T3_PARTIAL_W2_TRAJECTORY_RESCUE'
    else:
        classification='T0_NO_W2_TRAJECTORY_RESCUE'

    # Persist full audit trail.
    write_csv(a.out/'D7A_W2_CLAMP_TRACE.csv',run['trace'])
    d6a.d5b.write_csv(a.out/'D7A_B_LOCAL_ROWS.csv',B._d5b_rows)
    d6a.d5b.write_csv(a.out/'D7A_C_W2_SAFE_CLAMP_LOCAL_ROWS.csv',C._d5b_rows)
    save_terminal(a.out/'D7A_F_terminal.pkl',F,'F')
    save_terminal(a.out/'D7A_B_terminal.pkl',B,'B')
    save_terminal(a.out/'D7A_C_W2_safe_clamp_terminal.pkl',C,'C')

    npz={'validation_ids':validation_ids,'gt_tokens':gt.astype(np.int16),'shared_uniform':shared}
    for label,ev in gate.items():
        npz[f'{label}__self_metrics']=ev['self_metrics']; npz[f'{label}__teacher_metrics']=ev['teacher_metrics']
        npz[f'{label}__self_proposal']=ev['self_proposal']; npz[f'{label}__teacher_proposal']=ev['teacher_proposal']
        for m,v in ev['effects'].items(): npz[f'{label}__effect__{m}']=v
    np.savez_compressed(a.out/'D7A_EFFECT_ARRAYS.npz',**npz)

    summary={'stage':'GATE_D7A_CAUSAL_UPPER_BOUND_W2_READAPTATION_PREVENTION',
             'classification':classification,
             'source_runs':{'D6A':39687028646,'D6A1':39777693838,'D6A2':39806081064,'D6A3':39819736038},
             'learner_modified':True,
             'modification_scope':'privileged diagnostic W2 trajectory clamp only; no candidate final admissibility rule',
             'candidate_final_learner_claimed':False,
             'checkpoint_sha256':checkpoint_sha,
             'all_arm_stream_exact':True,'all_arm_external_rng_exact':True,
             'model_replay_to_D6A':model_replay,'GateA_replay_to_D6A':calibration,
             'W2_clamp':{'exact_after_every_context':True,
                         'max_postclamp_abs_error':run['max_postclamp_W2_abs_error'],
                         'cumulative_correction_path_length':run['cumulative_clamp_correction'],
                         'checkpoints':run['trace']},
             'local_training':{'F':localF,'B':localB,'C':localC},
             'oracle_valid':oracle_valid,
             'deployment_contrasts':contrasts,'gap_closure':recovery,
             'secondary_masked':masked,'secondary_rollout':rollout,'guardrails':guards,
             'decision':{'primary_C_minus_B_significant_improvement':primary_flags,
                         'primary_all_pass':primary_all,'primary_any_pass':primary_any,
                         'masked_ce_guardrail_pass':guards['masked_ce_all_pass'],
                         'rollout_js_guardrail_pass':guards['rollout_js_all_pass']},
             'interpretation_boundary':('D7A tests a privileged trajectory-level causal upper bound. A rescue shows that preventing W2 re-adaptation is sufficient under the time-matched F_EM reference trajectory; it does not provide a local, biological, static-projector, or SpiNNaker-ready learning rule. Gap-closure fractions are descriptive and are not equivalence tests.')}
    (a.out/'D7A_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    manifest={'classification':classification,'contexts':a.contexts,'eval_images':a.eval_images,
              'gen_samples':a.gen_samples,'bootstrap':a.bootstrap,'checkpoint_sha256':checkpoint_sha,
              'arms':['F','B','C_W2_safe_clamp'],'protected_baseline_modified':False,
              'no_PR_no_merge':True,'final_local_rule_claimed':False}
    (a.out/'D7A_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('D7A_RESULT',json.dumps({'classification':classification,'primary':primary_flags,
                                   'gap_closure':{m:recovery[m] for m in PRIMARY},
                                   'guardrails':guards,'oracle_valid':oracle_valid},indent=2),flush=True)


if __name__=='__main__':
    main()
