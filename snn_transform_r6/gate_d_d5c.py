#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D5C — local-descent / self-conditioning gradient-gap mapping.

Eval-only terminal sensitivity audit around the successful D5B F_native terminal.
No learner update is executed. Exact masked-token descent directions for W2, WO,
and bo are defined on a held-out validation batch disjoint from the frozen Gate-A
256 images, then tested under frozen-branch and full-CRN Gate-A perturbations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch

# Import before unpickling D5B terminal objects.
import gate_d_d5b  # noqa: F401
import gate_d_d5a as d5a
import gate_d_d3e_eval as d3eval
import gate_d_d3e_runtime as d3r
import gate_d_plasticity_audit as gd
import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
import r6b3_decoder_operating_point_audit as coarse
import r6b3_decoder_spatial_refinement_audit as spatial
from gate_d_d3e_eval import BASE_GEN_SEED, IX, model_dict, predict_round
from maskgit_msdl_training import sample_mask_set

PATHS=('W2','WO','bo')
ALPHAS=(0.25,0.5,1.0)
PRIMARY_FROZEN_ALPHA=0.25
PRIMARY_FULL_ALPHA=1.0
GRAD_IMAGES=256
GRAD_ID_SEED=551901
GRAD_MASK_SEED=552003
BOOTSTRAP=10000
DOM=61
MASK=64
ROUNDS=3
TOL=1e-7
EPS=1e-30


def clone(x):
    return pickle.loads(pickle.dumps(x,pickle.HIGHEST_PROTOCOL))


def locate(root:Path,name:str)->Path:
    return d3r.locate(root,name)


def load_terminal(path:Path):
    with path.open('rb') as f: obj=pickle.load(f)
    if not isinstance(obj,dict) or 'system' not in obj:
        raise RuntimeError(f'bad terminal schema: {path}')
    return obj['system']


def bootstrap_mean(x,seed,nboot=BOOTSTRAP,chunk=500):
    a=np.asarray(x,dtype=np.float64)
    a=a[np.isfinite(a)]
    if len(a)==0:
        return {'n':0,'mean':None,'median':None,'ci95':[None,None],
                'ci_entirely_positive':False,'ci_entirely_negative':False}
    rng=np.random.default_rng(seed); z=np.empty(nboot,dtype=np.float64); p=0
    while p<nboot:
        n=min(chunk,nboot-p); ix=rng.integers(0,len(a),size=(n,len(a)))
        z[p:p+n]=a[ix].mean(axis=1); p+=n
    q=np.quantile(z,[.025,.975])
    return {'n':int(len(a)),'mean':float(np.mean(a)),'median':float(np.median(a)),
            'ci95':[float(q[0]),float(q[1])],
            'fraction_positive':float(np.mean(a>0)),'fraction_negative':float(np.mean(a<0)),
            'ci_entirely_positive':bool(q[0]>0),'ci_entirely_negative':bool(q[1]<0)}


def nanaware_max_abs(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    if a.shape!=b.shape: return float('inf')
    if not np.array_equal(np.isnan(a),np.isnan(b)): return float('inf')
    if not np.array_equal(np.isposinf(a),np.isposinf(b)): return float('inf')
    if not np.array_equal(np.isneginf(a),np.isneginf(b)): return float('inf')
    good=np.isfinite(a)&np.isfinite(b)
    return 0.0 if not np.any(good) else float(np.max(np.abs(a[good]-b[good])))


def arr_sha(a):
    x=np.ascontiguousarray(np.asarray(a))
    h=hashlib.sha256(); h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()


def json_sha(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def read_numeric_column(path:Path,key:str):
    out=[]
    with path.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f): out.append(float(r[key]))
    return np.asarray(out,dtype=np.float64)


def select_gradient_ids(nval,gate_ids):
    used=set(int(x) for x in np.asarray(gate_ids).reshape(-1))
    comp=np.asarray([i for i in range(int(nval)) if i not in used],dtype=np.int64)
    if len(comp)<GRAD_IMAGES: raise RuntimeError('not enough validation images outside Gate-A set')
    return np.sort(np.random.default_rng(GRAD_ID_SEED).choice(comp,size=GRAD_IMAGES,replace=False).astype(np.int64))


def build_gradient_contexts(system,va,grad_ids):
    rng=np.random.default_rng(GRAD_MASK_SEED); contexts=[]; mask_sig=[]
    for ii in grad_ids:
        sm=sample_mask_set(system.c.N,rng); masked=np.asarray(sm.masked_positions,dtype=np.int64)
        truth=np.asarray(va[int(ii)],dtype=np.int64); arr=truth.copy(); arr[masked]=core.MASK
        contexts.append({'image_index':int(ii),'arr':arr,'truth':truth,'masked':masked})
        mask_sig.append({'image_index':int(ii),'masked':[int(x) for x in masked]})
    return contexts,mask_sig


def gradient_directions(base_system,contexts):
    s=clone(base_system)
    sums={k:np.zeros_like(getattr(s,k),dtype=np.float64) for k in PATHS}
    base_losses=[]; target_count=0
    for c in contexts:
        arr=np.asarray(c['arr'],dtype=np.int64); masked=np.asarray(c['masked'],dtype=np.int64); truth=np.asarray(c['truth'],dtype=np.int64)
        s._current_arr=arr.copy(); xn,xraw=s.context_from_arr(arr); cache=s.prepare_mask_batch(xn,xraw,masked)
        cg={k:np.zeros_like(sums[k]) for k in PATHS}; losses=[]
        for rec in cache:
            tgt=int(truth[int(rec['t'])]); o=d5a.exact_event_oracles(s,rec,tgt)
            for k in PATHS: cg[k]+=np.asarray(o[k],dtype=np.float64)
            losses.append(-math.log(max(float(np.asarray(rec['p'])[tgt]),1e-300)))
        n=max(1,len(cache)); target_count+=len(cache); base_losses.append(float(np.mean(losses)))
        for k in PATHS: sums[k]+=cg[k]/n
    out={};
    for k in PATHS:
        g=sums[k]/len(contexts); gn=float(np.linalg.norm(g))
        if gn<=EPS: raise RuntimeError(f'zero held-out gradient for {k}')
        out[k]={'gradient':g,'direction':g/gn,'gradient_norm':gn}
    return out,float(np.mean(base_losses)),int(target_count)


def local_batch_ce(system,contexts):
    s=clone(system); vals=[]
    for c in contexts:
        arr=np.asarray(c['arr'],dtype=np.int64); masked=np.asarray(c['masked'],dtype=np.int64); truth=np.asarray(c['truth'],dtype=np.int64)
        s._current_arr=arr.copy(); xn,xraw=s.context_from_arr(arr); cache=s.prepare_mask_batch(xn,xraw,masked)
        ls=[]
        for rec in cache:
            tgt=int(truth[int(rec['t'])]); ls.append(-math.log(max(float(np.asarray(rec['p'])[tgt]),1e-300)))
        vals.append(float(np.mean(ls)))
    return float(np.mean(vals))


def perturb(system,path,direction,step):
    s=clone(system); x=np.asarray(getattr(s,path),dtype=np.float64)+float(step)*np.asarray(direction,dtype=np.float64)
    if path=='W2': x*=s.mask2
    if path=='WO': x*=s.maskO
    setattr(s,path,np.array(x,copy=True)); return s


def baseline_gate_traces(system,schedules,gt,shared):
    traces=[]; sp=np.full((len(gt),ROUNDS,system.c.N),-1,np.int16); tp=np.full_like(sp,-1)
    for i in range(len(gt)):
        ps=core.P3ImageSystem(system.c,model_dict(system),BASE_GEN_SEED+100000+i)
        pt=core.P3ImageSystem(system.c,model_dict(system),BASE_GEN_SEED+100000+i)
        cs=np.full(system.c.N,MASK,int); ct=np.full(system.c.N,MASK,int); rows=[]
        for r in range(ROUNDS):
            tr=schedules[i][r]; masked=np.asarray(tr['masked_positions_before'],int); remask=np.asarray(tr['remasked_positions'],int)
            p1=predict_round(ps,cs,system.tok,system.pos,masked); p2=predict_round(pt,ct,system.tok,system.pos,masked)
            u=np.asarray(shared[i,r,masked],float); z1=d3eval.inverse_cdf_rows(p1,u); z2=d3eval.inverse_cdf_rows(p2,u)
            rows.append({'self_context':cs.copy(),'teacher_context':ct.copy(),'masked':masked.copy()})
            pr1=cs.copy(); pr1[masked]=z1; pr2=ct.copy(); pr2[masked]=z2; sp[i,r]=pr1; tp[i,r]=pr2
            ns=pr1.copy(); ns[remask]=MASK
            nt=ct.copy(); new=np.setdiff1d(masked,remask,assume_unique=True); nt[new]=gt[i,new]; nt[remask]=MASK
            old=np.flatnonzero(ct!=MASK)
            if len(old): nt[old]=gt[i,old]
            cs,ct=ns,nt
        traces.append(rows)
    return traces,sp,tp


def two_metrics(p,truth):
    p=np.asarray(p,dtype=np.float64); truth=np.asarray(truth,dtype=np.int64); non=truth!=DOM
    if not np.any(non): return float('nan'),float('nan')
    ce=-np.log(np.clip(p[np.arange(len(truth)),truth],1e-12,1.0))
    return float(np.mean(ce[non])),float(np.mean(p[non,DOM]))


def frozen_eval(system,traces,gt):
    # arrays: image, round, [self_D, teacher_D, self_F, teacher_F]
    out=np.full((len(gt),ROUNDS,4),np.nan,dtype=np.float64)
    for i,rows in enumerate(traces):
        ps=core.P3ImageSystem(system.c,model_dict(system),BASE_GEN_SEED+100000+i)
        pt=core.P3ImageSystem(system.c,model_dict(system),BASE_GEN_SEED+100000+i)
        for r,row in enumerate(rows):
            masked=np.asarray(row['masked'],int); truth=np.asarray(gt[i,masked],int)
            p1=predict_round(ps,np.asarray(row['self_context'],int),system.tok,system.pos,masked)
            p2=predict_round(pt,np.asarray(row['teacher_context'],int),system.tok,system.pos,masked)
            sd,sf=two_metrics(p1,truth); td,tf=two_metrics(p2,truth); out[i,r]=[sd,td,sf,tf]
    return out


def full_components(ev):
    r=2
    return {
        'self_D':np.asarray(ev['self_metrics'],float)[:,r,IX['ce_non61']],
        'teacher_D':np.asarray(ev['teacher_metrics'],float)[:,r,IX['ce_non61']],
        'self_F':np.asarray(ev['self_metrics'],float)[:,r,IX['p61_on_non61']],
        'teacher_F':np.asarray(ev['teacher_metrics'],float)[:,r,IX['p61_on_non61']],
    }


def frozen_components(a):
    r=2
    return {'self_D':a[:,r,0],'teacher_D':a[:,r,1],'self_F':a[:,r,2],'teacher_F':a[:,r,3]}


def response_summary(plus,minus,alpha,seed0,nboot):
    p=plus; m=minus; out={}; raw={}
    for j,key in enumerate(('self_D','teacher_D','self_F','teacher_F')):
        v=(np.asarray(p[key],float)-np.asarray(m[key],float))/(2.*float(alpha)); raw[key]=v
        out[key]=bootstrap_mean(v,seed0+j,nboot)
    raw['gap_D']=raw['self_D']-raw['teacher_D']; raw['gap_F']=raw['self_F']-raw['teacher_F']
    out['gap_D']=bootstrap_mean(raw['gap_D'],seed0+10,nboot); out['gap_F']=bootstrap_mean(raw['gap_F'],seed0+11,nboot)
    return out,raw


def proposal_switch(ev,base_self):
    a=np.asarray(ev['self_proposal']); b=np.asarray(base_self); diff=a!=b
    rows=[]
    for r in range(ROUNDS):
        d=diff[:,r]
        rows.append({'round':r+1,'token_fraction':float(np.mean(d)),'image_any_fraction':float(np.mean(np.any(d,axis=1)))})
    return rows


def gate_eval(name,s,schedules,gt,shared):
    e=d3eval.gate_for_system(name,s,schedules,gt,shared)
    if not e['gates']['all_pass']: raise RuntimeError(f'Gate-A structural failure {name}')
    return e


def calibration(ev,stored):
    d={'self_metrics_max_abs':nanaware_max_abs(ev['self_metrics'],stored['F_native__self_metrics']),
       'teacher_metrics_max_abs':nanaware_max_abs(ev['teacher_metrics'],stored['F_native__teacher_metrics']),
       'self_proposal_exact':bool(np.array_equal(ev['self_proposal'],stored['F_native__self_proposal'])),
       'teacher_proposal_exact':bool(np.array_equal(ev['teacher_proposal'],stored['F_native__teacher_proposal']))}
    d['pass']=bool(d['self_metrics_max_abs']<TOL and d['teacher_metrics_max_abs']<TOL and d['self_proposal_exact'] and d['teacher_proposal_exact'])
    return d


def ci_pos(q): return bool(q.get('ci_entirely_positive',False))
def ci_neg(q): return bool(q.get('ci_entirely_negative',False))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--d5b-artifact',type=Path,required=True); ap.add_argument('--d5a-artifact',type=Path,required=True); ap.add_argument('--anchor-artifact',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--encode-batch',type=int,default=256); ap.add_argument('--bootstrap',type=int,default=10000); ap.add_argument('--seed',type=int,default=7311); ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads); device=torch.device('cpu')

    d5b_summary=json.loads(locate(a.d5b_artifact,'D5B_SUMMARY.json').read_text(encoding='utf-8'))
    if d5b_summary.get('classification')!='L1_STRONG_LOCAL_OBJECTIVE_DEPLOYMENT_MISMATCH': raise RuntimeError('D5C requires certified D5B L1 prerequisite')
    base_system=load_terminal(locate(a.d5b_artifact,'D5B_F_native_terminal.pkl'))
    stored=np.load(locate(a.d5b_artifact,'D5B_EFFECT_ARRAYS.npz'),allow_pickle=False)
    schedules=json.loads(locate(a.anchor_artifact,'HISTORICAL_R6B3_R3_T8_REFERENCE_TRACE.json').read_text(encoding='utf-8'))
    anchor=np.load(locate(a.anchor_artifact,'R6B4_NATIVE_GATE_A_EXECUTION_ANCHOR.npz'),allow_pickle=False)
    gate_ids=np.asarray(anchor['validation_ids'],dtype=np.int64); gt=np.asarray(anchor['gt_tokens'],dtype=np.int64); shared=np.asarray(anchor['shared_uniform'],dtype=np.float64)

    xall,yall,_xte,_yte=core.load(a.data); x012,y012=core.subset012(xall,yall); _xtr,_ytr,xva,_yva=core.split(x012,y012,.1,7)
    toknet,_=base.load_frozen_tokenizer(a.tokenizer,device); va,_=base.encode_snn(toknet,xva,device,a.encode_batch)
    if not np.array_equal(np.asarray(va[gate_ids],dtype=np.int64),gt): raise RuntimeError('Gate-A validation token identity failed')

    # Hard base calibration to successful D5B.
    base_gate=gate_eval('D5C_base_F_native',base_system,schedules,gt,shared); cal=calibration(base_gate,stored)
    if not cal['pass']: raise RuntimeError({'D5C_base_D5B_calibration_failed':cal})
    traces,bsp,btp=baseline_gate_traces(base_system,schedules,gt,shared)
    trace_pass=bool(np.array_equal(bsp,stored['F_native__self_proposal']) and np.array_equal(btp,stored['F_native__teacher_proposal']))
    if not trace_pass: raise RuntimeError('frozen-branch baseline trace reconstruction failed')

    # Native-step reference scales from the successful F_EM trajectory.
    d5a_csv=locate(a.d5a_artifact,'D5A_F_EM_CONTEXT_ROWS.csv')
    step_ref={k:float(np.median(read_numeric_column(d5a_csv,f'{k}_u_norm'))) for k in PATHS}
    d5b_csv=locate(a.d5b_artifact,'D5B_F_native_LOCAL_ROWS.csv')
    cross={'WO':float(np.median(read_numeric_column(d5b_csv,'native_WO_norm'))),'bo':float(np.median(read_numeric_column(d5b_csv,'native_bo_norm')))}
    if abs(step_ref['WO']-cross['WO'])>1e-9 or abs(step_ref['bo']-cross['bo'])>1e-9: raise RuntimeError('D5A/D5B native O-step reference mismatch')

    # Define exact local directions on a Gate-A-disjoint held-out masked batch.
    grad_ids=select_gradient_ids(len(va),gate_ids); grad_contexts,mask_sig=build_gradient_contexts(base_system,va,grad_ids)
    dirs,grad_ce0,target_count=gradient_directions(base_system,grad_contexts)
    direction_audit={'gradient_ids_sha256':arr_sha(grad_ids),'mask_schedule_sha256':json_sha(mask_sig),'images':GRAD_IMAGES,'targets':target_count,'base_mean_context_ce':grad_ce0,'paths':{}}
    for k in PATHS:
        direction_audit['paths'][k]={'gradient_norm':dirs[k]['gradient_norm'],'native_step_reference':step_ref[k],'scales':{}}
        for alpha in ALPHAS:
            plus=perturb(base_system,k,dirs[k]['direction'],alpha*step_ref[k]); minus=perturb(base_system,k,dirs[k]['direction'],-alpha*step_ref[k])
            cp=local_batch_ce(plus,grad_contexts); cm=local_batch_ce(minus,grad_contexts)
            direction_audit['paths'][k]['scales'][str(alpha)]={'ce_plus':cp,'ce_base':grad_ce0,'ce_minus':cm,'plus_improves':bool(cp<grad_ce0),'minus_worsens':bool(cm>grad_ce0)}
        q=direction_audit['paths'][k]['scales'][str(PRIMARY_FROZEN_ALPHA)]
        direction_audit['paths'][k]['primary_direction_valid']=bool(q['plus_improves'] and q['minus_worsens'])
    if not all(direction_audit['paths'][k]['primary_direction_valid'] for k in PATHS): raise RuntimeError('held-out local direction validity failed')

    results={'frozen':{},'full_crn':{}}; npz={'gradient_ids':grad_ids,'gt_tokens':gt.astype(np.int16),'shared_uniform':shared,'base_self_proposal':np.asarray(base_gate['self_proposal'])}
    perturb_gates={}
    for pi,k in enumerate(PATHS):
        results['frozen'][k]={}; results['full_crn'][k]={}
        for ai,alpha in enumerate(ALPHAS):
            sp=perturb(base_system,k,dirs[k]['direction'],alpha*step_ref[k]); sm=perturb(base_system,k,dirs[k]['direction'],-alpha*step_ref[k])
            fp=frozen_eval(sp,traces,gt); fm=frozen_eval(sm,traces,gt)
            fr,frraw=response_summary(frozen_components(fp),frozen_components(fm),alpha,900000+10000*pi+100*ai,a.bootstrap)
            results['frozen'][k][str(alpha)]={'response_per_native_step':fr}
            ep=gate_eval(f'{k}_plus_{alpha}',sp,schedules,gt,shared); em=gate_eval(f'{k}_minus_{alpha}',sm,schedules,gt,shared)
            perturb_gates[f'{k}_plus_{alpha}']=ep; perturb_gates[f'{k}_minus_{alpha}']=em
            fu,furaw=response_summary(full_components(ep),full_components(em),alpha,910000+10000*pi+100*ai,a.bootstrap)
            results['full_crn'][k][str(alpha)]={'response_per_native_step':fu,'plus_switch_vs_base':proposal_switch(ep,base_gate['self_proposal']),'minus_switch_vs_base':proposal_switch(em,base_gate['self_proposal'])}
            for nm,v in frraw.items(): npz[f'frozen__{k}__{alpha}__{nm}']=v
            for nm,v in furaw.items(): npz[f'full__{k}__{alpha}__{nm}']=v
            npz[f'full__{k}__{alpha}__plus_self_proposal']=ep['self_proposal']; npz[f'full__{k}__{alpha}__minus_self_proposal']=em['self_proposal']

    # Preregistered classification.
    fwo=results['frozen']['WO'][str(PRIMARY_FROZEN_ALPHA)]['response_per_native_step']; uwo=results['full_crn']['WO'][str(PRIMARY_FULL_ALPHA)]['response_per_native_step']
    teacher_improve=ci_neg(fwo['teacher_D']); frozen_gap=bool(ci_pos(fwo['gap_D']) or ci_pos(fwo['gap_F'])); full_gap=bool(ci_pos(uwo['gap_D']) or ci_pos(uwo['gap_F']))
    control_patterns={}
    for k in ('W2','bo'):
        q=results['frozen'][k][str(PRIMARY_FROZEN_ALPHA)]['response_per_native_step']; control_patterns[k]=bool(ci_neg(q['teacher_D']) and (ci_pos(q['gap_D']) or ci_pos(q['gap_F'])))
    c1=bool(teacher_improve and frozen_gap and full_gap and not any(control_patterns.values()))
    no_frozen_gap_small=True
    for al in (0.25,0.5):
        q=results['frozen']['WO'][str(al)]['response_per_native_step']; no_frozen_gap_small=no_frozen_gap_small and not(ci_pos(q['gap_D']) or ci_pos(q['gap_F']))
    sw=results['full_crn']['WO'][str(PRIMARY_FULL_ALPHA)]['plus_switch_vs_base']; nonzero_switch=bool(any(x['image_any_fraction']>0 for x in sw))
    c3=bool((not c1) and no_frozen_gap_small and full_gap and nonzero_switch)
    case='C1_SELF_CONDITIONING_MISMATCH_CONFIRMED' if c1 else ('C3_DISCRETE_BRANCH_AMPLIFICATION' if c3 else 'C2_MISMATCH_NOT_LOCALIZED_TO_MEASURED_SELF_CONDITIONING_GAP')

    # Secondary ordinary masked and autonomous JS at the primary full perturbation, descriptive only.
    secondary={}
    _rc,real_p=coarse.probability_from_ids(va,base_system.c.V); _pc,real_pos_p=spatial.position_probabilities(va,base_system.c.V,base_system.c.N); real_pair=spatial.pair_reference(va,base_system.c.V,base_system.c.N,7)
    for k in PATHS:
        s=perturb(base_system,k,dirs[k]['direction'],PRIMARY_FULL_ALPHA*step_ref[k])
        secondary[k]={'masked':d3eval.eval_masked(s,va,a.seed+51000,256),'rollout_js':d3eval.rollout_js(s,real_p,real_pos_p,real_pair,BASE_GEN_SEED,256)}

    np.savez_compressed(a.out/'D5C_ARRAYS.npz',**npz)
    summary={'stage':'GATE_D5C_LOCAL_DESCENT_SELF_CONDITIONING_GRADIENT_GAP','classification':case,'source_D5B_run':39570284981,
             'learner_modified':False,'base_D5B_calibration':cal,'frozen_trace_reconstruction_exact':trace_pass,
             'gradient_batch_disjoint_from_gateA':bool(len(set(map(int,grad_ids))&set(map(int,gate_ids)))==0),
             'direction_audit':direction_audit,'step_reference':step_ref,'results':results,'secondary':secondary,
             'decision':{'teacher_WO_CE_improves_frozen_primary':teacher_improve,'WO_positive_frozen_gap_primary':frozen_gap,'WO_positive_full_gap_primary':full_gap,'control_complete_patterns':control_patterns,'WO_no_positive_frozen_gap_alpha025_05':no_frozen_gap_small,'WO_full_alpha1_nonzero_self_branch_switch':nonzero_switch,'C1':c1,'C3':c3},
             'interpretation_boundary':'D5C is a terminal eval-only directional sensitivity audit around D5B F_native. It localizes, but does not globally redefine, the masked-token objective or generation objective.'}
    (a.out/'D5C_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    manifest={'paths':list(PATHS),'alphas':list(ALPHAS),'primary_frozen_alpha':PRIMARY_FROZEN_ALPHA,'primary_full_crn_alpha':PRIMARY_FULL_ALPHA,'gradient_images':GRAD_IMAGES,'gradient_id_seed':GRAD_ID_SEED,'gradient_mask_seed':GRAD_MASK_SEED,'bootstrap':a.bootstrap,'gateA_images':int(len(gt)),'rounds':ROUNDS,'learner_modified':False}
    (a.out/'D5C_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('D5C_COMPLETE',json.dumps({'classification':case,'calibration_pass':cal['pass'],'direction_valid':{k:direction_audit['paths'][k]['primary_direction_valid'] for k in PATHS},'decision':summary['decision'],'primary_frozen_WO':fwo,'primary_full_WO':uwo},indent=2),flush=True)

if __name__=='__main__': main()
