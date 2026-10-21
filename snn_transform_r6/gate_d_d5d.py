#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D5D — emergence of negative deployment utility along exact-WO trajectory.

Replay/readout only. Replays the already-certified D5B F_native and F_WO_oracle
rules, saves seven paired checkpoints, and measures deployment onset, fixed-probe
WO gradient-field rotation, and path-vs-chord deployment utility.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch

import gate_d_d3e_eval as d3eval
import gate_d_d3e_runtime as d3r
import gate_d_d5b as d5b
import gate_d_d5c as d5c
import gate_d_plasticity_audit as gd
import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
import r6b3_context_epoch_training as trainer
import r6b3_decoder_operating_point_audit as coarse
import r6b3_decoder_spatial_refinement_audit as spatial
from maskgit_input_representation import build_input_embeddings
from r6b4_full_forward_plasticity import upgrade_r6b3_system

CHECKPOINTS=(0,64,128,256,512,768,1024)
PRIMARY=('F61','D_non61')
SUPPORTIVE=('C61','delta_P61')
ALL_METRICS=('F61','D_non61','C61','delta_P61','CE')
CHORD_LAMBDAS={0:0.0,64:0.0625,128:0.125,256:0.25,512:0.5,768:0.75,1024:1.0}
TOL=1e-7
DOM=61
V=64
EPS=1e-30


def clone(x):
    return pickle.loads(pickle.dumps(x,pickle.HIGHEST_PROTOCOL))


def locate(root:Path,name:str)->Path:
    return d3r.locate(root,name)


def jsonable(x):
    if isinstance(x,dict): return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [jsonable(v) for v in x]
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    return x


def write_csv(path:Path,rows):
    if not rows:
        path.write_text('',encoding='utf-8'); return
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); keys.append(k)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore'); w.writeheader(); w.writerows(rows)


def cos(a,b):
    a=np.asarray(a,float).reshape(-1); b=np.asarray(b,float).reshape(-1)
    na=float(np.linalg.norm(a)); nb=float(np.linalg.norm(b))
    if na<=EPS or nb<=EPS: return float('nan')
    return float(np.dot(a,b)/(na*nb))


def js(p,q):
    p=np.asarray(p,float); q=np.asarray(q,float)
    p=p/max(EPS,float(p.sum())); q=q/max(EPS,float(q.sum())); m=.5*(p+q)
    def kl(a,b):
        z=a>0
        return float(np.sum(a[z]*np.log(a[z]/np.clip(b[z],EPS,None))))
    return .5*kl(p,m)+.5*kl(q,m)


def gate_eval(name,s,schedules,gt,shared):
    e=d3eval.gate_for_system(name,s,schedules,gt,shared)
    if not e['gates']['all_pass']: raise RuntimeError(f'Gate-A structural failure: {name}')
    return e


def bootstrap(x,seed,B):
    return d5b.bootstrap_mean(np.asarray(x,float),seed,B)


def calibrate_gate(e,stored,prefix):
    d={
        'self_metrics_max_abs':d5b.nanaware_max_abs(e['self_metrics'],stored[f'{prefix}__self_metrics']),
        'teacher_metrics_max_abs':d5b.nanaware_max_abs(e['teacher_metrics'],stored[f'{prefix}__teacher_metrics']),
        'self_proposal_exact':bool(np.array_equal(e['self_proposal'],stored[f'{prefix}__self_proposal'])),
        'teacher_proposal_exact':bool(np.array_equal(e['teacher_proposal'],stored[f'{prefix}__teacher_proposal'])),
        'effects':{}
    }
    for m in ALL_METRICS:
        d['effects'][m]=d5b.nanaware_max_abs(e['effects'][m],stored[f'{prefix}__effect__{m}'])
    d['pass']=bool(d['self_metrics_max_abs']<TOL and d['teacher_metrics_max_abs']<TOL and
                   d['self_proposal_exact'] and d['teacher_proposal_exact'] and max(d['effects'].values())<TOL)
    return d


def set_WO(background,donor_WO):
    s=clone(background); s.WO=np.array(donor_WO,copy=True); s.WO*=s.maskO; return s


def save_checkpoint(path,s,arm,t):
    with path.open('wb') as f:
        pickle.dump({'arm':str(arm),'context':int(t),'system':s},f,pickle.HIGHEST_PROTOCOL)


def replay_pair(start,tokens,indices,rng_state):
    native=d5b.make_d5b(start,'F_native'); oracle=d5b.make_d5b(start,'F_WO_oracle')
    rn=gd.clone_rng(rng_state); ro=gd.clone_rng(rng_state)
    cpsN={0:clone(native)}; cpsO={0:clone(oracle)}; stream=[]
    for j,image_index in enumerate(indices,start=1):
        b1=d3r.jstate(rn); b2=d3r.jstate(ro)
        d3r.reset_boundary(native,'F_EM'); d3r.reset_boundary(oracle,'F_EM')
        if d3r.jstate(rn)!=b1 or d3r.jstate(ro)!=b2: raise RuntimeError('F_EM reset consumed RNG')
        truth=np.asarray(tokens[int(image_index)],dtype=np.int64)
        rnrec,nev=d5b.run_context_d5b(native,j,int(image_index),truth,rn)
        rorec,oev=d5b.run_context_d5b(oracle,j,int(image_index),truth,ro)
        if nev!=oev: raise RuntimeError(f'native/oracle stream divergence at context {j}')
        if d3r.jstate(rn)!=d3r.jstate(ro): raise RuntimeError(f'native/oracle external RNG divergence at context {j}')
        stream.append(d5b.stream_row(j,int(image_index),nev))
        if j in CHECKPOINTS:
            cpsN[j]=clone(native); cpsO[j]=clone(oracle)
            print('D5D_REPLAY',json.dumps({'context':j,'native_nll':float(rnrec['mean_nll']),'oracle_nll':float(rorec['mean_nll'])}),flush=True)
    if tuple(sorted(cpsN))!=CHECKPOINTS or tuple(sorted(cpsO))!=CHECKPOINTS: raise RuntimeError('missing D5D checkpoints')
    return cpsN,cpsO,stream,d3r.jstate(rn)


def training_row(s,t,prev_t):
    if t==0:
        return {'cum_online_nll':None,'recent_online_nll':None,'path_length_WO':0.0}
    rows=s._d5b_rows[:t]
    prev=s._d5b_rows[prev_t:t]
    return {
        'cum_online_nll':float(np.mean([r['mean_nll'] for r in rows])),
        'recent_online_nll':float(np.mean([r['mean_nll'] for r in prev])) if prev else None,
        'path_length_WO':float(np.sum([r['actual_WO_norm'] for r in rows])),
    }


def empirical_token_p(ids):
    a=np.asarray(ids,int).reshape(-1); c=np.bincount(a,minlength=V).astype(float); return c/max(1.,float(c.sum()))


def mean_position_entropy(ids):
    x=np.asarray(ids,int); vals=[]
    for j in range(x.shape[1]):
        p=np.bincount(x[:,j],minlength=V).astype(float); p/=max(1.,float(p.sum())); z=p>0
        vals.append(float(-np.sum(p[z]*np.log(p[z]))))
    return float(np.mean(vals))


def gate_markers(e,gt):
    sp=np.asarray(e['self_proposal'],int); sm=np.asarray(e['self_metrics'],float)
    final=sp[:,-1,:]; realp=empirical_token_p(gt); genp=empirical_token_p(final)
    return {
        'token_marginal_js_to_gt_nats':js(genp,realp),
        'mean_position_token_entropy_nats':mean_position_entropy(final),
        'final_P61':float(np.mean(final==DOM)),
        'round_sampled_confidence':[float(np.nanmean(sm[:,r,d3eval.IX['sampled_confidence']])) for r in range(sm.shape[1])],
        'round_proposal61_fraction':[float(np.nanmean(sm[:,r,d3eval.IX['proposal61_fraction']])) for r in range(sm.shape[1])],
        'round_ce_non61':[float(np.nanmean(sm[:,r,d3eval.IX['ce_non61']])) for r in range(sm.shape[1])],
    }


def proposal_switch(a,b):
    aa=np.asarray(a,int); bb=np.asarray(b,int); out=[]
    for r in range(aa.shape[1]):
        d=aa[:,r]!=bb[:,r]
        out.append({'round':r+1,'token_fraction':float(np.mean(d)),'image_any_fraction':float(np.mean(np.any(d,axis=1)))})
    return out


def near_monotone(vals):
    v=np.asarray(vals,float); d=np.diff(v); neg=-d[d<0]
    final=abs(float(v[-1]-v[0]))
    return bool(len(neg)<=1 and (len(neg)==0 or float(np.max(neg))<=0.25*max(final,1e-12)))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True); ap.add_argument('--tokenizer',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--d5b-artifact',type=Path,required=True); ap.add_argument('--d5c-artifact',type=Path,required=True); ap.add_argument('--anchor-artifact',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--encode-batch',type=int,default=256); ap.add_argument('--bootstrap',type=int,default=10000)
    ap.add_argument('--eval-images',type=int,default=256); ap.add_argument('--gen-samples',type=int,default=256); ap.add_argument('--seed',type=int,default=7311); ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    if a.eval_images!=256: raise ValueError('formal D5D requires frozen 256-image Gate-A')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads); device=torch.device('cpu')

    d5b_summary=json.loads(locate(a.d5b_artifact,'D5B_SUMMARY.json').read_text(encoding='utf-8'))
    d5c_summary=json.loads(locate(a.d5c_artifact,'D5C_SUMMARY.json').read_text(encoding='utf-8'))
    if d5b_summary.get('classification')!='L1_STRONG_LOCAL_OBJECTIVE_DEPLOYMENT_MISMATCH': raise RuntimeError('D5D requires certified D5B L1')
    if d5c_summary.get('classification')!='C2_MISMATCH_NOT_LOCALIZED_TO_MEASURED_SELF_CONDITIONING_GAP': raise RuntimeError('D5D requires certified D5C C2')
    stored=np.load(locate(a.d5b_artifact,'D5B_EFFECT_ARRAYS.npz'),allow_pickle=False)
    schedules=json.loads(locate(a.anchor_artifact,'HISTORICAL_R6B3_R3_T8_REFERENCE_TRACE.json').read_text(encoding='utf-8'))
    anchor=np.load(locate(a.anchor_artifact,'R6B4_NATIVE_GATE_A_EXECUTION_ANCHOR.npz'),allow_pickle=False)
    gate_ids=np.asarray(anchor['validation_ids'],dtype=np.int64); gt=np.asarray(anchor['gt_tokens'],dtype=np.int64); shared=np.asarray(anchor['shared_uniform'],dtype=np.float64)

    checkpoint_sha=d3r.sha_file(a.checkpoint); source,meta,train_rng,schedule=trainer.load_full_checkpoint(a.checkpoint)
    if checkpoint_sha!=d5b.EXPECTED_CHECKPOINT_SHA256 or abs(float(meta.get('context_epoch_equivalent',-1))-2.0)>1e-12: raise RuntimeError('D5D checkpoint identity failed')
    offset=int(schedule['offset']); indices=np.asarray(schedule['permutation'],dtype=np.int64)[offset:offset+1024]
    if len(indices)!=1024: raise RuntimeError('D5D requires exact next 1024 contexts')
    rng_state=train_rng.bit_generator.state

    xall,yall,_xte,_yte=core.load(a.data); x012,y012=core.subset012(xall,yall); xtr,_ytr,xva,_yva=core.split(x012,y012,.1,7)
    toknet,_=base.load_frozen_tokenizer(a.tokenizer,device); tr,_=base.encode_snn(toknet,xtr,device,a.encode_batch); va,_=base.encode_snn(toknet,xva,device,a.encode_batch)
    if not np.array_equal(np.asarray(va[gate_ids],dtype=np.int64),gt): raise RuntimeError('D5D Gate-A validation token identity failed')
    tok0,pos0,_=build_input_embeddings(toknet,source.c.D,grid=7); start=clone(source); upgrade_r6b3_system(start,tok0,pos0,aux_lr_scale=1.0)

    cpsN,cpsO,stream,final_rng=replay_pair(start,tr,indices,rng_state)
    for t in CHECKPOINTS:
        save_checkpoint(a.out/f'D5D_F_native_t{t:04d}.pkl',cpsN[t],'F_native',t)
        save_checkpoint(a.out/f'D5D_F_WO_oracle_t{t:04d}.pkl',cpsO[t],'F_WO_oracle',t)

    # Frozen held-out probe reused from D5C construction.
    grad_ids=d5c.select_gradient_ids(len(va),gate_ids); grad_contexts,mask_sig=d5c.build_gradient_contexts(start,va,grad_ids)
    if len(set(map(int,grad_ids))&set(map(int,gate_ids))): raise RuntimeError('D5D probe overlaps Gate-A')

    # Real-distribution references for trajectory-level autonomous JS.
    _rc,real_p=coarse.probability_from_ids(va,start.c.V); _pc,real_pos_p=spatial.position_probabilities(va,start.c.V,start.c.N); real_pair=spatial.pair_reference(va,start.c.V,start.c.N,7)

    gateN={}; gateO={}; maskedN={}; maskedO={}; jsN={}; jsO={}; markersN={}; markersO={}
    rotation_rows=[]; curve_rows=[]; effect_npz={'gradient_ids':grad_ids,'gt_tokens':gt.astype(np.int16),'shared_uniform':shared}
    dirsN={}; dirsO={}; cumangN=0.0; cumangO=0.0; prevN=None; prevO=None
    W0=np.asarray(cpsO[0].WO,float); WT=np.asarray(cpsO[1024].WO,float); chord=WT-W0; chord2=float(np.sum(chord*chord))

    prev_t=0
    delta_arrays={m:{} for m in PRIMARY}
    delta_stats={}
    for ti,t in enumerate(CHECKPOINTS):
        print('D5D_EVAL_CHECKPOINT',t,flush=True)
        gateN[t]=gate_eval(f'native_t{t}',cpsN[t],schedules,gt,shared); gateO[t]=gate_eval(f'oracle_t{t}',cpsO[t],schedules,gt,shared)
        if t==1024:
            calN=calibrate_gate(gateN[t],stored,'F_native'); calO=calibrate_gate(gateO[t],stored,'F_WO_oracle')
            if not(calN['pass'] and calO['pass']): raise RuntimeError({'D5D_D5B_terminal_calibration_failed':{'native':calN,'oracle':calO}})
        maskedN[t]=jsonable(d3eval.eval_masked(cpsN[t],va,a.seed+61000+t,a.eval_images)); maskedO[t]=jsonable(d3eval.eval_masked(cpsO[t],va,a.seed+61000+t,a.eval_images))
        jsN[t]=d3eval.rollout_js(cpsN[t],real_p,real_pos_p,real_pair,d3eval.BASE_GEN_SEED,a.gen_samples); jsO[t]=d3eval.rollout_js(cpsO[t],real_p,real_pos_p,real_pair,d3eval.BASE_GEN_SEED,a.gen_samples)
        markersN[t]=gate_markers(gateN[t],gt); markersO[t]=gate_markers(gateO[t],gt)

        gN,ceN,_=d5c.gradient_directions(cpsN[t],grad_contexts); gO,ceO,_=d5c.gradient_directions(cpsO[t],grad_contexts)
        dirsN[t]=np.asarray(gN['WO']['direction'],float); dirsO[t]=np.asarray(gO['WO']['direction'],float)
        r0N=cos(dirsN[t],dirsN[0]); r0O=cos(dirsO[t],dirsO[0]); rpN=1.0 if prevN is None else cos(dirsN[t],prevN); rpO=1.0 if prevO is None else cos(dirsO[t],prevO)
        if prevN is not None: cumangN+=math.acos(float(np.clip(rpN,-1,1)))
        if prevO is not None: cumangO+=math.acos(float(np.clip(rpO,-1,1)))
        rotation_rows.append({'context':t,'native_probe_ce':ceN,'oracle_probe_ce':ceO,'native_grad_norm':gN['WO']['gradient_norm'],'oracle_grad_norm':gO['WO']['gradient_norm'],
                              'native_R0':r0N,'oracle_R0':r0O,'native_Rprev':rpN,'oracle_Rprev':rpO,'cross_native_oracle_cos':cos(dirsN[t],dirsO[t]),
                              'native_Omega_rad':cumangN,'oracle_Omega_rad':cumangO,'native_Omega_deg':cumangN*180/math.pi,'oracle_Omega_deg':cumangO*180/math.pi})
        prevN=dirsN[t]; prevO=dirsO[t]

        dn=training_row(cpsN[t],t,prev_t); do=training_row(cpsO[t],t,prev_t)
        disp=np.asarray(cpsO[t].WO,float)-W0; dispnorm=float(np.linalg.norm(disp)); proj=float(np.sum(disp*chord)/(chord2+EPS)); perp=float(np.linalg.norm(disp-proj*chord))
        delta_stats[t]={}
        row={'context':t,'time_fraction':t/1024.,'WO_oracle_displacement_norm':dispnorm,'WO_oracle_chord_projection':proj,'WO_oracle_perp_to_chord_norm':perp,
             'native_cum_online_nll':dn['cum_online_nll'],'oracle_cum_online_nll':do['cum_online_nll'],'native_recent_online_nll':dn['recent_online_nll'],'oracle_recent_online_nll':do['recent_online_nll'],
             'native_WO_path_length':dn['path_length_WO'],'oracle_WO_path_length':do['path_length_WO'],'native_probe_ce':ceN,'oracle_probe_ce':ceO,
             'native_global_js':jsN[t]['global_js_nats'],'oracle_global_js':jsO[t]['global_js_nats'],'native_spatial_js':jsN[t]['spatial_js_nats'],'oracle_spatial_js':jsO[t]['spatial_js_nats'],
             'native_pair_js':jsN[t]['pair_js_nats'],'oracle_pair_js':jsO[t]['pair_js_nats'],'native_token_marginal_js':markersN[t]['token_marginal_js_to_gt_nats'],'oracle_token_marginal_js':markersO[t]['token_marginal_js_to_gt_nats'],
             'native_position_entropy':markersN[t]['mean_position_token_entropy_nats'],'oracle_position_entropy':markersO[t]['mean_position_token_entropy_nats'],'native_P61':markersN[t]['final_P61'],'oracle_P61':markersO[t]['final_P61']}
        for m in ALL_METRICS:
            arr=np.asarray(gateO[t]['effects'][m],float)-np.asarray(gateN[t]['effects'][m],float)
            st=bootstrap(arr,840000+1000*ti+ALL_METRICS.index(m),a.bootstrap); delta_stats[t][m]=st
            row[f'delta_{m}']=st['mean']; row[f'delta_{m}_ci_lo']=st['ci95'][0]; row[f'delta_{m}_ci_hi']=st['ci95'][1]
            if m in PRIMARY: delta_arrays[m][t]=arr
            effect_npz[f'native_t{t}__effect__{m}']=gateN[t]['effects'][m]; effect_npz[f'oracle_t{t}__effect__{m}']=gateO[t]['effects'][m]
        row['proposal_switch_round1']=proposal_switch(gateO[t]['self_proposal'],gateN[t]['self_proposal'])[0]['token_fraction']
        row['proposal_switch_round2']=proposal_switch(gateO[t]['self_proposal'],gateN[t]['self_proposal'])[1]['token_fraction']
        row['proposal_switch_round3']=proposal_switch(gateO[t]['self_proposal'],gateN[t]['self_proposal'])[2]['token_fraction']
        for r in range(3):
            row[f'native_conf_r{r+1}']=markersN[t]['round_sampled_confidence'][r]; row[f'oracle_conf_r{r+1}']=markersO[t]['round_sampled_confidence'][r]
        curve_rows.append(row); prev_t=t

    # Common-background path and chord evaluations.
    fixed_base=cpsN[1024]; fixed_base_gate=gateN[1024]
    transplant_rows=[]; path_fixed_gates={}; chord_gates={}; matched_gates={}; path_chord_stats={}
    for ti,t in enumerate(CHECKPOINTS):
        print('D5D_TRANSPLANT',t,flush=True)
        matched=set_WO(cpsN[t],cpsO[t].WO); matched_gates[t]=gate_eval(f'matched_path_t{t}',matched,schedules,gt,shared)
        pathsys=set_WO(fixed_base,cpsO[t].WO); path_fixed_gates[t]=gate_eval(f'fixed_path_t{t}',pathsys,schedules,gt,shared)
        lam=CHORD_LAMBDAS[t]; Wc=W0+lam*chord; chordsys=set_WO(fixed_base,Wc); chord_gates[t]=gate_eval(f'fixed_chord_t{t}',chordsys,schedules,gt,shared)
        path_chord_stats[t]={}
        for m in ALL_METRICS:
            matched_arr=np.asarray(matched_gates[t]['effects'][m])-np.asarray(gateN[t]['effects'][m])
            path_arr=np.asarray(path_fixed_gates[t]['effects'][m])-np.asarray(fixed_base_gate['effects'][m])
            chord_arr=np.asarray(chord_gates[t]['effects'][m])-np.asarray(fixed_base_gate['effects'][m])
            pc_arr=np.asarray(path_fixed_gates[t]['effects'][m])-np.asarray(chord_gates[t]['effects'][m])
            sm=bootstrap(matched_arr,850000+10000*ti+100*ALL_METRICS.index(m),a.bootstrap)
            sp=bootstrap(path_arr,851000+10000*ti+100*ALL_METRICS.index(m),a.bootstrap)
            sc=bootstrap(chord_arr,852000+10000*ti+100*ALL_METRICS.index(m),a.bootstrap)
            sd=bootstrap(pc_arr,853000+10000*ti+100*ALL_METRICS.index(m),a.bootstrap)
            path_chord_stats[t][m]=sd
            transplant_rows.append({'context':t,'lambda':lam,'metric':m,'matched_path_mean':sm['mean'],'matched_path_ci_lo':sm['ci95'][0],'matched_path_ci_hi':sm['ci95'][1],
                                    'fixed_path_mean':sp['mean'],'fixed_path_ci_lo':sp['ci95'][0],'fixed_path_ci_hi':sp['ci95'][1],
                                    'fixed_chord_mean':sc['mean'],'fixed_chord_ci_lo':sc['ci95'][0],'fixed_chord_ci_hi':sc['ci95'][1],
                                    'path_minus_chord_mean':sd['mean'],'path_minus_chord_ci_lo':sd['ci95'][0],'path_minus_chord_ci_hi':sd['ci95'][1]})
            effect_npz[f'matched_path_t{t}__effect__{m}']=matched_gates[t]['effects'][m]
            effect_npz[f'fixed_path_t{t}__effect__{m}']=path_fixed_gates[t]['effects'][m]
            effect_npz[f'fixed_chord_t{t}__effect__{m}']=chord_gates[t]['effects'][m]

    # Adjacent increments and onset.
    adjacent={m:{} for m in PRIMARY}; onset={}
    for mi,m in enumerate(PRIMARY):
        first=None; vals=[]
        for t in CHECKPOINTS: vals.append(float(delta_stats[t][m]['mean']))
        for k in range(1,len(CHECKPOINTS)):
            t0,t1=CHECKPOINTS[k-1],CHECKPOINTS[k]; arr=delta_arrays[m][t1]-delta_arrays[m][t0]
            adjacent[m][t1]=bootstrap(arr,860000+1000*mi+k,a.bootstrap)
        for t in CHECKPOINTS[1:]:
            if delta_stats[t][m].get('ci_entirely_positive',False): first=t; break
        onset[m]=first

    terminal_harm_any=any(delta_stats[1024][m].get('ci_entirely_positive',False) for m in PRIMARY)
    monotone_flags={m:near_monotone([delta_stats[t][m]['mean'] for t in CHECKPOINTS]) for m in PRIMARY}
    intermediate=[64,128,256,512,768]
    path_sig=any(path_chord_stats[t][m].get('ci_entirely_positive',False) or path_chord_stats[t][m].get('ci_entirely_negative',False) for t in intermediate for m in PRIMARY)
    chord_terminal_harm=any(bootstrap(np.asarray(chord_gates[1024]['effects'][m])-np.asarray(fixed_base_gate['effects'][m]),870000+PRIMARY.index(m),a.bootstrap).get('ci_entirely_positive',False) for m in PRIMARY)
    D1=bool(terminal_harm_any and all(monotone_flags.values()) and (not path_sig) and chord_terminal_harm)

    positive_onsets=[t for t in onset.values() if t is not None]; earliest=min(positive_onsets) if positive_onsets else 1024
    rrows={r['context']:r for r in rotation_rows}; rot_before=min(float(rrows[t]['oracle_R0']) for t in CHECKPOINTS if t<=earliest)
    rot_substantial=bool(rot_before<0.90 or float(rrows[1024]['oracle_Omega_deg'])>45.0)
    D2=bool(path_sig and rot_substantial)

    threshold_flags={}
    for m in PRIMARY:
        tstar=onset[m]; ok=False
        if tstar is not None and tstar>0:
            idx=CHECKPOINTS.index(tstar); inc=adjacent[m][tstar]
            prior=[abs(adjacent[m][CHECKPOINTS[j]]['mean']) for j in range(1,idx) if CHECKPOINTS[j] in adjacent[m]]
            scale=float(np.median(prior)) if prior else 0.0
            preceding=all(not delta_stats[t][m].get('ci_entirely_positive',False) for t in CHECKPOINTS[1:idx])
            ok=bool(preceding and inc.get('ci_entirely_positive',False) and inc['mean']>2.0*max(scale,1e-12))
        threshold_flags[m]=ok
    D3=bool(any(threshold_flags.values()))
    supported=[k for k,v in {'D1':D1,'D2':D2,'D3':D3}.items() if v]
    case={'D1':'D1_SMOOTH_CUMULATIVE_DISPLACEMENT','D2':'D2_ROTATION_PATH_DEPENDENCE_SUPPORTED','D3':'D3_THRESHOLD_LIKE_TRANSITION'}.get(supported[0]) if len(supported)==1 else 'D_MIXED_OR_UNRESOLVED'

    write_csv(a.out/'D5D_CHECKPOINT_CURVES.csv',curve_rows); write_csv(a.out/'D5D_GRADIENT_ROTATION.csv',rotation_rows); write_csv(a.out/'D5D_TRANSPLANT_CHORD.csv',transplant_rows)
    np.savez_compressed(a.out/'D5D_EFFECT_ARRAYS.npz',**effect_npz)
    summary={
        'stage':'GATE_D5D_EXACT_WO_TRAJECTORY_EMERGENCE','classification':case,'source_D5B_run':39570284981,'source_D5C_run':39597941545,
        'learner_modified':False,'replayed_rules':['F_native','F_WO_oracle'],'checkpoints':list(CHECKPOINTS),'checkpoint_sha256':checkpoint_sha,
        'stream_exact_between_arms':True,'external_rng_exact_between_arms':True,'D5B_terminal_calibration':{'F_native':calN,'F_WO_oracle':calO},
        'probe_disjoint_from_gateA':True,'probe_mask_signature':mask_sig,'deployment_delta':jsonable(delta_stats),'onset':onset,'adjacent_increments':jsonable(adjacent),
        'gradient_rotation':jsonable(rotation_rows),'path_chord':jsonable(path_chord_stats),'masked_eval_native':jsonable(maskedN),'masked_eval_oracle':jsonable(maskedO),
        'rollout_js_native':jsonable(jsN),'rollout_js_oracle':jsonable(jsO),'gate_markers_native':jsonable(markersN),'gate_markers_oracle':jsonable(markersO),
        'decision':{'terminal_harm_any_primary':terminal_harm_any,'near_monotone_primary':monotone_flags,'intermediate_path_chord_significant_primary':path_sig,'chord_terminal_harm_primary':chord_terminal_harm,
                    'rotation_before_or_by_onset_min_R0':rot_before,'oracle_terminal_cumulative_rotation_deg':rrows[1024]['oracle_Omega_deg'],'rotation_substantial':rot_substantial,
                    'threshold_flags':threshold_flags,'D1':D1,'D2':D2,'D3':D3,'supported_classes':supported},
        'interpretation_boundary':'D5D is replay + checkpoint + readout only. Gradient rotation is not interpreted causally without functional path-vs-chord divergence.'
    }
    (a.out/'D5D_SUMMARY.json').write_text(json.dumps(jsonable(summary),indent=2),encoding='utf-8')
    manifest={'checkpoints':list(CHECKPOINTS),'chord_lambdas':CHORD_LAMBDAS,'primary':list(PRIMARY),'supportive':list(SUPPORTIVE),'bootstrap':a.bootstrap,'eval_images':a.eval_images,'gen_samples':a.gen_samples,'learner_modified':False}
    (a.out/'D5D_MANIFEST.json').write_text(json.dumps(jsonable(manifest),indent=2),encoding='utf-8')
    print('D5D_COMPLETE',json.dumps({'classification':case,'calibration_native':calN['pass'],'calibration_oracle':calO['pass'],'onset':onset,'decision':summary['decision']},indent=2),flush=True)

if __name__=='__main__':
    main()
