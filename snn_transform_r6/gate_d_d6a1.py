#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D6A.1 — state-dependent repair interaction audit.

Replay-only audit of two already-defined D6A trajectories:
  A = F_EM_bo_oracle
  B = F_EM_WO_zero_bo_oracle
with time-matched F_EM used only as a common transplant background.

No new learner rule is introduced.
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

import gate_d_d3e_a0 as d3a0
import gate_d_d3e_runtime as d3r
import gate_d_d5a as d5a
import gate_d_d5c as d5c
import gate_d_d6a as d6a
import gate_d_plasticity_audit as gd
import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
import r6b3_context_epoch_training as trainer
from maskgit_input_representation import build_input_embeddings
from r6b4_full_forward_plasticity import upgrade_r6b3_system

CHECKPOINTS=(0,64,128,256,512,768,1024)
A_LABEL='F_EM_bo_oracle'
B_LABEL='F_EM_WO_zero_bo_oracle'
F_LABEL='F_EM'
PRIMARY=('F61','D_non61')
ALL_METRICS=('F61','D_non61','C61','delta_P61','CE')
EPS=1e-30
TOL=1e-7
FIELD_DIVERGENCE_COS=0.95
DIRECT_CAPTURE=0.50
MEDIATED_CAPTURE=0.25


def clone(x):
    return pickle.loads(pickle.dumps(x,pickle.HIGHEST_PROTOCOL))


def locate(root:Path,name:str)->Path:
    return d3r.locate(root,name)


def write_csv(path:Path,rows):
    if not rows:
        path.write_text('',encoding='utf-8'); return
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore'); w.writeheader(); w.writerows(rows)


def clean_checkpoint(s):
    q=clone(s)
    # Diagnostic history is not needed for frozen readout and is large.
    if hasattr(q,'_d5b_rows'): q._d5b_rows=[]
    return q


def replay_checkpoints(start,label,tokens,indices,rng_state):
    s=d6a.make_d6a(start,label); rng=gd.clone_rng(rng_state)
    ck={0:clean_checkpoint(s)}; stream=[]
    for j,image_index in enumerate(indices,start=1):
        before=d3r.jstate(rng); d3r.reset_boundary(s,'F_EM')
        if d3r.jstate(rng)!=before: raise RuntimeError('F_EM reset consumed RNG')
        truth=np.asarray(tokens[int(image_index)],dtype=np.int64)
        rec,ev=d6a.run_context(s,j,int(image_index),truth,rng)
        stream.append(d6a.stream_row(j,int(image_index),ev))
        if j in CHECKPOINTS:
            ck[j]=clean_checkpoint(s)
            print('D6A1_REPLAY',json.dumps({'arm':label,'context':j,'mean_nll':float(rec['mean_nll'])}),flush=True)
    if set(ck)!=set(CHECKPOINTS): raise RuntimeError({'missing_checkpoints':sorted(set(CHECKPOINTS)-set(ck))})
    return ck,stream,d3r.jstate(rng)


def exact_bo_probe_gradient(system,contexts):
    s=clone(system); total=np.zeros_like(s.bo,dtype=np.float64); losses=[]; n_targets=0
    for c in contexts:
        arr=np.asarray(c['arr'],dtype=np.int64); masked=np.asarray(c['masked'],dtype=np.int64); truth=np.asarray(c['truth'],dtype=np.int64)
        s._current_arr=arr.copy(); xn,xraw=s.context_from_arr(arr); cache=s.prepare_mask_batch(xn,xraw,masked)
        cg=np.zeros_like(total); ls=[]
        for rec in cache:
            tgt=int(truth[int(rec['t'])]); o=d5a.exact_event_oracles(s,rec,tgt)
            cg+=np.asarray(o['bo'],dtype=np.float64)
            ls.append(-math.log(max(float(np.asarray(rec['p'])[tgt]),1e-300)))
        n=max(1,len(cache)); total+=cg/n; n_targets+=len(cache); losses.append(float(np.mean(ls)))
    g=total/max(1,len(contexts)); gn=float(np.linalg.norm(g))
    if gn<=EPS: raise RuntimeError('zero fixed-probe bo gradient')
    return g,{'gradient_norm':gn,'mean_probe_ce':float(np.mean(losses)),'contexts':int(len(contexts)),'targets':int(n_targets)}


def unit(v):
    v=np.asarray(v,dtype=np.float64); n=float(np.linalg.norm(v))
    return np.zeros_like(v) if n<=EPS else v/n


def cos(a,b):
    return float(gd.cosine(np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)))


def angle_deg(a,b):
    c=np.clip(cos(a,b),-1.0,1.0)
    if not np.isfinite(c): return float('nan')
    return float(np.degrees(np.arccos(c)))


def direction_capture(source,target,swapped):
    hs=unit(source); ht=unit(target); hw=unit(swapped); den=float(np.linalg.norm(hs-ht))
    if den<=EPS: return float('nan')
    return float(1.0-np.linalg.norm(hw-ht)/den)


def transplant_bo(background,donor):
    s=clone(background); s.bo=np.asarray(donor.bo,dtype=np.float64).copy(); return s


def swap_WO(recipient,donor):
    s=clone(recipient); s.WO=np.asarray(donor.WO,dtype=np.float64).copy(); return s


def contrast(ev1,ev0,m,seed,nboot):
    x=np.asarray(ev1['effects'][m],dtype=np.float64)-np.asarray(ev0['effects'][m],dtype=np.float64)
    return d6a.bootstrap_mean(x,seed,nboot)


def gate_calibration(ev,stored,prefix):
    d={'self_metrics_max_abs':d6a.nanaware_max_abs(ev['self_metrics'],stored[f'{prefix}__self_metrics']),
       'teacher_metrics_max_abs':d6a.nanaware_max_abs(ev['teacher_metrics'],stored[f'{prefix}__teacher_metrics']),
       'self_proposal_exact':bool(np.array_equal(ev['self_proposal'],stored[f'{prefix}__self_proposal'])),
       'teacher_proposal_exact':bool(np.array_equal(ev['teacher_proposal'],stored[f'{prefix}__teacher_proposal'])),
       'effects':{}}
    for m in ALL_METRICS:
        d['effects'][m]=d6a.nanaware_max_abs(ev['effects'][m],stored[f'{prefix}__effect__{m}'])
    d['pass']=bool(d['self_metrics_max_abs']<TOL and d['teacher_metrics_max_abs']<TOL and d['self_proposal_exact'] and d['teacher_proposal_exact'] and max(d['effects'].values())<TOL)
    return d


class D6ACompatUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='__main__' and name=='D6ASystem': return d6a.D6ASystem
        return super().find_class(module,name)


def load_d6a_terminal(path):
    with path.open('rb') as f: obj=D6ACompatUnpickler(f).load()
    if not isinstance(obj,dict) or 'system' not in obj: raise RuntimeError(f'bad D6A terminal schema: {path}')
    return obj['system']


def model_field_compare(a,b):
    keys=list(d3a0.MODEL_KEYS)
    for k in ('tok','pos','aux_lr_scale'):
        if hasattr(a,k) or hasattr(b,k): keys.append(k)
    out={'pass':True,'mismatches':[]}
    for k in keys:
        if not hasattr(a,k) or not hasattr(b,k):
            out['pass']=False; out['mismatches'].append({'field':k,'kind':'missing'}); continue
        x=getattr(a,k); y=getattr(b,k)
        if isinstance(x,np.ndarray) or isinstance(y,np.ndarray):
            ok=isinstance(x,np.ndarray) and isinstance(y,np.ndarray) and x.dtype==y.dtype and x.shape==y.shape and np.array_equal(x,y)
            if not ok:
                out['pass']=False; out['mismatches'].append({'field':k,'kind':'array','max_abs':d6a.nanaware_max_abs(x,y) if isinstance(x,np.ndarray) and isinstance(y,np.ndarray) else None})
        else:
            ok=(x==y)
            if not ok:
                out['pass']=False; out['mismatches'].append({'field':k,'kind':'scalar','a':repr(x),'b':repr(y)})
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True); ap.add_argument('--tokenizer',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--d6a-artifact',type=Path,required=True); ap.add_argument('--anchor-artifact',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--contexts',type=int,default=1024); ap.add_argument('--eval-images',type=int,default=256); ap.add_argument('--bootstrap',type=int,default=10000)
    ap.add_argument('--encode-batch',type=int,default=256); ap.add_argument('--seed',type=int,default=7311); ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    if a.contexts!=1024 or a.eval_images!=256: raise ValueError('formal D6A.1 requires 1024 contexts and 256 Gate-A images')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads); device=torch.device('cpu')

    # D6A prerequisite.
    d6s=json.loads(locate(a.d6a_artifact,'D6A_SUMMARY.json').read_text(encoding='utf-8'))
    if d6s.get('classification')!='R0_NO_REPAIR_SUFFICIENCY': raise RuntimeError('D6A prerequisite is not frozen R0')
    if not d6s.get('decision',{}).get('oracle_valid',False): raise RuntimeError('D6A bo oracle prerequisite invalid')
    stored=np.load(locate(a.d6a_artifact,'D6A_EFFECT_ARRAYS.npz'),allow_pickle=False)

    source,meta,train_rng,schedule=trainer.load_full_checkpoint(a.checkpoint)
    if abs(float(meta.get('context_epoch_equivalent',-1.0))-2.0)>1e-12: raise RuntimeError('not exact 2.0-epoch checkpoint')
    offset=int(schedule['offset']); indices=np.asarray(schedule['permutation'],dtype=np.int64)[offset:offset+a.contexts]; rng_state=pickle.loads(pickle.dumps(train_rng.bit_generator.state,pickle.HIGHEST_PROTOCOL))
    if len(indices)!=1024: raise RuntimeError('checkpoint schedule lacks 1024 contexts')

    xall,yall,_xte,_yte=core.load(a.data); x012,y012=core.subset012(xall,yall); xtr,_ytr,xva,_yva=core.split(x012,y012,.1,7)
    toknet,_=base.load_frozen_tokenizer(a.tokenizer,device); tr,_=base.encode_snn(toknet,xtr,device,a.encode_batch); va,_=base.encode_snn(toknet,xva,device,a.encode_batch)
    tok0,pos0,_=build_input_embeddings(toknet,source.c.D,grid=7)
    start=clone(source); upgrade_r6b3_system(start,tok0,pos0,aux_lr_scale=1.0)
    oracle_test=d5a.synthetic_oracle_self_test()
    if not oracle_test['pass']: raise RuntimeError({'oracle_self_test_failed':oracle_test})

    anchor=np.load(locate(a.anchor_artifact,'R6B4_NATIVE_GATE_A_EXECUTION_ANCHOR.npz'),allow_pickle=False)
    schedules=json.loads(locate(a.anchor_artifact,'HISTORICAL_R6B3_R3_T8_REFERENCE_TRACE.json').read_text(encoding='utf-8'))
    validation_ids=np.asarray(anchor['validation_ids'],dtype=np.int64); gt=np.asarray(anchor['gt_tokens'],dtype=np.int64); shared=np.asarray(anchor['shared_uniform'],dtype=np.float64)
    if not np.array_equal(np.asarray(va[validation_ids],dtype=np.int64),gt): raise RuntimeError('Gate-A validation tokens mismatch')

    ckA,stA,rA=replay_checkpoints(start,A_LABEL,tr,indices,rng_state)
    ckB,stB,rB=replay_checkpoints(start,B_LABEL,tr,indices,rng_state)
    ckF,stF,rF=replay_checkpoints(start,F_LABEL,tr,indices,rng_state)
    if stA!=stB or stA!=stF or rA!=rB or rA!=rF: raise RuntimeError('D6A.1 replay stream/RNG mismatch')

    # Terminal model-field replay certification against successful D6A artifact.
    oldA=load_d6a_terminal(locate(a.d6a_artifact,'D6A_F_EM_bo_oracle_terminal.pkl'))
    oldB=load_d6a_terminal(locate(a.d6a_artifact,'D6A_F_EM_WO_zero_bo_oracle_terminal.pkl'))
    oldF=load_d6a_terminal(locate(a.d6a_artifact,'D6A_F_EM_terminal.pkl'))
    field_cert={'A':model_field_compare(ckA[1024],oldA),'B':model_field_compare(ckB[1024],oldB),'F_EM':model_field_compare(ckF[1024],oldF)}
    if not all(x['pass'] for x in field_cert.values()): raise RuntimeError({'terminal_model_field_replay_failed':field_cert})

    # Fixed probe, identical deterministic construction to D5C.
    probe_ids=d5c.select_gradient_ids(len(va),validation_ids); probe_contexts,probe_mask_sig=d5c.build_gradient_contexts(start,va,probe_ids)
    probe_prov={'ids_sha256':d5c.arr_sha(probe_ids),'mask_signature_sha256':d5c.json_sha(probe_mask_sig),'images':int(len(probe_ids)),
                'gate_overlap':int(len(set(int(x) for x in probe_ids)&set(int(x) for x in validation_ids)))}
    if probe_prov['gate_overlap']!=0: raise RuntimeError('probe overlaps Gate-A')

    checkpoint_rows=[]; utility={}; full={}; grad={}; raw_npz={'validation_ids':validation_ids,'gt_tokens':gt.astype(np.int16),'probe_ids':probe_ids}
    prevA=None; prevB=None; g0A=None; g0B=None; omegaA=0.; omegaB=0.; bo0=np.asarray(start.bo,dtype=np.float64).copy(); bo_path_A=0.; bo_path_B=0.; prev_boA=bo0; prev_boB=bo0

    terminal_gate_cache={}
    for ci,t in enumerate(CHECKPOINTS):
        SA,SB,SF=ckA[t],ckB[t],ckF[t]

        # A. common-background bo utility, plus full-system alignment.
        TA=transplant_bo(SF,SA); TB=transplant_bo(SF,SB)
        evF=d6a.gate_eval(f'F_EM_t{t}',SF,schedules,gt,shared)
        evTA=d6a.gate_eval(f'boA_on_F_t{t}',TA,schedules,gt,shared)
        evTB=d6a.gate_eval(f'boB_on_F_t{t}',TB,schedules,gt,shared)
        evA=d6a.gate_eval(f'A_full_t{t}',SA,schedules,gt,shared)
        evB=d6a.gate_eval(f'B_full_t{t}',SB,schedules,gt,shared)
        terminal_gate_cache[t]={'F':evF,'TA':evTA,'TB':evTB,'A':evA,'B':evB}
        utility[str(t)]={'B_minus_A':{},'A_minus_F':{},'B_minus_F':{}}
        full[str(t)]={'B_minus_A':{}}
        for mi,m in enumerate(ALL_METRICS):
            utility[str(t)]['B_minus_A'][m]=contrast(evTB,evTA,m,1010000+1000*ci+10*mi,a.bootstrap)
            utility[str(t)]['A_minus_F'][m]=contrast(evTA,evF,m,1020000+1000*ci+10*mi,a.bootstrap)
            utility[str(t)]['B_minus_F'][m]=contrast(evTB,evF,m,1030000+1000*ci+10*mi,a.bootstrap)
            full[str(t)]['B_minus_A'][m]=contrast(evB,evA,m,1040000+1000*ci+10*mi,a.bootstrap)
            raw_npz[f't{t}__F__{m}']=np.asarray(evF['effects'][m]); raw_npz[f't{t}__boA_on_F__{m}']=np.asarray(evTA['effects'][m]); raw_npz[f't{t}__boB_on_F__{m}']=np.asarray(evTB['effects'][m]); raw_npz[f't{t}__A_full__{m}']=np.asarray(evA['effects'][m]); raw_npz[f't{t}__B_full__{m}']=np.asarray(evB['effects'][m])

        # B/C. fixed-probe own-state and hybrid WO-swap gradients.
        gA,metaA=exact_bo_probe_gradient(SA,probe_contexts); gB,metaB=exact_bo_probe_gradient(SB,probe_contexts)
        HAB=swap_WO(SA,SB); HBA=swap_WO(SB,SA)
        gAB,metaAB=exact_bo_probe_gradient(HAB,probe_contexts); gBA,metaBA=exact_bo_probe_gradient(HBA,probe_contexts)
        if g0A is None: g0A=np.array(gA,copy=True); g0B=np.array(gB,copy=True)
        prev_cos_A=float('nan') if prevA is None else cos(gA,prevA); prev_cos_B=float('nan') if prevB is None else cos(gB,prevB)
        if prevA is not None: omegaA+=angle_deg(gA,prevA); omegaB+=angle_deg(gB,prevB)
        capA=direction_capture(gA,gB,gAB); capB=direction_capture(gB,gA,gBA); cap_vals=[x for x in (capA,capB) if np.isfinite(x)]; cap_mean=float(np.mean(cap_vals)) if cap_vals else float('nan')
        grad[str(t)]={'C_AB':cos(gA,gB),'rho_AB':float(metaB['gradient_norm']/metaA['gradient_norm']),
                      'R_A0':cos(gA,g0A),'R_B0':cos(gB,g0B),'prev_cos_A':prev_cos_A,'prev_cos_B':prev_cos_B,
                      'Omega_A_deg':float(omegaA),'Omega_B_deg':float(omegaB),'norm_A':metaA['gradient_norm'],'norm_B':metaB['gradient_norm'],
                      'probe_ce_A':metaA['mean_probe_ce'],'probe_ce_B':metaB['mean_probe_ce'],
                      'A_swapBWO_cos_to_B':cos(gAB,gB),'B_swapAWO_cos_to_A':cos(gBA,gA),
                      'A_swapBWO_norm_ratio_to_B':float(metaAB['gradient_norm']/metaB['gradient_norm']),
                      'B_swapAWO_norm_ratio_to_A':float(metaBA['gradient_norm']/metaA['gradient_norm']),
                      'capture_A':capA,'capture_B':capB,'capture_mean':cap_mean}
        raw_npz[f't{t}__gA']=gA; raw_npz[f't{t}__gB']=gB; raw_npz[f't{t}__gA_swapBWO']=gAB; raw_npz[f't{t}__gB_swapAWO']=gBA
        prevA=np.array(gA,copy=True); prevB=np.array(gB,copy=True)

        # D. bo parameter trajectory.
        boA=np.asarray(SA.bo,dtype=np.float64); boB=np.asarray(SB.bo,dtype=np.float64)
        if t!=0:
            bo_path_A+=float(np.linalg.norm(boA-prev_boA)); bo_path_B+=float(np.linalg.norm(boB-prev_boB))
        da=boA-bo0; db=boB-bo0
        cb=cos(da,db) if float(np.linalg.norm(da))>EPS and float(np.linalg.norm(db))>EPS else float('nan')
        row={'checkpoint':int(t),'C_AB':grad[str(t)]['C_AB'],'rho_AB':grad[str(t)]['rho_AB'],'R_A0':grad[str(t)]['R_A0'],'R_B0':grad[str(t)]['R_B0'],
             'prev_cos_A':prev_cos_A,'prev_cos_B':prev_cos_B,'Omega_A_deg':omegaA,'Omega_B_deg':omegaB,'capture_A':capA,'capture_B':capB,'capture_mean':cap_mean,
             'bo_delta_cos':cb,'bo_AB_distance':float(np.linalg.norm(boA-boB)),'bo_A_displacement':float(np.linalg.norm(da)),'bo_B_displacement':float(np.linalg.norm(db)),
             'bo_A_checkpoint_path':bo_path_A,'bo_B_checkpoint_path':bo_path_B}
        for m in PRIMARY:
            row[f'bo_B_minus_A_{m}']=utility[str(t)]['B_minus_A'][m]['mean']; row[f'bo_A_minus_F_{m}']=utility[str(t)]['A_minus_F'][m]['mean']; row[f'bo_B_minus_F_{m}']=utility[str(t)]['B_minus_F'][m]['mean']; row[f'full_B_minus_A_{m}']=full[str(t)]['B_minus_A'][m]['mean']
        checkpoint_rows.append(row); prev_boA=np.array(boA,copy=True); prev_boB=np.array(boB,copy=True)
        print('D6A1_CHECKPOINT',json.dumps({'t':t,'C_AB':row['C_AB'],'capture_mean':cap_mean,'bo_B_minus_A_F61':row['bo_B_minus_A_F61'],'bo_B_minus_A_D':row['bo_B_minus_A_D_non61']}),flush=True)

    # Terminal replay must reproduce the frozen successful D6A functional arrays exactly/tightly.
    tgate=terminal_gate_cache[1024]
    calibration={'A':gate_calibration(tgate['A'],stored,A_LABEL),'B':gate_calibration(tgate['B'],stored,B_LABEL),'F_EM':gate_calibration(tgate['F'],stored,F_LABEL)}
    if not all(x['pass'] for x in calibration.values()): raise RuntimeError({'D6A_terminal_GateA_replay_failed':calibration})

    termU=utility['1024']; termG=grad['1024']
    bma=termU['B_minus_A']
    intrinsic_degradation=bool(all(bma[m]['mean'] is not None and bma[m]['mean']>=0 for m in PRIMARY) and any(bma[m].get('ci_entirely_positive',False) for m in PRIMARY))
    retained_benefit=bool(not any(bma[m].get('ci_entirely_positive',False) for m in PRIMARY) and any(termU['B_minus_F'][m].get('ci_entirely_negative',False) for m in PRIMARY))
    field_divergence=bool(float(termG['C_AB'])<FIELD_DIVERGENCE_COS)
    capture=float(termG['capture_mean'])
    if intrinsic_degradation and field_divergence and np.isfinite(capture) and capture>=DIRECT_CAPTURE:
        classification='I1_DIRECT_PATHWAY_COUPLING'
    elif intrinsic_degradation and field_divergence and np.isfinite(capture) and capture<=MEDIATED_CAPTURE:
        classification='I2_MEDIATED_NETWORK_COUPLING'
    elif (not intrinsic_degradation) and retained_benefit:
        classification='I3_NO_INTRINSIC_BO_REPAIR_DEGRADATION'
    else:
        classification='I4_MIXED_OR_INSUFFICIENT'

    write_csv(a.out/'D6A1_CHECKPOINT_SUMMARY.csv',checkpoint_rows)
    np.savez_compressed(a.out/'D6A1_ARRAYS.npz',**raw_npz)
    summary={'stage':'GATE_D6A1_STATE_DEPENDENT_REPAIR_INTERACTION','classification':classification,'source_D6A_run':39687028646,
             'learner_modified':False,'new_learning_rule_introduced':False,'arms_replayed':[A_LABEL,B_LABEL],'reference_replayed':F_LABEL,
             'checkpoints':list(CHECKPOINTS),'oracle_self_test':oracle_test,'stream_exact':True,'external_rng_exact':True,
             'terminal_model_field_replay':field_cert,'terminal_GateA_replay_calibration':calibration,'probe_provenance':probe_prov,
             'common_background_bo_utility':utility,'full_system_alignment':full,'fixed_probe_gradient_field':grad,
             'decision':{'terminal_intrinsic_boB_degradation':intrinsic_degradation,'terminal_boB_retained_benefit':retained_benefit,
                         'terminal_field_divergence':field_divergence,'terminal_capture_mean':capture,
                         'field_divergence_cos_threshold':FIELD_DIVERGENCE_COS,'direct_capture_threshold':DIRECT_CAPTURE,'mediated_capture_threshold':MEDIATED_CAPTURE},
             'interpretation_boundary':('I1/I2 require common-background evidence that bo_B intrinsic deployment utility is worse before gradient divergence is interpreted as a repair-destroying mechanism. '
                 'Own-state gradient divergence alone is insufficient. Hybrid WO swaps distinguish direct WO-state dependence from broader co-adaptive mediation. D6A R0 remains unchanged.')}
    (a.out/'D6A1_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    manifest={'classification':classification,'contexts':a.contexts,'eval_images':a.eval_images,'bootstrap':a.bootstrap,'probe_images':int(len(probe_ids)),
              'protected_baseline_modified':False,'no_PR_no_merge':True,'learner_modified':False,'D6A_verdict_changed':False}
    (a.out/'D6A1_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('D6A1_RESULT',json.dumps({'classification':classification,'decision':summary['decision'],'terminal_utility':termU,'terminal_gradient':termG},indent=2),flush=True)


if __name__=='__main__':
    main()
