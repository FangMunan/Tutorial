#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D6A — mechanism-derived pathway-specific repair sufficiency test.

Five paired trajectories from the exact 2.0-epoch checkpoint:
  N
  F_EM
  F_EM_WO_zero
  F_EM_bo_oracle
  F_EM_WO_zero_bo_oracle

D6A is a synthesis/upper-bound diagnostic, not a final learner claim.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch

import gate_d_d3e_runtime as d3r
import gate_d_d5a as d5a
import gate_d_d5b as d5b
import gate_d_plasticity_audit as gd
import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
import r6b3_context_epoch_training as trainer
import r6b3_decoder_operating_point_audit as coarse
import r6b3_decoder_spatial_refinement_audit as spatial
from gate_d_d3e_eval import BASE_GEN_SEED, eval_masked, gate_for_system, rollout_js
from maskgit_input_representation import build_input_embeddings
from r6b4_full_forward_plasticity import upgrade_r6b3_system

CHECKPOINTS=(256,512,1024)
ARMS=('N','F_EM','F_EM_WO_zero','F_EM_bo_oracle','F_EM_WO_zero_bo_oracle')
REPAIR_ARMS=('F_EM_WO_zero','F_EM_bo_oracle','F_EM_WO_zero_bo_oracle')
ORACLE_ARMS=('F_EM_bo_oracle','F_EM_WO_zero_bo_oracle')
PRIMARY=('F61','D_non61')
ALL_METRICS=('F61','D_non61','C61','delta_P61','CE')
MASK_RATIOS=('0.25','0.5','0.9','1.0')
JS_KEYS=('global_js_nats','spatial_js_nats','pair_js_nats')
EXPECTED_CHECKPOINT_SHA256=d5a.EXPECTED_CHECKPOINT_SHA256
EPS=1e-30


def clone(x):
    return pickle.loads(pickle.dumps(x,pickle.HIGHEST_PROTOCOL))


def locate(root:Path,name:str)->Path:
    return d3r.locate(root,name)


def nanaware_max_abs(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    if a.shape!=b.shape: return float('inf')
    if not np.array_equal(np.isnan(a),np.isnan(b)): return float('inf')
    if not np.array_equal(np.isposinf(a),np.isposinf(b)): return float('inf')
    if not np.array_equal(np.isneginf(a),np.isneginf(b)): return float('inf')
    good=np.isfinite(a)&np.isfinite(b)
    return 0.0 if not np.any(good) else float(np.max(np.abs(a[good]-b[good])))


class D6ASystem(d5b.D5BSystem):
    """Reuse finite-difference-certified D5B bo oracle; optionally suppress WO update only."""
    def d6a_init(self,label:str):
        if label not in ARMS: raise ValueError(label)
        self._d6a_label=label
        self._d6a_freeze_WO=label in ('F_EM_WO_zero','F_EM_WO_zero_bo_oracle')
        base_mode='F_bo_oracle' if label in ORACLE_ARMS else 'F_native'
        self.d5b_init(base_mode)

    def apply_update(self):
        WO0=np.array(self.WO,copy=True)
        n=max(1,int(self.accum))
        native_WO_proposal=self.lro*np.asarray(self.ao,dtype=np.float64)/n
        native_WO_proposal*=self.maskO
        super().apply_update()
        row=self._d5b_rows[-1]
        row['d6a_arm']=self._d6a_label
        row['WO_native_proposal_norm']=float(np.linalg.norm(native_WO_proposal))
        if self._d6a_freeze_WO:
            self.WO=WO0
        row['WO_actual_post_intervention_norm']=float(np.linalg.norm(np.asarray(self.WO)-WO0))
        row['WO_zero_abs_error']=float(np.linalg.norm(np.asarray(self.WO)-WO0)) if self._d6a_freeze_WO else None


def make_d6a(start,label):
    s=clone(start); s.__class__=D6ASystem; s.d6a_init(label); return s


def stream_row(j,image_index,ev):
    return {'context':int(j),'image_index':int(image_index),'masked':ev['masked'],'E':ev['E'],
            'mirror_seed':int(ev['mirror_seed']),'target_order_sha256':str(ev['target_order_sha256'])}


def run_context(s,j,image_index,truth,rng):
    rec,ev=d5b.run_context_d5b(s,j,image_index,truth,rng)
    return rec,ev


def maybe_reset(s,label):
    if label!='N':
        before=None
        d3r.reset_boundary(s,'F_EM')


def run_control_parity(start,label,tokens,indices,rng_state):
    """Plain native vs D6A instrumentation, with the arm's exact boundary rule."""
    if label not in ('N','F_EM'): raise ValueError(label)
    plain=clone(start); inst=make_d6a(start,label)
    rp=gd.clone_rng(rng_state); ri=gd.clone_rng(rng_state)
    stream=[]; checks={}
    for j,image_index in enumerate(indices,start=1):
        if label=='F_EM':
            b1=d3r.jstate(rp); b2=d3r.jstate(ri)
            d3r.reset_boundary(plain,'F_EM'); d3r.reset_boundary(inst,'F_EM')
            if d3r.jstate(rp)!=b1 or d3r.jstate(ri)!=b2: raise RuntimeError('F_EM reset consumed RNG')
        truth=np.asarray(tokens[int(image_index)],dtype=np.int64)
        recp,evp=gd.run_one_context(plain,False,arm='T0',j=j,image_index=int(image_index),truth=truth,rng=rp)
        reci,evi=run_context(inst,j,int(image_index),truth,ri)
        if evp['masked']!=evi['masked'] or evp['E']!=evi['E'] or evp['mirror_seed']!=evi['mirror_seed']:
            raise RuntimeError(f'D6A {label} stream divergence at {j}')
        if d3r.jstate(rp)!=d3r.jstate(ri) or float(recp['mean_nll'])!=float(reci['mean_nll']):
            raise RuntimeError(f'D6A {label} RNG/NLL divergence at {j}')
        stream.append(stream_row(j,int(image_index),evi))
        if j in CHECKPOINTS:
            c=d3r.compare_native_fields_bitwise(plain,inst)
            c['external_rng_exact']=bool(d3r.jstate(rp)==d3r.jstate(ri))
            c['pass']=bool(c['bitwise_pass'] and c['external_rng_exact'])
            checks[str(j)]=c
            print('D6A_PARITY',json.dumps({'arm':label,'context':j,'pass':c['pass']}),flush=True)
    return inst,stream,d3r.jstate(ri),{'pass':bool(all(v['pass'] for v in checks.values())),'checkpoints':checks}


def run_repair(start,label,tokens,indices,rng_state):
    s=make_d6a(start,label); rng=gd.clone_rng(rng_state); stream=[]
    for j,image_index in enumerate(indices,start=1):
        before=d3r.jstate(rng); d3r.reset_boundary(s,'F_EM')
        if d3r.jstate(rng)!=before: raise RuntimeError('F_EM reset consumed RNG')
        truth=np.asarray(tokens[int(image_index)],dtype=np.int64)
        rec,ev=run_context(s,j,int(image_index),truth,rng)
        stream.append(stream_row(j,int(image_index),ev))
        if j in CHECKPOINTS:
            print('D6A_PROGRESS',json.dumps({'arm':label,'context':j,'mean_nll':float(rec['mean_nll'])}),flush=True)
    return s,stream,d3r.jstate(rng)


def bootstrap_mean(x,seed,nboot=10000,chunk=500):
    return d5b.bootstrap_mean(x,seed,nboot,chunk)


def bo_oracle_summary(rows,seed,nboot):
    native=np.asarray([r['native_bo'] for r in rows],dtype=np.float64)
    oracle=np.asarray([r['oracle_bo'] for r in rows],dtype=np.float64)
    diff=oracle-native
    out={'contexts':int(len(rows)),
         'mean_context_nll':float(np.mean([r['mean_nll'] for r in rows])),
         'sum_context_nll':float(np.sum([r['mean_nll'] for r in rows])),
         'event_weighted_nll':float(np.sum([r['mean_nll']*r['nmask'] for r in rows])/np.sum([r['nmask'] for r in rows])),
         'native_bo_deltaL':bootstrap_mean(native,seed,nboot),
         'oracle_bo_deltaL':bootstrap_mean(oracle,seed+1,nboot),
         'oracle_minus_native_deltaL':bootstrap_mean(diff,seed+2,nboot),
         'oracle_fraction_deltaL_negative':float(np.mean(oracle<0)),
         'max_loss_recompute_abs_error':float(np.max(np.abs([r['loss_recompute_minus_cached'] for r in rows]))),
         'max_bo_norm_match_abs_error':float(np.max([r['bo_norm_match_abs_error'] for r in rows])),
         'max_WO_zero_abs_error':float(np.max([0.0 if r.get('WO_zero_abs_error') is None else r['WO_zero_abs_error'] for r in rows]))}
    ci=out['oracle_minus_native_deltaL']['ci95']; om=out['oracle_bo_deltaL']['mean']
    out['finite_step_interpretability_pass']=bool(out['oracle_fraction_deltaL_negative']>0.95 and om is not None and om<0 and ci[1]<0)
    return out


def gate_eval(name,s,schedules,gt,shared):
    e=gate_for_system(name,s,schedules,gt,shared)
    if not e['gates']['all_pass']: raise RuntimeError(f'Gate-A structure failed: {name}')
    return e


def effect_contrast(a,b,m):
    return np.asarray(a['effects'][m],dtype=np.float64)-np.asarray(b['effects'][m],dtype=np.float64)


def compare_gate_to_d3e(ev,d3e_npz,prefix,tol=1e-7):
    d={'self_metrics_max_abs':nanaware_max_abs(ev['self_metrics'],d3e_npz[f'{prefix}__self_metrics']),
       'teacher_metrics_max_abs':nanaware_max_abs(ev['teacher_metrics'],d3e_npz[f'{prefix}__teacher_metrics']),
       'self_proposal_exact':bool(np.array_equal(ev['self_proposal'],d3e_npz[f'{prefix}__self_proposal'])),
       'teacher_proposal_exact':bool(np.array_equal(ev['teacher_proposal'],d3e_npz[f'{prefix}__teacher_proposal'])),
       'effects':{}}
    for m in ALL_METRICS:
        d['effects'][m]=nanaware_max_abs(ev['effects'][m],d3e_npz[f'{prefix}__effect__{m}'])
    d['pass']=bool(d['self_metrics_max_abs']<tol and d['teacher_metrics_max_abs']<tol and
                   d['self_proposal_exact'] and d['teacher_proposal_exact'] and max(d['effects'].values())<tol)
    return d


def save_terminal(path,s,arm):
    with path.open('wb') as f: pickle.dump({'arm':arm,'system':s},f,pickle.HIGHEST_PROTOCOL)


def primary_contrasts(gate,nboot):
    out={}
    for ai,arm in enumerate(REPAIR_ARMS):
        out[arm]={}
        for ref in ('N','F_EM'):
            out[arm][ref]={}
            for mi,m in enumerate(ALL_METRICS):
                out[arm][ref][m]=bootstrap_mean(effect_contrast(gate[arm],gate[ref],m),910000+10000*ai+100*(0 if ref=='N' else 1)+mi,nboot)
    return out


def interaction_summary(gate,nboot):
    c='F_EM_WO_zero'; d='F_EM_bo_oracle'; e='F_EM_WO_zero_bo_oracle'; f='F_EM'
    out={}
    for mi,m in enumerate(ALL_METRICS):
        x=(np.asarray(gate[e]['effects'][m],float)-np.asarray(gate[c]['effects'][m],float)
           -np.asarray(gate[d]['effects'][m],float)+np.asarray(gate[f]['effects'][m],float))
        out[m]=bootstrap_mean(x,920000+mi,nboot)
    return out


def guardrails(masked,rollout):
    comb='F_EM_WO_zero_bo_oracle'
    mg={}; mg_pass=True
    for r in MASK_RATIOS:
        best=min(float(masked['N'][r]['ce']),float(masked['F_EM'][r]['ce']))
        val=float(masked[comb][r]['ce']); lim=best*1.01+1e-8
        mg[r]={'combined_ce':val,'best_baseline_ce':best,'limit':lim,'pass':bool(val<=lim)}
        mg_pass=mg_pass and mg[r]['pass']
    jg={}; jg_pass=True
    for k in JS_KEYS:
        best=min(float(rollout['N'][k]),float(rollout['F_EM'][k]))
        val=float(rollout[comb][k]); lim=best*1.02+1e-6
        jg[k]={'combined':val,'best_baseline':best,'limit':lim,'pass':bool(val<=lim)}
        jg_pass=jg_pass and jg[k]['pass']
    return {'masked_ce':mg,'masked_ce_all_pass':bool(mg_pass),'rollout_js':jg,'rollout_js_all_pass':bool(jg_pass)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True); ap.add_argument('--tokenizer',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--d5a-artifact',type=Path,required=True); ap.add_argument('--d5b-artifact',type=Path,required=True); ap.add_argument('--d5d-artifact',type=Path,required=True)
    ap.add_argument('--d3e-artifact',type=Path,required=True); ap.add_argument('--anchor-artifact',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--contexts',type=int,default=1024); ap.add_argument('--eval-images',type=int,default=256); ap.add_argument('--gen-samples',type=int,default=256)
    ap.add_argument('--encode-batch',type=int,default=256); ap.add_argument('--bootstrap',type=int,default=10000); ap.add_argument('--seed',type=int,default=7311); ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    if a.contexts!=1024 or a.eval_images!=256: raise ValueError('formal D6A requires 1024 contexts and 256 Gate-A images')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads); device=torch.device('cpu')

    # Frozen prerequisite chain.
    d5a_summary=json.loads(locate(a.d5a_artifact,'D5A_SUMMARY.json').read_text(encoding='utf-8'))
    if not(d5a_summary['oracle_self_test']['pass'] and d5a_summary['same_run_instrumentation_parity']['N']['pass'] and d5a_summary['same_run_instrumentation_parity']['F_EM']['pass']):
        raise RuntimeError('D5A prerequisite not certified')
    oracle_test=d5a.synthetic_oracle_self_test()
    if not oracle_test['pass']: raise RuntimeError({'D6A_oracle_self_test_failed':oracle_test})
    d5b_summary=json.loads(locate(a.d5b_artifact,'D5B_SUMMARY.json').read_text(encoding='utf-8'))
    if not d5b_summary.get('local_objective',{}).get('F_bo_oracle',{}).get('finite_step_interpretability_pass',False):
        raise RuntimeError('D5B bO oracle prerequisite not certified')
    d5d_summary=json.loads(locate(a.d5d_artifact,'D5D_SUMMARY.json').read_text(encoding='utf-8'))
    if d5d_summary.get('classification')!='D2_ROTATION_PATH_DEPENDENCE_SUPPORTED':
        raise RuntimeError('D5D WO trajectory prerequisite not certified')

    checkpoint_sha=d3r.sha_file(a.checkpoint); source,meta,train_rng,schedule=trainer.load_full_checkpoint(a.checkpoint)
    if checkpoint_sha!=EXPECTED_CHECKPOINT_SHA256 or abs(float(meta.get('context_epoch_equivalent',-1))-2.)>1e-12: raise RuntimeError('D6A start checkpoint identity failed')
    offset=int(schedule['offset']); indices=np.asarray(schedule['permutation'],dtype=np.int64)[offset:offset+a.contexts]; rng_state=copy.deepcopy(train_rng.bit_generator.state)
    if len(indices)!=1024: raise RuntimeError('checkpoint schedule lacks 1024 contexts')

    xall,yall,_xte,_yte=core.load(a.data); x012,y012=core.subset012(xall,yall); xtr,_ytr,xva,_yva=core.split(x012,y012,.1,7)
    toknet,_=base.load_frozen_tokenizer(a.tokenizer,device); tr,_=base.encode_snn(toknet,xtr,device,a.encode_batch); va,_=base.encode_snn(toknet,xva,device,a.encode_batch)
    tok0,pos0,_=build_input_embeddings(toknet,source.c.D,grid=7)
    start=clone(source); upgrade_r6b3_system(start,tok0,pos0,aux_lr_scale=1.0)

    d3e_summary=json.loads(locate(a.d3e_artifact,'D3E_SUMMARY.json').read_text(encoding='utf-8'))
    if not d3e_summary.get('all_provenance_gates_pass',False): raise RuntimeError('D3E provenance failed')
    d3e_npz=np.load(locate(a.d3e_artifact,'D3E_EFFECT_ARRAYS.npz'),allow_pickle=False)
    with locate(a.d3e_artifact,'D3E_N_terminal.pkl').open('rb') as f: d3e_N=pickle.load(f)['system']
    with locate(a.d3e_artifact,'D3E_F_EM_terminal.pkl').open('rb') as f: d3e_F=pickle.load(f)['system']

    anchor_npz=np.load(locate(a.anchor_artifact,'R6B4_NATIVE_GATE_A_EXECUTION_ANCHOR.npz'),allow_pickle=False)
    schedules=json.loads(locate(a.anchor_artifact,'HISTORICAL_R6B3_R3_T8_REFERENCE_TRACE.json').read_text(encoding='utf-8'))
    validation_ids=np.asarray(anchor_npz['validation_ids'],dtype=np.int64); gt=np.asarray(anchor_npz['gt_tokens'],dtype=np.int64); shared=np.asarray(anchor_npz['shared_uniform'],dtype=np.float64)
    if not np.array_equal(np.asarray(va[validation_ids],dtype=np.int64),gt): raise RuntimeError('Gate-A validation tokens do not match frozen anchor')

    n,streamN,rngN,parityN=run_control_parity(start,'N',tr,indices,rng_state)
    f,streamF,rngF,parityF=run_control_parity(start,'F_EM',tr,indices,rng_state)
    if not parityN['pass'] or not parityF['pass']: raise RuntimeError('D6A control instrumentation parity failed')
    if streamF!=streamN or rngF!=rngN: raise RuntimeError('N/F_EM stream pairing failed')
    terminal_bitwise={'N':d3r.compare_native_fields_bitwise(n,d3e_N),'F_EM':d3r.compare_native_fields_bitwise(f,d3e_F)}
    terminal_bitwise['N']['pass']=bool(terminal_bitwise['N']['bitwise_pass']); terminal_bitwise['F_EM']['pass']=bool(terminal_bitwise['F_EM']['bitwise_pass'])
    if not terminal_bitwise['N']['pass'] or not terminal_bitwise['F_EM']['pass']: raise RuntimeError('D6A control terminal does not reproduce D3E')

    arms={'N':n,'F_EM':f}; streams={'N':streamN,'F_EM':streamF}; rng_after={'N':rngN,'F_EM':rngF}
    for label in REPAIR_ARMS:
        s,strows,rs=run_repair(start,label,tr,indices,rng_state)
        if strows!=streamN or rs!=rngN: raise RuntimeError(f'D6A stream/RNG divergence: {label}')
        arms[label]=s; streams[label]=strows; rng_after[label]=rs

    # Audit update identity and finite-step validity.
    local={}
    for i,label in enumerate(ARMS):
        rows=arms[label]._d5b_rows
        if max(abs(float(r['loss_recompute_minus_cached'])) for r in rows)>2e-10: raise RuntimeError(f'loss recompute failed: {label}')
        if label in ORACLE_ARMS:
            local[label]=bo_oracle_summary(rows,930000+100*i,a.bootstrap)
            if local[label]['max_bo_norm_match_abs_error']>2e-12: raise RuntimeError(f'bo norm match failed: {label}')
        else:
            local[label]={'contexts':len(rows),'mean_context_nll':float(np.mean([r['mean_nll'] for r in rows])),
                          'sum_context_nll':float(np.sum([r['mean_nll'] for r in rows])),
                          'event_weighted_nll':float(np.sum([r['mean_nll']*r['nmask'] for r in rows])/np.sum([r['nmask'] for r in rows]))}
        if label in ('F_EM_WO_zero','F_EM_WO_zero_bo_oracle'):
            z=max(float(r['WO_zero_abs_error']) for r in rows)
            local[label]['max_WO_zero_abs_error']=z
            if z>1e-15: raise RuntimeError(f'WO zero update failed: {label}')
        d5b.write_csv(a.out/f'D6A_{label}_LOCAL_ROWS.csv',rows)
        save_terminal(a.out/f'D6A_{label}_terminal.pkl',arms[label],label)

    gate={k:gate_eval(k,s,schedules,gt,shared) for k,s in arms.items()}
    calN=compare_gate_to_d3e(gate['N'],d3e_npz,'N'); calF=compare_gate_to_d3e(gate['F_EM'],d3e_npz,'F_EM')
    if not calN['pass'] or not calF['pass']: raise RuntimeError({'D6A_D3E_calibration_failed':{'N':calN,'F_EM':calF}})

    contrasts=primary_contrasts(gate,a.bootstrap)
    interaction=interaction_summary(gate,a.bootstrap)
    masked={k:eval_masked(s,va,a.seed+61000,a.eval_images) for k,s in arms.items()}
    _rc,real_p=coarse.probability_from_ids(va,source.c.V); _pc,real_pos_p=spatial.position_probabilities(va,source.c.V,source.c.N); real_pair=spatial.pair_reference(va,source.c.V,source.c.N,7)
    rollout={k:rollout_js(s,real_p,real_pos_p,real_pair,BASE_GEN_SEED,a.gen_samples) for k,s in arms.items()}
    if not all(v.get('all_trace_audits_pass',False) for v in rollout.values()): raise RuntimeError('rollout trace audit failed')
    guards=guardrails(masked,rollout)

    comb='F_EM_WO_zero_bo_oracle'
    primary_flags={}
    for ref in ('N','F_EM'):
        primary_flags[ref]={m:bool(contrasts[comb][ref][m]['ci_entirely_negative']) for m in PRIMARY}
    primary_all=bool(all(primary_flags[r][m] for r in primary_flags for m in PRIMARY))
    primary_any_both_refs=bool(any(primary_flags['N'][m] and primary_flags['F_EM'][m] for m in PRIMARY))
    oracle_valid=bool(all(local[a0]['finite_step_interpretability_pass'] for a0 in ORACLE_ARMS))
    if not oracle_valid:
        classification='RX_ORACLE_INTERPRETATION_INVALID'
    elif primary_all and guards['rollout_js_all_pass'] and guards['masked_ce_all_pass']:
        classification='R1_STRONG_PATHWAY_SPECIFIC_REPAIR_SUFFICIENCY'
    elif primary_all:
        classification='R2_PRIMARY_REPAIR_WITH_GUARDRAIL_TENSION'
    elif primary_any_both_refs:
        classification='R3_PARTIAL_PATHWAY_REPAIR'
    else:
        classification='R0_NO_REPAIR_SUFFICIENCY'

    npz={'validation_ids':validation_ids,'gt_tokens':gt.astype(np.int16),'shared_uniform':shared}
    for name,e in gate.items():
        npz[f'{name}__self_metrics']=e['self_metrics']; npz[f'{name}__teacher_metrics']=e['teacher_metrics']; npz[f'{name}__self_proposal']=e['self_proposal']; npz[f'{name}__teacher_proposal']=e['teacher_proposal']
        for m,v in e['effects'].items(): npz[f'{name}__effect__{m}']=v
    np.savez_compressed(a.out/'D6A_EFFECT_ARRAYS.npz',**npz)

    summary={'stage':'GATE_D6A_MECHANISM_DERIVED_PATHWAY_SPECIFIC_REPAIR','classification':classification,
             'source_D3E_run':39454173518,'source_D5A_run':39505406348,'source_D5B_run':39570284981,'source_D5D_run':39641577692,
             'checkpoint_sha256':checkpoint_sha,'learner_modified':True,
             'modification_scope':'diagnostic synthesis only: F_EM W2 boundary cleaning, WO update suppression control, exact norm-matched bo direction upper bound',
             'candidate_final_learner_claimed':False,'oracle_self_test':oracle_test,
             'same_run_plain_instrumented_parity':{'N':parityN,'F_EM':parityF},'terminal_bitwise_vs_D3E':terminal_bitwise,
             'all_arm_stream_exact':True,'all_arm_external_rng_exact':True,
             'GateA_calibration_to_D3E':{'N':calN,'F_EM':calF},'local_training':local,
             'deployment_contrasts':contrasts,'component_interaction':interaction,
             'secondary_masked':masked,'secondary_rollout':rollout,'guardrails':guards,
             'decision':{'oracle_valid':oracle_valid,'combined_primary_flags':primary_flags,'combined_primary_all_pass':primary_all,
                         'combined_any_primary_pass_both_refs':primary_any_both_refs,
                         'masked_ce_guardrail_pass':guards['masked_ce_all_pass'],'rollout_js_guardrail_pass':guards['rollout_js_all_pass']},
             'interpretation_boundary':('A successful D6A demonstrates sufficiency of pathway-specific plasticity organization under a diagnostic upper-bound repair. '
                 'It does not establish WO freeze or exact bo gradient as the final local/dynamical learning rule; those remain D6B/D6C engineering targets.')}
    (a.out/'D6A_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    manifest={'arms':list(ARMS),'contexts':a.contexts,'eval_images':a.eval_images,'gen_samples':a.gen_samples,'bootstrap':a.bootstrap,
              'checkpoint_sha256':checkpoint_sha,'training_stream_contexts':len(streamN),'classification':classification,
              'no_PR_no_merge':True,'protected_baseline_modified':False}
    (a.out/'D6A_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('D6A_RESULT',json.dumps({'classification':classification,'primary':primary_flags,'guardrails':guards,'oracle_valid':oracle_valid},indent=2),flush=True)

if __name__=='__main__':
    main()
