#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D5B — norm-matched exact-local-descent versus rollout-utility test.

Five paired F_EM trajectories are replayed from the exact 2.0-epoch checkpoint.
All O-side E/M dynamics remain native.  Only the context-boundary parameter update
for WO and/or bo is replaced by an arm-specific norm-matched exact negative
masked-token gradient, or suppressed in the diagnostic O-zero arm.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch

import gate_d_d3e_runtime as d3r
import gate_d_d5a as d5a
import gate_d_plasticity_audit as gd
import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
import r6b3_context_epoch_training as trainer
import r6b3_decoder_operating_point_audit as coarse
import r6b3_decoder_spatial_refinement_audit as spatial
from gate_d_d3e_eval import BASE_GEN_SEED, eval_masked, gate_for_system, rollout_js
from maskgit_input_representation import build_input_embeddings
from r6b4_full_forward_plasticity import FullForwardPlasticMaskGITSystem, upgrade_r6b3_system

EPS=1e-30
CHECKPOINTS=(256,512,1024)
ARMS=('F_native','F_WO_oracle','F_bo_oracle','F_O_oracle','F_O_zero')
ORACLE_ARMS=('F_WO_oracle','F_bo_oracle','F_O_oracle')
PRIMARY=('F61','D_non61')
SUPPORTIVE=('C61','delta_P61')
ALL_METRICS=('F61','D_non61','C61','delta_P61','CE')
LOCAL_DESCENT_FRACTION_GATE=0.95
EXPECTED_CHECKPOINT_SHA256=d5a.EXPECTED_CHECKPOINT_SHA256


def clone(x):
    return pickle.loads(pickle.dumps(x,pickle.HIGHEST_PROTOCOL))


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


def bootstrap_mean(x,seed,nboot=10000,chunk=500):
    a=np.asarray(x,dtype=np.float64); a=a[np.isfinite(a)]
    if len(a)==0:
        return {'n':0,'mean':None,'ci95':[None,None],'fraction_positive':None,'fraction_negative':None}
    rng=np.random.default_rng(seed); vals=np.empty(nboot,dtype=np.float64); p=0
    while p<nboot:
        n=min(chunk,nboot-p); ix=rng.integers(0,len(a),size=(n,len(a))); vals[p:p+n]=a[ix].mean(axis=1); p+=n
    q=np.quantile(vals,[.025,.975])
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


def locate(root:Path,name:str)->Path:
    return d3r.locate(root,name)


def load_terminal(path:Path):
    with path.open('rb') as f: obj=pickle.load(f)
    if not isinstance(obj,dict) or 'system' not in obj: raise RuntimeError(f'bad terminal schema: {path}')
    return obj['system']


def loss_from_cached_O(system,events,WO,bo):
    """Fixed-context mean masked CE with all non-O forward parameters held pre-update."""
    vals=[]
    WO=np.asarray(WO,dtype=np.float64); bo=np.asarray(bo,dtype=np.float64)
    for e in events:
        t=int(e['t']); a=np.asarray(e['a'],dtype=np.float64); target=int(e['target'])
        z=np.asarray(system._current_xraw[t],dtype=np.float64)+bo+WO@a
        mu=float(np.mean(z)); inv=1.0/math.sqrt(float(np.mean((z-mu)**2))+system.eps2)
        zhat=(z-mu)*inv; u=system.gamma2*zhat+system.beta2
        h=np.maximum(0.0,system.W1@u+system.b1-system.theta)
        y=z+system.W2@h+system.b2
        vals.append(gd._loss_from_y(y,system.WR,target))
    return float(np.mean(vals)) if vals else float('nan')


def scaled_oracle(native_update,gradient):
    u=np.asarray(native_update,dtype=np.float64); g=np.asarray(gradient,dtype=np.float64)
    un=float(np.linalg.norm(u)); gn=float(np.linalg.norm(g))
    if gn<=EPS:
        return np.zeros_like(g),un,gn
    return (un/gn)*g,un,gn


class D5BSystem(FullForwardPlasticMaskGITSystem):
    """Intervention system; E/M dynamics stay native, only final O update is altered."""
    def d5b_init(self,mode:str):
        if mode not in ARMS: raise ValueError(mode)
        self._d5b_mode=str(mode); self._d5b_rows=[]; self._d5b_ctx=None
        self._d5b_GWO=None; self._d5b_Gbo=None; self._d5b_events=[]; self._d5b_order=[]

    def d5b_begin_context(self,*,context,image_index,arr,masked,mirror_seed):
        self._d5b_ctx={'context':int(context),'image_index':int(image_index),
                       'masked':[int(x) for x in masked],'mirror_seed':int(mirror_seed),
                       'arr':np.asarray(arr,dtype=np.int64).copy()}
        self._d5b_GWO=np.zeros_like(self.WO,dtype=np.float64)
        self._d5b_Gbo=np.zeros_like(self.bo,dtype=np.float64)
        self._d5b_events=[]; self._d5b_order=[]

    def accumulate_cached_target(self,rec,target,weight=1.0):
        p=super().accumulate_cached_target(rec,target,weight)
        if self._d5b_ctx is None: return p
        o=d5a.exact_event_oracles(self,rec,int(target))
        self._d5b_GWO+=np.asarray(o['WO'],dtype=np.float64)
        self._d5b_Gbo+=np.asarray(o['bo'],dtype=np.float64)
        self._d5b_events.append({'t':int(rec['t']),'a':np.asarray(rec['a'],dtype=np.float64).copy(),
                                 'target':int(target),'p_target':float(np.asarray(rec['p'])[int(target)])})
        self._d5b_order.append((len(self._d5b_order),int(rec['t']),int(target)))
        return p

    def apply_update(self):
        if self._d5b_ctx is None: raise AssertionError('D5B context not initialized')
        n=max(1,int(self.accum))
        WO0=np.array(self.WO,copy=True); bo0=np.array(self.bo,copy=True)
        uWO_native=self.lro*np.asarray(self.ao,dtype=np.float64)/n
        uWO_native*=self.maskO
        ubo_native=self.aux_lr_scale*self.lro*np.asarray(self.abo,dtype=np.float64)/n
        GWO=np.asarray(self._d5b_GWO,dtype=np.float64)/n
        Gbo=np.asarray(self._d5b_Gbo,dtype=np.float64)/n
        uWO_oracle,nWO,gWO=scaled_oracle(uWO_native,GWO); uWO_oracle*=self.maskO
        ubo_oracle,nbo,gbo=scaled_oracle(ubo_native,Gbo)

        L0=loss_from_cached_O(self,self._d5b_events,WO0,bo0)
        cached=np.mean([-math.log(max(e['p_target'],1e-300)) for e in self._d5b_events])
        probe={
            'loss0':L0,'cached_loss':float(cached),'loss_recompute_minus_cached':float(L0-cached),
            'native_WO':loss_from_cached_O(self,self._d5b_events,WO0+uWO_native,bo0)-L0,
            'oracle_WO':loss_from_cached_O(self,self._d5b_events,WO0+uWO_oracle,bo0)-L0,
            'native_bo':loss_from_cached_O(self,self._d5b_events,WO0,bo0+ubo_native)-L0,
            'oracle_bo':loss_from_cached_O(self,self._d5b_events,WO0,bo0+ubo_oracle)-L0,
            'native_O':loss_from_cached_O(self,self._d5b_events,WO0+uWO_native,bo0+ubo_native)-L0,
            'oracle_O':loss_from_cached_O(self,self._d5b_events,WO0+uWO_oracle,bo0+ubo_oracle)-L0,
        }

        super().apply_update()

        mode=self._d5b_mode
        if mode=='F_WO_oracle':
            self.WO=WO0+uWO_oracle; self.WO*=self.maskO
        elif mode=='F_bo_oracle':
            self.bo=bo0+ubo_oracle
        elif mode=='F_O_oracle':
            self.WO=WO0+uWO_oracle; self.WO*=self.maskO; self.bo=bo0+ubo_oracle
        elif mode=='F_O_zero':
            self.WO=WO0; self.bo=bo0
        elif mode!='F_native':
            raise ValueError(mode)

        actualWO=np.asarray(self.WO)-WO0; actualbo=np.asarray(self.bo)-bo0
        if mode in ('F_WO_oracle','F_O_oracle'):
            wo_norm_err=abs(float(np.linalg.norm(actualWO))-nWO)
        elif mode=='F_O_zero':
            wo_norm_err=float(np.linalg.norm(actualWO))
        else:
            wo_norm_err=abs(float(np.linalg.norm(actualWO))-nWO)
        if mode in ('F_bo_oracle','F_O_oracle'):
            bo_norm_err=abs(float(np.linalg.norm(actualbo))-nbo)
        elif mode=='F_O_zero':
            bo_norm_err=float(np.linalg.norm(actualbo))
        else:
            bo_norm_err=abs(float(np.linalg.norm(actualbo))-nbo)

        order_sha=hashlib.sha256(json.dumps(self._d5b_order,separators=(',',':')).encode()).hexdigest()
        row={'arm':mode,'context':self._d5b_ctx['context'],'image_index':self._d5b_ctx['image_index'],
             'nmask':n,'mirror_seed':self._d5b_ctx['mirror_seed'],'target_order_sha256':order_sha,
             'native_WO_norm':nWO,'native_bo_norm':nbo,'G_WO_norm':gWO,'G_bo_norm':gbo,
             'actual_WO_norm':float(np.linalg.norm(actualWO)),'actual_bo_norm':float(np.linalg.norm(actualbo)),
             'WO_norm_match_abs_error':float(wo_norm_err),'bo_norm_match_abs_error':float(bo_norm_err),
             'G_WO_zero':bool(gWO<=EPS),'G_bo_zero':bool(gbo<=EPS),**probe}
        self._d5b_rows.append(row)
        self._d5b_ctx=None; self._d5b_GWO=None; self._d5b_Gbo=None; self._d5b_events=[]; self._d5b_order=[]


def make_d5b(start,mode):
    s=clone(start); s.__class__=D5BSystem; s.d5b_init(mode); return s


def run_context_d5b(s,j,image_index,truth,rng):
    sm=gd.sample_mask_set(s.c.N,rng); masked=np.asarray(sm.masked_positions,dtype=np.int64)
    E=gd.exposure_subset(masked,j-1,s.c.N)
    arr=np.asarray(truth,dtype=np.int64).copy(); arr[masked]=core.MASK
    mirror_seed=int(rng.integers(1,2**31-1)); core.seed_numba(mirror_seed)
    s.d5b_begin_context(context=j,image_index=image_index,arr=arr,masked=masked,mirror_seed=mirror_seed)
    ones=np.ones(s.c.V,dtype=np.float64)
    rec=s.learn_mask_batch_arr(arr,masked,truth,ones,rng)
    row=s._d5b_rows[-1]; row['mean_nll']=float(rec['mean_nll']); row['E_n']=int(len(E))
    ev={'masked':masked.astype(int).tolist(),'E':E.astype(int).tolist(),'mirror_seed':mirror_seed,'nmask':int(len(masked)),
        'target_order_sha256':row['target_order_sha256']}
    return rec,ev


def stream_row(j,image_index,ev):
    return {'context':int(j),'image_index':int(image_index),'masked':ev['masked'],'E':ev['E'],
            'mirror_seed':int(ev['mirror_seed']),'target_order_sha256':str(ev['target_order_sha256'])}


def run_native_parity(start,tokens,indices,rng_state):
    plain=clone(start); inst=make_d5b(start,'F_native'); rp=gd.clone_rng(rng_state); ri=gd.clone_rng(rng_state)
    stream=[]; checkpoints={}
    for j,image_index in enumerate(indices,start=1):
        d3r.reset_boundary(plain,'F_EM'); d3r.reset_boundary(inst,'F_EM')
        truth=np.asarray(tokens[int(image_index)],dtype=np.int64)
        recp,evp=gd.run_one_context(plain,False,arm='T0',j=j,image_index=int(image_index),truth=truth,rng=rp)
        reci,evi=run_context_d5b(inst,j,int(image_index),truth,ri)
        if evp['masked']!=evi['masked'] or evp['E']!=evi['E'] or evp['mirror_seed']!=evi['mirror_seed']:
            raise RuntimeError(f'D5B F_native stream divergence at {j}')
        if d3r.jstate(rp)!=d3r.jstate(ri) or float(recp['mean_nll'])!=float(reci['mean_nll']):
            raise RuntimeError(f'D5B F_native RNG/NLL divergence at {j}')
        stream.append(stream_row(j,int(image_index),evi))
        if j in CHECKPOINTS:
            c=d3r.compare_native_fields_bitwise(plain,inst); c['external_rng_exact']=bool(d3r.jstate(rp)==d3r.jstate(ri)); c['pass']=bool(c['bitwise_pass'] and c['external_rng_exact']); checkpoints[str(j)]=c
            print('D5B_PARITY',json.dumps({'context':j,'pass':c['pass']}),flush=True)
    return inst,stream,d3r.jstate(ri),{'pass':bool(all(v['pass'] for v in checkpoints.values())),'checkpoints':checkpoints}


def run_arm(start,mode,tokens,indices,rng_state):
    s=make_d5b(start,mode); rng=gd.clone_rng(rng_state); stream=[]
    for j,image_index in enumerate(indices,start=1):
        before=d3r.jstate(rng); d3r.reset_boundary(s,'F_EM')
        if d3r.jstate(rng)!=before: raise RuntimeError('F_EM reset consumed RNG')
        truth=np.asarray(tokens[int(image_index)],dtype=np.int64)
        _rec,ev=run_context_d5b(s,j,int(image_index),truth,rng); stream.append(stream_row(j,int(image_index),ev))
        if j in CHECKPOINTS:
            print('D5B_PROGRESS',json.dumps({'arm':mode,'context':j,'mean_nll':s._d5b_rows[-1]['mean_nll']}),flush=True)
    return s,stream,d3r.jstate(rng)


def local_summary(rows,arm,seed0,nboot):
    out={'contexts':int(len(rows)),
         'mean_context_nll':float(np.mean([r['mean_nll'] for r in rows])),
         'sum_context_nll':float(np.sum([r['mean_nll'] for r in rows])),
         'event_weighted_nll':float(np.sum([r['mean_nll']*r['nmask'] for r in rows])/np.sum([r['nmask'] for r in rows])),
         'max_loss_recompute_abs_error':float(np.max(np.abs([r['loss_recompute_minus_cached'] for r in rows]))),
         'max_WO_norm_match_abs_error':float(np.max([r['WO_norm_match_abs_error'] for r in rows])),
         'max_bo_norm_match_abs_error':float(np.max([r['bo_norm_match_abs_error'] for r in rows])),
         'zero_G_WO_contexts':int(np.sum([r['G_WO_zero'] for r in rows])),
         'zero_G_bo_contexts':int(np.sum([r['G_bo_zero'] for r in rows]))}
    for i,key in enumerate(('WO','bo','O')):
        n=np.asarray([r[f'native_{key}'] for r in rows],float); o=np.asarray([r[f'oracle_{key}'] for r in rows],float)
        out[key]={'native_deltaL':bootstrap_mean(n,seed0+10*i,nboot),
                  'oracle_deltaL':bootstrap_mean(o,seed0+10*i+1,nboot),
                  'oracle_minus_native_deltaL':bootstrap_mean(o-n,seed0+10*i+2,nboot),
                  'oracle_fraction_deltaL_negative':float(np.mean(o<0))}
    active={'F_WO_oracle':'WO','F_bo_oracle':'bo','F_O_oracle':'O'}.get(arm)
    if active:
        q=out[active]
        ci=q['oracle_minus_native_deltaL']['ci95']; om=q['oracle_deltaL']['mean']
        out['active_component']=active
        out['finite_step_interpretability_pass']=bool(q['oracle_fraction_deltaL_negative']>LOCAL_DESCENT_FRACTION_GATE and om is not None and om<0 and ci[1]<0)
    else:
        out['active_component']=None; out['finite_step_interpretability_pass']=None
    return out


def gate_eval(name,s,schedules,gt,shared):
    e=gate_for_system(name,s,schedules,gt,shared)
    if not e['gates']['all_pass']: raise RuntimeError(f'Gate-A structure failed: {name}')
    return e


def effect_contrast(a,b,m):
    return np.asarray(a['effects'][m],float)-np.asarray(b['effects'][m],float)


def compare_gate_to_d3e_F(ev,d3e_npz,tol=1e-7):
    d={'self_metrics_max_abs':nanaware_max_abs(ev['self_metrics'],d3e_npz['F_EM__self_metrics']),
       'teacher_metrics_max_abs':nanaware_max_abs(ev['teacher_metrics'],d3e_npz['F_EM__teacher_metrics']),
       'self_proposal_exact':bool(np.array_equal(ev['self_proposal'],d3e_npz['F_EM__self_proposal'])),
       'teacher_proposal_exact':bool(np.array_equal(ev['teacher_proposal'],d3e_npz['F_EM__teacher_proposal'])),
       'effects':{}}
    for m in ALL_METRICS: d['effects'][m]=nanaware_max_abs(ev['effects'][m],d3e_npz[f'F_EM__effect__{m}'])
    d['pass']=bool(d['self_metrics_max_abs']<tol and d['teacher_metrics_max_abs']<tol and d['self_proposal_exact'] and d['teacher_proposal_exact'] and max(d['effects'].values())<tol)
    return d


def set_fields(dst,donor,fields):
    for k in fields: setattr(dst,k,np.array(getattr(donor,k),copy=True))


def common_background_systems(nbase,fbase,arms):
    oo=arms['F_O_oracle']; oz=arms['F_O_zero']; wo=arms['F_WO_oracle']; bo=arms['F_bo_oracle']
    out={}
    for bg,w2 in (('N',False),('W2',True)):
        def mk():
            s=clone(nbase)
            if w2: s.W2=np.array(fbase.W2,copy=True)
            return s
        out[f'CB_{bg}_Onative']=mk()
        s=mk(); set_fields(s,fbase,('WO','bo')); out[f'CB_{bg}_OF']=s
        s=mk(); set_fields(s,oo,('WO','bo')); out[f'CB_{bg}_Ooracle']=s
        s=mk(); set_fields(s,oz,('WO','bo')); out[f'CB_{bg}_Ozero']=s
        s=mk(); set_fields(s,wo,('WO',)); out[f'CB_{bg}_WOoracle']=s
        s=mk(); set_fields(s,bo,('bo',)); out[f'CB_{bg}_booracle']=s
    return out


def save_terminal(path,s,arm):
    with path.open('wb') as f: pickle.dump({'arm':arm,'system':s},f,pickle.HIGHEST_PROTOCOL)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True); ap.add_argument('--tokenizer',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--d5a-artifact',type=Path,required=True); ap.add_argument('--d3e-artifact',type=Path,required=True); ap.add_argument('--anchor-artifact',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--contexts',type=int,default=1024); ap.add_argument('--encode-batch',type=int,default=256)
    ap.add_argument('--eval-images',type=int,default=256); ap.add_argument('--gen-samples',type=int,default=256); ap.add_argument('--bootstrap',type=int,default=10000)
    ap.add_argument('--seed',type=int,default=7311); ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    if a.contexts!=1024 or a.eval_images!=256: raise ValueError('formal D5B requires 1024 contexts and 256 Gate-A images')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads); device=torch.device('cpu')

    d5a_summary=json.loads(locate(a.d5a_artifact,'D5A_SUMMARY.json').read_text(encoding='utf-8'))
    if not(d5a_summary['oracle_self_test']['pass'] and d5a_summary['same_run_instrumentation_parity']['N']['pass'] and d5a_summary['same_run_instrumentation_parity']['F_EM']['pass']):
        raise RuntimeError('D5A prerequisite not certified')
    st=d5a.synthetic_oracle_self_test()
    if not st['pass']: raise RuntimeError({'D5B_oracle_self_test_failed':st})

    checkpoint_sha=d3r.sha_file(a.checkpoint); source,meta,train_rng,schedule=trainer.load_full_checkpoint(a.checkpoint)
    if checkpoint_sha!=EXPECTED_CHECKPOINT_SHA256 or abs(float(meta.get('context_epoch_equivalent',-1))-2.)>1e-12: raise RuntimeError('D5B start checkpoint identity failed')
    offset=int(schedule['offset']); indices=np.asarray(schedule['permutation'],dtype=np.int64)[offset:offset+a.contexts]; rng_state=copy.deepcopy(train_rng.bit_generator.state)
    if len(indices)!=1024: raise RuntimeError('checkpoint schedule lacks 1024 contexts')

    xall,yall,_xte,_yte=core.load(a.data); x012,y012=core.subset012(xall,yall); xtr,_ytr,xva,_yva=core.split(x012,y012,.1,7)
    toknet,_=base.load_frozen_tokenizer(a.tokenizer,device); tr,_=base.encode_snn(toknet,xtr,device,a.encode_batch); va,_=base.encode_snn(toknet,xva,device,a.encode_batch)
    tok0,pos0,_=build_input_embeddings(toknet,source.c.D,grid=7)
    start=clone(source); upgrade_r6b3_system(start,tok0,pos0,aux_lr_scale=1.0)

    d3e_summary=json.loads(locate(a.d3e_artifact,'D3E_SUMMARY.json').read_text(encoding='utf-8'))
    if not d3e_summary.get('all_provenance_gates_pass',False): raise RuntimeError('D3E source provenance failed')
    d3e_npz=np.load(locate(a.d3e_artifact,'D3E_EFFECT_ARRAYS.npz'),allow_pickle=False)
    nbase=load_terminal(locate(a.d3e_artifact,'D3E_N_terminal.pkl')); fbase=load_terminal(locate(a.d3e_artifact,'D3E_F_EM_terminal.pkl'))
    anchor_npz=np.load(locate(a.anchor_artifact,'R6B4_NATIVE_GATE_A_EXECUTION_ANCHOR.npz'),allow_pickle=False)
    schedules=json.loads(locate(a.anchor_artifact,'HISTORICAL_R6B3_R3_T8_REFERENCE_TRACE.json').read_text(encoding='utf-8'))
    validation_ids=np.asarray(anchor_npz['validation_ids'],dtype=np.int64); gt=np.asarray(anchor_npz['gt_tokens'],dtype=np.int64); shared=np.asarray(anchor_npz['shared_uniform'],dtype=np.float64)
    if not np.array_equal(np.asarray(va[validation_ids],dtype=np.int64),gt): raise RuntimeError('Gate-A validation tokens do not match frozen anchor')

    fn,ref_stream,ref_rng,parity=run_native_parity(start,tr,indices,rng_state)
    if not parity['pass']: raise RuntimeError('D5B F_native instrumentation parity failed')
    arms={'F_native':fn}; streams={'F_native':ref_stream}; rng_after={'F_native':ref_rng}
    for mode in ARMS[1:]:
        s,strows,rstate=run_arm(start,mode,tr,indices,rng_state); arms[mode]=s; streams[mode]=strows; rng_after[mode]=rstate
        if strows!=ref_stream or rstate!=ref_rng: raise RuntimeError(f'D5B stream/RNG pairing failed for {mode}')

    local={}
    for i,mode in enumerate(ARMS):
        local[mode]=local_summary(arms[mode]._d5b_rows,mode,810000+100*i,a.bootstrap)
        if local[mode]['max_loss_recompute_abs_error']>2e-10: raise RuntimeError(f'fixed-context loss recomputation failed: {mode}')
        if local[mode]['max_WO_norm_match_abs_error']>2e-12 or local[mode]['max_bo_norm_match_abs_error']>2e-12: raise RuntimeError(f'O update norm/zero gate failed: {mode}')
        write_csv(a.out/f'D5B_{mode}_LOCAL_ROWS.csv',arms[mode]._d5b_rows)
        save_terminal(a.out/f'D5B_{mode}_terminal.pkl',arms[mode],mode)

    gate={k:gate_eval(k,s,schedules,gt,shared) for k,s in arms.items()}
    fcal=compare_gate_to_d3e_F(gate['F_native'],d3e_npz)
    if not fcal['pass']: raise RuntimeError({'F_native_D3E_F_EM_gate_calibration_failed':fcal})

    deployment={}
    for i,mode in enumerate(ARMS[1:]):
        deployment[mode]={}
        for j,m in enumerate(ALL_METRICS):
            deployment[mode][m]={
                'J_arm_minus_F_native':bootstrap_mean(effect_contrast(gate[mode],gate['F_native'],m),820000+100*i+j,a.bootstrap),
                'F_native_rescue_positive':bootstrap_mean(-effect_contrast(gate[mode],gate['F_native'],m),821000+100*i+j,a.bootstrap),
                'J_arm_minus_O_zero':bootstrap_mean(effect_contrast(gate[mode],gate['F_O_zero'],m),822000+100*i+j,a.bootstrap),
            }

    cb_systems=common_background_systems(nbase,fbase,arms)
    cb_gate={k:gate_eval(k,s,schedules,gt,shared) for k,s in cb_systems.items()}
    cb={}
    for bg in ('N','W2'):
        cb[bg]={}
        for donor in ('OF','Ooracle','Ozero','WOoracle','booracle'):
            label=f'CB_{bg}_{donor}'; base_label=f'CB_{bg}_Onative'; cb[bg][donor]={}
            for j,m in enumerate(ALL_METRICS):
                cb[bg][donor][m]=bootstrap_mean(effect_contrast(cb_gate[label],cb_gate[base_label],m),830000+(0 if bg=='N' else 10000)+100*('OF','Ooracle','Ozero','WOoracle','booracle').index(donor)+j,a.bootstrap)

    masked={k:eval_masked(s,va,a.seed+41000,a.eval_images) for k,s in arms.items()}
    _rc,real_p=coarse.probability_from_ids(va,source.c.V); _pc,real_pos_p=spatial.position_probabilities(va,source.c.V,source.c.N); real_pair=spatial.pair_reference(va,source.c.V,source.c.N,7)
    rollout={k:rollout_js(s,real_p,real_pos_p,real_pair,BASE_GEN_SEED,a.gen_samples) for k,s in arms.items()}

    og=local['F_O_oracle']['finite_step_interpretability_pass']
    rescue_any=any(deployment['F_O_oracle'][m]['F_native_rescue_positive']['ci_entirely_positive'] for m in PRIMARY)
    harm_zero_any=any(deployment['F_O_oracle'][m]['J_arm_minus_O_zero']['ci_entirely_positive'] for m in PRIMARY)
    common_harm_N=any(cb['N']['Ooracle'][m]['ci_entirely_positive'] for m in PRIMARY)
    common_harm_W2=any(cb['W2']['Ooracle'][m]['ci_entirely_positive'] for m in PRIMARY)
    common_harm=bool(common_harm_N and common_harm_W2)
    if not og:
        case='L4_FINITE_STEP_ORACLE_INCONCLUSIVE'
    elif rescue_any and (harm_zero_any or common_harm):
        case='L3_MIXED_RESIDUAL_CREDIT_ERROR_AND_OBJECTIVE_MISMATCH'
    elif rescue_any and not harm_zero_any and not common_harm:
        case='L2_RESIDUAL_MSDL_CREDIT_ERROR'
    elif harm_zero_any and common_harm:
        case='L1_STRONG_LOCAL_OBJECTIVE_DEPLOYMENT_MISMATCH'
    else:
        case='L_UNRESOLVED_REQUIRES_INTERPRETATION'

    npz={'validation_ids':validation_ids,'gt_tokens':gt.astype(np.int16),'shared_uniform':shared}
    for name,e in {**gate,**cb_gate}.items():
        npz[f'{name}__self_metrics']=e['self_metrics']; npz[f'{name}__teacher_metrics']=e['teacher_metrics']; npz[f'{name}__self_proposal']=e['self_proposal']; npz[f'{name}__teacher_proposal']=e['teacher_proposal']
        for m,v in e['effects'].items(): npz[f'{name}__effect__{m}']=v
    np.savez_compressed(a.out/'D5B_EFFECT_ARRAYS.npz',**npz)

    summary={'stage':'GATE_D5B_NORM_MATCHED_EXACT_LOCAL_DESCENT_VS_ROLLOUT_UTILITY','classification':case,
             'source_D5A_run':39505406348,'source_D3E_run':39454173518,
             'training_rule_modified':True,'modification_scope':'diagnostic O-parameter update direction/zero only; O-side E/M dynamics and all non-O learner rules remain native',
             'candidate_learner_claimed':False,'checkpoint_sha256':checkpoint_sha,'oracle_self_test':st,
             'F_native_same_run_plain_instrumented_parity':parity,'all_arm_stream_exact':True,'all_arm_external_rng_exact':True,
             'F_native_vs_frozen_D3E_F_EM_GateA_calibration':fcal,'local_objective':local,'deployment':deployment,
             'common_background_terminal_utility':cb,'secondary_masked':masked,'secondary_rollout':rollout,
             'decision_flags':{'joint_oracle_finite_step_pass':og,'oracle_rescue_any_primary':rescue_any,
                               'oracle_harm_vs_Ozero_any_primary':harm_zero_any,'common_background_oracle_harm_N':common_harm_N,
                               'common_background_oracle_harm_W2':common_harm_W2,'common_background_harm_both':common_harm},
             'interpretation_boundary':('L1 supports a pathway-local objective/deployment mismatch only if the norm-matched joint O oracle realizes better finite-step masked-token descent yet its O terminal state remains harmful in common backgrounds. This does not imply gradient descent is generally unsuitable for generation.')}
    (a.out/'D5B_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    manifest={'contexts':1024,'arms':list(ARMS),'F_EM_boundary_reset_all_arms':True,'O_EM_states_native_all_arms':True,
              'oracle_update_boundary':'context','oracle_gradient':'exact pre-update mean masked-token negative gradient, support-matched for WO',
              'norm_matching':'arm-specific native proposal, component-wise WO and bo','local_finite_step_fraction_gate':LOCAL_DESCENT_FRACTION_GATE,
              'gate_a_images':256,'gate_a_rounds':3,'bootstrap':a.bootstrap,'common_backgrounds':['N','W2^F']}
    (a.out/'D5B_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    compact={'classification':case,'parity':parity['pass'],'F_calibration':fcal['pass'],
             'local':{k:{'nll':v['mean_context_nll'],'finite_pass':v['finite_step_interpretability_pass'],
                         'WO':v['WO'],'bo':v['bo'],'O':v['O']} for k,v in local.items()},
             'primary_deployment':{k:{m:deployment[k][m] for m in PRIMARY} for k in deployment},
             'primary_common_background':{bg:{donor:{m:cb[bg][donor][m] for m in PRIMARY} for donor in ('OF','Ooracle','Ozero')} for bg in cb},
             'decision_flags':summary['decision_flags']}
    print('D5B_COMPLETE',json.dumps(compact,indent=2),flush=True)

if __name__=='__main__': main()
