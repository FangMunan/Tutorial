#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D1.5/D1.6 diagnostics for SNN-Transform R6B4.

D1.5: exact oracle-rotation baseline for native T0 W2 plasticity.
D1.6A: same-pre-state GT/T1/T2 counterfactual context audit.

Diagnostic only: oracle quantities never enter candidate learning.
"""
from __future__ import annotations
import argparse, copy, csv, json, os, pickle, random
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import gate_d_plasticity_audit as gd
import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
import r6b3_context_epoch_training as trainer
from maskgit_input_representation import build_input_embeddings
from maskgit_msdl_training import sample_mask_set
from r6b4_full_forward_plasticity import upgrade_r6b3_system

LAGS=(1,2,4,8,16); SKETCH_DIMS=(64,128,256); SKETCH_SEEDS=(31001,41001,51001); EPS=1e-30

def _countsketch_map(n,dim,seed):
    rng=np.random.default_rng(seed); bins=rng.integers(0,dim,size=n,dtype=np.int64); signs=np.where(rng.random(n)<.5,-1.,1.)
    return bins,signs

def _countsketch(x,mapping,dim):
    bins,signs=mapping; z=np.asarray(x,dtype=np.float64).reshape(-1)
    return np.bincount(bins,weights=signs*z,minlength=dim).astype(np.float64)

def _effdim(X):
    A=np.asarray(X,dtype=np.float64); A=A-A.mean(0,keepdims=True); s=np.linalg.svd(A,full_matrices=False,compute_uv=False); e=s*s
    return {'effective_dimension':float((e.sum()**2)/(np.sum(e*e)+EPS)),'energy_top1_fraction':float(e[0]/(e.sum()+EPS)),'energy_top4_fraction':float(e[:4].sum()/(e.sum()+EPS)),'singular_values_top16':s[:16].astype(float).tolist()}

class RotationAuditSystem(gd.AuditFullForwardPlasticMaskGITSystem):
    def rotation_init(self):
        self._rot_g_hist=[]; self._rot_u_hist=[]; self._rot_cg={l:[] for l in LAGS}; self._rot_cu={l:[] for l in LAGS}
        self._rot_oracle_sum=np.zeros_like(self.W2,dtype=np.float64); self._rot_oracle_path=0.; self._rot_parallel_path=0.; self._rot_perp_path=0.; self._rot_rows=[]
        self._robust_maps={}; self._robust_series={}; n=int(np.asarray(self.W2).size)
        for dim in SKETCH_DIMS:
            for seed in SKETCH_SEEDS:
                k=f'k{dim}_s{seed}'; self._robust_maps[k]=(dim,_countsketch_map(n,dim,seed)); self._robust_series[k]=[]
    def audit_finish_context(self):
        nmask=max(1,int(len(self._audit_ctx['masked']))); g=np.asarray(self._audit_G,dtype=np.float64)/nmask; u=np.asarray(self.W2,dtype=np.float64)-self._audit_w_before['W2']
        gn=float(np.linalg.norm(g)); un=float(np.linalg.norm(u)); dot=float(np.sum(u*g))
        for lag in LAGS:
            if len(self._rot_g_hist)>=lag:
                self._rot_cg[lag].append(gd.cosine(g,self._rot_g_hist[-lag])); self._rot_cu[lag].append(gd.cosine(u,self._rot_u_hist[-lag]))
        self._rot_g_hist.append(np.array(g,copy=True)); self._rot_u_hist.append(np.array(u,copy=True))
        if len(self._rot_g_hist)>max(LAGS): self._rot_g_hist.pop(0); self._rot_u_hist.pop(0)
        if gn>EPS:
            oracle_step=(un/gn)*g; upar=(dot/(gn*gn+EPS))*g
        else:
            oracle_step=np.zeros_like(g); upar=np.zeros_like(g)
        uperp=u-upar; self._rot_oracle_sum+=oracle_step; self._rot_oracle_path+=float(np.linalg.norm(oracle_step))
        parn=float(np.linalg.norm(upar)); perpn=float(np.linalg.norm(uperp)); self._rot_parallel_path+=parn; self._rot_perp_path+=perpn
        for k,(dim,mapping) in self._robust_maps.items(): self._robust_series[k].append(_countsketch(u,mapping,dim))
        self._rot_rows.append({'context':int(self._audit_ctx['context']),'u_norm':un,'g_norm':gn,'cosine':gd.cosine(u,g),'signed_descent':dot,'parallel_norm':parn,'perp_norm':perpn})
        super().audit_finish_context()
    def rotation_summary(self):
        native=self.audit_terminal_mobility()['W2']; oracle_net=float(np.linalg.norm(self._rot_oracle_sum)); t_oracle=float(self._rot_oracle_path/(oracle_net+EPS))
        rot={}
        for lag in LAGS:
            cg=np.asarray(self._rot_cg[lag]); cu=np.asarray(self._rot_cu[lag]); rot[str(lag)]={'C_g_mean':float(np.nanmean(cg)),'C_g_median':float(np.nanmedian(cg)),'C_u_mean':float(np.nanmean(cu)),'C_u_median':float(np.nanmedian(cu)),'N_pairs':int(np.sum(np.isfinite(cg)&np.isfinite(cu)))}
        robust={}
        for k,rows in self._robust_series.items():
            X=np.stack(rows); r=_effdim(X); A=X-X.mean(0,keepdims=True); U,s,_=np.linalg.svd(A,full_matrices=False); a1=U[:,0]*s[0]; r['pc1_adjacent_sign_flip_probability']=float(np.mean(a1[:-1]*a1[1:]<0)); robust[k]=r
        return {'native_W2':native,'oracle_matched':{'path_abs':float(self._rot_oracle_path),'net_abs':oracle_net,'tortuosity':t_oracle},'excess_tortuosity_native_over_oracle':float(native['tortuosity']/(t_oracle+EPS)),'parallel_path_abs':float(self._rot_parallel_path),'orthogonal_path_abs':float(self._rot_perp_path),'orthogonal_path_fraction':float(self._rot_perp_path/(self._rot_parallel_path+self._rot_perp_path+EPS)),'rotation':rot,'sketch_robustness':robust,'interpretation_boundary':'High native tortuosity is only excess temporal regulation if it substantially exceeds norm-matched oracle tortuosity; low sketch rank is observed geometry, not an invariant manifold.'}

def make_rotation_clone(start):
    s=pickle.loads(pickle.dumps(start,pickle.HIGHEST_PROTOCOL)); s.__class__=RotationAuditSystem; s.audit_init(sketch_dim=64); s.rotation_init(); return s

def preview_t1_event(system,truth,j,image_index,rng_state):
    rng=gd.clone_rng(rng_state); sm=sample_mask_set(system.c.N,rng); masked=np.asarray(sm.masked_positions,dtype=np.int64); E=gd.exposure_subset(masked,j-1,system.c.N)
    z,pp,ur=gd.model_exposure_tokens(system,truth,masked,E,j-1); gt=np.asarray(truth)[E]; corr=z==gt
    return {'context':int(j),'image_index':int(image_index),'masked':masked.astype(int).tolist(),'E':E.astype(int).tolist(),'tokens':z.astype(int).tolist(),'gt':gt.astype(int).tolist(),'correct':corr.astype(bool).tolist(),'uniform':ur.tolist(),'selected_prob':(pp[np.arange(len(E)),z].tolist() if len(E) else [])}

def write_csv(path,rows):
    if not rows: path.write_text('',encoding='utf-8'); return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()),extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def bootstrap_ci(x,stat='mean',B=3000,seed=77001):
    x=np.asarray(x,dtype=np.float64); x=x[np.isfinite(x)]
    if not len(x): return [float('nan'),float('nan')]
    rng=np.random.default_rng(seed); vals=np.empty(B)
    for b in range(B):
        z=x[rng.integers(0,len(x),len(x))]; vals[b]=np.mean(z) if stat=='mean' else np.median(z)
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]

def summarize_branch(vals):
    vals=np.asarray(vals,dtype=np.float64); return {'N':int(np.sum(np.isfinite(vals))),'mean':float(np.nanmean(vals)),'median':float(np.nanmedian(vals)),'p_negative':float(np.nanmean(vals<0)),'mean_95ci':bootstrap_ci(vals,'mean'),'median_95ci':bootstrap_ci(vals,'median')}

def summarize_d16(rows):
    out={'branches':{},'paired_differences':{}}
    for arm in ('GT','T1','T2'): out['branches'][arm]=summarize_branch(np.array([r[f'{arm}_cosine'] for r in rows]))
    for a,b in (('T1','GT'),('T2','GT'),('T1','T2')):
        d=np.array([r[f'{a}_cosine']-r[f'{b}_cosine'] for r in rows]); out['paired_differences'][f'{a}_minus_{b}']={'mean':float(np.nanmean(d)),'median':float(np.nanmedian(d)),'mean_95ci':bootstrap_ci(d,'mean'),'median_95ci':bootstrap_ci(d,'median'),'p_less_than_zero':float(np.nanmean(d<0))}
    return out

def load_start(args):
    device=torch.device('cpu'); source,meta,train_rng,schedule=trainer.load_full_checkpoint(args.checkpoint)
    if abs(float(meta.get('context_epoch_equivalent',-1.))-2.)>1e-12: raise ValueError('requires exact 2.0-epoch R6B3 checkpoint')
    offset=int(schedule.get('offset',0)); perm=np.asarray(schedule['permutation'],dtype=np.int64); indices=perm[offset:offset+args.contexts]; rng_state=copy.deepcopy(train_rng.bit_generator.state)
    xall,yall,_xte,_yte=core.load(args.data); x012,y012=core.subset012(xall,yall); xtr,_ytr,_xva,_yva=core.split(x012,y012,.1,7)
    tokenizer,tok_meta=base.load_frozen_tokenizer(args.tokenizer,device); tr,sp_tr=base.encode_snn(tokenizer,xtr,device,args.encode_batch); tok0,pos0,input_audit=build_input_embeddings(tokenizer,source.c.D,grid=7)
    start=pickle.loads(pickle.dumps(source,pickle.HIGHEST_PROTOCOL)); upgrade_r6b3_system(start,tok0,pos0,aux_lr_scale=1.0)
    return start,tr,indices,rng_state,meta,tok_meta,sp_tr,input_audit

def run_d15(start,tokens,indices,rng_state,out):
    s=make_rotation_clone(start); rng=gd.clone_rng(rng_state); t1_previews=[]
    for j,image_index in enumerate(indices,start=1):
        truth=np.asarray(tokens[int(image_index)],dtype=np.int64); pre_rng=copy.deepcopy(rng.bit_generator.state); prev=preview_t1_event(s,truth,j,int(image_index),pre_rng)
        _rec,ev0=gd.run_one_context(s,True,arm='T0',j=j,image_index=int(image_index),truth=truth,rng=rng)
        if ev0['masked']!=prev['masked'] or ev0['E']!=prev['E']: raise RuntimeError(f'preview/native mismatch {j}')
        t1_previews.append(prev)
        if j in gd.CHECKPOINTS or j==len(indices): print('D15_PROGRESS',json.dumps({'context':j,'W2_tortuosity':s.audit_terminal_mobility()['W2']['tortuosity']}),flush=True)
    plan,t2_audit=gd.make_t2_plan(t1_previews); summary=s.rotation_summary(); write_csv(out/'D15_CONTEXT_ROWS.csv',s._rot_rows); (out/'D15_ORACLE_ROTATION.json').write_text(json.dumps(summary,indent=2)); (out/'D16_T1_PREVIEW_EVENTS.json').write_text(json.dumps(t1_previews)); (out/'D16_T2_PLAN.json').write_text(json.dumps({'audit':t2_audit,'plan':plan})); return t1_previews,plan,t2_audit,summary

def branch_one(base_state,truth,j,image_index,rng_state,arm,plan):
    b=gd.make_audit_clone(base_state,sketch_dim=8); brng=gd.clone_rng(rng_state); run_arm='T0' if arm=='GT' else arm
    _rec,ev=gd.run_one_context(b,True,arm=run_arm,j=j,image_index=image_index,truth=truth,rng=brng,t2_plan=plan if arm=='T2' else None); row=b._audit_context_rows[-1]
    return b,ev,{'cosine':float(row['context_cosine']),'signed_descent':float(row['context_signed_descent']),'efficiency':float(row['context_efficiency']),'u_norm':float(row['w2_actual_step_norm']),'g_norm':float(row['w2_oracle_mean_norm'])}

def run_d16(start,tokens,indices,rng_state,t1_previews,plan,out):
    base_state=pickle.loads(pickle.dumps(start,pickle.HIGHEST_PROTOCOL)); rng=gd.clone_rng(rng_state); rows=[]
    for j,image_index in enumerate(indices,start=1):
        truth=np.asarray(tokens[int(image_index)],dtype=np.int64); ctx_rng_state=copy.deepcopy(rng.bit_generator.state); br={}; evs={}; systems={}
        for arm in ('GT','T1','T2'):
            b,ev,met=branch_one(base_state,truth,j,int(image_index),ctx_rng_state,arm,plan); systems[arm]=b; evs[arm]=ev; br[arm]=met
        sig=[(evs[a]['masked'],evs[a]['E'],evs[a]['mirror_seed']) for a in ('GT','T1','T2')]
        if not(sig[0]==sig[1]==sig[2]): raise RuntimeError(f'branch stream mismatch {j}')
        if evs['T1'].get('tokens',[])!=t1_previews[j-1].get('tokens',[]): raise RuntimeError(f'T1 preview mismatch {j}')
        _rec_native,ev_native=gd.run_one_context(base_state,False,arm='T0',j=j,image_index=int(image_index),truth=truth,rng=rng)
        if (ev_native['masked'],ev_native['E'],ev_native['mirror_seed'])!=sig[0]: raise RuntimeError(f'native/GT stream mismatch {j}')
        if not gd.compare_native_states(base_state,systems['GT'])['bitwise_pass']: raise RuntimeError(f'GT/native state mismatch {j}')
        row={'context':int(j),'image_index':int(image_index),'nmask':int(ev_native['nmask']),'exposure_n':int(len(ev_native['E']))}
        for arm in ('GT','T1','T2'):
            for k,v in br[arm].items(): row[f'{arm}_{k}']=v
        row['T1_minus_GT_cosine']=row['T1_cosine']-row['GT_cosine']; row['T2_minus_GT_cosine']=row['T2_cosine']-row['GT_cosine']; row['T1_minus_T2_cosine']=row['T1_cosine']-row['T2_cosine']; rows.append(row)
        if j in gd.CHECKPOINTS or j==len(indices): print('D16_PROGRESS',json.dumps({'context':j,'GT':row['GT_cosine'],'T1':row['T1_cosine'],'T2':row['T2_cosine']}),flush=True)
    write_csv(out/'D16_SAME_STATE_CONTEXTS.csv',rows); summary=summarize_d16(rows); summary['causal_boundary']='GT/T1/T2 start from identical pre-context parameter, persistent E/M and RNG state; only visible context differs. Native GT alone advances between contexts.'; (out/'D16_SAME_STATE_SUMMARY.json').write_text(json.dumps(summary,indent=2)); return summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--tokenizer',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--contexts',type=int,default=1024); ap.add_argument('--encode-batch',type=int,default=256); ap.add_argument('--seed',type=int,default=7311); ap.add_argument('--threads',type=int,default=4); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads)
    st=gd.oracle_self_test();
    if not st['pass']: raise RuntimeError(st)
    start,tokens,indices,rng_state,meta,tok_meta,sp_tr,input_audit=load_start(a); t1,plan,t2audit,d15=run_d15(start,tokens,indices,rng_state,a.out); d16=run_d16(start,tokens,indices,rng_state,t1,plan,a.out)
    manifest={'stage':'GATE_D15_D16_ORACLE_ROTATION_AND_SAME_STATE_CONTEXT_AUDIT','contexts':int(a.contexts),'seed':int(a.seed),'source_context_epoch':float(meta['context_epoch_equivalent']),'gradient_oracle_enters_candidate_learning':False,'native_training_rule_changed':False,'D15_excess_tortuosity':d15['excess_tortuosity_native_over_oracle'],'T2_audit':t2audit,'github_sha':os.environ.get('GITHUB_SHA'),'scientific_boundary':'D1.5 distinguishes objective-gradient rotation from excess native cancelling motion; D1.6 tests immediate causal visible-context effects on W2 credit. Neither establishes an optimal learning-rate hierarchy or spectral submanifold.'}
    (a.out/'D15_D16_MANIFEST.json').write_text(json.dumps(manifest,indent=2)); print('D15_D16_COMPLETE',json.dumps({'excess_tortuosity':manifest['D15_excess_tortuosity'],'D16_branch_medians':{k:v['median'] for k,v in d16['branches'].items()},'T2_audit':t2audit},indent=2),flush=True)
if __name__=='__main__': main()
