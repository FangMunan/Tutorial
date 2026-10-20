#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D3 — temporal credit-memory matching audit for SNN-Transform R6B4.

D3A: observational temporal kernel / lag analysis of native W2 updates versus
     exact instantaneous W2 descent oracles.
D3B: cloned-state W2-direct target-order sensitivity with reciprocal result and
     forward cache frozen.
D3C: cloned-state E2/M2 persistence decomposition, separating within-context
     carry from cross-context inherited state.
D3D: frozen-parameter future-context oracle at H={1,2,4,8}; no intermediate
     learning and no BPTT.

Diagnostic only: counterfactual replay never enters the native candidate learner.
"""
from __future__ import annotations
import argparse, copy, csv, json, math, os, pickle, random
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np
import torch
import gate_d_plasticity_audit as gd
import gate_d_d15_d16 as d15
import p3i_cb_exact_msdl as core
from maskgit_msdl_training import sample_mask_set

EPS=1e-30
LAGS=(0,1,2,4,8)
FIR_MAX_LAG=8
HORIZONS=(1,2,4,8)
REPLAY_TOL=2e-14
ORACLE_TOL=2e-14
D3_SKETCH_DIM=256
D3_SKETCH_SEED=620261020
PERSISTENCE_MODES=('native','no_E_within','no_M_within','no_EM_within','fresh_E_context','fresh_M_context','fresh_EM_context')

def write_csv(path:Path, rows:List[Dict]):
    if not rows:
        path.write_text('',encoding='utf-8'); return
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); keys.append(k)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def bootstrap_ci(x,stat='mean',B=3000,seed=93001):
    a=np.asarray(x,dtype=np.float64); a=a[np.isfinite(a)]
    if len(a)==0: return [float('nan'),float('nan')]
    rng=np.random.default_rng(seed); z=np.empty(B,dtype=np.float64)
    for b in range(B):
        q=a[rng.integers(0,len(a),len(a))]; z[b]=np.mean(q) if stat=='mean' else np.median(q)
    return [float(np.quantile(z,.025)),float(np.quantile(z,.975))]

def summarize_scalar(x,seed=93001):
    a=np.asarray(x,dtype=np.float64); a=a[np.isfinite(a)]
    if len(a)==0:
        return {'N':0,'mean':float('nan'),'median':float('nan'),'mean_95ci':[float('nan'),float('nan')],'median_95ci':[float('nan'),float('nan')]}
    return {'N':int(len(a)),'mean':float(np.mean(a)),'median':float(np.median(a)),
            'mean_95ci':bootstrap_ci(a,'mean',seed=seed),'median_95ci':bootstrap_ci(a,'median',seed=seed+1),
            'fraction_positive':float(np.mean(a>0)),'fraction_negative':float(np.mean(a<0))}

def _countsketch_map(n,dim,seed):
    rng=np.random.default_rng(seed); bins=rng.integers(0,dim,size=n,dtype=np.int64); signs=np.where(rng.random(n)<.5,-1.,1.).astype(np.float64); return bins,signs

def _countsketch(x,mapping,dim):
    bins,signs=mapping; z=np.asarray(x,dtype=np.float64).reshape(-1); return np.bincount(bins,weights=signs*z,minlength=dim).astype(np.float64)

def _r2(y,pred):
    y=np.asarray(y,dtype=np.float64); pred=np.asarray(pred,dtype=np.float64); den=float(np.sum((y-np.mean(y))**2)); return float(1.-np.sum((y-pred)**2)/(den+EPS))

def _fit_linear(Xtr,ytr,Xte,yte):
    Xtr=np.asarray(Xtr,dtype=np.float64); Xte=np.asarray(Xte,dtype=np.float64); ytr=np.asarray(ytr,dtype=np.float64); yte=np.asarray(yte,dtype=np.float64)
    A=np.column_stack([np.ones(len(Xtr)),Xtr]); coef,*_=np.linalg.lstsq(A,ytr,rcond=None); pred=np.column_stack([np.ones(len(Xte)),Xte])@coef; return coef,_r2(yte,pred)

class D3AuditSystem(gd.AuditFullForwardPlasticMaskGITSystem):
    def d3_init(self):
        n=int(np.asarray(self.W2).size); self._audit_d3_map=_countsketch_map(n,D3_SKETCH_DIM,D3_SKETCH_SEED)
        self._audit_d3_u_sketch=[]; self._audit_d3_g_sketch=[]; self._audit_d3_rows=[]; self._audit_d3_u_hist=[]; self._audit_d3_g_hist=[]
        self._audit_d3_last_u=None; self._audit_d3_last_g=None
    def audit_finish_context(self):
        nmask=max(1,int(len(self._audit_ctx['masked']))); g=np.asarray(self._audit_G,dtype=np.float64)/nmask; u=np.asarray(self.W2,dtype=np.float64)-self._audit_w_before['W2']
        row={'context':int(self._audit_ctx['context']),'u_norm':float(np.linalg.norm(u)),'g_norm':float(np.linalg.norm(g)),'C_ug_0':gd.cosine(u,g)}
        for lag in (1,2,4,8):
            if len(self._audit_d3_g_hist)>=lag:
                row[f'C_ug_{lag}']=gd.cosine(u,self._audit_d3_g_hist[-lag]); row[f'C_gg_{lag}']=gd.cosine(g,self._audit_d3_g_hist[-lag]); row[f'C_uu_{lag}']=gd.cosine(u,self._audit_d3_u_hist[-lag])
            else:
                row[f'C_ug_{lag}']=float('nan'); row[f'C_gg_{lag}']=float('nan'); row[f'C_uu_{lag}']=float('nan')
        self._audit_d3_rows.append(row); self._audit_d3_u_sketch.append(_countsketch(u,self._audit_d3_map,D3_SKETCH_DIM)); self._audit_d3_g_sketch.append(_countsketch(g,self._audit_d3_map,D3_SKETCH_DIM))
        self._audit_d3_u_hist.append(np.array(u,copy=True)); self._audit_d3_g_hist.append(np.array(g,copy=True))
        if len(self._audit_d3_u_hist)>8: self._audit_d3_u_hist.pop(0); self._audit_d3_g_hist.pop(0)
        self._audit_d3_last_u=np.array(u,copy=True); self._audit_d3_last_g=np.array(g,copy=True)
        super().audit_finish_context()
    def d3_temporal_summary(self):
        rows=self._audit_d3_rows; lag_summary={'0':{'C_ug':summarize_scalar([r['C_ug_0'] for r in rows],seed=94000)}}
        for lag in (1,2,4,8):
            lag_summary[str(lag)]={'C_ug':summarize_scalar([r[f'C_ug_{lag}'] for r in rows],seed=94000+lag),
                                   'C_gg':summarize_scalar([r[f'C_gg_{lag}'] for r in rows],seed=95000+lag),
                                   'C_uu':summarize_scalar([r[f'C_uu_{lag}'] for r in rows],seed=96000+lag)}
        U=np.asarray(self._audit_d3_u_sketch,dtype=np.float64); G=np.asarray(self._audit_d3_g_sketch,dtype=np.float64); Uc=U-U.mean(0,keepdims=True); Gc=G-G.mean(0,keepdims=True)
        _Us,s,Vt=np.linalg.svd(Uc,full_matrices=False); v1=Vt[0]; au=Uc@v1; ag=Gc@v1; energy=s*s
        t0=FIR_MAX_LAG; y=au[t0:]; Xfull=np.column_stack([ag[t0-lag:len(ag)-lag] for lag in range(FIR_MAX_LAG+1)]); X0=ag[t0:,None]
        cut=max(1,int(math.floor(.70*len(y)))); cut=min(cut,len(y)-1); cf,r2f=_fit_linear(Xfull[:cut],y[:cut],Xfull[cut:],y[cut:]); c0,r20=_fit_linear(X0[:cut],y[:cut],X0[cut:],y[cut:])
        return {'lags':lag_summary,'dominant_mode':{'sketch_dim':D3_SKETCH_DIM,'sketch_seed':D3_SKETCH_SEED,
                'native_PC1_energy_fraction':float(energy[0]/(np.sum(energy)+EPS)),
                'native_adjacent_sign_flip_probability':float(np.mean(au[:-1]*au[1:]<0)),
                'oracle_adjacent_sign_flip_probability_in_native_PC1':float(np.mean(ag[:-1]*ag[1:]<0)),
                'same_time_native_oracle_sign_match':float(np.mean(au*ag>0)),
                'lag1_native_vs_previous_oracle_sign_match':float(np.mean(au[1:]*ag[:-1]>0))},
                'descriptive_FIR':{'basis':'PC1 of centered 256-D CountSketch of native W2 updates','train_fraction':.70,'heldout_N':int(len(y)-cut),
                'coeff_intercept':float(cf[0]),'coeff_k0_to_k8':np.asarray(cf[1:],dtype=float).tolist(),'heldout_R2_full_lag0_to_8':float(r2f),
                'heldout_R2_tau0_only':float(r20),'heldout_R2_gain_from_history':float(r2f-r20),'tau0_coeff_intercept':float(c0[0]),'tau0_coeff':float(c0[1])},
                'scientific_boundary':'The FIR is descriptive system identification in a fixed compressed dominant W2 mode. Lag coefficients do not by themselves prove causal stale credit.'}

def make_d3_audit_clone(start):
    s=pickle.loads(pickle.dumps(start,pickle.HIGHEST_PROTOCOL)); s.__class__=D3AuditSystem; s.audit_init(sketch_dim=64); s.d3_init(); return s

def choose_probe_contexts(contexts,nprobe,hmax=8):
    hi=int(contexts)-int(hmax)+1
    if hi<1: raise ValueError('contexts must be >= max horizon')
    n=min(int(nprobe),hi); vals=np.unique(np.linspace(1,hi,num=n,dtype=np.int64)); return set(int(x) for x in vals),[int(x) for x in vals]

def freeze_native_context(prestate,truth,j,rng_state):
    s=pickle.loads(pickle.dumps(prestate,pickle.HIGHEST_PROTOCOL)); rng=gd.clone_rng(rng_state); sm=sample_mask_set(s.c.N,rng); masked=np.asarray(sm.masked_positions,dtype=np.int64)
    arr=np.asarray(truth,dtype=np.int64).copy(); arr[masked]=core.MASK; mirror_seed=int(rng.integers(1,2**31-1)); core.seed_numba(mirror_seed); s._current_arr=np.asarray(arr,dtype=np.int64).copy(); xn,xraw=s.context_from_arr(arr)
    s._reciprocal_batch_microsteps(len(masked)); cache=s.prepare_mask_batch(xn,xraw,masked); order=np.asarray(rng.permutation(len(cache)),dtype=np.int64); return s,cache,order,masked,mirror_seed

def context_oracle(frozen,cache,truth):
    G=np.zeros_like(frozen.W2,dtype=np.float64)
    for rec in cache:
        g,_post,_h=gd.w2_descent_oracle(frozen,rec,int(truth[int(rec['t'])])); G+=g
    return G/max(1,len(cache))

def _eq_state(v,u,z):
    return np.full_like(v,core.V0,dtype=np.float64),np.full_like(u,core.U0,dtype=np.float64),np.zeros_like(z,dtype=np.float64)

def replay_w2_direct(frozen,cache,truth,order:Sequence[int],mode='native'):
    if mode not in PERSISTENCE_MODES: raise ValueError(mode)
    e_inherited=(np.array(frozen.ve2,copy=True),np.array(frozen.ue2,copy=True),np.array(frozen.ze2,copy=True)); m_inherited=(np.array(frozen.vm2,copy=True),np.array(frozen.um2,copy=True),np.array(frozen.zm2,copy=True))
    e_base=_eq_state(*e_inherited) if mode in ('fresh_E_context','fresh_EM_context') else tuple(np.array(x,copy=True) for x in e_inherited)
    m_base=_eq_state(*m_inherited) if mode in ('fresh_M_context','fresh_EM_context') else tuple(np.array(x,copy=True) for x in m_inherited)
    ve,ue,ze=[np.array(x,copy=True) for x in e_base]; vm,um,zm=[np.array(x,copy=True) for x in m_base]; acc=np.zeros_like(frozen.W2,dtype=np.float64); event_rows=[]
    for rank,ci0 in enumerate(order):
        ci=int(ci0); rec=cache[ci]
        if mode in ('no_E_within','no_EM_within'): ve[:],ue[:],ze[:]=e_inherited[0],e_inherited[1],e_inherited[2]
        if mode in ('no_M_within','no_EM_within'): vm[:],um[:],zm[:]=m_inherited[0],m_inherited[1],m_inherited[2]
        target=int(truth[int(rec['t'])]); p=np.asarray(rec['p'],dtype=np.float64); one=np.zeros(frozen.c.V,dtype=np.float64); one[target]=1.; dy=frozen.BR@(one-p)
        E2,_=core.fast_izh_step(ve,ue,np.asarray(rec['h'],dtype=np.float64),ze,core.G_E,core.CV_E,core.CU_E); M2,_=core.fast_izh_step(vm,um,dy,zm,core.G_M,core.CV_M,core.CU_M)
        du=np.outer(M2,E2)*frozen.mask2; acc+=du; event_rows.append({'rank':int(rank),'cache_index':ci,'position':int(rec['t']),'target':target,'contribution_norm':float(np.linalg.norm(du))})
    return frozen.lr2*acc/max(1,len(cache)),event_rows

def vector_metrics(u,g):
    u=np.asarray(u,dtype=np.float64); g=np.asarray(g,dtype=np.float64); gn2=float(np.sum(g*g)); dot=float(np.sum(u*g)); up=(dot/(gn2+EPS))*g if gn2>EPS else np.zeros_like(g); perp=u-up; un=float(np.linalg.norm(u)); pn=float(np.linalg.norm(perp))
    return {'cosine':gd.cosine(u,g),'u_norm':un,'g_norm':float(np.linalg.norm(g)),'signed_descent':dot,'parallel_norm':float(np.linalg.norm(up)),'perp_norm':pn,'perp_fraction_of_decomposed_path':float(pn/(pn+np.linalg.norm(up)+EPS))},perp

def order_probe(frozen,cache,truth,native_order,context,nrandom):
    perms=[('native',0,np.asarray(native_order,dtype=np.int64)),('reverse',0,np.asarray(native_order[::-1],dtype=np.int64))]; prng=np.random.default_rng(880001+int(context))
    for k in range(int(nrandom)): perms.append(('random',k,prng.permutation(len(cache)).astype(np.int64)) )
    G=context_oracle(frozen,cache,truth); rows=[]; Us=[]; U_native=None
    for typ,pi,order in perms:
        U,_=replay_w2_direct(frozen,cache,truth,order,'native'); Us.append(U)
        if typ=='native': U_native=U
        rows.append({'context':int(context),'perm_type':typ,'perm_index':int(pi),'alignment_to_oracle':gd.cosine(U,G),'u_norm':float(np.linalg.norm(U)),'cosine_to_native':float('nan')})
    for r,U in zip(rows,Us): r['cosine_to_native']=gd.cosine(U,U_native)
    meanU=np.mean(np.stack(Us),axis=0); mean_sq=float(np.mean([np.sum(U*U) for U in Us])); order_var=float(np.mean([np.sum((U-meanU)**2) for U in Us])/(mean_sq+EPS)); pcs=[gd.cosine(Us[a],Us[b]) for a in range(len(Us)) for b in range(a+1,len(Us))]
    ctx={'context':int(context),'n_orders':int(len(Us)),'pairwise_cosine_mean':float(np.nanmean(pcs)),'pairwise_cosine_median':float(np.nanmedian(pcs)),'V_order':order_var,'native_alignment':gd.cosine(U_native,G),'mean_order_alignment':gd.cosine(meanU,G),'alignment_gain_mean_order_minus_native':gd.cosine(meanU,G)-gd.cosine(U_native,G)}
    return U_native,G,rows,ctx

def persistence_probe(frozen,cache,truth,native_order,context,G):
    rows=[]
    for mode in PERSISTENCE_MODES:
        U,_=replay_w2_direct(frozen,cache,truth,native_order,mode); met,_=vector_metrics(U,G); row={'context':int(context),'mode':mode}; row.update(met); rows.append(row)
    return rows

def frozen_context_oracle(system,truth,masked):
    s=pickle.loads(pickle.dumps(system,pickle.HIGHEST_PROTOCOL)); arr=np.asarray(truth,dtype=np.int64).copy(); arr[np.asarray(masked,dtype=np.int64)]=core.MASK; s._current_arr=np.asarray(arr,dtype=np.int64).copy(); xn,xraw=s.context_from_arr(arr); cache=s.prepare_mask_batch(xn,xraw,np.asarray(masked,dtype=np.int64)); return context_oracle(s,cache,truth)

def future_oracles(prestate,tokens,indices,j,rng_state,horizons=HORIZONS):
    hmax=max(int(h) for h in horizons); rng=gd.clone_rng(rng_state); grads=[]
    for off in range(hmax):
        idx=int(indices[(j-1)+off]); truth=np.asarray(tokens[idx],dtype=np.int64); sm=sample_mask_set(prestate.c.N,rng); masked=np.asarray(sm.masked_positions,dtype=np.int64); _mirror_seed=int(rng.integers(1,2**31-1)); _order=rng.permutation(len(masked)); grads.append(frozen_context_oracle(prestate,truth,masked))
    out={}; acc=np.zeros_like(prestate.W2,dtype=np.float64); wanted=set(int(h) for h in horizons)
    for k,g in enumerate(grads,start=1):
        acc+=g
        if k in wanted: out[k]=np.array(acc,copy=True)
    return out

def summarize_order(context_rows):
    return {'N_contexts':int(len(context_rows)),'V_order':summarize_scalar([r['V_order'] for r in context_rows],seed=101001),'pairwise_cosine':summarize_scalar([r['pairwise_cosine_mean'] for r in context_rows],seed=101101),'native_alignment':summarize_scalar([r['native_alignment'] for r in context_rows],seed=101201),'mean_order_alignment':summarize_scalar([r['mean_order_alignment'] for r in context_rows],seed=101301),'alignment_gain_mean_order_minus_native':summarize_scalar([r['alignment_gain_mean_order_minus_native'] for r in context_rows],seed=101401),'interpretation_boundary':'Order effects are measured in a W2-direct replay with reciprocal result and forward cache frozen; they are not whole-network order effects.'}

def summarize_persistence(rows):
    out={'modes':{},'paired_vs_native':{}}; by={m:[r for r in rows if r['mode']==m] for m in PERSISTENCE_MODES}
    for m,rr in by.items():
        i=PERSISTENCE_MODES.index(m); out['modes'][m]={'alignment':summarize_scalar([r['cosine'] for r in rr],seed=102000+i),'perp_fraction':summarize_scalar([r['perp_fraction_of_decomposed_path'] for r in rr],seed=102100+i),'u_norm':summarize_scalar([r['u_norm'] for r in rr],seed=102200+i)}
    native={int(r['context']):r for r in by['native']}
    for m in PERSISTENCE_MODES[1:]:
        i=PERSISTENCE_MODES.index(m); diffa=[]; diffp=[]
        for r in by[m]:
            n=native[int(r['context'])]; diffa.append(r['cosine']-n['cosine']); diffp.append(r['perp_fraction_of_decomposed_path']-n['perp_fraction_of_decomposed_path'])
        out['paired_vs_native'][m]={'delta_alignment':summarize_scalar(diffa,seed=103000+i),'delta_perp_fraction':summarize_scalar(diffp,seed=103100+i)}
    out['scientific_boundary']='All persistence variants are counterfactual diagnostics on copied E2/M2 state. They are not candidate learners or optimization proposals.'; return out

def summarize_horizon(rows):
    out={'horizons':{}}
    for H in HORIZONS:
        rr=[r for r in rows if int(r['H'])==H]; out['horizons'][str(H)]={'native_alignment_to_GH':summarize_scalar([r['cos_u_GH'] for r in rr],seed=104000+H),'instantaneous_orthogonal_alignment_to_GH':summarize_scalar([r['cos_uperp_GH'] for r in rr],seed=104100+H)}
    out['scientific_boundary']='GH is a frozen-parameter future-context oracle over the native future GT-visible mask stream. No intermediate learning occurs; this is not BPTT and does not claim an exact training adjoint.'; return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--tokenizer',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--contexts',type=int,default=1024); ap.add_argument('--probe-contexts',type=int,default=64); ap.add_argument('--order-permutations',type=int,default=8); ap.add_argument('--encode-batch',type=int,default=256); ap.add_argument('--seed',type=int,default=7311); ap.add_argument('--threads',type=int,default=4); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads); oracle_test=gd.oracle_self_test()
    if not oracle_test['pass']: raise RuntimeError(f'oracle self-test failed: {oracle_test}')
    start,tokens,indices,rng_state,meta,tok_meta,sp_tr,input_audit=d15.load_start(a)
    if len(indices)!=a.contexts: raise ValueError('requested contexts exceed available frozen schedule')
    probe_set,probe_list=choose_probe_contexts(a.contexts,a.probe_contexts,max(HORIZONS)); reference=pickle.loads(pickle.dumps(start,pickle.HIGHEST_PROTOCOL)); audit=make_d3_audit_clone(start); rr=gd.clone_rng(rng_state); ra=gd.clone_rng(rng_state)
    parity_checkpoints={}; replay_errors=[]; oracle_errors=[]; d3b_rows=[]; d3b_ctx=[]; d3c_rows=[]; d3d_rows=[]
    for j,image_index in enumerate(indices,start=1):
        truth=np.asarray(tokens[int(image_index)],dtype=np.int64); probe=None; independent_G1=None; U_replay=None
        if j in probe_set:
            pre_audit=pickle.loads(pickle.dumps(audit,pickle.HIGHEST_PROTOCOL)); ctx_rng_state=copy.deepcopy(ra.bit_generator.state); frozen,cache,native_order,masked,mirror_seed=freeze_native_context(pre_audit,truth,j,ctx_rng_state)
            U_replay,G_direct,orows,octx=order_probe(frozen,cache,truth,native_order,j,a.order_permutations); d3b_rows.extend(orows); d3b_ctx.append(octx); d3c_rows.extend(persistence_probe(frozen,cache,truth,native_order,j,G_direct))
            GH=future_oracles(pre_audit,tokens,indices,j,ctx_rng_state,HORIZONS); independent_G1=GH[1]; gm=float(np.max(np.abs(independent_G1-G_direct)))
            if gm>ORACLE_TOL: raise RuntimeError(f'D3 oracle rebuild mismatch before native at context {j}: {gm}')
            _met,uperp=vector_metrics(U_replay,independent_G1)
            for H in HORIZONS:
                d3d_rows.append({'context':int(j),'H':int(H),'cos_u_GH':gd.cosine(U_replay,GH[H]),'cos_uperp_GH':gd.cosine(uperp,GH[H]),'u_norm':float(np.linalg.norm(U_replay)),'uperp_norm':float(np.linalg.norm(uperp)),'GH_norm':float(np.linalg.norm(GH[H]))})
            probe={'masked':masked,'mirror_seed':mirror_seed,'order':native_order}
        rec_r,ev_r=gd.run_one_context(reference,False,arm='T0',j=j,image_index=int(image_index),truth=truth,rng=rr); rec_a,ev_a=gd.run_one_context(audit,True,arm='T0',j=j,image_index=int(image_index),truth=truth,rng=ra)
        if ev_r['masked']!=ev_a['masked'] or ev_r['E']!=ev_a['E'] or ev_r['mirror_seed']!=ev_a['mirror_seed']: raise RuntimeError(f'D3 reference/audit stream divergence at {j}')
        if gd.json_state(rr.bit_generator.state)!=gd.json_state(ra.bit_generator.state): raise RuntimeError(f'D3 external RNG divergence at {j}')
        if float(rec_r['mean_nll'])!=float(rec_a['mean_nll']): raise RuntimeError(f'D3 mean_nll divergence at {j}')
        if j in probe_set:
            if ev_a['masked']!=probe['masked'].astype(int).tolist() or int(ev_a['mirror_seed'])!=int(probe['mirror_seed']): raise RuntimeError(f'D3 frozen/native stream mismatch at probe context {j}')
            uerr=float(np.max(np.abs(np.asarray(U_replay)-np.asarray(audit._audit_d3_last_u)))); gerr=float(np.max(np.abs(np.asarray(independent_G1)-np.asarray(audit._audit_d3_last_g)))); replay_errors.append({'context':int(j),'max_abs':uerr}); oracle_errors.append({'context':int(j),'max_abs':gerr})
            if uerr>REPLAY_TOL: raise RuntimeError(f'D3 W2 direct replay identity failed at {j}: {uerr}')
            if gerr>ORACLE_TOL: raise RuntimeError(f'D3 oracle reconstruction identity failed at {j}: {gerr}')
        if j in gd.CHECKPOINTS or j==len(indices):
            cmp=gd.compare_native_states(reference,audit); cmp['context']=int(j); cmp['training_rng_equal']=bool(gd.json_state(rr.bit_generator.state)==gd.json_state(ra.bit_generator.state)); cmp['pass']=bool(cmp['bitwise_pass'] and cmp['training_rng_equal']); parity_checkpoints[str(j)]=cmp; print('D3_PARITY_CHECKPOINT',json.dumps({'context':j,'pass':cmp['pass']}),flush=True)
            if not cmp['pass']: raise RuntimeError(f'D3 parity failed at context {j}')
        if j in (256,512,1024) or j==len(indices): print('D3_PROGRESS',json.dumps({'context':j,'probe_done':int(sum(x<=j for x in probe_list))}),flush=True)
    temporal=audit.d3_temporal_summary(); order_summary=summarize_order(d3b_ctx); persistence_summary=summarize_persistence(d3c_rows); horizon_summary=summarize_horizon(d3d_rows)
    write_csv(a.out/'D3A_TEMPORAL_ROWS.csv',audit._audit_d3_rows); (a.out/'D3A_TEMPORAL_KERNEL.json').write_text(json.dumps(temporal,indent=2),encoding='utf-8'); write_csv(a.out/'D3B_ORDER_ROWS.csv',d3b_rows); write_csv(a.out/'D3B_ORDER_CONTEXTS.csv',d3b_ctx); (a.out/'D3B_ORDER_SUMMARY.json').write_text(json.dumps(order_summary,indent=2),encoding='utf-8'); write_csv(a.out/'D3C_PERSISTENCE_ROWS.csv',d3c_rows); (a.out/'D3C_PERSISTENCE_SUMMARY.json').write_text(json.dumps(persistence_summary,indent=2),encoding='utf-8'); write_csv(a.out/'D3D_HORIZON_ROWS.csv',d3d_rows); (a.out/'D3D_HORIZON_SUMMARY.json').write_text(json.dumps(horizon_summary,indent=2),encoding='utf-8')
    parity={'gate':'D3_DIAGNOSTIC_PARITY','contexts':int(a.contexts),'checkpoints':parity_checkpoints,'reference_native_bitwise_pass':bool(all(v['pass'] for v in parity_checkpoints.values())),'probe_contexts':probe_list,'W2_direct_replay_tolerance':REPLAY_TOL,'W2_direct_replay_max_abs':float(max([x['max_abs'] for x in replay_errors],default=0.)),'oracle_reconstruction_tolerance':ORACLE_TOL,'oracle_reconstruction_max_abs':float(max([x['max_abs'] for x in oracle_errors],default=0.)),'W2_direct_replay_rows':replay_errors,'oracle_reconstruction_rows':oracle_errors}
    parity['pass']=bool(parity['reference_native_bitwise_pass'] and parity['W2_direct_replay_max_abs']<=REPLAY_TOL and parity['oracle_reconstruction_max_abs']<=ORACLE_TOL); (a.out/'D3_PARITY.json').write_text(json.dumps(parity,indent=2),encoding='utf-8')
    if not parity['pass']: raise RuntimeError('D3 parity gate failed')
    manifest={'stage':'GATE_D3_TEMPORAL_CREDIT_MEMORY_MATCHING_AUDIT','contexts':int(a.contexts),'probe_contexts':int(len(probe_list)),'order_random_permutations_per_probe':int(a.order_permutations),'source_context_epoch':float(meta['context_epoch_equivalent']),'seed':int(a.seed),'github_sha':os.environ.get('GITHUB_SHA'),'D3_parity_pass':True,'gradient_oracle_enters_candidate_learning':False,'counterfactual_replay_enters_candidate_learning':False,'native_training_rule_changed':False,'tokenizer_metadata':tok_meta,'input_audit':input_audit,'train_tokenizer_activity':sp_tr,'scientific_boundary':'D3 localizes temporal credit-memory mechanisms. D3B/C are cloned-state diagnostics; D3D is a frozen-parameter future-context oracle. No result by itself licenses changing learning rates, resetting native E/M, or claiming a biological mechanism.'}
    (a.out/'D3_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print('D3_COMPLETE',json.dumps({'parity':parity['pass'],'C_ug_lags_mean':{k:v['C_ug']['mean'] for k,v in temporal['lags'].items()},'FIR_R2_gain':temporal['descriptive_FIR']['heldout_R2_gain_from_history'],'V_order_median':order_summary['V_order']['median'],'persistence_delta_alignment_median':{k:v['delta_alignment']['median'] for k,v in persistence_summary['paired_vs_native'].items()},'future_orthogonal_alignment_median':{h:horizon_summary['horizons'][h]['instantaneous_orthogonal_alignment_to_GH']['median'] for h in horizon_summary['horizons']}},indent=2),flush=True)

if __name__=='__main__': main()
