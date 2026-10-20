#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D5A — W2 versus O-path local-credit divergence audit.

Diagnostic only. Replays paired N and F_EM trajectories from the exact 2.0-epoch
R6B3 checkpoint after the frozen R6B4 aux=1.0 upgrade. The candidate learner is
unchanged; only read-only instrumentation is added.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch

import gate_d_plasticity_audit as gd
import gate_d_d3e_runtime as d3r
import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
import r6b3_context_epoch_training as trainer
from maskgit_input_representation import build_input_embeddings
from r6b4_full_forward_plasticity import FullForwardPlasticMaskGITSystem, upgrade_r6b3_system

EPS=1e-30
CHECKPOINTS=(256,512,1024)
LAGS=(1,2,4,8)
PATHS=('W2','WO','bo')
EXPECTED_CHECKPOINT_SHA256='4a42e60ada31f6ec5167cafd2bd860dd8ca937f7970665c5b95bf8fb2dfc89d0'


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


def bootstrap_mean(x,seed=1,B=5000):
    a=np.asarray(x,dtype=np.float64); a=a[np.isfinite(a)]
    if len(a)==0:
        return {'N':0,'mean':float('nan'),'median':float('nan'),'p_negative':float('nan'),'mean_95ci':[float('nan'),float('nan')],'median_95ci':[float('nan'),float('nan')]}
    rng=np.random.default_rng(seed); vm=np.empty(B); vd=np.empty(B)
    for b in range(B):
        q=a[rng.integers(0,len(a),len(a))]; vm[b]=np.mean(q); vd[b]=np.median(q)
    return {'N':int(len(a)),'mean':float(np.mean(a)),'median':float(np.median(a)),
            'p_negative':float(np.mean(a<0)),
            'mean_95ci':[float(np.quantile(vm,.025)),float(np.quantile(vm,.975))],
            'median_95ci':[float(np.quantile(vd,.025)),float(np.quantile(vd,.975))]}


def paired_summary(a,b,seed=1,B=5000):
    x=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64)
    x=x[np.isfinite(x)]
    if len(x)==0: return {'N':0,'mean':float('nan'),'ci95':[float('nan'),float('nan')]}
    rng=np.random.default_rng(seed); z=np.empty(B)
    for k in range(B):
        q=x[rng.integers(0,len(x),len(x))]; z[k]=np.mean(q)
    return {'N':int(len(x)),'mean':float(np.mean(x)),'median':float(np.median(x)),
            'ci95':[float(np.quantile(z,.025)),float(np.quantile(z,.975))],
            'fraction_positive':float(np.mean(x>0)),'fraction_negative':float(np.mean(x<0))}


def tail_loss_from_z(z,W1,W2,b1,b2,theta,gamma,beta,eps2,WR,target):
    z=np.asarray(z,dtype=np.float64)
    mu=float(np.mean(z)); inv=1.0/math.sqrt(float(np.mean((z-mu)**2))+eps2)
    zhat=(z-mu)*inv; u=gamma*zhat+beta
    h=np.maximum(0.0,W1@u+b1-theta); y=z+W2@h+b2
    return gd._loss_from_y(y,WR,int(target))


def synthetic_oracle_self_test(seed=520260):
    rng=np.random.default_rng(seed); D,FF,V=11,15,7; eps2=1e-5
    z=rng.normal(size=D); a=rng.normal(size=D)
    WO=rng.normal(scale=.2,size=(D,D)); W1=rng.normal(scale=.2,size=(FF,D)); W2=rng.normal(scale=.2,size=(D,FF))
    b1=rng.normal(scale=.1,size=FF); b2=rng.normal(scale=.1,size=D); theta=-0.2
    gamma=1.+rng.normal(scale=.05,size=D); beta=rng.normal(scale=.05,size=D); WR=rng.normal(scale=.2,size=(V,D+1)); target=int(rng.integers(V))
    # Move away from ReLU kinks if necessary.
    for _ in range(20):
        mu=float(np.mean(z)); inv=1.0/math.sqrt(float(np.mean((z-mu)**2))+eps2); zh=(z-mu)*inv; u=gamma*zh+beta
        pre=W1@u+b1-theta
        if float(np.min(np.abs(pre)))>2e-3: break
        b1+=0.03
    h=np.maximum(0.0,pre); y=z+W2@h+b2
    r=math.sqrt(float(np.mean(y*y)))+1e-8; rf=np.empty(D+1); rf[:-1]=.5*y/r; rf[-1]=.5
    logits=WR@rf; logits-=np.max(logits); p=np.exp(logits); p/=np.sum(p); one=np.zeros(V); one[target]=1.
    q=(WR.T@(p-one))[:-1]; s=math.sqrt(float(np.mean(y*y)))
    grad_y=.5*q/r-.5*float(np.dot(q,y))*y/(D*s*r*r)
    grad_h=W2.T@grad_y; grad_pre=grad_h*(pre>0); grad_u=W1.T@grad_pre; grad_zh=gamma*grad_u
    grad_z=grad_y+inv*(grad_zh-np.mean(grad_zh)-zh*np.mean(grad_zh*zh))
    Gz=-grad_z; GW2=-np.outer(grad_y,h); GWO=-np.outer(grad_z,a); Gbo=-grad_z
    e=1e-6
    dz=rng.normal(size=D); dW2=rng.normal(size=(D,FF)); dWO=rng.normal(size=(D,D)); dbo=rng.normal(size=D)
    L=lambda zz,WW2: tail_loss_from_z(zz,W1,WW2,b1,b2,theta,gamma,beta,eps2,WR,target)
    fd_z=(L(z+e*dz,W2)-L(z-e*dz,W2))/(2*e); an_z=float(np.dot(-Gz,dz))
    # W2 only changes y through h at fixed z.
    lp=tail_loss_from_z(z,W1,W2+e*dW2,b1,b2,theta,gamma,beta,eps2,WR,target)
    lm=tail_loss_from_z(z,W1,W2-e*dW2,b1,b2,theta,gamma,beta,eps2,WR,target)
    fd_w2=(lp-lm)/(2*e); an_w2=float(np.sum((-GW2)*dW2))
    fd_wo=(L(z+e*(dWO@a),W2)-L(z-e*(dWO@a),W2))/(2*e); an_wo=float(np.sum((-GWO)*dWO))
    fd_bo=(L(z+e*dbo,W2)-L(z-e*dbo,W2))/(2*e); an_bo=float(np.dot(-Gbo,dbo))
    def rel(fd,an): return abs(fd-an)/max(1.,abs(fd),abs(an))
    out={'z':{'fd':fd_z,'analytic':an_z,'relative_error':rel(fd_z,an_z)},
         'W2':{'fd':fd_w2,'analytic':an_w2,'relative_error':rel(fd_w2,an_w2)},
         'WO':{'fd':fd_wo,'analytic':an_wo,'relative_error':rel(fd_wo,an_wo)},
         'bo':{'fd':fd_bo,'analytic':an_bo,'relative_error':rel(fd_bo,an_bo)}}
    out['pass']=bool(max(v['relative_error'] for k,v in out.items() if k!='pass')<2e-7)
    return out


def exact_event_oracles(s,rec,target):
    G2,gy_desc,h=gd.w2_descent_oracle(s,rec,int(target))
    grad_y=-np.asarray(gy_desc,dtype=np.float64)
    t=int(rec['t']); a=np.asarray(rec['a'],dtype=np.float64); u=np.asarray(rec['u'],dtype=np.float64)
    z=np.asarray(s._current_xraw[t],dtype=np.float64)+s.bo+s.WO@a
    mu=float(np.mean(z)); inv=1.0/math.sqrt(float(np.mean((z-mu)**2))+s.eps2); zhat=(z-mu)*inv
    pre=s.W1@u+s.b1-s.theta
    grad_h=s.W2.T@grad_y; grad_pre=grad_h*(pre>0); grad_u=s.W1.T@grad_pre; grad_zhat=s.gamma2*grad_u
    grad_z=grad_y+inv*(grad_zhat-np.mean(grad_zhat)-zhat*np.mean(grad_zhat*zhat))
    gz_desc=-grad_z
    GO=np.outer(gz_desc,a)*s.maskO; Gbo=np.asarray(gz_desc,dtype=np.float64)
    return {'W2':G2,'WO':GO,'bo':Gbo,'gy':np.asarray(gy_desc,dtype=np.float64),'gz':gz_desc,'h':h,'a':a}


def reconstructed_states(s):
    E2=core.CV_E*(s.ve2-core.V0)+core.CU_E*(s.ue2-core.U0)
    M2=core.CV_M*(s.vm2-core.V0)+core.CU_M*(s.um2-core.U0)
    EO=core.CV_E*(s.veo-core.V0)+core.CU_E*(s.ueo-core.U0)
    MO=core.CV_M*(s.vmo-core.V0)+core.CU_M*(s.umo-core.U0)
    Eb=core.CV_E*(s.veconst-core.V0)+core.CU_E*(s.ueconst-core.U0)
    return E2,M2,EO,MO,float(Eb[0])


class D5AuditSystem(FullForwardPlasticMaskGITSystem):
    def d5_init(self):
        self._d5_event_rows=[]; self._d5_context_rows=[]; self._d5_ctx=None; self._d5_rank=0
        self._d5_hist={k:{'g':[],'u':[]} for k in PATHS}
        self._d5_path={k:0. for k in PATHS}; self._d5_net={k:np.zeros_like(getattr(self,k),dtype=np.float64) for k in PATHS}
        self._d5_oracle_path={k:0. for k in PATHS}; self._d5_oracle_net={k:np.zeros_like(getattr(self,k),dtype=np.float64) for k in PATHS}
        self._d5_parallel={k:0. for k in PATHS}; self._d5_perp={k:0. for k in PATHS}

    def audit_begin_context(self,*,arm,context,image_index,truth,masked,E,mirror_seed):
        self._d5_ctx={'arm':str(arm),'context':int(context),'image_index':int(image_index),'truth':np.asarray(truth,dtype=np.int64).copy(),
                      'masked':np.asarray(masked,dtype=np.int64).copy(),'E':np.asarray(E,dtype=np.int64).copy(),'mirror_seed':int(mirror_seed)}
        self._d5_before={k:np.array(getattr(self,k),copy=True) for k in PATHS}
        self._d5_U={k:np.zeros_like(getattr(self,k),dtype=np.float64) for k in PATHS}; self._d5_G={k:np.zeros_like(getattr(self,k),dtype=np.float64) for k in PATHS}; self._d5_rank=0

    def accumulate_cached_target(self,rec,target,weight=1.0):
        p=super().accumulate_cached_target(rec,target,weight)
        if self._d5_ctx is None: return p
        E2,M2,EO,MO,eb=reconstructed_states(self); o=exact_event_oracles(self,rec,int(target))
        u2=np.outer(M2,E2)*self.mask2; uo=np.outer(MO,EO)*self.maskO; ub=MO*eb
        U={'W2':u2,'WO':uo,'bo':ub}; G={'W2':o['W2'],'WO':o['WO'],'bo':o['bo']}
        for k in PATHS: self._d5_U[k]+=U[k]; self._d5_G[k]+=G[k]
        pp=np.asarray(rec['p'],dtype=np.float64); tgt=int(target); pred=int(np.argmax(pp))
        row={'arm':self._d5_ctx['arm'],'context':self._d5_ctx['context'],'rank':int(self._d5_rank),'position':int(rec['t']),'target':tgt,'pred':pred,
             'p_target':float(pp[tgt]),'eb':eb,'eb_positive':bool(eb>0),
             'W2_A_M':gd.cosine(M2,o['gy']),'W2_A_E':gd.cosine(E2,o['h']),
             'WO_A_M':gd.cosine(MO,o['gz']),'WO_A_E':gd.cosine(EO,o['a']),
             'bo_A_M':gd.cosine(MO,o['gz']),'bo_A_E_const':gd.cosine(np.asarray([eb]),np.ones(1))}
        for k in PATHS:
            uu=np.asarray(U[k],dtype=np.float64); gg=np.asarray(G[k],dtype=np.float64); un=float(np.linalg.norm(uu)); dot=float(np.sum(uu*gg))
            row[f'{k}_cosine']=gd.cosine(uu,gg); row[f'{k}_signed_descent']=dot; row[f'{k}_efficiency']=float(dot/(un*un+EPS)); row[f'{k}_u_norm']=un; row[f'{k}_g_norm']=float(np.linalg.norm(gg))
        self._d5_event_rows.append(row); self._d5_rank+=1
        return p

    def audit_finish_context(self):
        if self._d5_ctx is None: raise AssertionError('D5 finish without begin')
        n=max(1,len(self._d5_ctx['masked'])); row={'arm':self._d5_ctx['arm'],'context':self._d5_ctx['context'],'image_index':self._d5_ctx['image_index'],'nmask':int(n)}
        for k in PATHS:
            u=np.asarray(getattr(self,k),dtype=np.float64)-self._d5_before[k]; g=np.asarray(self._d5_G[k],dtype=np.float64)/n
            lr=self.lr2 if k=='W2' else (self.lro if k=='WO' else self.aux_lr_scale*self.lro)
            expected=lr*np.asarray(self._d5_U[k],dtype=np.float64)/n
            ident=float(np.max(np.abs(u-expected))) if u.size else 0.; gn=float(np.linalg.norm(g)); un=float(np.linalg.norm(u)); dot=float(np.sum(u*g))
            row[f'{k}_native_update_identity_max_abs']=ident; row[f'{k}_cosine']=gd.cosine(u,g); row[f'{k}_signed_descent']=dot; row[f'{k}_efficiency']=float(dot/(un*un+EPS)); row[f'{k}_u_norm']=un; row[f'{k}_g_norm']=gn
            if gn>EPS:
                oracle=(un/gn)*g; par=(dot/(gn*gn+EPS))*g
            else:
                oracle=np.zeros_like(g); par=np.zeros_like(g)
            perp=u-par; self._d5_path[k]+=un; self._d5_net[k]+=u; self._d5_oracle_path[k]+=float(np.linalg.norm(oracle)); self._d5_oracle_net[k]+=oracle
            self._d5_parallel[k]+=float(np.linalg.norm(par)); self._d5_perp[k]+=float(np.linalg.norm(perp))
            for lag in LAGS:
                row[f'{k}_C_ug_{lag}']=gd.cosine(u,self._d5_hist[k]['g'][-lag]) if len(self._d5_hist[k]['g'])>=lag else float('nan')
            self._d5_hist[k]['g'].append(np.array(g,copy=True)); self._d5_hist[k]['u'].append(np.array(u,copy=True))
            if len(self._d5_hist[k]['g'])>max(LAGS): self._d5_hist[k]['g'].pop(0); self._d5_hist[k]['u'].pop(0)
        self._d5_context_rows.append(row); self._d5_ctx=None

    def d5_summary(self):
        out={}
        for pi,k in enumerate(PATHS):
            ctx=self._d5_context_rows; ev=self._d5_event_rows
            net=float(np.linalg.norm(self._d5_net[k])); onet=float(np.linalg.norm(self._d5_oracle_net[k])); tn=float(self._d5_path[k]/(net+EPS)); to=float(self._d5_oracle_path[k]/(onet+EPS))
            q={'context_alignment':bootstrap_mean([r[f'{k}_cosine'] for r in ctx],61000+pi),
               'context_signed_descent':bootstrap_mean([r[f'{k}_signed_descent'] for r in ctx],62000+pi),
               'context_efficiency':bootstrap_mean([r[f'{k}_efficiency'] for r in ctx],63000+pi),
               'event_alignment':bootstrap_mean([r[f'{k}_cosine'] for r in ev],64000+pi),
               'event_signed_descent':bootstrap_mean([r[f'{k}_signed_descent'] for r in ev],65000+pi),
               'event_efficiency':bootstrap_mean([r[f'{k}_efficiency'] for r in ev],66000+pi),
               'native':{'path_abs':float(self._d5_path[k]),'net_abs':net,'tortuosity':tn},
               'oracle_matched':{'path_abs':float(self._d5_oracle_path[k]),'net_abs':onet,'tortuosity':to},
               'excess_tortuosity':float(tn/(to+EPS)),
               'orthogonal_path_fraction':float(self._d5_perp[k]/(self._d5_perp[k]+self._d5_parallel[k]+EPS)),
               'lags':{str(l):bootstrap_mean([r[f'{k}_C_ug_{l}'] for r in ctx],67000+pi*100+l) for l in LAGS},
               'identity_max_abs':float(max(r[f'{k}_native_update_identity_max_abs'] for r in ctx))}
            if k=='W2':
                q['M_alignment']=bootstrap_mean([r['W2_A_M'] for r in ev],68001); q['E_alignment']=bootstrap_mean([r['W2_A_E'] for r in ev],68002)
            elif k=='WO':
                q['M_alignment']=bootstrap_mean([r['WO_A_M'] for r in ev],68101); q['E_alignment']=bootstrap_mean([r['WO_A_E'] for r in ev],68102)
            else:
                q['M_alignment']=bootstrap_mean([r['bo_A_M'] for r in ev],68201); q['constant_E_alignment']=bootstrap_mean([r['bo_A_E_const'] for r in ev],68202); q['Eb_amplitude']=bootstrap_mean([r['eb'] for r in ev],68203); q['Eb_fraction_positive']=float(np.mean([r['eb_positive'] for r in ev]))
            out[k]=q
        return out


def make_audit(start):
    s=pickle.loads(pickle.dumps(start,pickle.HIGHEST_PROTOCOL)); s.__class__=D5AuditSystem; s.d5_init(); return s


def stream_signature(ev,event_rows):
    order=[(int(r['rank']),int(r['position']),int(r['target'])) for r in event_rows]
    return {'masked':list(ev['masked']),'E':list(ev['E']),'mirror_seed':int(ev['mirror_seed']),'order':order}


def run_pair(arm,start,tokens,indices,rng_state):
    plain=pickle.loads(pickle.dumps(start,pickle.HIGHEST_PROTOCOL)); inst=make_audit(start); rp=gd.clone_rng(rng_state); ri=gd.clone_rng(rng_state)
    checkpoints={}; stream=[]
    for j,image_index in enumerate(indices,start=1):
        if arm=='F_EM':
            d3r.reset_boundary(plain,'F_EM'); d3r.reset_boundary(inst,'F_EM')
        truth=np.asarray(tokens[int(image_index)],dtype=np.int64)
        recp,evp=gd.run_one_context(plain,False,arm='T0',j=j,image_index=int(image_index),truth=truth,rng=rp)
        e0=len(inst._d5_event_rows); reci,evi=gd.run_one_context(inst,True,arm='T0',j=j,image_index=int(image_index),truth=truth,rng=ri); er=inst._d5_event_rows[e0:]
        sigp=(evp['masked'],evp['E'],evp['mirror_seed']); sigi=(evi['masked'],evi['E'],evi['mirror_seed'])
        if sigp!=sigi or d3r.jstate(rp)!=d3r.jstate(ri) or float(recp['mean_nll'])!=float(reci['mean_nll']): raise RuntimeError(f'{arm} plain/instrumented stream divergence at {j}')
        stream.append(stream_signature(evi,er))
        if j in CHECKPOINTS:
            cmp=d3r.compare_native_fields_bitwise(plain,inst); cmp['external_rng_exact']=bool(d3r.jstate(rp)==d3r.jstate(ri)); cmp['pass']=bool(cmp['bitwise_pass'] and cmp['external_rng_exact']); checkpoints[str(j)]=cmp
            print('D5A_PARITY',json.dumps({'arm':arm,'context':j,'pass':cmp['pass']}),flush=True)
    return inst,stream,{'pass':bool(all(x['pass'] for x in checkpoints.values())),'checkpoints':checkpoints,'rng_after':d3r.jstate(ri)}


def assert_paired_events(nrows,frows):
    if len(nrows)!=len(frows): raise RuntimeError('N/F event count mismatch')
    for i,(a,b) in enumerate(zip(nrows,frows)):
        ka=(a['context'],a['rank'],a['position'],a['target']); kb=(b['context'],b['rank'],b['position'],b['target'])
        if ka!=kb: raise RuntimeError(f'N/F event pairing mismatch at {i}: {ka} vs {kb}')


def paired_arm_shift(ns,fs,nrows,frows,nctx,fctx):
    out={}
    for pi,k in enumerate(PATHS):
        d={'context_alignment_F_minus_N':paired_summary([r[f'{k}_cosine'] for r in fctx],[r[f'{k}_cosine'] for r in nctx],71000+pi),
           'context_signed_descent_F_minus_N':paired_summary([r[f'{k}_signed_descent'] for r in fctx],[r[f'{k}_signed_descent'] for r in nctx],71100+pi),
           'context_efficiency_F_minus_N':paired_summary([r[f'{k}_efficiency'] for r in fctx],[r[f'{k}_efficiency'] for r in nctx],71200+pi),
           'event_alignment_F_minus_N':paired_summary([r[f'{k}_cosine'] for r in frows],[r[f'{k}_cosine'] for r in nrows],71300+pi)}
        if k=='W2':
            d['M_alignment_F_minus_N']=paired_summary([r['W2_A_M'] for r in frows],[r['W2_A_M'] for r in nrows],71401); d['E_alignment_F_minus_N']=paired_summary([r['W2_A_E'] for r in frows],[r['W2_A_E'] for r in nrows],71402)
        elif k=='WO':
            d['M_alignment_F_minus_N']=paired_summary([r['WO_A_M'] for r in frows],[r['WO_A_M'] for r in nrows],71501); d['E_alignment_F_minus_N']=paired_summary([r['WO_A_E'] for r in frows],[r['WO_A_E'] for r in nrows],71502)
        else:
            d['M_alignment_F_minus_N']=paired_summary([r['bo_A_M'] for r in frows],[r['bo_A_M'] for r in nrows],71601); d['Eb_F_minus_N']=paired_summary([r['eb'] for r in frows],[r['eb'] for r in nrows],71602)
        d['excess_tortuosity_F_minus_N']=float(fs[k]['excess_tortuosity']-ns[k]['excess_tortuosity'])
        d['orthogonal_path_fraction_F_minus_N']=float(fs[k]['orthogonal_path_fraction']-ns[k]['orthogonal_path_fraction'])
        d['lag_shift_F_minus_N']={str(l):float(fs[k]['lags'][str(l)]['mean']-ns[k]['lags'][str(l)]['mean']) for l in LAGS}
        out[k]=d
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--tokenizer',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--contexts',type=int,default=1024); ap.add_argument('--encode-batch',type=int,default=256); ap.add_argument('--seed',type=int,default=7311); ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    if a.contexts!=1024: raise ValueError('formal D5A requires exactly 1024 contexts')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads)
    st=synthetic_oracle_self_test(); (a.out/'D5A_ORACLE_SELF_TEST.json').write_text(json.dumps(st,indent=2),encoding='utf-8')
    if not st['pass']: raise RuntimeError({'D5A_exact_oracle_self_test_failed':st})

    checkpoint_sha=d3r.sha_file(a.checkpoint); source,meta,train_rng,schedule=trainer.load_full_checkpoint(a.checkpoint)
    if checkpoint_sha!=EXPECTED_CHECKPOINT_SHA256 or abs(float(meta.get('context_epoch_equivalent',-1))-2.)>1e-12: raise RuntimeError('D5A start checkpoint identity failed')
    offset=int(schedule['offset']); indices=np.asarray(schedule['permutation'],dtype=np.int64)[offset:offset+a.contexts]; rng_state=copy.deepcopy(train_rng.bit_generator.state)
    if len(indices)!=1024: raise RuntimeError('checkpoint schedule lacks 1024 contexts')
    xall,yall,_xte,_yte=core.load(a.data); x012,y012=core.subset012(xall,yall); xtr,_ytr,_xva,_yva=core.split(x012,y012,.1,7)
    toknet,_=base.load_frozen_tokenizer(a.tokenizer,torch.device('cpu')); tr,_=base.encode_snn(toknet,xtr,torch.device('cpu'),a.encode_batch); tok0,pos0,_=build_input_embeddings(toknet,source.c.D,grid=7)
    start=pickle.loads(pickle.dumps(source,pickle.HIGHEST_PROTOCOL)); upgrade_r6b3_system(start,tok0,pos0,aux_lr_scale=1.0)

    N,nstream,npar=run_pair('N',start,tr,indices,rng_state); F,fstream,fpar=run_pair('F_EM',start,tr,indices,rng_state)
    if not(npar['pass'] and fpar['pass']): raise RuntimeError('same-run instrumentation parity failed')
    if nstream!=fstream or npar['rng_after']!=fpar['rng_after']: raise RuntimeError('N/F_EM stream or external RNG pairing failed')
    assert_paired_events(N._d5_event_rows,F._d5_event_rows)
    ns=N.d5_summary(); fs=F.d5_summary(); shifts=paired_arm_shift(ns,fs,N._d5_event_rows,F._d5_event_rows,N._d5_context_rows,F._d5_context_rows)
    max_ident=max(max(ns[k]['identity_max_abs'],fs[k]['identity_max_abs']) for k in PATHS)
    if max_ident>2e-14: raise RuntimeError(f'native accumulator/update identity failed: {max_ident}')

    write_csv(a.out/'D5A_N_EVENT_ROWS.csv',N._d5_event_rows); write_csv(a.out/'D5A_F_EM_EVENT_ROWS.csv',F._d5_event_rows); write_csv(a.out/'D5A_N_CONTEXT_ROWS.csv',N._d5_context_rows); write_csv(a.out/'D5A_F_EM_CONTEXT_ROWS.csv',F._d5_context_rows)
    result={'stage':'GATE_D5A_PATHWAY_SPECIFIC_LOCAL_CREDIT_DIVERGENCE','source_D4B_run':39496138727,
            'training_rule_modified':False,'learning_rates_modified':False,'candidate_repair_introduced':False,
            'checkpoint_sha256':checkpoint_sha,'oracle_self_test':st,'same_run_instrumentation_parity':{'N':npar,'F_EM':fpar},
            'N_F_stream_exact':True,'N_F_external_rng_after_exact':True,'native_update_identity_max_abs':max_ident,
            'N':ns,'F_EM':fs,'paired_F_EM_minus_N':shifts,
            'interpretation_boundary':('Exact-gradient alignment diagnoses local masked-token credit quality. Good local descent alignment does not imply autonomous-rollout benefit. D5A is observational/instrumented except for the frozen F_EM E2/M2 boundary reset; it introduces no repair.')}
    (a.out/'D5A_SUMMARY.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    manifest={'contexts':1024,'arms':['N','F_EM'],'paths':list(PATHS),'lags':list(LAGS),'checkpoint_sha256':checkpoint_sha,
              'oracle_fd_tolerance':2e-7,'update_identity_tolerance':2e-14,'same_run_parity_checkpoints':list(CHECKPOINTS),
              'F_EM_definition':'reset E2/M2 to canonical fresh initialization at each context boundary; no other learner changes'}
    (a.out/'D5A_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    compact={'oracle_pass':st['pass'],'parity_N':npar['pass'],'parity_F_EM':fpar['pass'],'identity_max':max_ident,
             'N':{k:{'A':ns[k]['context_alignment'],'M':ns[k].get('M_alignment'),'E':ns[k].get('E_alignment',ns[k].get('constant_E_alignment')),'T_excess':ns[k]['excess_tortuosity'],'R_perp':ns[k]['orthogonal_path_fraction'],'lags':ns[k]['lags']} for k in PATHS},
             'F_EM':{k:{'A':fs[k]['context_alignment'],'M':fs[k].get('M_alignment'),'E':fs[k].get('E_alignment',fs[k].get('constant_E_alignment')),'T_excess':fs[k]['excess_tortuosity'],'R_perp':fs[k]['orthogonal_path_fraction'],'lags':fs[k]['lags']} for k in PATHS},
             'paired':shifts}
    print('D5A_COMPLETE',json.dumps(compact,indent=2),flush=True)

if __name__=='__main__': main()
