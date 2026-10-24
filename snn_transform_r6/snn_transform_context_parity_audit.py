#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SNN-Transform Context/Parity Audit v1.

Read-only diagnostic of the completed Adaptive Timescale v1 checkpoint.
No training, no parameter update, no timescale update, no MaskGIT generation.

Questions:
  A. Does changing the other tokens change a fixed target prediction?
  B. Does resident H/g attention equal an explicit token-scan reference?
  C. Is attention selective / does it use visible context?
  D. Does the attention output actually contribute to the final prediction?
  E. Does correct context help more than all-MASK / shuffled / wrong-image context?

The checkpoint was created by running adaptive_timescale_v1.py as __main__, so
pickle compatibility aliases are installed before loading. This is only a loader
compatibility step; the loaded model is never modified by learning.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'snn_transform_p3_image'))
sys.path.insert(0, str(HERE))

import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
import adaptive_timescale_v1 as ats

FORMAT = 'SNN_TRANSFORM_CONTEXT_PARITY_AUDIT_V1'
CONTEXT_VARIANTS = ('correct','shuffled','wrong','allmask')
PARENT_FORMAT = 'SNN_TRANSFORM_ADAPTIVE_TIMESCALE_V1'


def load_checkpoint_compat(path: Path):
    # The parent script was executed as __main__, hence these class names were
    # serialized under __main__. Rebind them to the actual implementation before
    # unpickling. No state is changed after loading except resident forward caches.
    import __main__
    __main__.AdaptiveTimescaleFullForwardSystem = ats.AdaptiveTimescaleFullForwardSystem
    __main__.TimescaleConfig = ats.TimescaleConfig
    __main__.CausalTimescaleController = ats.CausalTimescaleController
    with path.open('rb') as f:
        payload = pickle.load(f)
    if payload.get('format') != PARENT_FORMAT:
        raise ValueError(f'expected parent format {PARENT_FORMAT}, got {payload.get("format")}')
    return payload


def js_divergence(p, q, eps=1e-12):
    p=np.asarray(p,dtype=np.float64);q=np.asarray(q,dtype=np.float64)
    p=np.clip(p,eps,None);q=np.clip(q,eps,None)
    p/=p.sum();q/=q.sum();m=0.5*(p+q)
    return float(0.5*np.sum(p*np.log(p/m))+0.5*np.sum(q*np.log(q/m)))


def bootstrap_mean_ci(x, seed, n_boot=4000):
    x=np.asarray(x,dtype=np.float64)
    x=x[np.isfinite(x)]
    if len(x)==0:
        return {'n':0,'mean':None,'ci95':[None,None]}
    rng=np.random.default_rng(seed)
    means=np.empty(int(n_boot),dtype=np.float64)
    n=len(x)
    for b in range(int(n_boot)):
        means[b]=np.mean(x[rng.integers(0,n,size=n)])
    lo,hi=np.quantile(means,[0.025,0.975])
    return {'n':int(n),'mean':float(np.mean(x)),'ci95':[float(lo),float(hi)]}


def explicit_attention(system, xn, target):
    """Independent explicit token scan for the exact positive-RF kernel."""
    c=system.c;N=c.N;H=c.H;hd=c.D//c.H
    xn=np.asarray(xn,dtype=np.float64)
    q_all=(xn@system.WQ.T+system.bq).reshape(N,H,hd)
    k_all=(xn@system.WK.T+system.bk).reshape(N,H,hd)
    v_all=(xn@system.WV.T+system.bv).reshape(N,H,hd)
    a=np.empty(c.D,dtype=np.float64)
    g=np.empty((H,c.M),dtype=np.float64)
    HH=np.empty((H,c.M,hd),dtype=np.float64)
    alpha=np.empty((H,N),dtype=np.float64)
    den=np.empty(H,dtype=np.float64)
    num=np.empty((H,hd),dtype=np.float64)
    pq_all=np.empty((H,c.M),dtype=np.float64)
    pk_all=np.empty((H,N,c.M),dtype=np.float64)
    for h in range(H):
        pq=core.phi_q(q_all[int(target),h],system.omega[h])
        pk=np.stack([core.phi_k(k_all[n,h],system.omega[h]) for n in range(N)],axis=0)
        scores=pk@pq
        dd=max(1e-8,float(np.sum(scores)))
        aa=scores/dd
        gg=np.sum(pk,axis=0)
        hh=np.einsum('nm,nd->md',pk,v_all[:,h,:])
        nn=pq@hh
        sl=slice(h*hd,(h+1)*hd)
        a[sl]=nn/dd
        g[h]=gg;HH[h]=hh;alpha[h]=aa;den[h]=dd;num[h]=nn
        pq_all[h]=pq;pk_all[h]=pk
    return {
        'a':a,'g':g,'H':HH,'alpha':alpha,'den':den,'num':num,
        'q':q_all[int(target)],'pq':pq_all,'pk':pk_all,
    }


def resident_prediction(system, arr, target, true_token=None, with_explicit=False):
    """Stateless-with-respect-to-learning forward diagnostic for one context."""
    arr=np.asarray(arr,dtype=np.int64)
    xn,xraw=system.context_from_arr(arr)
    system.rebuild(xn)
    a,q,num,den=system._read(int(target))
    u,h,rf,p=system._tail(a,xraw[int(target)])
    out={
        'p':np.asarray(p,dtype=np.float64).copy(),
        'a':np.asarray(a,dtype=np.float64).copy(),
        'q':np.asarray(q,dtype=np.float64).copy(),
        'num':np.asarray(num,dtype=np.float64).copy(),
        'den':np.asarray(den,dtype=np.float64).copy(),
        'xn':np.asarray(xn,dtype=np.float64).copy(),
        'xraw_target':np.asarray(xraw[int(target)],dtype=np.float64).copy(),
    }
    if true_token is not None:
        z=int(true_token)
        out['p_true']=float(out['p'][z])
        out['ce']=float(-math.log(max(out['p_true'],1e-12)))
        out['correct']=int(np.argmax(out['p'])==z)
    if with_explicit:
        ex=explicit_attention(system,xn,int(target))
        out['explicit']=ex
        out['parity']={
            'a_max_abs':float(np.max(np.abs(out['a']-ex['a']))),
            'H_max_abs':float(np.max(np.abs(system.Hstate-ex['H']))),
            'g_max_abs':float(np.max(np.abs(system.gstate-ex['g']))),
            'q_max_abs':float(np.max(np.abs(out['q']-ex['q']))),
            'num_max_abs':float(np.max(np.abs(out['num']-ex['num']))),
            'den_max_abs':float(np.max(np.abs(out['den']-ex['den']))),
        }
    return out


def no_attention_prediction(system, a_record, true_token):
    # Same xraw target and downstream network, but ablate only attention output a.
    zero=np.zeros(system.c.D,dtype=np.float64)
    _u,_h,_rf,p0=system._tail(zero,a_record['xraw_target'])
    p0=np.asarray(p0,dtype=np.float64)
    z=int(true_token)
    residual=a_record['xraw_target']+system.bo
    context_vec=system.WO@a_record['a']
    return {
        'p':p0,
        'p_true':float(p0[z]),
        'ce':float(-math.log(max(float(p0[z]),1e-12))),
        'correct':int(np.argmax(p0)==z),
        'js_normal_vs_no_attention':js_divergence(a_record['p'],p0),
        'WOa_norm':float(np.linalg.norm(context_vec)),
        'residual_base_norm':float(np.linalg.norm(residual)),
        'context_to_residual_ratio':float(np.linalg.norm(context_vec)/(np.linalg.norm(residual)+1e-12)),
    }


def attention_stats(alpha, visible_mask, target):
    alpha=np.asarray(alpha,dtype=np.float64)
    H,N=alpha.shape
    visible_mask=np.asarray(visible_mask,dtype=bool)
    masked_mask=~visible_mask
    rows=[]
    for h in range(H):
        a=np.clip(alpha[h],1e-15,1.0)
        ent=-float(np.sum(a*np.log(a)))
        vmax=float(np.sum(a[visible_mask])) if np.any(visible_mask) else 0.0
        mmass=float(np.sum(a[masked_mask])) if np.any(masked_mask) else 0.0
        vmean=float(np.mean(a[visible_mask])) if np.any(visible_mask) else 0.0
        mmean=float(np.mean(a[masked_mask])) if np.any(masked_mask) else 0.0
        rows.append({
            'head':h,
            'entropy':ent,
            'normalized_entropy':float(ent/max(math.log(N),1e-12)),
            'effective_tokens':float(math.exp(ent)),
            'max_alpha':float(np.max(a)),
            'self_alpha':float(a[int(target)]),
            'visible_mass':vmax,
            'masked_mass':mmass,
            'visible_mean_alpha':vmean,
            'masked_mean_alpha':mmean,
            'visible_to_masked_per_token_ratio':float(vmean/(mmean+1e-15)),
        })
    return rows


def choose_diagnostics(tokens, labels, dominant, n_fg, n_bg, seed):
    rng=np.random.default_rng(seed)
    tokens=np.asarray(tokens,dtype=np.int64);labels=np.asarray(labels,dtype=np.int64)
    fg=np.argwhere(tokens!=int(dominant))
    bg=np.argwhere(tokens==int(dominant))
    if len(fg)<n_fg or len(bg)<n_bg:
        raise ValueError('not enough foreground/background target candidates')
    fg=fg[rng.choice(len(fg),size=n_fg,replace=False)]
    bg=bg[rng.choice(len(bg),size=n_bg,replace=False)]
    out=[]
    for kind,arr in [('foreground',fg),('background',bg)]:
        for ii,t in arr:
            # choose a wrong-image partner with a different MNIST label.
            candidates=np.flatnonzero(labels!=labels[int(ii)])
            jj=int(candidates[int(rng.integers(len(candidates)))])
            order=np.asarray([x for x in rng.permutation(tokens.shape[1]) if x!=int(t)],dtype=np.int64)
            out.append({
                'kind':kind,'image_index':int(ii),'target':int(t),
                'wrong_image_index':jj,'context_order':order,
            })
    rng.shuffle(out)
    return out


def make_context(truth, wrong_truth, target, order, visible_fraction, variant, rng, mask_id):
    N=len(truth);arr=np.full(N,int(mask_id),dtype=np.int64)
    nvis=int(round(float(visible_fraction)*(N-1)))
    nvis=max(0,min(N-1,nvis))
    visible=np.asarray(order[:nvis],dtype=np.int64)
    if variant=='correct':
        arr[visible]=truth[visible]
    elif variant=='wrong':
        arr[visible]=wrong_truth[visible]
    elif variant=='shuffled':
        vals=np.asarray(truth[visible],dtype=np.int64).copy()
        if len(vals)>1:
            # deterministic per-sample random permutation; force a non-identity
            # positional permutation where possible.
            perm=rng.permutation(len(vals))
            if np.array_equal(perm,np.arange(len(vals))):
                perm=np.roll(perm,1)
            vals=vals[perm]
        arr[visible]=vals
    else:
        raise ValueError(variant)
    arr[int(target)]=int(mask_id)
    vmask=np.zeros(N,dtype=bool);vmask[visible]=True;vmask[int(target)]=False
    return arr,vmask


def save_csv(path, rows):
    if not rows:return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)


def summarize_rows(rows, fractions, primary, seed):
    summary={'by_kind':{},'paired_primary':{}}
    for kind in ('foreground','background','all'):
        rr=rows if kind=='all' else [r for r in rows if r['kind']==kind]
        byf={}
        for frac in fractions:
            q=[r for r in rr if r['variant']=='correct' and abs(r['visible_fraction']-frac)<1e-12]
            byf[str(frac)]={
                'n':len(q),
                'mean_p_true':float(np.mean([x['p_true'] for x in q])) if q else None,
                'mean_ce':float(np.mean([x['ce'] for x in q])) if q else None,
                'accuracy':float(np.mean([x['correct'] for x in q])) if q else None,
                'mean_js_vs_allmask':float(np.mean([x['js_vs_allmask'] for x in q])) if q else None,
                'mean_a_rel_change_vs_allmask':float(np.mean([x['a_rel_change_vs_allmask'] for x in q])) if q else None,
            }
        summary['by_kind'][kind]=byf

    # Build paired records by sample id at the preregistered primary context level.
    keyed={}
    for r in rows:
        if abs(r['visible_fraction']-primary)>1e-12 and r['variant']!='allmask':
            continue
        keyed.setdefault(r['sample_id'],{})[(r['variant'],r['visible_fraction'])]=r
    pairs=[]
    for sid,d in keyed.items():
        c=d.get(('correct',primary));s=d.get(('shuffled',primary));w=d.get(('wrong',primary));z=d.get(('allmask',0.0))
        if c and s and w and z:
            pairs.append((c,s,w,z))
    for kind in ('foreground','background','all'):
        pp=[x for x in pairs if kind=='all' or x[0]['kind']==kind]
        def diff(fn_a,fn_b):return np.asarray([fn_a(x)-fn_b(x) for x in pp],dtype=np.float64)
        if not pp:continue
        summary['paired_primary'][kind]={
            'n':len(pp),
            'ptrue_correct_minus_allmask':bootstrap_mean_ci(diff(lambda x:x[0]['p_true'],lambda x:x[3]['p_true']),seed+11),
            'ptrue_correct_minus_shuffled':bootstrap_mean_ci(diff(lambda x:x[0]['p_true'],lambda x:x[1]['p_true']),seed+12),
            'ptrue_correct_minus_wrong':bootstrap_mean_ci(diff(lambda x:x[0]['p_true'],lambda x:x[2]['p_true']),seed+13),
            'ce_correct_minus_allmask':bootstrap_mean_ci(diff(lambda x:x[0]['ce'],lambda x:x[3]['ce']),seed+14),
            'ce_correct_minus_shuffled':bootstrap_mean_ci(diff(lambda x:x[0]['ce'],lambda x:x[1]['ce']),seed+15),
            'ce_correct_minus_wrong':bootstrap_mean_ci(diff(lambda x:x[0]['ce'],lambda x:x[2]['ce']),seed+16),
            'mean_js_correct_vs_allmask':float(np.mean([x[0]['js_vs_allmask'] for x in pp])),
            'mean_a_rel_change_correct_vs_allmask':float(np.mean([x[0]['a_rel_change_vs_allmask'] for x in pp])),
        }
    return summary


def classify(parity, context_summary, contribution_summary, tol):
    mx=max(parity.values()) if parity else float('inf')
    if mx>tol:
        return 'F0_FORWARD_IMPLEMENTATION_PARITY_FAIL'
    fg=context_summary['paired_primary'].get('foreground',{})
    js=float(fg.get('mean_js_correct_vs_allmask',0.0) or 0.0)
    ar=float(fg.get('mean_a_rel_change_correct_vs_allmask',0.0) or 0.0)
    if js<1e-8 and ar<1e-8:
        return 'F1_CONTEXT_PATH_EFFECTIVELY_INACTIVE'
    if contribution_summary['mean_js_normal_vs_no_attention']<1e-8:
        return 'F2_CONTEXT_COMPUTED_BUT_DOWNSTREAM_EFFECT_WEAK'
    def ci_positive(key):
        d=fg.get(key,{});ci=d.get('ci95',[None,None])
        return ci[0] is not None and ci[0]>0
    semantic=(ci_positive('ptrue_correct_minus_allmask') and
              ci_positive('ptrue_correct_minus_shuffled') and
              ci_positive('ptrue_correct_minus_wrong'))
    if semantic:
        return 'F4_CONTEXTUAL_FORWARD_FUNCTION_SUPPORTED'
    return 'F3_CONTEXT_ACTIVE_BUT_SEMANTIC_BENEFIT_NOT_ESTABLISHED'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--parent-results',type=Path,default=None)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--foreground-samples',type=int,default=256)
    ap.add_argument('--background-samples',type=int,default=128)
    ap.add_argument('--visible-fractions',type=str,default='0,0.10,0.25,0.50,0.75,0.90')
    ap.add_argument('--primary-visible',type=float,default=0.75)
    ap.add_argument('--parity-tol',type=float,default=1e-10)
    ap.add_argument('--encode-batch',type=int,default=256)
    ap.add_argument('--seed',type=int,default=7311)
    ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    fractions=[float(x) for x in a.visible_fractions.split(',') if x.strip()!='']
    if 0.0 not in fractions:fractions=[0.0]+fractions
    fractions=sorted(set(fractions))
    if a.primary_visible not in fractions:
        fractions.append(a.primary_visible);fractions=sorted(set(fractions))

    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(a.threads)
    payload=load_checkpoint_compat(a.checkpoint)
    system=payload['system']
    # Freeze a compact hash-like snapshot to detect accidental learning mutation.
    before={k:np.asarray(getattr(system,k),dtype=np.float64).copy() for k in ats.GROUP_PARAMETERS['fast']+ats.GROUP_PARAMETERS['medium']+ats.GROUP_PARAMETERS['slow']}
    tau_before=system.timescale_snapshot()
    controller_count_before=int(system.timescale_controller.context_count)

    device=torch.device('cpu')
    xall,yall,xte0,yte0=core.load(a.data)
    _x012,_y012=core.subset012(xall,yall)
    xte,yte=core.subset012(xte0,yte0)
    tokenizer,_tok_meta=base.load_frozen_tokenizer(a.tokenizer,device)
    te,_sp=base.encode_snn(tokenizer,xte,device,a.encode_batch)

    dominant=None
    if a.parent_results is not None and a.parent_results.exists():
        parent=json.loads(a.parent_results.read_text(encoding='utf-8'))
        try: dominant=int(parent['masked_eval']['1.0']['dominant_token'])
        except Exception: dominant=None
    if dominant is None:
        dominant=int(np.argmax(np.bincount(np.asarray(te).reshape(-1),minlength=system.c.V)))

    diags=choose_diagnostics(te,yte,dominant,a.foreground_samples,a.background_samples,a.seed+91)
    rows=[];parity_rows=[];attn_rows=[];contrib_rows=[]
    max_parity={'a_max_abs':0.0,'H_max_abs':0.0,'g_max_abs':0.0,'q_max_abs':0.0,'num_max_abs':0.0,'den_max_abs':0.0}

    for sid,d in enumerate(diags):
        truth=np.asarray(te[d['image_index']],dtype=np.int64)
        wrong=np.asarray(te[d['wrong_image_index']],dtype=np.int64)
        target=d['target'];z=int(truth[target]);order=d['context_order']
        srng=np.random.default_rng(a.seed+100000+sid)

        # all-MASK baseline is identical for all fractions and variants.
        arr0,vm0=make_context(truth,wrong,target,order,0.0,'correct',srng,core.MASK)
        rec0=resident_prediction(system,arr0,target,z,with_explicit=True)
        base_a_norm=max(float(np.linalg.norm(rec0['a'])),1e-12)
        row0={
            'sample_id':sid,'kind':d['kind'],'image_index':d['image_index'],'target':target,
            'true_token':z,'wrong_image_index':d['wrong_image_index'],'visible_fraction':0.0,
            'variant':'allmask','visible_count':0,'p_true':rec0['p_true'],'ce':rec0['ce'],
            'correct':rec0['correct'],'js_vs_allmask':0.0,'a_rel_change_vs_allmask':0.0,
        }
        rows.append(row0)
        pr={'sample_id':sid,'kind':d['kind'],**rec0['parity']};parity_rows.append(pr)
        for k,v in rec0['parity'].items():max_parity[k]=max(max_parity[k],float(v))

        for frac in fractions:
            if frac==0.0:continue
            arr,vm=make_context(truth,wrong,target,order,frac,'correct',srng,core.MASK)
            rec=resident_prediction(system,arr,target,z,with_explicit=(abs(frac-a.primary_visible)<1e-12))
            rr={
                'sample_id':sid,'kind':d['kind'],'image_index':d['image_index'],'target':target,
                'true_token':z,'wrong_image_index':d['wrong_image_index'],'visible_fraction':frac,
                'variant':'correct','visible_count':int(np.sum(vm)),'p_true':rec['p_true'],'ce':rec['ce'],
                'correct':rec['correct'],'js_vs_allmask':js_divergence(rec['p'],rec0['p']),
                'a_rel_change_vs_allmask':float(np.linalg.norm(rec['a']-rec0['a'])/base_a_norm),
            }
            rows.append(rr)
            if abs(frac-a.primary_visible)<1e-12:
                if 'parity' in rec:
                    pr={'sample_id':sid,'kind':d['kind'],**rec['parity']};parity_rows.append(pr)
                    for k,v in rec['parity'].items():max_parity[k]=max(max_parity[k],float(v))
                nctx=no_attention_prediction(system,rec,z)
                contrib_rows.append({
                    'sample_id':sid,'kind':d['kind'],'p_true_normal':rec['p_true'],
                    'p_true_no_attention':nctx['p_true'],'delta_p_true':rec['p_true']-nctx['p_true'],
                    'ce_normal':rec['ce'],'ce_no_attention':nctx['ce'],
                    'correct_normal':rec['correct'],'correct_no_attention':nctx['correct'],
                    **{k:nctx[k] for k in ('js_normal_vs_no_attention','WOa_norm','residual_base_norm','context_to_residual_ratio')},
                })
                for hs in attention_stats(rec['explicit']['alpha'],vm,target):
                    attn_rows.append({'sample_id':sid,'kind':d['kind'],'visible_fraction':frac,**hs})

                # Primary controlled context corruptions.
                for variant in ('shuffled','wrong'):
                    av,vv=make_context(truth,wrong,target,order,frac,variant,srng,core.MASK)
                    rv=resident_prediction(system,av,target,z,with_explicit=False)
                    rows.append({
                        'sample_id':sid,'kind':d['kind'],'image_index':d['image_index'],'target':target,
                        'true_token':z,'wrong_image_index':d['wrong_image_index'],'visible_fraction':frac,
                        'variant':variant,'visible_count':int(np.sum(vv)),'p_true':rv['p_true'],'ce':rv['ce'],
                        'correct':rv['correct'],'js_vs_allmask':js_divergence(rv['p'],rec0['p']),
                        'a_rel_change_vs_allmask':float(np.linalg.norm(rv['a']-rec0['a'])/base_a_norm),
                    })

    context_summary=summarize_rows(rows,fractions,a.primary_visible,a.seed+500)
    fg_contrib=[r for r in contrib_rows if r['kind']=='foreground']
    contribution_summary={
        'n_foreground':len(fg_contrib),
        'mean_js_normal_vs_no_attention':float(np.mean([r['js_normal_vs_no_attention'] for r in fg_contrib])) if fg_contrib else None,
        'mean_context_to_residual_ratio':float(np.mean([r['context_to_residual_ratio'] for r in fg_contrib])) if fg_contrib else None,
        'delta_p_true_normal_minus_no_attention':bootstrap_mean_ci([r['delta_p_true'] for r in fg_contrib],a.seed+701),
        'accuracy_normal':float(np.mean([r['correct_normal'] for r in fg_contrib])) if fg_contrib else None,
        'accuracy_no_attention':float(np.mean([r['correct_no_attention'] for r in fg_contrib])) if fg_contrib else None,
    }
    fg_attn=[r for r in attn_rows if r['kind']=='foreground']
    attention_summary={
        'n_head_samples':len(fg_attn),
        'mean_normalized_entropy':float(np.mean([r['normalized_entropy'] for r in fg_attn])) if fg_attn else None,
        'mean_effective_tokens':float(np.mean([r['effective_tokens'] for r in fg_attn])) if fg_attn else None,
        'mean_max_alpha':float(np.mean([r['max_alpha'] for r in fg_attn])) if fg_attn else None,
        'mean_self_alpha':float(np.mean([r['self_alpha'] for r in fg_attn])) if fg_attn else None,
        'mean_visible_mass':float(np.mean([r['visible_mass'] for r in fg_attn])) if fg_attn else None,
        'mean_visible_to_masked_per_token_ratio':float(np.mean([r['visible_to_masked_per_token_ratio'] for r in fg_attn])) if fg_attn else None,
    }

    # Verify the diagnostic itself did not perform any learning or controller update.
    max_param_mut=0.0
    for k,v0 in before.items():
        max_param_mut=max(max_param_mut,float(np.max(np.abs(np.asarray(getattr(system,k),dtype=np.float64)-v0))))
    no_learning_mutation=(max_param_mut<=0.0 and system.timescale_snapshot()==tau_before and
                          int(system.timescale_controller.context_count)==controller_count_before)

    classification=classify(max_parity,context_summary,contribution_summary,a.parity_tol)
    gates={
        'resident_explicit_parity_pass':bool(max(max_parity.values())<=a.parity_tol),
        'audit_no_parameter_or_controller_learning_mutation':bool(no_learning_mutation),
        'foreground_primary_context_output_changes':bool(context_summary['paired_primary']['foreground']['mean_js_correct_vs_allmask']>1e-8),
    }
    summary={
        'format':FORMAT,
        'parent_run_id':40115465892,
        'parent_artifact_id':12569296115,
        'parent_artifact_digest':'sha256:ed2aaa58eae5c648e6c98e0ab66803eeac0b39ef0dbe579384af77abccf70d47',
        'parent_checkpoint':str(a.checkpoint),
        'scientific_scope':'read-only forward context computation and resident-vs-explicit parity audit; no retraining',
        'softmax_transformer_parity_claimed':False,
        'designed_operator':'positive-random-feature kernel attention with resident H/g state',
        'dominant_token':dominant,
        'visible_fractions':fractions,
        'primary_visible_fraction':a.primary_visible,
        'diagnostic_samples':{'foreground':a.foreground_samples,'background':a.background_samples},
        'parity_tolerance':a.parity_tol,
        'parity_max_abs':max_parity,
        'context_sensitivity':context_summary,
        'attention_structure_foreground_primary':attention_summary,
        'context_contribution_foreground_primary':contribution_summary,
        'max_forward_parameter_mutation_during_audit':max_param_mut,
        'tau_before_after_identical':bool(system.timescale_snapshot()==tau_before),
        'controller_context_count_before':controller_count_before,
        'controller_context_count_after':int(system.timescale_controller.context_count),
        'gates':gates,
        'classification':classification,
        'classification_meaning':{
            'F0_FORWARD_IMPLEMENTATION_PARITY_FAIL':'resident H/g implementation disagrees with explicit reference; fix forward code before training analysis',
            'F1_CONTEXT_PATH_EFFECTIVELY_INACTIVE':'parity passes but changing other tokens produces essentially no target-level context signal',
            'F2_CONTEXT_COMPUTED_BUT_DOWNSTREAM_EFFECT_WEAK':'attention state responds to context but downstream prediction barely uses it',
            'F3_CONTEXT_ACTIVE_BUT_SEMANTIC_BENEFIT_NOT_ESTABLISHED':'context changes predictions, but correct context is not reliably more useful than controls',
            'F4_CONTEXTUAL_FORWARD_FUNCTION_SUPPORTED':'resident math is correct and correct context provides paired semantic benefit at the preregistered 75% context level',
        }[classification],
    }

    save_csv(a.out/'CONTEXT_SENSITIVITY_ROWS.csv',rows)
    save_csv(a.out/'PARITY_ROWS.csv',parity_rows)
    save_csv(a.out/'ATTENTION_STRUCTURE_ROWS.csv',attn_rows)
    save_csv(a.out/'CONTEXT_CONTRIBUTION_ROWS.csv',contrib_rows)
    (a.out/'CONTEXT_PARITY_AUDIT_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (a.out/'CONTEXT_PARITY_DIAGNOSTIC_IDS.json').write_text(json.dumps([
        {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in d.items()} for d in diags
    ],indent=2),encoding='utf-8')

    # Minimal plot: correct-context performance as visible context increases.
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fg=context_summary['by_kind']['foreground']
        xs=fractions;ys=[fg[str(x)]['mean_p_true'] for x in xs]
        fig,ax=plt.subplots(figsize=(6.5,4.2));ax.plot(xs,ys,marker='o')
        ax.set_xlabel('Visible fraction of other tokens');ax.set_ylabel('Mean P(true target token)')
        ax.set_title('Foreground target: controlled correct-context sensitivity');ax.grid(True,alpha=.2)
        fig.tight_layout();fig.savefig(a.out/'CONTEXT_SENSITIVITY_FOREGROUND.png',dpi=180);plt.close(fig)
    except Exception as e:
        summary['plot_warning']=repr(e)
        (a.out/'CONTEXT_PARITY_AUDIT_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

    print('CONTEXT_PARITY_AUDIT_FINAL',json.dumps({
        'classification':classification,'gates':gates,'parity_max_abs':max_parity,
        'foreground_primary':context_summary['paired_primary'].get('foreground'),
        'contribution':contribution_summary,
    }),flush=True)


if __name__=='__main__':
    main()
