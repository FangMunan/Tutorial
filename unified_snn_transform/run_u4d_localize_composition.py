#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""U4D — localize Izh error amplification through the U4 sparse tail.

No architecture change. Rebuild the frozen U4C 25%-25%-25% sparse block,
select the best trained-distribution Izh Q/K pair, then measure Q-only, K-only,
and stage-by-stage error growth through:
  attention output -> residual -> FFN preactivation -> active hidden -> FFN output
  -> block output -> task readout.
"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import run_unified_gate as u
import run_u3_trained_retrieval as r
import run_u31_izh_localization as u31
import run_u4b_task_sparse_block as u4b
import run_u4c_full_sparse_izh as u4c


def replace(ref,qitem=None,kitem=None):
    return u4c.replace_phi(ref,qitem,kitem)


def flat_rel(a,b): return r.rel_l2(a,b)
def flat_cos(a,b): return r.cosine(a,b)

@torch.no_grad()
def stage_forward(model,x):
    b=model.block
    changed=torch.ones(x.shape[:2],dtype=torch.bool)
    event_ratio=changed.float().mean(1);alpha=b.gate(event_ratio)
    a,state,stats=b.attn.step(b.n1(x),None,alpha,changed)
    z=x+a
    pre=b.f1(b.n2(z))
    h=F.relu(pre-b.theta)
    ff=b.f2(h)
    y=z+ff
    pred=model.readout(y[:,-1])
    return {'attn_out':a,'residual':z,'ff_pre':pre,'ff_hidden':h,'ff_out':ff,
            'block_out':y,'prediction':pred,'H':state.H,'g':state.g,
            'hidden_gate':(h>0).float()}

@torch.no_grad()
def stage_compare(ref,cand,gen,seed,batches=8,batch=64):
    g=torch.Generator();g.manual_seed(seed)
    names=['attn_out','residual','ff_pre','ff_hidden','ff_out','block_out','prediction','H','g']
    num={n:0. for n in names};den={n:0. for n in names};cs={n:[] for n in names if n not in ('H','g')}
    gate_flip=0.;gate_total=0.;target=[];pr=[];pc=[]
    for _ in range(batches):
        x,t,_,_=gen.sample(batch,g);A=stage_forward(ref,x);B=stage_forward(cand,x)
        for n in names:
            num[n]+=float(torch.sum((B[n]-A[n])**2));den[n]+=float(torch.sum(A[n]**2))
            if n in cs: cs[n].append(flat_cos(B[n],A[n]))
        gate_flip+=float((A['hidden_gate']!=B['hidden_gate']).float().sum());gate_total+=A['hidden_gate'].numel()
        target.append(t);pr.append(A['prediction']);pc.append(B['prediction'])
    rel={n:float(np.sqrt(num[n]/max(den[n],1e-30))) for n in names}
    cos={n:float(np.mean(v)) for n,v in cs.items()}
    pr=torch.cat(pr);pc=torch.cat(pc);target=torch.cat(target)
    return {'rel_l2_by_stage':rel,'cosine_by_stage':cos,
            'hidden_gate_flip_fraction':gate_flip/gate_total,
            'reference_task':r.metrics(pr,target),'candidate_task':r.metrics(pc,target)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='unified_snn_transform/run/U4D')
    ap.add_argument('--seed',type=int,default=71);ap.add_argument('--tokens',type=int,default=16)
    ap.add_argument('--d',type=int,default=32);ap.add_argument('--heads',type=int,default=2)
    ap.add_argument('--features',type=int,default=32);ap.add_argument('--ff',type=int,default=64)
    ap.add_argument('--value-dim',type=int,default=8);ap.add_argument('--theta-init',type=float,default=0.)
    ap.add_argument('--train-steps',type=int,default=900);ap.add_argument('--train-batch',type=int,default=128)
    ap.add_argument('--eval-batches',type=int,default=8);ap.add_argument('--eval-batch',type=int,default=64)
    ap.add_argument('--lr',type=float,default=2e-3);ap.add_argument('--report-every',type=int,default=300)
    ap.add_argument('--topn',type=int,default=12);ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args();torch.set_num_threads(args.threads);u4b.seed_all(args.seed)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    gen=r.AssocGenerator(args.tokens,args.d,args.value_dim,args.seed)
    ref=u4b.SparseRetrievalModel(args,args.seed+1001,.25,.25,.25)
    ref,hist=u4b.train_model(ref,gen,args,args.seed+2001)
    aq,ak=u4c.collect_a(ref,gen,args.seed+4001)
    qtop,_=u31.expanded_candidates(aq,args.features,args.seed+5001,args.topn)
    ktop,_=u31.expanded_candidates(ak,args.features,args.seed+5002,args.topn)
    # same pair selection rule as U4C
    best=None
    for qi,qitem in enumerate(qtop):
        for ki,kitem in enumerate(ktop):
            m=u4c.parity(ref,replace(ref,qitem,kitem),gen,args.seed+6001,batches=3,batch=64)
            score=m['prediction_rel_l2']+1e-5*(qitem[1]['window_ms']+kitem[1]['window_ms'])
            if best is None or score<best[0]: best=(score,qi,ki,qitem,kitem,m)
    _,qi,ki,qitem,kitem,sel=best
    cases={
        'Q_only':stage_compare(ref,replace(ref,qitem,None),gen,args.seed+7001,args.eval_batches,args.eval_batch),
        'K_only':stage_compare(ref,replace(ref,None,kitem),gen,args.seed+7001,args.eval_batches,args.eval_batch),
        'QK':stage_compare(ref,replace(ref,qitem,kitem),gen,args.seed+7001,args.eval_batches,args.eval_batch),
    }
    qk=cases['QK']['rel_l2_by_stage']
    amp_pred=qk['prediction']/max(qk['attn_out'],1e-12)
    result={'stage':'UNIFIED_U4D_COMPOSITION_LOCALIZATION','config':vars(args),
            'selected':{'q_rank':qi,'k_rank':ki,'Q':qitem[3],'K':kitem[3],'selection_metrics':sel},
            'cases':cases,'amplification':{'prediction_over_attn_out':amp_pred,
                'ff_hidden_over_ff_pre':qk['ff_hidden']/max(qk['ff_pre'],1e-12),
                'block_over_attn_out':qk['block_out']/max(qk['attn_out'],1e-12)},
            'decision':'localization only; use stage errors to decide whether to improve Izh interface or make sparse tail robust'}
    (out/'U4D_RESULTS.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
