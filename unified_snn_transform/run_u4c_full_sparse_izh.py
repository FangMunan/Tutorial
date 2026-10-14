#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""U4C — integrate the U4B sparse tail with the validated Izh feature interface.

This is the first single lightweight block in the unified line that combines:
  * positive-RF linear attention / persistent H-g state,
  * neuronal Izh trajectory feature map on Q and K,
  * 25% structural attention-output projection,
  * 25% structural FFN input/output connectivity,
  * exact-zero thresholded FFN hidden population.

The model is trained once with the exact positive-RF feature map; only the
feature-map interface is then replaced by calibrated Izh dynamics. No task or
attention weights are changed during the Izh parity gate.
"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np
import torch

import run_unified_gate as u
import run_u3_trained_retrieval as r
import run_u31_izh_localization as u31
import run_u4b_task_sparse_block as u4b


@torch.no_grad()
def collect_a(model,gen,seed,batches=8,batch=64):
    g=torch.Generator();g.manual_seed(seed);qs=[];ks=[];b=model.block
    for _ in range(batches):
        x,_,_,_=gen.sample(batch,g);z=b.n1(x)
        q=b.attn._split(b.attn.q(z));k=b.attn._split(b.attn.k(z))
        qs.append(b.attn.phi_q.stable_a(q,True).reshape(-1).cpu())
        ks.append(b.attn.phi_k.stable_a(k,False).reshape(-1).cpu())
    return torch.cat(qs),torch.cat(ks)


def replace_phi(ref,qitem=None,kitem=None):
    cand=copy.deepcopy(ref)
    if qitem is not None:
        _,spec,dec,_=qitem;cand.block.attn.phi_q=u.IzhPhi(ref.block.attn.phi_q,spec,dec)
    if kitem is not None:
        _,spec,dec,_=kitem;cand.block.attn.phi_k=u.IzhPhi(ref.block.attn.phi_k,spec,dec)
    return cand.eval()


@torch.no_grad()
def parity(ref,cand,gen,seed,batches=6,batch=64):
    g=torch.Generator();g.manual_seed(seed);pr=[];pc=[];tt=[];acts=[]
    Hnum=Hden=gnum=gden=0.0
    for _ in range(batches):
        x,t,_,_=gen.sample(batch,g)
        rr,sr=ref.fresh(x);cc,sc=cand.fresh(x)
        pr.append(rr);pc.append(cc);tt.append(t);acts.append(sc.get('hidden_activity',float('nan')))
        # fresh() does not expose H/g; repeat block step only for state parity.
        ch=torch.ones(x.shape[:2],dtype=torch.bool)
        _,hrs,_=ref.block.step(x,None,ch);_,hcs,_=cand.block.step(x,None,ch)
        Hnum+=float(torch.sum((hcs.H-hrs.H)**2));Hden+=float(torch.sum(hrs.H**2))
        gnum+=float(torch.sum((hcs.g-hrs.g)**2));gden+=float(torch.sum(hrs.g**2))
    pr=torch.cat(pr);pc=torch.cat(pc);tt=torch.cat(tt)
    return {'prediction_rel_l2':r.rel_l2(pc,pr),'prediction_cosine':r.cosine(pc,pr),
            'reference_task':r.metrics(pr,tt),'candidate_task':r.metrics(pc,tt),
            'H_rel_l2':float(np.sqrt(Hnum/max(Hden,1e-30))),
            'g_rel_l2':float(np.sqrt(gnum/max(gden,1e-30))),
            'hidden_activity':float(np.mean(acts))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='unified_snn_transform/run/U4C')
    ap.add_argument('--seed',type=int,default=71);ap.add_argument('--tokens',type=int,default=16)
    ap.add_argument('--d',type=int,default=32);ap.add_argument('--heads',type=int,default=2)
    ap.add_argument('--features',type=int,default=32);ap.add_argument('--ff',type=int,default=64)
    ap.add_argument('--value-dim',type=int,default=8);ap.add_argument('--theta-init',type=float,default=0.0)
    ap.add_argument('--train-steps',type=int,default=900);ap.add_argument('--train-batch',type=int,default=128)
    ap.add_argument('--eval-batches',type=int,default=8);ap.add_argument('--eval-batch',type=int,default=128)
    ap.add_argument('--lr',type=float,default=2e-3);ap.add_argument('--report-every',type=int,default=300)
    ap.add_argument('--topn',type=int,default=5);ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args();torch.set_num_threads(args.threads);u4b.seed_all(args.seed)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    gen=r.AssocGenerator(args.tokens,args.d,args.value_dim,args.seed)

    # Freeze the strongest U4B structural point: 25% for output, FF1 and FF2.
    ref=u4b.SparseRetrievalModel(args,args.seed+1001,.25,.25,.25)
    ref,hist=u4b.train_model(ref,gen,args,args.seed+2001)
    exact_task,act,acct=u4b.eval_sparse(ref,gen,args,args.seed+3001)
    shuffled=u4b.shuffled_control(ref,gen,args,args.seed+3001)

    aq,ak=collect_a(ref,gen,args.seed+4001)
    qtop,_=u31.expanded_candidates(aq,args.features,args.seed+5001,args.topn)
    ktop,_=u31.expanded_candidates(ak,args.features,args.seed+5002,args.topn)

    # Pair selection uses only parity to the exact sparse model, not task labels.
    best=None;pair_rows=[]
    for qi,qitem in enumerate(qtop):
        for ki,kitem in enumerate(ktop):
            cand=replace_phi(ref,qitem,kitem)
            m=parity(ref,cand,gen,args.seed+6001,batches=3,batch=64)
            row={'q_rank':qi,'k_rank':ki,'prediction_rel_l2':m['prediction_rel_l2'],
                 'prediction_cosine':m['prediction_cosine'],
                 'q_window':qitem[1]['window_ms'],'k_window':kitem[1]['window_ms']}
            pair_rows.append(row)
            score=m['prediction_rel_l2']+1e-5*(qitem[1]['window_ms']+kitem[1]['window_ms'])
            if best is None or score<best[0]: best=(score,qi,ki,qitem,kitem)
    _,qi,ki,qitem,kitem=best
    cand=replace_phi(ref,qitem,kitem)
    held=parity(ref,cand,gen,args.seed+7001,batches=args.eval_batches,batch=min(64,args.eval_batch))

    gates={
        'exact_sparse_task_sign_ge_0p95':exact_task['sign_accuracy']>=.95,
        'binding_control_drop_ge_0p20':exact_task['sign_accuracy']-shuffled['sign_accuracy']>=.20,
        'izh_prediction_rel_l2_le_1pct':held['prediction_rel_l2']<=.01,
        'izh_prediction_cosine_ge_0p999':held['prediction_cosine']>=.999,
        'izh_task_sign_drop_le_0p02':held['candidate_task']['sign_accuracy']>=held['reference_task']['sign_accuracy']-.02,
        'tail_effective_fraction_le_0p30':acct['effective_fraction']<=.30,
    }
    result={'stage':'UNIFIED_U4C_SPARSE_TAIL_PLUS_IZH','config':vars(args),
            'architecture':{'output_density':.25,'ff1_density':.25,'ff2_density':.25,
                            'feature_map':'trained-distribution Izh Q/K','attention':'persistent positive-RF H/g'},
            'exact_sparse_task':exact_task,'shuffled_control':shuffled,
            'binding_sign_drop':exact_task['sign_accuracy']-shuffled['sign_accuracy'],
            'hidden_activity':act,'tail_compute':acct,'train_history':hist,
            'selected_Q':qitem[3],'selected_K':kitem[3],
            'pair_selection':sorted(pair_rows,key=lambda z:z['prediction_rel_l2']),
            'heldout_izh_parity':held,'gates':gates,'PASS':all(gates.values())}
    result['decision']=('freeze U4 sparse-tail + Izh lightweight block and proceed to canonical integration/robustness gate'
                        if result['PASS'] else 'do not freeze U4; localize Izh/sparse-tail composition error')
    (out/'U4C_RESULTS.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    if not result['PASS']: raise SystemExit(2)

if __name__=='__main__':main()
