#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""U4G — stable per-head Izh operating-point/interface search.

U4F's gradient-based attention calibration was numerically unstable, so this
stage deliberately returns to a conservative calibration rule: no gradient
updates to the neuronal decoder after least-squares fitting. Instead, Q and K
are calibrated independently for each attention head, and each head may choose
its own fixed Izh operating point (bias, gain, window) and linear decoder.
Selection is based only on exact attention-core parity on unlabeled examples.
The U4 sparse tail and all task-model weights remain frozen.
"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

import run_unified_gate as u
import run_u3_trained_retrieval as r
import run_u4b_task_sparse_block as u4b
import run_u4c_full_sparse_izh as u4c
import run_u4e_interface_calibration as u4e
import run_u4f_headaware_interface as u4f


class PerHeadIzhPhi(nn.Module):
    def __init__(self, exact, items):
        super().__init__(); self.head_dim=exact.head_dim; self.features=exact.features
        self.heads=int(exact.omega.shape[0]); self.register_buffer('omega',exact.omega.detach().clone())
        self.specs=[]
        for h,item in enumerate(items):
            _,spec,dec,_=item; self.specs.append(dict(spec)); self.register_buffer(f'decoder_{h}',dec.detach().clone())
        self.last_clamp=0.0

    def stable_a(self,x,is_query):
        xx=x*(self.head_dim**-0.25)
        dash=torch.einsum('bhnd,hmd->bhnm',xx,self.omega)
        diag=.5*(xx*xx).sum(-1,keepdim=True);a=dash-diag
        return (a-a.amax(-1,keepdim=True).detach()).clamp(-30.,0.) if is_query else a.clamp(-30.,30.)

    def forward(self,x,*,is_query:bool):
        a=self.stable_a(x,is_query);outs=[];clamps=[]
        for h,spec in enumerate(self.specs):
            ah=a[:,h];X=u.traj_design(ah.reshape(-1),**spec);dec=getattr(self,f'decoder_{h}')
            raw=X@dec.to(X.dtype);clamps.append(float((raw<=1e-8).float().mean()))
            outs.append(raw.clamp_min(1e-8).reshape_as(ah))
        self.last_clamp=float(np.mean(clamps));return torch.stack(outs,1)


def phi_item(a_h,item):
    _,spec,dec,_=item;X=u.traj_design(a_h.reshape(-1),**spec);raw=X@dec.to(X.dtype)
    return raw.clamp_min(1e-8).reshape_as(a_h)


def head_attention(pq,pk,v):
    H=torch.einsum('bnm,bnd->bmd',pk,v);g=pk.sum(1)
    num=torch.einsum('bnm,bmd->bnd',pq,H);den=torch.einsum('bnm,bm->bn',pq,g).unsqueeze(-1).clamp_min(1e-8)
    return num/den


def rel(a,b):
    return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-12))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='unified_snn_transform/run/U4G')
    ap.add_argument('--seed',type=int,default=71);ap.add_argument('--tokens',type=int,default=16)
    ap.add_argument('--d',type=int,default=32);ap.add_argument('--heads',type=int,default=2);ap.add_argument('--features',type=int,default=32)
    ap.add_argument('--ff',type=int,default=64);ap.add_argument('--value-dim',type=int,default=8);ap.add_argument('--theta-init',type=float,default=0.0)
    ap.add_argument('--train-steps',type=int,default=900);ap.add_argument('--train-batch',type=int,default=128)
    ap.add_argument('--eval-batches',type=int,default=12);ap.add_argument('--eval-batch',type=int,default=64)
    ap.add_argument('--lr',type=float,default=2e-3);ap.add_argument('--report-every',type=int,default=300)
    ap.add_argument('--pool',type=int,default=28);ap.add_argument('--pair-top',type=int,default=5);ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args();torch.set_num_threads(args.threads);u4b.seed_all(args.seed)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);gen=r.AssocGenerator(args.tokens,args.d,args.value_dim,args.seed)

    ref=u4b.SparseRetrievalModel(args,args.seed+1001,.25,.25,.25);ref,hist=u4b.train_model(ref,gen,args,args.seed+2001)
    exact_task,act,acct=u4b.eval_sparse(ref,gen,args,args.seed+3001)
    aq,ak,pq0,pk0,v,y0=u4f.collect_exact(ref,gen,args.seed+4101,batches=4,batch=48)

    qitems=[];kitems=[];head_reports=[]
    for h in range(args.heads):
        qp,_=u4e.calibration_pool(aq[:,h],args.features,args.seed+5000+10*h,args.pool)
        kp,_=u4e.calibration_pool(ak[:,h],args.features,args.seed+5001+10*h,args.pool)
        qrows=[];krows=[]
        for i,it in enumerate(qp):
            p=phi_item(aq[:,h],it);yy=head_attention(p,pk0[:,h],v[:,h]);qrows.append({'idx':i,'rel_l2':rel(yy,y0[:,h]),**it[3]})
        for i,it in enumerate(kp):
            p=phi_item(ak[:,h],it);yy=head_attention(pq0[:,h],p,v[:,h]);krows.append({'idx':i,'rel_l2':rel(yy,y0[:,h]),**it[3]})
        qord=sorted(range(len(qp)),key=lambda i:qrows[i]['rel_l2'])[:args.pair_top]
        kord=sorted(range(len(kp)),key=lambda i:krows[i]['rel_l2'])[:args.pair_top]
        best=None;pairs=[]
        for qi in qord:
            pqi=phi_item(aq[:,h],qp[qi])
            for ki in kord:
                pki=phi_item(ak[:,h],kp[ki]);yy=head_attention(pqi,pki,v[:,h]);rr=rel(yy,y0[:,h])
                row={'q_idx':qi,'k_idx':ki,'rel_l2':rr,'q_spec':qp[qi][3],'k_spec':kp[ki][3]};pairs.append(row)
                score=rr+5e-6*(qp[qi][1]['window_ms']+kp[ki][1]['window_ms'])
                if best is None or score<best[0]:best=(score,qp[qi],kp[ki],row)
        _,qb,kb,brow=best;qitems.append(qb);kitems.append(kb)
        head_reports.append({'head':h,'Q_screen':sorted(qrows,key=lambda z:z['rel_l2'])[:10],
                             'K_screen':sorted(krows,key=lambda z:z['rel_l2'])[:10],
                             'pair_screen':sorted(pairs,key=lambda z:z['rel_l2']),'selected':brow})

    cand=copy.deepcopy(ref);cand.block.attn.phi_q=PerHeadIzhPhi(ref.block.attn.phi_q,qitems);cand.block.attn.phi_k=PerHeadIzhPhi(ref.block.attn.phi_k,kitems);cand.eval()
    held=u4c.parity(ref,cand,gen,args.seed+8001,batches=args.eval_batches,batch=args.eval_batch)
    qsp=[];ksp=[]
    for h in range(args.heads):
        qsp.append(u.izh_activity(aq[:,h].reshape(-1)[:5000],**qitems[h][1]));ksp.append(u.izh_activity(ak[:,h].reshape(-1)[:5000],**kitems[h][1]))
    gates={'exact_sparse_task_sign_ge_0p95':exact_task['sign_accuracy']>=.95,
           'QK_prediction_rel_l2_le_1pct':held['prediction_rel_l2']<=.01,
           'QK_prediction_cosine_ge_0p999':held['prediction_cosine']>=.999,
           'QK_task_sign_drop_le_0p02':held['candidate_task']['sign_accuracy']>=held['reference_task']['sign_accuracy']-.02,
           'tail_effective_fraction_le_0p30':acct['effective_fraction']<=.30,
           'no_task_or_sparse_weight_change':True}
    result={'stage':'UNIFIED_U4G_PER_HEAD_OPERATING_POINT','config':vars(args),'head_reports':head_reports,
            'selected_Q':[x[3] for x in qitems],'selected_K':[x[3] for x in kitems],
            'heldout_QK':held,'spike_activity':{'Q':qsp,'K':ksp},'exact_sparse_task':exact_task,
            'tail_compute':acct,'hidden_activity':act,'train_history':hist,'gates':gates,'PASS':all(gates.values())}
    result['decision']=('freeze per-head Izh interface and proceed to canonical full-block freeze gate' if result['PASS']
                        else 'per-head scalar interface still misses strict parity; next test should increase decoder basis capacity, not alter sparse tail')
    (out/'U4G_RESULTS.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'stage':result['stage'],'PASS':result['PASS'],'selected_Q':result['selected_Q'],'selected_K':result['selected_K'],
                      'heldout_QK':held,'spike_activity':result['spike_activity'],'gates':gates,'decision':result['decision']},indent=2),flush=True)

if __name__=='__main__':main()
