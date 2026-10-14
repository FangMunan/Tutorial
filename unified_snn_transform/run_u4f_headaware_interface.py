#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""U4F — head-aware, attention-aware Q/K Izh interface calibration.

U4E showed that wider/clamp-aware scalar search improves K but does not meet
the strict <=1% full-block parity gate. U4D showed the sparse tail is not the
source of amplification. This stage therefore changes ONLY fixed Izh decoder
parameters at the Q/K interface:

  * same Izh RS dynamics family and fixed operating points;
  * one fixed linear membrane-trajectory decoder per attention head;
  * decoder calibration uses exact attention-core outputs, never task labels;
  * final joint Q/K calibration keeps all network/task weights frozen.

The forward interface remains: scalar preactivation -> Izh trajectory -> fixed
linear decoder -> positivity floor. No new learned network layer is added.
"""
from __future__ import annotations

import argparse, copy, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import run_unified_gate as u
import run_u3_trained_retrieval as r
import run_u4b_task_sparse_block as u4b
import run_u4c_full_sparse_izh as u4c


class HeadAwareIzhPhi(nn.Module):
    def __init__(self, exact, spec, decoders):
        super().__init__()
        self.head_dim = exact.head_dim; self.features = exact.features
        self.heads = int(exact.omega.shape[0])
        self.register_buffer('omega', exact.omega.detach().clone())
        self.register_buffer('decoders', decoders.detach().clone())
        self.bias=float(spec['bias']); self.gain=float(spec['gain']); self.window_ms=int(spec['window_ms'])
        self.last_clamp = 0.0

    def stable_a(self,x,is_query):
        xx=x*(self.head_dim**-0.25)
        dash=torch.einsum('bhnd,hmd->bhnm',xx,self.omega)
        diag=.5*(xx*xx).sum(-1,keepdim=True); a=dash-diag
        return (a-a.amax(-1,keepdim=True).detach()).clamp(-30.,0.) if is_query else a.clamp(-30.,30.)

    def forward(self,x,*,is_query:bool):
        a=self.stable_a(x,is_query); outs=[]; clamps=[]
        for h in range(self.heads):
            ah=a[:,h]
            X=u.traj_design(ah.reshape(-1),self.bias,self.gain,self.window_ms)
            raw=X@self.decoders[h].to(X.dtype)
            clamps.append(float((raw<=1e-8).float().mean()))
            outs.append(raw.clamp_min(1e-8).reshape_as(ah))
        self.last_clamp=float(np.mean(clamps))
        return torch.stack(outs,dim=1)


def ste_floor(raw, eps=1e-8):
    hard=raw.clamp_min(eps)
    return raw + (hard-raw).detach()


@torch.no_grad()
def collect_exact(model,gen,seed,batches=4,batch=48):
    g=torch.Generator(); g.manual_seed(seed); rec=[]; b=model.block
    for _ in range(batches):
        x,_,_,_=gen.sample(batch,g); z=b.n1(x)
        q=b.attn._split(b.attn.q(z)); k=b.attn._split(b.attn.k(z)); v=b.attn._split(b.attn.v(z))
        aq=b.attn.phi_q.stable_a(q,True); ak=b.attn.phi_k.stable_a(k,False)
        pq=b.attn.phi_q(q,is_query=True); pk=b.attn.phi_k(k,is_query=False)
        H=torch.einsum('bhnm,bhnd->bhmd',pk,v); gg=pk.sum(2)
        num=torch.einsum('bhnm,bhmd->bhnd',pq,H)
        den=torch.einsum('bhnm,bhm->bhn',pq,gg).unsqueeze(-1).clamp_min(1e-8)
        y=num/den
        rec.append((aq.cpu(),ak.cpu(),pq.cpu(),pk.cpu(),v.cpu(),y.cpu()))
    return tuple(torch.cat([z[i] for z in rec],dim=0) for i in range(6))


def initial_decoder(a,features,spec):
    aa=a.detach().flatten().cpu(); g=torch.Generator(); g.manual_seed(9017+len(aa)+int(spec['window_ms']))
    cap=min(len(aa),30000)
    if len(aa)>cap: aa=aa[torch.randperm(len(aa),generator=g)[:cap]]
    y=torch.exp(aa.clamp(-30,30))*(features**-.5)+1e-6
    return u.fit_decoder(aa,y,**spec)


def make_X(a,spec):
    # a [B,H,N,M] -> list of [B,N,M,P]
    xs=[]
    for h in range(a.shape[1]):
        ah=a[:,h]
        X=u.traj_design(ah.reshape(-1),**spec).reshape(*ah.shape,-1)
        xs.append(X)
    return xs


def attention_from_decoders(Xq,Xk,dq,dk,v,eps=1e-8):
    pqs=[];pks=[];neg=[]
    for h in range(len(Xq)):
        rq=torch.einsum('bnmp,p->bnm',Xq[h],dq[h]); rk=torch.einsum('bnmp,p->bnm',Xk[h],dk[h])
        neg.append(F.relu(eps-rq).mean()); neg.append(F.relu(eps-rk).mean())
        pqs.append(ste_floor(rq,eps));pks.append(ste_floor(rk,eps))
    pq=torch.stack(pqs,1);pk=torch.stack(pks,1)
    H=torch.einsum('bhnm,bhnd->bhmd',pk,v);gg=pk.sum(2)
    num=torch.einsum('bhnm,bhmd->bhnd',pq,H)
    den=torch.einsum('bhnm,bhm->bhn',pq,gg).unsqueeze(-1).clamp_min(1e-8)
    return num/den, torch.stack(neg).mean(), pq, pk


def optimize_decoders(data,qspec,kspec,features,steps=220,lr=8e-3):
    aq,ak,pq0,pk0,v,target=data
    Xq=make_X(aq,qspec);Xk=make_X(ak,kspec);H=aq.shape[1]
    q0=initial_decoder(aq,features,qspec);k0=initial_decoder(ak,features,kspec)
    dq=nn.Parameter(q0.repeat(H,1));dk=nn.Parameter(k0.repeat(H,1))
    opt=torch.optim.Adam([dq,dk],lr=lr)
    hist=[]
    scale=target.square().mean().detach().clamp_min(1e-8)
    for st in range(1,steps+1):
        y,neg,pq,pk=attention_from_decoders(Xq,Xk,dq,dk,v)
        attn=((y-target)**2).mean()/scale
        # Keep the neuronal interface close to the intended positive-RF map;
        # this regularizes the calibration and protects held-out generalization.
        phi=((pq-pq0)**2).mean()/pq0.square().mean().clamp_min(1e-8)
        phi=phi+((pk-pk0)**2).mean()/pk0.square().mean().clamp_min(1e-8)
        loss=attn + 0.02*phi + 0.25*neg
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([dq,dk],5.0);opt.step()
        if st in (1,steps//2,steps):
            hist.append({'step':st,'loss':float(loss.detach()),'attn_loss':float(attn.detach()),
                         'phi_reg':float(phi.detach()),'neg_penalty':float(neg.detach())})
    return dq.detach(),dk.detach(),hist


@torch.no_grad()
def scalar_metrics(module,exact,a,is_query):
    # Evaluate feature-map error directly on supplied preactivations using the
    # same trajectory/readout, without rerunning projections.
    ys=[];ps=[];cl=[]
    target=torch.exp(a.clamp(-30,30))*(module.features**-.5)+1e-6
    for h in range(a.shape[1]):
        ah=a[:,h];X=u.traj_design(ah.reshape(-1),module.bias,module.gain,module.window_ms)
        raw=X@module.decoders[h]
        cl.append(float((raw<=1e-8).float().mean()))
        ps.append(raw.clamp_min(1e-8).reshape_as(ah));ys.append(target[:,h])
    p=torch.stack(ps,1);y=torch.stack(ys,1)
    return {'rel_l2':float(torch.linalg.vector_norm(p-y)/torch.linalg.vector_norm(y).clamp_min(1e-12)),
            'clamp_fraction':float(np.mean(cl))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='unified_snn_transform/run/U4F')
    ap.add_argument('--seed',type=int,default=71);ap.add_argument('--tokens',type=int,default=16)
    ap.add_argument('--d',type=int,default=32);ap.add_argument('--heads',type=int,default=2)
    ap.add_argument('--features',type=int,default=32);ap.add_argument('--ff',type=int,default=64)
    ap.add_argument('--value-dim',type=int,default=8);ap.add_argument('--theta-init',type=float,default=0.0)
    ap.add_argument('--train-steps',type=int,default=900);ap.add_argument('--train-batch',type=int,default=128)
    ap.add_argument('--eval-batches',type=int,default=12);ap.add_argument('--eval-batch',type=int,default=64)
    ap.add_argument('--lr',type=float,default=2e-3);ap.add_argument('--report-every',type=int,default=300)
    ap.add_argument('--cal-steps',type=int,default=220);ap.add_argument('--cal-lr',type=float,default=8e-3)
    ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args();torch.set_num_threads(args.threads);u4b.seed_all(args.seed)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    gen=r.AssocGenerator(args.tokens,args.d,args.value_dim,args.seed)

    ref=u4b.SparseRetrievalModel(args,args.seed+1001,.25,.25,.25)
    ref,hist=u4b.train_model(ref,gen,args,args.seed+2001)
    exact_task,act,acct=u4b.eval_sparse(ref,gen,args,args.seed+3001)

    # Fixed operating points are taken from U4E's best held-out interface.
    # U4F changes only the fixed linear trajectory decoders, per head.
    qspec={'bias':4.0,'gain':0.75,'window_ms':16}
    kspec={'bias':1.0,'gain':0.35,'window_ms':24}
    cal=collect_exact(ref,gen,args.seed+4101,batches=4,batch=48)
    dq,dk,cal_hist=optimize_decoders(cal,qspec,kspec,args.features,args.cal_steps,args.cal_lr)

    cand=copy.deepcopy(ref)
    cand.block.attn.phi_q=HeadAwareIzhPhi(ref.block.attn.phi_q,qspec,dq)
    cand.block.attn.phi_k=HeadAwareIzhPhi(ref.block.attn.phi_k,kspec,dk)
    cand.eval()
    held=u4c.parity(ref,cand,gen,args.seed+8001,batches=args.eval_batches,batch=args.eval_batch)

    # Independent held-out feature audit on a different unlabeled sample.
    test=collect_exact(ref,gen,args.seed+9001,batches=3,batch=48)
    aq,ak=test[0],test[1]
    q_scalar=scalar_metrics(cand.block.attn.phi_q,ref.block.attn.phi_q,aq,True)
    k_scalar=scalar_metrics(cand.block.attn.phi_k,ref.block.attn.phi_k,ak,False)
    qsp=u.izh_activity(aq.reshape(-1)[:5000],**qspec);ksp=u.izh_activity(ak.reshape(-1)[:5000],**kspec)

    gates={
        'exact_sparse_task_sign_ge_0p95':exact_task['sign_accuracy']>=.95,
        'QK_prediction_rel_l2_le_1pct':held['prediction_rel_l2']<=.01,
        'QK_prediction_cosine_ge_0p999':held['prediction_cosine']>=.999,
        'QK_task_sign_drop_le_0p02':held['candidate_task']['sign_accuracy']>=held['reference_task']['sign_accuracy']-.02,
        'tail_effective_fraction_le_0p30':acct['effective_fraction']<=.30,
        'task_and_sparse_architecture_weights_frozen_during_interface_calibration':True,
    }
    result={'stage':'UNIFIED_U4F_HEADAWARE_ATTENTION_CALIBRATION','config':vars(args),
            'qspec':qspec,'kspec':kspec,'calibration_history':cal_hist,
            'exact_sparse_task':exact_task,'tail_compute':acct,'hidden_activity':act,'train_history':hist,
            'heldout_QK':held,'heldout_scalar':{'Q':q_scalar,'K':k_scalar},
            'spike_activity':{'Q_mean_spikes':qsp[0],'Q_active_fraction':qsp[1],
                              'K_mean_spikes':ksp[0],'K_active_fraction':ksp[1]},
            'gates':gates,'PASS':all(gates.values())}
    result['decision']=('freeze head-aware Izh interface and proceed to canonical full-block freeze gate'
                        if result['PASS'] else
                        'head-aware decoder calibration still misses strict gate; next step is per-head operating-point search, not sparse-tail redesign')
    (out/'U4F_RESULTS.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'stage':result['stage'],'PASS':result['PASS'],'qspec':qspec,'kspec':kspec,
                      'calibration_history':cal_hist,'heldout_QK':held,'heldout_scalar':result['heldout_scalar'],
                      'spike_activity':result['spike_activity'],'gates':gates,'decision':result['decision']},indent=2),flush=True)


if __name__=='__main__':main()
