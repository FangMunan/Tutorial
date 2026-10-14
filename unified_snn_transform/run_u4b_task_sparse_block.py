#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""U4B — task-trained sparse output projection + sparse/activity-sparse FFN.

This corrects the deliberately harsh U4A teacher-matching screen. U4A asked a
sparse tail to reproduce an arbitrary frozen dense random mapping, which is not
the mathematical claim behind sparse neural networks. U4B asks the right
architecture question: when trained on the SAME task from scratch, can a
structurally sparse block reach the same function while using fewer synapses
and fewer active hidden fan-outs?

The already-validated linear-attention/H-g core is kept unchanged. Only:
  1) attention output projection; and
  2) FFN channel mixing
are replaced by fixed structural sparse connections. FFN uses a thresholded
ReLU population so inactive hidden units generate exactly zero output.
"""
from __future__ import annotations

import argparse, json, math, os, random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import run_unified_gate as u
import run_u3_trained_retrieval as r


def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def balanced_mask(out_features:int,in_features:int,density:float):
    # Equal-ish fan-in per postsynaptic neuron; mask uses Linear weight layout.
    k=max(1,min(in_features,int(round(density*in_features))))
    m=torch.zeros(out_features,in_features)
    for o in range(out_features):
        idx=torch.randperm(in_features)[:k]
        m[o,idx]=1.0
    return m


class SparseLinear(nn.Module):
    def __init__(self,in_features:int,out_features:int,density:float,bias:bool=True):
        super().__init__(); self.in_features=in_features;self.out_features=out_features
        self.weight=nn.Parameter(torch.empty(out_features,in_features))
        self.bias=nn.Parameter(torch.zeros(out_features)) if bias else None
        self.register_buffer('mask',balanced_mask(out_features,in_features,density))
        nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        if self.bias is not None:
            bound=1/math.sqrt(in_features);nn.init.uniform_(self.bias,-bound,bound)
    def forward(self,x):
        return F.linear(x,self.weight*self.mask,self.bias)
    @property
    def nnz(self): return int(self.mask.sum().item())
    @property
    def density(self): return float(self.mask.mean().item())


class SparseUnifiedBlock(nn.Module):
    """Same attention/H-g computation as UnifiedBlock; sparse tail only."""
    def __init__(self,d:int,heads:int,features:int,ff:int,seed:int,
                 out_density:float,ff1_density:float,ff2_density:float,theta_init:float):
        super().__init__(); self.d=d;self.heads=heads;self.ff=ff
        self.n1=nn.LayerNorm(d)
        self.attn=u.LinearAttentionCore(d,heads,features,seed)
        # Replace only the post-attention channel mixer.
        self.attn.out=SparseLinear(d,d,out_density,bias=True)
        self.n2=nn.LayerNorm(d)
        self.f1=SparseLinear(d,ff,ff1_density,bias=True)
        self.f2=SparseLinear(ff,d,ff2_density,bias=True)
        self.theta=nn.Parameter(torch.full((ff,),float(theta_init)))
        self.gate=u.SelectiveRetention(heads)

    def step(self,x,state,changed):
        event_ratio=changed.float().mean(1); alpha=self.gate(event_ratio)
        a,state,stats=self.attn.step(self.n1(x),state,alpha,changed)
        z=x+a
        pre=self.f1(self.n2(z))
        # Exact zeros below threshold; no soft-vs-hard train/eval mismatch.
        h=F.relu(pre-self.theta)
        y=z+self.f2(h)
        stats.update({'alpha_mean':float(alpha.mean()),'event_ratio':float(event_ratio.mean()),
                      'hidden_activity':float((h>0).float().mean().detach())})
        return y,state,stats

    def synapse_account(self,hidden_activity:float):
        d=self.d;ff=self.ff
        dense=d*d+d*ff+ff*d
        static=self.attn.out.nnz+self.f1.nnz+self.f2.nnz
        effective=self.attn.out.nnz+self.f1.nnz+hidden_activity*self.f2.nnz
        return {'dense_tail_synapses':dense,'static_sparse_synapses':static,
                'static_fraction':static/dense,'effective_synops_proxy':effective,
                'effective_fraction':effective/dense,'effective_reduction':1-effective/dense}


class SparseRetrievalModel(nn.Module):
    def __init__(self,args,seed:int,out_d:float,f1_d:float,f2_d:float):
        super().__init__()
        self.block=SparseUnifiedBlock(args.d,args.heads,args.features,args.ff,seed,
                                      out_d,f1_d,f2_d,args.theta_init)
        self.readout=nn.Linear(args.d,args.value_dim)
    def fresh(self,x):
        changed=torch.ones(x.shape[:2],dtype=torch.bool,device=x.device)
        y,_,stats=self.block.step(x,None,changed)
        return self.readout(y[:,-1]),stats


def train_model(model,gen,args,seed):
    model.train();opt=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(.9,.98),weight_decay=1e-4)
    g=torch.Generator();g.manual_seed(seed);hist=[]
    for st in range(1,args.train_steps+1):
        x,tgt,_,_=gen.sample(args.train_batch,g)
        pred,stats=model.fresh(x)
        task=F.mse_loss(pred,tgt)
        # Smooth activity proxy only as a regularizer; forward path stays hard-zero ReLU.
        b=model.block
        # Recompute a cheap proxy from theta magnitude is not meaningful; instead use observed
        # hard activity as a detached diagnostic and L1 hidden-output pressure via f2 input is
        # implicit in weight decay. Keep loss task-pure in this first functional gate.
        loss=task
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if st==1 or st%args.report_every==0 or st==args.train_steps:
            m=r.metrics(pred,tgt);row={'step':st,'loss':float(loss.detach()),**m,
                                      'hidden_activity':stats.get('hidden_activity',float('nan'))}
            hist.append(row);print('TRAIN',json.dumps(row),flush=True)
    return model.eval(),hist


@torch.no_grad()
def eval_sparse(model,gen,args,seed):
    g=torch.Generator();g.manual_seed(seed);ps=[];ts=[];acts=[]
    for _ in range(args.eval_batches):
        x,tgt,_,_=gen.sample(args.eval_batch,g);p,s=model.fresh(x)
        ps.append(p);ts.append(tgt);acts.append(s['hidden_activity'])
    met=r.metrics(torch.cat(ps),torch.cat(ts));a=float(np.mean(acts))
    return met,a,model.block.synapse_account(a)


@torch.no_grad()
def shuffled_control(model,gen,args,seed):
    model.eval();g=torch.Generator();g.manual_seed(seed);ps=[];ts=[]
    for _ in range(args.eval_batches):
        x,tgt,_,vals=gen.sample(args.eval_batch,g)
        perm=torch.stack([torch.randperm(gen.slots,generator=g) for _ in range(args.eval_batch)])
        kd,vd=gen.key_dim,gen.value_dim
        sh=vals[torch.arange(args.eval_batch)[:,None],perm]
        x[:,:gen.slots,kd:kd+vd]=sh
        p,_=model.fresh(x);ps.append(p);ts.append(tgt)
    return r.metrics(torch.cat(ps),torch.cat(ts))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='unified_snn_transform/run/U4B')
    ap.add_argument('--seed',type=int,default=53);ap.add_argument('--tokens',type=int,default=16)
    ap.add_argument('--d',type=int,default=32);ap.add_argument('--heads',type=int,default=2)
    ap.add_argument('--features',type=int,default=32);ap.add_argument('--ff',type=int,default=64)
    ap.add_argument('--value-dim',type=int,default=8);ap.add_argument('--theta-init',type=float,default=0.0)
    ap.add_argument('--train-steps',type=int,default=900);ap.add_argument('--train-batch',type=int,default=128)
    ap.add_argument('--eval-batches',type=int,default=8);ap.add_argument('--eval-batch',type=int,default=128)
    ap.add_argument('--lr',type=float,default=2e-3);ap.add_argument('--report-every',type=int,default=300)
    ap.add_argument('--task-gate',type=float,default=.95);ap.add_argument('--binding-drop-gate',type=float,default=.20)
    ap.add_argument('--effective-fraction-gate',type=float,default=.55);ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args();torch.set_num_threads(args.threads);seed_all(args.seed)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    gen=r.AssocGenerator(args.tokens,args.d,args.value_dim,args.seed)

    grid=[
        (1.0,0.50,0.50),
        (0.50,0.50,0.50),
        (0.50,0.25,0.50),
        (0.50,0.25,0.25),
        (0.25,0.50,0.50),
        (0.25,0.25,0.50),
        (0.25,0.25,0.25),
    ]
    rows=[]
    for i,(do,d1,d2) in enumerate(grid):
        seed_all(args.seed+100+i)
        model=SparseRetrievalModel(args,args.seed+1000+i,do,d1,d2)
        model,hist=train_model(model,gen,args,args.seed+2000+i)
        met,act,acct=eval_sparse(model,gen,args,args.seed+3000+i)
        shuf=shuffled_control(model,gen,args,args.seed+3000+i)
        drop=met['sign_accuracy']-shuf['sign_accuracy']
        ok=(met['sign_accuracy']>=args.task_gate and drop>=args.binding_drop_gate and
            acct['effective_fraction']<=args.effective_fraction_gate)
        row={'output_density':do,'ff1_density':d1,'ff2_density':d2,
             'task':met,'shuffled_control':shuf,'binding_sign_drop':drop,
             'hidden_activity':act,'compute':acct,'PASS':ok,'train_history':hist}
        rows.append(row);print('CAND',json.dumps({k:v for k,v in row.items() if k!='train_history'}),flush=True)

    passing=[x for x in rows if x['PASS']]
    # Prefer least effective work; break ties by task MSE.
    best=min(passing,key=lambda x:(x['compute']['effective_fraction'],x['task']['mse'])) if passing else None
    result={'stage':'UNIFIED_U4B_TASK_TRAINED_SPARSE_BLOCK','config':vars(args),'rows':rows,
            'best_passing':best,
            'gates':{'task_sign_accuracy_ge':args.task_gate,'binding_drop_ge':args.binding_drop_gate,
                     'effective_fraction_le':args.effective_fraction_gate},
            'PASS':bool(passing),
            'decision':('promote best sparse output+FFN candidate to full unified-block integration'
                        if passing else 'structural sparse tail not yet established on task; revise topology before integration')}
    (out/'U4B_RESULTS.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'stage':result['stage'],'PASS':result['PASS'],
                      'best_passing':None if best is None else {k:v for k,v in best.items() if k!='train_history'},
                      'decision':result['decision']},indent=2),flush=True)
    if not result['PASS']: raise SystemExit(2)

if __name__=='__main__': main()
