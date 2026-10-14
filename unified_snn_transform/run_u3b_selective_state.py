#!/usr/bin/env python3
from __future__ import annotations

import argparse, copy, json, math, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import run_unified_gate as u


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


class EventSelectiveGate(nn.Module):
    """Generic Mamba-inspired per-head retention driven by the current event content.

    This is deliberately not called a Mamba block: only the retention timescale is
    input dependent.  The summary contains only changed-token content, so no task
    label or host-side reset flag is fed to the gate.
    """
    def __init__(self,d:int,heads:int,init_alpha=.99):
        super().__init__()
        target=max(1e-5,-math.log(init_alpha));s0=math.log(max(1e-8,math.exp(target)-1.))
        self.theta0=nn.Parameter(torch.full((heads,),s0))
        self.proj=nn.Linear(d,heads,bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self,x:torch.Tensor,changed:torch.Tensor):
        w=changed.to(x.dtype)[:,:,None]
        summary=(x*w).sum(1)/w.sum(1).clamp_min(1.)
        s=self.theta0[None]+self.proj(summary)
        return torch.exp(-F.softplus(s)).clamp(1e-4,1.)


class FixedGate(nn.Module):
    def __init__(self,heads:int,init_alpha=.99):
        super().__init__();target=max(1e-5,-math.log(init_alpha));s0=math.log(max(1e-8,math.exp(target)-1.))
        self.theta0=nn.Parameter(torch.full((heads,),s0))
    def forward(self,x,changed):
        a=torch.exp(-F.softplus(self.theta0)).clamp(1e-4,1.)
        return a[None].expand(x.shape[0],-1)


class TemporalBlock(nn.Module):
    def __init__(self,d,heads,features,ff,seed,gate_kind='selective'):
        super().__init__();self.d=d;self.heads=heads
        self.n1=nn.LayerNorm(d);self.attn=u.LinearAttentionCore(d,heads,features,seed)
        self.n2=nn.LayerNorm(d);self.f1=nn.Linear(d,ff);self.f2=nn.Linear(ff,d)
        self.gate=EventSelectiveGate(d,heads) if gate_kind=='selective' else FixedGate(heads)

    def step(self,x,state,changed,force_identity=False):
        alpha=torch.ones(x.shape[0],self.heads,device=x.device,dtype=x.dtype) if force_identity else self.gate(x,changed)
        a,state,stats=self.attn.step(self.n1(x),state,alpha,changed)
        z=x+a;y=z+self.f2(F.gelu(self.f1(self.n2(z))))
        return y,state,{'alpha':alpha,'alpha_mean':float(alpha.mean()),**stats}


class TemporalModel(nn.Module):
    def __init__(self,d,heads,features,ff,value_dim,seed,gate_kind):
        super().__init__();self.block=TemporalBlock(d,heads,features,ff,seed,gate_kind)
        self.readout=nn.Linear(d,value_dim)


class ResetMemoryTask:
    """Two-episode associative memory.

    Episode-1 and episode-2 use the SAME address keys but independent random values.
    Episode-2 writes into new physical slots, so old episode-1 traces remain unless
    the persistent state is forgotten.  A reset event is encoded only in token
    content.  This makes input-dependent retention useful for a reason that cannot be
    solved by a fixed key-to-value lookup.
    """
    def __init__(self,tokens,d,value_dim,nkeys=4,reset_amp=8.0):
        if 2*nkeys+2>tokens: raise ValueError('Need 2*nkeys + reset + query tokens')
        if nkeys+value_dim+3>d: raise ValueError('d too small')
        self.tokens=tokens;self.d=d;self.vd=value_dim;self.k=nkeys;self.reset_amp=reset_amp
        self.v0=nkeys;self.mem_role=nkeys+value_dim;self.reset_dim=self.mem_role+1;self.query_role=self.mem_role+2
        self.reset_idx=2*nkeys;self.query_idx=2*nkeys+1;self.keys=torch.eye(nkeys)

    def _mem(self,x,slot0,vals):
        B=vals.shape[0];K=self.k
        x[:,slot0:slot0+K,:K]=self.keys[None]
        x[:,slot0:slot0+K,self.v0:self.v0+self.vd]=vals
        x[:,slot0:slot0+K,self.mem_role]=1.

    def sample(self,batch,g):
        B=batch;K=self.k;N=self.tokens;D=self.d;vd=self.vd
        v1=(torch.randint(0,2,(B,K,vd),generator=g).float()*2-1)
        v2=(torch.randint(0,2,(B,K,vd),generator=g).float()*2-1)
        q0=torch.randint(0,K,(B,),generator=g);q1=torch.randint(0,K,(B,),generator=g)
        q2=torch.randint(0,K,(B,),generator=g);q3=torch.randint(0,K,(B,),generator=g)
        def qtoken(x,q):
            x[:,self.query_idx,:K]=self.keys[q];x[:,self.query_idx,self.query_role]=1.
        # step0: episode 1 written
        x0=torch.zeros(B,N,D);self._mem(x0,0,v1);qtoken(x0,q0)
        # step1: only query changes, episode 1 must still be readable
        x1=x0.clone();x1[:,self.query_idx]=0.;qtoken(x1,q1)
        # step2: reset event only; no supervised readout
        x2=x1.clone();x2[:,self.reset_idx,self.reset_dim]=self.reset_amp
        # step3: episode 2 writes into new slots with same semantic keys
        x3=x2.clone();self._mem(x3,K,v2);x3[:,self.query_idx]=0.;qtoken(x3,q2)
        # step4: only query changes, episode 2 must persist
        x4=x3.clone();x4[:,self.query_idx]=0.;qtoken(x4,q3)
        xs=[x0,x1,x2,x3,x4]
        ms=[]
        ms.append(torch.ones(B,N,dtype=torch.bool))
        m=torch.zeros(B,N,dtype=torch.bool);m[:,self.query_idx]=True;ms.append(m)
        m=torch.zeros(B,N,dtype=torch.bool);m[:,self.reset_idx]=True;ms.append(m)
        m=torch.zeros(B,N,dtype=torch.bool);m[:,K:2*K]=True;m[:,self.query_idx]=True;ms.append(m)
        m=torch.zeros(B,N,dtype=torch.bool);m[:,self.query_idx]=True;ms.append(m)
        t0=v1[torch.arange(B),q0];t1=v1[torch.arange(B),q1]
        t3=v2[torch.arange(B),q2];t4=v2[torch.arange(B),q3]
        return xs,ms,{0:t0,1:t1,3:t3,4:t4}


def sign_acc(p,t): return float(((p>=0)==(t>=0)).float().mean().detach())


def forward_sequence(model,task,batch,g,force_identity=False):
    xs,ms,targets=task.sample(batch,g);state=None;loss=0.;rows={};alphas=[]
    for i,(x,m) in enumerate(zip(xs,ms)):
        y,state,st=model.block.step(x,state,m,force_identity=force_identity)
        alphas.append(st['alpha'])
        if i in targets:
            p=model.readout(y[:,task.query_idx]);t=targets[i]
            li=F.mse_loss(p,t);loss=loss+li
            rows[i]=(p,t,li)
    return loss/len(rows),rows,alphas


def clone_shared(src:TemporalModel,dst:TemporalModel):
    for n in ('n1','attn','n2','f1','f2'):
        getattr(dst.block,n).load_state_dict(copy.deepcopy(getattr(src.block,n).state_dict()))
    dst.readout.load_state_dict(copy.deepcopy(src.readout.state_dict()))


def train(model,task,args,seed):
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(.9,.98),weight_decay=1e-4)
    g=torch.Generator();g.manual_seed(seed);hist=[]
    for st in range(1,args.train_steps+1):
        loss,rows,alphas=forward_sequence(model,task,args.train_batch,g)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if st==1 or st%args.report_every==0 or st==args.train_steps:
            r={'step':st,'loss':float(loss.detach()),'pre_reset_sign':np.mean([sign_acc(*rows[i][:2]) for i in (0,1)]),
               'post_reset_sign':np.mean([sign_acc(*rows[i][:2]) for i in (3,4)]),
               'alpha_reset':float(alphas[2].mean().detach()),
               'alpha_nonreset':float(torch.stack([alphas[i].mean() for i in (0,1,3,4)]).mean().detach())}
            hist.append(r);print('TRAIN',json.dumps(r),flush=True)
    return hist


@torch.no_grad()
def evaluate(model,task,args,seed,force_identity=False):
    g=torch.Generator();g.manual_seed(seed);pre=[];post=[];mse=[];ares=[];anon=[]
    for _ in range(args.eval_batches):
        loss,rows,alphas=forward_sequence(model,task,args.eval_batch,g,force_identity=force_identity)
        pre += [sign_acc(*rows[i][:2]) for i in (0,1)]
        post += [sign_acc(*rows[i][:2]) for i in (3,4)]
        mse.append(float(loss));ares.append(float(alphas[2].mean()))
        anon.append(float(torch.stack([alphas[i].mean() for i in (0,1,3,4)]).mean()))
    return {'pre_reset_sign_accuracy':float(np.mean(pre)),'post_reset_sign_accuracy':float(np.mean(post)),
            'mse':float(np.mean(mse)),'alpha_reset':float(np.mean(ares)),'alpha_nonreset':float(np.mean(anon))}


def run(args):
    seed_all(args.seed);torch.set_num_threads(args.threads);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    task=ResetMemoryTask(args.tokens,args.d,args.value_dim,args.nkeys,args.reset_amp)
    selective=TemporalModel(args.d,args.heads,args.features,args.ff,args.value_dim,args.seed+101,'selective')
    fixed=TemporalModel(args.d,args.heads,args.features,args.ff,args.value_dim,args.seed+202,'fixed')
    clone_shared(selective,fixed)
    sh=train(selective,task,args,args.seed+1000)
    fh=train(fixed,task,args,args.seed+2000)
    se=evaluate(selective,task,args,args.seed+3000,False)
    sid=evaluate(selective,task,args,args.seed+3000,True)
    fe=evaluate(fixed,task,args,args.seed+3000,False)
    advantage=se['post_reset_sign_accuracy']-max(sid['post_reset_sign_accuracy'],fe['post_reset_sign_accuracy'])
    selectivity=se['alpha_nonreset']-se['alpha_reset']
    gates={
        'selective_pre_reset_ge_0p90':se['pre_reset_sign_accuracy']>=.90,
        'selective_post_reset_ge_0p90':se['post_reset_sign_accuracy']>=.90,
        'reset_alpha_lower_than_nonreset_by_0p20':selectivity>=.20,
        'selective_advantage_ge_0p10':advantage>=.10,
    }
    result={'stage':'UNIFIED_U3B_SELECTIVE_STATE_FUNCTION','config':vars(args),
            'selective_train':sh,'fixed_train':fh,'selective_eval':se,'selective_identity_ablation':sid,
            'matched_fixed_eval':fe,'post_reset_selective_advantage':advantage,'alpha_selectivity_gap':selectivity,
            'gates':gates,'PASS':all(gates.values())}
    if result['PASS']:
        result['decision']='retain generic event-content-dependent selective retention in the canonical block; it is functionally required on this temporal memory control.'
    else:
        result['decision']='do not claim Mamba-like selective retention is required; inspect whether the task or gate failed before keeping it canonical.'
    (out/'U3B_RESULTS.json').write_text(json.dumps(result,indent=2));torch.save({'model':selective.state_dict(),'config':vars(args)},out/'U3B_SELECTIVE.pt')
    print(json.dumps(result,indent=2),flush=True);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='unified_snn_transform/run/U3B')
    ap.add_argument('--seed',type=int,default=23);ap.add_argument('--tokens',type=int,default=16);ap.add_argument('--d',type=int,default=32)
    ap.add_argument('--heads',type=int,default=2);ap.add_argument('--features',type=int,default=32);ap.add_argument('--ff',type=int,default=64)
    ap.add_argument('--value-dim',type=int,default=8);ap.add_argument('--nkeys',type=int,default=4);ap.add_argument('--reset-amp',type=float,default=8.)
    ap.add_argument('--train-steps',type=int,default=1200);ap.add_argument('--train-batch',type=int,default=128);ap.add_argument('--lr',type=float,default=2e-3)
    ap.add_argument('--report-every',type=int,default=200);ap.add_argument('--eval-batches',type=int,default=8);ap.add_argument('--eval-batch',type=int,default=128)
    ap.add_argument('--threads',type=int,default=4);run(ap.parse_args())
if __name__=='__main__':main()
