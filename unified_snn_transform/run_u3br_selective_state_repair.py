#!/usr/bin/env python3
from __future__ import annotations

import argparse, copy, json, math, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import run_unified_gate as u


def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def sign_acc(p,t):
    return float(((p>=0)==(t>=0)).float().mean().detach())


class ContentSelectiveGate(nn.Module):
    """Per-head retention from changed-input content only.

    The gate sees a summary of changed input events.  A control event is allowed to
    affect retention without being written into H/g; this separates state control
    from associative-memory content and avoids the exp-feature blow-up in U3B.
    """
    def __init__(self,d:int,heads:int,init_alpha:float=.995):
        super().__init__()
        target=max(1e-6,-math.log(init_alpha))
        s0=math.log(max(1e-8,math.exp(target)-1.0))
        self.theta0=nn.Parameter(torch.full((heads,),s0))
        self.proj=nn.Linear(d,heads,bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self,gate_x:torch.Tensor,gate_changed:torch.Tensor):
        w=gate_changed.to(gate_x.dtype)[:,:,None]
        summary=(gate_x*w).sum(1)/w.sum(1).clamp_min(1.0)
        s=self.theta0[None]+self.proj(summary)
        return torch.exp(-F.softplus(s)).clamp(1e-5,1.0)


class FixedGate(nn.Module):
    def __init__(self,heads:int,init_alpha:float=.995):
        super().__init__()
        target=max(1e-6,-math.log(init_alpha))
        s0=math.log(max(1e-8,math.exp(target)-1.0))
        self.theta0=nn.Parameter(torch.full((heads,),s0))
    def forward(self,B:int):
        a=torch.exp(-F.softplus(self.theta0)).clamp(1e-5,1.0)
        return a[None].expand(B,-1)


class RepairModel(nn.Module):
    def __init__(self,d,heads,features,ff,value_dim,seed):
        super().__init__()
        self.d=d; self.heads=heads
        self.n1=nn.LayerNorm(d)
        self.attn=u.LinearAttentionCore(d,heads,features,seed)
        self.n2=nn.LayerNorm(d)
        self.f1=nn.Linear(d,ff); self.f2=nn.Linear(ff,d)
        self.readout=nn.Linear(d,value_dim)
        self.selective_gate=ContentSelectiveGate(d,heads)
        self.fixed_gate=FixedGate(heads)

    def core_step(self,x,state,memory_changed,alpha):
        a,state,st=self.attn.step(self.n1(x),state,alpha,memory_changed)
        z=x+a
        y=z+self.f2(F.gelu(self.f1(self.n2(z))))
        return y,state,st


class ResetTask:
    """Two-episode associative recall with a control-only reset event.

    Memory tokens contain address+random value.  Query tokens contain address only.
    Episode 2 uses new physical slots but the same semantic addresses and independent
    values.  The reset token is visible to the retention gate but is explicitly
    excluded from H/g writes.
    """
    def __init__(self,tokens:int,d:int,value_dim:int,nkeys:int=4,control_amp:float=1.0):
        if 2*nkeys+2>tokens: raise ValueError('tokens too small')
        if nkeys+value_dim+3>d: raise ValueError('d too small')
        self.N=tokens; self.D=d; self.vd=value_dim; self.K=nkeys
        self.v0=nkeys; self.mem_role=nkeys+value_dim
        self.control_dim=self.mem_role+1; self.query_role=self.mem_role+2
        self.reset_idx=2*nkeys; self.query_idx=2*nkeys+1
        self.control_amp=control_amp; self.keys=torch.eye(nkeys)

    def _write(self,x,slot0,vals):
        K=self.K
        x[:,slot0:slot0+K,:K]=self.keys[None]
        x[:,slot0:slot0+K,self.v0:self.v0+self.vd]=vals
        x[:,slot0:slot0+K,self.mem_role]=1.0

    def _query(self,x,q):
        x[:,self.query_idx,:self.K]=self.keys[q]
        x[:,self.query_idx,self.query_role]=1.0

    def one_episode(self,B,g):
        v=(torch.randint(0,2,(B,self.K,self.vd),generator=g).float()*2-1)
        q0=torch.randint(0,self.K,(B,),generator=g); q1=torch.randint(0,self.K,(B,),generator=g)
        x0=torch.zeros(B,self.N,self.D); self._write(x0,0,v); self._query(x0,q0)
        x1=x0.clone(); x1[:,self.query_idx]=0.0; self._query(x1,q1)
        m0=torch.ones(B,self.N,dtype=torch.bool)
        m1=torch.zeros(B,self.N,dtype=torch.bool); m1[:,self.query_idx]=True
        t0=v[torch.arange(B),q0]; t1=v[torch.arange(B),q1]
        return [x0,x1],[m0,m1],{0:t0,1:t1}

    def two_episode(self,B,g):
        v1=(torch.randint(0,2,(B,self.K,self.vd),generator=g).float()*2-1)
        v2=(torch.randint(0,2,(B,self.K,self.vd),generator=g).float()*2-1)
        qs=[torch.randint(0,self.K,(B,),generator=g) for _ in range(4)]
        x0=torch.zeros(B,self.N,self.D); self._write(x0,0,v1); self._query(x0,qs[0])
        x1=x0.clone(); x1[:,self.query_idx]=0.0; self._query(x1,qs[1])
        x2=x1.clone(); x2[:,self.reset_idx,self.control_dim]=self.control_amp
        # reset is transient: it disappears before episode 2.
        x3=x1.clone(); self._write(x3,self.K,v2); x3[:,self.query_idx]=0.0; self._query(x3,qs[2])
        x4=x3.clone(); x4[:,self.query_idx]=0.0; self._query(x4,qs[3])
        xs=[x0,x1,x2,x3,x4]
        # Gate events: all actual input changes, including reset.
        gate=[]
        gate.append(torch.ones(B,self.N,dtype=torch.bool))
        for a,b in zip(xs[:-1],xs[1:]): gate.append((a-b).abs().amax(-1)>0)
        # Memory events: reset slot is excluded from H/g writes/removals.
        mem=[z.clone() for z in gate]
        for z in mem: z[:,self.reset_idx]=False
        targets={0:v1[torch.arange(B),qs[0]],1:v1[torch.arange(B),qs[1]],
                 3:v2[torch.arange(B),qs[2]],4:v2[torch.arange(B),qs[3]]}
        return xs,mem,gate,targets


def alpha_schedule(kind,model,x,gate_mask,step,oracle_reset=.0,normal_alpha=1.0):
    B=x.shape[0]
    if kind=='identity': return torch.ones(B,model.heads)
    if kind=='oracle':
        a=oracle_reset if step==2 else normal_alpha
        return torch.full((B,model.heads),float(a))
    if kind=='selective': return model.selective_gate(x,gate_mask)
    if kind=='fixed': return model.fixed_gate(B)
    raise ValueError(kind)


def run_one_episode(model,task,B,g,train=True):
    xs,mem,targets=task.one_episode(B,g); state=None; loss=0.0; rows={}
    for i,(x,m) in enumerate(zip(xs,mem)):
        alpha=torch.ones(B,model.heads)
        y,state,_=model.core_step(x,state,m,alpha)
        p=model.readout(y[:,task.query_idx]); t=targets[i]
        li=F.mse_loss(p,t); loss=loss+li; rows[i]=(p,t)
    return loss/2,rows


def run_two_episode(model,task,B,g,kind,oracle_reset=0.0):
    xs,mem,gate,targets=task.two_episode(B,g); state=None; loss=0.0; rows={}; alphas=[]
    for i,(x,mm,gm) in enumerate(zip(xs,mem,gate)):
        a=alpha_schedule(kind,model,x,gm,i,oracle_reset=oracle_reset)
        y,state,_=model.core_step(x,state,mm,a); alphas.append(a)
        if i in targets:
            p=model.readout(y[:,task.query_idx]); t=targets[i]
            loss=loss+F.mse_loss(p,t); rows[i]=(p,t)
    return loss/4,rows,alphas


def set_core_trainable(model,flag:bool):
    for mod in (model.n1,model.attn,model.n2,model.f1,model.f2,model.readout):
        for p in mod.parameters(): p.requires_grad_(flag)


def train_base(model,task,args):
    for p in model.selective_gate.parameters(): p.requires_grad_(False)
    for p in model.fixed_gate.parameters(): p.requires_grad_(False)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.base_lr,betas=(.9,.98),weight_decay=1e-4)
    g=torch.Generator(); g.manual_seed(args.seed+1000); hist=[]
    for st in range(1,args.base_steps+1):
        loss,rows=run_one_episode(model,task,args.batch,g)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if st==1 or st%args.report_every==0 or st==args.base_steps:
            r={'step':st,'loss':float(loss.detach()),'write_sign':sign_acc(*rows[0]),'persistent_query_sign':sign_acc(*rows[1])}
            hist.append(r); print('BASE',json.dumps(r),flush=True)
    return hist


def train_gate(model,task,args,kind,seed):
    set_core_trainable(model,False)
    for p in model.selective_gate.parameters(): p.requires_grad_(kind=='selective')
    for p in model.fixed_gate.parameters(): p.requires_grad_(kind=='fixed')
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(params,lr=args.gate_lr)
    g=torch.Generator(); g.manual_seed(seed); hist=[]
    for st in range(1,args.gate_steps+1):
        loss,rows,alphas=run_two_episode(model,task,args.batch,g,kind)
        # Small regularizer keeps non-reset retention high; reset alpha is task-driven.
        if kind=='selective':
            normal=torch.stack([alphas[i].mean() for i in (0,1,3,4)])
            loss=loss+args.alpha_reg*((1.0-normal)**2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params,5.0); opt.step()
        if st==1 or st%args.report_every==0 or st==args.gate_steps:
            r={'step':st,'loss':float(loss.detach()),
               'pre_sign':float(np.mean([sign_acc(*rows[i]) for i in (0,1)])),
               'post_sign':float(np.mean([sign_acc(*rows[i]) for i in (3,4)])),
               'alpha_reset':float(alphas[2].mean().detach()),
               'alpha_nonreset':float(torch.stack([alphas[i].mean() for i in (0,1,3,4)]).mean().detach())}
            hist.append(r); print(kind.upper(),json.dumps(r),flush=True)
    return hist


@torch.no_grad()
def evaluate(model,task,args,kind,seed,oracle_reset=0.0):
    g=torch.Generator();g.manual_seed(seed);pre=[];post=[];mse=[];ares=[];anon=[]
    for _ in range(args.eval_batches):
        loss,rows,alphas=run_two_episode(model,task,args.eval_batch,g,kind,oracle_reset)
        pre += [sign_acc(*rows[i]) for i in (0,1)]; post += [sign_acc(*rows[i]) for i in (3,4)]
        mse.append(float(loss)); ares.append(float(alphas[2].mean()));
        anon.append(float(torch.stack([alphas[i].mean() for i in (0,1,3,4)]).mean()))
    return {'pre_sign_accuracy':float(np.mean(pre)),'post_sign_accuracy':float(np.mean(post)),
            'mse':float(np.mean(mse)),'alpha_reset':float(np.mean(ares)),'alpha_nonreset':float(np.mean(anon))}


def clone_model(m): return copy.deepcopy(m)


def run(args):
    seed_all(args.seed); torch.set_num_threads(args.threads); out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    task=ResetTask(args.tokens,args.d,args.value_dim,args.nkeys,args.control_amp)
    base=RepairModel(args.d,args.heads,args.features,args.ff,args.value_dim,args.seed+101)
    base_hist=train_base(base,task,args)
    base_eval=evaluate(base,task,args,'identity',args.seed+2500)

    # First prove the two-episode task is solvable if the state is reset at the right time.
    oracle=evaluate(base,task,args,'oracle',args.seed+3000,oracle_reset=0.0)
    identity=evaluate(base,task,args,'identity',args.seed+3000)

    selective=clone_model(base); fixed=clone_model(base)
    sh=train_gate(selective,task,args,'selective',args.seed+4000)
    fh=train_gate(fixed,task,args,'fixed',args.seed+5000)
    se=evaluate(selective,task,args,'selective',args.seed+6000)
    fe=evaluate(fixed,task,args,'fixed',args.seed+6000)
    sid=evaluate(selective,task,args,'identity',args.seed+6000)

    selectivity=se['alpha_nonreset']-se['alpha_reset']
    advantage=se['post_sign_accuracy']-max(fe['post_sign_accuracy'],sid['post_sign_accuracy'])
    gates={
        'base_persistent_memory_pretrained':base_eval['pre_sign_accuracy']>=args.base_gate,
        'oracle_reset_task_solvable':oracle['pre_sign_accuracy']>=args.task_gate and oracle['post_sign_accuracy']>=args.task_gate,
        'identity_reset_control_worse':oracle['post_sign_accuracy']-identity['post_sign_accuracy']>=args.oracle_gap,
        'selective_pre_ge_gate':se['pre_sign_accuracy']>=args.task_gate,
        'selective_post_ge_gate':se['post_sign_accuracy']>=args.task_gate,
        'selective_alpha_gap_ge_0p20':selectivity>=.20,
        'selective_advantage_ge_gate':advantage>=args.selective_gap,
    }
    result={'stage':'UNIFIED_U3BR_SELECTIVE_STATE_REPAIR','config':vars(args),'base_train':base_hist,
            'base_identity_eval':base_eval,'oracle_reset_eval':oracle,'identity_eval':identity,
            'selective_train':sh,'fixed_train':fh,'selective_eval':se,'fixed_eval':fe,
            'selective_identity_ablation':sid,'alpha_selectivity_gap':selectivity,
            'post_reset_selective_advantage':advantage,'gates':gates,'PASS':all(gates.values())}
    if not gates['oracle_reset_task_solvable']:
        result['decision']='invalid temporal task/core: oracle reset itself does not solve it; do not interpret selective retention.'
    elif result['PASS']:
        result['decision']='selective retention has a functional causal role on the repaired temporal-memory control; keep it as a narrow Mamba-inspired primitive.'
    else:
        result['decision']='task is valid under oracle reset, but learned selective retention has not earned canonical status; localize gate learning before proceeding.'
    (out/'U3BR_RESULTS.json').write_text(json.dumps(result,indent=2));
    torch.save({'base':base.state_dict(),'selective':selective.state_dict(),'fixed':fixed.state_dict(),'config':vars(args)},out/'U3BR_MODELS.pt')
    print(json.dumps(result,indent=2),flush=True); return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='unified_snn_transform/run/U3BR')
    ap.add_argument('--seed',type=int,default=31);ap.add_argument('--tokens',type=int,default=16);ap.add_argument('--d',type=int,default=32)
    ap.add_argument('--heads',type=int,default=2);ap.add_argument('--features',type=int,default=32);ap.add_argument('--ff',type=int,default=64)
    ap.add_argument('--value-dim',type=int,default=8);ap.add_argument('--nkeys',type=int,default=4);ap.add_argument('--control-amp',type=float,default=1.0)
    ap.add_argument('--base-steps',type=int,default=500);ap.add_argument('--gate-steps',type=int,default=350);ap.add_argument('--batch',type=int,default=96)
    ap.add_argument('--base-lr',type=float,default=2e-3);ap.add_argument('--gate-lr',type=float,default=2e-2);ap.add_argument('--alpha-reg',type=float,default=.05)
    ap.add_argument('--report-every',type=int,default=100);ap.add_argument('--eval-batches',type=int,default=6);ap.add_argument('--eval-batch',type=int,default=128)
    ap.add_argument('--base-gate',type=float,default=.90);ap.add_argument('--task-gate',type=float,default=.90)
    ap.add_argument('--oracle-gap',type=float,default=.15);ap.add_argument('--selective-gap',type=float,default=.10);ap.add_argument('--threads',type=int,default=4)
    run(ap.parse_args())
if __name__=='__main__':main()
