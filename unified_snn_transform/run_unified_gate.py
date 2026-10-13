#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, json, math, random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def rel_l2(a,b):
    return float((torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(b).clamp_min(1e-12)).detach())


def cosine(a,b):
    aa=a.reshape(-1); bb=b.reshape(-1)
    return float(F.cosine_similarity(aa[None],bb[None]).detach().item())


class PositiveRF(nn.Module):
    def __init__(self, heads:int, head_dim:int, features:int, seed:int=17):
        super().__init__(); self.heads=heads; self.head_dim=head_dim; self.features=features
        g=torch.Generator(device='cpu'); g.manual_seed(seed)
        omega=torch.randn(heads,features,head_dim,generator=g)
        omega=omega/omega.norm(dim=-1,keepdim=True).clamp_min(1e-8)*math.sqrt(head_dim)
        self.register_buffer('omega',omega)

    def stable_a(self,x,is_query:bool):
        xx=x*(self.head_dim**-0.25)
        dash=torch.einsum('bhnd,hmd->bhnm',xx,self.omega)
        diag=.5*(xx*xx).sum(-1,keepdim=True)
        a=dash-diag
        if is_query:
            return (a-a.amax(-1,keepdim=True).detach()).clamp(-30.,0.)
        return a.clamp(-30.,30.)

    def forward(self,x,*,is_query:bool):
        a=self.stable_a(x,is_query)
        return torch.exp(a)*(self.features**-.5)+1e-6


@torch.no_grad()
def izh_voltage(a,bias,gain,window_ms):
    x=a.float(); v=torch.full_like(x,-65.); u=.2*v
    I=float(bias)+float(gain)*x; out=[]
    for _ in range(int(window_ms)):
        dv=.04*v*v+5*v+140-u+I; v=v+.5*dv
        dv=.04*v*v+5*v+140-u+I; v=v+.5*dv
        u=u+.02*(.2*v-u)
        fired=v>=30.
        v=torch.where(fired,torch.full_like(v,-65.),v)
        u=torch.where(fired,u+8.,u)
        out.append(v)
    return torch.stack(out,-1)


@torch.no_grad()
def traj_design(a,bias,gain,window_ms):
    v=izh_voltage(a,bias,gain,window_ms)
    one=torch.ones(*a.shape,1,device=a.device,dtype=v.dtype)
    return torch.cat([one,v],-1)


@torch.no_grad()
def fit_decoder(a,y,bias,gain,window_ms,ridge=1e-5):
    X=traj_design(a,bias,gain,window_ms).double(); yy=y.double(); p=X.shape[-1]
    eye=torch.eye(p,dtype=torch.float64,device=X.device); eye[0,0]=0
    return torch.linalg.solve(X.T@X+ridge*eye,X.T@yy).float()


@torch.no_grad()
def izh_activity(a,bias,gain,window_ms):
    x=a.float(); v=torch.full_like(x,-65.); u=.2*v
    I=float(bias)+float(gain)*x; spikes=torch.zeros_like(x)
    for _ in range(int(window_ms)):
        dv=.04*v*v+5*v+140-u+I; v=v+.5*dv
        dv=.04*v*v+5*v+140-u+I; v=v+.5*dv
        u=u+.02*(.2*v-u); fired=v>=30.; spikes+=fired.float()
        v=torch.where(fired,torch.full_like(v,-65.),v); u=torch.where(fired,u+8.,u)
    return float(spikes.mean()),float((spikes>0).float().mean())


class IzhPhi(nn.Module):
    def __init__(self,exact:PositiveRF,spec:dict,decoder:torch.Tensor):
        super().__init__(); self.head_dim=exact.head_dim; self.features=exact.features
        self.register_buffer('omega',exact.omega.detach().clone())
        self.register_buffer('decoder',decoder.detach().clone())
        self.bias=float(spec['bias']);self.gain=float(spec['gain']);self.window_ms=int(spec['window_ms'])
        self.last_clamp=0.

    def stable_a(self,x,is_query):
        xx=x*(self.head_dim**-0.25)
        dash=torch.einsum('bhnd,hmd->bhnm',xx,self.omega)
        diag=.5*(xx*xx).sum(-1,keepdim=True); a=dash-diag
        return (a-a.amax(-1,keepdim=True).detach()).clamp(-30.,0.) if is_query else a.clamp(-30.,30.)

    def forward(self,x,*,is_query:bool):
        a=self.stable_a(x,is_query)
        X=traj_design(a.reshape(-1),self.bias,self.gain,self.window_ms)
        raw=X@self.decoder.to(X.dtype)
        self.last_clamp=float((raw<=1e-8).float().mean())
        return raw.clamp_min(1e-8).reshape_as(a)


def calibrate_side(a:torch.Tensor,features:int):
    a=a.detach().flatten().cpu(); g=torch.Generator(); g.manual_seed(9917+features+len(a))
    perm=torch.randperm(len(a),generator=g)
    n=max(100,min(int(.7*len(a)),len(a)-100)); fit=a[perm[:n]]; val=a[perm[n:]]
    yf=torch.exp(fit.clamp(-30,30))*(features**-.5)+1e-6
    yv=torch.exp(val.clamp(-30,30))*(features**-.5)+1e-6
    grid=[{'bias':b,'gain':gg,'window_ms':w}
          for b in (2.,4.,6.) for gg in (.35,.5,.75,1.,1.4) for w in (6,8,10)]
    best=None; rows=[]
    for spec in grid:
        dec=fit_decoder(fit,yf,**spec)
        raw=traj_design(val,**spec)@dec; pred=raw.clamp_min(1e-8)
        rr=float(torch.linalg.vector_norm(pred-yv)/torch.linalg.vector_norm(yv).clamp_min(1e-12))
        clamp=float((raw<=1e-8).float().mean())
        spk,active=izh_activity(val[:min(2000,len(val))],**spec)
        score=rr+.15*clamp+.001*spec['window_ms']+.005*spk
        row={**spec,'scalar_rel_l2':rr,'clamp_fraction':clamp,
             'mean_spikes':spk,'active_fraction':active,'score':score}
        rows.append(row)
        if best is None or score<best[0]: best=(score,spec,dec,row)
    return best[1],best[2],best[3],rows


@dataclass
class AttnState:
    H:torch.Tensor
    g:torch.Tensor
    pk:torch.Tensor
    v:torch.Tensor
    weight:torch.Tensor


class LinearAttentionCore(nn.Module):
    def __init__(self,d=32,heads=2,features=32,seed=17):
        super().__init__(); assert d%heads==0
        self.d=d;self.heads=heads;self.dh=d//heads;self.m=features
        self.q=nn.Linear(d,d);self.k=nn.Linear(d,d);self.v=nn.Linear(d,d);self.out=nn.Linear(d,d)
        self.phi_q=PositiveRF(heads,self.dh,features,seed)
        self.phi_k=PositiveRF(heads,self.dh,features,seed)

    def _split(self,x):
        B,N,D=x.shape
        return x.reshape(B,N,self.heads,self.dh).transpose(1,2)

    def q_features(self,x):
        return self.phi_q(self._split(self.q(x)),is_query=True)

    def kv_all(self,x):
        k=self._split(self.k(x)); v=self._split(self.v(x))
        return self.phi_k(k,is_query=False),v

    def step(self,x,state:Optional[AttnState],alpha:torch.Tensor,changed:torch.Tensor):
        B,N,D=x.shape; pq=self.q_features(x)
        a4=alpha[:,:,None,None]; a3=alpha[:,:,None]
        if state is None:
            pk,v=self.kv_all(x)
            H=torch.einsum('bhnm,bhnd->bhmd',pk,v); g=pk.sum(2)
            weight=torch.ones(B,self.heads,N,device=x.device,dtype=x.dtype)
            state=AttnState(H,g,pk,v,weight); kv_evals=B*N
        else:
            H=state.H*a4; g=state.g*a3; weight=state.weight*alpha[:,:,None]
            pk_cache=state.pk.clone(); v_cache=state.v.clone(); kv_evals=0
            for b in range(B):
                idx=torch.nonzero(changed[b],as_tuple=False).flatten()
                if idx.numel()==0: continue
                xb=x[b:b+1,idx]
                kk=self._split(self.k(xb)); vv=self._split(self.v(xb)); pp=self.phi_k(kk,is_query=False)
                oldpk=pk_cache[b:b+1,:,idx,:]; oldv=v_cache[b:b+1,:,idx,:]; ow=weight[b:b+1,:,idx]
                H=H-torch.einsum('bhn,bhnm,bhnd->bhmd',ow,oldpk,oldv)+torch.einsum('bhnm,bhnd->bhmd',pp,vv)
                g=g-(oldpk*ow[:,:,:,None]).sum(2)+pp.sum(2)
                pk_cache[b,:,idx,:]=pp[0]; v_cache[b,:,idx,:]=vv[0]; weight[b,:,idx]=1.
                kv_evals+=int(idx.numel())
            state=AttnState(H,g,pk_cache,v_cache,weight)
        num=torch.einsum('bhnm,bhmd->bhnd',pq,state.H)
        den=torch.einsum('bhnm,bhm->bhn',pq,state.g).unsqueeze(-1).clamp_min(1e-8)
        y=num/den; out=self.out(y.transpose(1,2).reshape(B,N,D))
        return out,state,{'kv_token_evals':kv_evals,'q_token_evals':B*N}


class SelectiveRetention(nn.Module):
    def __init__(self,heads:int,init_alpha=.99):
        super().__init__(); target=max(1e-5,-math.log(init_alpha))
        s0=math.log(max(1e-8,math.exp(target)-1))
        self.register_buffer('theta0',torch.full((heads,),s0))
        self.register_buffer('theta_event',torch.linspace(-.35,.35,heads))

    def forward(self,event_ratio):
        s=self.theta0[None]+event_ratio[:,None]*self.theta_event[None]
        return torch.exp(-F.softplus(s)).clamp(1e-4,1.)


class UnifiedBlock(nn.Module):
    def __init__(self,d=32,heads=2,features=32,ff=64,seed=17):
        super().__init__(); self.d=d;self.heads=heads
        self.n1=nn.LayerNorm(d); self.attn=LinearAttentionCore(d,heads,features,seed)
        self.n2=nn.LayerNorm(d); self.f1=nn.Linear(d,ff);self.f2=nn.Linear(ff,d)
        self.gate=SelectiveRetention(heads)

    def step(self,x,state,changed):
        event_ratio=changed.float().mean(1); alpha=self.gate(event_ratio)
        a,state,stats=self.attn.step(self.n1(x),state,alpha,changed)
        z=x+a; y=z+self.f2(F.gelu(self.f1(self.n2(z))))
        stats.update({'alpha_mean':float(alpha.mean()),'event_ratio':float(event_ratio.mean())})
        return y,state,stats


def collect_calibration(block:UnifiedBlock,seed:int,B=8,N=16,rounds=8):
    g=torch.Generator();g.manual_seed(seed); qs=[];ks=[]
    for _ in range(rounds):
        x=torch.randn(B,N,block.d,generator=g); z=block.n1(x)
        q=block.attn._split(block.attn.q(z)); k=block.attn._split(block.attn.k(z))
        qs.append(block.attn.phi_q.stable_a(q,True).reshape(-1))
        ks.append(block.attn.phi_k.stable_a(k,False).reshape(-1))
    return torch.cat(qs),torch.cat(ks)


def make_candidate(ref:UnifiedBlock,qspec,qdec,kspec,kdec):
    cand=copy.deepcopy(ref)
    cand.attn.phi_q=IzhPhi(ref.attn.phi_q,qspec,qdec)
    cand.attn.phi_k=IzhPhi(ref.attn.phi_k,kspec,kdec)
    return cand


def sequence(seed:int,B:int,N:int,D:int,T:int,change_frac:float):
    g=torch.Generator();g.manual_seed(seed)
    xs=[torch.randn(B,N,D,generator=g)]; masks=[torch.ones(B,N,dtype=torch.bool)]
    nchange=max(1,round(N*change_frac))
    for _ in range(1,T):
        x=xs[-1].clone(); m=torch.zeros(B,N,dtype=torch.bool)
        for b in range(B):
            idx=torch.randperm(N,generator=g)[:nchange]
            x[b,idx]=torch.randn(nchange,D,generator=g); m[b,idx]=True
        xs.append(x);masks.append(m)
    return xs,masks


@torch.no_grad()
def delta_projection_audit(linear:nn.Linear,xs):
    exact=[linear(x) for x in xs]; cur=exact[0].clone(); errs=[]
    changed_features=0;total_features=0
    for t in range(1,len(xs)):
        dx=xs[t]-xs[t-1]
        cur=cur+F.linear(dx,linear.weight,None)
        errs.append(rel_l2(cur,exact[t]))
        changed_features+=int((dx.abs()>1e-12).sum()); total_features+=dx.numel()
    return {'max_rel_l2':max(errs) if errs else 0.,
            'feature_delta_density':changed_features/max(1,total_features)}


def run(args):
    seed_all(args.seed); torch.set_num_threads(args.threads)
    ref=UnifiedBlock(args.d,args.heads,args.features,args.ff,args.seed+101).eval()
    aq,ak=collect_calibration(ref,args.seed+1000,B=args.batch,N=args.tokens,rounds=8)
    qspec,qdec,qrow,_=calibrate_side(aq,args.features)
    kspec,kdec,krow,_=calibrate_side(ak,args.features)
    cand=make_candidate(ref,qspec,qdec,kspec,kdec).eval()

    xs,masks=sequence(args.seed+2000,args.batch,args.tokens,args.d,args.steps,args.change_frac)
    sr=sc=None; rows=[]; counts={'q':0,'kv':0,'dense_kv':0}
    with torch.no_grad():
        for t,(x,m) in enumerate(zip(xs,masks)):
            yr,sr,rs=ref.step(x,sr,m); yc,sc,cs=cand.step(x,sc,m)
            rows.append({'step':t,'output_rel_l2':rel_l2(yc,yr),
                         'output_cosine':cosine(yc,yr),
                         'H_rel_l2':rel_l2(sc.H,sr.H),'g_rel_l2':rel_l2(sc.g,sr.g),
                         'alpha_mean':rs['alpha_mean'],'event_ratio':rs['event_ratio'],
                         'kv_token_evals':cs['kv_token_evals']})
            counts['q']+=cs['q_token_evals'];counts['kv']+=cs['kv_token_evals']
            counts['dense_kv']+=args.batch*args.tokens

    nx=[ref.n1(x) for x in xs]
    delta={name:delta_projection_audit(getattr(ref.attn,name),nx) for name in ('q','k','v')}
    max_out=max(r['output_rel_l2'] for r in rows);max_H=max(r['H_rel_l2'] for r in rows)
    max_g=max(r['g_rel_l2'] for r in rows);min_cos=min(r['output_cosine'] for r in rows)
    gate={'operator_rel_l2_le_1pct':max_out<=.01,
          'state_H_rel_l2_le_2pct':max_H<=.02,
          'state_g_rel_l2_le_2pct':max_g<=.02,
          'output_cosine_ge_0p999':min_cos>=.999,
          'delta_projection_exact_le_1e6':max(v['max_rel_l2'] for v in delta.values())<=1e-6}
    result={'stage':'UNIFIED_LIGHTWEIGHT_A0_A1','config':vars(args),
            'architecture':{
                'baseline':'positive-RF normalized linear attention + selective persistent H/g + FFN',
                'candidate':'same block with Izh trajectory feature map; event-sparse K/V writes',
                'pending_neuralization':['Q/K/V/O channel projections','H/g read arithmetic','FFN','residual/norm hardware mapping']},
            'calibration':{'Q':{**qspec,**qrow},'K':{**kspec,**krow}},
            'trajectory':rows,'delta_projection_audit':delta,
            'compute':{'q_token_evals':counts['q'],
                       'kv_token_evals_event_sparse':counts['kv'],
                       'kv_token_evals_dense_reference':counts['dense_kv'],
                       'kv_token_eval_reduction':1-counts['kv']/counts['dense_kv'],
                       'note':'Projection MAC payload is still dense within each active token; this gate measures token-event sparsity only.'},
            'summary':{'max_output_rel_l2':max_out,'min_output_cosine':min_cos,
                       'max_H_rel_l2':max_H,'max_g_rel_l2':max_g},
            'gates':gate,'PASS':all(gate.values())}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'UNIFIED_RESULTS.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='UNIFIED_RUN');ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--tokens',type=int,default=16);ap.add_argument('--d',type=int,default=32)
    ap.add_argument('--heads',type=int,default=2);ap.add_argument('--features',type=int,default=32)
    ap.add_argument('--ff',type=int,default=64);ap.add_argument('--steps',type=int,default=6)
    ap.add_argument('--batch',type=int,default=4);ap.add_argument('--change-frac',type=float,default=.25)
    ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args();r=run(args);raise SystemExit(0 if r['PASS'] else 2)
