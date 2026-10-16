from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIDE=28; PATCH=4; GRID=7; L=49; K=64; MASK=64; IN_V=65
D=128; LAYERS=4; HEADS=4; HEAD_DIM=32; FF=512; DROP=.05
RF_FEATURES=64; RF_EPS=1e-6

def image_patches(x):
    x=np.asarray(x,np.float32)
    return x.reshape(len(x),GRID,PATCH,GRID,PATCH).transpose(0,1,3,2,4).reshape(-1,16)

def encode(x,c):
    p=image_patches(x); c=np.asarray(c,np.float32); c2=(c*c).sum(1); out=[]
    for i in range(0,len(p),200000):
        q=p[i:i+200000]; d=(q*q).sum(1,keepdims=True)-2*q@c.T+c2[None]
        out.append(d.argmin(1).astype(np.int64))
    return np.concatenate(out).reshape(len(x),L)

def decode(t,c):
    q=np.asarray(c,np.float32)[np.asarray(t,np.int64)].reshape(-1,GRID,GRID,PATCH,PATCH).transpose(0,1,3,2,4).reshape(-1,SIDE,SIDE)
    return np.clip(q,0,1).astype(np.float32)

def random_mask(tokens,g=None):
    B,N=tokens.shape; u=torch.rand(B,device=tokens.device,generator=g); ratio=torch.cos(.5*math.pi*u)
    nmask=torch.round(ratio*N).long().clamp(1,N); score=torch.rand(B,N,device=tokens.device,generator=g)
    rank=score.argsort(1).argsort(1); m=rank<nmask[:,None]; inp=tokens.clone(); inp[m]=MASK
    return inp,m,ratio

class PositiveRF(nn.Module):
    def __init__(self,heads=HEADS,hd=HEAD_DIM,features=RF_FEATURES,seed=100):
        super().__init__(); gen=torch.Generator(device='cpu'); gen.manual_seed(seed)
        w=torch.randn(heads,features,hd,generator=gen); w=w/w.norm(dim=-1,keepdim=True).clamp_min(1e-12)*math.sqrt(hd)
        self.register_buffer('omega',w); self.hd=hd; self.features=features
    def _raw(self,x):
        xx=x*(self.hd**-.25); return torch.einsum('bhnd,hmd->bhnm',xx,self.omega)-.5*(xx*xx).sum(-1,keepdim=True)
    def q(self,x):
        a=self._raw(x); a=a-a.amax(-1,keepdim=True); return torch.exp(a.clamp(-30,0))*self.features**-.5+RF_EPS
    def k(self,x):
        a=self._raw(x); return torch.exp(a.clamp(-30,30))*self.features**-.5+RF_EPS

@dataclass
class Trace:
    x_in:torch.Tensor; q:torch.Tensor; k:torch.Tensor; v:torch.Tensor
    phi_q:torch.Tensor; phi_k:torch.Tensor; H_state:torch.Tensor; g_state:torch.Tensor
    attention_read:torch.Tensor; post_attention:torch.Tensor; ff_hidden:torch.Tensor; x_out:torch.Tensor

class HgAttention(nn.Module):
    def __init__(self,d=D,heads=HEADS,features=RF_FEATURES,seed=100):
        super().__init__(); self.h=heads; self.hd=d//heads
        self.WQ=nn.Linear(d,d); self.WK=nn.Linear(d,d); self.WV=nn.Linear(d,d); self.WO=nn.Linear(d,d)
        self.rf=PositiveRF(heads,self.hd,features,seed)
    def forward(self,x,internal=False):
        B,N,D0=x.shape
        def sp(y): return y.reshape(B,N,self.h,self.hd).transpose(1,2)
        q,k,v=sp(self.WQ(x)),sp(self.WK(x)),sp(self.WV(x)); pq,pk=self.rf.q(q),self.rf.k(k)
        g=pk.sum(2); H=torch.einsum('bhnm,bhnd->bhmd',pk,v)
        num=torch.einsum('bhnm,bhmd->bhnd',pq,H); den=torch.einsum('bhnm,bhm->bhn',pq,g).clamp_min(1e-8)
        a=(num/den[...,None]).transpose(1,2).reshape(B,N,D0); out=self.WO(a)
        return (out,dict(q=q,k=k,v=v,phi_q=pq,phi_k=pk,H_state=H,g_state=g,attention_read=a)) if internal else out

class Block(nn.Module):
    def __init__(self,d=D,heads=HEADS,ff=FF,drop=DROP,seed=100):
        super().__init__(); self.attn=HgAttention(d,heads,2*(d//heads),seed); self.n1=nn.LayerNorm(d); self.n2=nn.LayerNorm(d)
        self.f1=nn.Linear(d,ff); self.f2=nn.Linear(ff,d); self.drop=drop
    def forward(self,x,trace=False):
        ao,z=self.attn(x,True); xa=self.n1(x+F.dropout(ao,self.drop,self.training)); h=F.gelu(self.f1(xa)); y=self.n2(xa+F.dropout(self.f2(h),self.drop,self.training))
        if not trace:return y
        return y,Trace(x,z['q'],z['k'],z['v'],z['phi_q'],z['phi_k'],z['H_state'],z['g_state'],z['attention_read'],xa,h,y)

class Model(nn.Module):
    def __init__(self,seed=26001):
        super().__init__(); self.tok=nn.Embedding(IN_V,D); self.pos=nn.Parameter(torch.empty(1,L,D))
        self.blocks=nn.ModuleList([Block(seed=seed+1009*i) for i in range(LAYERS)])
        self.dense=nn.Linear(D,D); self.ln=nn.LayerNorm(D); self.bias=nn.Parameter(torch.zeros(K)); self.reset(seed)
    def reset(self,seed):
        torch.manual_seed(seed); nn.init.trunc_normal_(self.tok.weight,std=.02,a=-.04,b=.04); nn.init.trunc_normal_(self.pos,std=.02,a=-.04,b=.04)
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.trunc_normal_(m.weight,std=.02,a=-.04,b=.04); nn.init.zeros_(m.bias)
    def forward(self,tokens,traces=False):
        z=self.tok(tokens)+self.pos; tt=[]
        for b in self.blocks:
            if traces:z,t=b(z,True);tt.append(t)
            else:z=b(z)
        z=self.ln(F.gelu(self.dense(z))); logits=F.linear(z,self.tok.weight[:K],self.bias)
        return (logits,tt) if traces else logits

@torch.no_grad()
def generate(model,batch_size,device,steps=16,temp=0.,seed=1007):
    model.eval(); cur=torch.full((batch_size,L),MASK,dtype=torch.long,device=device); initial=torch.full((batch_size,),L,dtype=torch.long,device=device)
    g=torch.Generator(device=device);g.manual_seed(seed)
    for st in range(steps):
        unk=cur.eq(MASK); p=model(cur).softmax(-1); s=torch.multinomial(p.reshape(-1,K),1,generator=g).reshape(batch_size,L)
        prop=torch.where(unk,s,cur); conf=p.gather(-1,prop[...,None]).squeeze(-1)
        if st==steps-1:cur=prop;continue
        base=torch.where(unk,conf,torch.full_like(conf,float('inf'))); frac=(st+1)/steps
        desired=torch.floor(initial.float()*math.cos(.5*math.pi*frac)).long(); desired=torch.minimum(desired,torch.clamp(unk.sum(1)-1,min=0))
        u=torch.rand(base.shape,device=device,generator=g).clamp_(1e-6,1-1e-6); gum=-torch.log(-torch.log(u)); score=torch.log(base.clamp_min(1e-12))+temp*(1-frac)*gum
        score=torch.where(torch.isinf(base),torch.full_like(score,float('inf')),score); new=prop.clone()
        for bi in range(batch_size):
            k=int(desired[bi]);
            if k:new[bi,torch.topk(score[bi],k,largest=False).indices]=MASK
        cur=new
    return cur
