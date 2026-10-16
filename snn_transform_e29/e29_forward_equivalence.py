#!/usr/bin/env python3
"""P4-E29: zero-training B0 -> resident H/g forward-equivalence audit.

Scientific rule: the trained B0 checkpoint is the reference solution. E29 does not
train the H/g model. It copies every learned B0 forward parameter exactly and
changes only the attention operator.

For causal cleanliness E29 RETAINS B0 class conditioning. Removing class
conditioning belongs to a later experiment; doing it here would confound the
attention replacement test.
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from maskgit_b0_mnist012 import (
    Model as B0Model, Block as B0Block,
    load, subset012, split, encode, decode,
    fixed_mask_batch, generate as b0_generate,
    L, K, MASK, IN_V, NCLASS,
)

D=128; LAYERS=4; HEADS=4; HEAD_DIM=32; FF=512

def seed_all(s:int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

@dataclass
class ExactTrace:
    x_in: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    attention_read: torch.Tensor
    post_attention: torch.Tensor
    x_out: torch.Tensor

@torch.no_grad()
def exact_block_trace(block:B0Block, x:torch.Tensor) -> Tuple[torch.Tensor,ExactTrace]:
    B,N,D0=x.shape
    qkv=block.qkv(x).reshape(B,N,3,block.h,block.dh).permute(2,0,3,1,4)
    q,k,v=qkv.unbind(0)
    a=F.scaled_dot_product_attention(q,k,v,dropout_p=0.0)
    read=a.transpose(1,2).reshape(B,N,D0)
    xa=block.n1(x+block.proj(read))
    y=block.n2(xa+block.f2(F.gelu(block.f1(xa))))
    return y,ExactTrace(x,q,k,v,read,xa,y)

class PositiveRF(nn.Module):
    """Eq.14-style positive RF for exp(q^T k / sqrt(d_h)).

    gaussian: omega_ij ~ N(0,1), faithful to the paper's random projection statement.
    p3norm: previous E26/P3 row-normalized RF control.
    """
    def __init__(self, heads:int, hd:int, features:int, seed:int, kind:str='gaussian'):
        super().__init__(); self.hd=hd; self.features=features; self.kind=kind
        g=torch.Generator(device='cpu'); g.manual_seed(seed)
        omega=torch.randn(heads,features,hd,generator=g)
        if kind=='p3norm':
            omega=omega/omega.norm(dim=-1,keepdim=True).clamp_min(1e-12)*math.sqrt(hd)
        elif kind!='gaussian':
            raise ValueError(f'unknown rf kind {kind}')
        self.register_buffer('omega',omega)
    def _raw(self,x):
        xx=x*(self.hd**-0.25)
        return torch.einsum('bhnd,hmd->bhnm',xx,self.omega)-0.5*(xx*xx).sum(-1,keepdim=True)
    def q(self,x):
        a=self._raw(x); a=a-a.amax(-1,keepdim=True)
        return torch.exp(a.clamp(-30.,0.))/math.sqrt(self.features)+1e-8
    def k(self,x):
        a=self._raw(x)
        return torch.exp(a.clamp(-30.,30.))/math.sqrt(self.features)+1e-8

@dataclass
class HgTrace:
    x_in: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    H: torch.Tensor
    g: torch.Tensor
    attention_read: torch.Tensor
    post_attention: torch.Tensor
    x_out: torch.Tensor

class HgAttention(nn.Module):
    def __init__(self,d=D,heads=HEADS,features=256,seed=100,rf_kind='gaussian'):
        super().__init__(); self.h=heads; self.hd=d//heads
        self.WQ=nn.Linear(d,d); self.WK=nn.Linear(d,d); self.WV=nn.Linear(d,d); self.WO=nn.Linear(d,d)
        self.rf=PositiveRF(heads,self.hd,features,seed,rf_kind)
    def forward(self,x,trace=False):
        B,N,D0=x.shape
        def sp(z):return z.reshape(B,N,self.h,self.hd).transpose(1,2)
        q,k,v=sp(self.WQ(x)),sp(self.WK(x)),sp(self.WV(x))
        pq,pk=self.rf.q(q),self.rf.k(k)
        g=pk.sum(2)
        H=torch.einsum('bhnm,bhnd->bhmd',pk,v)
        num=torch.einsum('bhnm,bhmd->bhnd',pq,H)
        den=torch.einsum('bhnm,bhm->bhn',pq,g).clamp_min(1e-12)
        read=(num/den[...,None]).transpose(1,2).reshape(B,N,D0)
        out=self.WO(read)
        return (out,(q,k,v,H,g,read)) if trace else out

class HgBlock(nn.Module):
    def __init__(self,d=D,heads=HEADS,ff=FF,features=256,seed=100,rf_kind='gaussian'):
        super().__init__(); self.attn=HgAttention(d,heads,features,seed,rf_kind)
        self.n1=nn.LayerNorm(d);self.n2=nn.LayerNorm(d);self.f1=nn.Linear(d,ff);self.f2=nn.Linear(ff,d)
    def forward(self,x,trace=False):
        ao,inn=self.attn(x,True)
        xa=self.n1(x+ao); y=self.n2(xa+self.f2(F.gelu(self.f1(xa))))
        if trace:
            q,k,v,H,g,read=inn
            return y,HgTrace(x,q,k,v,H,g,read,xa,y)
        return y

class HgB0Model(nn.Module):
    """Diagnostic model: identical to B0 including class embedding; only attention differs."""
    def __init__(self,features=256,rf_kind='gaussian',seed=29001):
        super().__init__();self.tok=nn.Embedding(IN_V,D);self.pos=nn.Parameter(torch.empty(1,L,D));self.cls=nn.Embedding(NCLASS,D)
        self.blocks=nn.ModuleList([HgBlock(features=features,seed=seed+1009*i,rf_kind=rf_kind) for i in range(LAYERS)])
        self.dense=nn.Linear(D,D);self.ln=nn.LayerNorm(D);self.bias=nn.Parameter(torch.zeros(K))
    def forward(self,x,y,traces=False):
        z=self.tok(x)+self.pos+self.cls(y)[:,None]; tt=[]
        for b in self.blocks:
            if traces:z,t=b(z,True);tt.append(t)
            else:z=b(z)
        z=self.ln(F.gelu(self.dense(z)));log=F.linear(z,self.tok.weight[:K],self.bias)
        return (log,tt) if traces else log

@torch.no_grad()
def transfer_b0(src:B0Model,dst:HgB0Model):
    dst.tok.weight.copy_(src.tok.weight);dst.pos.copy_(src.pos);dst.cls.weight.copy_(src.cls.weight)
    for s,d in zip(src.blocks,dst.blocks):
        d.attn.WQ.weight.copy_(s.qkv.weight[:D]); d.attn.WQ.bias.copy_(s.qkv.bias[:D])
        d.attn.WK.weight.copy_(s.qkv.weight[D:2*D]); d.attn.WK.bias.copy_(s.qkv.bias[D:2*D])
        d.attn.WV.weight.copy_(s.qkv.weight[2*D:]); d.attn.WV.bias.copy_(s.qkv.bias[2*D:])
        d.attn.WO.weight.copy_(s.proj.weight); d.attn.WO.bias.copy_(s.proj.bias)
        d.n1.load_state_dict(s.n1.state_dict());d.n2.load_state_dict(s.n2.state_dict());d.f1.load_state_dict(s.f1.state_dict());d.f2.load_state_dict(s.f2.state_dict())
    dst.dense.load_state_dict(src.dense.state_dict());dst.ln.load_state_dict(src.ln.state_dict());dst.bias.copy_(src.bias)

def rel_l2(a,b):
    return float((a-b).norm().div(a.norm().clamp_min(1e-12)).item())

@torch.no_grad()
def exact_model_traces(m:B0Model,x,y):
    z=m.tok(x)+m.pos+m.cls(y)[:,None]; ts=[]
    for b in m.blocks:
        z,t=exact_block_trace(b,z);ts.append(t)
    out=m.ln(F.gelu(m.dense(z)));log=F.linear(out,m.tok.weight[:K],m.bias)
    return log,ts

@torch.no_grad()
def paired_metrics(exact:B0Model,hg:HgB0Model,tok,y,dev,ratio=.5,batch=128,limit=512):
    exact.eval();hg.eval();T=torch.from_numpy(tok[:limit]).long();Y=torch.from_numpy(y[:limit]).long()
    agg={'n':0,'ce_exact':0.,'ce_hg':0.,'kl':0.,'top1_agree':0.,'logit_rel':0.}
    cum=[0.]*LAYERS; local_block=[0.]*LAYERS; local_attn=[0.]*LAYERS; nb=0
    for i in range(0,len(T),batch):
        t=T[i:i+batch].to(dev);yy=Y[i:i+batch].to(dev);inp,ma=fixed_mask_batch(t,ratio,seed=12345+i+int(ratio*1000))
        le,te=exact_model_traces(exact,inp,yy); lh,th=hg(inp,yy,True)
        n=int(ma.sum()); agg['n']+=n
        agg['ce_exact']+=float(F.cross_entropy(le[ma],t[ma],reduction='sum'))
        agg['ce_hg']+=float(F.cross_entropy(lh[ma],t[ma],reduction='sum'))
        pe=le[ma].log_softmax(-1); ph=lh[ma].log_softmax(-1)
        agg['kl']+=float((pe.exp()*(pe-ph)).sum(-1).sum())
        agg['top1_agree']+=float((le[ma].argmax(-1)==lh[ma].argmax(-1)).sum())
        agg['logit_rel']+=rel_l2(le[ma],lh[ma]);nb+=1
        for l in range(LAYERS):
            cum[l]+=rel_l2(te[l].x_out,th[l].x_out)
            xe=te[l].x_in
            ye,tel=exact_block_trace(exact.blocks[l],xe)
            yh,thl=hg.blocks[l](xe,True)
            local_block[l]+=rel_l2(ye,yh);local_attn[l]+=rel_l2(tel.attention_read,thl.attention_read)
    n=max(1,agg['n'])
    return {
        'ratio':ratio,'masked_tokens':agg['n'],
        'ce_exact':agg['ce_exact']/n,'ce_hg':agg['ce_hg']/n,'ce_delta':agg['ce_hg']/n-agg['ce_exact']/n,
        'kl_exact_to_hg':agg['kl']/n,'top1_agreement':agg['top1_agree']/n,
        'mean_batch_logit_rel_l2':agg['logit_rel']/max(1,nb),
        'cumulative_block_rel_l2':[v/max(1,nb) for v in cum],
        'local_block_rel_l2':[v/max(1,nb) for v in local_block],
        'local_attention_read_rel_l2':[v/max(1,nb) for v in local_attn],
    }

@torch.no_grad()
def hg_generate(m:HgB0Model,labels,dev,steps=16,seed=1007,greedy=False):
    m.eval();labels=labels.to(dev);B=len(labels);cur=torch.full((B,L),MASK,dtype=torch.long,device=dev);initial=torch.full((B,),L,dtype=torch.long,device=dev)
    g=torch.Generator(device=dev);g.manual_seed(seed)
    for st in range(steps):
        unk=cur.eq(MASK);p=m(cur,labels).softmax(-1)
        s=p.argmax(-1) if greedy else torch.multinomial(p.reshape(-1,K),1,generator=g).reshape(B,L)
        prop=torch.where(unk,s,cur);conf=p.gather(-1,prop[...,None]).squeeze(-1)
        if st==steps-1:cur=prop;continue
        base=torch.where(unk,conf,torch.full_like(conf,float('inf')));frac=(st+1)/steps
        desired=torch.floor(initial.float()*math.cos(.5*math.pi*frac)).long();desired=torch.minimum(desired,torch.clamp(unk.sum(1)-1,min=0))
        new=prop.clone()
        for bi in range(B):
            k=int(desired[bi]);
            if k:new[bi,torch.topk(base[bi],k,largest=False).indices]=MASK
        cur=new
    return cur.cpu().numpy().astype(np.int16)

@torch.no_grad()
def b0_generate_greedy(m:B0Model,labels,dev,steps=16):
    m.eval();labels=labels.to(dev);B=len(labels);cur=torch.full((B,L),MASK,dtype=torch.long,device=dev);initial=torch.full((B,),L,dtype=torch.long,device=dev)
    for st in range(steps):
        unk=cur.eq(MASK);p=m(cur,labels).softmax(-1);s=p.argmax(-1);prop=torch.where(unk,s,cur);conf=p.gather(-1,prop[...,None]).squeeze(-1)
        if st==steps-1:cur=prop;continue
        base=torch.where(unk,conf,torch.full_like(conf,float('inf')));frac=(st+1)/steps
        desired=torch.floor(initial.float()*math.cos(.5*math.pi*frac)).long();desired=torch.minimum(desired,torch.clamp(unk.sum(1)-1,min=0))
        new=prop.clone()
        for bi in range(B):
            k=int(desired[bi]);
            if k:new[bi,torch.topk(base[bi],k,largest=False).indices]=MASK
        cur=new
    return cur.cpu().numpy().astype(np.int16)

def save_grid(img,lab,path,title):
    n=len(img);cols=15;rows=math.ceil(n/cols);fig,ax=plt.subplots(rows,cols,figsize=(cols,rows*1.15));ax=np.asarray(ax).reshape(-1)
    for i,a in enumerate(ax):
        a.axis('off')
        if i<n:a.imshow(img[i],cmap='gray',vmin=0,vmax=1,interpolation='nearest');a.set_title(str(int(lab[i])),fontsize=7)
    fig.suptitle(title);plt.tight_layout();fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)

def classifier(xtr,ytr):
    return LogisticRegression(max_iter=250,solver='liblinear',random_state=7).fit(xtr.reshape(len(xtr),-1),ytr)

def gen_metrics(clf,img,labels):
    pred=clf.predict(img.reshape(len(img),-1));prob=clf.predict_proba(img.reshape(len(img),-1))
    return {'condition_match':float((pred==labels).mean()),'mean_conf':float(prob.max(1).mean()),'pred_counts':np.bincount(pred,minlength=NCLASS).tolist()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--codebook',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--features',type=str,default='64,128,256,512');ap.add_argument('--rf-kinds',type=str,default='gaussian,p3norm');ap.add_argument('--seed',type=int,default=29001);ap.add_argument('--threads',type=int,default=2);ap.add_argument('--audit-limit',type=int,default=512);ap.add_argument('--gen-per-class',type=int,default=20)
    a=ap.parse_args();seed_all(a.seed);torch.set_num_threads(a.threads);dev=torch.device('cpu');a.out.mkdir(parents=True,exist_ok=True)
    xall,yall,xte0,yte0=load(a.data);x012,y012=subset012(xall,yall);xte,yte=subset012(xte0,yte0);xtr,ytr,xva,yva=split(x012,y012,.1,7)
    cent=np.load(a.codebook);te=encode(xte,cent)
    exact=B0Model(128,4,4,512,.05).to(dev);ck=torch.load(a.checkpoint,map_location=dev,weights_only=False);exact.load_state_dict(ck['model']);exact.eval()
    clf=classifier(xtr,ytr)
    labels_np=np.tile(np.arange(NCLASS,dtype=np.int64),a.gen_per_class);labels=torch.from_numpy(labels_np)
    exact_s,_=b0_generate(exact,labels,dev,16,0.0,a.seed+1000,False);exact_g=b0_generate_greedy(exact,labels,dev,16)
    img_es=decode(exact_s,cent);img_eg=decode(exact_g,cent);save_grid(img_es,labels_np,a.out/'exact_stochastic.png','B0 exact stochastic');save_grid(img_eg,labels_np,a.out/'exact_greedy.png','B0 exact greedy')
    results={'stage':'P4-E29 zero-training B0->H/g forward equivalence','checkpoint_epoch':int(ck.get('epoch',-1)),'checkpoint_best_val_ce':float(ck.get('best',float('nan'))),'diagnostic_keeps_class_condition':True,'hg_training_performed':False,'exact_generation_stochastic':gen_metrics(clf,img_es,labels_np),'exact_generation_greedy':gen_metrics(clf,img_eg,labels_np),'variants':[]}
    for kind in [s.strip() for s in a.rf_kinds.split(',') if s.strip()]:
      for feat in [int(s) for s in a.features.split(',') if s.strip()]:
        hg=HgB0Model(feat,kind,a.seed+feat).to(dev);transfer_b0(exact,hg);hg.eval()
        pm=sum(p.numel() for p in hg.parameters()); pe=sum(p.numel() for p in exact.parameters())
        ratios=[paired_metrics(exact,hg,te,yte,dev,r,128,a.audit_limit) for r in (.25,.5,.9,1.0)]
        hs=hg_generate(hg,labels,dev,16,a.seed+1000,False);hgre=hg_generate(hg,labels,dev,16,a.seed+1000,True)
        ims=decode(hs,cent);img=decode(hgre,cent)
        tag=f'{kind}_m{feat}';save_grid(ims,labels_np,a.out/f'hg_{tag}_stochastic.png',f'H/g {tag} stochastic');save_grid(img,labels_np,a.out/f'hg_{tag}_greedy.png',f'H/g {tag} greedy')
        var={'rf_kind':kind,'features':feat,'trainable_params_exact':pe,'trainable_params_hg':pm,'parameter_count_parity':bool(pe==pm),'ratio_metrics':ratios,'generation_stochastic':gen_metrics(clf,ims,labels_np),'generation_greedy':gen_metrics(clf,img,labels_np),'greedy_token_agreement_with_exact':float((hgre==exact_g).mean()),'stochastic_token_agreement_with_exact':float((hs==exact_s).mean())}
        results['variants'].append(var);print(json.dumps(var),flush=True)
    (a.out/'metrics.json').write_text(json.dumps(results,indent=2));print('FINAL '+json.dumps(results),flush=True)

if __name__=='__main__':main()
