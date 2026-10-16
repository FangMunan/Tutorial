from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from model import D,K,L,LAYERS,HEADS,HEAD_DIM,FF,Model,random_mask
AF=.62;AS=.92;WF=.70;WS=.30;MF_A=.62;MS_A=.92;MWF=.70;MWS=.30

def masked_mean(x,m):
    w=m.to(x.dtype); return (x*w[...,None]).sum(1)/w.sum(1,keepdim=True).clamp_min(1.)

@dataclass
class DS: ef:torch.Tensor; es:torch.Tensor; mf:torch.Tensor; ms:torch.Tensor
class Bank:
    def __init__(self,dev):self.dev=dev;self.s:Dict[str,DS]={}
    def outer(self,name,pre,mod):
        if len(pre)==0:return torch.zeros((mod.shape[1],pre.shape[1]),device=pre.device),torch.zeros(mod.shape[1],device=pre.device)
        st=self.s.get(name)
        if st is None:
            st=DS(*(torch.zeros(n,device=self.dev,dtype=pre.dtype) for n in (pre.shape[1],pre.shape[1],mod.shape[1],mod.shape[1])));self.s[name]=st
        er=[];mr=[];ef,es,mf,ms=st.ef,st.es,st.mf,st.ms
        for i in range(len(pre)):
            ef=AF*ef+(1-AF)*pre[i];es=AS*es+(1-AS)*pre[i];mf=MF_A*mf+(1-MF_A)*mod[i];ms=MS_A*ms+(1-MS_A)*mod[i]
            er.append(WF*ef+WS*es);mr.append(MWF*mf+MWS*ms)
        st.ef,st.es,st.mf,st.ms=ef.detach(),es.detach(),mf.detach(),ms.detach();E=torch.stack(er);M=torch.stack(mr)
        return M.T@E/len(E),M.mean(0)
    def cpu(self):return {k:{q:getattr(v,q).cpu() for q in ('ef','es','mf','ms')} for k,v in self.s.items()}
    def load(self,o):self.s={k:DS(**{q:v[q].to(self.dev) for q in ('ef','es','mf','ms')}) for k,v in o.items()}

class Reciprocal:
    def __init__(self,m,dev,seed):
        self.m=m;self.dev=dev;self.g=torch.Generator(device=dev);self.g.manual_seed(seed);self.b=Bank(dev);self.sig=.08;self.eta=2e-3;self.leak=2e-5
        self.BR=torch.randn(D,K,generator=self.g,device=dev)*.01;self.Bd=torch.randn(D,D,generator=self.g,device=dev)*.01
        self.BO=[torch.randn(D,D,generator=self.g,device=dev)*.01 for _ in range(LAYERS)];self.B2=[torch.randn(FF,D,generator=self.g,device=dev)*.01 for _ in range(LAYERS)]
    @torch.no_grad()
    def probe(self,n=2):
        for _ in range(n):
            x=torch.randn(D,generator=self.g,device=self.dev)*self.sig;y=self.m.tok.weight[:K]@x;dw,_=self.b.outer('BR',y[None],x[None]);self.BR.mul_(1-self.leak).add_(self.eta*dw)
            x=torch.randn(D,generator=self.g,device=self.dev)*self.sig;y=self.m.dense.weight@x;dw,_=self.b.outer('Bd',y[None],x[None]);self.Bd.mul_(1-self.leak).add_(self.eta*dw)
            for i,bl in enumerate(self.m.blocks):
                x=torch.randn(D,generator=self.g,device=self.dev)*self.sig;y=bl.attn.WO.weight@x;dw,_=self.b.outer(f'BO{i}',y[None],x[None]);self.BO[i].mul_(1-self.leak).add_(self.eta*dw)
                x=torch.randn(FF,generator=self.g,device=self.dev)*self.sig;y=bl.f2.weight@x;dw,_=self.b.outer(f'B2{i}',y[None],x[None]);self.B2[i].mul_(1-self.leak).add_(self.eta*dw)
    def cpu(self):return dict(BR=self.BR.cpu(),Bd=self.Bd.cpu(),BO=[x.cpu() for x in self.BO],B2=[x.cpu() for x in self.B2],bank=self.b.cpu())
    def load(self,o):self.BR=o['BR'].to(self.dev);self.Bd=o['Bd'].to(self.dev);self.BO=[x.to(self.dev) for x in o['BO']];self.B2=[x.to(self.dev) for x in o['B2']];self.b.load(o['bank'])

class Trainer:
    def __init__(self,m:Model,dev,seed=27101,scale=1.,R=4):
        self.m=m;self.dev=dev;self.g=torch.Generator(device=dev);self.g.manual_seed(seed+11);self.bank=Bank(dev);self.r=Reciprocal(m,dev,seed+22);self.R=R;self.step=0
        self.lr={k:v*scale for k,v in dict(read=.020,dense=.004,W2=.0015,W1=.001,WO=.0015,WV=.0008,WQ=.00025,WK=.00025,pos=.00015).items()};self.sigq=.08;self.sigk=.08;self.clip=.10
    @torch.no_grad()
    def add(self,p,d,lr):
        n=d.norm().item();d=d*(self.clip/(n+1e-12)) if n>self.clip else d;p.add_(lr*d)
    @torch.no_grad()
    def qdrive(self,bl,tr,da,m):
        B=len(m);qb=masked_mean(tr.q.transpose(1,2).reshape(B,L,D),m).reshape(B,HEADS,HEAD_DIM);db=masked_mean(da,m).reshape(B,HEADS,HEAD_DIM)
        xi=torch.where(torch.rand((B,HEADS,HEAD_DIM),generator=self.g,device=self.dev)<.5,-1.,1.);qp=bl.attn.rf.q((qb+self.sigq*xi)[:,:,None,:]).squeeze(2);qm=bl.attn.rf.q((qb-self.sigq*xi)[:,:,None,:]).squeeze(2)
        dp=torch.einsum('bhm,bhm->bh',qp,tr.g_state).clamp_min(1e-8);dm=torch.einsum('bhm,bhm->bh',qm,tr.g_state).clamp_min(1e-8)
        ap=torch.einsum('bhm,bhmd->bhd',qp,tr.H_state)/dp[...,None];am=torch.einsum('bhm,bhmd->bhd',qm,tr.H_state)/dm[...,None]
        return (((db*(ap-am)).sum(-1)/(2*self.sigq))[...,None]*xi).reshape(B,D)
    @torch.no_grad()
    def k_events(self,bl,tr,da,m):
        B=len(m);pq=masked_mean(tr.phi_q.transpose(1,2).reshape(B,L,HEADS*tr.phi_q.shape[-1]),m).reshape(B,HEADS,-1);db=masked_mean(da,m).reshape(B,HEADS,HEAD_DIM);grp=torch.arange(L,device=self.dev)%self.R;pp=[];mm=[]
        den0=torch.einsum('bhm,bhm->bh',pq,tr.g_state).clamp_min(1e-8);num0=torch.einsum('bhm,bhmd->bhd',pq,tr.H_state)
        for r in range(self.R):
            idx=torch.nonzero(grp==r).flatten();gc=max(1,int(idx.numel()));norm=math.sqrt(gc);xg=tr.x_in[:,idx,:].sum(1)/norm;xi=torch.where(torch.rand((B,HEADS,HEAD_DIM),generator=self.g,device=self.dev)<.5,-1.,1.)
            k0=tr.k[:,:,idx,:];v=tr.v[:,:,idx,:];pk0=tr.phi_k[:,:,idx,:];pkp=bl.attn.rf.k(k0+self.sigk*xi[:,:,None,:]);pkm=bl.attn.rf.k(k0-self.sigk*xi[:,:,None,:])
            dgp=(pkp-pk0).sum(2);dgm=(pkm-pk0).sum(2);dHp=torch.einsum('bhgm,bhgd->bhmd',pkp-pk0,v);dHm=torch.einsum('bhgm,bhgd->bhmd',pkm-pk0,v)
            dp=(den0+torch.einsum('bhm,bhm->bh',pq,dgp)).clamp_min(1e-8);dm=(den0+torch.einsum('bhm,bhm->bh',pq,dgm)).clamp_min(1e-8)
            ap=(num0+torch.einsum('bhm,bhmd->bhd',pq,dHp))/dp[...,None];am=(num0+torch.einsum('bhm,bhmd->bhd',pq,dHm))/dm[...,None]
            kd=(((db*(ap-am)).sum(-1)/(2*self.sigk*norm))[...,None]*xi).reshape(B,D);pp.append(xg);mm.append(kd)
        return torch.cat(pp),torch.cat(mm)
    @torch.no_grad()
    def vpre(self,tr,m):
        B=len(m);X=torch.einsum('bhnm,bnc->bhmc',tr.phi_k,tr.x_in);pq=masked_mean(tr.phi_q.transpose(1,2).reshape(B,L,HEADS*tr.phi_q.shape[-1]),m).reshape(B,HEADS,-1);den=torch.einsum('bhm,bhm->bh',pq,tr.g_state).clamp_min(1e-8)
        return torch.einsum('bhm,bhmc->bhc',pq,X)/den[...,None]
    @torch.no_grad()
    def batch(self,tok,qk=True):
        self.m.train();inp,ma,ratio=random_mask(tok,self.g);log,tr=self.m(inp,True);p=log.softmax(-1);err=torch.zeros_like(p);err.scatter_(-1,tok[...,None],1.);err.sub_(p);err.mul_(ma[...,None]);ce=F.cross_entropy(log[ma],tok[ma]).item();acc=(log[ma].argmax(-1)==tok[ma]).float().mean().item();self.r.probe(2)
        hz=self.m.ln(F.gelu(self.m.dense(tr[-1].x_out)));dw,db=self.bank.outer('read',masked_mean(hz,ma),masked_mean(err,ma));self.add(self.m.tok.weight[:K],dw,self.lr['read']);self.m.bias.add_(self.lr['read']*db.clamp(-.05,.05))
        dh=torch.einsum('dk,bnk->bnd',self.r.BR,err);dw,db=self.bank.outer('dense',masked_mean(tr[-1].x_out,ma),masked_mean(dh,ma));self.add(self.m.dense.weight,dw,self.lr['dense']);self.m.dense.bias.add_(self.lr['dense']*db.clamp(-.05,.05));dy=torch.einsum('de,bne->bnd',self.r.Bd,dh)
        for i in reversed(range(LAYERS)):
            bl= self.m.blocks[i];t=tr[i];mod=masked_mean(dy,ma);dw,db=self.bank.outer(f'{i}W2',masked_mean(t.ff_hidden,ma),mod);self.add(bl.f2.weight,dw,self.lr['W2']);bl.f2.bias.add_(self.lr['W2']*db.clamp(-.05,.05))
            dhid=torch.einsum('fd,bnd->bnf',self.r.B2[i],dy);dw,db=self.bank.outer(f'{i}W1',masked_mean(t.post_attention,ma),masked_mean(dhid,ma));self.add(bl.f1.weight,dw,self.lr['W1']);bl.f1.bias.add_(self.lr['W1']*db.clamp(-.05,.05))
            dw,db=self.bank.outer(f'{i}WO',masked_mean(t.attention_read,ma),mod);self.add(bl.attn.WO.weight,dw,self.lr['WO']);bl.attn.WO.bias.add_(self.lr['WO']*db.clamp(-.05,.05));da=torch.einsum('de,bne->bnd',self.r.BO[i],dy)
            vp=self.vpre(t,ma);vm=masked_mean(da,ma).reshape(len(tok),HEADS,HEAD_DIM);dWV=torch.zeros_like(bl.attn.WV.weight)
            for h in range(HEADS):dWV[h*HEAD_DIM:(h+1)*HEAD_DIM]=self.bank.outer(f'{i}WV{h}',vp[:,h],vm[:,h])[0]
            self.add(bl.attn.WV.weight,dWV,self.lr['WV']);bl.attn.WV.bias.add_(self.lr['WV']*masked_mean(da,ma).mean(0).clamp(-.05,.05))
            if qk:
                qd=self.qdrive(bl,t,da,ma);dw,db=self.bank.outer(f'{i}WQ',masked_mean(t.x_in,ma),qd);self.add(bl.attn.WQ.weight,dw,self.lr['WQ']);bl.attn.WQ.bias.add_(self.lr['WQ']*db.clamp(-.05,.05))
                kp,km=self.k_events(bl,t,da,ma);dw,db=self.bank.outer(f'{i}WK',kp,km);self.add(bl.attn.WK.weight,dw,self.lr['WK']);bl.attn.WK.bias.add_(self.lr['WK']*db.clamp(-.05,.05))
        self.m.pos.add_(self.lr['pos']*dy.mean(0)[None].clamp(-.02,.02));self.step+=1
        return dict(ce=ce,acc=acc,mask=float(ratio.mean()))
    def cpu(self):return dict(step=self.step,bank=self.bank.cpu(),recip=self.r.cpu())
    def load(self,o):self.step=int(o.get('step',0));self.bank.load(o['bank']);self.r.load(o['recip'])
