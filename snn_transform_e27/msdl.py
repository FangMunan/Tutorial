from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn.functional as F
from model import D,K,L,LAYERS,HEADS,HEAD_DIM,FF,MASK,Model,random_mask
AF=.62;AS=.92;WF=.70;WS=.30;MF_A=.62;MS_A=.92;MWF=.70;MWS=.30

@dataclass
class DS: ef:torch.Tensor; es:torch.Tensor; mf:torch.Tensor; ms:torch.Tensor
class Bank:
    def __init__(self,dev):self.dev=dev;self.s:Dict[str,DS]={}
    def _state(self,name,pre_dim,mod_dim,dtype):
        st=self.s.get(name)
        if st is None:
            st=DS(torch.zeros(pre_dim,device=self.dev,dtype=dtype),torch.zeros(pre_dim,device=self.dev,dtype=dtype),torch.zeros(mod_dim,device=self.dev,dtype=dtype),torch.zeros(mod_dim,device=self.dev,dtype=dtype));self.s[name]=st
        return st
    def outer(self,name,pre,mod):
        if len(pre)==0:return torch.zeros((mod.shape[1],pre.shape[1]),device=pre.device),torch.zeros(mod.shape[1],device=pre.device)
        st=self._state(name,pre.shape[1],mod.shape[1],pre.dtype);er=[];mr=[];ef,es,mf,ms=st.ef,st.es,st.mf,st.ms
        for i in range(len(pre)):
            ef=AF*ef+(1-AF)*pre[i];es=AS*es+(1-AS)*pre[i];mf=MF_A*mf+(1-MF_A)*mod[i];ms=MS_A*ms+(1-MS_A)*mod[i]
            er.append(WF*ef+WS*es);mr.append(MWF*mf+MWS*ms)
        st.ef,st.es,st.mf,st.ms=ef.detach(),es.detach(),mf.detach(),ms.detach();E=torch.stack(er);M=torch.stack(mr)
        return M.T@E/len(E),M.mean(0)
    def events(self,name,pre,mod):
        """Matched token-wise MSDL events evaluated in parallel within one minibatch."""
        if len(pre)==0:return torch.zeros((mod.shape[1],pre.shape[1]),device=pre.device),torch.zeros(mod.shape[1],device=pre.device)
        st=self._state(name,pre.shape[1],mod.shape[1],pre.dtype)
        ef=AF*st.ef[None]+(1-AF)*pre;es=AS*st.es[None]+(1-AS)*pre
        mf=MF_A*st.mf[None]+(1-MF_A)*mod;ms=MS_A*st.ms[None]+(1-MS_A)*mod
        E=WF*ef+WS*es;M=MWF*mf+MWS*ms
        st.ef,st.es,st.mf,st.ms=ef.mean(0).detach(),es.mean(0).detach(),mf.mean(0).detach(),ms.mean(0).detach()
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
        self.lr={k:v*scale for k,v in dict(read=.020,dense=.004,W2=.0015,W1=.001,WO=.0015,WV=.0008,WQ=.00025,WK=.00025,pos=.00015,mask_embed=.0005).items()};self.sigq=.08;self.sigk=.08;self.clip=.10
    @torch.no_grad()
    def add(self,p,d,lr):
        n=d.norm().item();d=d*(self.clip/(n+1e-12)) if n>self.clip else d;p.add_(lr*d)
    @torch.no_grad()
    def q_events(self,bl,tr,da,m):
        bi,_=m.nonzero(as_tuple=True);q=tr.q.transpose(1,2)[m];dm=da.reshape(len(m),L,HEADS,HEAD_DIM)[m];gs=tr.g_state[bi];Hs=tr.H_state[bi]
        xi=torch.where(torch.rand(q.shape,generator=self.g,device=self.dev)<.5,-1.,1.)
        qp=bl.attn.rf.q((q+self.sigq*xi)[:,:,None,:]).squeeze(2);qm=bl.attn.rf.q((q-self.sigq*xi)[:,:,None,:]).squeeze(2)
        dp=torch.einsum('thm,thm->th',qp,gs).clamp_min(1e-8);dn=torch.einsum('thm,thm->th',qm,gs).clamp_min(1e-8)
        ap=torch.einsum('thm,thmd->thd',qp,Hs)/dp[...,None];am=torch.einsum('thm,thmd->thd',qm,Hs)/dn[...,None]
        qd=(((dm*(ap-am)).sum(-1)/(2*self.sigq))[...,None]*xi).reshape(len(q),D)
        return tr.x_in[m],qd
    @torch.no_grad()
    def k_events(self,bl,tr,da,m):
        B=len(m);pq=tr.phi_q;db=da.reshape(B,L,HEADS,HEAD_DIM).permute(0,2,1,3);grp=torch.arange(L,device=self.dev)%self.R;pp=[];mm=[]
        den0=torch.einsum('bhnm,bhm->bhn',pq,tr.g_state).clamp_min(1e-8);num0=torch.einsum('bhnm,bhmd->bhnd',pq,tr.H_state);cnt=m.sum(1).clamp_min(1).to(da.dtype)[:,None]
        for r in range(self.R):
            idx=torch.nonzero(grp==r).flatten();gc=max(1,int(idx.numel()));norm=math.sqrt(gc);xg=tr.x_in[:,idx,:].sum(1)/norm;xi=torch.where(torch.rand((B,HEADS,HEAD_DIM),generator=self.g,device=self.dev)<.5,-1.,1.)
            k0=tr.k[:,:,idx,:];v=tr.v[:,:,idx,:];pk0=tr.phi_k[:,:,idx,:];pkp=bl.attn.rf.k(k0+self.sigk*xi[:,:,None,:]);pkm=bl.attn.rf.k(k0-self.sigk*xi[:,:,None,:])
            dgp=(pkp-pk0).sum(2);dgm=(pkm-pk0).sum(2);dHp=torch.einsum('bhgm,bhgd->bhmd',pkp-pk0,v);dHm=torch.einsum('bhgm,bhgd->bhmd',pkm-pk0,v)
            dp=(den0+torch.einsum('bhnm,bhm->bhn',pq,dgp)).clamp_min(1e-8);dn=(den0+torch.einsum('bhnm,bhm->bhn',pq,dgm)).clamp_min(1e-8)
            ap=(num0+torch.einsum('bhnm,bhmd->bhnd',pq,dHp))/dp[...,None];am=(num0+torch.einsum('bhnm,bhmd->bhnd',pq,dHm))/dn[...,None]
            score=((db*(ap-am))*m[:,None,:,None]).sum((2,3))/(cnt*(2*self.sigk*norm));kd=(score[...,None]*xi).reshape(B,D);pp.append(xg);mm.append(kd)
        return torch.cat(pp),torch.cat(mm)
    @torch.no_grad()
    def v_events(self,tr,da,m):
        B=len(m);X=torch.einsum('bhnm,bnc->bhmc',tr.phi_k,tr.x_in);pq=tr.phi_q.transpose(1,2)
        den=torch.einsum('bnhm,bhm->bnh',pq,tr.g_state).clamp_min(1e-8);pre=torch.einsum('bnhm,bhmc->bnhc',pq,X)/den[...,None];mod=da.reshape(B,L,HEADS,HEAD_DIM)
        return pre[m],mod[m]
    @torch.no_grad()
    def batch(self,tok,qk=True):
        self.m.train();inp,ma,ratio=random_mask(tok,self.g);log,tr=self.m(inp,True);p=log.softmax(-1);err=torch.zeros_like(p);err.scatter_(-1,tok[...,None],1.);err.sub_(p);err.mul_(ma[...,None]);ce=F.cross_entropy(log[ma],tok[ma]).item();acc=(log[ma].argmax(-1)==tok[ma]).float().mean().item();self.r.probe(2)
        hz=self.m.ln(F.gelu(self.m.dense(tr[-1].x_out)));dw,db=self.bank.events('read',hz[ma],err[ma]);self.add(self.m.tok.weight[:K],dw,self.lr['read']);self.m.bias.add_(self.lr['read']*db.clamp(-.05,.05))
        dh=torch.einsum('dk,bnk->bnd',self.r.BR,err);dw,db=self.bank.events('dense',tr[-1].x_out[ma],dh[ma]);self.add(self.m.dense.weight,dw,self.lr['dense']);self.m.dense.bias.add_(self.lr['dense']*db.clamp(-.05,.05));dy=torch.einsum('de,bne->bnd',self.r.Bd,dh)
        for i in reversed(range(LAYERS)):
            bl=self.m.blocks[i];t=tr[i];mod=dy[ma];dw,db=self.bank.events(f'{i}W2',t.ff_hidden[ma],mod);self.add(bl.f2.weight,dw,self.lr['W2']);bl.f2.bias.add_(self.lr['W2']*db.clamp(-.05,.05))
            dhid=torch.einsum('fd,bnd->bnf',self.r.B2[i],dy);dw,db=self.bank.events(f'{i}W1',t.post_attention[ma],dhid[ma]);self.add(bl.f1.weight,dw,self.lr['W1']);bl.f1.bias.add_(self.lr['W1']*db.clamp(-.05,.05))
            dw,db=self.bank.events(f'{i}WO',t.attention_read[ma],mod);self.add(bl.attn.WO.weight,dw,self.lr['WO']);bl.attn.WO.bias.add_(self.lr['WO']*db.clamp(-.05,.05));da=torch.einsum('de,bne->bnd',self.r.BO[i],dy)
            vp,vm=self.v_events(t,da,ma);dWV=torch.zeros_like(bl.attn.WV.weight)
            for h in range(HEADS):dWV[h*HEAD_DIM:(h+1)*HEAD_DIM]=self.bank.events(f'{i}WV{h}',vp[:,h],vm[:,h])[0]
            self.add(bl.attn.WV.weight,dWV,self.lr['WV']);bl.attn.WV.bias.add_(self.lr['WV']*vm.reshape(-1,D).mean(0).clamp(-.05,.05))
            if qk:
                qp,qm=self.q_events(bl,t,da,ma);dw,db=self.bank.events(f'{i}WQ',qp,qm);self.add(bl.attn.WQ.weight,dw,self.lr['WQ']);bl.attn.WQ.bias.add_(self.lr['WQ']*db.clamp(-.05,.05))
                kp,km=self.k_events(bl,t,da,ma);dw,db=self.bank.events(f'{i}WK',kp,km);self.add(bl.attn.WK.weight,dw,self.lr['WK']);bl.attn.WK.bias.add_(self.lr['WK']*db.clamp(-.05,.05))
        ones=torch.ones((int(ma.sum()),1),device=self.dev,dtype=dy.dtype);dmask,_=self.bank.events('mask_embed',ones,dy[ma]);self.add(self.m.tok.weight[MASK],dmask[:,0],self.lr['mask_embed'])
        self.m.pos.add_(self.lr['pos']*dy.mean(0)[None].clamp(-.02,.02));self.step+=1
        return dict(ce=ce,acc=acc,mask=float(ratio.mean()),events=int(ma.sum()))
    def cpu(self):return dict(step=self.step,bank=self.bank.cpu(),recip=self.r.cpu())
    def load(self,o):self.step=int(o.get('step',0));self.bank.load(o['bank']);self.r.load(o['recip'])
