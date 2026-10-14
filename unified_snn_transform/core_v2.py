from __future__ import annotations

import torch
import torch.nn as nn

import run_unified_gate as legacy


class LinearAttentionCoreV2(legacy.LinearAttentionCore):
    """Corrected persistent H/g implementation.

    The legacy prototype updated the full batch H/g tensor inside a per-example loop,
    so a changed token from sample b was broadcast into every sample's persistent
    state.  V2 keeps every update strictly within b while retaining the same
    event-cached semantics and mathematical operator.
    """
    def empty_state(self,x:torch.Tensor):
        B,N,_=x.shape
        return legacy.AttnState(
            H=torch.zeros(B,self.heads,self.m,self.dh,device=x.device,dtype=x.dtype),
            g=torch.zeros(B,self.heads,self.m,device=x.device,dtype=x.dtype),
            pk=torch.zeros(B,self.heads,N,self.m,device=x.device,dtype=x.dtype),
            v=torch.zeros(B,self.heads,N,self.dh,device=x.device,dtype=x.dtype),
            weight=torch.zeros(B,self.heads,N,device=x.device,dtype=x.dtype))

    def step(self,x,state,alpha,changed):
        B,N,D=x.shape; pq=self.q_features(x)
        a4=alpha[:,:,None,None]; a3=alpha[:,:,None]
        if state is None:
            pk,v=self.kv_all(x)
            H=torch.einsum('bhnm,bhnd->bhmd',pk,v); g=pk.sum(2)
            weight=torch.ones(B,self.heads,N,device=x.device,dtype=x.dtype)
            state=legacy.AttnState(H,g,pk,v,weight); kv_evals=B*N
        else:
            H0=state.H*a4; g0=state.g*a3; weight=state.weight*alpha[:,:,None]
            pk_cache=state.pk.clone(); v_cache=state.v.clone(); kv_evals=0
            Hs=[]; gs=[]
            for b in range(B):
                Hb=H0[b:b+1]; gb=g0[b:b+1]
                idx=torch.nonzero(changed[b],as_tuple=False).flatten()
                if idx.numel():
                    xb=x[b:b+1,idx]
                    kk=self._split(self.k(xb)); vv=self._split(self.v(xb)); pp=self.phi_k(kk,is_query=False)
                    oldpk=pk_cache[b:b+1,:,idx,:]; oldv=v_cache[b:b+1,:,idx,:]; ow=weight[b:b+1,:,idx]
                    Hb=Hb-torch.einsum('bhn,bhnm,bhnd->bhmd',ow,oldpk,oldv)+torch.einsum('bhnm,bhnd->bhmd',pp,vv)
                    gb=gb-(oldpk*ow[:,:,:,None]).sum(2)+pp.sum(2)
                    pk_cache[b,:,idx,:]=pp[0]; v_cache[b,:,idx,:]=vv[0]; weight[b,:,idx]=1.0
                    kv_evals+=int(idx.numel())
                Hs.append(Hb); gs.append(gb)
            H=torch.cat(Hs,0); g=torch.cat(gs,0)
            state=legacy.AttnState(H,g,pk_cache,v_cache,weight)
        num=torch.einsum('bhnm,bhmd->bhnd',pq,state.H)
        den=torch.einsum('bhnm,bhm->bhn',pq,state.g).unsqueeze(-1).clamp_min(1e-8)
        y=num/den; out=self.out(y.transpose(1,2).reshape(B,N,D))
        return out,state,{'kv_token_evals':kv_evals,'q_token_evals':B*N}


class UnifiedBlockV2(legacy.UnifiedBlock):
    def __init__(self,d=32,heads=2,features=32,ff=64,seed=17):
        nn.Module.__init__(self)
        self.d=d; self.heads=heads
        self.n1=nn.LayerNorm(d); self.attn=LinearAttentionCoreV2(d,heads,features,seed)
        self.n2=nn.LayerNorm(d); self.f1=nn.Linear(d,ff); self.f2=nn.Linear(ff,d)
        self.gate=legacy.SelectiveRetention(heads)
