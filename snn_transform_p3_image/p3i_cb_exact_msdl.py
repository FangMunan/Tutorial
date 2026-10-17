#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3I-A: first image/patch gate for the developed P3 MSDL core.

Scientific boundary
-------------------
This is an image-interface gate, not a redefinition of P3-D.
It preserves the developed P3 learning/computation core:
  * resident H/g linear-attention state
  * resident Xphi V eligibility
  * fixed-R collective K-write eligibility
  * persistent Izhikevich-derived E/M populations
  * local dW ~ M_post E_pre^T for WQ/WK/WV/WO/W1/W2/WR
  * continuously adaptive reciprocal populations BR/BO/B2
  * one uninterrupted plastic state trajectory during image training

To isolate the image/task interface, this first gate uses the exact positive random
feature map instead of the P1-G neuronal Q/K LUT. P1-G can be reintroduced only
after image functionality passes; the accepted P3 report already established that
this interface swap is functionally near-neutral on the synthetic gate.

The only new scientific variable is optional bounded frequency balancing of the
observed categorical mismatch. Evaluation remains unweighted.

Class conditioning is retained in this gate, exactly as in R6, to isolate the
learning mechanism. This is therefore NOT a label-free/autonomous-concept result.
No candidate autograd/backpropagation/torch optimizer is used.
"""
from __future__ import annotations
import argparse, csv, json, math, random, sys, time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from numba import njit

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'snn_transform_e29'))
from maskgit_b0_mnist012 import load, subset012, split, encode, decode, K as TOKENS, MASK

FORCE_DT_MS=80.0
AF,AS=0.62,0.92
WF,WS=0.70,0.30
TAU_F_MS=-FORCE_DT_MS/math.log(AF); TAU_S_MS=-FORCE_DT_MS/math.log(AS)
RF,RS=1.0/TAU_F_MS,1.0/TAU_S_MS
A_IZH=1e-3; S_IZH=-(RF+RS); P_IZH=RF*RS; Q_IZH=S_IZH+A_IZH
B_IZH=Q_IZH+P_IZH/A_IZH
V0=(Q_IZH-5.0)/0.08; U0=B_IZH*V0
I0=-0.04*V0*V0-5.0*V0-140.0+U0
NUM_S=WF*RF+WS*RS; NUM_0=RF*RS
G_E=1e-5; G_M=1e-6; N_SUB=80; FORCE_STEPS=27
MIRROR_SIGMA=0.1; MIRROR_LEAK=5e-5
ETA_BR=0.0338; ETA_BO=0.0335; ETA_B2=0.0335
BASE_LRQ=0.135; BASE_LRK=0.1125; BASE_LRV=0.039375
BASE_LRO=0.1125; BASE_LR1=0.225; BASE_LR2=0.225; BASE_LRR=0.001
SIGQ=0.05; SIGK=0.05

def readout_coeff(gain):
    cv=NUM_S/gain; cu=(NUM_0/gain-cv*A_IZH)/(A_IZH*B_IZH); return cv,cu
CV_E,CU_E=readout_coeff(G_E); CV_M,CU_M=readout_coeff(G_M)

@njit(cache=True)
def seed_numba(seed): np.random.seed(seed)

@njit(cache=True)
def softmax_numba(z):
    mx=np.max(z); out=np.exp(z-mx); return out/np.sum(out)

@njit(cache=True)
def fast_izh_step(v,u,target_input,pred_input,gain,cv,cu):
    spikes=0
    for _ in range(N_SUB):
        for j in range(v.shape[0]):
            I=I0+gain*target_input[j]-gain*pred_input[j]
            dv1=0.04*v[j]*v[j]+5.0*v[j]+140.0-u[j]+I
            du1=A_IZH*(B_IZH*v[j]-u[j])
            vm=v[j]+0.5*dv1; um=u[j]+0.5*du1
            dv2=0.04*vm*vm+5.0*vm+140.0-um+I
            du2=A_IZH*(B_IZH*vm-um)
            v[j]+=dv2; u[j]+=du2
            if v[j]>=30.0:
                v[j]=-65.0; u[j]+=8.0; spikes+=1
    out=np.empty(v.shape[0],np.float64)
    for j in range(v.shape[0]): out[j]=cv*(v[j]-V0)+cu*(u[j]-U0)
    return out,spikes

@njit(cache=True)
def reciprocal_microsteps(BR,BO,B2,WR,WO,W2,
    vm_br,um_br,ve_br,ue_br,zm_br,ze_br,
    vm_bo,um_bo,ve_bo,ue_bo,zm_bo,ze_bo,
    vm_b2,um_b2,ve_b2,ue_b2,zm_b2,ze_b2,
    nsteps,sigma,eta_br,eta_bo,eta_b2,leak):
    D=WO.shape[0]; ff=W2.shape[1]; V=WR.shape[0]
    for _ in range(nsteps):
        df=np.random.normal(0.0,sigma,D); dlogit=np.zeros(V,np.float64)
        for vv in range(V):
            for o in range(D): dlogit[vv]+=WR[vv,o]*df[o]
        M,_=fast_izh_step(vm_br,um_br,df,zm_br,G_M,CV_M,CU_M)
        E,_=fast_izh_step(ve_br,ue_br,dlogit,ze_br,G_E,CV_E,CU_E)
        for o in range(D):
            for vv in range(V): BR[o,vv]=(1.0-leak)*BR[o,vv]+eta_br*M[o]*E[vv]
        da=np.random.normal(0.0,sigma,D); dz=np.zeros(D,np.float64)
        for o in range(D):
            for j in range(D): dz[o]+=WO[o,j]*da[j]
        M,_=fast_izh_step(vm_bo,um_bo,da,zm_bo,G_M,CV_M,CU_M)
        E,_=fast_izh_step(ve_bo,ue_bo,dz,ze_bo,G_E,CV_E,CU_E)
        for j in range(D):
            for o in range(D): BO[j,o]=(1.0-leak)*BO[j,o]+eta_bo*M[j]*E[o]
        dh=np.random.normal(0.0,sigma,ff); dy=np.zeros(D,np.float64)
        for o in range(D):
            for i in range(ff): dy[o]+=W2[o,i]*dh[i]
        M,_=fast_izh_step(vm_b2,um_b2,dh,zm_b2,G_M,CV_M,CU_M)
        E,_=fast_izh_step(ve_b2,ue_b2,dy,ze_b2,G_E,CV_E,CU_E)
        for i in range(ff):
            for o in range(D): B2[i,o]=(1.0-leak)*B2[i,o]+eta_b2*M[i]*E[o]

@njit(cache=True)
def train_readout_msdl(features,targets,target_weights,n_paths,lr,seed):
    np.random.seed(seed); nctx,nin=features.shape; V=target_weights.shape[0]
    W=np.zeros((V,nin),np.float64)
    ve=np.full(nin,V0);ue=np.full(nin,U0);vm=np.full(V,V0);um=np.full(V,U0)
    ze=np.zeros(nin);zm=np.zeros(V)
    for _p in range(n_paths):
        W0=W.copy();acc=np.zeros_like(W)
        for _k in range(FORCE_STEPS):
            ci=np.random.randint(0,nctx);x=features[ci];target=int(targets[ci]);p=softmax_numba(W0@x)
            E,_=fast_izh_step(ve,ue,x,ze,G_E,CV_E,CU_E)
            one=np.zeros(V);one[target]=1.0;ww=target_weights[target]
            M,_=fast_izh_step(vm,um,ww*one,ww*p,G_M,CV_M,CU_M)
            acc+=np.outer(M,E)
        W+=lr*acc/FORCE_STEPS
    return W

def sparse_mask(rng,shape,density=0.25):
    out=np.zeros(shape,np.float64); k=max(1,int(round(shape[1]*density)))
    for i in range(shape[0]): out[i,rng.choice(shape[1],size=k,replace=False)]=1.0
    return out

def phi_q(q,omega):
    hd=q.shape[0];xx=q*(hd**-0.25);a=omega@xx-0.5*np.dot(xx,xx);a-=a.max();a=np.clip(a,-30,0)
    return np.exp(a)/math.sqrt(omega.shape[0])+1e-8

def phi_k(k,omega):
    hd=k.shape[0];xx=k*(hd**-0.25);a=omega@xx-0.5*np.dot(xx,xx);a=np.clip(a,-30,30)
    return np.exp(a)/math.sqrt(omega.shape[0])+1e-8

def rotating_groups(N,R,phase):
    if N%R: raise ValueError('R must divide N')
    shift=phase%N; return np.array([((n+shift)%N)%R for n in range(N)],np.int64)

def make_dirs(rng,R,H,hd): return np.where(rng.random((R,H,hd))<0.5,-1.0,1.0).astype(np.float64)

@dataclass
class Config:
    D:int=96; H:int=8; M:int=24; FF:int=192; N:int=49; V:int=64; R:int=7
    density:float=0.25; reciprocal_micro:int=4

def init_model(cfg,seed):
    if cfg.D%cfg.H: raise ValueError('D/H')
    if cfg.N%cfg.R: raise ValueError('N/R')
    rng=np.random.default_rng(seed);hd=cfg.D//cfg.H
    def W(o,i):return rng.normal(0,1/math.sqrt(i),size=(o,i)).astype(np.float64)
    maskO=sparse_mask(rng,(cfg.D,cfg.D),cfg.density);mask1=sparse_mask(rng,(cfg.FF,cfg.D),cfg.density);mask2=sparse_mask(rng,(cfg.D,cfg.FF),cfg.density)
    WQ,WK,WV=W(cfg.D,cfg.D),W(cfg.D,cfg.D),W(cfg.D,cfg.D);WO=W(cfg.D,cfg.D)*maskO;W1=W(cfg.FF,cfg.D)*mask1;W2=W(cfg.D,cfg.FF)*mask2
    WR=np.zeros((cfg.V,cfg.D+1));BR=np.zeros((cfg.D,cfg.V));BO=np.zeros((cfg.D,cfg.D));B2=np.zeros((cfg.FF,cfg.D))
    omega=rng.normal(size=(cfg.H,cfg.M,hd));omega/=np.linalg.norm(omega,axis=2,keepdims=True)+1e-12;omega*=math.sqrt(hd)
    scale=math.sqrt(24.0/cfg.D)
    return dict(WQ=WQ,WK=WK,WV=WV,WO=WO,W1=W1,W2=W2,WR=WR,BR=BR,BO=BO,B2=B2,
        maskO=maskO,mask1=mask1,mask2=mask2,omega=omega,bq=np.zeros(cfg.D),bk=np.zeros(cfg.D),bv=np.zeros(cfg.D),bo=np.zeros(cfg.D),b1=np.zeros(cfg.FF),b2=np.zeros(cfg.D),theta=np.zeros(cfg.FF),gamma2=np.ones(cfg.D),beta2=np.zeros(cfg.D),eps2=1e-5,
        lrq=BASE_LRQ*scale,lrk=BASE_LRK*scale,lrv=BASE_LRV*scale,lro=BASE_LRO*scale,lr1=BASE_LR1*scale,lr2=BASE_LR2*scale,lrr=BASE_LRR,recip_scale=scale,sigq=SIGQ,sigk=SIGK)

def build_embeddings(cfg,seed):
    rng=np.random.default_rng(seed);tok=rng.normal(size=(cfg.V+1,cfg.D));tok/=np.linalg.norm(tok,axis=1,keepdims=True)+1e-12;tok*=math.sqrt(cfg.D)*0.70
    pos=rng.normal(size=(cfg.N,cfg.D));pos/=np.linalg.norm(pos,axis=1,keepdims=True)+1e-12;pos*=math.sqrt(cfg.D)*0.35
    cls=rng.normal(size=(3,cfg.D));cls/=np.linalg.norm(cls,axis=1,keepdims=True)+1e-12;cls*=math.sqrt(cfg.D)*0.35
    return tok,pos,cls

def context_from_arr(arr,label,tok,pos,cls):
    x=tok[arr]+pos+cls[int(label)][None,:]
    mu=x.mean(1,keepdims=True);var=np.mean((x-mu)**2,1,keepdims=True);xn=(x-mu)/np.sqrt(var+1e-5)
    return xn,x

def static_feature(model,cfg,xn,xres,t):
    WQ,WK,WV=model['WQ'],model['WK'],model['WV'];hd=cfg.D//cfg.H
    q=(WQ@xn[t]+model['bq']).reshape(cfg.H,hd);kk=(xn@WK.T+model['bk']).reshape(cfg.N,cfg.H,hd).transpose(1,0,2);vv=(xn@WV.T+model['bv']).reshape(cfg.N,cfg.H,hd).transpose(1,0,2)
    a=np.empty(cfg.D)
    for h in range(cfg.H):
        pq=phi_q(q[h],model['omega'][h]);P=np.array([phi_k(kk[h,n],model['omega'][h]) for n in range(cfg.N)]);scores=P@pq;den=max(1e-8,float(scores.sum()));a[h*hd:(h+1)*hd]=(scores[:,None]*vv[h]).sum(0)/den
    z=xres+model['bo']+model['WO']@a;mu=z.mean();inv=1/math.sqrt(float(np.mean((z-mu)**2))+model['eps2']);u=(z-mu)*inv*model['gamma2']+model['beta2'];hh=np.maximum(0,model['W1']@u+model['b1']-model['theta']);y=z+model['W2']@hh+model['b2'];rms=math.sqrt(float(np.mean(y*y)))+1e-8
    rf=np.empty(cfg.D+1);rf[:-1]=0.5*y/rms;rf[-1]=0.5;return rf

class P3ImageSystem:
    def __init__(self,cfg,model,seed):
        self.c=cfg
        for k,v in model.items(): setattr(self,k,np.array(v,copy=True) if isinstance(v,np.ndarray) else v)
        c=cfg;D=c.D;ff=c.FF;V=c.V;P=c.R;hd=D//c.H
        self.rng=np.random.default_rng(seed);np.random.seed(seed+8181);seed_numba(seed+9181)
        self.vm_br=np.full(D,V0);self.um_br=np.full(D,U0);self.ve_br=np.full(V,V0);self.ue_br=np.full(V,U0);self.zm_br=np.zeros(D);self.ze_br=np.zeros(V)
        self.vm_bo=np.full(D,V0);self.um_bo=np.full(D,U0);self.ve_bo=np.full(D,V0);self.ue_bo=np.full(D,U0);self.zm_bo=np.zeros(D);self.ze_bo=np.zeros(D)
        self.vm_b2=np.full(ff,V0);self.um_b2=np.full(ff,U0);self.ve_b2=np.full(D,V0);self.ue_b2=np.full(D,U0);self.zm_b2=np.zeros(ff);self.ze_b2=np.zeros(D)
        def pop(n): return np.full(n,V0),np.full(n,U0),np.zeros(n)
        self.ver,self.uer,self.zer=pop(D+1);self.vmr,self.umr,self.zmr=pop(V)
        self.veq,self.ueq,self.zeq=pop(D);self.vmq,self.umq,self.zmq=pop(D)
        self.vek,self.uek,self.zek=pop(P*D);self.vmk,self.umk,self.zmk=pop(P*D)
        self.vev,self.uev,self.zev=pop(c.H*D);self.vmv,self.umv,self.zmv=pop(D)
        self.veo,self.ueo,self.zeo=pop(D);self.vmo,self.umo,self.zmo=pop(D)
        self.ve1,self.ue1,self.ze1=pop(D);self.vm1,self.um1,self.zm1=pop(ff)
        self.ve2,self.ue2,self.ze2=pop(ff);self.vm2,self.um2,self.zm2=pop(D)
        self.xcache=np.zeros((c.N,D));self.qraw=np.zeros((c.H,c.N,hd));self.kraw=np.zeros_like(self.qraw);self.vcache=np.zeros_like(self.qraw);self.pq=np.zeros((c.H,c.N,c.M));self.pk=np.zeros_like(self.pq);self.Hstate=np.zeros((c.H,c.M,hd));self.gstate=np.zeros((c.H,c.M));self.Xphi=np.zeros((c.H,c.M,D))
        self.groups=np.zeros(c.N,np.int64);self.dirs=np.ones((P,c.H,hd));self.xgrp=np.zeros((P,D));self.Hp=np.zeros((P,c.H,c.M,hd));self.Gp=np.zeros((P,c.H,c.M));self.Hm=np.zeros_like(self.Hp);self.Gm=np.zeros_like(self.Gp)
        self.initialized=False;self.image_count=0;self._clear();self.event_count=0;self.update_count=0;self.obs_nll=0.;self.obs_count=0
    def _clear(self):
        self.aq=np.zeros_like(self.WQ);self.ak=np.zeros_like(self.WK);self.av=np.zeros_like(self.WV);self.ao=np.zeros_like(self.WO);self.a1=np.zeros_like(self.W1);self.a2=np.zeros_like(self.W2);self.ar=np.zeros_like(self.WR);self.accum=0
    def weights(self):return {k:np.array(getattr(self,k),copy=True) for k in ('WQ','WK','WV','WO','W1','W2','WR','BR','BO','B2')}
    def _project(self,x):
        c=self.c;hd=c.D//c.H;return (self.WQ@x+self.bq).reshape(c.H,hd),(self.WK@x+self.bk).reshape(c.H,hd),(self.WV@x+self.bv).reshape(c.H,hd)
    def _reld(self,h,k,v,xi):
        pk=phi_k(k,self.omega[h]);pp=phi_k(k+self.sigk*xi,self.omega[h]);pm=phi_k(k-self.sigk*xi,self.omega[h]);dp=pp-pk;dm=pm-pk
        return dp[:,None]*v[None,:],dp,dm[:,None]*v[None,:],dm
    def rebuild(self,xn):
        c=self.c;hd=c.D//c.H;self.groups=rotating_groups(c.N,c.R,self.image_count);self.dirs=make_dirs(self.rng,c.R,c.H,hd);self.image_count+=1
        self.Hstate.fill(0);self.gstate.fill(0);self.Xphi.fill(0);self.xgrp.fill(0);self.Hp.fill(0);self.Gp.fill(0);self.Hm.fill(0);self.Gm.fill(0);norm=math.sqrt(c.N//c.R)
        for n in range(c.N):
            x=np.asarray(xn[n],float);q,k,v=self._project(x);self.xcache[n]=x;r=int(self.groups[n]);self.xgrp[r]+=x/norm
            for h in range(c.H):
                pq=phi_q(q[h],self.omega[h]);pk=phi_k(k[h],self.omega[h]);self.qraw[h,n]=q[h];self.kraw[h,n]=k[h];self.vcache[h,n]=v[h];self.pq[h,n]=pq;self.pk[h,n]=pk;self.gstate[h]+=pk;self.Hstate[h]+=pk[:,None]*v[h];self.Xphi[h]+=pk[:,None]*x
                hp,gp,hm,gm=self._reld(h,k[h],v[h],self.dirs[r,h]);self.Hp[r,h]+=hp;self.Gp[r,h]+=gp;self.Hm[r,h]+=hm;self.Gm[r,h]+=gm
        self.initialized=True
    def update_token(self,n,newx):
        c=self.c;old=self.xcache[n].copy()
        if np.array_equal(old,newx):return
        r=int(self.groups[n]);norm=math.sqrt(c.N//c.R);self.xgrp[r]+=(newx-old)/norm
        for h in range(c.H):
            op=self.pk[h,n].copy();ov=self.vcache[h,n].copy();ok=self.kraw[h,n].copy();self.gstate[h]-=op;self.Hstate[h]-=op[:,None]*ov;self.Xphi[h]-=op[:,None]*old;hp,gp,hm,gm=self._reld(h,ok,ov,self.dirs[r,h]);self.Hp[r,h]-=hp;self.Gp[r,h]-=gp;self.Hm[r,h]-=hm;self.Gm[r,h]-=gm
        q,k,v=self._project(newx)
        for h in range(c.H):
            pq=phi_q(q[h],self.omega[h]);pk=phi_k(k[h],self.omega[h]);self.qraw[h,n]=q[h];self.kraw[h,n]=k[h];self.vcache[h,n]=v[h];self.pq[h,n]=pq;self.pk[h,n]=pk;self.gstate[h]+=pk;self.Hstate[h]+=pk[:,None]*v[h];self.Xphi[h]+=pk[:,None]*newx;hp,gp,hm,gm=self._reld(h,k[h],v[h],self.dirs[r,h]);self.Hp[r,h]+=hp;self.Gp[r,h]+=gp;self.Hm[r,h]+=hm;self.Gm[r,h]+=gm
        self.xcache[n]=newx
    def _read(self,t):
        c=self.c;hd=c.D//c.H;a=np.empty(c.D);q=np.empty((c.H,hd));num=np.empty((c.H,hd));den=np.empty(c.H)
        for h in range(c.H):q[h]=self.qraw[h,t];pq=self.pq[h,t];dd=max(1e-8,float(pq@self.gstate[h]));den[h]=dd;nn=pq@self.Hstate[h];num[h]=nn;a[h*hd:(h+1)*hd]=nn/dd
        return a,q,num,den
    def _qdrive(self,q,drive_a):
        c=self.c;hd=c.D//c.H;out=np.zeros(c.D)
        for h in range(c.H):
            xi=np.where(self.rng.random(hd)<.5,-1.,1.);p=phi_q(q[h]+self.sigq*xi,self.omega[h]);m=phi_q(q[h]-self.sigq*xi,self.omega[h]);dp=max(1e-8,float(p@self.gstate[h]));dm=max(1e-8,float(m@self.gstate[h]));ap=(p@self.Hstate[h])/dp;am=(m@self.Hstate[h])/dm;sc=float(drive_a[h*hd:(h+1)*hd]@(ap-am))/(2*self.sigq);out[h*hd:(h+1)*hd]=sc*xi
        return out
    def _kdrive(self,q,num,den,drive_a):
        c=self.c;hd=c.D//c.H;out=np.zeros((c.R,c.D));norm=math.sqrt(c.N//c.R)
        for r in range(c.R):
            for h in range(c.H):
                pq=phi_q(q[h],self.omega[h]);dgp=float(pq@self.Gp[r,h]);dgm=float(pq@self.Gm[r,h]);dHp=pq@self.Hp[r,h];dHm=pq@self.Hm[r,h];dp=max(1e-8,float(den[h]+dgp));dm=max(1e-8,float(den[h]+dgm));ap=(num[h]+dHp)/dp;am=(num[h]+dHm)/dm;sc=float(drive_a[h*hd:(h+1)*hd]@(ap-am))/(2*self.sigk*norm);out[r,h*hd:(h+1)*hd]=sc*self.dirs[r,h]
        return out
    def _vinput(self,t,den):
        c=self.c;s=np.empty(c.H*c.D)
        for h in range(c.H):s[h*c.D:(h+1)*c.D]=(self.pq[h,t]@self.Xphi[h])/max(1e-8,float(den[h]))
        return s
    def _tail(self,a,xres):
        z=xres+self.bo+self.WO@a;mu=z.mean();inv=1/math.sqrt(float(np.mean((z-mu)**2))+self.eps2);u=(z-mu)*inv*self.gamma2+self.beta2;h=np.maximum(0,self.W1@u+self.b1-self.theta);y=z+self.W2@h+self.b2;rms=math.sqrt(float(np.mean(y*y)))+1e-8;rf=np.empty(self.c.D+1);rf[:-1]=.5*y/rms;rf[-1]=.5;return u,h,rf,softmax_numba(self.WR@rf)
    def forward_only(self,xn,xres,t,new_image=False,changed=None):
        if new_image or not self.initialized:self.rebuild(xn)
        elif changed is not None:self.update_token(int(changed),np.asarray(xn[int(changed)],float))
        a,q,num,den=self._read(int(t));u,h,rf,p=self._tail(a,xres);return np.asarray(p,float)
    def step(self,xn,xres,t,target,weight=1.0,new_image=False,changed=None):
        c=self.c
        reciprocal_microsteps(self.BR,self.BO,self.B2,self.WR,self.WO,self.W2,self.vm_br,self.um_br,self.ve_br,self.ue_br,self.zm_br,self.ze_br,self.vm_bo,self.um_bo,self.ve_bo,self.ue_bo,self.zm_bo,self.ze_bo,self.vm_b2,self.um_b2,self.ve_b2,self.ue_b2,self.zm_b2,self.ze_b2,c.reciprocal_micro,MIRROR_SIGMA,ETA_BR*self.recip_scale,ETA_BO*self.recip_scale,ETA_B2*self.recip_scale,MIRROR_LEAK)
        if new_image or not self.initialized:self.rebuild(xn)
        elif changed is not None:self.update_token(int(changed),np.asarray(xn[int(changed)],float))
        a,q,num,den=self._read(int(t));u,h,rf,p=self._tail(a,xres);one=np.zeros(c.V);one[int(target)]=1.;ww=float(weight);err=ww*(one-p);self.obs_nll-=math.log(max(float(p[int(target)]),1e-12));self.obs_count+=1
        ER,_=fast_izh_step(self.ver,self.uer,rf,self.zer,G_E,CV_E,CU_E);MR,_=fast_izh_step(self.vmr,self.umr,ww*one,ww*p,G_M,CV_M,CU_M);self.ar+=np.outer(MR,ER)
        dy=self.BR@err;dh=self.B2@dy;da=self.BO@dy
        E2,_=fast_izh_step(self.ve2,self.ue2,h,self.ze2,G_E,CV_E,CU_E);M2,_=fast_izh_step(self.vm2,self.um2,dy,self.zm2,G_M,CV_M,CU_M);self.a2+=np.outer(M2,E2)*self.mask2
        E1,_=fast_izh_step(self.ve1,self.ue1,u,self.ze1,G_E,CV_E,CU_E);M1,_=fast_izh_step(self.vm1,self.um1,dh,self.zm1,G_M,CV_M,CU_M);self.a1+=np.outer(M1,E1)*self.mask1
        EO,_=fast_izh_step(self.veo,self.ueo,a,self.zeo,G_E,CV_E,CU_E);MO,_=fast_izh_step(self.vmo,self.umo,dy,self.zmo,G_M,CV_M,CU_M);self.ao+=np.outer(MO,EO)*self.maskO
        sv=self._vinput(int(t),den);EV,_=fast_izh_step(self.vev,self.uev,sv,self.zev,G_E,CV_E,CU_E);MV,_=fast_izh_step(self.vmv,self.umv,da,self.zmv,G_M,CV_M,CU_M);hd=c.D//c.H
        for r in range(c.D):self.av[r]+=MV[r]*EV[(r//hd)*c.D:(r//hd+1)*c.D]
        EK,_=fast_izh_step(self.vek,self.uek,self.xgrp.reshape(-1),self.zek,G_E,CV_E,CU_E);kd=self._kdrive(q,num,den,da);MK,_=fast_izh_step(self.vmk,self.umk,kd.reshape(-1),self.zmk,G_M,CV_M,CU_M);self.ak+=(MK.reshape(c.R,c.D).T@EK.reshape(c.R,c.D))/c.R
        qd=self._qdrive(q,da);EQ,_=fast_izh_step(self.veq,self.ueq,xn[int(t)],self.zeq,G_E,CV_E,CU_E);MQ,_=fast_izh_step(self.vmq,self.umq,qd,self.zmq,G_M,CV_M,CU_M);self.aq+=np.outer(MQ,EQ)
        self.event_count+=1;self.accum+=1
        if self.accum>=FORCE_STEPS:self.apply_update()
        return np.asarray(p,float)
    def apply_update(self):
        inv=1/max(1,self.accum);self.WR+=self.lrr*self.ar*inv;self.WQ+=self.lrq*self.aq*inv;self.WK+=self.lrk*self.ak*inv;self.WV+=self.lrv*self.av*inv;self.WO+=self.lro*self.ao*inv;self.WO*=self.maskO;self.W1+=self.lr1*self.a1*inv;self.W1*=self.mask1;self.W2+=self.lr2*self.a2*inv;self.W2*=self.mask2;self.update_count+=1;self._clear()

def class_weights(tokens,variant,beta=0.9999,cap=4.0):
    counts=np.bincount(tokens.reshape(-1),minlength=TOKENS).astype(np.float64);freq=counts/counts.sum();w=np.ones(TOKENS)
    if variant!='plain':
        eff=(1-beta)/(1-np.power(beta,np.maximum(counts,1)))
        eff/=np.sum(freq*eff)
        if variant=='cb_sqrt4':eff=np.sqrt(eff)
        w=eff/np.sum(freq*eff);w=np.clip(w,0.5,cap);w/=np.sum(freq*w);w=np.minimum(w,cap)
    return counts,w

def build_scaffold_contexts(model,cfg,tokens,labels,tok,pos,cls,nctx,seed):
    rng=np.random.default_rng(seed);F=np.empty((nctx,cfg.D+1));T=np.empty(nctx,np.int64)
    for i in range(nctx):
        ii=int(rng.integers(len(tokens)));order=rng.permutation(cfg.N);revealed=int(rng.integers(0,cfg.N));t=int(order[revealed]);arr=np.full(cfg.N,MASK,np.int64);arr[order[:revealed]]=tokens[ii,order[:revealed]];xn,xraw=context_from_arr(arr,labels[ii],tok,pos,cls);F[i]=static_feature(model,cfg,xn,xraw[t],t);T[i]=int(tokens[ii,t])
    return F,T

def eval_masked(model,cfg,tokens,labels,tok,pos,cls,ratio,seed,nimages=128,targets_per_image=6):
    rng=np.random.default_rng(seed);ce=[];acc=[];fgce=[];fgacc=[];fgmass=[]
    dom=int(np.argmax(np.bincount(tokens.reshape(-1),minlength=cfg.V)))
    sys=P3ImageSystem(cfg,model,seed+30000)
    for ii in rng.choice(len(tokens),size=min(nimages,len(tokens)),replace=False):
        nmask=max(1,int(round(ratio*cfg.N)));masked=rng.choice(cfg.N,size=nmask,replace=False);arr=tokens[ii].copy();arr[masked]=MASK;xn,xraw=context_from_arr(arr,labels[ii],tok,pos,cls)
        picks=rng.choice(masked,size=min(targets_per_image,nmask),replace=False)
        for t in picks:
            p=sys.forward_only(xn,xraw[t],int(t),new_image=True);z=int(tokens[ii,t]);ce.append(-math.log(max(p[z],1e-12)));acc.append(int(np.argmax(p)==z))
            if z!=dom:fgce.append(ce[-1]);fgacc.append(acc[-1]);fgmass.append(1-p[dom])
    return {'ce':float(np.mean(ce)),'acc':float(np.mean(acc)),'foreground_ce':float(np.mean(fgce)) if fgce else None,'foreground_acc':float(np.mean(fgacc)) if fgacc else None,'foreground_non_dominant_mass':float(np.mean(fgmass)) if fgmass else None,'samples':len(ce),'fg_samples':len(fgce),'dominant_token':dom}

def train_full(system,cfg,tr,ytr,tok,pos,cls,weights,events,seed,report_every=5000):
    rng=np.random.default_rng(seed);done=0;hist=[];t0=time.time()
    while done<events:
        ii=int(rng.integers(len(tr)));order=rng.permutation(cfg.N);arr=np.full(cfg.N,MASK,np.int64);changed=None
        for j,t in enumerate(order):
            if done>=events:break
            xn,xraw=context_from_arr(arr,ytr[ii],tok,pos,cls);target=int(tr[ii,t]);system.step(xn,xraw[t],int(t),target,float(weights[target]),new_image=(j==0),changed=changed);arr[t]=target;changed=int(t);done+=1
            if done%report_every==0:
                rec={'events':done,'online_nll':system.obs_nll/max(1,system.obs_count),'updates':system.update_count,'elapsed_s':time.time()-t0,'mean_abs_BR':float(np.mean(np.abs(system.BR))),'mean_abs_BO':float(np.mean(np.abs(system.BO))),'mean_abs_B2':float(np.mean(np.abs(system.B2)))};hist.append(rec);print('TRAIN',json.dumps(rec),flush=True)
    if system.accum:system.apply_update()
    return hist

def generate_images(model,cfg,labels,tok,pos,cls,codebook,seed):
    rng=np.random.default_rng(seed);out=[]
    for gi,lab in enumerate(labels):
        s=P3ImageSystem(cfg,model,seed+100000+gi);arr=np.full(cfg.N,MASK,np.int64);order=rng.permutation(cfg.N);changed=None
        for j,t in enumerate(order):
            xn,xraw=context_from_arr(arr,int(lab),tok,pos,cls);p=s.forward_only(xn,xraw[t],int(t),new_image=(j==0),changed=changed);z=int(rng.choice(cfg.V,p=p/p.sum()));arr[t]=z;changed=int(t)
        out.append(arr.copy())
    arr=np.asarray(out,np.int16);return arr,decode(arr,codebook)

def save_grid(img,lab,path):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    n=len(img);cols=12;rows=math.ceil(n/cols);fig,ax=plt.subplots(rows,cols,figsize=(cols,rows*1.2));ax=np.asarray(ax).reshape(-1)
    for i,a in enumerate(ax):
        a.axis('off')
        if i<n:a.imshow(img[i],cmap='gray',vmin=0,vmax=1,interpolation='nearest');a.set_title(str(int(lab[i])),fontsize=7)
    plt.tight_layout();fig.savefig(path,dpi=170,bbox_inches='tight');plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--codebook',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--variant',choices=['plain','cb_sqrt4'],default='cb_sqrt4');ap.add_argument('--D',type=int,default=96);ap.add_argument('--heads',type=int,default=8);ap.add_argument('--features',type=int,default=24);ap.add_argument('--ff',type=int,default=192);ap.add_argument('--R',type=int,default=7);ap.add_argument('--scaffold-contexts',type=int,default=2000);ap.add_argument('--scaffold-paths',type=int,default=2500);ap.add_argument('--scaffold-lr',type=float,default=.6);ap.add_argument('--events',type=int,default=30000);ap.add_argument('--eval-images',type=int,default=128);ap.add_argument('--gen-per-class',type=int,default=12);ap.add_argument('--seed',type=int,default=7311);ap.add_argument('--cb-beta',type=float,default=.9999);ap.add_argument('--cb-cap',type=float,default=4.0);ap.add_argument('--report-every',type=int,default=5000);a=ap.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);a.out.mkdir(parents=True,exist_ok=True);cfg=Config(D=a.D,H=a.heads,M=a.features,FF=a.ff,R=a.R)
    xall,yall,xte0,yte0=load(a.data);x012,y012=subset012(xall,yall);xte,yte=subset012(xte0,yte0);xtr,ytr,xva,yva=split(x012,y012,.1,7);cb=np.load(a.codebook);tr=encode(xtr,cb);va=encode(xva,cb);te=encode(xte,cb);counts,cw=class_weights(tr,a.variant,a.cb_beta,a.cb_cap)
    model=init_model(cfg,a.seed);tok,pos,cls=build_embeddings(cfg,a.seed+99)
    F,T=build_scaffold_contexts(model,cfg,tr,ytr,tok,pos,cls,a.scaffold_contexts,a.seed+101);t0=time.time();model['WR']=train_readout_msdl(F,T,cw,a.scaffold_paths,a.scaffold_lr,a.seed+202);print('SCAFFOLD_DONE',json.dumps({'elapsed_s':time.time()-t0,'contexts':a.scaffold_contexts,'paths':a.scaffold_paths}),flush=True)
    scaffold_eval={str(r):eval_masked(model,cfg,va,yva,tok,pos,cls,r,a.seed+300+int(100*r),a.eval_images) for r in (.25,.5,.9,1.0)};print('SCAFFOLD_EVAL',json.dumps(scaffold_eval),flush=True)
    labs=np.tile(np.arange(3),a.gen_per_class)
    from sklearn.linear_model import LogisticRegression
    clf=LogisticRegression(max_iter=250,solver='lbfgs',random_state=7).fit(xtr.reshape(len(xtr),-1),ytr)
    def gm(img):
        pred=clf.predict(img.reshape(len(img),-1));prob=clf.predict_proba(img.reshape(len(img),-1));return {'condition_match':float(np.mean(pred==labs)),'mean_confidence':float(np.mean(prob.max(1))),'pred_counts':np.bincount(pred,minlength=3).tolist()}
    stok,simg=generate_images(model,cfg,labs,tok,pos,cls,cb,a.seed+606);save_grid(simg,labs,a.out/'generation_scaffold.png');gen_scaffold=gm(simg)
    sysm=P3ImageSystem(cfg,model,a.seed+303);hist=train_full(sysm,cfg,tr,ytr,tok,pos,cls,cw,a.events,a.seed+404,a.report_every);final=model.copy();final.update(sysm.weights())
    final_eval={str(r):eval_masked(final,cfg,te,yte,tok,pos,cls,r,a.seed+500+int(100*r),a.eval_images) for r in (.25,.5,.9,1.0)}
    gtok,gimg=generate_images(final,cfg,labs,tok,pos,cls,cb,a.seed+606);save_grid(gimg,labs,a.out/'generation_final.png');gen_final=gm(gimg)
    moves={k:float(np.linalg.norm(final[k]-model[k])/(np.linalg.norm(model[k])+1e-12)) for k in ('WQ','WK','WV','WO','W1','W2')}
    rec={'BR':float(np.mean(np.abs(sysm.BR))),'BO':float(np.mean(np.abs(sysm.BO))),'B2':float(np.mean(np.abs(sysm.B2)))}
    sc50=scaffold_eval['0.5'];fn50=final_eval['0.5']
    gates={'ce50_improves_from_scaffold':fn50['ce']<sc50['ce'],'foreground_mass50_improves_from_scaffold':fn50['foreground_non_dominant_mass']>sc50['foreground_non_dominant_mass'],'generation_final_match_ge_0p60':gen_final['condition_match']>=0.60,'reciprocal_populations_alive':all(v>0 for v in rec.values())}
    diag={'variant':a.variant,'candidate_backprop':False,'torch_optimizer':False,'developed_p3_core':{'resident_Hg':True,'resident_Xphi':True,'collective_K_R':cfg.R,'persistent_EM_Izh':True,'seven_forward_groups':True,'adaptive_reciprocal':True,'qk_interface':'exact_positive_RF (P1-G LUT deferred)'},'class_conditioning_retained':True,'from_scratch_internal_core':True,'startup_msdl_readout_scaffold':True,'token_counts':counts.tolist(),'token_weights':cw.tolist(),'dominant_token':int(np.argmax(counts)),'dominant_fraction':float(counts.max()/counts.sum()),'config':vars(cfg),'scaffold_eval':scaffold_eval,'final_eval':final_eval,'generation_scaffold':gen_scaffold,'generation_final':gen_final,'events':a.events,'slow_updates':sysm.update_count,'mean_abs_reciprocal':rec,'relative_internal_weight_move':moves,'gates':gates,'pass_numeric':bool(all(gates.values())),'history':hist}
    (a.out/'metrics.json').write_text(json.dumps(diag,indent=2));np.savez_compressed(a.out/'final_state.npz',**{k:v for k,v in final.items() if isinstance(v,np.ndarray)},tok=tok,pos=pos,cls=cls,token_weights=cw);np.savetxt(a.out/'generated_tokens_final.csv',gtok,fmt='%d',delimiter=',');np.savetxt(a.out/'generated_tokens_scaffold.csv',stok,fmt='%d',delimiter=',')
    if hist:
        with (a.out/'history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(hist[0].keys()));w.writeheader();w.writerows(hist)
    print('FINAL',json.dumps(diag),flush=True)
if __name__=='__main__':main()
