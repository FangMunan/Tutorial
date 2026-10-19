#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R6B.4: complete forward-parameter plasticity for the label-free MaskGIT/P3 route.

R6B.3 already trains WQ/WK/WV/WO/W1/W2/WR with persistent local MSDL.
This module restores plasticity for the forward parameters that R6B.3 left fixed:

  * visual token embeddings (initialized from frozen SNN-VQ geometry),
  * positional embeddings (initialized from deterministic 2-D sinusoids),
  * bq/bk/bv/bo/b1/b2,
  * gamma2/beta2,
  * theta.

The MaskGIT objective is unchanged.  No label, autograd, BPTT, or conventional
optimizer is introduced.  Missing credit coverage is supplied by extending the
same adaptive reciprocal-mirror mechanism already used by P3 to the local
interfaces that were previously frozen.  Those reciprocal matrices are learning
machinery, not forward inference parameters.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'snn_transform_p3_image'))
sys.path.insert(0, str(HERE))

import p3i_cb_exact_msdl as core
from maskgit_msdl_training import MaskGITP3System


FORWARD_PARAMETER_KEYS = (
    'tok','pos',
    'WQ','WK','WV','WO','W1','W2','WR',
    'bq','bk','bv','bo','b1','b2',
    'gamma2','beta2','theta',
)

EXTRA_RECIPROCAL_KEYS = ('BU1','BXQ','BXK','BXV','BXN')


def _row_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x=np.asarray(x,dtype=np.float64)
    mu=float(np.mean(x))
    var=float(np.mean((x-mu)**2))
    return (x-mu)/math.sqrt(var+float(eps))


def _copy_array(x):
    return np.array(x,copy=True,dtype=np.float64)


class FullForwardPlasticMaskGITSystem(MaskGITP3System):
    """R6B.3 P3/MSDL system with all ordinary forward parameters plastic."""

    def __init__(self, cfg, model, seed: int, tok: np.ndarray, pos: np.ndarray,
                 aux_lr_scale: float = 0.10, extra_reciprocal_scale: float = 1.0):
        super().__init__(cfg,model,seed)
        self._init_full_forward_plasticity(
            tok,pos,seed=seed+700001,
            aux_lr_scale=aux_lr_scale,
            extra_reciprocal_scale=extra_reciprocal_scale,
        )

    def _init_full_forward_plasticity(self, tok, pos, *, seed: int,
                                      aux_lr_scale: float,
                                      extra_reciprocal_scale: float):
        c=self.c
        tok=np.asarray(tok,dtype=np.float64)
        pos=np.asarray(pos,dtype=np.float64)
        if tok.shape != (c.V+1,c.D):
            raise ValueError(f'token embedding shape {tok.shape}, expected {(c.V+1,c.D)}')
        if pos.shape != (c.N,c.D):
            raise ValueError(f'position embedding shape {pos.shape}, expected {(c.N,c.D)}')
        if aux_lr_scale <= 0:
            raise ValueError('aux_lr_scale must be positive')
        if extra_reciprocal_scale <= 0:
            raise ValueError('extra_reciprocal_scale must be positive')

        self.tok=_copy_array(tok)
        self.pos=_copy_array(pos)
        self.full_forward_plasticity_version='R6B4_FULL_FORWARD_PLASTICITY_V1'
        self.aux_lr_scale=float(aux_lr_scale)
        self.extra_reciprocal_scale=float(extra_reciprocal_scale)
        self.aux_rng=np.random.default_rng(int(seed))

        D=c.D; ff=c.FF
        # Same orientation as the existing P3 reciprocal matrices: reciprocal
        # maps local post modulation back toward the preceding forward variable.
        self.BU1=np.zeros((D,ff),dtype=np.float64)  # h -> u, mirrors W1
        self.BXQ=np.zeros((D,D),dtype=np.float64)   # q -> xn, mirrors WQ
        self.BXK=np.zeros((D,D),dtype=np.float64)   # k -> xn, mirrors WK
        self.BXV=np.zeros((D,D),dtype=np.float64)   # v -> xn, mirrors WV
        self.BXN=np.zeros((D,D),dtype=np.float64)   # xn -> xraw, local norm mirror

        def pop(n):
            return np.full(n,core.V0),np.full(n,core.U0),np.zeros(n,dtype=np.float64)

        # Extra reciprocal mirror populations.
        for name,nin,nout in (
            ('bu1',D,ff),('bxq',D,D),('bxk',D,D),('bxv',D,D),('bxn',D,D),
        ):
            vm,um,zm=pop(nin); ve,ue,ze=pop(nout)
            setattr(self,'vm_'+name,vm);setattr(self,'um_'+name,um);setattr(self,'zm_'+name,zm)
            setattr(self,'ve_'+name,ve);setattr(self,'ue_'+name,ue);setattr(self,'ze_'+name,ze)

        # Local E/M states for the newly plastic normalization/affine stages.
        self.vmn,self.umn,self.zmn=pop(D)
        self.veg,self.ueg,self.zeg=pop(D)
        self.veconst,self.ueconst,self.zeconst=pop(1)

        # Embedding-output M is advanced once per MaskGIT context after the
        # target-wise local drives have been accumulated.  This avoids O(N)
        # Izh populations being stepped once for every masked receiver.
        self.vmemb,self.umemb,self.zmemb=pop(c.N*D)

        self._mirror_xraw=None
        self._current_arr=None
        self._current_xraw=None
        self._current_xn=None
        self._clear_aux_accumulators()

    def context_from_arr(self, arr: np.ndarray):
        arr=np.asarray(arr,dtype=np.int64).reshape(-1)
        if len(arr) != self.c.N:
            raise ValueError('token length mismatch')
        if np.any(arr<0) or np.any(arr>=len(self.tok)):
            raise ValueError('token id outside embedding table')
        x=self.tok[arr]+self.pos
        mu=x.mean(1,keepdims=True)
        var=np.mean((x-mu)**2,1,keepdims=True)
        xn=(x-mu)/np.sqrt(var+self.eps2)
        return xn,x

    def _clear_aux_accumulators(self):
        self.abq=np.zeros_like(self.bq)
        self.abk=np.zeros_like(self.bk)
        self.abv=np.zeros_like(self.bv)
        self.abo=np.zeros_like(self.bo)
        self.ab1=np.zeros_like(self.b1)
        self.ab2=np.zeros_like(self.b2)
        self.agamma2=np.zeros_like(self.gamma2)
        self.abeta2=np.zeros_like(self.beta2)
        self.atheta=np.zeros_like(self.theta)
        self.aembed_drive=np.zeros((self.c.N,self.c.D),dtype=np.float64)
        self.aconst=0.0

    def _mirror_linear_step(self, B: np.ndarray, W: np.ndarray, prefix: str, eta: float):
        nin=W.shape[1]
        dx=self.aux_rng.normal(0.0,core.MIRROR_SIGMA,nin)
        dy=W@dx
        vm=getattr(self,'vm_'+prefix); um=getattr(self,'um_'+prefix); zm=getattr(self,'zm_'+prefix)
        ve=getattr(self,'ve_'+prefix); ue=getattr(self,'ue_'+prefix); ze=getattr(self,'ze_'+prefix)
        M,_=core.fast_izh_step(vm,um,dx,zm,core.G_M,core.CV_M,core.CU_M)
        E,_=core.fast_izh_step(ve,ue,dy,ze,core.G_E,core.CV_E,core.CU_E)
        B *= (1.0-core.MIRROR_LEAK)
        B += float(eta)*np.outer(M,E)

    def _mirror_input_norm_step(self, eta: float):
        if self._mirror_xraw is None:
            return
        n=int(self.aux_rng.integers(self.c.N))
        x=np.asarray(self._mirror_xraw[n],dtype=np.float64)
        dx=self.aux_rng.normal(0.0,core.MIRROR_SIGMA,self.c.D)
        # Local response of the same per-token normalization used by the forward path.
        dy=_row_norm(x+dx,self.eps2)-_row_norm(x,self.eps2)
        M,_=core.fast_izh_step(self.vm_bxn,self.um_bxn,dx,self.zm_bxn,
                               core.G_M,core.CV_M,core.CU_M)
        E,_=core.fast_izh_step(self.ve_bxn,self.ue_bxn,dy,self.ze_bxn,
                               core.G_E,core.CV_E,core.CU_E)
        self.BXN *= (1.0-core.MIRROR_LEAK)
        self.BXN += float(eta)*np.outer(M,E)

    def _extra_reciprocal_microsteps(self, nsteps: int):
        eta=float(core.ETA_BO*self.recip_scale*self.extra_reciprocal_scale)
        for _ in range(int(nsteps)):
            self._mirror_linear_step(self.BU1,self.W1,'bu1',eta)
            self._mirror_linear_step(self.BXQ,self.WQ,'bxq',eta)
            self._mirror_linear_step(self.BXK,self.WK,'bxk',eta)
            self._mirror_linear_step(self.BXV,self.WV,'bxv',eta)
            self._mirror_input_norm_step(eta)

    def _reciprocal_batch_microsteps(self, n_targets: int):
        # Preserve all existing R6B.3 reciprocal dynamics first.
        super()._reciprocal_batch_microsteps(n_targets)
        self._extra_reciprocal_microsteps(self.c.reciprocal_micro*int(n_targets))

    def prepare_mask_batch(self, xn: np.ndarray, xraw: np.ndarray,
                           masked_positions: np.ndarray):
        if self.accum != 0:
            raise AssertionError('P3 local accumulators must be empty at mask-batch start')
        self._current_xn=np.asarray(xn,dtype=np.float64).copy()
        self._current_xraw=np.asarray(xraw,dtype=np.float64).copy()
        self.rebuild(self._current_xn)
        cache=[]
        for t0 in np.asarray(masked_positions,dtype=np.int64):
            t=int(t0)
            a,q,num,den=self._read(t)
            z=self._current_xraw[t]+self.bo+self.WO@a
            mu=float(np.mean(z)); inv=1.0/math.sqrt(float(np.mean((z-mu)**2))+self.eps2)
            zhat=(z-mu)*inv
            u=zhat*self.gamma2+self.beta2
            h=np.maximum(0.0,self.W1@u+self.b1-self.theta)
            y=z+self.W2@h+self.b2
            rms=math.sqrt(float(np.mean(y*y)))+1e-8
            rf=np.empty(self.c.D+1,dtype=np.float64);rf[:-1]=0.5*y/rms;rf[-1]=0.5
            p=core.softmax_numba(self.WR@rf)
            cache.append({
                't':t,'a':a.copy(),'q':q.copy(),'num':num.copy(),'den':den.copy(),
                'z':z.copy(),'zhat':zhat.copy(),'u':u.copy(),'h':h.copy(),
                'rf':rf.copy(),'p':np.asarray(p,dtype=np.float64).copy(),
            })
        return cache

    def _value_drives_by_position(self, t: int, den: np.ndarray, da: np.ndarray):
        c=self.c; hd=c.D//c.H
        dv=np.zeros((c.N,c.D),dtype=np.float64)
        for h in range(c.H):
            pq=self.pq[h,t]
            scores=self.pk[h]@pq
            alpha=scores/max(1e-8,float(den[h]))
            sl=slice(h*hd,(h+1)*hd)
            dv[:,sl]=alpha[:,None]*da[sl][None,:]
        return dv

    def accumulate_cached_target(self, rec: Dict, target: int, weight: float = 1.0):
        c=self.c
        if self._current_arr is None:
            raise AssertionError('full-plastic system requires learn_mask_batch_arr()')
        t=int(rec['t']);a=rec['a'];q=rec['q'];num=rec['num'];den=rec['den']
        u=rec['u'];h=rec['h'];rf=rec['rf'];p=rec['p'];zhat=rec['zhat']
        one=np.zeros(c.V,dtype=np.float64);one[int(target)]=1.0
        ww=float(weight);err=ww*(one-p)
        self.obs_nll-=math.log(max(float(p[int(target)]),1e-12));self.obs_count+=1

        # Existing seven forward groups: identical local MSDL mechanics to R6B.3.
        ER,_=core.fast_izh_step(self.ver,self.uer,rf,self.zer,core.G_E,core.CV_E,core.CU_E)
        MR,_=core.fast_izh_step(self.vmr,self.umr,ww*one,ww*p,core.G_M,core.CV_M,core.CU_M)
        self.ar+=np.outer(MR,ER)

        dy=self.BR@err
        dh=self.B2@dy
        da=self.BO@dy
        du=self.BU1@dh

        E2,_=core.fast_izh_step(self.ve2,self.ue2,h,self.ze2,core.G_E,core.CV_E,core.CU_E)
        M2,_=core.fast_izh_step(self.vm2,self.um2,dy,self.zm2,core.G_M,core.CV_M,core.CU_M)
        self.a2+=np.outer(M2,E2)*self.mask2

        E1,_=core.fast_izh_step(self.ve1,self.ue1,u,self.ze1,core.G_E,core.CV_E,core.CU_E)
        M1,_=core.fast_izh_step(self.vm1,self.um1,dh,self.zm1,core.G_M,core.CV_M,core.CU_M)
        self.a1+=np.outer(M1,E1)*self.mask1

        EO,_=core.fast_izh_step(self.veo,self.ueo,a,self.zeo,core.G_E,core.CV_E,core.CU_E)
        MO,_=core.fast_izh_step(self.vmo,self.umo,dy,self.zmo,core.G_M,core.CV_M,core.CU_M)
        self.ao+=np.outer(MO,EO)*self.maskO

        sv=self._vinput(t,den)
        EV,_=core.fast_izh_step(self.vev,self.uev,sv,self.zev,core.G_E,core.CV_E,core.CU_E)
        MV,_=core.fast_izh_step(self.vmv,self.umv,da,self.zmv,core.G_M,core.CV_M,core.CU_M)
        hd=c.D//c.H
        for rr in range(c.D):
            self.av[rr]+=MV[rr]*EV[(rr//hd)*c.D:(rr//hd+1)*c.D]

        EK,_=core.fast_izh_step(self.vek,self.uek,self.xgrp.reshape(-1),self.zek,
                                core.G_E,core.CV_E,core.CU_E)
        kd=self._kdrive(q,num,den,da)
        MK,_=core.fast_izh_step(self.vmk,self.umk,kd.reshape(-1),self.zmk,
                                core.G_M,core.CV_M,core.CU_M)
        self.ak+=(MK.reshape(c.R,c.D).T@EK.reshape(c.R,c.D))/c.R

        qd=self._qdrive(q,da)
        EQ,_=core.fast_izh_step(self.veq,self.ueq,self.xcache[t],self.zeq,
                                core.G_E,core.CV_E,core.CU_E)
        MQ,_=core.fast_izh_step(self.vmq,self.umq,qd,self.zmq,
                                core.G_M,core.CV_M,core.CU_M)
        self.aq+=np.outer(MQ,EQ)

        # Constant presynaptic eligibility is shared by all affine biases.
        Eb,_=core.fast_izh_step(self.veconst,self.ueconst,np.ones(1),self.zeconst,
                                core.G_E,core.CV_E,core.CU_E)
        eb=float(Eb[0]); self.aconst+=eb

        # Restore all affine biases.  bk is shared across collective K groups.
        self.abq+=MQ*eb
        self.abk+=np.mean(MK.reshape(c.R,c.D),axis=0)*eb
        self.abv+=MV*eb
        self.abo+=MO*eb
        self.ab1+=M1*eb
        self.ab2+=M2*eb
        self.atheta+=(-M1)*eb

        # Restore normalization affine gamma/beta.  BU1 supplies the local M drive
        # at the normalization output u, using the same reciprocal-mirror principle.
        MN,_=core.fast_izh_step(self.vmn,self.umn,du,self.zmn,core.G_M,core.CV_M,core.CU_M)
        EG,_=core.fast_izh_step(self.veg,self.ueg,zhat,self.zeg,core.G_E,core.CV_E,core.CU_E)
        self.agamma2+=MN*EG
        self.abeta2+=MN*eb

        # Restore token/position embedding plasticity.  Build local modulation at
        # every raw input position from receiver residual credit plus Q/K/V credit.
        mxn=np.zeros((c.N,c.D),dtype=np.float64)
        mxn[t]+=self.BXQ@qd
        norm=math.sqrt(c.N//c.R)
        for r in range(c.R):
            drive=(self.BXK@kd[r])/norm
            mxn[self.groups==r]+=drive[None,:]
        dv=self._value_drives_by_position(t,den,da)
        mxn+=dv@self.BXV.T

        # Q/K/V consume normalized inputs.  BXN locally mirrors that normalization
        # back to the raw embedding coordinates; the residual path is already raw.
        mxraw=mxn@self.BXN.T
        mxraw[t]+=dy
        self.aembed_drive+=mxraw

        self.event_count+=1
        self.accum+=1
        return np.asarray(p,dtype=np.float64)

    def apply_update(self):
        inv=1.0/max(1,self.accum)
        # Existing R6B.3 forward groups.
        self.WR+=self.lrr*self.ar*inv
        self.WQ+=self.lrq*self.aq*inv
        self.WK+=self.lrk*self.ak*inv
        self.WV+=self.lrv*self.av*inv
        self.WO+=self.lro*self.ao*inv; self.WO*=self.maskO
        self.W1+=self.lr1*self.a1*inv; self.W1*=self.mask1
        self.W2+=self.lr2*self.a2*inv; self.W2*=self.mask2

        s=self.aux_lr_scale
        # Biases/threshold use their parent layer learning-rate scale.
        self.bq+=s*self.lrq*self.abq*inv
        self.bk+=s*self.lrk*self.abk*inv
        self.bv+=s*self.lrv*self.abv*inv
        self.bo+=s*self.lro*self.abo*inv
        self.b1+=s*self.lr1*self.ab1*inv
        self.b2+=s*self.lr2*self.ab2*inv
        self.theta+=s*self.lr1*self.atheta*inv
        self.gamma2+=s*self.lr1*self.agamma2*inv
        self.beta2+=s*self.lr1*self.abeta2*inv

        # One context-level persistent E/M step for embedding outputs, after all
        # masked receivers have supplied local modulation under the same context.
        mean_drive=self.aembed_drive*inv
        ME,_=core.fast_izh_step(
            self.vmemb,self.umemb,mean_drive.reshape(-1),self.zmemb,
            core.G_M,core.CV_M,core.CU_M)
        ME=ME.reshape(self.c.N,self.c.D)
        eb=self.aconst*inv
        lr_emb=s*min(self.lrq,self.lrk,self.lrv)
        lr_pos=s*self.lrq
        for n,zid in enumerate(np.asarray(self._current_arr,dtype=np.int64)):
            self.tok[int(zid)]+=lr_emb*ME[n]*eb
            self.pos[n]+=lr_pos*ME[n]*eb

        self.update_count+=1
        # Clear base accumulators without invoking a second update.
        self.aq.fill(0);self.ak.fill(0);self.av.fill(0);self.ao.fill(0)
        self.a1.fill(0);self.a2.fill(0);self.ar.fill(0);self.accum=0
        self._clear_aux_accumulators()

    def learn_mask_batch_arr(self, arr: np.ndarray, masked_positions: np.ndarray,
                             truth_tokens: np.ndarray, weights: np.ndarray,
                             rng: np.random.Generator):
        arr=np.asarray(arr,dtype=np.int64).copy()
        self._current_arr=arr
        xn,xraw=self.context_from_arr(arr)
        self._mirror_xraw=xraw.copy()
        try:
            return super().learn_mask_batch(
                xn,xraw,np.asarray(masked_positions,dtype=np.int64),
                np.asarray(truth_tokens,dtype=np.int64),weights,rng)
        finally:
            self._mirror_xraw=None

    def forward_parameter_snapshot(self):
        return {k:_copy_array(getattr(self,k)) for k in FORWARD_PARAMETER_KEYS}

    def full_weights(self):
        out=self.weights()
        for k in FORWARD_PARAMETER_KEYS:
            if k not in out:
                out[k]=_copy_array(getattr(self,k))
        for k in EXTRA_RECIPROCAL_KEYS:
            out[k]=_copy_array(getattr(self,k))
        return out


def upgrade_r6b3_system(system, tok: np.ndarray, pos: np.ndarray, *, seed: int,
                         aux_lr_scale: float = 0.10,
                         extra_reciprocal_scale: float = 1.0):
    """In-place class upgrade preserving every R6B.3 dynamical state exactly."""
    if isinstance(system,FullForwardPlasticMaskGITSystem):
        return system
    if not isinstance(system,MaskGITP3System):
        raise TypeError(f'expected MaskGITP3System, got {type(system)}')
    if getattr(system,'accum',0) != 0:
        raise ValueError('upgrade requires a clean context boundary (accum==0)')
    system.__class__=FullForwardPlasticMaskGITSystem
    system._init_full_forward_plasticity(
        tok,pos,seed=int(seed),aux_lr_scale=float(aux_lr_scale),
        extra_reciprocal_scale=float(extra_reciprocal_scale))
    return system


def movement_report(before: Dict[str,np.ndarray], system,
                    atol: float = 1e-12) -> Dict:
    rows={}; all_move=True; all_finite=True
    for k in FORWARD_PARAMETER_KEYS:
        a=np.asarray(before[k],dtype=np.float64)
        b=np.asarray(getattr(system,k),dtype=np.float64)
        d=b-a
        mx=float(np.max(np.abs(d))) if d.size else 0.0
        absn=float(np.linalg.norm(d))
        rel=float(absn/(np.linalg.norm(a)+1e-12))
        finite=bool(np.all(np.isfinite(b)))
        moved=bool(mx>float(atol))
        rows[k]={'moved':moved,'finite':finite,'max_abs_change':mx,
                 'delta_norm':absn,'relative_change':rel,'shape':list(b.shape)}
        all_move=all_move and moved; all_finite=all_finite and finite

    tok0=np.asarray(before['tok']);tok1=np.asarray(system.tok)
    visual=np.max(np.abs(tok1[:system.c.V]-tok0[:system.c.V]),axis=1)
    pos0=np.asarray(before['pos']);pos1=np.asarray(system.pos)
    prow=np.max(np.abs(pos1-pos0),axis=1)
    return {
        'all_forward_groups_move':bool(all_move),
        'all_forward_groups_finite':bool(all_finite),
        'visual_token_rows_moved':int(np.sum(visual>atol)),
        'visual_token_rows_total':int(system.c.V),
        'mask_token_row_moved':bool(np.max(np.abs(tok1[system.c.V]-tok0[system.c.V]))>atol),
        'position_rows_moved':int(np.sum(prow>atol)),
        'position_rows_total':int(system.c.N),
        'groups':rows,
    }
