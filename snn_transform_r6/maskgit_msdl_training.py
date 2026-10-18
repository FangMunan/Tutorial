#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MaskGIT-style masked-token training adapter for the exact P3/MSDL core.

The public MaskGIT training rule is represented at the task/protocol level:
  * sample r ~ Uniform[0,1)
  * gamma(r) = cos(pi*r/2)
  * uniformly mask ceil(gamma(r) * N) token positions
  * keep that corrupted token context fixed
  * predict/learn every masked target from that same context
  * do not reveal any ground-truth masked token inside the batch

P3/MSDL remains local and dynamical rather than backpropagation.  All forward
quantities for every masked receiver are cached before any forward weight update;
local E/M contributions are then accumulated and one P3 update is applied after
all masked targets in the context have contributed.  Reciprocal mirror dynamics
run once per mask-context with the same total number of microsteps that the old
per-target route would have used, and remain fixed while the masked-target batch
is accumulated.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

import p3i_cb_exact_msdl as core


@dataclass(frozen=True)
class MaskSample:
    ratio_r: float
    mask_ratio: float
    masked_positions: np.ndarray


def cosine_mask_ratio(r: float) -> float:
    """MaskGIT training schedule gamma(r)=cos(pi*r/2)."""
    r=float(r)
    if not (0.0 <= r < 1.0):
        raise ValueError(f'r must be in [0,1), got {r}')
    return float(math.cos(math.pi * 0.5 * r))


def sample_mask_set(n_tokens: int, rng: np.random.Generator) -> MaskSample:
    """Sample a MaskGIT training mask: ceil(gamma(r)*N), positions uniform."""
    if n_tokens < 1:
        raise ValueError('n_tokens must be positive')
    r=float(rng.random())
    gamma=cosine_mask_ratio(r)
    nmask=int(math.ceil(gamma * int(n_tokens)))
    nmask=max(1,min(int(n_tokens),nmask))
    pos=np.sort(rng.choice(int(n_tokens),size=nmask,replace=False).astype(np.int64))
    return MaskSample(ratio_r=r,mask_ratio=gamma,masked_positions=pos)


def masking_self_test(n_tokens: int = 49, vocab_size: int = 64,
                      trials: int = 128, seed: int = 9173) -> Dict:
    """Pure structural audit of the training-mask sampler; no neural computation."""
    rng=np.random.default_rng(seed)
    truth=(np.arange(n_tokens,dtype=np.int64) % vocab_size)
    counts=[]
    for _ in range(int(trials)):
        sm=sample_mask_set(n_tokens,rng)
        expected=int(math.ceil(sm.mask_ratio*n_tokens))
        if len(sm.masked_positions) != expected:
            raise AssertionError('MaskGIT training mask length does not equal ceil(gamma(r)*N)')
        if len(np.unique(sm.masked_positions)) != len(sm.masked_positions):
            raise AssertionError('training mask contains duplicate positions')
        arr=truth.copy(); arr[sm.masked_positions]=vocab_size
        if not np.all(arr[sm.masked_positions] == vocab_size):
            raise AssertionError('masked positions were not replaced by MASK')
        known=np.setdiff1d(np.arange(n_tokens),sm.masked_positions,assume_unique=True)
        if not np.array_equal(arr[known],truth[known]):
            raise AssertionError('unmasked ground-truth tokens were modified')
        counts.append(len(sm.masked_positions))
    return {
        'pass':True,
        'schedule':'gamma(r)=cos(pi*r/2), r~Uniform[0,1)',
        'mask_count_rule':'ceil(gamma(r)*N)',
        'uniform_position_sampling_without_replacement':True,
        'ground_truth_reveal_inside_mask_context':False,
        'trials':int(trials),
        'mask_count_min':int(np.min(counts)),
        'mask_count_max':int(np.max(counts)),
        'mask_count_mean':float(np.mean(counts)),
    }


def build_maskgit_scaffold_contexts(model, cfg, tokens, labels, tok, pos, cls,
                                    nctx: int, seed: int):
    """Build scaffold rows only from MaskGIT-style corrupted contexts.

    Every target row belonging to one sampled mask set is computed from exactly
    the same corrupted image.  No target is revealed before another target row is
    formed.  `nctx` continues to mean the number of target rows used by the
    historical startup scaffold, not the number of sampled images.
    """
    rng=np.random.default_rng(seed)
    F=np.empty((nctx,cfg.D+1),dtype=np.float64)
    T=np.empty(nctx,dtype=np.int64)
    row=0; batches=0; counts=[]
    while row<nctx:
        ii=int(rng.integers(len(tokens)))
        sm=sample_mask_set(cfg.N,rng)
        truth=np.asarray(tokens[ii],dtype=np.int64)
        arr=truth.copy(); arr[sm.masked_positions]=core.MASK
        xn,xraw=core.context_from_arr(arr,labels[ii],tok,pos,cls)
        counts.append(len(sm.masked_positions)); batches+=1
        # All rows below use the same xn/xraw; there is no progressive reveal.
        for t in sm.masked_positions:
            if row>=nctx:
                break
            F[row]=core.static_feature(model,cfg,xn,xraw[int(t)],int(t))
            T[row]=int(truth[int(t)])
            row+=1
    audit={
        'protocol':'MaskGIT-style scaffold contexts',
        'target_rows':int(nctx),
        'mask_contexts_sampled':int(batches),
        'mean_masked_targets_per_context':float(np.mean(counts)),
        'same_corrupted_context_for_all_targets':True,
        'progressive_ground_truth_reveal':False,
        'mask_schedule':'cosine with ceil(gamma(r)*N)',
    }
    return F,T,audit


class MaskGITP3System(core.P3ImageSystem):
    """Exact P3 system with a MaskGIT-style multi-target update boundary."""

    def _reciprocal_batch_microsteps(self, n_targets: int):
        """Advance reciprocal mirror populations before the frozen target batch.

        Total microsteps equal the old per-target amount, but BR/BO/B2 are then
        held fixed while all masked targets from this context contribute.
        """
        c=self.c
        nsteps=int(c.reciprocal_micro)*int(n_targets)
        core.reciprocal_microsteps(
            self.BR,self.BO,self.B2,self.WR,self.WO,self.W2,
            self.vm_br,self.um_br,self.ve_br,self.ue_br,self.zm_br,self.ze_br,
            self.vm_bo,self.um_bo,self.ve_bo,self.ue_bo,self.zm_bo,self.ze_bo,
            self.vm_b2,self.um_b2,self.ve_b2,self.ue_b2,self.zm_b2,self.ze_b2,
            nsteps,core.MIRROR_SIGMA,
            core.ETA_BR*self.recip_scale,core.ETA_BO*self.recip_scale,
            core.ETA_B2*self.recip_scale,core.MIRROR_LEAK)

    def prepare_mask_batch(self, xn: np.ndarray, xraw: np.ndarray,
                           masked_positions: np.ndarray) -> List[Dict]:
        """Cache all masked-site forward quantities before any forward update."""
        if self.accum != 0:
            raise AssertionError('P3 local accumulators must be empty at mask-batch start')
        self.rebuild(np.asarray(xn,dtype=np.float64))
        cache=[]
        for t0 in np.asarray(masked_positions,dtype=np.int64):
            t=int(t0)
            a,q,num,den=self._read(t)
            u,h,rf,p=self._tail(a,np.asarray(xraw[t],dtype=np.float64))
            cache.append({
                't':t,
                'a':np.asarray(a,dtype=np.float64).copy(),
                'q':np.asarray(q,dtype=np.float64).copy(),
                'num':np.asarray(num,dtype=np.float64).copy(),
                'den':np.asarray(den,dtype=np.float64).copy(),
                'u':np.asarray(u,dtype=np.float64).copy(),
                'h':np.asarray(h,dtype=np.float64).copy(),
                'rf':np.asarray(rf,dtype=np.float64).copy(),
                'p':np.asarray(p,dtype=np.float64).copy(),
            })
        return cache

    def accumulate_cached_target(self, rec: Dict, target: int, weight: float = 1.0):
        """Accumulate one local MSDL contribution without changing forward weights."""
        c=self.c
        t=int(rec['t']);a=rec['a'];q=rec['q'];num=rec['num'];den=rec['den']
        u=rec['u'];h=rec['h'];rf=rec['rf'];p=rec['p']
        one=np.zeros(c.V,dtype=np.float64);one[int(target)]=1.0
        ww=float(weight);err=ww*(one-p)
        self.obs_nll-=math.log(max(float(p[int(target)]),1e-12));self.obs_count+=1

        ER,_=core.fast_izh_step(self.ver,self.uer,rf,self.zer,core.G_E,core.CV_E,core.CU_E)
        MR,_=core.fast_izh_step(self.vmr,self.umr,ww*one,ww*p,core.G_M,core.CV_M,core.CU_M)
        self.ar+=np.outer(MR,ER)

        dy=self.BR@err;dh=self.B2@dy;da=self.BO@dy
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

        EK,_=core.fast_izh_step(self.vek,self.uek,self.xgrp.reshape(-1),self.zek,core.G_E,core.CV_E,core.CU_E)
        kd=self._kdrive(q,num,den,da)
        MK,_=core.fast_izh_step(self.vmk,self.umk,kd.reshape(-1),self.zmk,core.G_M,core.CV_M,core.CU_M)
        self.ak+=(MK.reshape(c.R,c.D).T@EK.reshape(c.R,c.D))/c.R

        qd=self._qdrive(q,da)
        EQ,_=core.fast_izh_step(self.veq,self.ueq,self.xcache[t],self.zeq,core.G_E,core.CV_E,core.CU_E)
        MQ,_=core.fast_izh_step(self.vmq,self.umq,qd,self.zmq,core.G_M,core.CV_M,core.CU_M)
        self.aq+=np.outer(MQ,EQ)

        self.event_count+=1
        self.accum+=1
        return np.asarray(p,dtype=np.float64)

    def learn_mask_batch(self, xn: np.ndarray, xraw: np.ndarray,
                         masked_positions: np.ndarray, truth_tokens: np.ndarray,
                         weights: np.ndarray, rng: np.random.Generator) -> Dict:
        """One MaskGIT corrupted context -> all masked local contributions -> one update."""
        masked=np.asarray(masked_positions,dtype=np.int64)
        if len(masked)<1:
            raise ValueError('mask batch must contain at least one target')
        truth=np.asarray(truth_tokens,dtype=np.int64)

        # Reciprocal learning is completed before target predictions are cached;
        # BR/BO/B2 and all forward weights are fixed across the target batch.
        self._reciprocal_batch_microsteps(len(masked))
        cache=self.prepare_mask_batch(xn,xraw,masked)

        # MSDL E/M populations are persistent dynamical states, so contribution
        # accumulation has a temporal order. Randomize it to avoid a fixed spatial
        # bias; importantly, the context and cached forward predictions never change.
        order=rng.permutation(len(cache))
        nll=[]
        for ci in order:
            rec=cache[int(ci)]
            target=int(truth[rec['t']])
            p=self.accumulate_cached_target(rec,target,float(weights[target]))
            nll.append(-math.log(max(float(p[target]),1e-12)))

        if self.accum != len(masked):
            raise AssertionError('not every masked target contributed exactly once')
        # P3 apply_update averages accumulated local contributions by self.accum.
        # No target can alter forward weights before every masked target has contributed.
        self.apply_update()
        return {
            'masked_targets':int(len(masked)),
            'mean_nll':float(np.mean(nll)),
            'same_corrupted_context':True,
            'ground_truth_reveal_inside_batch':False,
            'forward_weights_fixed_until_all_targets_contribute':True,
            'one_forward_update_after_mask_batch':True,
        }


def train_maskgit_msdl(system: MaskGITP3System, cfg, tr, ytr, tok, pos, cls,
                       weights: np.ndarray, target_events: int, seed: int,
                       report_every: int = 5000) -> Tuple[List[Dict],Dict]:
    """Train P3/MSDL with the MaskGIT multi-mask objective protocol.

    `target_events` preserves the historical compute-budget unit: number of masked
    target contributions.  A full mask context is never truncated merely to hit
    that number exactly; the final batch may therefore overshoot slightly.
    """
    rng=np.random.default_rng(seed)
    done=0;batches=0;hist=[];t0=time.time();next_report=int(report_every)
    mask_counts=[];sampled_r=[]

    while done<int(target_events):
        ii=int(rng.integers(len(tr)))
        sm=sample_mask_set(cfg.N,rng)
        truth=np.asarray(tr[ii],dtype=np.int64)
        arr=truth.copy();arr[sm.masked_positions]=core.MASK
        xn,xraw=core.context_from_arr(arr,ytr[ii],tok,pos,cls)

        rec=system.learn_mask_batch(xn,xraw,sm.masked_positions,truth,weights,rng)
        nmask=int(rec['masked_targets']);done+=nmask;batches+=1
        mask_counts.append(nmask);sampled_r.append(sm.ratio_r)

        if done>=next_report:
            row={
                'target_events':int(done),
                'mask_batches':int(batches),
                'online_nll':system.obs_nll/max(1,system.obs_count),
                'updates':int(system.update_count),
                'mean_masked_targets_per_batch':float(np.mean(mask_counts)),
                'last_masked_targets':nmask,
                'elapsed_s':time.time()-t0,
                'mean_abs_BR':float(np.mean(np.abs(system.BR))),
                'mean_abs_BO':float(np.mean(np.abs(system.BO))),
                'mean_abs_B2':float(np.mean(np.abs(system.B2))),
            }
            hist.append(row);print('TRAIN',json.dumps(row),flush=True)
            while next_report<=done:
                next_report+=int(report_every)

    audit={
        'pass':True,
        'requested_target_events':int(target_events),
        'actual_target_events':int(done),
        'mask_batches':int(batches),
        'mean_masked_targets_per_batch':float(np.mean(mask_counts)),
        'min_masked_targets':int(np.min(mask_counts)),
        'max_masked_targets':int(np.max(mask_counts)),
        'mean_sampled_r':float(np.mean(sampled_r)),
        'schedule':'r~Uniform[0,1), gamma(r)=cos(pi*r/2), nmask=ceil(gamma(r)*N)',
        'uniform_mask_positions':True,
        'all_masked_targets_learned_from_same_corrupted_context':True,
        'ground_truth_reveal_inside_batch':False,
        'forward_predictions_cached_before_weight_update':True,
        'forward_weights_fixed_across_masked_targets':True,
        'one_forward_weight_update_per_mask_context':True,
        'local_EM_state':'persistent and sequentially advanced across masked-target contributions in randomized order',
        'backpropagation':False,
    }
    return hist,audit
