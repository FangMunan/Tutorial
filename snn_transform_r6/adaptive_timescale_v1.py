#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adaptive Timescale v1 for the current SNN_Transform MaskGIT learner.

Scientific scope
----------------
This file makes exactly one learning-mechanism change relative to the active
R6B4 full-forward plastic system: trainable forward-parameter groups are given
causal, adaptive plasticity timescales.  The MaskGIT objective, forward model,
MSDL E/M credit signals, reciprocal BR/BO/B2 dynamics, frozen SNN-VQ tokenizer,
and scheduled-parallel generation procedure are not redefined.

The original local update direction for group g is preserved:

    DeltaTheta_g(original) = eta_g * F_g(E, M)

Adaptive Timescale v1 changes only its magnitude:

    DeltaTheta_g = DeltaTheta_g(original) / tau_g(t)

The three initial groups are:

    FAST   = WR, WQ, WK, WV
    MEDIUM = WO, W1, W2
    SLOW   = bq,bk,bv,bo,b1,b2,gamma2,beta2,tok,pos

Initial timescales are a prior, not a permanent classification:

    tau_fast : tau_medium : tau_slow = 1 : 5 : 25

Feedback is strictly causal.  Context t is predicted and learned using tau(t).
Only after that context has completed is its already-observed mean masked-token
NLL supplied to the controller, producing tau(t+1).  No BP, BPTT, autograd,
future-loss gradient, or retroactive rewrite is used.

v1 controller (experimental hypothesis, not claimed as a published law):
  * exponentially smooth the observed context NLL;
  * at fixed feedback intervals compare current EMA with an earlier EMA;
  * if loss improved, modestly speed the currently active timescale;
  * if loss worsened, modestly slow it;
  * after sustained plateau, hand plasticity focus from FAST -> MEDIUM -> SLOW;
  * all tau values are bounded and remain positive.

Primary endpoint for this first run is functional: can the learner acquire
MNIST-like autonomous image generation?  Detailed mechanism analysis is deferred
until after the generation result is available.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'snn_transform_p3_image'))
sys.path.insert(0, str(HERE))

import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
import r6b3_context_epoch_training as trainer
from maskgit_input_representation import build_input_embeddings, representation_self_test
from maskgit_msdl_training import sample_mask_set
from maskgit_msdl_training_unconditional import (
    build_maskgit_scaffold_contexts_unconditional,
    eval_masked_unconditional,
    masking_self_test,
)
from maskgit_scheduled_parallel import controller_self_test as maskgit_controller_self_test
from r6b3_snn_encoder_maskgit_msdl import (
    generation_tokens_maskgit,
    posthoc_generation_metric,
    save_grid_unlabeled,
)
from r6b4_full_forward_plasticity import (
    FORWARD_PARAMETER_KEYS,
    FullForwardPlasticMaskGITSystem,
)

FORMAT = 'SNN_TRANSFORM_ADAPTIVE_TIMESCALE_V1'
GROUP_ORDER = ('fast', 'medium', 'slow')
GROUP_PARAMETERS = {
    'fast': ('WR', 'WQ', 'WK', 'WV'),
    'medium': ('WO', 'W1', 'W2'),
    'slow': ('bq','bk','bv','bo','b1','b2','gamma2','beta2','tok','pos'),
}


@dataclass
class TimescaleConfig:
    tau_fast: float = 1.0
    tau_medium: float = 5.0
    tau_slow: float = 25.0
    tau_min: float = 0.25
    tau_max: float = 100.0

    # Loss feedback is intentionally slow relative to one MaskGIT context.
    ema_beta: float = 0.05
    feedback_interval: int = 32
    trend_window: int = 128
    relative_trend_deadband: float = 0.005
    feedback_log_step: float = 0.08
    warmup_contexts: int = 128

    # A plateau is a sequence of feedback observations whose smoothed loss
    # changed by less than the deadband over the trend window.
    plateau_patience: int = 3
    retire_factor: float = 4.0
    release_ratio: float = 1.25

    adaptive: bool = True

    def validate(self):
        vals=(self.tau_fast,self.tau_medium,self.tau_slow,self.tau_min,self.tau_max)
        if not all(np.isfinite(v) and v>0 for v in vals):
            raise ValueError('all timescales and bounds must be finite and positive')
        if self.tau_min >= self.tau_max:
            raise ValueError('tau_min must be smaller than tau_max')
        if self.feedback_interval < 1 or self.trend_window < 1:
            raise ValueError('feedback_interval and trend_window must be positive')
        if self.plateau_patience < 1 or self.warmup_contexts < 0:
            raise ValueError('invalid plateau/warmup setting')
        if not (0.0 < self.ema_beta <= 1.0):
            raise ValueError('ema_beta must be in (0,1]')
        if self.relative_trend_deadband <= 0 or self.feedback_log_step <= 0:
            raise ValueError('deadband and feedback_log_step must be positive')
        if self.retire_factor <= 1.0 or self.release_ratio <= 0:
            raise ValueError('handover factors are invalid')
        return self


class CausalTimescaleController:
    """Forward-only performance feedback controlling future plasticity speed."""

    def __init__(self, cfg: TimescaleConfig):
        self.cfg=copy.deepcopy(cfg).validate()
        self.log_tau={
            'fast':math.log(self.cfg.tau_fast),
            'medium':math.log(self.cfg.tau_medium),
            'slow':math.log(self.cfg.tau_slow),
        }
        self.context_count=0
        self.ema_loss=None
        self.ema_history=[]
        self.focus_index=0
        self.plateau_count=0
        self.feedback_count=0
        self.handover_count=0
        self.history=[]

    @property
    def focus(self):
        return GROUP_ORDER[self.focus_index]

    def _clip_log_tau(self, x: float) -> float:
        return float(np.clip(x, math.log(self.cfg.tau_min), math.log(self.cfg.tau_max)))

    def tau(self, group: str) -> float:
        return float(math.exp(self.log_tau[group]))

    def snapshot(self) -> Dict[str,float]:
        return {g:self.tau(g) for g in GROUP_ORDER}

    def scale(self, group: str) -> float:
        return 1.0/self.tau(group)

    def _handover(self):
        if self.focus_index >= len(GROUP_ORDER)-1:
            return None
        old=GROUP_ORDER[self.focus_index]
        new=GROUP_ORDER[self.focus_index+1]
        old_before=self.tau(old)
        new_before=self.tau(new)

        # Retire the previous coarse group and release the next one smoothly in
        # timescale space.  The new group cannot become faster than tau_min.
        old_after=np.clip(old_before*self.cfg.retire_factor,
                          self.cfg.tau_min,self.cfg.tau_max)
        new_target=max(self.cfg.tau_min,old_before*self.cfg.release_ratio)
        new_after=np.clip(min(new_before,new_target),
                          self.cfg.tau_min,self.cfg.tau_max)
        self.log_tau[old]=math.log(float(old_after))
        self.log_tau[new]=math.log(float(new_after))
        self.focus_index+=1
        self.handover_count+=1
        self.plateau_count=0
        return {
            'from':old,'to':new,
            'old_tau_before':float(old_before),'old_tau_after':float(old_after),
            'new_tau_before':float(new_before),'new_tau_after':float(new_after),
        }

    def observe(self, loss: float) -> Dict:
        """Observe L_t after context t; update tau for context t+1 only."""
        loss=float(loss)
        if not np.isfinite(loss):
            raise FloatingPointError(f'non-finite feedback loss: {loss}')
        self.context_count+=1
        tau_before=self.snapshot()

        if self.ema_loss is None:
            self.ema_loss=loss
        else:
            b=self.cfg.ema_beta
            self.ema_loss=(1.0-b)*self.ema_loss+b*loss
        self.ema_history.append(float(self.ema_loss))

        event='observe_only'
        rel_trend=None
        handover=None
        active=self.focus

        due=(self.context_count >= self.cfg.warmup_contexts and
             self.context_count % self.cfg.feedback_interval == 0 and
             len(self.ema_history) > self.cfg.trend_window)

        if self.cfg.adaptive and due:
            old=float(self.ema_history[-1-self.cfg.trend_window])
            now=float(self.ema_history[-1])
            rel_trend=(now-old)/max(abs(old),1e-12)
            d=self.cfg.relative_trend_deadband
            # Normalize/cap the feedback amplitude.  Positive rel_trend means
            # worsening; negative means improvement.
            signal=float(np.clip(rel_trend/d,-1.0,1.0))
            self.feedback_count+=1

            if rel_trend < -d:
                self.log_tau[active]=self._clip_log_tau(
                    self.log_tau[active] + self.cfg.feedback_log_step*signal)
                self.plateau_count=0
                event='improving_speed_active'
            elif rel_trend > d:
                self.log_tau[active]=self._clip_log_tau(
                    self.log_tau[active] + self.cfg.feedback_log_step*signal)
                self.plateau_count=0
                event='worsening_slow_active'
            else:
                self.plateau_count+=1
                event='plateau_wait'
                if self.plateau_count >= self.cfg.plateau_patience:
                    handover=self._handover()
                    event='plateau_handover' if handover is not None else 'plateau_slow_terminal'

        rec={
            'context':int(self.context_count),
            'loss_observed':loss,
            'ema_loss':float(self.ema_loss),
            'relative_trend':None if rel_trend is None else float(rel_trend),
            'event':event,
            'focus_before':active,
            'focus_after':self.focus,
            'tau_before':tau_before,
            'tau_after':self.snapshot(),
            'plateau_count':int(self.plateau_count),
            'feedback_count':int(self.feedback_count),
            'handover_count':int(self.handover_count),
            'handover':handover,
            'causal_semantics':'loss_t updates tau_(t+1); no retroactive parameter rewrite',
        }
        self.history.append(rec)
        return rec


class AdaptiveTimescaleFullForwardSystem(FullForwardPlasticMaskGITSystem):
    """R6B4 full-forward learner with only group-wise timescale modulation added."""

    def __init__(self,cfg,model,seed:int,tok:np.ndarray,pos:np.ndarray,
                 *,aux_lr_scale:float=1.0,timescale_cfg:TimescaleConfig|None=None):
        super().__init__(cfg,model,seed,tok,pos,aux_lr_scale=aux_lr_scale)
        self.timescale_cfg=copy.deepcopy(timescale_cfg or TimescaleConfig()).validate()
        self.timescale_controller=CausalTimescaleController(self.timescale_cfg)
        self.adaptive_timescale_version=FORMAT

    def timescale_snapshot(self):
        return self.timescale_controller.snapshot()

    def apply_update(self):
        """Exact R6B4 update directions, multiplied only by 1/tau(group)."""
        inv=1.0/max(1,self.accum)
        sf=self.timescale_controller.scale('fast')
        sm=self.timescale_controller.scale('medium')
        ss=self.timescale_controller.scale('slow')

        # FAST: output readout + relation-forming Q/K/V.
        self.WR+=sf*self.lrr*self.ar*inv
        self.WQ+=sf*self.lrq*self.aq*inv
        self.WK+=sf*self.lrk*self.ak*inv
        self.WV+=sf*self.lrv*self.av*inv

        # MEDIUM: output mixing / FFN body.
        self.WO+=sm*self.lro*self.ao*inv;self.WO*=self.maskO
        self.W1+=sm*self.lr1*self.a1*inv;self.W1*=self.mask1
        self.W2+=sm*self.lr2*self.a2*inv;self.W2*=self.mask2

        # SLOW: biases, normalization affine terms, and representation geometry.
        s=self.aux_lr_scale
        self.bq+=ss*s*self.lrq*self.abq*inv
        self.bk+=ss*s*self.lrk*self.abk*inv
        self.bv+=ss*s*self.lrv*self.abv*inv
        self.bo+=ss*s*self.lro*self.abo*inv
        self.b1+=ss*s*self.lr1*self.ab1*inv
        self.b2+=ss*s*self.lr2*self.ab2*inv
        self.gamma2+=ss*s*self.lr1*self.agamma2*inv
        self.beta2+=ss*s*self.lr1*self.abeta2*inv

        # Preserve the existing embedding-output M state and update route.  Only
        # the final parameter displacement is slowed by the SLOW timescale.
        mean_drive=self.aembed_drive*inv
        ME,_=core.fast_izh_step(self.vmemb,self.umemb,mean_drive.reshape(-1),self.zmemb,
                                core.G_M,core.CV_M,core.CU_M)
        ME=ME.reshape(self.c.N,self.c.D)
        eb=self.aconst*inv
        lr_emb=ss*s*min(self.lrq,self.lrk,self.lrv)
        lr_pos=ss*s*self.lrq
        for n,zid in enumerate(np.asarray(self._current_arr,dtype=np.int64)):
            self.tok[int(zid)]+=lr_emb*ME[n]*eb
            self.pos[n]+=lr_pos*ME[n]*eb

        self.update_count+=1
        self.aq.fill(0);self.ak.fill(0);self.av.fill(0);self.ao.fill(0)
        self.a1.fill(0);self.a2.fill(0);self.ar.fill(0);self.accum=0
        self._clear_aux_accumulators()

    def learn_mask_batch_arr(self,arr:np.ndarray,masked_positions:np.ndarray,
                             truth_tokens:np.ndarray,weights:np.ndarray,
                             rng:np.random.Generator):
        # tau_used is sampled before the current update.  The loss from this
        # context is observed only after super() has finished that update.
        tau_used=self.timescale_snapshot()
        rec=super().learn_mask_batch_arr(arr,masked_positions,truth_tokens,weights,rng)
        fb=self.timescale_controller.observe(float(rec['mean_nll']))
        out=dict(rec)
        out['tau_used_for_this_context']=tau_used
        out['tau_after_feedback_for_next_context']=self.timescale_snapshot()
        out['timescale_feedback']=fb
        return out


def _array_group_snapshot(system) -> Dict[str,Dict[str,np.ndarray]]:
    return {
        g:{k:np.asarray(getattr(system,k),dtype=np.float64).copy() for k in keys}
        for g,keys in GROUP_PARAMETERS.items()
    }


def _group_movement(before,system):
    rows={}
    for g,keys in GROUP_PARAMETERS.items():
        d2=0.0;n2=0.0;mx=0.0
        for k in keys:
            a=np.asarray(before[g][k],dtype=np.float64)
            b=np.asarray(getattr(system,k),dtype=np.float64)
            d=b-a
            d2+=float(np.sum(d*d));n2+=float(np.sum(a*a))
            if d.size:mx=max(mx,float(np.max(np.abs(d))))
        rows[g]={
            'delta_norm':math.sqrt(d2),
            'relative_change':math.sqrt(d2)/(math.sqrt(n2)+1e-12),
            'max_abs_change':mx,
        }
    return rows


def run_one_context(system,truth,ones,rng):
    sm=sample_mask_set(system.c.N,rng)
    truth=np.asarray(truth,dtype=np.int64)
    arr=truth.copy();arr[sm.masked_positions]=core.MASK
    mirror_seed=int(rng.integers(1,2**31-1))
    core.seed_numba(mirror_seed)
    rec=system.learn_mask_batch_arr(arr,sm.masked_positions,truth,ones,rng)
    rec=dict(rec)
    rec['mask_ratio']=float(sm.mask_ratio)
    rec['ratio_r']=float(sm.ratio_r)
    rec['mirror_seed']=mirror_seed
    return rec


def adaptive_controller_self_test():
    cfg=TimescaleConfig(
        tau_fast=1,tau_medium=5,tau_slow=25,
        ema_beta=.2,feedback_interval=4,trend_window=8,
        relative_trend_deadband=.002,feedback_log_step=.1,
        warmup_contexts=8,plateau_patience=2,
    )
    c=CausalTimescaleController(cfg)
    t0=c.snapshot()
    # Decreasing loss should eventually speed FAST.
    for x in np.linspace(4.0,2.0,40): c.observe(float(x))
    t1=c.snapshot()
    improved=t1['fast'] < t0['fast']

    # Long flat loss should hand over at least once.
    for _ in range(80): c.observe(2.0)
    handed=(c.handover_count>=1 and c.focus in ('medium','slow'))

    # Increasing loss should slow whichever group is active.
    active=c.focus;before=c.tau(active)
    for x in np.linspace(2.0,3.0,40): c.observe(float(x))
    after=c.tau(active)
    worsened=after > before

    bounds=all(cfg.tau_min-1e-12 <= v <= cfg.tau_max+1e-12 for v in c.snapshot().values())
    return {
        'pass':bool(improved and handed and worsened and bounds),
        'decreasing_loss_speeds_initial_fast':bool(improved),
        'plateau_causes_ordered_handover':bool(handed),
        'increasing_loss_slows_active':bool(worsened),
        'bounds_respected':bool(bounds),
        'final_focus':c.focus,
        'final_tau':c.snapshot(),
    }


def update_parity_self_test(seed=7311):
    """tau=1, adaptive off must reproduce one exact R6B4 update."""
    cfg=core.Config(D=24,H=4,M=8,FF=48,N=7,V=64,R=7)
    model=core.init_model(cfg,seed)
    rng=np.random.default_rng(seed+1)
    tok=rng.normal(size=(cfg.V+1,cfg.D))
    pos=rng.normal(size=(cfg.N,cfg.D))
    base_sys=FullForwardPlasticMaskGITSystem(cfg,model,seed+2,tok,pos,aux_lr_scale=1.0)
    ats_sys=AdaptiveTimescaleFullForwardSystem(
        cfg,model,seed+2,tok,pos,aux_lr_scale=1.0,
        timescale_cfg=TimescaleConfig(tau_fast=1,tau_medium=1,tau_slow=1,adaptive=False))
    truth=rng.integers(0,cfg.V,size=cfg.N,dtype=np.int64)
    masked=np.asarray([0,3,6],dtype=np.int64)
    arr=truth.copy();arr[masked]=core.MASK
    ones=np.ones(cfg.V,dtype=np.float64)

    # Keep all stochastic streams matched between the two arms.
    rng_state=copy.deepcopy(rng.bit_generator.state)
    ra=np.random.default_rng();ra.bit_generator.state=copy.deepcopy(rng_state)
    rb=np.random.default_rng();rb.bit_generator.state=copy.deepcopy(rng_state)
    core.seed_numba(seed+100)
    rec_a=base_sys.learn_mask_batch_arr(arr,masked,truth,ones,ra)
    core.seed_numba(seed+100)
    rec_b=ats_sys.learn_mask_batch_arr(arr,masked,truth,ones,rb)

    keys=list(FORWARD_PARAMETER_KEYS)+['BR','BO','B2']
    max_abs=0.0;bad=[]
    for k in keys:
        a=np.asarray(getattr(base_sys,k),dtype=np.float64)
        b=np.asarray(getattr(ats_sys,k),dtype=np.float64)
        d=float(np.max(np.abs(a-b))) if a.size else 0.0
        max_abs=max(max_abs,d)
        if d>1e-12:bad.append(k)
    pred_diff=abs(float(rec_a['mean_nll'])-float(rec_b['mean_nll']))
    return {
        'pass':bool(not bad and pred_diff<=1e-12),
        'max_parameter_abs_diff':max_abs,
        'mean_nll_abs_diff':pred_diff,
        'mismatched_fields':bad,
        'meaning':'with tau=1 and feedback disabled, v1 leaves the R6B4 update unchanged',
    }


def _token_stats(ids,vocab_size):
    a=np.asarray(ids,dtype=np.int64).reshape(-1)
    c=np.bincount(a,minlength=vocab_size).astype(np.float64)
    p=c/max(1.0,c.sum());nz=p[p>0]
    return {
        'active_codes':int(np.sum(c>0)),
        'perplexity':float(np.exp(-(nz*np.log(nz)).sum())),
        'top1_fraction':float(np.max(p)),
    }


def _save_controller_csv(path:Path,history:Iterable[Dict]):
    rows=[]
    for r in history:
        rows.append({
            'context':r['context'],'loss_observed':r['loss_observed'],'ema_loss':r['ema_loss'],
            'relative_trend':r['relative_trend'],'event':r['event'],
            'focus_before':r['focus_before'],'focus_after':r['focus_after'],
            'tau_fast_before':r['tau_before']['fast'],'tau_medium_before':r['tau_before']['medium'],'tau_slow_before':r['tau_before']['slow'],
            'tau_fast_after':r['tau_after']['fast'],'tau_medium_after':r['tau_after']['medium'],'tau_slow_after':r['tau_after']['slow'],
            'plateau_count':r['plateau_count'],'feedback_count':r['feedback_count'],'handover_count':r['handover_count'],
        })
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path)
    ap.add_argument('--tokenizer',type=Path)
    ap.add_argument('--out',type=Path,default=Path('run_adaptive_timescale_v1/result'))
    ap.add_argument('--fractional-epoch',type=float,default=1.0,
                    help='first functional run: fraction of one shuffled training epoch, (0,1]')
    ap.add_argument('--D',type=int,default=96);ap.add_argument('--heads',type=int,default=8)
    ap.add_argument('--features',type=int,default=24);ap.add_argument('--ff',type=int,default=192);ap.add_argument('--R',type=int,default=7)
    ap.add_argument('--scaffold-contexts',type=int,default=2000);ap.add_argument('--scaffold-paths',type=int,default=2500);ap.add_argument('--scaffold-lr',type=float,default=.6)
    ap.add_argument('--internal-lr-scale',type=float,default=.5)
    ap.add_argument('--aux-lr-scale',type=float,default=1.0)
    ap.add_argument('--tau-fast',type=float,default=1.0);ap.add_argument('--tau-medium',type=float,default=5.0);ap.add_argument('--tau-slow',type=float,default=25.0)
    ap.add_argument('--tau-min',type=float,default=.25);ap.add_argument('--tau-max',type=float,default=100.0)
    ap.add_argument('--ema-beta',type=float,default=.05);ap.add_argument('--feedback-interval',type=int,default=32)
    ap.add_argument('--trend-window',type=int,default=128);ap.add_argument('--trend-deadband',type=float,default=.005)
    ap.add_argument('--feedback-log-step',type=float,default=.08);ap.add_argument('--warmup-contexts',type=int,default=128)
    ap.add_argument('--plateau-patience',type=int,default=3)
    ap.add_argument('--retire-factor',type=float,default=4.0);ap.add_argument('--release-ratio',type=float,default=1.25)
    ap.add_argument('--eval-images',type=int,default=256);ap.add_argument('--gen-samples',type=int,default=100)
    ap.add_argument('--grid-samples',type=int,default=100);ap.add_argument('--decode-iterations',type=int,default=8)
    ap.add_argument('--choice-temperature',type=float,default=4.5)
    ap.add_argument('--encode-batch',type=int,default=256);ap.add_argument('--report-every-contexts',type=int,default=128)
    ap.add_argument('--seed',type=int,default=7311);ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--self-test-only',action='store_true')
    a=ap.parse_args()

    own_controller_test=adaptive_controller_self_test()
    parity_test=update_parity_self_test(a.seed)
    print('ADAPTIVE_CONTROLLER_SELF_TEST',json.dumps(own_controller_test),flush=True)
    print('R6B4_TAU1_PARITY_SELF_TEST',json.dumps(parity_test),flush=True)
    if not own_controller_test['pass'] or not parity_test['pass']:
        raise RuntimeError('Adaptive Timescale v1 self-test failed')
    if a.self_test_only:
        return
    if a.data is None or a.tokenizer is None:
        raise ValueError('--data and --tokenizer are required unless --self-test-only is used')
    if not (0.0 < a.fractional_epoch <= 1.0):
        raise ValueError('--fractional-epoch must be in (0,1] for v1')

    a.out.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(a.threads)
    device=torch.device('cpu')

    mg_mask_test=masking_self_test(n_tokens=49,vocab_size=64,trials=128,seed=a.seed+17)
    mg_controller_test=maskgit_controller_self_test()
    if not mg_mask_test['pass'] or not mg_controller_test['pass']:
        raise RuntimeError('existing MaskGIT structural self-test failed')

    xall,yall,xte0,yte0=core.load(a.data)
    x012,y012=core.subset012(xall,yall);xte,yte=core.subset012(xte0,yte0)
    xtr,ytr,xva,yva=core.split(x012,y012,.1,7)

    tokenizer,tok_meta=base.load_frozen_tokenizer(a.tokenizer,device)
    t0=time.time()
    tr,sp_tr=base.encode_snn(tokenizer,xtr,device,a.encode_batch)
    va,sp_va=base.encode_snn(tokenizer,xva,device,a.encode_batch)
    te,sp_te=base.encode_snn(tokenizer,xte,device,a.encode_batch)
    print('TOKENIZATION_DONE',json.dumps({'elapsed_s':time.time()-t0,'train':len(tr),'val':len(va),'test':len(te)}),flush=True)

    cfg=core.Config(D=a.D,H=a.heads,M=a.features,FF=a.ff,R=a.R)
    model=core.init_model(cfg,a.seed)
    for k in ('lrq','lrk','lrv','lro','lr1','lr2'):
        model[k]*=a.internal_lr_scale

    tok,pos,input_audit=build_input_embeddings(tokenizer,cfg.D,grid=7)
    rep_audit=representation_self_test(tokenizer,cfg.D,grid=7)
    if not rep_audit['pass']:
        raise RuntimeError('input representation audit failed')

    # Preserve the existing startup scaffold; it is not part of the new feedback.
    ones=np.ones(cfg.V,dtype=np.float64)
    F,T,scaffold_audit=build_maskgit_scaffold_contexts_unconditional(
        model,cfg,tr,tok,pos,a.scaffold_contexts,a.seed+101)
    ts=time.time()
    model['WR']=core.train_readout_msdl(F,T,ones,a.scaffold_paths,a.scaffold_lr,a.seed+202)
    print('SCAFFOLD_DONE',json.dumps({'elapsed_s':time.time()-ts,'masking':scaffold_audit}),flush=True)

    tcfg=TimescaleConfig(
        tau_fast=a.tau_fast,tau_medium=a.tau_medium,tau_slow=a.tau_slow,
        tau_min=a.tau_min,tau_max=a.tau_max,ema_beta=a.ema_beta,
        feedback_interval=a.feedback_interval,trend_window=a.trend_window,
        relative_trend_deadband=a.trend_deadband,feedback_log_step=a.feedback_log_step,
        warmup_contexts=a.warmup_contexts,plateau_patience=a.plateau_patience,
        retire_factor=a.retire_factor,release_ratio=a.release_ratio,adaptive=True).validate()
    system=AdaptiveTimescaleFullForwardSystem(
        cfg,model,a.seed+303,tok,pos,aux_lr_scale=a.aux_lr_scale,timescale_cfg=tcfg)

    # One visit per image, matching the existing context-epoch semantics.
    train_rng=np.random.default_rng(a.seed+404)
    n_contexts=max(1,int(round(a.fractional_epoch*len(tr))))
    indices=train_rng.permutation(len(tr))[:n_contexts]
    start_snapshot=_array_group_snapshot(system)
    report_snapshot=_array_group_snapshot(system)
    reports=[];target_events=0;train_start=time.time()

    for j,image_index in enumerate(indices,start=1):
        rec=run_one_context(system,np.asarray(tr[int(image_index)]),ones,train_rng)
        target_events+=int(rec['masked_targets'])
        if j==1 or j==n_contexts or (a.report_every_contexts>0 and j%a.report_every_contexts==0):
            move=_group_movement(report_snapshot,system)
            report_snapshot=_array_group_snapshot(system)
            row={
                'context':int(j),'image_index':int(image_index),
                'masked_targets':int(rec['masked_targets']),'target_events':int(target_events),
                'batch_mean_nll':float(rec['mean_nll']),
                'cumulative_online_nll':float(system.obs_nll/max(1,system.obs_count)),
                'tau':system.timescale_snapshot(),'focus':system.timescale_controller.focus,
                'last_feedback_event':rec['timescale_feedback']['event'],
                'movement_since_last_report':move,'elapsed_s':float(time.time()-train_start),
            }
            reports.append(row);print('ADAPTIVE_TRAIN_PROGRESS',json.dumps(row),flush=True)

    # First-run endpoint: masked validation plus autonomous generation and images.
    final_model=trainer.model_dict_from_system(system)
    masked_eval={
        str(r):eval_masked_unconditional(final_model,cfg,te,system.tok,system.pos,r,
                                         a.seed+5000+int(1000*r),a.eval_images)
        for r in (.25,.5,.9,1.0)
    }

    posthoc_clf=LogisticRegression(max_iter=250,solver='lbfgs',random_state=7).fit(
        xtr.reshape(len(xtr),-1),ytr)
    ids,traces,audits=generation_tokens_maskgit(
        final_model,cfg,a.gen_samples,system.tok,system.pos,a.seed+6060,
        num_iter=a.decode_iterations,choice_temperature=a.choice_temperature)
    imgs=base.decode_snn(tokenizer,ids,device)
    save_grid_unlabeled(imgs[:min(a.grid_samples,len(imgs))],a.out/'adaptive_timescale_v1_final_generation.png')
    np.savez_compressed(a.out/'adaptive_timescale_v1_generated.npz',tokens=ids,images=imgs)
    (a.out/'adaptive_timescale_v1_generation_trace.json').write_text(json.dumps(traces,indent=2),encoding='utf-8')

    generation={
        'samples':int(len(ids)),
        'decode_iterations':int(a.decode_iterations),
        'choice_temperature':float(a.choice_temperature),
        'posthoc':posthoc_generation_metric(posthoc_clf,imgs),
        'token_stats':_token_stats(ids,cfg.V),
        'all_trace_audits_pass':bool(all(x['pass'] for x in audits)),
    }

    final_move=_group_movement(start_snapshot,system)
    finite=bool(all(np.all(np.isfinite(v)) for v in vars(system).values() if isinstance(v,np.ndarray)))
    summary={
        'format':FORMAT,
        'scientific_question':'Can causal adaptive plasticity timescales rescue functional MaskGIT image learning without changing the existing local learning rule?',
        'only_new_learning_mechanism':'group-wise 1/tau modulation plus causal loss->future-tau feedback',
        'backpropagation_or_bptt_used':False,
        'future_loss_gradient_used':False,
        'retroactive_parameter_rewrite':False,
        'parameter_groups':GROUP_PARAMETERS,
        'timescale_config':asdict(tcfg),
        'initial_tau':{'fast':a.tau_fast,'medium':a.tau_medium,'slow':a.tau_slow},
        'final_tau':system.timescale_snapshot(),
        'final_focus':system.timescale_controller.focus,
        'handover_count':int(system.timescale_controller.handover_count),
        'contexts_trained':int(n_contexts),'fractional_epoch':float(a.fractional_epoch),
        'target_events':int(target_events),
        'masked_eval':masked_eval,
        'generation':generation,
        'group_movement_total':final_move,
        'all_system_arrays_finite':finite,
        'self_tests':{
            'adaptive_controller':own_controller_test,
            'tau1_r6b4_update_parity':parity_test,
            'maskgit_masking':mg_mask_test,
            'maskgit_controller':mg_controller_test,
            'input_representation':rep_audit,
        },
        'tokenizer':{'metadata':tok_meta,'activity_train':sp_tr,'activity_val':sp_va,'activity_test':sp_te},
        'scaffold_audit':scaffold_audit,
        'reports':reports,
    }
    (a.out/'ADAPTIVE_TIMESCALE_V1_RESULTS.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    _save_controller_csv(a.out/'adaptive_timescale_v1_controller_trace.csv',system.timescale_controller.history)
    (a.out/'adaptive_timescale_v1_controller_trace.json').write_text(
        json.dumps(system.timescale_controller.history,indent=2),encoding='utf-8')
    np.savez_compressed(a.out/'adaptive_timescale_v1_final_state.npz',**system.full_weights())
    with (a.out/'adaptive_timescale_v1_checkpoint.pkl').open('wb') as f:
        pickle.dump({
            'format':FORMAT,'system':system,
            'train_rng_state':copy.deepcopy(train_rng.bit_generator.state),
            'contexts_trained':int(n_contexts),'fractional_epoch':float(a.fractional_epoch),
            'timescale_config':asdict(tcfg),
        },f,protocol=pickle.HIGHEST_PROTOCOL)

    print('ADAPTIVE_TIMESCALE_V1_FINAL',json.dumps({
        'final_tau':summary['final_tau'],'final_focus':summary['final_focus'],
        'handover_count':summary['handover_count'],'masked_eval':masked_eval,
        'generation':generation,'all_system_arrays_finite':finite,
    }),flush=True)


if __name__=='__main__':
    main()
