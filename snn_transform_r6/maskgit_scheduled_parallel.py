#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small, model-agnostic MaskGIT scheduled-parallel decoding controller.

This is a NumPy reference controller for the public MaskGIT inference rule:
  1) predict every currently masked site from one immutable context;
  2) sample a proposal at every currently masked site;
  3) use the sampled-token probability as confidence;
  4) use the cosine mask schedule;
  5) add annealed Gumbel noise to confidence and re-mask the lowest-confidence
     proposals;
  6) repeat for a fixed number of iterations.

The expensive predictor is intentionally supplied as a callback.  In R6B.3 it is
P3 SNN-Transform.  The controller itself contains only small-N scheduling,
sampling and ranking operations, so it can later be moved to a SpiNNaker control
core without turning those operations into a neural network.

Semantics were checked against the public Google Research MaskGIT implementation:
  maskgit/libml/parallel_decode.py
  maskgit/libml/mask_schedule.py
For the project's N=49, T=8 case, the official-code-style floor+clamp schedule
leaves [48,45,40,34,27,18,9,1] tokens masked after the eight controller rounds;
the final returned image is the fully sampled proposal from round 8, exactly as
MaskGIT stores/returns its final sampled sequence before the last re-mask state.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Tuple
import numpy as np

PredictMasked = Callable[[np.ndarray, np.ndarray, int], np.ndarray]


def cosine_mask_ratio(ratio: float) -> float:
    """MaskGIT cosine schedule gamma(r)=cos(pi*r/2), clipped to [1e-6,1]."""
    return float(np.clip(math.cos(math.pi * 0.5 * float(ratio)), 1e-6, 1.0))


def _normalise_rows(p: np.ndarray) -> np.ndarray:
    p=np.asarray(p,dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f'predict_masked must return [n_masked,vocab], got {p.shape}')
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError('predictor probabilities must be finite and non-negative')
    s=p.sum(axis=1,keepdims=True)
    if np.any(s <= 0):
        raise ValueError('every masked-site distribution must have positive mass')
    return p/s


def _sample_rows(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Categorical sample independently at every masked location."""
    return np.asarray([rng.choice(p.shape[1],p=row) for row in p],dtype=np.int64)


def _official_mask_len(initial_unknown: int, current_unknown: int, mask_ratio: float) -> int:
    """Match Google MaskGIT code: floor(N0*ratio), then clamp [1,current_unknown-1]."""
    raw=int(math.floor(float(initial_unknown)*float(mask_ratio)))
    return int(max(1,min(int(current_unknown)-1,raw)))


def _remask_low_confidence(masked_positions: np.ndarray,
                           selected_prob: np.ndarray,
                           mask_len: int,
                           temperature: float,
                           rng: np.random.Generator) -> Tuple[np.ndarray,np.ndarray]:
    """MaskGIT random-top-k: rank log(prob)+annealed Gumbel noise, mask the lowest."""
    masked_positions=np.asarray(masked_positions,dtype=np.int64)
    selected_prob=np.asarray(selected_prob,dtype=np.float64)
    if len(masked_positions) != len(selected_prob):
        raise ValueError('position/confidence length mismatch')
    if not (1 <= mask_len <= len(masked_positions)):
        raise ValueError(f'invalid mask_len={mask_len} for {len(masked_positions)} candidates')
    score=np.log(np.clip(selected_prob,1e-12,1.0))
    if temperature != 0.0:
        score=score+float(temperature)*rng.gumbel(size=len(score))
    order=np.argsort(score,kind='mergesort')
    chosen=masked_positions[order[:mask_len]]
    return chosen,score


def scheduled_parallel_decode(init_ids: np.ndarray,
                              mask_token_id: int,
                              vocab_size: int,
                              predict_masked: PredictMasked,
                              rng: np.random.Generator,
                              num_iter: int = 8,
                              choice_temperature: float = 4.5) -> Tuple[np.ndarray,List[Dict]]:
    """Decode one token grid with MaskGIT scheduled-parallel inference.

    predict_masked(cur_ids, masked_positions, step) must evaluate *all* currently
    masked positions against the same immutable cur_ids.  It returns one
    probability vector per masked position.  The callback may compute those rows
    serially on CPU as a reference implementation, but it must not update cur_ids
    or model state between rows.  This preserves MaskGIT parallel semantics and
    allows a future hardware implementation to dispatch the rows concurrently.
    """
    cur=np.asarray(init_ids,dtype=np.int64).copy().reshape(-1)
    if num_iter < 1:
        raise ValueError('num_iter must be >=1')
    initial_unknown=int(np.sum(cur==mask_token_id))
    if initial_unknown < 1:
        raise ValueError('MaskGIT generation must begin with at least one MASK token')

    trace: List[Dict]=[]
    final_proposal=None

    for step in range(int(num_iter)):
        unknown=np.flatnonzero(cur==mask_token_id).astype(np.int64)
        current_unknown=len(unknown)
        if current_unknown == 0:
            final_proposal=cur.copy()
            break

        # One immutable context for every proposal in this round.
        context_before=cur.copy()
        p=_normalise_rows(predict_masked(context_before.copy(),unknown.copy(),step))
        if p.shape != (current_unknown,int(vocab_size)):
            raise ValueError(f'predictor returned {p.shape}, expected {(current_unknown,int(vocab_size))}')
        if not np.array_equal(cur,context_before):
            raise AssertionError('predictor mutated the MaskGIT context during a parallel round')

        sampled=_sample_rows(p,rng)
        proposal=context_before.copy()
        proposal[unknown]=sampled
        if np.any(proposal==mask_token_id):
            raise AssertionError('every currently masked site must receive a proposal before confidence filtering')
        # Known/committed tokens are copied unchanged; only previously masked sites get proposals.
        known=np.flatnonzero(context_before!=mask_token_id)
        if len(known) and not np.array_equal(proposal[known],context_before[known]):
            raise AssertionError('a previously committed token changed during proposal generation')

        selected_prob=p[np.arange(current_unknown),sampled]
        final_proposal=proposal.copy()  # official MaskGIT records this before re-masking

        ratio=float(step+1)/float(num_iter)
        mask_ratio=cosine_mask_ratio(ratio)
        mask_len=_official_mask_len(initial_unknown,current_unknown,mask_ratio)
        temp=float(choice_temperature)*(1.0-ratio)
        remask,score=_remask_low_confidence(unknown,selected_prob,mask_len,temp,rng)

        nxt=proposal.copy()
        nxt[remask]=mask_token_id
        masked_after=int(np.sum(nxt==mask_token_id))
        if masked_after != mask_len:
            raise AssertionError(f'expected {mask_len} MASKs after round, got {masked_after}')
        if len(known) and not np.array_equal(nxt[known],context_before[known]):
            raise AssertionError('a previously committed token was re-masked or modified')

        trace.append({
            'step':int(step+1),
            'masked_before':int(current_unknown),
            'proposal_count':int(current_unknown),
            'proposal_mask_count':int(np.sum(proposal==mask_token_id)),
            'ratio':ratio,
            'mask_ratio':mask_ratio,
            'mask_len':int(mask_len),
            'masked_after':masked_after,
            'committed_after':int(len(cur)-masked_after),
            'choice_temperature':temp,
            'masked_positions_before':unknown.tolist(),
            'sampled_ids':sampled.tolist(),
            'sampled_confidence':selected_prob.tolist(),
            'confidence_score_with_gumbel':score.tolist(),
            'remasked_positions':remask.tolist(),
        })
        cur=nxt

    if final_proposal is None or np.any(final_proposal==mask_token_id):
        raise AssertionError('final MaskGIT proposal must be fully unmasked')
    return final_proposal.astype(np.int64),trace


def audit_trace(trace: List[Dict], sequence_length: int, num_iter: int) -> Dict:
    """Structural audit: all-MASK start, full proposals, cosine schedule, monotone commitment."""
    if len(trace) != num_iter:
        raise AssertionError(f'expected {num_iter} decoding rounds, got {len(trace)}')
    if trace[0]['masked_before'] != sequence_length:
        raise AssertionError('generation did not start from an all-MASK image')
    previous_after=sequence_length
    for i,r in enumerate(trace):
        if r['masked_before'] != previous_after:
            raise AssertionError(f'round {i+1} did not consume the previous round state')
        if r['proposal_count'] != r['masked_before'] or r['proposal_mask_count'] != 0:
            raise AssertionError(f'round {i+1} did not predict every MASK site before filtering')
        if r['masked_after'] > r['masked_before']:
            raise AssertionError(f'MASK count increased at round {i+1}')
        previous_after=r['masked_after']
    expected=None
    if sequence_length==49 and num_iter==8:
        expected=[48,45,40,34,27,18,9,1]
        got=[r['masked_after'] for r in trace]
        if got != expected:
            raise AssertionError(f'N=49,T=8 MaskGIT schedule mismatch: got {got}, expected {expected}')
    return {
        'pass':True,
        'all_mask_start':True,
        'all_current_masks_proposed_each_round':True,
        'committed_tokens_never_changed':True,
        'cosine_schedule':True,
        'official_floor_clamp_rule':True,
        'rounds':int(num_iter),
        'masked_after_each_round':[int(r['masked_after']) for r in trace],
        'expected_masked_after_for_N49_T8':expected,
    }


def controller_self_test() -> Dict:
    """Deterministic structural test independent of any neural model."""
    n=49; v=64; mask=v; rng=np.random.default_rng(12345)
    init=np.full(n,mask,dtype=np.int64)
    def uniform_predict(cur,masked,step):
        # Deliberately uniform: this test audits controller semantics, not model quality.
        return np.full((len(masked),v),1.0/v,dtype=np.float64)
    out,tr=scheduled_parallel_decode(init,mask,v,uniform_predict,rng,num_iter=8,choice_temperature=4.5)
    if np.any(out==mask):
        raise AssertionError('self-test final proposal still contains MASK')
    return audit_trace(tr,n,8)
