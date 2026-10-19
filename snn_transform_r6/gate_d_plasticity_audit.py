#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D: native MSDL plasticity-organization audit for R6B4.

Scientific scope
----------------
This is an instrumentation/diagnostic gate.  It does NOT change the candidate
learning rule, learning rates, target order, reciprocal microsteps, E/M
persistence, sparse masks, or update boundary.

D0: prove the instrumented T0 trajectory is bitwise-identical to an
    uninstrumented R6B4 aux=1.0 reference at contexts 256/512/1024.
D1: audit W2 local credit at event and context level against an exact
    instantaneous masked-support descent oracle under GT/T1/T2 visible context.
D2: record exact per-context group movement and deterministic low-dimensional
    CountSketches for joint update-spectrum analysis.

Important interpretation boundary
---------------------------------
The gradient/descent oracle is analysis-only.  It never enters the candidate
learner and never modifies a parameter or dynamical state.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'snn_transform_p3_image'))
sys.path.insert(0, str(HERE))

import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
import r6b3_context_epoch_training as trainer
from maskgit_input_representation import build_input_embeddings
from maskgit_msdl_training import sample_mask_set
from r6b4_full_forward_plasticity import (
    FullForwardPlasticMaskGITSystem,
    upgrade_r6b3_system,
)

TRACK = ('WQ', 'WK', 'WV', 'WO', 'W1', 'W2', 'WR')
CHECKPOINTS = (256, 512, 1024)
EXPOSURE_Q = 1.0 / 7.0
DEFAULT_SKETCH_DIM = 64
FALSE_TOKEN = 61
MODEL_KEYS = (
    'WQ', 'WK', 'WV', 'WO', 'W1', 'W2', 'WR',
    'BR', 'BO', 'B2', 'maskO', 'mask1', 'mask2', 'omega',
    'bq', 'bk', 'bv', 'bo', 'b1', 'b2', 'theta', 'gamma2', 'beta2',
    'eps2', 'lrq', 'lrk', 'lrv', 'lro', 'lr1', 'lr2', 'lrr',
    'recip_scale', 'sigq', 'sigk',
)


def clone_rng(state):
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(state)
    return rng


def json_state(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(',', ':'))


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(aa)); nb = float(np.linalg.norm(bb))
    if na <= eps or nb <= eps:
        return float('nan')
    return float(np.dot(aa, bb) / (na * nb))


def model_dict(s):
    out = {}
    for k in MODEL_KEYS:
        v = getattr(s, k)
        out[k] = np.array(v, copy=True) if isinstance(v, np.ndarray) else v
    return out


def inverse_cdf_rows(p: np.ndarray, u: np.ndarray) -> np.ndarray:
    c = np.cumsum(p, axis=1)
    c[:, -1] = 1.0
    return np.minimum(np.sum(c < np.asarray(u)[:, None], axis=1), p.shape[1] - 1).astype(np.int64)


def exposure_subset(masked: np.ndarray, j0: int, n: int) -> np.ndarray:
    vis = np.setdiff1d(np.arange(n, dtype=np.int64), np.asarray(masked, dtype=np.int64), assume_unique=True)
    k = int(math.floor(EXPOSURE_Q * len(vis) + 0.5))
    if k <= 0:
        return np.empty(0, np.int64)
    rng = np.random.default_rng(550001 + int(j0))
    return np.sort(rng.choice(vis, size=k, replace=False).astype(np.int64))


def model_exposure_tokens(s, truth, masked, E, j0):
    """C2-compatible model exposure without consuming the training RNG stream."""
    if len(E) == 0:
        return np.empty(0, np.int64), np.empty((0, s.c.V)), np.empty(0)
    arr = np.asarray(truth, dtype=np.int64).copy()
    arr[np.asarray(masked, dtype=np.int64)] = core.MASK
    arr[np.asarray(E, dtype=np.int64)] = core.MASK
    xn, xraw = s.context_from_arr(arr)
    pred = core.P3ImageSystem(s.c, model_dict(s), 870001 + int(j0))
    pred.rebuild(xn)
    p = np.empty((len(E), s.c.V), dtype=np.float64)
    for qi, t0 in enumerate(E):
        t = int(t0)
        a, q, num, den = pred._read(t)
        _u, _h, _rf, pp = pred._tail(a, np.asarray(xraw[t], dtype=np.float64))
        p[qi] = pp
    ur = np.random.default_rng(990001 + int(j0)).random(len(E))
    z = inverse_cdf_rows(p, ur)
    return z, p, ur


def make_t2_plan(t1_exposure_events: List[Dict]):
    """Preserve T1 error locations and global wrong-token marginal, derange identities."""
    err = []
    for ci, e in enumerate(t1_exposure_events):
        for li, (z, g, c) in enumerate(zip(e.get('tokens', []), e.get('gt', []), e.get('correct', []))):
            if not c:
                err.append((ci, li, int(z), int(g)))
    toks = np.array([x[2] for x in err], dtype=np.int64)
    gts = np.array([x[3] for x in err], dtype=np.int64)
    n = len(err)
    rng = np.random.default_rng(771337)
    assign = toks[rng.permutation(n)].copy()
    for _pass in range(100):
        conf = np.flatnonzero(assign == gts)
        if not len(conf):
            break
        for i in conf:
            if assign[i] != gts[i]:
                continue
            cand = np.flatnonzero((assign != gts[i]) & (assign[i] != gts))
            cand = cand[cand != i]
            if len(cand):
                j = int(cand[rng.integers(len(cand))])
                assign[i], assign[j] = assign[j], assign[i]
    if np.any(assign == gts):
        raise RuntimeError('T2 could not satisfy wrong-token constraint')
    for _pass in range(100):
        fix = np.flatnonzero(assign == toks)
        if not len(fix):
            break
        changed = 0
        for i in fix:
            if assign[i] != toks[i]:
                continue
            cand = np.flatnonzero(
                (assign != gts[i]) & (assign != toks[i]) &
                (assign[i] != gts) & (assign[i] != toks)
            )
            cand = cand[cand != i]
            if len(cand):
                j = int(cand[rng.integers(len(cand))])
                assign[i], assign[j] = assign[j], assign[i]
                changed += 1
        if changed == 0:
            break
    if np.any(assign == gts) or np.any(assign == toks):
        raise RuntimeError('T2 could not construct strict error-matched derangement')
    plan = []
    lookup = {(ci, li): int(assign[k]) for k, (ci, li, _, _) in enumerate(err)}
    for ci, e in enumerate(t1_exposure_events):
        E = np.asarray(e['E'], dtype=np.int64)
        gt = np.asarray(e.get('gt', []), dtype=np.int64)
        corr = np.asarray(e.get('correct', []), dtype=bool)
        zz = np.array(gt, copy=True)
        for li in range(len(E)):
            if not corr[li]:
                zz[li] = lookup[(ci, li)]
        plan.append({'E': E.tolist(), 'tokens': zz.tolist()})
    audit = {
        'error_events': int(n),
        'wrong_token_marginal_exact': bool(np.array_equal(np.sort(assign), np.sort(toks))),
        'all_assigned_wrong': bool(np.all(assign != gts)),
        'strict_event_derangement': bool(np.all(assign != toks)),
        'same_token_as_T1_fraction': float(np.mean(assign == toks)) if n else 0.0,
        'interpretation_boundary': (
            'T2 removes the exact T1 wrong-token identity association while preserving '
            'the T1 exposure positions and correct/error-location pattern.'
        ),
    }
    return plan, audit


def w2_descent_oracle(system, rec: Dict, target: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact instantaneous negative gradient wrt W2, restricted to native sparse support."""
    p = np.asarray(rec['p'], dtype=np.float64)
    h = np.asarray(rec['h'], dtype=np.float64)
    t = int(rec['t'])
    a = np.asarray(rec['a'], dtype=np.float64)
    one = np.zeros(system.c.V, dtype=np.float64)
    one[int(target)] = 1.0
    z = np.asarray(system._current_xraw[t], dtype=np.float64) + system.bo + system.WO @ a
    y = z + system.W2 @ h + system.b2
    s = math.sqrt(float(np.mean(y * y)))
    r = s + 1e-8
    dlogits = p - one
    drf = system.WR.T @ dlogits
    q = np.asarray(drf[:-1], dtype=np.float64)
    if s <= 1e-30:
        grad_y = 0.5 * q / r
    else:
        grad_y = 0.5 * q / r - 0.5 * float(np.dot(q, y)) * y / (system.c.D * s * r * r)
    post_descent = -grad_y
    g = np.outer(post_descent, h) * system.mask2
    return g, post_descent, h


def _loss_from_y(y, WR, target):
    y = np.asarray(y, dtype=np.float64)
    r = math.sqrt(float(np.mean(y * y))) + 1e-8
    rf = np.empty(len(y) + 1, dtype=np.float64)
    rf[:-1] = 0.5 * y / r
    rf[-1] = 0.5
    z = WR @ rf
    z = z - np.max(z)
    p = np.exp(z); p /= np.sum(p)
    return -math.log(max(float(p[int(target)]), 1e-300))


def oracle_self_test(seed: int = 12031):
    rng = np.random.default_rng(seed)
    D, FF, V = 12, 17, 9
    y = rng.normal(size=D); h = rng.normal(size=FF)
    WR = rng.normal(scale=0.2, size=(V, D + 1)); target = int(rng.integers(V))
    r = math.sqrt(float(np.mean(y * y))) + 1e-8
    rf = np.empty(D + 1); rf[:-1] = 0.5 * y / r; rf[-1] = 0.5
    logits = WR @ rf; logits -= logits.max(); p = np.exp(logits); p /= p.sum()
    one = np.zeros(V); one[target] = 1.0
    q = (WR.T @ (p - one))[:-1]
    s = math.sqrt(float(np.mean(y * y)))
    grad_y = 0.5 * q / r - 0.5 * float(np.dot(q, y)) * y / (D * s * r * r)
    gradW = np.outer(grad_y, h)
    dW = rng.normal(size=(D, FF)); eps = 1e-6
    lp = _loss_from_y(y + eps * (dW @ h), WR, target)
    lm = _loss_from_y(y - eps * (dW @ h), WR, target)
    fd = (lp - lm) / (2.0 * eps); an = float(np.sum(gradW * dW))
    rel = abs(fd - an) / max(1.0, abs(fd), abs(an))
    return {'finite_difference': float(fd), 'analytic': float(an), 'relative_error': float(rel), 'pass': bool(rel < 2e-7)}


class AuditFullForwardPlasticMaskGITSystem(FullForwardPlasticMaskGITSystem):
    """Read-only instrumentation subclass. Native learning calls super() unchanged."""

    def audit_init(self, sketch_dim: int, sketch_seed: int = 20261020):
        self._audit_sketch_dim = int(sketch_dim)
        self._audit_sketch_seed = int(sketch_seed)
        self._audit_maps = {}
        for gi, k in enumerate(TRACK):
            n = int(np.asarray(getattr(self, k)).size)
            rng = np.random.default_rng(self._audit_sketch_seed + 1009 * (gi + 1))
            bins = rng.integers(0, self._audit_sketch_dim, size=n, dtype=np.int64)
            signs = np.where(rng.random(n) < 0.5, -1.0, 1.0).astype(np.float64)
            self._audit_maps[k] = (bins, signs)
        self._audit_event_rows = []; self._audit_context_rows = []; self._audit_sketch_rows = []
        self._audit_path_norm = {k: 0.0 for k in TRACK}
        self._audit_start = {k: np.array(getattr(self, k), copy=True) for k in TRACK}
        self._audit_ctx = None; self._audit_U = None; self._audit_G = None
        self._audit_rank = 0; self._audit_w_before = None

    def _audit_countsketch(self, k: str, delta: np.ndarray) -> np.ndarray:
        bins, signs = self._audit_maps[k]
        x = np.asarray(delta, dtype=np.float64).reshape(-1)
        return np.bincount(bins, weights=signs * x, minlength=self._audit_sketch_dim).astype(np.float64)

    def audit_begin_context(self, *, arm: str, context: int, image_index: int,
                            truth: np.ndarray, masked: np.ndarray, E: np.ndarray,
                            mirror_seed: int):
        self._audit_ctx = {
            'arm': str(arm), 'context': int(context), 'image_index': int(image_index),
            'masked': np.asarray(masked, dtype=np.int64).copy(),
            'E': np.asarray(E, dtype=np.int64).copy(),
            'truth': np.asarray(truth, dtype=np.int64).copy(),
            'mirror_seed': int(mirror_seed),
        }
        self._audit_U = np.zeros_like(self.W2, dtype=np.float64)
        self._audit_G = np.zeros_like(self.W2, dtype=np.float64)
        self._audit_rank = 0
        self._audit_w_before = {k: np.array(getattr(self, k), copy=True) for k in TRACK}

    def accumulate_cached_target(self, rec: Dict, target: int, weight: float = 1.0):
        p = super().accumulate_cached_target(rec, target, weight)
        if self._audit_ctx is None:
            return p
        E2 = core.CV_E * (self.ve2 - core.V0) + core.CU_E * (self.ue2 - core.U0)
        M2 = core.CV_M * (self.vm2 - core.V0) + core.CU_M * (self.um2 - core.U0)
        u = np.outer(M2, E2) * self.mask2
        g, post_descent, h = w2_descent_oracle(self, rec, int(target))
        self._audit_U += u; self._audit_G += g
        pp = np.asarray(rec['p'], dtype=np.float64); pred = int(np.argmax(pp)); tgt = int(target)
        dot = float(np.sum(u * g)); un = float(np.linalg.norm(u)); gn = float(np.linalg.norm(g))
        self._audit_event_rows.append({
            'arm': self._audit_ctx['arm'], 'context': int(self._audit_ctx['context']),
            'image_index': int(self._audit_ctx['image_index']), 'rank': int(self._audit_rank),
            'position': int(rec['t']), 'target': tgt, 'pred': pred,
            'correct': bool(pred == tgt), 'false61': bool(tgt != FALSE_TOKEN and pred == FALSE_TOKEN),
            'gt61': bool(tgt == FALSE_TOKEN), 'pred61': bool(pred == FALSE_TOKEN),
            'p_target': float(pp[tgt]), 'p61': float(pp[FALSE_TOKEN]) if FALSE_TOKEN < len(pp) else float('nan'),
            'u_norm': un, 'g_norm': gn, 'signed_descent': dot,
            'efficiency': float(dot / (un * un + 1e-30)), 'cosine': cosine(u, g),
            'A_M': cosine(M2, post_descent), 'A_E': cosine(E2, h),
        })
        self._audit_rank += 1
        return p

    def audit_finish_context(self):
        if self._audit_ctx is None:
            raise AssertionError('audit_finish_context without begin')
        nmask = max(1, int(len(self._audit_ctx['masked'])))
        U = np.asarray(self._audit_U, dtype=np.float64); G = np.asarray(self._audit_G, dtype=np.float64)
        actual_w2 = self.W2 - self._audit_w_before['W2']
        expected_w2 = self.lr2 * U / nmask
        native_identity_max = float(np.max(np.abs(actual_w2 - expected_w2)))
        Gmean = G / nmask
        dot = float(np.sum(actual_w2 * Gmean)); un = float(np.linalg.norm(actual_w2)); gn = float(np.linalg.norm(Gmean))
        self._audit_context_rows.append({
            'arm': self._audit_ctx['arm'], 'context': int(self._audit_ctx['context']),
            'image_index': int(self._audit_ctx['image_index']), 'nmask': int(nmask),
            'exposure_n': int(len(self._audit_ctx['E'])), 'mirror_seed': int(self._audit_ctx['mirror_seed']),
            'w2_native_update_identity_max_abs': native_identity_max,
            'w2_actual_step_norm': un, 'w2_oracle_mean_norm': gn,
            'context_signed_descent': dot, 'context_efficiency': float(dot / (un * un + 1e-30)),
            'context_cosine': cosine(actual_w2, Gmean),
        })
        sketch_row = {'arm': self._audit_ctx['arm'], 'context': int(self._audit_ctx['context'])}
        for k in TRACK:
            d = np.asarray(getattr(self, k), dtype=np.float64) - self._audit_w_before[k]
            dn = float(np.linalg.norm(d)); self._audit_path_norm[k] += dn
            sketch_row[f'{k}__norm'] = dn; sketch_row[f'{k}__sketch'] = self._audit_countsketch(k, d)
        self._audit_sketch_rows.append(sketch_row)
        self._audit_ctx = None; self._audit_U = None; self._audit_G = None; self._audit_w_before = None

    def audit_terminal_mobility(self):
        out = {}
        for k in TRACK:
            a = self._audit_start[k]; b = np.asarray(getattr(self, k), dtype=np.float64)
            net = float(np.linalg.norm(b - a)); denom = float(np.linalg.norm(a)) + 1e-12
            path = float(self._audit_path_norm[k])
            out[k] = {'net_abs': net, 'path_abs': path, 'net_relative': net / denom,
                      'path_relative': path / denom, 'tortuosity': path / (net + 1e-30)}
        return out


def make_audit_clone(system, sketch_dim: int):
    a = pickle.loads(pickle.dumps(system, pickle.HIGHEST_PROTOCOL))
    a.__class__ = AuditFullForwardPlasticMaskGITSystem
    a.audit_init(sketch_dim=sketch_dim)
    return a


def native_state_digest(s) -> str:
    h = hashlib.sha256()
    for k in sorted(vars(s)):
        if k.startswith('_audit_'):
            continue
        v = getattr(s, k); h.update(k.encode('utf-8') + b'\0')
        if isinstance(v, np.ndarray):
            a = np.ascontiguousarray(v); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
        elif isinstance(v, np.random.Generator):
            h.update(json_state(v.bit_generator.state).encode())
        elif isinstance(v, (int, float, bool, str, type(None), np.integer, np.floating)):
            h.update(repr(v).encode())
    return h.hexdigest()


def compare_native_states(ref, aud) -> Dict:
    array_diffs = []; scalar_diffs = []; rng_diffs = []
    for k, v in vars(ref).items():
        if k.startswith('_audit_') or not hasattr(aud, k):
            continue
        w = getattr(aud, k)
        if isinstance(v, np.ndarray) and isinstance(w, np.ndarray):
            if v.shape != w.shape or v.dtype != w.dtype or not np.array_equal(v, w, equal_nan=True):
                mx = float(np.max(np.abs(v - w))) if v.shape == w.shape and v.size else float('inf')
                array_diffs.append({'field': k, 'max_abs': mx, 'shape_ref': list(v.shape), 'shape_audit': list(w.shape)})
        elif isinstance(v, np.random.Generator) and isinstance(w, np.random.Generator):
            if json_state(v.bit_generator.state) != json_state(w.bit_generator.state):
                rng_diffs.append(k)
        elif isinstance(v, (int, float, bool, str, type(None), np.integer, np.floating)) and v != w:
            scalar_diffs.append({'field': k, 'ref': repr(v), 'audit': repr(w)})
    rd = native_state_digest(ref); ad = native_state_digest(aud)
    return {'ref_digest': rd, 'audit_digest': ad, 'array_mismatch_count': len(array_diffs),
            'scalar_mismatch_count': len(scalar_diffs), 'rng_mismatch_count': len(rng_diffs),
            'array_diffs': array_diffs[:20], 'scalar_diffs': scalar_diffs[:20], 'rng_diffs': rng_diffs,
            'bitwise_pass': bool(not array_diffs and not scalar_diffs and not rng_diffs and rd == ad)}


def target_order_signature(rows: List[Dict], context: int) -> str:
    x = [(int(r['rank']), int(r['position']), int(r['target'])) for r in rows if int(r['context']) == int(context)]
    return hashlib.sha256(json.dumps(x, separators=(',', ':')).encode()).hexdigest()


def run_one_context(system, audit_mode: bool, *, arm: str, j: int, image_index: int,
                    truth: np.ndarray, rng: np.random.Generator,
                    t2_plan: Optional[List[Dict]] = None):
    sm = sample_mask_set(system.c.N, rng); masked = np.asarray(sm.masked_positions, dtype=np.int64)
    E = exposure_subset(masked, j - 1, system.c.N)
    arr = np.asarray(truth, dtype=np.int64).copy(); arr[masked] = core.MASK
    exposure_event = {'context': int(j), 'image_index': int(image_index), 'masked': masked.astype(int).tolist(), 'E': E.astype(int).tolist()}
    if arm == 'T1':
        z, pp, ur = model_exposure_tokens(system, truth, masked, E, j - 1); arr[E] = z
        corr = z == np.asarray(truth)[E]
        exposure_event.update({'tokens': z.astype(int).tolist(), 'gt': np.asarray(truth)[E].astype(int).tolist(),
                               'correct': corr.astype(bool).tolist(), 'uniform': ur.tolist(),
                               'selected_prob': (pp[np.arange(len(E)), z].tolist() if len(E) else [])})
    elif arm == 'T2':
        if t2_plan is None:
            raise ValueError('T2 requires plan')
        plan = t2_plan[j - 1]
        if not np.array_equal(E, np.asarray(plan['E'], dtype=np.int64)):
            raise AssertionError('T2 exposure set mismatch')
        z = np.asarray(plan['tokens'], dtype=np.int64); arr[E] = z
        exposure_event.update({'tokens': z.astype(int).tolist(), 'gt': np.asarray(truth)[E].astype(int).tolist(),
                               'correct': (z == np.asarray(truth)[E]).astype(bool).tolist()})
    elif arm != 'T0':
        raise ValueError(f'unknown arm {arm}')
    mirror_seed = int(rng.integers(1, 2**31 - 1)); core.seed_numba(mirror_seed)
    if audit_mode:
        system.audit_begin_context(arm=arm, context=j, image_index=image_index, truth=truth,
                                   masked=masked, E=E, mirror_seed=mirror_seed)
    ones = np.ones(system.c.V, dtype=np.float64)
    rec = system.learn_mask_batch_arr(arr, masked, truth, ones, rng)
    if audit_mode:
        system.audit_finish_context()
    exposure_event['mirror_seed'] = mirror_seed; exposure_event['nmask'] = int(len(masked))
    return rec, exposure_event


def run_d0_pair(reference, audit, tokens, indices, rng_state, out: Path):
    rr = clone_rng(rng_state); ra = clone_rng(rng_state); checkpoints = {}; stream_rows = []
    for j, image_index in enumerate(indices, start=1):
        truth = np.asarray(tokens[int(image_index)], dtype=np.int64)
        rec_r, ev_r = run_one_context(reference, False, arm='T0', j=j, image_index=int(image_index), truth=truth, rng=rr)
        rec_a, ev_a = run_one_context(audit, True, arm='T0', j=j, image_index=int(image_index), truth=truth, rng=ra)
        stream_equal = (ev_r['masked'] == ev_a['masked'] and ev_r['E'] == ev_a['E'] and
                        ev_r['mirror_seed'] == ev_a['mirror_seed'] and json_state(rr.bit_generator.state) == json_state(ra.bit_generator.state))
        if not stream_equal:
            raise RuntimeError(f'D0 stream divergence at context {j}')
        if abs(float(rec_r['mean_nll']) - float(rec_a['mean_nll'])) != 0.0:
            raise RuntimeError(f'D0 mean_nll divergence at context {j}')
        if j in CHECKPOINTS or j == len(indices):
            cmp = compare_native_states(reference, audit); cmp['context'] = int(j)
            cmp['training_rng_equal'] = bool(json_state(rr.bit_generator.state) == json_state(ra.bit_generator.state))
            cmp['target_order_signature_audit'] = target_order_signature(audit._audit_event_rows, j)
            cmp['pass'] = bool(cmp['bitwise_pass'] and cmp['training_rng_equal']); checkpoints[str(j)] = cmp
            print('D0_CHECKPOINT', json.dumps({'context': j, 'pass': cmp['pass'], 'array_mismatch_count': cmp['array_mismatch_count']}), flush=True)
        stream_rows.append({'context': j, 'image_index': int(image_index), 'mirror_seed': int(ev_r['mirror_seed']), 'nmask': int(ev_r['nmask'])})
    d0_pass = bool(all(v['pass'] for v in checkpoints.values()))
    report = {'gate': 'D0_INSTRUMENTATION_PARITY', 'contexts': int(len(indices)), 'checkpoints': checkpoints,
              'pass': d0_pass, 'requirement': 'instrumented and uninstrumented native state must be bitwise-identical'}
    (out / 'D0_PARITY.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (out / 'D0_STREAM.json').write_text(json.dumps(stream_rows), encoding='utf-8')
    if not d0_pass:
        raise RuntimeError('D0 FAIL: instrumentation changed the native trajectory')
    return report


def run_audit_arm(start_system, arm: str, tokens, indices, rng_state, sketch_dim: int,
                  t2_plan: Optional[List[Dict]] = None):
    s = make_audit_clone(start_system, sketch_dim); rng = clone_rng(rng_state); exposures = []
    for j, image_index in enumerate(indices, start=1):
        truth = np.asarray(tokens[int(image_index)], dtype=np.int64)
        _rec, ev = run_one_context(s, True, arm=arm, j=j, image_index=int(image_index), truth=truth, rng=rng, t2_plan=t2_plan)
        exposures.append(ev)
        if j in CHECKPOINTS or j == len(indices):
            print('AUDIT_PROGRESS', json.dumps({'arm': arm, 'context': j, 'mobility_W2': s.audit_terminal_mobility()['W2']}), flush=True)
    return s, exposures, rng


def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        path.write_text('', encoding='utf-8'); return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction='ignore'); w.writeheader(); w.writerows(rows)


def save_audit_outputs(out: Path, arm: str, s: AuditFullForwardPlasticMaskGITSystem, exposures: List[Dict]):
    write_csv(out / f'{arm}_D1_EVENTS.csv', s._audit_event_rows)
    write_csv(out / f'{arm}_D1_CONTEXTS.csv', s._audit_context_rows)
    (out / f'{arm}_EXPOSURE_EVENTS.json').write_text(json.dumps(exposures), encoding='utf-8')
    (out / f'{arm}_MOBILITY.json').write_text(json.dumps(s.audit_terminal_mobility(), indent=2), encoding='utf-8')
    arrs = {}
    for k in TRACK:
        arrs[f'{k}__norm'] = np.array([r[f'{k}__norm'] for r in s._audit_sketch_rows], dtype=np.float64)
        arrs[f'{k}__sketch'] = np.stack([r[f'{k}__sketch'] for r in s._audit_sketch_rows]).astype(np.float64)
    arrs['context'] = np.array([r['context'] for r in s._audit_sketch_rows], dtype=np.int64)
    np.savez_compressed(out / f'{arm}_D2_SKETCHES.npz', **arrs)
    np.savez_compressed(out / f'{arm}_TERMINAL_WEIGHTS.npz', **{k: np.asarray(getattr(s, k)) for k in TRACK})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True); ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--contexts', type=int, default=1024); ap.add_argument('--encode-batch', type=int, default=256)
    ap.add_argument('--seed', type=int, default=7311); ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--sketch-dim', type=int, default=DEFAULT_SKETCH_DIM)
    ap.add_argument('--skip-t1t2', action='store_true', help='Run D0 + T0 audit only.')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads)
    device = torch.device('cpu')
    self_test = oracle_self_test(); (a.out / 'W2_ORACLE_SELF_TEST.json').write_text(json.dumps(self_test, indent=2), encoding='utf-8')
    if not self_test['pass']:
        raise RuntimeError(f'W2 oracle finite-difference self-test failed: {self_test}')
    source, meta, train_rng, schedule = trainer.load_full_checkpoint(a.checkpoint)
    if abs(float(meta.get('context_epoch_equivalent', -1.0)) - 2.0) > 1e-12:
        raise ValueError('Gate D requires exact 2.0-epoch R6B3 checkpoint')
    offset = int(schedule.get('offset', 0)); perm = np.asarray(schedule['permutation'], dtype=np.int64)
    if offset + a.contexts > len(perm):
        raise ValueError('requested contexts exceed saved epoch-3 schedule')
    indices = perm[offset:offset + a.contexts]; rng_state = copy.deepcopy(train_rng.bit_generator.state)
    xall, yall, _xte, _yte = core.load(a.data); x012, y012 = core.subset012(xall, yall)
    xtr, _ytr, _xva, _yva = core.split(x012, y012, .1, 7)
    tokenizer, tok_meta = base.load_frozen_tokenizer(a.tokenizer, device)
    tr, sp_tr = base.encode_snn(tokenizer, xtr, device, a.encode_batch)
    tok0, pos0, input_audit = build_input_embeddings(tokenizer, source.c.D, grid=7)
    start = pickle.loads(pickle.dumps(source, pickle.HIGHEST_PROTOCOL)); upgrade_r6b3_system(start, tok0, pos0, aux_lr_scale=1.0)
    reference = pickle.loads(pickle.dumps(start, pickle.HIGHEST_PROTOCOL)); audit_t0 = make_audit_clone(start, a.sketch_dim)
    d0 = run_d0_pair(reference, audit_t0, tr, indices, rng_state, a.out); save_audit_outputs(a.out, 'T0', audit_t0, [])
    t2_audit = None
    if not a.skip_t1t2:
        audit_t1, exp1, _rng1 = run_audit_arm(start, 'T1', tr, indices, rng_state, a.sketch_dim)
        plan, t2_audit = make_t2_plan(exp1); (a.out / 'T2_PLAN.json').write_text(json.dumps({'audit': t2_audit, 'plan': plan}), encoding='utf-8')
        save_audit_outputs(a.out, 'T1', audit_t1, exp1)
        audit_t2, exp2, _rng2 = run_audit_arm(start, 'T2', tr, indices, rng_state, a.sketch_dim, t2_plan=plan)
        save_audit_outputs(a.out, 'T2', audit_t2, exp2)
    manifest = {
        'stage': 'GATE_D_NATIVE_PLASTICITY_ORGANIZATION_AUDIT', 'contexts': int(a.contexts), 'seed': int(a.seed),
        'sketch_dim': int(a.sketch_dim), 'source_context_epoch': float(meta['context_epoch_equivalent']),
        'schedule_offset': int(offset), 'D0_pass': bool(d0['pass']), 'T1_T2_run': bool(not a.skip_t1t2),
        'T2_audit': t2_audit, 'gradient_oracle_enters_candidate_learning': False, 'native_rule_changed': False,
        'tokenizer_metadata': tok_meta, 'input_audit': input_audit, 'train_tokenizer_activity': sp_tr,
        'scientific_boundary': ('D0-D2 are diagnostic. They do not establish an optimal plasticity schedule, '
                                'a causal slow manifold, or a biological learning mechanism.'),
    }
    (a.out / 'GATE_D_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('GATE_D_RUN_COMPLETE', json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
