#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D3E evaluation helpers: frozen Gate-A evaluator and secondary endpoints."""
from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np

import p3i_cb_exact_msdl as core
import r6b3_context_epoch_training as trainer
import r6b3_decoder_operating_point_audit as coarse
import r6b3_decoder_spatial_refinement_audit as spatial
from maskgit_input_representation import context_from_arr_unconditional
from r6b3_snn_encoder_maskgit_msdl import generation_tokens_maskgit

MASK = 64
V = 64
DOM = 61
ROUNDS = 3
BASE_GEN_SEED = 27311
GATE_A_TEMP = 8.0
METRIC_NAMES = (
    'ce','acc','ce61','acc61','ce_non61','acc_non61','mean_p61',
    'p61_on_gt61','p61_on_non61','proposal61_fraction','context61_fraction',
    'sampled_accuracy','sampled_confidence','brier')
IX = {k:i for i,k in enumerate(METRIC_NAMES)}


def bootstrap_mean(x, seed, nboot=10000):
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {'n':0, 'mean':None, 'ci95':[None,None]}
    rng = np.random.default_rng(seed)
    v = np.empty(nboot)
    for b in range(nboot):
        v[b] = a[rng.integers(0, len(a), size=len(a))].mean()
    q = np.quantile(v, [.025,.975])
    return {
        'n': int(len(a)), 'mean': float(a.mean()),
        'ci95': [float(q[0]), float(q[1])],
        'fraction_positive': float(np.mean(a > 0)),
        'fraction_negative': float(np.mean(a < 0)),
    }


def model_dict(s):
    return trainer.model_dict_from_system(s)


def inverse_cdf_rows(p, u):
    c = np.cumsum(np.asarray(p, float), axis=1)
    c[:, -1] = 1.0
    return np.minimum(np.sum(c < np.asarray(u, float)[:, None], axis=1), p.shape[1]-1).astype(np.int64)


def predict_round(pred, cur, tok, pos, masked):
    xn, xraw = context_from_arr_unconditional(np.asarray(cur, int), tok, pos)
    pred.rebuild(xn)
    p = np.empty((len(masked), V), float)
    for qi, t0 in enumerate(np.asarray(masked, int)):
        t = int(t0)
        a, q, num, den = pred._read(t)
        _u, _h, _rf, pp = pred._tail(a, np.asarray(xraw[t], float))
        p[qi] = pp
    p /= p.sum(1, keepdims=True)
    return p


def one_metrics(p, gt, sampled, cur):
    p = np.asarray(p, float)
    gt = np.asarray(gt, int)
    sampled = np.asarray(sampled, int)
    truep = p[np.arange(len(gt)), gt]
    ce = -np.log(np.clip(truep, 1e-12, 1))
    arg = np.argmax(p, 1)
    acc = arg == gt
    is61 = gt == DOM
    non = ~is61
    sconf = p[np.arange(len(sampled)), sampled]
    scorr = sampled == gt
    one = np.zeros_like(p)
    one[np.arange(len(gt)), gt] = 1.0
    vis = np.asarray(cur) != MASK
    def m(x):
        return float(np.mean(x)) if np.size(x) else np.nan
    return np.asarray([
        m(ce), m(acc), m(ce[is61]), m(acc[is61]), m(ce[non]), m(acc[non]),
        m(p[:,DOM]), m(p[is61,DOM]), m(p[non,DOM]), m(sampled == DOM),
        m(np.asarray(cur)[vis] == DOM), m(scorr), m(sconf),
        m(np.sum((p-one)**2, 1))], float)


def effect_arrays(sm, tm, r=2):
    s = np.asarray(sm)[:,r]
    t = np.asarray(tm)[:,r]
    return {
        'CE': s[:,IX['ce']] - t[:,IX['ce']],
        'D_non61': s[:,IX['ce_non61']] - t[:,IX['ce_non61']],
        'F61': s[:,IX['p61_on_non61']] - t[:,IX['p61_on_non61']],
        'C61': s[:,IX['context61_fraction']] - t[:,IX['context61_fraction']],
        'delta_P61': s[:,IX['mean_p61']] - t[:,IX['mean_p61']],
    }


def eval_gate_a(name, cfg, m, tok, pos, schedules, gt, shared_u):
    n = len(gt)
    sm = np.full((n, ROUNDS, len(METRIC_NAMES)), np.nan)
    tm = np.full_like(sm, np.nan)
    sp = np.full((n, ROUNDS, cfg.N), -1, np.int16)
    tp = np.full_like(sp, -1)
    gates = {
        'round1_probability_bitwise_parity': True,
        'round1_sample_exact_parity': True,
        'committed_position_parity': True,
        'intervention_purity': True,
        'common_random_number_integrity': True,
        'image_count_after_three_rounds_exact': True,
    }
    for i in range(n):
        ps = core.P3ImageSystem(cfg, m, BASE_GEN_SEED + 100000 + i)
        pt = core.P3ImageSystem(cfg, m, BASE_GEN_SEED + 100000 + i)
        cs = np.full(cfg.N, MASK, int)
        ct = np.full(cfg.N, MASK, int)
        for r in range(ROUNDS):
            tr = schedules[i][r]
            masked = np.asarray(tr['masked_positions_before'], int)
            remask = np.asarray(tr['remasked_positions'], int)
            if (not np.array_equal(np.flatnonzero(cs == MASK), masked)
                    or not np.array_equal(np.flatnonzero(ct == MASK), masked)):
                gates['committed_position_parity'] = False
            if r == 0:
                if not np.array_equal(cs, ct) or not np.all(cs == MASK):
                    gates['intervention_purity'] = False
            else:
                if not np.array_equal(cs == MASK, ct == MASK):
                    gates['intervention_purity'] = False
                vis = np.flatnonzero(ct != MASK)
                if len(vis) and not np.array_equal(ct[vis], gt[i,vis]):
                    gates['intervention_purity'] = False
                if np.any((cs != ct) & (ct == MASK)):
                    gates['intervention_purity'] = False
            p1 = predict_round(ps, cs, tok, pos, masked)
            p2 = predict_round(pt, ct, tok, pos, masked)
            u = np.asarray(shared_u[i,r,masked], float)
            if not np.array_equal(u, shared_u[i,r,masked]):
                gates['common_random_number_integrity'] = False
            z1 = inverse_cdf_rows(p1, u)
            z2 = inverse_cdf_rows(p2, u)
            if r == 0 and not np.array_equal(p1, p2):
                gates['round1_probability_bitwise_parity'] = False
            if r == 0 and not np.array_equal(z1, z2):
                gates['round1_sample_exact_parity'] = False
            sm[i,r] = one_metrics(p1, gt[i,masked], z1, cs)
            tm[i,r] = one_metrics(p2, gt[i,masked], z2, ct)
            pr1 = cs.copy(); pr1[masked] = z1
            pr2 = ct.copy(); pr2[masked] = z2
            sp[i,r] = pr1; tp[i,r] = pr2
            ns = pr1.copy(); ns[remask] = MASK
            nt = ct.copy()
            new = np.setdiff1d(masked, remask, assume_unique=True)
            nt[new] = gt[i,new]
            nt[remask] = MASK
            old = np.flatnonzero(ct != MASK)
            if len(old):
                nt[old] = gt[i,old]
            cs, ct = ns, nt
        if ps.image_count != ROUNDS or pt.image_count != ROUNDS:
            gates['image_count_after_three_rounds_exact'] = False
    gates['all_pass'] = bool(all(gates.values()))
    effects = effect_arrays(sm, tm, 2)
    summary = {k:bootstrap_mean(v, 710000 + 100*i) for i,(k,v) in enumerate(effects.items())}
    return {
        'name': name, 'self_metrics': sm, 'teacher_metrics': tm,
        'self_proposal': sp, 'teacher_proposal': tp,
        'effects': effects, 'summary': summary, 'gates': gates,
    }


def eval_masked(s, tokens, seed, n):
    return {str(r):trainer.eval_masked_unconditional(
        model_dict(s), s.c, tokens, s.tok, s.pos, r,
        seed + int(round(1000*r)), n) for r in (.25,.5,.9,1.0)}


def rollout_js(s, real_p, real_pos_p, real_pair, seed, samples):
    ids,traces,audits = generation_tokens_maskgit(
        model_dict(s), s.c, samples, s.tok, s.pos, seed,
        num_iter=3, choice_temperature=8.0)
    gl = coarse.distribution_metrics(ids,real_p,s.c.V)
    sp = spatial.positionwise_metrics(ids,real_pos_p,s.c.V,s.c.N)
    pa = spatial.pair_metrics(ids,real_pair,s.c.V,s.c.N,7)
    return {
        'global_js_nats':float(gl['js_to_real_nats']),
        'spatial_js_nats':float(sp['mean_js_nats']),
        'pair_js_nats':float(pa['mean_orientation_js_nats']),
        'all_trace_audits_pass':bool(all(x['pass'] for x in audits)),
    }


def gate_for_system(name, s, schedules, gt, shared):
    return eval_gate_a(name, s.c, model_dict(s), np.asarray(s.tok), np.asarray(s.pos),
                       schedules, gt, shared)


def transplant(native, fem, which):
    s = pickle.loads(pickle.dumps(native,pickle.HIGHEST_PROTOCOL))
    if which in ('W2','W2_b2'):
        s.W2 = np.array(fem.W2,copy=True)
    if which in ('b2','W2_b2'):
        s.b2 = np.array(fem.b2,copy=True)
    return s


def rescue(native_eval, other_eval, seed0):
    return {k:bootstrap_mean(
        np.asarray(native_eval['effects'][k]) - np.asarray(other_eval['effects'][k]),
        seed0+i) for i,k in enumerate(('F61','D_non61','C61','CE','delta_P61'))}


def write_csv(path: Path, rows):
    if not rows:
        path.write_text('',encoding='utf-8'); return
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
