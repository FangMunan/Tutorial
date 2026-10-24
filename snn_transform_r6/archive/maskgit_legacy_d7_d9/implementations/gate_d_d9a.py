#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D9A — trajectory-level rank-1 functional regulation upper bound.

Four synchronized 1024-context trajectories from the exact D6/D7 start state:
  F  = historical F_EM reference
  B  = historical combined failure learner (WO zero + norm-matched exact bo oracle)
  Q1 = B + hard one-sided rank-1 y-space regulation during training only
  Q2 = B + fixed slow scalar rank-1 regulator during training only

All terminal deployment evaluations run with the regulator OFF.  D9A therefore
asks whether controlling the D8B causal coordinate during learning reshapes the
learner, rather than reusing the D8B inference-time intervention.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch

import gate_d_d3e_runtime as d3r
import gate_d_d6a as d6a
import gate_d_d6a1 as d6a1
import gate_d_d7a as d7a
import gate_d_d7a1 as d71
import gate_d_d8a as d8a
import gate_d_plasticity_audit as gd
import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
import r6b3_context_epoch_training as trainer
import r6b3_decoder_operating_point_audit as coarse
import r6b3_decoder_spatial_refinement_audit as spatial
from gate_d_d3e_eval import BASE_GEN_SEED, eval_masked, rollout_js
from maskgit_input_representation import build_input_embeddings
from r6b4_full_forward_plasticity import upgrade_r6b3_system

CHECKPOINTS = (64, 128, 256, 512, 768, 1024)
PRIMARY = ('F61', 'D_non61')
ALL_METRICS = ('F61', 'D_non61', 'C61', 'delta_P61', 'CE')
MASK_RATIOS = ('0.25', '0.5', '0.9', '1.0')
JS_KEYS = ('global_js_nats', 'spatial_js_nats', 'pair_js_nats')
F_LABEL = 'F_EM'
B_LABEL = 'F_EM_WO_zero_bo_oracle'
RHO = 0.92
KAPPA = 1.0
EPS = 1e-30
TOL = 1e-7


def clone(x):
    return pickle.loads(pickle.dumps(x, pickle.HIGHEST_PROTOCOL))


def locate(root: Path, name: str) -> Path:
    return d3r.locate(root, name)


def write_csv(path: Path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def prob_from_y(system, y):
    y = np.asarray(y, dtype=np.float64)
    rms = math.sqrt(float(np.mean(y * y))) + 1e-8
    rf = np.empty(system.c.D + 1, dtype=np.float64)
    rf[:-1] = 0.5 * y / rms
    rf[-1] = 0.5
    p = np.asarray(core.softmax_numba(np.asarray(system.WR, dtype=np.float64) @ rf), dtype=np.float64)
    return rf, p


def grad_loss_y(system, y, p, target):
    """Exact d(CE)/dy for the normalized readout used by the frozen learner."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    one = np.zeros(system.c.V, dtype=np.float64); one[int(target)] = 1.0
    q = (np.asarray(system.WR, dtype=np.float64).T @ (p - one))[:-1]
    s = math.sqrt(float(np.mean(y * y))); r = s + 1e-8
    if s <= EPS:
        return 0.5 * q / r
    return 0.5 * q / r - 0.5 * float(np.dot(q, y)) * y / (system.c.D * s * r * r)


def corrected_O_descent(system, rec, target):
    """Exact current-context descent wrt WO/bo under the D9 regulator.

    Q1 includes J_reg = I-vv^T when the one-sided clamp is active.  Q2 uses the
    previous-context controller state, so its current correction is parameter
    independent and J_reg=I for the current context.
    """
    yreg = np.asarray(rec['d9_y_reg'], dtype=np.float64)
    p = np.asarray(rec['p'], dtype=np.float64)
    grad_y = grad_loss_y(system, yreg, p, int(target))
    if system._d9_mode == 'Q1' and bool(rec['d9_projection_active']):
        v = np.asarray(system._d9_v, dtype=np.float64)
        grad_y = grad_y - v * float(np.dot(v, grad_y))

    t = int(rec['t']); a = np.asarray(rec['a'], dtype=np.float64)
    u = np.asarray(rec['u'], dtype=np.float64)
    zhat = np.asarray(rec['zhat'], dtype=np.float64)
    z = np.asarray(system._current_xraw[t], dtype=np.float64) + system.bo + system.WO @ a
    mu = float(np.mean(z))
    inv = 1.0 / math.sqrt(float(np.mean((z - mu) ** 2)) + float(system.eps2))
    pre = np.asarray(system.W1, dtype=np.float64) @ u + np.asarray(system.b1, dtype=np.float64) - np.asarray(system.theta, dtype=np.float64)
    grad_h = np.asarray(system.W2, dtype=np.float64).T @ grad_y
    grad_pre = grad_h * (pre > 0)
    grad_u = np.asarray(system.W1, dtype=np.float64).T @ grad_pre
    grad_zh = np.asarray(system.gamma2, dtype=np.float64) * grad_u
    grad_z = grad_y + inv * (grad_zh - np.mean(grad_zh) - zhat * np.mean(grad_zh * zhat))
    gz_desc = -grad_z
    return np.outer(gz_desc, a) * system.maskO, np.asarray(gz_desc, dtype=np.float64)


def projected_oracle_self_test(seed=1902601):
    """Finite-difference test of the Q1 projection-aware bo gradient."""
    rng = np.random.default_rng(seed)
    D, FF, V = 13, 19, 11
    x = rng.normal(size=D); a = rng.normal(size=D)
    WO = rng.normal(scale=.15, size=(D, D)); W1 = rng.normal(scale=.17, size=(FF, D)); W2 = rng.normal(scale=.15, size=(D, FF))
    bo = rng.normal(scale=.08, size=D); b1 = rng.normal(scale=.08, size=FF); b2 = rng.normal(scale=.08, size=D)
    gamma = 1.0 + rng.normal(scale=.04, size=D); beta = rng.normal(scale=.04, size=D); theta = rng.normal(scale=.03, size=FF)
    WR = rng.normal(scale=.15, size=(V, D + 1)); v = rng.normal(size=D); v /= np.linalg.norm(v)
    target = int(rng.integers(V)); eps2 = 1e-5

    def forward(bov):
        z = x + bov + WO @ a
        mu = float(np.mean(z)); inv = 1.0 / math.sqrt(float(np.mean((z - mu) ** 2)) + eps2)
        zh = (z - mu) * inv; u = zh * gamma + beta
        pre = W1 @ u + b1 - theta; h = np.maximum(0.0, pre)
        y = z + W2 @ h + b2
        return z, zh, u, pre, h, y

    # Move away from ReLU kinks.
    for _ in range(30):
        _z, _zh, _u, pre, _h, _y = forward(bo)
        if float(np.min(np.abs(pre))) > 2e-3: break
        b1 += 0.02

    def loss(bov, qref):
        _z, _zh, _u, _pre, _h, y = forward(bov)
        e = max(0.0, float(np.dot(v, y)) - qref)
        return gd._loss_from_y(y - e * v, WR, target)

    out = {}
    for name, margin in (('active', -0.35), ('inactive', +0.35)):
        z, zh, u, pre, h, y = forward(bo)
        q0 = float(np.dot(v, y)); qref = q0 + margin
        e = max(0.0, q0 - qref); yreg = y - e * v
        rms = math.sqrt(float(np.mean(yreg * yreg))) + 1e-8
        rf = np.empty(D + 1); rf[:-1] = .5 * yreg / rms; rf[-1] = .5
        logits = WR @ rf; logits -= np.max(logits); p = np.exp(logits); p /= np.sum(p)
        one = np.zeros(V); one[target] = 1.0
        qq = (WR.T @ (p - one))[:-1]
        ss = math.sqrt(float(np.mean(yreg * yreg))); rr = ss + 1e-8
        gy = .5 * qq / rr if ss <= EPS else .5 * qq / rr - .5 * float(np.dot(qq, yreg)) * yreg / (D * ss * rr * rr)
        if e > 0: gy = gy - v * float(np.dot(v, gy))
        gh = W2.T @ gy; gp = gh * (pre > 0); gu = W1.T @ gp; gzh = gamma * gu
        inv = 1.0 / math.sqrt(float(np.mean((z - np.mean(z)) ** 2)) + eps2)
        gz = gy + inv * (gzh - np.mean(gzh) - zh * np.mean(gzh * zh))
        dbo = rng.normal(size=D); ee = 1e-6
        fd = (loss(bo + ee * dbo, qref) - loss(bo - ee * dbo, qref)) / (2 * ee)
        an = float(np.dot(gz, dbo))
        rel = abs(fd - an) / max(1.0, abs(fd), abs(an))
        out[name] = {'fd': float(fd), 'analytic': an, 'relative_error': float(rel), 'projection_active': bool(e > 0)}
    out['pass'] = bool(max(out[k]['relative_error'] for k in ('active', 'inactive')) < 2e-7)
    return out


class D9ASystem(d6a.D6ASystem):
    def d9a_init(self, mode: str, v, qref: float):
        if mode not in ('Q1', 'Q2'): raise ValueError(mode)
        self.d6a_init(B_LABEL)
        vv = np.asarray(v, dtype=np.float64).reshape(-1)
        if vv.shape != (self.c.D,): raise ValueError('D9A basis shape mismatch')
        vv = vv / (np.linalg.norm(vv) + 1e-30)
        self._d9_mode = mode
        self._d9_v = vv
        self._d9_qref = float(qref)
        self._d9_rho = float(RHO); self._d9_kappa = float(KAPPA)
        self._d9_m = float(qref)
        self._d9_reg_rows = []
        self._d9_cum_work = 0.0
        self._d9_cum_pressure = 0.0
        self._d9_active_events = 0
        self._d9_total_events = 0
        self._d9_max_correction = 0.0

    def prepare_mask_batch(self, xn, xraw, masked_positions):
        cache = super().prepare_mask_batch(xn, xraw, masked_positions)
        qraw = []
        yraws = []
        for rec in cache:
            t = int(rec['t']); a = np.asarray(rec['a'], dtype=np.float64)
            z = np.asarray(self._current_xraw[t], dtype=np.float64) + self.bo + self.WO @ a
            y = z + self.W2 @ np.asarray(rec['h'], dtype=np.float64) + self.b2
            yraws.append(np.asarray(y, dtype=np.float64))
            qraw.append(float(np.dot(self._d9_v, y)))
        qraw = np.asarray(qraw, dtype=np.float64)
        m_before = float(self._d9_m)
        if self._d9_mode == 'Q2':
            shared_corr = self._d9_kappa * max(0.0, m_before - self._d9_qref)
        else:
            shared_corr = None

        corrections = []
        for rec, y, q in zip(cache, yraws, qraw):
            corr = max(0.0, q - self._d9_qref) if self._d9_mode == 'Q1' else float(shared_corr)
            yreg = y - corr * self._d9_v
            rf, p = prob_from_y(self, yreg)
            rec['rf'] = rf; rec['p'] = p
            rec['d9_y_raw'] = y.copy(); rec['d9_y_reg'] = yreg.copy()
            rec['d9_q_raw'] = float(q); rec['d9_correction'] = float(corr)
            rec['d9_projection_active'] = bool(self._d9_mode == 'Q1' and corr > 0.0)
            corrections.append(float(corr))

        qbar = float(np.mean(qraw)) if len(qraw) else m_before
        if self._d9_mode == 'Q2':
            self._d9_m = self._d9_rho * m_before + (1.0 - self._d9_rho) * qbar
        m_after = float(self._d9_m)
        corrections = np.asarray(corrections, dtype=np.float64)
        pressure = np.maximum(qraw - self._d9_qref, 0.0)
        work = float(np.sum(np.abs(corrections)))
        press = float(np.sum(pressure))
        active = int(np.sum(corrections > 0.0))
        self._d9_cum_work += work; self._d9_cum_pressure += press
        self._d9_active_events += active; self._d9_total_events += int(len(corrections))
        if len(corrections): self._d9_max_correction = max(self._d9_max_correction, float(np.max(corrections)))
        ctx = None if self._d5b_ctx is None else int(self._d5b_ctx['context'])
        self._d9_reg_rows.append({
            'context': ctx, 'arm': self._d9_mode, 'nmask': int(len(cache)),
            'q_raw_mean': qbar, 'q_raw_min': None if not len(qraw) else float(np.min(qraw)),
            'q_raw_max': None if not len(qraw) else float(np.max(qraw)),
            'native_harmful_pressure': press, 'context_regulator_work': work,
            'active_events': active, 'active_fraction': 0.0 if not len(cache) else float(active / len(cache)),
            'mean_correction': 0.0 if not len(corrections) else float(np.mean(corrections)),
            'max_correction': 0.0 if not len(corrections) else float(np.max(corrections)),
            'm_before': m_before, 'm_after': m_after,
            'cumulative_regulator_work': float(self._d9_cum_work),
            'cumulative_native_harmful_pressure': float(self._d9_cum_pressure),
        })
        return cache

    def accumulate_cached_target(self, rec, target, weight=1.0):
        # D5B adds its native oracle contribution. Replace only the O-oracle
        # contribution with the regulator-aware exact current-context result.
        before_bo = np.array(self._d5b_Gbo, copy=True)
        before_wo = np.array(self._d5b_GWO, copy=True)
        p = super().accumulate_cached_target(rec, target, weight)
        GO, Gbo = corrected_O_descent(self, rec, int(target))
        self._d5b_Gbo = before_bo + Gbo
        self._d5b_GWO = before_wo + GO
        return p

    def apply_update(self):
        super().apply_update()
        row = self._d5b_rows[-1]
        row['d9a_arm'] = self._d9_mode
        row['d9a_regulator_work_cumulative'] = float(self._d9_cum_work)
        row['d9a_native_pressure_cumulative'] = float(self._d9_cum_pressure)
        row['d9a_d5b_loss_probe_boundary'] = 'D5B finite-step loss probe is native-tail only and is not used to validate D9 regulator-aware oracle'


def make_d9a(start, mode, v, qref):
    s = clone(start); s.__class__ = D9ASystem; s.d9a_init(mode, v, qref); return s


def discover_q_reference(R, va, yva, validation_ids, d8_modes, v):
    stored_ids = np.asarray(d8_modes['discover_ids'], dtype=np.int64)
    calc_ids, calc_split = d8a.select_discovery_ids(yva, validation_ids, 256, d8a.DISCOVERY_SEED)
    if not np.array_equal(stored_ids, calc_ids):
        raise RuntimeError('D9A cannot reproduce frozen D8A discovery IDs')
    if np.intersect1d(stored_ids, validation_ids).size:
        raise RuntimeError('D9A discovery/Gate-A overlap')
    vals = []
    for did in stored_ids:
        gt = np.asarray(va[int(did)], dtype=np.int64)
        for _depth, _vis, cur, masked in d8a.teacher_contexts(gt, int(did), R.c.N, R.c.V):
            acts, terr = d8a.forward_probe(R, cur, masked, do_tail_check=False)
            if terr != 0.0: raise RuntimeError('unexpected D8A probe tail error')
            vals.append(np.asarray(acts['y'], dtype=np.float64) @ np.asarray(v, dtype=np.float64))
    q = np.concatenate(vals).astype(np.float64)
    return {
        'qref': float(np.mean(q)), 'std': float(np.std(q)), 'n': int(len(q)),
        'q01': float(np.quantile(q, .01)), 'q05': float(np.quantile(q, .05)),
        'median': float(np.median(q)), 'q95': float(np.quantile(q, .95)), 'q99': float(np.quantile(q, .99)),
        'min': float(np.min(q)), 'max': float(np.max(q)),
        'discover_ids_exact': True, 'GateA_overlap_count': 0,
        'split_exact': bool(np.array_equal(calc_split, np.asarray(d8_modes['discover_split'], dtype=np.int8))),
    }


def run_four(start, tokens, indices, rng_state, v, qref):
    systems = {
        'F': d6a.make_d6a(start, F_LABEL),
        'B': d6a.make_d6a(start, B_LABEL),
        'Q1': make_d9a(start, 'Q1', v, qref),
        'Q2': make_d9a(start, 'Q2', v, qref),
    }
    rngs = {k: gd.clone_rng(rng_state) for k in systems}
    streams = {k: [] for k in systems}
    trace = []
    for j, image_index in enumerate(indices, start=1):
        records = {}; events = {}
        truth = np.asarray(tokens[int(image_index)], dtype=np.int64)
        for label, s in systems.items():
            before = d3r.jstate(rngs[label]); d3r.reset_boundary(s, 'F_EM')
            if d3r.jstate(rngs[label]) != before: raise RuntimeError('D9A reset consumed RNG')
            rec, ev = d6a.run_context(s, j, int(image_index), truth, rngs[label])
            records[label] = rec; events[label] = ev
            streams[label].append(d6a.stream_row(j, int(image_index), ev))
        ref = streams['F'][-1]
        if not all(streams[k][-1] == ref for k in systems):
            raise RuntimeError({'D9A_stream_divergence': {'context': j}})
        states = [d3r.jstate(rngs[k]) for k in systems]
        if not all(x == states[0] for x in states[1:]):
            raise RuntimeError({'D9A_external_rng_divergence': {'context': j}})
        if j in CHECKPOINTS:
            row = {
                'context': int(j),
                'F_mean_nll': float(records['F']['mean_nll']),
                'B_mean_nll': float(records['B']['mean_nll']),
                'Q1_mean_nll': float(records['Q1']['mean_nll']),
                'Q2_mean_nll': float(records['Q2']['mean_nll']),
            }
            for label in ('Q1', 'Q2'):
                s = systems[label]
                row[f'{label}_cum_work'] = float(s._d9_cum_work)
                row[f'{label}_cum_pressure'] = float(s._d9_cum_pressure)
                row[f'{label}_active_fraction'] = float(s._d9_active_events / max(1, s._d9_total_events))
                row[f'{label}_max_correction'] = float(s._d9_max_correction)
                row[f'{label}_m'] = float(s._d9_m)
            trace.append(row)
            print('D9A_PROGRESS', json.dumps(row), flush=True)
    return {'systems': systems, 'streams': streams,
            'rng_after': {k: d3r.jstate(rngs[k]) for k in systems}, 'trace': trace}


def bootstrap_ratio(num, den, seed, nboot):
    num = np.asarray(num, dtype=np.float64); den = np.asarray(den, dtype=np.float64)
    point_den = float(np.mean(den)); point = None if point_den <= EPS else float(np.mean(num) / point_den)
    rng = np.random.default_rng(seed); vals = np.full(nboot, np.nan)
    n = len(num)
    for b in range(nboot):
        ii = rng.integers(0, n, size=n); d = float(np.mean(den[ii]))
        if d > EPS: vals[b] = float(np.mean(num[ii]) / d)
    good = vals[np.isfinite(vals)]
    return {'point_estimate': point,
            'ci95': [None, None] if not len(good) else [float(x) for x in np.quantile(good, [.025, .975])],
            'valid_bootstrap_fraction': float(len(good) / nboot)}


def arm_guardrails(masked, rollout, arm):
    mg = {}; mg_pass = True
    for r in MASK_RATIOS:
        best = min(float(masked['F'][r]['ce']), float(masked['B'][r]['ce']))
        val = float(masked[arm][r]['ce']); lim = best * 1.01 + 1e-8
        mg[r] = {'value': val, 'best_F_or_B': best, 'limit': lim, 'pass': bool(val <= lim)}
        mg_pass = mg_pass and mg[r]['pass']
    jg = {}; jg_pass = True
    for k in JS_KEYS:
        best = min(float(rollout['F'][k]), float(rollout['B'][k]))
        val = float(rollout[arm][k]); lim = best * 1.02 + 1e-6
        jg[k] = {'value': val, 'best_F_or_B': best, 'limit': lim, 'pass': bool(val <= lim)}
        jg_pass = jg_pass and jg[k]['pass']
    return {'masked_ce': mg, 'masked_ce_all_pass': bool(mg_pass),
            'rollout_js': jg, 'rollout_js_all_pass': bool(jg_pass),
            'all_pass': bool(mg_pass and jg_pass)}


def regulator_summary(s):
    return {
        'mode': s._d9_mode, 'qref': float(s._d9_qref), 'rho': float(s._d9_rho), 'kappa': float(s._d9_kappa),
        'cumulative_regulator_work': float(s._d9_cum_work),
        'cumulative_native_harmful_pressure': float(s._d9_cum_pressure),
        'active_events': int(s._d9_active_events), 'total_events': int(s._d9_total_events),
        'active_fraction': float(s._d9_active_events / max(1, s._d9_total_events)),
        'mean_correction_per_event': float(s._d9_cum_work / max(1, s._d9_total_events)),
        'max_correction': float(s._d9_max_correction), 'final_m': float(s._d9_m),
    }


def save_terminal(path, s, arm):
    with path.open('wb') as f: pickle.dump({'arm': arm, 'system': s}, f, pickle.HIGHEST_PROTOCOL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--d6a-artifact', type=Path, required=True)
    ap.add_argument('--d7a-artifact', type=Path, required=True)
    ap.add_argument('--d8a-artifact', type=Path, required=True)
    ap.add_argument('--d8b-artifact', type=Path, required=True)
    ap.add_argument('--anchor-artifact', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--contexts', type=int, default=1024)
    ap.add_argument('--eval-images', type=int, default=256)
    ap.add_argument('--gen-samples', type=int, default=256)
    ap.add_argument('--encode-batch', type=int, default=256)
    ap.add_argument('--bootstrap', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=7311)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if a.contexts != 1024 or a.eval_images != 256: raise ValueError('formal D9A requires 1024 contexts and 256 Gate-A images')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads)
    device = torch.device('cpu')

    # Frozen D8 prerequisites.
    s8a = json.loads(locate(a.d8a_artifact, 'D8A_SUMMARY.json').read_text(encoding='utf-8'))
    s8b = json.loads(locate(a.d8b_artifact, 'D8B_SUMMARY.json').read_text(encoding='utf-8'))
    if s8a.get('classification') != 'I1_SHARED_LOW_DIMENSIONAL_FUNCTIONAL_CANDIDATE' or s8a.get('primary_y_gate', {}).get('selected_k') != 1:
        raise RuntimeError('D9A prerequisite D8A is not frozen I1 k=1')
    if s8b.get('classification') != 'B1_BIDIRECTIONAL_CAUSAL_HARMFUL_FUNCTIONAL_MODE':
        raise RuntimeError('D9A prerequisite D8B is not frozen B1')
    if not s8b.get('decision', {}).get('discovery_fixed_amplitude_bidirectional', False):
        raise RuntimeError('D9A requires frozen D8B discovery-fixed bidirectional support')
    if float(s8b.get('discovery_fixed_amplitude', 0.0)) <= 0.0:
        raise RuntimeError('D9A frozen D8B orientation is not positive-harm')

    d8m = np.load(locate(a.d8a_artifact, 'D8A_MODES.npz'), allow_pickle=False)
    basis = np.asarray(d8m['D8B_frozen_y_pair_basis'], dtype=np.float64)
    if basis.shape[1] != 1: raise RuntimeError('D9A frozen basis is not rank-1')
    v = basis[:, 0]; v /= np.linalg.norm(v)

    oracle_self = projected_oracle_self_test()
    if not oracle_self['pass']: raise RuntimeError({'D9A_projection_oracle_self_test_failed': oracle_self})

    # Exact 2.0-epoch start and frozen stream.
    checkpoint_sha = d3r.sha_file(a.checkpoint)
    source, meta, train_rng, schedule = trainer.load_full_checkpoint(a.checkpoint)
    if checkpoint_sha != d6a.EXPECTED_CHECKPOINT_SHA256 or abs(float(meta.get('context_epoch_equivalent', -1.0)) - 2.0) > 1e-12:
        raise RuntimeError('D9A start checkpoint identity failed')
    offset = int(schedule['offset'])
    indices = np.asarray(schedule['permutation'], dtype=np.int64)[offset:offset + a.contexts]
    rng_state = copy.deepcopy(train_rng.bit_generator.state)
    if len(indices) != 1024: raise RuntimeError('D9A checkpoint schedule lacks 1024 contexts')

    # Data/tokenizer coordinates identical to D6-D8.
    xall, yall, _xte, _yte = core.load(a.data); x012, y012 = core.subset012(xall, yall)
    xtr, _ytr, xva, yva = core.split(x012, y012, .1, 7)
    toknet, _ = base.load_frozen_tokenizer(a.tokenizer, device)
    tr, _ = base.encode_snn(toknet, xtr, device, a.encode_batch)
    va, _ = base.encode_snn(toknet, xva, device, a.encode_batch)
    va = np.asarray(va, dtype=np.int64); yva = np.asarray(yva, dtype=np.int64)
    tok0, pos0, _ = build_input_embeddings(toknet, source.c.D, grid=7)
    start = clone(source); upgrade_r6b3_system(start, tok0, pos0, aux_lr_scale=1.0)

    stored6 = np.load(locate(a.d6a_artifact, 'D6A_EFFECT_ARRAYS.npz'), allow_pickle=False)
    schedules = json.loads(locate(a.anchor_artifact, 'HISTORICAL_R6B3_R3_T8_REFERENCE_TRACE.json').read_text(encoding='utf-8'))
    validation_ids = np.asarray(stored6['validation_ids'], dtype=np.int64)
    gt = np.asarray(stored6['gt_tokens'], dtype=np.int64); shared = np.asarray(stored6['shared_uniform'], dtype=np.float64)
    if not np.array_equal(va[validation_ids], gt): raise RuntimeError('D9A Gate-A token coordinate mismatch')

    # Privileged discovery-derived safe q reference from the exact D8A rescued R.
    oldB = d6a1.load_d6a_terminal(locate(a.d6a_artifact, 'D6A_F_EM_WO_zero_bo_oracle_terminal.pkl'))
    Fsafe = d71.load_terminal(locate(a.d7a_artifact, 'D7A_F_terminal.pkl'))
    Rterm = d71.hybrid(oldB, Fsafe, ('W2',))
    qref_stats = discover_q_reference(Rterm, va, yva, validation_ids, d8m, v)
    qref = float(qref_stats['qref'])

    run = run_four(start, tr, indices, rng_state, v, qref)
    systems = run['systems']; F, B, Q1, Q2 = systems['F'], systems['B'], systems['Q1'], systems['Q2']
    if not (run['streams']['F'] == run['streams']['B'] == run['streams']['Q1'] == run['streams']['Q2']):
        raise RuntimeError('D9A all-arm stream exactness failed')
    rs = list(run['rng_after'].values())
    if not all(x == rs[0] for x in rs[1:]): raise RuntimeError('D9A all-arm external RNG exactness failed')

    # F/B replay must reproduce historical D6A before interpreting Q arms.
    oldF = d6a1.load_d6a_terminal(locate(a.d6a_artifact, 'D6A_F_EM_terminal.pkl'))
    model_replay = {'F': d7a.model_field_compare_tolerant(F, oldF), 'B': d7a.model_field_compare_tolerant(B, oldB)}
    if not all(x['pass'] for x in model_replay.values()): raise RuntimeError({'D9A_D6A_terminal_replay_failed': model_replay})

    # Regulator is OFF for all terminal deployment evaluations below.
    gate = {k: d6a.gate_eval('D9A_' + k, s, schedules, gt, shared) for k, s in systems.items()}
    calibration = {
        'F_to_D6A': d7a.gate_calibration(gate['F'], stored6, F_LABEL),
        'B_to_D6A': d7a.gate_calibration(gate['B'], stored6, B_LABEL),
    }
    if not all(x['pass'] for x in calibration.values()): raise RuntimeError({'D9A_D6A_GateA_replay_failed': calibration})

    contrasts = {'Q1_minus_B': {}, 'Q2_minus_B': {}, 'B_minus_F': {}}
    for mi, m in enumerate(ALL_METRICS):
        contrasts['Q1_minus_B'][m] = d7a.contrast(gate['Q1'], gate['B'], m, 1910000 + mi, a.bootstrap)
        contrasts['Q2_minus_B'][m] = d7a.contrast(gate['Q2'], gate['B'], m, 1920000 + mi, a.bootstrap)
        contrasts['B_minus_F'][m] = d7a.contrast(gate['B'], gate['F'], m, 1930000 + mi, a.bootstrap)

    gap = {'Q1': {}, 'Q2': {}}
    for ai, arm in enumerate(('Q1', 'Q2')):
        for mi, m in enumerate(ALL_METRICS):
            load = np.asarray(gate['B']['effects'][m], float) - np.asarray(gate['F']['effects'][m], float)
            rescue = np.asarray(gate['B']['effects'][m], float) - np.asarray(gate[arm]['effects'][m], float)
            gap[arm][m] = bootstrap_ratio(rescue, load, 1940000 + ai * 100 + mi, a.bootstrap)
            gap[arm][m].update({'F_mean': d7a.mean_effect(gate['F'], m), 'B_mean': d7a.mean_effect(gate['B'], m),
                                f'{arm}_mean': d7a.mean_effect(gate[arm], m)})

    # Multi-objective synthesis guardrails, regulator OFF.
    mask_seed = a.seed + 71000
    masked = {k: eval_masked(s, va, mask_seed, a.eval_images) for k, s in systems.items()}
    _rc, real_p = coarse.probability_from_ids(va, source.c.V)
    _pc, real_pos_p = spatial.position_probabilities(va, source.c.V, source.c.N)
    real_pair = spatial.pair_reference(va, source.c.V, source.c.N, 7)
    rollout = {k: rollout_js(s, real_p, real_pos_p, real_pair, BASE_GEN_SEED, a.gen_samples) for k, s in systems.items()}
    if not all(vv.get('all_trace_audits_pass', False) for vv in rollout.values()): raise RuntimeError('D9A rollout trace audit failed')
    guards = {'Q1': arm_guardrails(masked, rollout, 'Q1'), 'Q2': arm_guardrails(masked, rollout, 'Q2')}

    q1_primary = bool(all(contrasts['Q1_minus_B'][m]['ci_entirely_negative'] for m in PRIMARY))
    q2_primary = bool(all(contrasts['Q2_minus_B'][m]['ci_entirely_negative'] for m in PRIMARY))
    q1_synth = bool(q1_primary and guards['Q1']['all_pass'])
    q2_synth = bool(q2_primary and guards['Q2']['all_pass'])
    if q1_synth and q2_synth:
        classification = 'R3_HARD_AND_DYNAMIC_FUNCTIONAL_RESCUE'
    elif q1_synth:
        classification = 'R2_HARD_FUNCTIONAL_RESCUE_DYNAMIC_APPROX_FAIL'
    elif q1_primary:
        classification = 'R1_HARD_PRIMARY_RESCUE_WITH_GUARDRAIL_TENSION'
    else:
        classification = 'R0_NO_HARD_FUNCTIONAL_TRAJECTORY_RESCUE'

    # Persist.
    write_csv(a.out / 'D9A_REGULATOR_TRACE.csv', run['trace'])
    write_csv(a.out / 'D9A_Q1_CONTEXT_REGULATOR.csv', Q1._d9_reg_rows)
    write_csv(a.out / 'D9A_Q2_CONTEXT_REGULATOR.csv', Q2._d9_reg_rows)
    d6a.d5b.write_csv(a.out / 'D9A_Q1_LOCAL_ROWS.csv', Q1._d5b_rows)
    d6a.d5b.write_csv(a.out / 'D9A_Q2_LOCAL_ROWS.csv', Q2._d5b_rows)
    for label, s in systems.items(): save_terminal(a.out / f'D9A_{label}_terminal.pkl', s, label)

    npz = {'validation_ids': validation_ids, 'gt_tokens': gt.astype(np.int16), 'shared_uniform': shared,
           'frozen_y_basis': basis, 'qref': np.asarray([qref], dtype=np.float64)}
    for label, ev in gate.items():
        npz[f'{label}__self_metrics'] = ev['self_metrics']; npz[f'{label}__teacher_metrics'] = ev['teacher_metrics']
        npz[f'{label}__self_proposal'] = ev['self_proposal']; npz[f'{label}__teacher_proposal'] = ev['teacher_proposal']
        for m in ALL_METRICS: npz[f'{label}__effect__{m}'] = ev['effects'][m]
    np.savez_compressed(a.out / 'D9A_EFFECT_ARRAYS.npz', **npz)

    reg = {'Q1': regulator_summary(Q1), 'Q2': regulator_summary(Q2)}
    # WO-zero and bo norm-match are still exact update identities. D5B's native-tail
    # finite-step loss probe is not used as the regulator-aware oracle certificate.
    local_identity = {}
    for label, s in (('Q1', Q1), ('Q2', Q2)):
        local_identity[label] = {
            'max_bo_norm_match_abs_error': float(np.max([r['bo_norm_match_abs_error'] for r in s._d5b_rows])),
            'max_WO_zero_abs_error': float(np.max([0.0 if r.get('WO_zero_abs_error') is None else r['WO_zero_abs_error'] for r in s._d5b_rows])),
        }
        if local_identity[label]['max_bo_norm_match_abs_error'] > 2e-12 or local_identity[label]['max_WO_zero_abs_error'] > 1e-15:
            raise RuntimeError({'D9A_local_update_identity_failed': {label: local_identity[label]}})

    summary = {
        'stage': 'GATE_D9A_TRAJECTORY_LEVEL_RANK1_FUNCTIONAL_REGULATION',
        'classification': classification,
        'source_runs': {'D8B': 40021397096, 'D8A': 39979543621, 'D7A': 39871827217, 'D6A': 39687028646},
        'checkpoint_sha256': checkpoint_sha,
        'training_contexts': int(a.contexts),
        'evaluation_regulator_off': True,
        'D8_prerequisites': {'D8A': s8a.get('classification'), 'D8B': s8b.get('classification'),
                             'D8B_fixed_amplitude_bidirectional': s8b.get('decision', {}).get('discovery_fixed_amplitude_bidirectional')},
        'frozen_basis_rank': 1, 'frozen_basis_norm': float(np.linalg.norm(v)),
        'qref_source': 'D8A independent discovery rows projected through rescued terminal R',
        'qref_is_privileged_future_reference': True,
        'qref_stats': qref_stats,
        'controller_constants': {'Q1': {'hard_projection_lambda': 1.0},
                                 'Q2': {'rho': RHO, 'kappa': KAPPA, 'one_context_delay': True, 'initialized_m': qref}},
        'projection_aware_bo_oracle_self_test': oracle_self,
        'all_arm_stream_exact': True, 'all_arm_external_rng_exact': True,
        'model_replay_to_D6A': model_replay, 'GateA_replay_to_D6A': calibration,
        'local_update_identity': local_identity,
        'regulator_work': reg, 'regulator_checkpoints': run['trace'],
        'terminal_deployment_contrasts': contrasts, 'gap_closure': gap,
        'secondary_masked_regulator_off': masked, 'secondary_rollout_regulator_off': rollout,
        'guardrails': guards,
        'decision': {'Q1_primary_both': q1_primary, 'Q1_synthesis_pass': q1_synth,
                     'Q2_primary_both': q2_primary, 'Q2_synthesis_pass': q2_synth,
                     'D9B_authorized': bool(q1_synth)},
        'interpretation_boundary': (
            'D9A tests whether direct trajectory-level control of the frozen D8B causal y-space coordinate can reshape learning. '
            'Q1 remains a privileged activation-space upper bound using a discovery-derived rescued reference. Q2 is a fixed non-tuned scalar dynamical approximation. '
            'Neither is the final local/SNN/MSDL plasticity coordinator. Terminal evaluation is regulator-off.'
        ),
    }
    (a.out / 'D9A_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    manifest = {'classification': classification, 'contexts': int(a.contexts), 'eval_images': int(a.eval_images),
                'bootstrap': int(a.bootstrap), 'basis_rank': 1, 'evaluation_regulator_off': True,
                'parameter_search_performed': False, 'controller_parameter_sweep_performed': False,
                'protected_baseline_modified': False, 'no_PR_no_merge': True}
    (a.out / 'D9A_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print('D9A_RESULT', json.dumps({
        'classification': classification, 'qref': qref,
        'Q1_primary': {m: contrasts['Q1_minus_B'][m] for m in PRIMARY},
        'Q2_primary': {m: contrasts['Q2_minus_B'][m] for m in PRIMARY},
        'Q1_gap': {m: gap['Q1'][m] for m in PRIMARY},
        'Q2_gap': {m: gap['Q2'][m] for m in PRIMARY},
        'Q1_guardrails': guards['Q1'], 'Q2_guardrails': guards['Q2'],
        'regulator_work': reg, 'decision': summary['decision'],
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
