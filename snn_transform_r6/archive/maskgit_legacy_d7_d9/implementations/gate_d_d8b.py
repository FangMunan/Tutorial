#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D8B — frozen rank-1 bidirectional causal functional-mode intervention.

Eval-only causal test after D8A=I1, k=1.  The y-space basis is loaded unchanged
from the independent D8A discovery artifact.  Primary interventions use a
privileged matched R/C counterfactual amplitude on the same current discrete
context.  Sign-flip and discovery-fixed-amplitude controls are supportive only.

No training, learner modification, parameter transplant search, trajectory
replay, basis fitting, logits intervention, or h intervention occurs here.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

import gate_d_d3e_eval as d3e
import gate_d_d3e_runtime as d3r
import gate_d_d6a as d6a
import gate_d_d6a1 as d6a1
import gate_d_d7a1 as d71
import p3i_cb_exact_msdl as core
from maskgit_input_representation import context_from_arr_unconditional

PRIMARY = ('F61', 'D_non61')
ALL_METRICS = ('F61', 'D_non61', 'C61', 'delta_P61', 'CE')
TOL = 1e-7
ZERO_TOL = 1e-10
BASIS_TOL = 1e-10


def locate(root: Path, name: str) -> Path:
    return d3r.locate(root, name)


def bootstrap_vector(x, seed: int, nboot: int):
    x = np.asarray(x, dtype=np.float64)
    out = d6a.bootstrap_mean(x, seed, nboot)
    lo, hi = out['ci95']
    out['ci_entirely_positive'] = bool(lo is not None and lo > 0)
    out['ci_entirely_negative'] = bool(hi is not None and hi < 0)
    out['ci_contains_zero'] = bool(lo is not None and hi is not None and lo <= 0 <= hi)
    return out


def bootstrap_ratio(num, den, seed: int, nboot: int):
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    if num.shape != den.shape or num.ndim != 1:
        raise ValueError('bootstrap_ratio requires same-shape 1-D paired vectors')
    point_den = float(np.mean(den))
    point = None if abs(point_den) <= 1e-30 else float(np.mean(num) / point_den)
    rng = np.random.default_rng(seed)
    vals = np.full(nboot, np.nan, dtype=np.float64)
    n = len(num)
    for b in range(nboot):
        ii = rng.integers(0, n, size=n)
        d = float(np.mean(den[ii]))
        if d > 1e-30:
            vals[b] = float(np.mean(num[ii]) / d)
    good = vals[np.isfinite(vals)]
    ci = [None, None] if not len(good) else [float(x) for x in np.quantile(good, [0.025, 0.975])]
    return {
        'point_estimate': point,
        'ci95': ci,
        'valid_bootstrap_fraction': float(len(good) / nboot),
        'majority_point_capture': bool(point is not None and point > 0.50),
        'majority_certified_capture': bool(ci[0] is not None and ci[0] > 0.50),
    }


def vec(ev, metric):
    return np.asarray(ev['effects'][metric], dtype=np.float64)


def model_dict(system):
    return d3e.model_dict(system)


def tail_y(pred, a, xres):
    z = np.asarray(xres, dtype=np.float64) + np.asarray(pred.bo, dtype=np.float64) + np.asarray(pred.WO, dtype=np.float64) @ np.asarray(a, dtype=np.float64)
    mu = float(np.mean(z))
    inv = 1.0 / math.sqrt(float(np.mean((z - mu) ** 2)) + float(pred.eps2))
    u = (z - mu) * inv * np.asarray(pred.gamma2, dtype=np.float64) + np.asarray(pred.beta2, dtype=np.float64)
    h = np.maximum(0.0, np.asarray(pred.W1, dtype=np.float64) @ u + np.asarray(pred.b1, dtype=np.float64) - np.asarray(pred.theta, dtype=np.float64))
    y = z + np.asarray(pred.W2, dtype=np.float64) @ h + np.asarray(pred.b2, dtype=np.float64)
    return y


def prob_from_y(pred, y):
    y = np.asarray(y, dtype=np.float64)
    rms = math.sqrt(float(np.mean(y * y))) + 1e-8
    rf = np.empty(pred.c.D + 1, dtype=np.float64)
    rf[:-1] = 0.5 * y / rms
    rf[-1] = 0.5
    return np.asarray(core.softmax_numba(np.asarray(pred.WR, dtype=np.float64) @ rf), dtype=np.float64)


def predict_pair(rpred, cpred, cur, tokR, posR, tokC, posC, masked, v, mode, fixed_shift):
    """Predict one Gate-A round with R/C evaluated on the same discrete context.

    mode is one of R_base, C_base, C_minus, R_plus, C_wrong, R_wrong,
    C_fixed_minus, R_fixed_plus.
    """
    cur = np.asarray(cur, dtype=np.int64)
    xnR, xrawR = context_from_arr_unconditional(cur, np.asarray(tokR), np.asarray(posR))
    xnC, xrawC = context_from_arr_unconditional(cur, np.asarray(tokC), np.asarray(posC))
    rpred.rebuild(xnR)
    cpred.rebuild(xnC)
    probs = np.empty((len(masked), rpred.c.V), dtype=np.float64)
    amps = np.empty(len(masked), dtype=np.float64)
    proj_frac = np.empty(len(masked), dtype=np.float64)

    for qi, t0 in enumerate(np.asarray(masked, dtype=np.int64)):
        t = int(t0)
        aR, _qR, _nR, _dR = rpred._read(t)
        aC, _qC, _nC, _dC = cpred._read(t)
        yR = tail_y(rpred, aR, xrawR[t])
        yC = tail_y(cpred, aC, xrawC[t])
        dy = yC - yR
        amp = float(np.dot(v, dy))
        proj = amp * v
        amps[qi] = amp
        den = float(np.dot(dy, dy))
        proj_frac[qi] = 0.0 if den <= 1e-30 else float((amp * amp) / den)

        if mode == 'R_base':
            base, pred, shift = yR, rpred, 0.0
        elif mode == 'C_base':
            base, pred, shift = yC, cpred, 0.0
        elif mode == 'C_minus':
            base, pred, shift = yC, cpred, -proj
        elif mode == 'R_plus':
            base, pred, shift = yR, rpred, proj
        elif mode == 'C_wrong':
            base, pred, shift = yC, cpred, proj
        elif mode == 'R_wrong':
            base, pred, shift = yR, rpred, -proj
        elif mode == 'C_fixed_minus':
            base, pred, shift = yC, cpred, -fixed_shift
        elif mode == 'R_fixed_plus':
            base, pred, shift = yR, rpred, fixed_shift
        else:
            raise ValueError(mode)
        probs[qi] = prob_from_y(pred, np.asarray(base, dtype=np.float64) + shift)

    probs /= probs.sum(axis=1, keepdims=True)
    return probs, amps, proj_frac


def diag_summary(x):
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return {'n': 0}
    return {
        'n': int(len(x)),
        'mean': float(np.mean(x)),
        'std': float(np.std(x)),
        'mean_abs': float(np.mean(np.abs(x))),
        'q05': float(np.quantile(x, 0.05)),
        'median': float(np.median(x)),
        'q95': float(np.quantile(x, 0.95)),
        'min': float(np.min(x)),
        'max': float(np.max(x)),
    }


def eval_gate_mode(name, R, C, schedules, gt, shared_u, v, mode, fixed_shift):
    n = len(gt)
    sm = np.full((n, d3e.ROUNDS, len(d3e.METRIC_NAMES)), np.nan)
    tm = np.full_like(sm, np.nan)
    sp = np.full((n, d3e.ROUNDS, R.c.N), -1, np.int16)
    tp = np.full_like(sp, -1)
    amps_self, amps_teacher = [], []
    frac_self, frac_teacher = [], []
    gates = {
        'round1_probability_bitwise_parity': True,
        'round1_sample_exact_parity': True,
        'committed_position_parity': True,
        'intervention_purity': True,
        'common_random_number_integrity': True,
        'image_count_after_three_rounds_exact': True,
        'matched_counterfactual_same_discrete_context': True,
    }
    mR = model_dict(R)
    mC = model_dict(C)
    tokR, posR = np.asarray(R.tok), np.asarray(R.pos)
    tokC, posC = np.asarray(C.tok), np.asarray(C.pos)

    for i in range(n):
        seed = d3e.BASE_GEN_SEED + 100000 + i
        rs = core.P3ImageSystem(R.c, mR, seed)
        csref = core.P3ImageSystem(C.c, mC, seed)
        rt = core.P3ImageSystem(R.c, mR, seed)
        ctref = core.P3ImageSystem(C.c, mC, seed)
        cur_s = np.full(R.c.N, d3e.MASK, int)
        cur_t = np.full(R.c.N, d3e.MASK, int)

        for r in range(d3e.ROUNDS):
            tr = schedules[i][r]
            masked = np.asarray(tr['masked_positions_before'], int)
            remask = np.asarray(tr['remasked_positions'], int)
            if (not np.array_equal(np.flatnonzero(cur_s == d3e.MASK), masked)
                    or not np.array_equal(np.flatnonzero(cur_t == d3e.MASK), masked)):
                gates['committed_position_parity'] = False
            if r == 0:
                if not np.array_equal(cur_s, cur_t) or not np.all(cur_s == d3e.MASK):
                    gates['intervention_purity'] = False
            else:
                if not np.array_equal(cur_s == d3e.MASK, cur_t == d3e.MASK):
                    gates['intervention_purity'] = False
                vis = np.flatnonzero(cur_t != d3e.MASK)
                if len(vis) and not np.array_equal(cur_t[vis], gt[i, vis]):
                    gates['intervention_purity'] = False
                if np.any((cur_s != cur_t) & (cur_t == d3e.MASK)):
                    gates['intervention_purity'] = False

            p1, a1, f1 = predict_pair(rs, csref, cur_s, tokR, posR, tokC, posC,
                                      masked, v, mode, fixed_shift)
            p2, a2, f2 = predict_pair(rt, ctref, cur_t, tokR, posR, tokC, posC,
                                      masked, v, mode, fixed_shift)
            amps_self.append(a1); amps_teacher.append(a2)
            frac_self.append(f1); frac_teacher.append(f2)
            u = np.asarray(shared_u[i, r, masked], float)
            if not np.array_equal(u, shared_u[i, r, masked]):
                gates['common_random_number_integrity'] = False
            z1 = d3e.inverse_cdf_rows(p1, u)
            z2 = d3e.inverse_cdf_rows(p2, u)
            if r == 0 and not np.array_equal(p1, p2):
                gates['round1_probability_bitwise_parity'] = False
            if r == 0 and not np.array_equal(z1, z2):
                gates['round1_sample_exact_parity'] = False

            sm[i, r] = d3e.one_metrics(p1, gt[i, masked], z1, cur_s)
            tm[i, r] = d3e.one_metrics(p2, gt[i, masked], z2, cur_t)
            pr1 = cur_s.copy(); pr1[masked] = z1
            pr2 = cur_t.copy(); pr2[masked] = z2
            sp[i, r] = pr1; tp[i, r] = pr2
            ns = pr1.copy(); ns[remask] = d3e.MASK
            nt = cur_t.copy()
            new = np.setdiff1d(masked, remask, assume_unique=True)
            nt[new] = gt[i, new]
            nt[remask] = d3e.MASK
            old = np.flatnonzero(cur_t != d3e.MASK)
            if len(old):
                nt[old] = gt[i, old]
            cur_s, cur_t = ns, nt

        counts = (rs.image_count, csref.image_count, rt.image_count, ctref.image_count)
        if any(int(x) != d3e.ROUNDS for x in counts):
            gates['image_count_after_three_rounds_exact'] = False

    gates['all_pass'] = bool(all(gates.values()))
    effects = d3e.effect_arrays(sm, tm, 2)
    amp_s = np.concatenate(amps_self) if amps_self else np.empty(0)
    amp_t = np.concatenate(amps_teacher) if amps_teacher else np.empty(0)
    fr_s = np.concatenate(frac_self) if frac_self else np.empty(0)
    fr_t = np.concatenate(frac_teacher) if frac_teacher else np.empty(0)
    return {
        'name': name,
        'self_metrics': sm,
        'teacher_metrics': tm,
        'self_proposal': sp,
        'teacher_proposal': tp,
        'effects': effects,
        'gates': gates,
        'matched_amplitude_self': amp_s,
        'matched_amplitude_teacher': amp_t,
        'matched_projection_fraction_self': fr_s,
        'matched_projection_fraction_teacher': fr_t,
        'amplitude_summary': {
            'self': diag_summary(amp_s),
            'teacher': diag_summary(amp_t),
            'projection_fraction_self': diag_summary(fr_s),
            'projection_fraction_teacher': diag_summary(fr_t),
        },
    }


def functional_match(ev, ref, tol):
    return d71.functional_match(ev, ref, tol)


def calibrate(ev, stored, prefix, tol):
    return d71.calibrate_full(ev, stored, prefix, tol)


def endpoint_contrasts(ev1, ev0, seed0, nboot):
    out = {}
    for mi, m in enumerate(ALL_METRICS):
        out[m] = bootstrap_vector(vec(ev1, m) - vec(ev0, m), seed0 + mi, nboot)
    return out


def capture_summary(num_ev, base_ev, load_v, direction, seed0, nboot):
    out = {}
    for mi, m in enumerate(ALL_METRICS):
        if direction == 'removal':
            num = vec(base_ev, m) - vec(num_ev, m)
        elif direction == 'addition':
            num = vec(num_ev, m) - vec(base_ev, m)
        else:
            raise ValueError(direction)
        out[m] = bootstrap_ratio(num, load_v[m], seed0 + mi, nboot)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d6a-artifact', type=Path, required=True)
    ap.add_argument('--d7a-artifact', type=Path, required=True)
    ap.add_argument('--d7a2-artifact', type=Path, required=True)
    ap.add_argument('--d8a-artifact', type=Path, required=True)
    ap.add_argument('--anchor-artifact', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--bootstrap', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=7311)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)

    # D8A frozen prerequisite and basis.
    s8 = json.loads(locate(a.d8a_artifact, 'D8A_SUMMARY.json').read_text(encoding='utf-8'))
    if s8.get('classification') != 'I1_SHARED_LOW_DIMENSIONAL_FUNCTIONAL_CANDIDATE':
        raise RuntimeError('D8B prerequisite failed: D8A is not I1')
    if int(s8.get('primary_y_gate', {}).get('selected_k', -1)) != 1:
        raise RuntimeError('D8B prerequisite failed: D8A selected_k is not 1')
    if not s8.get('decision', {}).get('proceed_to_D8B', False):
        raise RuntimeError('D8B prerequisite failed: D8A did not authorize D8B')
    if s8.get('decision', {}).get('D8B_basis_source') != 'pair_y_discovery_topk':
        raise RuntimeError('D8B basis source changed')
    modes = np.load(locate(a.d8a_artifact, 'D8A_MODES.npz'), allow_pickle=False)
    basis = np.asarray(modes['D8B_frozen_y_pair_basis'], dtype=np.float64)
    if basis.shape != (96, 1) or not np.all(np.isfinite(basis)):
        raise RuntimeError({'D8B_basis_shape_or_finite_failure': basis.shape})
    v = basis[:, 0].copy()
    basis_norm = float(np.linalg.norm(v))
    if abs(basis_norm - 1.0) >= BASIS_TOL:
        raise RuntimeError({'D8B_basis_not_unit': basis_norm})

    stored72 = np.load(locate(a.d7a2_artifact, 'D7A2_EFFECT_ARRAYS.npz'), allow_pickle=False)
    schedules = json.loads(locate(a.anchor_artifact, 'HISTORICAL_R6B3_R3_T8_REFERENCE_TRACE.json').read_text(encoding='utf-8'))
    validation_ids = np.asarray(stored72['validation_ids'], dtype=np.int64)
    gt = np.asarray(stored72['gt_tokens'], dtype=np.int64)
    shared = np.asarray(stored72['shared_uniform'], dtype=np.float64)
    discover_ids = np.asarray(modes['discover_ids'], dtype=np.int64)
    overlap = int(np.intersect1d(discover_ids, validation_ids).size)
    if overlap != 0 or int(s8.get('discovery', {}).get('gateA_overlap_count', -1)) != 0:
        raise RuntimeError({'D8B_discovery_GateA_overlap': overlap})

    # Reconstruct exact R/C used in D8A.
    B = d6a1.load_d6a_terminal(locate(a.d6a_artifact, 'D6A_F_EM_WO_zero_bo_oracle_terminal.pkl'))
    Fsafe = d71.load_terminal(locate(a.d7a_artifact, 'D7A_F_terminal.pkl'))
    C = d71.load_terminal(locate(a.d7a_artifact, 'D7A_C_W2_safe_clamp_terminal.pkl'))
    R = d71.hybrid(B, Fsafe, ('W2',))
    invariants = {
        'W2_R_equals_C': d71.arr_exact(R.W2, C.W2),
        'W2_R_C_max_abs': d71.max_abs(R.W2, C.W2),
        'WO_R_equals_C': d71.arr_exact(R.WO, C.WO),
        'WO_R_C_max_abs': d71.max_abs(R.WO, C.WO),
    }
    if not (invariants['W2_R_equals_C'] and invariants['W2_R_C_max_abs'] == 0.0
            and invariants['WO_R_equals_C'] and invariants['WO_R_C_max_abs'] == 0.0):
        raise RuntimeError({'D8B_R_C_invariant_failure': invariants})

    # Ordinary frozen Gate-A calibration.
    ordinary = {
        'R': d71.gate_eval('D8B_ORD_R', R, schedules, gt, shared),
        'C': d71.gate_eval('D8B_ORD_C', C, schedules, gt, shared),
    }
    ordinary_cal = {
        'R': calibrate(ordinary['R'], stored72, 'R', TOL),
        'C': calibrate(ordinary['C'], stored72, 'C', TOL),
    }
    if not all(x['pass'] for x in ordinary_cal.values()):
        raise RuntimeError({'D8B_D7A2_ordinary_calibration_failure': ordinary_cal})

    pair_mean = np.asarray(modes['y__pair__mean'], dtype=np.float64)
    if pair_mean.shape != (96,) or not np.all(np.isfinite(pair_mean)):
        raise RuntimeError('D8B frozen y pair mean invalid')
    a_disc = float(np.dot(v, pair_mean))
    fixed_shift = a_disc * v

    # Custom zero evaluator must reproduce ordinary/frozen Gate-A before intervention.
    zero = {
        'R': eval_gate_mode('D8B_ZERO_R', R, C, schedules, gt, shared, v, 'R_base', fixed_shift),
        'C': eval_gate_mode('D8B_ZERO_C', R, C, schedules, gt, shared, v, 'C_base', fixed_shift),
    }
    zero_cal = {
        'R': calibrate(zero['R'], stored72, 'R', ZERO_TOL),
        'C': calibrate(zero['C'], stored72, 'C', ZERO_TOL),
    }
    zero_vs_ordinary = {
        'R': functional_match(zero['R'], ordinary['R'], ZERO_TOL),
        'C': functional_match(zero['C'], ordinary['C'], ZERO_TOL),
    }
    if not all(x['pass'] for x in zero_cal.values()) or not all(x['pass'] for x in zero_vs_ordinary.values()):
        raise RuntimeError({'D8B_zero_intervention_parity_failure': {'cal': zero_cal, 'ordinary': zero_vs_ordinary}})

    # Primary matched interventions and supportive controls.
    evals = dict(zero)
    specs = {
        'C_minus': 'C_minus',
        'R_plus': 'R_plus',
        'C_wrong': 'C_wrong',
        'R_wrong': 'R_wrong',
        'C_fixed_minus': 'C_fixed_minus',
        'R_fixed_plus': 'R_fixed_plus',
    }
    for label, mode in specs.items():
        evals[label] = eval_gate_mode('D8B_' + label, R, C, schedules, gt, shared, v, mode, fixed_shift)
        if not evals[label]['gates']['all_pass']:
            raise RuntimeError({'D8B_GateA_structure_failure': label, 'gates': evals[label]['gates']})
        print('D8B_GATE', json.dumps({'system': label}), flush=True)

    # Per-image original load and primary causal effects.
    load_v = {m: vec(evals['C'], m) - vec(evals['R'], m) for m in ALL_METRICS}
    loads = {m: bootstrap_vector(load_v[m], 1700000 + i, a.bootstrap)
             for i, m in enumerate(ALL_METRICS)}
    removal = endpoint_contrasts(evals['C_minus'], evals['C'], 1710000, a.bootstrap)
    addition = endpoint_contrasts(evals['R_plus'], evals['R'], 1720000, a.bootstrap)
    wrong_C = endpoint_contrasts(evals['C_wrong'], evals['C'], 1730000, a.bootstrap)
    wrong_R = endpoint_contrasts(evals['R_wrong'], evals['R'], 1740000, a.bootstrap)
    fixed_removal = endpoint_contrasts(evals['C_fixed_minus'], evals['C'], 1750000, a.bootstrap)
    fixed_addition = endpoint_contrasts(evals['R_fixed_plus'], evals['R'], 1760000, a.bootstrap)

    capture_removal = capture_summary(evals['C_minus'], evals['C'], load_v, 'removal', 1770000, a.bootstrap)
    capture_addition = capture_summary(evals['R_plus'], evals['R'], load_v, 'addition', 1780000, a.bootstrap)
    fixed_capture_removal = capture_summary(evals['C_fixed_minus'], evals['C'], load_v, 'removal', 1790000, a.bootstrap)
    fixed_capture_addition = capture_summary(evals['R_fixed_plus'], evals['R'], load_v, 'addition', 1800000, a.bootstrap)

    removal_primary = bool(all(removal[m]['ci_entirely_negative'] for m in PRIMARY))
    addition_primary = bool(all(addition[m]['ci_entirely_positive'] for m in PRIMARY))
    if removal_primary and addition_primary:
        classification = 'B1_BIDIRECTIONAL_CAUSAL_HARMFUL_FUNCTIONAL_MODE'
    elif removal_primary:
        classification = 'B2_RESCUE_ONLY_FUNCTIONAL_MODE'
    elif addition_primary:
        classification = 'B3_ADDITION_ONLY_FUNCTIONAL_MODE'
    else:
        classification = 'B4_DESCRIPTIVE_ONLY_FUNCTIONAL_MODE'

    orientation = {
        'C_wrong_worsens_both_primary': bool(all(wrong_C[m]['ci_entirely_positive'] for m in PRIMARY)),
        'R_wrong_no_positive_harm_on_either_primary': bool(all(not wrong_R[m]['ci_entirely_positive'] for m in PRIMARY)),
    }
    orientation['supportive_pass'] = bool(orientation['C_wrong_worsens_both_primary']
                                          and orientation['R_wrong_no_positive_harm_on_either_primary'])
    fixed_support = {
        'removal_both_primary': bool(all(fixed_removal[m]['ci_entirely_negative'] for m in PRIMARY)),
        'addition_both_primary': bool(all(fixed_addition[m]['ci_entirely_positive'] for m in PRIMARY)),
    }
    fixed_support['bidirectional'] = bool(fixed_support['removal_both_primary'] and fixed_support['addition_both_primary'])

    # Persist full audit arrays.
    npz = {
        'validation_ids': validation_ids,
        'gt_tokens': gt.astype(np.int16),
        'shared_uniform': shared,
        'frozen_y_basis': basis,
        'discovery_pair_y_mean': pair_mean,
        'discovery_fixed_amplitude': np.asarray([a_disc], dtype=np.float64),
        'discovery_fixed_shift': fixed_shift,
    }
    for label, ev in evals.items():
        npz[f'{label}__self_metrics'] = ev['self_metrics']
        npz[f'{label}__teacher_metrics'] = ev['teacher_metrics']
        npz[f'{label}__self_proposal'] = ev['self_proposal']
        npz[f'{label}__teacher_proposal'] = ev['teacher_proposal']
        npz[f'{label}__matched_amplitude_self'] = ev['matched_amplitude_self']
        npz[f'{label}__matched_amplitude_teacher'] = ev['matched_amplitude_teacher']
        npz[f'{label}__matched_projection_fraction_self'] = ev['matched_projection_fraction_self']
        npz[f'{label}__matched_projection_fraction_teacher'] = ev['matched_projection_fraction_teacher']
        for m in ALL_METRICS:
            npz[f'{label}__effect__{m}'] = vec(ev, m)
    np.savez_compressed(a.out / 'D8B_EFFECT_ARRAYS.npz', **npz)

    summary = {
        'stage': 'GATE_D8B_FROZEN_RANK1_BIDIRECTIONAL_CAUSAL_FUNCTIONAL_MODE',
        'classification': classification,
        'source_runs': {'D8A': 39979543621, 'D7A2': 39942308903, 'D7A': 39871827217, 'D6A': 39687028646},
        'training_performed': False,
        'learner_modified': False,
        'parameter_search_performed': False,
        'trajectory_replay_performed': False,
        'activation_space': 'y',
        'rank': 1,
        'basis_source': 'D8A independent-discovery pair-y top-1 right-singular vector',
        'matched_intervention_is_privileged_counterfactual_upper_bound': True,
        'basis_norm': basis_norm,
        'discovery_GateA_overlap_count': overlap,
        'invariants': invariants,
        'ordinary_calibration_to_D7A2': ordinary_cal,
        'zero_custom_calibration_to_D7A2': zero_cal,
        'zero_custom_vs_ordinary': zero_vs_ordinary,
        'discovery_fixed_amplitude': a_disc,
        'discovery_fixed_shift_norm': float(np.linalg.norm(fixed_shift)),
        'discovery_mean_projection_fraction': float((a_disc * a_disc) / max(1e-30, float(np.dot(pair_mean, pair_mean)))),
        'total_compensation_load': loads,
        'matched_removal_Cminus_vs_C': removal,
        'matched_addition_Rplus_vs_R': addition,
        'matched_removal_load_capture': capture_removal,
        'matched_addition_load_capture': capture_addition,
        'sign_flip_Cwrong_vs_C': wrong_C,
        'sign_flip_Rwrong_vs_R': wrong_R,
        'orientation_control': orientation,
        'fixed_amplitude_Cminus_vs_C': fixed_removal,
        'fixed_amplitude_Rplus_vs_R': fixed_addition,
        'fixed_amplitude_removal_load_capture': fixed_capture_removal,
        'fixed_amplitude_addition_load_capture': fixed_capture_addition,
        'fixed_amplitude_support': fixed_support,
        'amplitude_diagnostics': {label: ev['amplitude_summary'] for label, ev in evals.items()},
        'decision': {
            'matched_removal_both_primary': removal_primary,
            'matched_addition_both_primary': addition_primary,
            'bidirectional_primary': bool(removal_primary and addition_primary),
            'orientation_control_supportive_pass': orientation['supportive_pass'],
            'discovery_fixed_amplitude_bidirectional': fixed_support['bidirectional'],
            'causal_harmful_functional_mode_supported': bool(classification == 'B1_BIDIRECTIONAL_CAUSAL_HARMFUL_FUNCTIONAL_MODE'),
        },
        'interpretation_boundary': (
            'B1 establishes bidirectional causal negative deployment utility for the frozen rank-1 y-space coordinate under a privileged matched-counterfactual amplitude upper bound. '
            'The discovery-fixed control tests whether a Gate-A-independent scalar shift generalizes. Neither result establishes a final online/local/MSDL/SNN controller or trajectory emergence of the mode.'
        ),
    }
    (a.out / 'D8B_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    manifest = {
        'classification': classification,
        'eval_images': int(len(gt)),
        'bootstrap': int(a.bootstrap),
        'basis_rank': 1,
        'basis_norm': basis_norm,
        'discovery_GateA_overlap_count': overlap,
        'training_performed': False,
        'learner_modified': False,
        'parameter_search_performed': False,
        'protected_baseline_modified': False,
        'no_PR_no_merge': True,
    }
    (a.out / 'D8B_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print('D8B_RESULT', json.dumps({
        'classification': classification,
        'loads': {m: loads[m] for m in PRIMARY},
        'removal': {m: removal[m] for m in PRIMARY},
        'addition': {m: addition[m] for m in PRIMARY},
        'capture_removal': {m: capture_removal[m] for m in PRIMARY},
        'capture_addition': {m: capture_addition[m] for m in PRIMARY},
        'orientation': orientation,
        'fixed_removal': {m: fixed_removal[m] for m in PRIMARY},
        'fixed_addition': {m: fixed_addition[m] for m in PRIMARY},
        'fixed_support': fixed_support,
        'a_disc': a_disc,
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
