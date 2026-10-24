#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D9A.3 — pair-versus-residual mode-consolidation decomposition.

Diagnostic-only replay of frozen D7A F/B/C trajectories. At each frozen
checkpoint construct R_t, pair-transplant P_t and reciprocal Q_t, then decompose
the same D9A.2-R C_t-R_t functional geometry into the known (b2,bO) subsystem
and the residual network. No learning rule, update, controller, mode, or
parameter-search logic is changed.
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
import gate_d_d7a as d7a
import gate_d_d9a1 as d91
import gate_d_plasticity_audit as gd
import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
import r6b3_context_epoch_training as trainer
from maskgit_input_representation import build_input_embeddings
from r6b4_full_forward_plasticity import FORWARD_PARAMETER_KEYS, upgrade_r6b3_system

CHECKPOINTS = (64, 128, 256, 512, 768, 1024)
PAIR_FIELDS = ('b2', 'bo')
ALIGN_GATE = 0.80
PAIR_MAJORITY = 0.50
JOINT_MIN_SHARE = 0.25
DECOMP_TOL = 1e-12
CAL_TOL = 1e-10
EPS = 1e-30


def locate(root: Path, name: str) -> Path:
    return d3r.locate(root, name)


def clone(x):
    return pickle.loads(pickle.dumps(x, pickle.HIGHEST_PROTOCOL))


def write_csv(path: Path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    keys, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def load_terminal(path: Path):
    with path.open('rb') as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or 'system' not in obj:
        raise RuntimeError(f'bad terminal schema: {path}')
    return obj['system']


def pair_only_check(base, hybrid, donor):
    out = {}
    ok = True
    for k in FORWARD_PARAMETER_KEYS:
        b = np.asarray(getattr(base, k))
        h = np.asarray(getattr(hybrid, k))
        d = np.asarray(getattr(donor, k))
        if k in PAIR_FIELDS:
            same = bool(np.array_equal(h, d))
            err = float(np.max(np.abs(h.astype(np.float64) - d.astype(np.float64)))) if h.size else 0.0
        else:
            same = bool(np.array_equal(h, b))
            err = float(np.max(np.abs(h.astype(np.float64) - b.astype(np.float64)))) if h.size else 0.0
        out[k] = {'pass': same, 'max_abs': err}
        ok = ok and same
    out['pass'] = bool(ok)
    return out


def build_states(F, B, C):
    R = clone(B)
    R.W2 = np.array(F.W2, copy=True)
    if d7a.max_abs(C.W2, R.W2) != 0.0:
        raise RuntimeError('D9A3 W2 C/R identity failure')

    P = clone(R)
    P.b2 = np.array(C.b2, copy=True)
    P.bo = np.array(C.bo, copy=True)

    Q = clone(C)
    Q.b2 = np.array(R.b2, copy=True)
    Q.bo = np.array(R.bo, copy=True)

    pc = pair_only_check(R, P, C)
    qc = pair_only_check(C, Q, R)
    if not (pc['pass'] and qc['pass']):
        raise RuntimeError({'D9A3_pair_only_transplant_failure': {'P': pc, 'Q': qc}})
    return R, P, Q, pc, qc


def collect_y(systems, diag_tokens, diag_ids):
    snaps = {k: d91.snapshot_forward(v) for k, v in systems.items()}
    chunks = {k: [] for k in systems}
    for i, did in enumerate(np.asarray(diag_ids, dtype=np.int64)):
        gt = np.asarray(diag_tokens[i], dtype=np.int64)
        for _depth, cur, masked in d91.diagnostic_contexts(gt, int(did), systems['C'].c.N, systems['C'].c.V):
            for k in systems:
                chunks[k].append(d91.stateless_y(snaps[k], systems['C'].c, cur, masked))
    return {k: np.vstack(v) for k, v in chunks.items()}


def component_geometry(D, v):
    D = np.asarray(D, dtype=np.float64)
    p = D @ np.asarray(v, dtype=np.float64)
    energy = float(np.sum(D * D))
    proj_energy = float(np.sum(p * p))
    frob = math.sqrt(max(0.0, energy))
    if energy <= EPS:
        return {
            'rows': int(D.shape[0]), 'a': 0.0, 'median_a': 0.0, 'projection_std': 0.0,
            'energy': energy, 'frob': frob, 'rho_v': 0.0, 'EV1': 0.0,
            'cos_v1_to_terminal_v': 0.0, 'v1': np.zeros(D.shape[1], dtype=np.float64),
            'singular_values_sq': np.zeros(D.shape[1], dtype=np.float64),
            'row_projection': p, 'projected_energy': proj_energy,
        }
    G = D.T @ D
    G = 0.5 * (G + G.T)
    vals, vecs = np.linalg.eigh(G)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(0.0, vals[order])
    vecs = vecs[:, order]
    total = float(np.sum(vals))
    v1 = np.asarray(vecs[:, 0], dtype=np.float64)
    return {
        'rows': int(D.shape[0]),
        'a': float(np.mean(p)),
        'median_a': float(np.median(p)),
        'projection_std': float(np.std(p)),
        'energy': energy,
        'frob': frob,
        'rho_v': float(proj_energy / (energy + EPS)),
        'EV1': float(vals[0] / (total + EPS)),
        'cos_v1_to_terminal_v': float(abs(np.dot(v1, v))),
        'v1': v1,
        'singular_values_sq': vals,
        'row_projection': p,
        'projected_energy': proj_energy,
    }


def frob_cos(A, B):
    a = np.asarray(A, dtype=np.float64).reshape(-1)
    b = np.asarray(B, dtype=np.float64).reshape(-1)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > EPS else 0.0


def opposing_ratio(a_pair, a_res):
    if abs(float(a_pair)) <= 1e-15:
        return float('inf')
    return float(max(0.0, -float(a_res) / float(a_pair)))


def decompose_checkpoint(F, B, C, diag_tokens, diag_ids, v):
    R, P, Q, pcheck, qcheck = build_states(F, B, C)
    Y = collect_y({'R': R, 'C': C, 'P': P, 'Q': Q}, diag_tokens, diag_ids)

    Dfull = Y['C'] - Y['R']
    Dpair = Y['P'] - Y['R']
    Dres = Y['C'] - Y['P']
    Drelease = Y['C'] - Y['Q']
    decomp_err = float(np.max(np.abs(Dfull - (Dpair + Dres)))) if Dfull.size else 0.0
    if decomp_err > DECOMP_TOL:
        raise RuntimeError({'D9A3_decomposition_identity_failure': decomp_err})

    gf = component_geometry(Dfull, v)
    gp = component_geometry(Dpair, v)
    gr = component_geometry(Dres, v)
    gl = component_geometry(Drelease, v)

    chi = frob_cos(Dpair, Dres)
    pair_release_cos = frob_cos(Dpair, Drelease)
    pair_v_energy_over_full = float(gp['projected_energy'] / (gf['projected_energy'] + EPS))

    return {
        'R': R, 'P': P, 'Q': Q,
        'P_transplant_check': pcheck, 'Q_transplant_check': qcheck,
        'Dfull': Dfull, 'Dpair': Dpair, 'Dres': Dres, 'Drelease': Drelease,
        'full': gf, 'pair': gp, 'res': gr, 'release': gl,
        'decomp_max_abs': decomp_err,
        'chi_pair_res': chi,
        'pair_release_frob_cos': pair_release_cos,
        'a_pair_times_a_res': float(gp['a'] * gr['a']),
        'pair_v_projected_energy_over_full': pair_v_energy_over_full,
        'opposing_ratio': opposing_ratio(gp['a'], gr['a']),
    }


def classify(rows):
    by = {int(r['context']): r for r in rows}
    r128, r256 = by[128], by[256]
    gfull = float(r256['full_a'] - r128['full_a'])
    gpair = float(r256['pair_a'] - r128['pair_a'])
    gres = float(r256['res_a'] - r128['res_a'])
    identity_err = abs(gfull - (gpair + gres))
    if identity_err > 1e-12:
        raise RuntimeError({'D9A3_growth_identity_failure': identity_err})
    if gfull > 1e-12:
        spair = gpair / gfull
        sres = gres / gfull
    else:
        spair = None
        sres = None

    m2 = bool(
        gfull > 1e-12
        and r128['pair_cos_v1'] >= ALIGN_GATE
        and r128['pair_a'] * r128['res_a'] < 0.0
        and r128['chi_pair_res'] < 0.0
        and r256['opposing_ratio'] < r128['opposing_ratio']
    )
    m3 = bool(
        (not m2)
        and gfull > 1e-12
        and gpair > 0.0 and gres > 0.0
        and spair is not None and sres is not None
        and spair >= JOINT_MIN_SHARE and sres >= JOINT_MIN_SHARE
        and r256['pair_cos_v1'] >= ALIGN_GATE
        and r256['res_cos_v1'] >= ALIGN_GATE
    )
    m1 = bool(
        (not m2) and (not m3)
        and gfull > 1e-12
        and spair is not None and spair >= PAIR_MAJORITY
        and r256['pair_cos_v1'] >= ALIGN_GATE
    )

    if m2:
        cls = 'M2_RESIDUAL_UNMASKING_CANCELLATION_RELEASE'
    elif m3:
        cls = 'M3_JOINT_CO_CONSOLIDATION'
    elif m1:
        cls = 'M1_PAIR_DRIVEN_CONSOLIDATION'
    else:
        cls = 'M4_PAIR_DOES_NOT_EXPLAIN_TRAJECTORY_CONSOLIDATION'

    return cls, {
        'transition': [128, 256],
        'g_full': gfull, 'g_pair': gpair, 'g_res': gres,
        'growth_identity_abs_error': identity_err,
        'pair_growth_share': spair, 'res_growth_share': sres,
        'pair_aligned_128': bool(r128['pair_cos_v1'] >= ALIGN_GATE),
        'pair_aligned_256': bool(r256['pair_cos_v1'] >= ALIGN_GATE),
        'res_aligned_256': bool(r256['res_cos_v1'] >= ALIGN_GATE),
        'scalar_opposition_128': bool(r128['pair_a'] * r128['res_a'] < 0.0),
        'matrix_cancellation_128': bool(r128['chi_pair_res'] < 0.0),
        'opposing_ratio_128': r128['opposing_ratio'],
        'opposing_ratio_256': r256['opposing_ratio'],
        'opposing_ratio_decreased': bool(r256['opposing_ratio'] < r128['opposing_ratio']),
        'M1_condition': m1, 'M2_condition': m2, 'M3_condition': m3,
    }


def replay(start, tokens, indices, rng_state, diag_tokens, diag_ids, v,
           frozen2_summary, frozen2_arrays):
    F = d6a.make_d6a(start, d7a.F_LABEL)
    B = d6a.make_d6a(start, d7a.B_LABEL)
    C = d6a.make_d6a(start, d7a.B_LABEL)
    rf = gd.clone_rng(rng_state)
    rb = gd.clone_rng(rng_state)
    rc = gd.clone_rng(rng_state)
    streams = {'F': [], 'B': [], 'C': []}
    rows = []
    arrays = {}
    max_post_clamp = 0.0
    cumulative_clamp = 0.0
    max_full_projection_cal_error = 0.0
    max_full_summary_cal_error = 0.0
    max_decomp_error = 0.0
    parity_checks = []
    transplant_checks = []

    frozen_rows = {int(r['context']): r for r in frozen2_summary['checkpoint_geometry']}

    for j, image_index in enumerate(indices, start=1):
        if d7a.max_abs(C.W2, F.W2) != 0.0:
            raise RuntimeError({'D9A3_precontext_W2_not_safe': j})

        for s, rr in ((F, rf), (B, rb), (C, rc)):
            before = d3r.jstate(rr)
            d3r.reset_boundary(s, 'F_EM')
            if d3r.jstate(rr) != before:
                raise RuntimeError('reset consumed external RNG')

        truth = np.asarray(tokens[int(image_index)], dtype=np.int64)
        _rF, evF = d6a.run_context(F, j, int(image_index), truth, rf)
        _rB, evB = d6a.run_context(B, j, int(image_index), truth, rb)
        _rC, evC = d6a.run_context(C, j, int(image_index), truth, rc)

        sF = d6a.stream_row(j, int(image_index), evF)
        sB = d6a.stream_row(j, int(image_index), evB)
        sC = d6a.stream_row(j, int(image_index), evC)
        if not (sF == sB == sC):
            raise RuntimeError({'D9A3_stream_divergence': j})
        if not (d3r.jstate(rf) == d3r.jstate(rb) == d3r.jstate(rc)):
            raise RuntimeError({'D9A3_rng_divergence': j})
        streams['F'].append(sF); streams['B'].append(sB); streams['C'].append(sC)

        correction = np.asarray(F.W2, dtype=np.float64) - np.asarray(C.W2, dtype=np.float64)
        cumulative_clamp += float(np.linalg.norm(correction))
        C.W2 = np.array(F.W2, copy=True)
        err = d7a.max_abs(C.W2, F.W2)
        max_post_clamp = max(max_post_clamp, err)
        if err != 0.0:
            raise RuntimeError({'D9A3_postclamp_W2_not_exact': j, 'err': err})

        if j in CHECKPOINTS:
            dec = decompose_checkpoint(F, B, C, diag_tokens, diag_ids, v)
            max_decomp_error = max(max_decomp_error, dec['decomp_max_abs'])

            frozen_proj = np.asarray(frozen2_arrays[f'checkpoint_{j}__row_projection'], dtype=np.float64)
            proj_err = float(np.max(np.abs(dec['full']['row_projection'] - frozen_proj)))
            max_full_projection_cal_error = max(max_full_projection_cal_error, proj_err)
            if proj_err > CAL_TOL:
                raise RuntimeError({'D9A3_D9A2R_projection_calibration_failure': j, 'err': proj_err})

            fr = frozen_rows[j]
            summary_errs = [
                abs(dec['full']['a'] - float(fr['signed_mean_amplitude'])),
                abs(dec['full']['rho_v'] - float(fr['rho_v'])),
                abs(dec['full']['frob'] - float(fr['frob_norm'])),
                abs(dec['full']['EV1'] - float(fr['EV1'])),
                abs(dec['full']['cos_v1_to_terminal_v'] - float(fr['cos_v1_to_terminal_v'])),
            ]
            serr = max(summary_errs)
            max_full_summary_cal_error = max(max_full_summary_cal_error, serr)
            if serr > CAL_TOL:
                raise RuntimeError({'D9A3_D9A2R_summary_calibration_failure': j, 'err': serr})

            if j == 64:
                gt0 = np.asarray(diag_tokens[0], dtype=np.int64)
                _depth, cur0, masked0 = d91.diagnostic_contexts(gt0, int(diag_ids[0]), C.c.N, C.c.V)[0]
                for label, sys in (('R', dec['R']), ('C', C), ('P', dec['P']), ('Q', dec['Q'])):
                    pc = d91.parity_check(sys, cur0, masked0)
                    pc.update({'arm': label, 'context': j})
                    parity_checks.append(pc)
                    if not pc['pass']:
                        raise RuntimeError({'D9A3_stateless_parity_failure': pc})

            transplant_checks.append({
                'context': j,
                'P_pass': bool(dec['P_transplant_check']['pass']),
                'Q_pass': bool(dec['Q_transplant_check']['pass']),
            })

            row = {
                'context': int(j),
                'rows': int(dec['full']['rows']),
                'full_a': dec['full']['a'],
                'pair_a': dec['pair']['a'],
                'res_a': dec['res']['a'],
                'release_a': dec['release']['a'],
                'full_frob': dec['full']['frob'],
                'pair_frob': dec['pair']['frob'],
                'res_frob': dec['res']['frob'],
                'release_frob': dec['release']['frob'],
                'full_rho_v': dec['full']['rho_v'],
                'pair_rho_v': dec['pair']['rho_v'],
                'res_rho_v': dec['res']['rho_v'],
                'release_rho_v': dec['release']['rho_v'],
                'full_EV1': dec['full']['EV1'],
                'pair_EV1': dec['pair']['EV1'],
                'res_EV1': dec['res']['EV1'],
                'release_EV1': dec['release']['EV1'],
                'full_cos_v1': dec['full']['cos_v1_to_terminal_v'],
                'pair_cos_v1': dec['pair']['cos_v1_to_terminal_v'],
                'res_cos_v1': dec['res']['cos_v1_to_terminal_v'],
                'release_cos_v1': dec['release']['cos_v1_to_terminal_v'],
                'chi_pair_res': dec['chi_pair_res'],
                'a_pair_times_a_res': dec['a_pair_times_a_res'],
                'opposing_ratio': dec['opposing_ratio'],
                'pair_v_projected_energy_over_full': dec['pair_v_projected_energy_over_full'],
                'pair_release_frob_cos': dec['pair_release_frob_cos'],
                'decomp_max_abs': dec['decomp_max_abs'],
                'D9A2R_full_projection_max_abs_error': proj_err,
                'D9A2R_full_summary_max_abs_error': serr,
                'C_cumulative_W2_clamp_correction': float(cumulative_clamp),
            }
            rows.append(row)

            for name, g in (('full', dec['full']), ('pair', dec['pair']), ('res', dec['res']), ('release', dec['release'])):
                arrays[f'checkpoint_{j}__{name}__row_projection'] = np.asarray(g['row_projection'], dtype=np.float64)
                arrays[f'checkpoint_{j}__{name}__v1'] = np.asarray(g['v1'], dtype=np.float64)
                arrays[f'checkpoint_{j}__{name}__singular_values_sq'] = np.asarray(g['singular_values_sq'], dtype=np.float64)

            print('D9A3_PROGRESS', json.dumps(row), flush=True)

    return {
        'F': F, 'B': B, 'C': C, 'rows': rows, 'arrays': arrays,
        'streams': streams,
        'rng_after': {'F': d3r.jstate(rf), 'B': d3r.jstate(rb), 'C': d3r.jstate(rc)},
        'max_post_clamp_W2_abs_error': float(max_post_clamp),
        'cumulative_W2_clamp_correction': float(cumulative_clamp),
        'max_decomp_error': float(max_decomp_error),
        'max_full_projection_cal_error': float(max_full_projection_cal_error),
        'max_full_summary_cal_error': float(max_full_summary_cal_error),
        'parity_checks': parity_checks,
        'transplant_checks': transplant_checks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--d7a-artifact', type=Path, required=True)
    ap.add_argument('--d7a2-artifact', type=Path, required=True)
    ap.add_argument('--d8a-artifact', type=Path, required=True)
    ap.add_argument('--d8b-artifact', type=Path, required=True)
    ap.add_argument('--d9a1-artifact', type=Path, required=True)
    ap.add_argument('--d9a2r-artifact', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--contexts', type=int, default=1024)
    ap.add_argument('--encode-batch', type=int, default=256)
    ap.add_argument('--seed', type=int, default=7311)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.contexts != 1024:
        raise ValueError('formal D9A.3 requires exactly 1024 contexts')

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    device = torch.device('cpu')

    s72 = json.loads(locate(a.d7a2_artifact, 'D7A2_SUMMARY.json').read_text(encoding='utf-8'))
    s8a = json.loads(locate(a.d8a_artifact, 'D8A_SUMMARY.json').read_text(encoding='utf-8'))
    s8b = json.loads(locate(a.d8b_artifact, 'D8B_SUMMARY.json').read_text(encoding='utf-8'))
    s91 = json.loads(locate(a.d9a1_artifact, 'D9A1_SUMMARY.json').read_text(encoding='utf-8'))
    s92 = json.loads(locate(a.d9a2r_artifact, 'D9A2R_SUMMARY.json').read_text(encoding='utf-8'))

    if s72.get('classification') != 'A1_COMPACT_TWO_BLOCK_COMPENSATION' or list(s72.get('pair', [])) != ['b2', 'bO']:
        raise RuntimeError('D7A.2 pair prerequisite mismatch')
    if s8a.get('classification') != 'I1_SHARED_LOW_DIMENSIONAL_FUNCTIONAL_CANDIDATE':
        raise RuntimeError('D8A prerequisite mismatch')
    if s8b.get('classification') != 'B1_BIDIRECTIONAL_CAUSAL_HARMFUL_FUNCTIONAL_MODE':
        raise RuntimeError('D8B prerequisite mismatch')
    if s91.get('classification') != 'V0_NO_HARMFUL_PLASTIC_VELOCITY_EXCESS':
        raise RuntimeError('D9A.1 prerequisite mismatch')
    if s92.get('classification') != 'T2_LATE_MODE_CONSOLIDATION':
        raise RuntimeError('D9A.2-R prerequisite mismatch')

    d8m = np.load(locate(a.d8a_artifact, 'D8A_MODES.npz'), allow_pickle=False)
    d91arr = np.load(locate(a.d9a1_artifact, 'D9A1_VELOCITY_ARRAYS.npz'), allow_pickle=False)
    d92arr = np.load(locate(a.d9a2r_artifact, 'D9A2R_ARRAYS.npz'), allow_pickle=False)
    basis = np.asarray(d8m['D8B_frozen_y_pair_basis'], dtype=np.float64)
    if basis.shape != (96, 1):
        raise RuntimeError(f'bad basis shape {basis.shape}')
    v = basis[:, 0].copy()
    if abs(float(np.linalg.norm(v)) - 1.0) > 1e-10:
        raise RuntimeError('D8A basis not unit norm')
    if not np.allclose(np.asarray(d91arr['basis_v']), v, rtol=0.0, atol=1e-10):
        raise RuntimeError('D9A.1 basis mismatch')
    if not np.allclose(np.asarray(d92arr['basis_v']), v, rtol=0.0, atol=1e-10):
        raise RuntimeError('D9A.2-R basis mismatch')

    # Preserve exact D9A.1 dtypes because dtype is part of its provenance hash.
    diag_ids = np.asarray(d91arr['diagnostic_ids'])
    diag_labels = np.asarray(d91arr['diagnostic_labels'])
    diag_tokens = np.asarray(d91arr['diagnostic_tokens'])
    diag_hash = d91.sha_arrays(
        diag_ids, diag_labels, diag_tokens,
        np.asarray([d91.DIAG_SEED, d91.DIAG_MASK_SEED], dtype=np.int64)
    )
    if diag_hash != str(s91['diagnostic_probe']['provenance_sha256']):
        raise RuntimeError('D9A.1 diagnostic provenance mismatch')
    if not np.array_equal(np.asarray(d92arr['diagnostic_ids']), diag_ids):
        raise RuntimeError('D9A.2-R diagnostic IDs mismatch')
    if not np.array_equal(np.asarray(d92arr['diagnostic_tokens']), diag_tokens):
        raise RuntimeError('D9A.2-R diagnostic tokens mismatch')

    checkpoint_sha = d3r.sha_file(a.checkpoint)
    source, meta, train_rng, schedule = trainer.load_full_checkpoint(a.checkpoint)
    if checkpoint_sha != d6a.EXPECTED_CHECKPOINT_SHA256 or abs(float(meta.get('context_epoch_equivalent', -1.0)) - 2.0) > 1e-12:
        raise RuntimeError('start checkpoint identity failed')
    offset = int(schedule['offset'])
    indices = np.asarray(schedule['permutation'], dtype=np.int64)[offset:offset + a.contexts]
    rng_state = copy.deepcopy(train_rng.bit_generator.state)
    if len(indices) != 1024:
        raise RuntimeError('checkpoint schedule lacks 1024 contexts')

    xall, yall, _xte, _yte = core.load(a.data)
    x012, y012 = core.subset012(xall, yall)
    xtr, _ytr, _xva, _yva = core.split(x012, y012, .1, 7)
    toknet, _ = base.load_frozen_tokenizer(a.tokenizer, device)
    tr, _ = base.encode_snn(toknet, xtr, device, a.encode_batch)
    tr = np.asarray(tr, dtype=np.int64)
    tok0, pos0, _ = build_input_embeddings(toknet, source.c.D, grid=7)
    start = clone(source)
    upgrade_r6b3_system(start, tok0, pos0, aux_lr_scale=1.0)

    run = replay(start, tr, indices, rng_state, diag_tokens, diag_ids, v, s92, d92arr)
    if not (run['streams']['F'] == run['streams']['B'] == run['streams']['C']):
        raise RuntimeError('stream exactness failed')
    if not (run['rng_after']['F'] == run['rng_after']['B'] == run['rng_after']['C']):
        raise RuntimeError('external RNG exactness failed')
    if run['max_post_clamp_W2_abs_error'] != 0.0 or not np.array_equal(run['C'].W2, run['F'].W2):
        raise RuntimeError('C safe-W2 identity failed')
    if run['max_decomp_error'] > DECOMP_TOL:
        raise RuntimeError('functional decomposition audit failed')
    if run['max_full_projection_cal_error'] > CAL_TOL or run['max_full_summary_cal_error'] > CAL_TOL:
        raise RuntimeError('D9A.2-R full-geometry calibration failed')

    oldF = load_terminal(locate(a.d7a_artifact, 'D7A_F_terminal.pkl'))
    oldB = load_terminal(locate(a.d7a_artifact, 'D7A_B_terminal.pkl'))
    oldC = load_terminal(locate(a.d7a_artifact, 'D7A_C_W2_safe_clamp_terminal.pkl'))
    model_replay = {
        'F': d7a.model_field_compare_tolerant(run['F'], oldF),
        'B': d7a.model_field_compare_tolerant(run['B'], oldB),
        'C': d7a.model_field_compare_tolerant(run['C'], oldC),
    }
    if not all(x['pass'] for x in model_replay.values()):
        raise RuntimeError({'D9A3_terminal_model_replay_failure': model_replay})

    classification, transition = classify(run['rows'])

    write_csv(a.out / 'D9A3_CHECKPOINT_DECOMPOSITION.csv', run['rows'])
    write_csv(a.out / 'D9A3_DIAGNOSTIC_IDS.csv', [
        {'order': i, 'validation_id': int(diag_ids[i]), 'digit': int(diag_labels[i])}
        for i in range(len(diag_ids))
    ])

    npz = {
        'basis_v': v,
        'diagnostic_ids': diag_ids,
        'diagnostic_labels': diag_labels,
        'diagnostic_tokens': diag_tokens,
        'contexts': np.asarray(CHECKPOINTS, dtype=np.int64),
    }
    npz.update(run['arrays'])
    np.savez_compressed(a.out / 'D9A3_ARRAYS.npz', **npz)

    summary = {
        'stage': 'GATE_D9A3_PAIR_VERSUS_RESIDUAL_MODE_CONSOLIDATION_DECOMPOSITION',
        'classification': classification,
        'source_runs': {
            'D7A': 39871827217,
            'D7A2': 39942308903,
            'D8A': 39979543621,
            'D8B': 40021397096,
            'D9A1': 40044084395,
            'D9A2R': 40094105097,
        },
        'training_modified': False,
        'learner_modified': False,
        'intervention_performed': False,
        'additional_block_search_performed': False,
        'pair': ['b2', 'bO'],
        'basis_source': 'frozen D8A/D8B y-space rank-1 basis',
        'probe_source': 'exact frozen D9A.1 128-image diagnostic set and masks',
        'formal_transition_window': [128, 256],
        'thresholds': {
            'terminal_alignment_cos': ALIGN_GATE,
            'pair_majority_growth_share': PAIR_MAJORITY,
            'joint_min_growth_share_each': JOINT_MIN_SHARE,
        },
        'checkpoint_decomposition': run['rows'],
        'transition_128_256': transition,
        'trajectory_provenance': {
            'all_arm_stream_exact': True,
            'all_arm_external_rng_exact': True,
            'C_max_postclamp_W2_abs_error': run['max_post_clamp_W2_abs_error'],
            'C_cumulative_W2_clamp_correction': run['cumulative_W2_clamp_correction'],
            'terminal_model_replay_to_D7A': model_replay,
            'max_full_projection_calibration_error_to_D9A2R': run['max_full_projection_cal_error'],
            'max_full_summary_calibration_error_to_D9A2R': run['max_full_summary_cal_error'],
            'max_decomposition_identity_error': run['max_decomp_error'],
            'stateless_forward_parity': {
                'all_pass': bool(all(x['pass'] for x in run['parity_checks'])),
                'max_abs': float(max(x['max_abs'] for x in run['parity_checks'])),
                'checks': run['parity_checks'],
            },
            'pair_only_transplant_checks': run['transplant_checks'],
            'D9A1_diagnostic_provenance_sha256': diag_hash,
        },
        'decision': {
            'mechanism_classification': classification,
            'controller_authorized': False,
            'additional_block_search_authorized': False,
        },
        'interpretation_boundary': (
            'D9A.3 is a binary functional-geometry decomposition of the already-known b2+bO subsystem versus the residual network. '
            'It does not establish causal trajectory control, does not make b2+bO the only causal parameters, and does not authorize a learning-rule change. '
            'M1/M2/M3 identify which mechanism should receive the next targeted causal intervention; M4 means terminal pair attribution does not explain trajectory consolidation.'
        ),
    }
    (a.out / 'D9A3_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    manifest = {
        'classification': classification,
        'contexts': 1024,
        'checkpoint_contexts': list(CHECKPOINTS),
        'diagnostic_images': int(len(diag_ids)),
        'pair': ['b2', 'bO'],
        'training_modified': False,
        'learner_modified': False,
        'intervention_performed': False,
        'additional_block_search_performed': False,
        'protected_baseline_modified': False,
        'no_PR_no_merge': True,
    }
    (a.out / 'D9A3_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print('D9A3_RESULT', json.dumps({
        'classification': classification,
        'transition_128_256': transition,
        'checkpoint_decomposition': run['rows'],
        'provenance': {
            'max_decomp_error': run['max_decomp_error'],
            'max_D9A2R_projection_cal_error': run['max_full_projection_cal_error'],
            'cumulative_W2_clamp_correction': run['cumulative_W2_clamp_correction'],
        },
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
