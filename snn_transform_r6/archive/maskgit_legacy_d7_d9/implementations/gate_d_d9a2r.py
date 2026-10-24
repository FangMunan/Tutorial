#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D9A.2-R — matched trajectory-relative functional displacement audit.

Diagnostic-only replay of frozen D7A F/B/C. At each frozen checkpoint, construct
R_t = B_t with W2 replaced by time-matched F_t W2, then measure C_t - R_t
geometry in the frozen D8A/D8B rank-1 y-space basis and the historical Gate-A
deployment gap. No learner update, controller, projection, or mode refit occurs.
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
from r6b4_full_forward_plasticity import upgrade_r6b3_system

CHECKPOINTS = (64, 128, 256, 512, 768, 1024)
PRIMARY = ('F61', 'D_non61')
RHO_GATE = 0.50
COS_GATE = 0.80
ENERGY_TOL = 1e-20
GATE_TOL = 1e-7


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
                seen.add(k); keys.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def load_terminal(path: Path):
    with path.open('rb') as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or 'system' not in obj:
        raise RuntimeError(f'bad terminal schema: {path}')
    return obj['system']


def geometry_at(F, B, C, diag_tokens, diag_ids, v):
    R = clone(B)
    R.W2 = np.array(F.W2, copy=True)
    w2_err = d7a.max_abs(C.W2, R.W2)
    if w2_err != 0.0:
        raise RuntimeError({'D9A2R_W2_identity_failed': w2_err})

    sr = d91.snapshot_forward(R)
    sc = d91.snapshot_forward(C)
    chunks = []
    for i, did in enumerate(np.asarray(diag_ids, dtype=np.int64)):
        gt = np.asarray(diag_tokens[i], dtype=np.int64)
        for _depth, cur, masked in d91.diagnostic_contexts(gt, int(did), C.c.N, C.c.V):
            yr = d91.stateless_y(sr, C.c, cur, masked)
            yc = d91.stateless_y(sc, C.c, cur, masked)
            chunks.append(np.asarray(yc - yr, dtype=np.float64))
    D = np.vstack(chunks) if chunks else np.empty((0, C.c.D), dtype=np.float64)
    if D.shape[0] < 2:
        raise RuntimeError('insufficient diagnostic rows')

    proj = D @ np.asarray(v, dtype=np.float64)
    energy = float(np.sum(D * D))
    proj_energy = float(np.sum(proj * proj))
    rho = float(proj_energy / (energy + 1e-30))
    frob = float(math.sqrt(energy))
    mean_vec = np.mean(D, axis=0)
    mean_energy_fraction = float(D.shape[0] * np.dot(mean_vec, mean_vec) / (energy + 1e-30))

    G = D.T @ D
    G = 0.5 * (G + G.T)
    vals, vecs = np.linalg.eigh(G)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(0.0, vals[order])
    vecs = vecs[:, order]
    total = float(np.sum(vals))
    ev1 = float(vals[0] / (total + 1e-30))
    v1 = np.asarray(vecs[:, 0], dtype=np.float64)
    cos = float(abs(np.dot(v1, v)))
    present = bool(rho >= RHO_GATE and cos >= COS_GATE)

    return {
        'R': R,
        'rows': int(D.shape[0]),
        'signed_mean_amplitude': float(np.mean(proj)),
        'signed_median_amplitude': float(np.median(proj)),
        'projection_std': float(np.std(proj)),
        'rho_v': rho,
        'total_energy': energy,
        'frob_norm': frob,
        'mean_displacement_energy_fraction': mean_energy_fraction,
        'EV1': ev1,
        'cos_v1_to_terminal_v': cos,
        'frozen_mode_present': present,
        'W2_C_R_max_abs': float(w2_err),
        'v1': v1,
        'singular_values_sq': vals,
        'row_projection': proj,
    }


def gate_gap(R, C, schedules, gt_gate, shared_gate, nboot, seed):
    evR = d6a.gate_eval('D9A2R_R', clone(R), schedules, gt_gate, shared_gate)
    evC = d6a.gate_eval('D9A2R_C', clone(C), schedules, gt_gate, shared_gate)
    gaps = {}
    diffs = {}
    for i, m in enumerate(PRIMARY):
        x = np.asarray(evC['effects'][m], dtype=np.float64) - np.asarray(evR['effects'][m], dtype=np.float64)
        gaps[m] = d6a.bootstrap_mean(x, seed + i, nboot)
        diffs[m] = x
    return evR, evC, gaps, diffs


def classify(rows):
    present = [bool(r['frozen_mode_present']) for r in rows]
    by = {int(r['context']): bool(r['frozen_mode_present']) for r in rows}
    energy = {int(r['context']): float(r['total_energy']) for r in rows}
    if by[64] and by[128] and sum(present) >= 5 and by[1024]:
        return 'T1_FIXED_RELATIVE_MODE_ACCUMULATION'
    if (not by[64]) and (not by[128]) and by[768] and by[1024] and any(
        energy[c] > ENERGY_TOL for c in (64, 128, 256, 512)
    ):
        return 'T2_LATE_MODE_CONSOLIDATION'
    if by[1024] and not any(by[c] for c in (64, 128, 256, 512, 768)):
        return 'T4_TERMINAL_ONLY_CAUSAL_GEOMETRY'
    return 'T3_ROTATING_OR_PATH_DEPENDENT_RELATIVE_GEOMETRY'


def run_replay(start, tokens, indices, rng_state, diag_tokens, diag_ids, v,
               schedules, gt_gate, shared_gate, nboot):
    F = d6a.make_d6a(start, d7a.F_LABEL)
    B = d6a.make_d6a(start, d7a.B_LABEL)
    C = d6a.make_d6a(start, d7a.B_LABEL)
    rf = gd.clone_rng(rng_state); rb = gd.clone_rng(rng_state); rc = gd.clone_rng(rng_state)
    streams = {'F': [], 'B': [], 'C': []}
    rows, gap_rows = [], []
    arrays = {}
    max_post_clamp = 0.0
    cumulative_clamp = 0.0
    terminal_gate = None

    for j, image_index in enumerate(indices, start=1):
        if d7a.max_abs(C.W2, F.W2) != 0.0:
            raise RuntimeError({'D9A2R_precontext_W2_not_safe': j})

        for s, r in ((F, rf), (B, rb), (C, rc)):
            before = d3r.jstate(r)
            d3r.reset_boundary(s, 'F_EM')
            if d3r.jstate(r) != before:
                raise RuntimeError('reset consumed external RNG')

        truth = np.asarray(tokens[int(image_index)], dtype=np.int64)
        _recF, evF = d6a.run_context(F, j, int(image_index), truth, rf)
        _recB, evB = d6a.run_context(B, j, int(image_index), truth, rb)
        _recC, evC = d6a.run_context(C, j, int(image_index), truth, rc)

        sF = d6a.stream_row(j, int(image_index), evF)
        sB = d6a.stream_row(j, int(image_index), evB)
        sC = d6a.stream_row(j, int(image_index), evC)
        if not (sF == sB == sC):
            raise RuntimeError({'D9A2R_stream_divergence': j})
        if not (d3r.jstate(rf) == d3r.jstate(rb) == d3r.jstate(rc)):
            raise RuntimeError({'D9A2R_rng_divergence': j})
        streams['F'].append(sF); streams['B'].append(sB); streams['C'].append(sC)

        correction = np.asarray(F.W2, dtype=np.float64) - np.asarray(C.W2, dtype=np.float64)
        cumulative_clamp += float(np.linalg.norm(correction))
        C.W2 = np.array(F.W2, copy=True)
        err = d7a.max_abs(C.W2, F.W2)
        max_post_clamp = max(max_post_clamp, err)
        if err != 0.0:
            raise RuntimeError({'D9A2R_postclamp_W2_not_exact': j, 'err': err})

        if j in CHECKPOINTS:
            g = geometry_at(F, B, C, diag_tokens, diag_ids, v)
            evR, evCg, gaps, diffs = gate_gap(
                g['R'], C, schedules, gt_gate, shared_gate, nboot, 3700000 + j * 10
            )
            row = {
                'context': int(j),
                'rows': g['rows'],
                'signed_mean_amplitude': g['signed_mean_amplitude'],
                'signed_median_amplitude': g['signed_median_amplitude'],
                'projection_std': g['projection_std'],
                'rho_v': g['rho_v'],
                'total_energy': g['total_energy'],
                'frob_norm': g['frob_norm'],
                'mean_displacement_energy_fraction': g['mean_displacement_energy_fraction'],
                'EV1': g['EV1'],
                'cos_v1_to_terminal_v': g['cos_v1_to_terminal_v'],
                'frozen_mode_present': g['frozen_mode_present'],
                'W2_C_R_max_abs': g['W2_C_R_max_abs'],
                'C_cumulative_W2_clamp_correction': float(cumulative_clamp),
            }
            for m in PRIMARY:
                row[f'{m}_gap_mean'] = gaps[m]['mean']
                row[f'{m}_gap_ci95_lo'] = gaps[m]['ci95'][0]
                row[f'{m}_gap_ci95_hi'] = gaps[m]['ci95'][1]
                row[f'{m}_gap_ci_entirely_positive'] = bool(gaps[m]['ci95'][0] > 0)
                gap_rows.append({
                    'context': int(j), 'metric': m,
                    'mean': gaps[m]['mean'],
                    'ci95_lo': gaps[m]['ci95'][0],
                    'ci95_hi': gaps[m]['ci95'][1],
                    'ci_entirely_positive': bool(gaps[m]['ci95'][0] > 0),
                })
                arrays[f'checkpoint_{j}__{m}__C_minus_R'] = np.asarray(diffs[m], dtype=np.float64)
            arrays[f'checkpoint_{j}__row_projection'] = np.asarray(g['row_projection'], dtype=np.float64)
            arrays[f'checkpoint_{j}__v1'] = np.asarray(g['v1'], dtype=np.float64)
            arrays[f'checkpoint_{j}__singular_values_sq'] = np.asarray(g['singular_values_sq'], dtype=np.float64)
            rows.append(row)
            if j == 1024:
                terminal_gate = (evR, evCg)
            print('D9A2R_PROGRESS', json.dumps(row), flush=True)

    return {
        'F': F, 'B': B, 'C': C,
        'rows': rows, 'gap_rows': gap_rows, 'arrays': arrays,
        'streams': streams,
        'rng_after': {'F': d3r.jstate(rf), 'B': d3r.jstate(rb), 'C': d3r.jstate(rc)},
        'max_post_clamp_W2_abs_error': float(max_post_clamp),
        'cumulative_W2_clamp_correction': float(cumulative_clamp),
        'terminal_gate': terminal_gate,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--d7a-artifact', type=Path, required=True)
    ap.add_argument('--d8a-artifact', type=Path, required=True)
    ap.add_argument('--d8b-artifact', type=Path, required=True)
    ap.add_argument('--d9a-artifact', type=Path, required=True)
    ap.add_argument('--d9a1-artifact', type=Path, required=True)
    ap.add_argument('--anchor-artifact', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--contexts', type=int, default=1024)
    ap.add_argument('--bootstrap', type=int, default=10000)
    ap.add_argument('--encode-batch', type=int, default=256)
    ap.add_argument('--seed', type=int, default=7311)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if a.contexts != 1024:
        raise ValueError('formal D9A.2-R requires exactly 1024 contexts')

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads)
    device = torch.device('cpu')

    s7 = json.loads(locate(a.d7a_artifact, 'D7A_SUMMARY.json').read_text(encoding='utf-8'))
    s8a = json.loads(locate(a.d8a_artifact, 'D8A_SUMMARY.json').read_text(encoding='utf-8'))
    s8b = json.loads(locate(a.d8b_artifact, 'D8B_SUMMARY.json').read_text(encoding='utf-8'))
    s9 = json.loads(locate(a.d9a_artifact, 'D9A_SUMMARY.json').read_text(encoding='utf-8'))
    s91 = json.loads(locate(a.d9a1_artifact, 'D9A1_SUMMARY.json').read_text(encoding='utf-8'))
    if s7.get('classification') != 'T0_NO_W2_TRAJECTORY_RESCUE':
        raise RuntimeError('D7A prerequisite mismatch')
    if s8a.get('classification') != 'I1_SHARED_LOW_DIMENSIONAL_FUNCTIONAL_CANDIDATE':
        raise RuntimeError('D8A prerequisite mismatch')
    selected_k = s8a.get('selected_k', s8a.get('primary_y_gate', {}).get('selected_k'))
    if int(selected_k) != 1:
        raise RuntimeError('D8A selected k mismatch')
    if s8b.get('classification') != 'B1_BIDIRECTIONAL_CAUSAL_HARMFUL_FUNCTIONAL_MODE':
        raise RuntimeError('D8B prerequisite mismatch')
    if s9.get('classification') != 'R0_NO_HARD_FUNCTIONAL_TRAJECTORY_RESCUE':
        raise RuntimeError('D9A prerequisite mismatch')
    if s91.get('classification') != 'V0_NO_HARMFUL_PLASTIC_VELOCITY_EXCESS':
        raise RuntimeError('D9A.1 prerequisite mismatch')

    d8m = np.load(locate(a.d8a_artifact, 'D8A_MODES.npz'), allow_pickle=False)
    d8be = np.load(locate(a.d8b_artifact, 'D8B_EFFECT_ARRAYS.npz'), allow_pickle=False)
    d91arr = np.load(locate(a.d9a1_artifact, 'D9A1_VELOCITY_ARRAYS.npz'), allow_pickle=False)
    basis = np.asarray(d8m['D8B_frozen_y_pair_basis'], dtype=np.float64)
    if basis.shape != (96, 1):
        raise RuntimeError(f'bad frozen basis shape {basis.shape}')
    v = basis[:, 0].copy()
    if abs(float(np.linalg.norm(v)) - 1.0) > 1e-10:
        raise RuntimeError('D8A basis not unit norm')
    if not np.allclose(np.asarray(d8be['frozen_y_basis'])[:, 0], v, rtol=0.0, atol=1e-10):
        raise RuntimeError('D8A/D8B basis mismatch')
    if not np.allclose(np.asarray(d91arr['basis_v']), v, rtol=0.0, atol=1e-10):
        raise RuntimeError('D9A.1 basis mismatch')

    diag_ids = np.asarray(d91arr['diagnostic_ids'], dtype=np.int64)
    diag_labels = np.asarray(d91arr['diagnostic_labels'], dtype=np.int8)
    diag_tokens = np.asarray(d91arr['diagnostic_tokens'], dtype=np.int64)
    diag_hash = d91.sha_arrays(
        diag_ids, diag_labels, diag_tokens,
        np.asarray([d91.DIAG_SEED, d91.DIAG_MASK_SEED], dtype=np.int64)
    )
    expected_diag_hash = str(s91['diagnostic_probe']['provenance_sha256'])
    if diag_hash != expected_diag_hash:
        raise RuntimeError('D9A.1 diagnostic provenance hash mismatch')

    validation_ids = np.asarray(d8be['validation_ids'], dtype=np.int64)
    gt_gate = np.asarray(d8be['gt_tokens'], dtype=np.int64)
    shared_gate = np.asarray(d8be['shared_uniform'], dtype=np.float64)
    if np.intersect1d(diag_ids, validation_ids).size:
        raise RuntimeError('D9A.2-R diagnostic/Gate-A overlap')

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
    xtr, _ytr, xva, yva = core.split(x012, y012, .1, 7)
    toknet, _ = base.load_frozen_tokenizer(a.tokenizer, device)
    tr, _ = base.encode_snn(toknet, xtr, device, a.encode_batch)
    va, _ = base.encode_snn(toknet, xva, device, a.encode_batch)
    tr = np.asarray(tr, dtype=np.int64); va = np.asarray(va, dtype=np.int64)
    if not np.array_equal(va[validation_ids], gt_gate):
        raise RuntimeError('Gate-A coordinate mismatch')
    if not np.array_equal(va[diag_ids], diag_tokens):
        raise RuntimeError('D9A.1 diagnostic token mismatch')

    tok0, pos0, _ = build_input_embeddings(toknet, source.c.D, grid=7)
    start = clone(source); upgrade_r6b3_system(start, tok0, pos0, aux_lr_scale=1.0)

    sample_gt = diag_tokens[0]
    _depth, sample_cur, sample_masked = d91.diagnostic_contexts(
        sample_gt, int(diag_ids[0]), start.c.N, start.c.V
    )[0]
    parity = d91.parity_check(start, sample_cur, sample_masked)
    if not parity['pass']:
        raise RuntimeError({'stateless_y_parity_failed': parity})

    schedules = json.loads(locate(a.anchor_artifact, 'HISTORICAL_R6B3_R3_T8_REFERENCE_TRACE.json').read_text(encoding='utf-8'))
    run = run_replay(
        start, tr, indices, rng_state, diag_tokens, diag_ids, v,
        schedules, gt_gate, shared_gate, a.bootstrap
    )

    if not (run['streams']['F'] == run['streams']['B'] == run['streams']['C']):
        raise RuntimeError('stream exactness failed')
    if not (run['rng_after']['F'] == run['rng_after']['B'] == run['rng_after']['C']):
        raise RuntimeError('external RNG exactness failed')
    if run['max_post_clamp_W2_abs_error'] != 0.0:
        raise RuntimeError('post-clamp W2 exactness failed')

    frozenF = load_terminal(locate(a.d7a_artifact, 'D7A_F_terminal.pkl'))
    frozenB = load_terminal(locate(a.d7a_artifact, 'D7A_B_terminal.pkl'))
    frozenC = load_terminal(locate(a.d7a_artifact, 'D7A_C_W2_safe_clamp_terminal.pkl'))
    model_replay = {
        'F': d7a.model_field_compare_tolerant(run['F'], frozenF),
        'B': d7a.model_field_compare_tolerant(run['B'], frozenB),
        'C': d7a.model_field_compare_tolerant(run['C'], frozenC),
    }
    if not all(x['pass'] for x in model_replay.values()):
        raise RuntimeError({'terminal_model_replay_failed': model_replay})

    evR1024, evC1024 = run['terminal_gate']
    calR = d7a.gate_calibration(evR1024, d8be, 'R', GATE_TOL)
    calC = d7a.gate_calibration(evC1024, d8be, 'C', GATE_TOL)
    if not (calR['pass'] and calC['pass']):
        raise RuntimeError({'terminal_D8B_gate_calibration_failed': {'R': calR, 'C': calC}})

    classification = classify(run['rows'])
    present_sequence = [bool(r['frozen_mode_present']) for r in run['rows']]
    summary = {
        'stage': 'D9A.2-R',
        'classification': classification,
        'training_modified': False,
        'learner_modified': False,
        'intervention_performed': False,
        'mode_refit_performed': False,
        'basis_source': 'D8A/D8B frozen rank-1 pair-derived y-space basis',
        'probe_source': 'exact D9A.1 independent 128-image diagnostic set',
        'mode_presence_thresholds': {'rho_v_min': RHO_GATE, 'cos_v1_to_terminal_v_min': COS_GATE},
        'checkpoint_contexts': list(CHECKPOINTS),
        'present_sequence': present_sequence,
        'stateless_forward_parity': parity,
        'diagnostic_probe': {
            'images': int(len(diag_ids)),
            'provenance_sha256': diag_hash,
            'GateA_overlap_count': int(np.intersect1d(diag_ids, validation_ids).size),
        },
        'trajectory_provenance': {
            'all_arm_stream_exact': True,
            'all_arm_external_rng_exact': True,
            'max_post_clamp_W2_abs_error': run['max_post_clamp_W2_abs_error'],
            'C_cumulative_W2_clamp_correction': run['cumulative_W2_clamp_correction'],
            'terminal_model_replay_to_D7A': model_replay,
            'terminal_gate_calibration_to_D8B': {'R': calR, 'C': calC},
        },
        'checkpoint_geometry': run['rows'],
        'deployment_gaps': run['gap_rows'],
        'decision': {
            'T1_fixed_relative_mode_accumulation': classification == 'T1_FIXED_RELATIVE_MODE_ACCUMULATION',
            'T2_late_mode_consolidation': classification == 'T2_LATE_MODE_CONSOLIDATION',
            'T3_rotating_or_path_dependent': classification == 'T3_ROTATING_OR_PATH_DEPENDENT_RELATIVE_GEOMETRY',
            'T4_terminal_only': classification == 'T4_TERMINAL_ONLY_CAUSAL_GEOMETRY',
            'controller_authorized': False,
        },
        'interpretation_boundary': (
            'This gate characterizes time-matched C_t-R_t functional geometry only. '
            'It does not prove a deployable online controller, an evolving local learning rule, '
            'or parameterization invariance across architectures.'
        ),
    }

    write_csv(a.out / 'D9A2R_CHECKPOINT_GEOMETRY.csv', run['rows'])
    write_csv(a.out / 'D9A2R_DEPLOYMENT_GAPS.csv', run['gap_rows'])
    write_csv(a.out / 'D9A2R_DIAGNOSTIC_IDS.csv', [
        {'order': i, 'validation_id': int(x), 'digit': int(diag_labels[i])}
        for i, x in enumerate(diag_ids)
    ])
    arrays = {
        'basis_v': np.asarray(v, dtype=np.float64),
        'diagnostic_ids': diag_ids,
        'diagnostic_labels': diag_labels,
        'diagnostic_tokens': diag_tokens.astype(np.int16),
        'contexts': np.asarray(CHECKPOINTS, dtype=np.int64),
        'present_sequence': np.asarray(present_sequence, dtype=np.int8),
    }
    arrays.update(run['arrays'])
    np.savez_compressed(a.out / 'D9A2R_ARRAYS.npz', **arrays)
    (a.out / 'D9A2R_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    manifest = {
        'stage': 'D9A.2-R',
        'classification': classification,
        'checkpoint_sha256': checkpoint_sha,
        'D9A1_diagnostic_provenance_sha256': diag_hash,
        'source_runs': {
            'D7A': 39871827217,
            'D8A': 39979543621,
            'D8B': 40021397096,
            'D9A': 40037197274,
            'D9A1': 40044084395,
        },
        'files': [
            'D9A2R_SUMMARY.json', 'D9A2R_CHECKPOINT_GEOMETRY.csv',
            'D9A2R_DEPLOYMENT_GAPS.csv', 'D9A2R_DIAGNOSTIC_IDS.csv',
            'D9A2R_ARRAYS.npz'
        ],
    }
    (a.out / 'D9A2R_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('D9A2R_FINAL', json.dumps({
        'classification': classification,
        'present_sequence': present_sequence,
        'checkpoints': run['rows'],
    }), flush=True)


if __name__ == '__main__':
    main()
