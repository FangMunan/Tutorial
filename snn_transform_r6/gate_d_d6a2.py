#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D6A.2 — terminal suppressor mapping of beneficial bo_B.

Eval-only terminal causal map. No training or learner modification occurs.
Primary reference H = F_EM with bo transplanted from the combined D6A arm B.
Each non-bo forward block is tested bidirectionally for suppressor sufficiency
and necessity using the frozen Gate-A coordinate system.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch

import gate_d_d6a as d6a
import gate_d_d6a1 as d6a1
import gate_d_d3e_runtime as d3r
import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
import r6b3_decoder_operating_point_audit as coarse
import r6b3_decoder_spatial_refinement_audit as spatial
from gate_d_d3e_eval import BASE_GEN_SEED, eval_masked, rollout_js
from r6b4_full_forward_plasticity import FORWARD_PARAMETER_KEYS

TOL = 1e-7
PRIMARY = ('F61', 'D_non61')
SUPPORTIVE = ('C61', 'delta_P61')
ALL_METRICS = ('F61', 'D_non61', 'C61', 'delta_P61', 'CE')
MASK_RATIOS = ('0.25', '0.5', '0.9', '1.0')
JS_KEYS = ('global_js_nats', 'spatial_js_nats', 'pair_js_nats')

BLOCKS = {
    'Q': ('WQ', 'bq'),
    'K': ('WK', 'bk'),
    'V': ('WV', 'bv'),
    'O_W': ('WO',),
    'Norm': ('gamma2', 'beta2'),
    'FF1': ('W1', 'b1'),
    'W2': ('W2',),
    'b2': ('b2',),
    'R': ('WR',),
    'Emb': ('tok', 'pos'),
}

F_LABEL = 'F_EM'
A_LABEL = 'F_EM_bo_oracle'
B_LABEL = 'F_EM_WO_zero_bo_oracle'


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
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def set_fields(dst, donor, fields):
    for k in fields:
        setattr(dst, k, np.array(getattr(donor, k), copy=True))


def hybrid(background, donor, fields):
    s = clone(background)
    set_fields(s, donor, fields)
    return s


def arr_exact(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return bool(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b))


def forward_exact(a, b):
    rows = {}
    ok = True
    for k in FORWARD_PARAMETER_KEYS:
        eq = arr_exact(getattr(a, k), getattr(b, k))
        rows[k] = eq
        ok = ok and eq
    return {'pass': bool(ok), 'fields': rows}


def gate_eval(name, system, schedules, gt, shared):
    return d6a.gate_eval(name, system, schedules, gt, shared)


def contrast(ev1, ev0, metric, seed, nboot):
    x = (np.asarray(ev1['effects'][metric], dtype=np.float64)
         - np.asarray(ev0['effects'][metric], dtype=np.float64))
    d = d6a.bootstrap_mean(x, seed, nboot)
    lo, hi = d['ci95']
    d['ci_entirely_positive'] = bool(lo is not None and lo > 0)
    d['ci_entirely_negative'] = bool(hi is not None and hi < 0)
    return d


def calibrate_effects(ev, stored, prefix, tol=TOL):
    out = {
        'self_metrics_max_abs': d6a.nanaware_max_abs(ev['self_metrics'], stored[f'{prefix}__self_metrics']),
        'teacher_metrics_max_abs': d6a.nanaware_max_abs(ev['teacher_metrics'], stored[f'{prefix}__teacher_metrics']),
        'self_proposal_exact': bool(np.array_equal(ev['self_proposal'], stored[f'{prefix}__self_proposal'])),
        'teacher_proposal_exact': bool(np.array_equal(ev['teacher_proposal'], stored[f'{prefix}__teacher_proposal'])),
        'effects': {},
    }
    for m in ALL_METRICS:
        out['effects'][m] = d6a.nanaware_max_abs(ev['effects'][m], stored[f'{prefix}__effect__{m}'])
    out['pass'] = bool(
        out['self_metrics_max_abs'] < tol
        and out['teacher_metrics_max_abs'] < tol
        and out['self_proposal_exact']
        and out['teacher_proposal_exact']
        and max(out['effects'].values()) < tol
    )
    return out


def calibrate_d6a1_effects(ev, stored, label, tol=TOL):
    # D6A.1 ARRAY file stores per-metric effect arrays for terminal systems.
    out = {'effects': {}}
    for m in ALL_METRICS:
        out['effects'][m] = d6a.nanaware_max_abs(ev['effects'][m], stored[f't1024__{label}__{m}'])
    out['pass'] = bool(max(out['effects'].values()) < tol)
    return out


def secondary_eval(label, system, va, real_p, real_pos_p, real_pair, seed, eval_images, gen_samples):
    masked = eval_masked(system, va, seed, eval_images)
    rollout = rollout_js(system, real_p, real_pos_p, real_pair, BASE_GEN_SEED, gen_samples)
    if not rollout.get('all_trace_audits_pass', False):
        raise RuntimeError(f'rollout trace audit failed: {label}')
    return {'masked': masked, 'rollout': rollout}


def score_block(result, loads):
    vals = []
    for m in PRIMARY:
        L = float(loads[m]['mean'])
        if abs(L) > 1e-30:
            vals.append(float(result['sufficiency'][m]['mean']) / L)
            vals.append(float(result['necessity'][m]['mean']) / L)
    return float(np.mean(vals)) if vals else float('-inf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--d6a-artifact', type=Path, required=True)
    ap.add_argument('--d6a1-artifact', type=Path, required=True)
    ap.add_argument('--anchor-artifact', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--eval-images', type=int, default=256)
    ap.add_argument('--gen-samples', type=int, default=256)
    ap.add_argument('--encode-batch', type=int, default=256)
    ap.add_argument('--bootstrap', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=7311)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.eval_images != 256:
        raise ValueError('formal D6A.2 requires exact 256-image Gate-A evaluation')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device('cpu')

    # Frozen prerequisite chain.
    d6s = json.loads(locate(args.d6a_artifact, 'D6A_SUMMARY.json').read_text(encoding='utf-8'))
    if d6s.get('classification') != 'R0_NO_REPAIR_SUFFICIENCY':
        raise RuntimeError('D6A prerequisite is not frozen R0')
    d61s = json.loads(locate(args.d6a1_artifact, 'D6A1_SUMMARY.json').read_text(encoding='utf-8'))
    if d61s.get('classification') != 'I3_NO_INTRINSIC_BO_REPAIR_DEGRADATION':
        raise RuntimeError('D6A.1 prerequisite is not frozen I3')

    stored6 = np.load(locate(args.d6a_artifact, 'D6A_EFFECT_ARRAYS.npz'), allow_pickle=False)
    stored61 = np.load(locate(args.d6a1_artifact, 'D6A1_ARRAYS.npz'), allow_pickle=False)
    schedules = json.loads(locate(args.anchor_artifact, 'HISTORICAL_R6B3_R3_T8_REFERENCE_TRACE.json').read_text(encoding='utf-8'))

    F = d6a1.load_d6a_terminal(locate(args.d6a_artifact, 'D6A_F_EM_terminal.pkl'))
    A = d6a1.load_d6a_terminal(locate(args.d6a_artifact, 'D6A_F_EM_bo_oracle_terminal.pkl'))
    B = d6a1.load_d6a_terminal(locate(args.d6a_artifact, 'D6A_F_EM_WO_zero_bo_oracle_terminal.pkl'))

    expected = {'tok','pos','WQ','WK','WV','WO','W1','W2','WR','bq','bk','bv','bo','b1','b2','gamma2','beta2'}
    if set(FORWARD_PARAMETER_KEYS) != expected:
        raise RuntimeError({'forward_parameter_key_set_changed': sorted(FORWARD_PARAMETER_KEYS)})

    partition_fields = [k for fields in BLOCKS.values() for k in fields]
    expected_non_bo = set(FORWARD_PARAMETER_KEYS) - {'bo'}
    partition_ok = bool(len(partition_fields) == len(set(partition_fields)) and set(partition_fields) == expected_non_bo)
    partition_report = {
        'pass': partition_ok,
        'expected_non_bo': sorted(expected_non_bo),
        'covered': sorted(set(partition_fields)),
        'duplicates': sorted({k for k in partition_fields if partition_fields.count(k) > 1}),
    }
    if not partition_ok:
        summary = {'stage': 'GATE_D6A2_TERMINAL_SUPPRESSOR_MAP', 'classification': 'S4_PARTITION_FAILURE',
                   'partition': partition_report, 'learner_modified': False}
        (args.out/'D6A2_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        raise RuntimeError({'D6A2_partition_failure': partition_report})

    # H has all F forward state except beneficial bo_B.
    H = hybrid(F, B, ('bo',))
    all_non_bo = tuple(partition_fields)
    H_to_B = hybrid(H, B, all_non_bo)
    B_to_H = hybrid(B, F, all_non_bo)
    closure = {
        'H_all_B_equals_B': forward_exact(H_to_B, B),
        'B_all_F_equals_H': forward_exact(B_to_H, H),
        'partition': partition_report,
    }
    closure['pass'] = bool(closure['H_all_B_equals_B']['pass'] and closure['B_all_F_equals_H']['pass'] and partition_ok)
    if not closure['pass']:
        summary = {'stage': 'GATE_D6A2_TERMINAL_SUPPRESSOR_MAP', 'classification': 'S4_PARTITION_FAILURE',
                   'closure': closure, 'learner_modified': False}
        (args.out/'D6A2_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        raise RuntimeError({'D6A2_closure_failure': closure})

    gt = np.asarray(stored6['gt_tokens'], dtype=np.int64)
    shared = np.asarray(stored6['shared_uniform'], dtype=np.float64)
    validation_ids = np.asarray(stored6['validation_ids'], dtype=np.int64)
    if len(gt) != 256 or len(schedules) != 256:
        raise RuntimeError('frozen Gate-A coordinate system is not 256 images')

    # Build all primary Gate-A systems.
    systems = {'F': F, 'A': A, 'B': B, 'H': H}
    for block, fields in BLOCKS.items():
        systems[f'H_plus_{block}'] = hybrid(H, B, fields)
        systems[f'B_minus_{block}'] = hybrid(B, F, fields)

    gate = {}
    for label, system in systems.items():
        gate[label] = gate_eval(label, system, schedules, gt, shared)
        print('D6A2_GATE', json.dumps({'system': label}), flush=True)

    # Provenance: reproduce frozen D6A and D6A.1 terminal effects.
    calibration = {
        'F_to_D6A': calibrate_effects(gate['F'], stored6, F_LABEL),
        'A_to_D6A': calibrate_effects(gate['A'], stored6, A_LABEL),
        'B_to_D6A': calibrate_effects(gate['B'], stored6, B_LABEL),
        'H_to_D6A1': calibrate_d6a1_effects(gate['H'], stored61, 'boB_on_F'),
    }
    if not all(v['pass'] for v in calibration.values()):
        raise RuntimeError({'D6A2_terminal_calibration_failed': calibration})

    # Total suppression load B-H.
    loads = {}
    for mi, m in enumerate(ALL_METRICS):
        loads[m] = contrast(gate['B'], gate['H'], m, 1200000 + mi, args.bootstrap)

    # Single-block bidirectional map.
    block_results = {}
    rows = []
    for bi, (block, fields) in enumerate(BLOCKS.items()):
        br = {'fields': list(fields), 'sufficiency': {}, 'necessity': {}}
        for mi, m in enumerate(ALL_METRICS):
            suff = contrast(gate[f'H_plus_{block}'], gate['H'], m, 1210000 + 1000*bi + 10*mi, args.bootstrap)
            nec = contrast(gate['B'], gate[f'B_minus_{block}'], m, 1220000 + 1000*bi + 10*mi, args.bootstrap)
            L = float(loads[m]['mean']) if loads[m]['mean'] is not None else float('nan')
            suff['fraction_of_total_load'] = None if not np.isfinite(L) or abs(L) <= 1e-30 else float(suff['mean']/L)
            nec['fraction_of_total_load'] = None if not np.isfinite(L) or abs(L) <= 1e-30 else float(nec['mean']/L)
            br['sufficiency'][m] = suff
            br['necessity'][m] = nec
        br['bidirectional_primary'] = bool(all(
            br['sufficiency'][m]['ci_entirely_positive'] and br['necessity'][m]['ci_entirely_positive']
            for m in PRIMARY))
        nec_both = bool(all(br['necessity'][m]['ci_entirely_positive'] for m in PRIMARY))
        suff_both = bool(all(br['sufficiency'][m]['ci_entirely_positive'] for m in PRIMARY))
        br['necessity_only_primary'] = bool(nec_both and not suff_both)
        br['rank_score'] = score_block(br, loads)
        block_results[block] = br
        row = {'block': block, 'fields': '+'.join(fields), 'bidirectional_primary': br['bidirectional_primary'],
               'necessity_only_primary': br['necessity_only_primary'], 'rank_score': br['rank_score']}
        for m in PRIMARY:
            row[f'S_{m}'] = br['sufficiency'][m]['mean']
            row[f'S_{m}_lo'] = br['sufficiency'][m]['ci95'][0]
            row[f'S_{m}_hi'] = br['sufficiency'][m]['ci95'][1]
            row[f'N_{m}'] = br['necessity'][m]['mean']
            row[f'N_{m}_lo'] = br['necessity'][m]['ci95'][0]
            row[f'N_{m}_hi'] = br['necessity'][m]['ci95'][1]
        rows.append(row)

    bidirectional = [b for b, r in block_results.items() if r['bidirectional_primary']]
    necessity_only = [b for b, r in block_results.items() if r['necessity_only_primary']]
    positive_dual_load = bool(all(loads[m]['mean'] is not None and loads[m]['mean'] > 0 for m in PRIMARY))
    if bidirectional:
        classification = 'S1_DOMINANT_SINGLE_SUPPRESSOR'
    elif necessity_only:
        classification = 'S2_CONTEXT_DEPENDENT_SUPPRESSOR'
    elif positive_dual_load:
        classification = 'S3_DISTRIBUTED_HIGHER_ORDER_SUPPRESSION'
    else:
        classification = 'SX_NO_POSITIVE_SUPPRESSION_LOAD'

    ranked = sorted(BLOCKS, key=lambda b: block_results[b]['rank_score'], reverse=True)
    top_blocks = ranked[:2]

    # Supportive cross-background check only for top two blocks.
    A_star = hybrid(A, B, ('bo',))
    cross_background = {}
    for ti, block in enumerate(top_blocks):
        sys = hybrid(A_star, B, BLOCKS[block])
        ev0 = gate_eval('A_star', A_star, schedules, gt, shared) if ti == 0 else gate['A_star']
        if ti == 0:
            gate['A_star'] = ev0
        ev = gate_eval(f'A_star_plus_{block}', sys, schedules, gt, shared)
        gate[f'A_star_plus_{block}'] = ev
        cross_background[block] = {}
        for mi, m in enumerate(ALL_METRICS):
            cross_background[block][m] = contrast(ev, ev0, m, 1230000 + 1000*ti + 10*mi, args.bootstrap)

    # Secondary readouts for H/B and top-two primary hybrids only.
    xall, yall, _xte, _yte = core.load(args.data)
    x012, y012 = core.subset012(xall, yall)
    _xtr, _ytr, xva, _yva = core.split(x012, y012, .1, 7)
    toknet, _ = base.load_frozen_tokenizer(args.tokenizer, device)
    va, _ = base.encode_snn(toknet, xva, device, args.encode_batch)
    if not np.array_equal(np.asarray(va[validation_ids], dtype=np.int64), gt):
        raise RuntimeError('D6A.2 validation tokens mismatch frozen Gate-A ground truth')
    V = int(F.c.V); N = int(F.c.N)
    _rc, real_p = coarse.probability_from_ids(va, V)
    _pc, real_pos_p = spatial.position_probabilities(va, V, N)
    real_pair = spatial.pair_reference(va, V, N, 7)
    secondary_labels = ['F', 'H', 'B']
    for block in top_blocks:
        secondary_labels += [f'H_plus_{block}', f'B_minus_{block}']
    secondary = {}
    for si, label in enumerate(secondary_labels):
        secondary[label] = secondary_eval(label, systems[label], va, real_p, real_pos_p, real_pair,
                                          args.seed + 70000 + si, args.eval_images, args.gen_samples)

    write_csv(args.out/'D6A2_BLOCK_MAP.csv', rows)
    npz = {'validation_ids': validation_ids, 'gt_tokens': gt.astype(np.int16), 'shared_uniform': shared}
    for label, ev in gate.items():
        for m in ALL_METRICS:
            npz[f'{label}__effect__{m}'] = np.asarray(ev['effects'][m])
    np.savez_compressed(args.out/'D6A2_EFFECT_ARRAYS.npz', **npz)

    summary = {
        'stage': 'GATE_D6A2_TERMINAL_SUPPRESSOR_MAP',
        'classification': classification,
        'source_D6A_run': 39687028646,
        'source_D6A1_run': 39777693838,
        'learner_modified': False,
        'training_performed': False,
        'bo_fixed_to_B_in_causal_hybrids': True,
        'closure': closure,
        'calibration': calibration,
        'total_suppression_load_B_minus_H': loads,
        'block_results': block_results,
        'bidirectional_primary_blocks': bidirectional,
        'necessity_only_primary_blocks': necessity_only,
        'ranked_blocks': ranked,
        'top_two_blocks': top_blocks,
        'A_star_cross_background_top_blocks': cross_background,
        'secondary_selected_systems': secondary,
        'decision': {
            'positive_dual_primary_suppression_load': positive_dual_load,
            'S1_blocks': bidirectional,
            'S2_blocks': necessity_only,
            'pair_search_performed': False,
        },
        'interpretation_boundary': (
            'D6A.2 localizes terminal forward-state suppression of an already demonstrated beneficial bo_B state. '
            'It does not test whether credit-field coupling is harmful and does not modify plasticity. '
            'If S3 occurs, pair/top-k interaction testing requires a separate preregistered extension.'
        ),
    }
    (args.out/'D6A2_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    manifest = {
        'classification': classification,
        'eval_images': args.eval_images,
        'gen_samples': args.gen_samples,
        'bootstrap': args.bootstrap,
        'blocks': list(BLOCKS),
        'protected_baseline_modified': False,
        'no_PR_no_merge': True,
        'training_performed': False,
        'pair_search_performed': False,
    }
    (args.out/'D6A2_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('D6A2_RESULT', json.dumps({
        'classification': classification,
        'loads': {m: loads[m] for m in PRIMARY},
        'S1_blocks': bidirectional,
        'S2_blocks': necessity_only,
        'top_two_blocks': top_blocks,
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
