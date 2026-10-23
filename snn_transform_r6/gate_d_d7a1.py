#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D7A.1 — compensation mapping around the safe-W2 constraint.

Eval-only terminal causal map between:
  R = D6A combined terminal with W2 replaced by W2^F (rescued terminal hybrid)
  C = D7A time-matched safe-W2-clamped terminal (unrescued learner)

W2 is exact-identical in R and C. WO must also be exact-identical and is not a
candidate. Nine remaining forward blocks are mapped bidirectionally for
compensation sufficiency and necessity. No training, gradient audit, trajectory
replay, pair search, or learner modification occurs.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch

import gate_d_d3e_runtime as d3r
import gate_d_d6a as d6a
import gate_d_d6a1 as d6a1
from r6b4_full_forward_plasticity import FORWARD_PARAMETER_KEYS

TOL = 1e-7
PRIMARY = ('F61', 'D_non61')
ALL_METRICS = ('F61', 'D_non61', 'C61', 'delta_P61', 'CE')

BLOCKS = {
    'Q': ('WQ', 'bq'),
    'K': ('WK', 'bk'),
    'V': ('WV', 'bv'),
    'bO': ('bo',),
    'Norm': ('gamma2', 'beta2'),
    'FF1': ('W1', 'b1'),
    'b2': ('b2',),
    'Readout': ('WR',),
    'Embedding': ('tok', 'pos'),
}

F_LABEL = 'F_EM'
B_LABEL = 'F_EM_WO_zero_bo_oracle'


def clone(x):
    return pickle.loads(pickle.dumps(x, pickle.HIGHEST_PROTOCOL))


def locate(root: Path, name: str) -> Path:
    return d3r.locate(root, name)


def load_terminal(path: Path):
    with path.open('rb') as f:
        obj = pickle.load(f)
    return obj['system'] if isinstance(obj, dict) and 'system' in obj else obj


def write_csv(path: Path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def set_fields(dst, donor, fields):
    for k in fields:
        x = getattr(donor, k)
        setattr(dst, k, np.array(x, copy=True) if isinstance(x, np.ndarray) else x)


def hybrid(background, donor, fields):
    s = clone(background)
    set_fields(s, donor, fields)
    return s


def arr_exact(a, b):
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        return bool(a == b)
    return bool(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b))


def max_abs(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float('inf')
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return float('inf')
    if not np.array_equal(np.isposinf(a), np.isposinf(b)):
        return float('inf')
    if not np.array_equal(np.isneginf(a), np.isneginf(b)):
        return float('inf')
    good = np.isfinite(a) & np.isfinite(b)
    return 0.0 if not np.any(good) else float(np.max(np.abs(a[good] - b[good])))


def forward_exact(a, b):
    fields = {}
    ok = True
    for k in FORWARD_PARAMETER_KEYS:
        eq = arr_exact(getattr(a, k), getattr(b, k))
        fields[k] = eq
        ok = ok and eq
    return {'pass': bool(ok), 'fields': fields}


def gate_eval(name, system, schedules, gt, shared):
    return d6a.gate_eval(name, system, schedules, gt, shared)


def bootstrap_contrast(ev1, ev0, metric, seed, nboot):
    x = (np.asarray(ev1['effects'][metric], dtype=np.float64)
         - np.asarray(ev0['effects'][metric], dtype=np.float64))
    d = d6a.bootstrap_mean(x, seed, nboot)
    lo, hi = d['ci95']
    d['ci_entirely_positive'] = bool(lo is not None and lo > 0)
    d['ci_entirely_negative'] = bool(hi is not None and hi < 0)
    return d


def calibrate_effects_only(ev, stored, prefix, tol=TOL):
    out = {'historical_fields_available': 'effects_only', 'effects': {}}
    for m in ALL_METRICS:
        out['effects'][m] = d6a.nanaware_max_abs(
            ev['effects'][m], stored[f'{prefix}__effect__{m}'])
    out['pass'] = bool(max(out['effects'].values()) < tol)
    return out


def calibrate_full(ev, stored, prefix, tol=TOL):
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


def functional_match(ev, ref, tol=TOL):
    out = {
        'self_metrics_max_abs': d6a.nanaware_max_abs(ev['self_metrics'], ref['self_metrics']),
        'teacher_metrics_max_abs': d6a.nanaware_max_abs(ev['teacher_metrics'], ref['teacher_metrics']),
        'self_proposal_exact': bool(np.array_equal(ev['self_proposal'], ref['self_proposal'])),
        'teacher_proposal_exact': bool(np.array_equal(ev['teacher_proposal'], ref['teacher_proposal'])),
        'effects': {},
    }
    for m in ALL_METRICS:
        out['effects'][m] = d6a.nanaware_max_abs(ev['effects'][m], ref['effects'][m])
    out['pass'] = bool(
        out['self_metrics_max_abs'] < tol
        and out['teacher_metrics_max_abs'] < tol
        and out['self_proposal_exact']
        and out['teacher_proposal_exact']
        and max(out['effects'].values()) < tol
    )
    return out


def rank_score(br, loads):
    vals = []
    for m in PRIMARY:
        L = loads[m]['mean']
        if L is not None and float(L) > 1e-30:
            vals.append(float(br['sufficiency'][m]['mean']) / float(L))
            vals.append(float(br['necessity'][m]['mean']) / float(L))
    return float(np.mean(vals)) if vals else float('-inf')


def save_summary(path, summary):
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d6a-artifact', type=Path, required=True)
    ap.add_argument('--d6a2-artifact', type=Path, required=True)
    ap.add_argument('--d6a3-artifact', type=Path, required=True)
    ap.add_argument('--d7a-artifact', type=Path, required=True)
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

    # Frozen prerequisite chain.
    s6 = json.loads(locate(a.d6a_artifact, 'D6A_SUMMARY.json').read_text(encoding='utf-8'))
    s62 = json.loads(locate(a.d6a2_artifact, 'D6A2_SUMMARY.json').read_text(encoding='utf-8'))
    s63 = json.loads(locate(a.d6a3_artifact, 'D6A3_SUMMARY.json').read_text(encoding='utf-8'))
    s7 = json.loads(locate(a.d7a_artifact, 'D7A_SUMMARY.json').read_text(encoding='utf-8'))
    if s6.get('classification') != 'R0_NO_REPAIR_SUFFICIENCY' or not s6.get('decision', {}).get('oracle_valid', False):
        raise RuntimeError('D7A.1 prerequisite failed: D6A is not frozen R0 with valid oracle')
    if s62.get('classification') != 'S1_DOMINANT_SINGLE_SUPPRESSOR':
        raise RuntimeError('D7A.1 prerequisite failed: D6A.2 is not frozen S1')
    if s63.get('classification') != 'P4_MIXED_BACKGROUND_DEPENDENT':
        raise RuntimeError('D7A.1 prerequisite failed: D6A.3 is not frozen P4')
    if s7.get('classification') != 'T0_NO_W2_TRAJECTORY_RESCUE' or not s7.get('oracle_valid', False):
        raise RuntimeError('D7A.1 prerequisite failed: D7A is not frozen T0 with valid oracle')

    stored62 = np.load(locate(a.d6a2_artifact, 'D6A2_EFFECT_ARRAYS.npz'), allow_pickle=False)
    stored7 = np.load(locate(a.d7a_artifact, 'D7A_EFFECT_ARRAYS.npz'), allow_pickle=False)
    schedules = json.loads(locate(a.anchor_artifact, 'HISTORICAL_R6B3_R3_T8_REFERENCE_TRACE.json').read_text(encoding='utf-8'))

    F = d6a1.load_d6a_terminal(locate(a.d6a_artifact, 'D6A_F_EM_terminal.pkl'))
    B = d6a1.load_d6a_terminal(locate(a.d6a_artifact, 'D6A_F_EM_WO_zero_bo_oracle_terminal.pkl'))
    C = load_terminal(locate(a.d7a_artifact, 'D7A_C_W2_safe_clamp_terminal.pkl'))
    R = hybrid(B, F, ('W2',))

    expected = {'tok','pos','WQ','WK','WV','WO','W1','W2','WR','bq','bk','bv','bo','b1','b2','gamma2','beta2'}
    if set(FORWARD_PARAMETER_KEYS) != expected:
        raise RuntimeError({'D7A1_forward_parameter_key_set_changed': sorted(FORWARD_PARAMETER_KEYS)})

    # Exact W2/WO invariants.
    invariants = {
        'W2_R_equals_C': arr_exact(R.W2, C.W2),
        'W2_R_equals_F': arr_exact(R.W2, F.W2),
        'W2_C_equals_F': arr_exact(C.W2, F.W2),
        'W2_R_C_max_abs': max_abs(R.W2, C.W2),
        'WO_R_equals_C': arr_exact(R.WO, C.WO),
        'WO_R_C_max_abs': max_abs(R.WO, C.WO),
    }
    if not (invariants['W2_R_equals_C'] and invariants['W2_R_equals_F'] and invariants['W2_C_equals_F']
            and invariants['W2_R_C_max_abs'] == 0.0 and invariants['WO_R_equals_C']
            and invariants['WO_R_C_max_abs'] == 0.0):
        raise RuntimeError({'D7A1_W2_WO_invariant_failure': invariants})

    # Partition must cover all forward fields except exact-invariant W2 and WO.
    partition_fields = [k for fields in BLOCKS.values() for k in fields]
    expected_candidates = set(FORWARD_PARAMETER_KEYS) - {'W2', 'WO'}
    partition = {
        'expected': sorted(expected_candidates),
        'covered': sorted(set(partition_fields)),
        'duplicates': sorted({k for k in partition_fields if partition_fields.count(k) > 1}),
    }
    partition['pass'] = bool(
        len(partition_fields) == len(set(partition_fields))
        and set(partition_fields) == expected_candidates)
    if not partition['pass']:
        raise RuntimeError({'D7A1_partition_failure': partition})

    # Exact forward closure with W2 and WO held fixed-identical.
    candidate_fields = tuple(partition_fields)
    R_all_C = hybrid(R, C, candidate_fields)
    C_all_R = hybrid(C, R, candidate_fields)
    forward_closure = {
        'R_all_C_equals_C': forward_exact(R_all_C, C),
        'C_all_R_equals_R': forward_exact(C_all_R, R),
        'partition': partition,
        'invariants': invariants,
    }
    forward_closure['pass'] = bool(
        forward_closure['R_all_C_equals_C']['pass']
        and forward_closure['C_all_R_equals_R']['pass'])
    if not forward_closure['pass']:
        raise RuntimeError({'D7A1_forward_closure_failure': forward_closure})

    gt = np.asarray(stored7['gt_tokens'], dtype=np.int64)
    shared = np.asarray(stored7['shared_uniform'], dtype=np.float64)
    validation_ids = np.asarray(stored7['validation_ids'], dtype=np.int64)
    if len(gt) != 256 or len(schedules) != 256:
        raise RuntimeError('D7A.1 frozen Gate-A coordinate system is not 256 images')

    # Evaluate base systems and closure hybrids first.
    gate = {
        'R': gate_eval('D7A1_R_rescued', R, schedules, gt, shared),
        'C': gate_eval('D7A1_C_unrescued', C, schedules, gt, shared),
        'R_all_C': gate_eval('D7A1_R_all_C', R_all_C, schedules, gt, shared),
        'C_all_R': gate_eval('D7A1_C_all_R', C_all_R, schedules, gt, shared),
    }

    calibration = {
        'R_to_D6A2_B_minus_W2': calibrate_effects_only(gate['R'], stored62, 'B_minus_W2'),
        'C_to_D7A': calibrate_full(gate['C'], stored7, 'C'),
    }
    if not all(v['pass'] for v in calibration.values()):
        raise RuntimeError({'D7A1_frozen_calibration_failure': calibration})

    functional_closure = {
        'R_all_C_to_C': functional_match(gate['R_all_C'], gate['C']),
        'C_all_R_to_R': functional_match(gate['C_all_R'], gate['R']),
    }
    functional_closure['pass'] = bool(all(v['pass'] for v in functional_closure.values()))
    if not functional_closure['pass']:
        summary = {
            'stage': 'GATE_D7A1_SAFE_W2_COMPENSATION_MAP',
            'classification': 'C4_FUNCTIONAL_STATE_CLOSURE_FAILURE',
            'training_performed': False,
            'learner_modified': False,
            'forward_closure': forward_closure,
            'functional_closure': functional_closure,
            'calibration': calibration,
            'interpretation_allowed': False,
        }
        save_summary(a.out/'D7A1_SUMMARY.json', summary)
        (a.out/'D7A1_MANIFEST.json').write_text(json.dumps({
            'classification': summary['classification'],
            'protected_baseline_modified': False,
            'no_PR_no_merge': True,
            'training_performed': False,
            'pair_search_performed': False,
        }, indent=2), encoding='utf-8')
        print('D7A1_RESULT', json.dumps({'classification': summary['classification'],
                                         'functional_closure': functional_closure}, indent=2), flush=True)
        return

    # Build/evaluate single-block bidirectional hybrids only after closure is certified.
    systems = {'R': R, 'C': C}
    for block, fields in BLOCKS.items():
        systems[f'R_plus_C_{block}'] = hybrid(R, C, fields)
        systems[f'C_minus_{block}'] = hybrid(C, R, fields)
    for label, system in systems.items():
        if label not in gate:
            gate[label] = gate_eval(label, system, schedules, gt, shared)
            print('D7A1_GATE', json.dumps({'system': label}), flush=True)

    # Total compensation load C-R.
    loads = {}
    for mi, m in enumerate(ALL_METRICS):
        loads[m] = bootstrap_contrast(gate['C'], gate['R'], m, 1400000 + 10*mi, a.bootstrap)

    # Single-block causal map.
    block_results = {}
    rows = []
    for bi, (block, fields) in enumerate(BLOCKS.items()):
        br = {'fields': list(fields), 'sufficiency': {}, 'necessity': {}}
        for mi, m in enumerate(ALL_METRICS):
            suff = bootstrap_contrast(gate[f'R_plus_C_{block}'], gate['R'], m,
                                      1410000 + 1000*bi + 10*mi, a.bootstrap)
            nec = bootstrap_contrast(gate['C'], gate[f'C_minus_{block}'], m,
                                     1420000 + 1000*bi + 10*mi, a.bootstrap)
            L = loads[m]['mean']
            if L is not None and float(L) > 1e-30:
                suff['fraction_of_total_load'] = float(suff['mean']) / float(L)
                nec['fraction_of_total_load'] = float(nec['mean']) / float(L)
            else:
                suff['fraction_of_total_load'] = None
                nec['fraction_of_total_load'] = None
            br['sufficiency'][m] = suff
            br['necessity'][m] = nec

        suff_both = bool(all(br['sufficiency'][m]['ci_entirely_positive'] for m in PRIMARY))
        nec_both = bool(all(br['necessity'][m]['ci_entirely_positive'] for m in PRIMARY))
        br['bidirectional_primary'] = bool(suff_both and nec_both)
        br['necessity_only_primary'] = bool(nec_both and not suff_both)
        br['rank_score'] = rank_score(br, loads)
        block_results[block] = br

        row = {
            'block': block,
            'fields': '+'.join(fields),
            'bidirectional_primary': br['bidirectional_primary'],
            'necessity_only_primary': br['necessity_only_primary'],
            'rank_score': br['rank_score'],
        }
        for m in PRIMARY:
            row[f'S_{m}'] = br['sufficiency'][m]['mean']
            row[f'S_{m}_lo'] = br['sufficiency'][m]['ci95'][0]
            row[f'S_{m}_hi'] = br['sufficiency'][m]['ci95'][1]
            row[f'S_{m}_frac'] = br['sufficiency'][m]['fraction_of_total_load']
            row[f'N_{m}'] = br['necessity'][m]['mean']
            row[f'N_{m}_lo'] = br['necessity'][m]['ci95'][0]
            row[f'N_{m}_hi'] = br['necessity'][m]['ci95'][1]
            row[f'N_{m}_frac'] = br['necessity'][m]['fraction_of_total_load']
        rows.append(row)

    c1_blocks = [b for b, r in block_results.items() if r['bidirectional_primary']]
    c2_blocks = [b for b, r in block_results.items() if r['necessity_only_primary']]
    dual_load_significant = bool(all(loads[m]['ci_entirely_positive'] for m in PRIMARY))

    if c1_blocks:
        classification = 'C1_SINGLE_COMPENSATORY_CARRIER'
    elif c2_blocks:
        classification = 'C2_CONTEXT_DEPENDENT_COMPENSATION'
    elif dual_load_significant:
        classification = 'C3_DISTRIBUTED_COMPENSATION'
    else:
        classification = 'CX_INSUFFICIENT_DUAL_COMPENSATION_LOAD'

    ranked = sorted(BLOCKS, key=lambda b: block_results[b]['rank_score'], reverse=True)

    write_csv(a.out/'D7A1_COMPENSATION_BLOCK_MAP.csv', rows)
    npz = {
        'validation_ids': validation_ids,
        'gt_tokens': gt.astype(np.int16),
        'shared_uniform': shared,
    }
    for label, ev in gate.items():
        npz[f'{label}__self_metrics'] = ev['self_metrics']
        npz[f'{label}__teacher_metrics'] = ev['teacher_metrics']
        npz[f'{label}__self_proposal'] = ev['self_proposal']
        npz[f'{label}__teacher_proposal'] = ev['teacher_proposal']
        for m in ALL_METRICS:
            npz[f'{label}__effect__{m}'] = np.asarray(ev['effects'][m])
    np.savez_compressed(a.out/'D7A1_EFFECT_ARRAYS.npz', **npz)

    summary = {
        'stage': 'GATE_D7A1_SAFE_W2_COMPENSATION_MAP',
        'classification': classification,
        'source_runs': {
            'D6A': 39687028646,
            'D6A2': 39806081064,
            'D6A3': 39819736038,
            'D7A': 39871827217,
        },
        'training_performed': False,
        'learner_modified': False,
        'gradient_audit_performed': False,
        'trajectory_replay_performed': False,
        'pair_search_performed': False,
        'R_definition': 'D6A B terminal with W2 replaced by D6A F_EM W2',
        'C_definition': 'D7A B_W2_safe_clamp terminal',
        'forward_closure': forward_closure,
        'functional_closure': functional_closure,
        'calibration': calibration,
        'total_compensation_load_C_minus_R': loads,
        'block_results': block_results,
        'C1_bidirectional_blocks': c1_blocks,
        'C2_necessity_only_blocks': c2_blocks,
        'ranked_blocks': ranked,
        'decision': {
            'dual_primary_compensation_load_significant': dual_load_significant,
            'C1_blocks': c1_blocks,
            'C2_blocks': c2_blocks,
            'interpretation_allowed': True,
        },
        'interpretation_boundary': (
            'D7A.1 localizes terminal non-W2 forward-state carriers of the R-versus-C deployment gap under exact W2/WO identity. '
            'It does not establish when compensation emerged or prove literal migration of a functional mode during learning. '
            'Trajectory analysis is deferred to a later gate for the top carrier only.'
        ),
    }
    save_summary(a.out/'D7A1_SUMMARY.json', summary)
    manifest = {
        'classification': classification,
        'eval_images': 256,
        'bootstrap': a.bootstrap,
        'blocks': list(BLOCKS),
        'protected_baseline_modified': False,
        'no_PR_no_merge': True,
        'training_performed': False,
        'pair_search_performed': False,
    }
    (a.out/'D7A1_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print('D7A1_RESULT', json.dumps({
        'classification': classification,
        'loads': {m: loads[m] for m in PRIMARY},
        'C1_blocks': c1_blocks,
        'C2_blocks': c2_blocks,
        'ranked_blocks': ranked,
        'top_block': ranked[0] if ranked else None,
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
