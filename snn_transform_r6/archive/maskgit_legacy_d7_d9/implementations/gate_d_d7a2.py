#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D7A.2 — final parameter-level top-two compensation stopping gate.

Eval-only test of the single preregistered pair (b2, bO) between the rescued R
and unrescued C terminal systems from D7A.1. No training, learner change,
gradient audit, trajectory replay, or additional block/pair search occurs.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

import gate_d_d3e_runtime as d3r
import gate_d_d6a as d6a
import gate_d_d6a1 as d6a1
import gate_d_d7a1 as d71

PRIMARY = ('F61', 'D_non61')
ALL_METRICS = ('F61', 'D_non61', 'C61', 'delta_P61', 'CE')
TOL = 1e-7
PAIR_FIELDS = ('b2', 'bo')


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


def vec(ev, metric):
    return np.asarray(ev['effects'][metric], dtype=np.float64)


def calibrate(ev, stored, prefix, tol=TOL):
    return d71.calibrate_full(ev, stored, prefix, tol)


def fraction(effect, load):
    L = load.get('mean')
    if L is None or float(L) <= 1e-30:
        return None
    return float(effect['mean']) / float(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d6a-artifact', type=Path, required=True)
    ap.add_argument('--d7a-artifact', type=Path, required=True)
    ap.add_argument('--d7a1-artifact', type=Path, required=True)
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

    # Frozen prerequisite and stopping-map provenance.
    s71 = json.loads(locate(a.d7a1_artifact, 'D7A1_SUMMARY.json').read_text(encoding='utf-8'))
    if s71.get('classification') != 'C3_DISTRIBUTED_COMPENSATION':
        raise RuntimeError('D7A.2 prerequisite failed: D7A.1 is not frozen C3')
    ranked = list(s71.get('ranked_blocks', []))
    if ranked[:2] != ['b2', 'bO']:
        raise RuntimeError({'D7A2_top_two_changed': ranked[:2]})

    stored = np.load(locate(a.d7a1_artifact, 'D7A1_EFFECT_ARRAYS.npz'), allow_pickle=False)
    schedules = json.loads(locate(a.anchor_artifact, 'HISTORICAL_R6B3_R3_T8_REFERENCE_TRACE.json').read_text(encoding='utf-8'))

    # Reconstruct exactly the successful D7A.1 operational systems.
    B = d6a1.load_d6a_terminal(locate(a.d6a_artifact, 'D6A_F_EM_WO_zero_bo_oracle_terminal.pkl'))
    F_same_run = d71.load_terminal(locate(a.d7a_artifact, 'D7A_F_terminal.pkl'))
    C = d71.load_terminal(locate(a.d7a_artifact, 'D7A_C_W2_safe_clamp_terminal.pkl'))
    R = d71.hybrid(B, F_same_run, ('W2',))

    invariants = {
        'W2_R_equals_C': d71.arr_exact(R.W2, C.W2),
        'W2_R_C_max_abs': d71.max_abs(R.W2, C.W2),
        'WO_R_equals_C': d71.arr_exact(R.WO, C.WO),
        'WO_R_C_max_abs': d71.max_abs(R.WO, C.WO),
    }
    if not (invariants['W2_R_equals_C'] and invariants['W2_R_C_max_abs'] == 0.0
            and invariants['WO_R_equals_C'] and invariants['WO_R_C_max_abs'] == 0.0):
        raise RuntimeError({'D7A2_R_C_invariant_failure': invariants})

    gt = np.asarray(stored['gt_tokens'], dtype=np.int64)
    shared = np.asarray(stored['shared_uniform'], dtype=np.float64)
    validation_ids = np.asarray(stored['validation_ids'], dtype=np.int64)
    if len(gt) != 256 or len(schedules) != 256:
        raise RuntimeError('D7A.2 frozen Gate-A coordinate system is not 256 images')

    # Reconstruct all D7A.1 frozen base/single systems used in the interaction.
    systems = {
        'R': R,
        'C': C,
        'R_plus_C_b2': d71.hybrid(R, C, ('b2',)),
        'R_plus_C_bO': d71.hybrid(R, C, ('bo',)),
        'C_minus_b2': d71.hybrid(C, R, ('b2',)),
        'C_minus_bO': d71.hybrid(C, R, ('bo',)),
    }
    # The only two new scientific systems.
    systems['R_plus_C_b2_bO'] = d71.hybrid(R, C, PAIR_FIELDS)
    systems['C_minus_b2_bO'] = d71.hybrid(C, R, PAIR_FIELDS)

    gate = {}
    for label, system in systems.items():
        gate[label] = d71.gate_eval(label, system, schedules, gt, shared)
        print('D7A2_GATE', json.dumps({'system': label}), flush=True)

    calibration = {}
    for label in ('R', 'C', 'R_plus_C_b2', 'R_plus_C_bO', 'C_minus_b2', 'C_minus_bO'):
        calibration[label] = calibrate(gate[label], stored, label)
    if not all(x['pass'] for x in calibration.values()):
        raise RuntimeError({'D7A2_D7A1_frozen_calibration_failure': calibration})

    loads = {}
    suff_pair = {}
    nec_pair = {}
    interaction_s = {}
    interaction_n = {}
    pair_minus_b2_s = {}

    for mi, m in enumerate(ALL_METRICS):
        r = vec(gate['R'], m)
        c = vec(gate['C'], m)
        rb2 = vec(gate['R_plus_C_b2'], m)
        rbo = vec(gate['R_plus_C_bO'], m)
        rp = vec(gate['R_plus_C_b2_bO'], m)
        cb2 = vec(gate['C_minus_b2'], m)
        cbo = vec(gate['C_minus_bO'], m)
        cp = vec(gate['C_minus_b2_bO'], m)

        load_v = c - r
        s12_v = rp - r
        n12_v = c - cp
        nb2_v = c - cb2
        nbo_v = c - cbo
        is_v = rp - rb2 - rbo + r
        in_v = n12_v - nb2_v - nbo_v
        pmb2_v = rp - rb2

        loads[m] = bootstrap_vector(load_v, 1500000 + 10*mi, a.bootstrap)
        suff_pair[m] = bootstrap_vector(s12_v, 1510000 + 10*mi, a.bootstrap)
        nec_pair[m] = bootstrap_vector(n12_v, 1520000 + 10*mi, a.bootstrap)
        interaction_s[m] = bootstrap_vector(is_v, 1530000 + 10*mi, a.bootstrap)
        interaction_n[m] = bootstrap_vector(in_v, 1540000 + 10*mi, a.bootstrap)
        pair_minus_b2_s[m] = bootstrap_vector(pmb2_v, 1550000 + 10*mi, a.bootstrap)
        suff_pair[m]['fraction_of_total_load'] = fraction(suff_pair[m], loads[m])
        nec_pair[m]['fraction_of_total_load'] = fraction(nec_pair[m], loads[m])

    pair_suff_primary = bool(all(suff_pair[m]['ci_entirely_positive'] for m in PRIMARY))
    pair_nec_primary = bool(all(nec_pair[m]['ci_entirely_positive'] for m in PRIMARY))
    is_pos = bool(all(interaction_s[m]['ci_entirely_positive'] for m in PRIMARY))
    in_pos = bool(all(interaction_n[m]['ci_entirely_positive'] for m in PRIMARY))
    is_neg = bool(all(interaction_s[m]['ci_entirely_negative'] for m in PRIMARY))
    pair_close_b2 = bool(all(pair_minus_b2_s[m]['ci_contains_zero'] for m in PRIMARY))
    majority = bool(all(
        suff_pair[m]['fraction_of_total_load'] is not None
        and nec_pair[m]['fraction_of_total_load'] is not None
        and suff_pair[m]['fraction_of_total_load'] >= 0.50
        and nec_pair[m]['fraction_of_total_load'] >= 0.50
        for m in PRIMARY
    ))

    if pair_suff_primary and pair_nec_primary and is_pos and in_pos:
        classification = 'A3_SYNERGISTIC_PAIR'
    elif pair_suff_primary and pair_nec_primary and is_neg and pair_close_b2:
        classification = 'A2_REDUNDANT_OFFSET_MODE'
    elif pair_suff_primary and pair_nec_primary and majority:
        classification = 'A1_COMPACT_TWO_BLOCK_COMPENSATION'
    else:
        classification = 'A4_DISTRIBUTED_COMPENSATION_PERSISTS'

    # Persist compact audit trail.
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
    np.savez_compressed(a.out/'D7A2_EFFECT_ARRAYS.npz', **npz)

    summary = {
        'stage': 'GATE_D7A2_TOP_TWO_COMPENSATION_STOPPING_GATE',
        'classification': classification,
        'source_runs': {'D7A1': 39929247723, 'D7A': 39871827217, 'D6A': 39687028646},
        'training_performed': False,
        'learner_modified': False,
        'gradient_audit_performed': False,
        'trajectory_replay_performed': False,
        'additional_pair_search_performed': False,
        'pair': ['b2', 'bO'],
        'invariants': invariants,
        'calibration_to_D7A1': calibration,
        'total_compensation_load': loads,
        'pair_sufficiency': suff_pair,
        'pair_necessity': nec_pair,
        'interaction_sufficiency': interaction_s,
        'interaction_necessity': interaction_n,
        'pair_minus_b2_sufficiency': pair_minus_b2_s,
        'decision': {
            'pair_sufficiency_primary_positive': pair_suff_primary,
            'pair_necessity_primary_positive': pair_nec_primary,
            'interaction_sufficiency_primary_positive': is_pos,
            'interaction_necessity_primary_positive': in_pos,
            'interaction_sufficiency_primary_negative': is_neg,
            'pair_minus_b2_sufficiency_CI_contains_zero_both_primary': pair_close_b2,
            'majority_load_all_four_point_estimates_ge_0p50': majority,
            'parameter_decomposition_must_stop': bool(classification == 'A4_DISTRIBUTED_COMPENSATION_PERSISTS'),
            'proceed_to_D8_functional_mode_identification': bool(classification == 'A4_DISTRIBUTED_COMPENSATION_PERSISTS'),
        },
        'interpretation_boundary': (
            'D7A.2 is the final preregistered parameter-level stopping gate. It tests only the b2+bO pair. '
            'A4 mandates stopping block/pair transplantation and moving to parameterization-invariant functional-mode identification. '
            'No result here establishes trajectory emergence or literal migration of a harmful mode during learning.'
        ),
    }
    (a.out/'D7A2_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (a.out/'D7A2_MANIFEST.json').write_text(json.dumps({
        'classification': classification,
        'eval_images': 256,
        'bootstrap': a.bootstrap,
        'pair': ['b2', 'bO'],
        'protected_baseline_modified': False,
        'no_PR_no_merge': True,
        'training_performed': False,
        'additional_pair_search_performed': False,
    }, indent=2), encoding='utf-8')

    print('D7A2_RESULT', json.dumps({
        'classification': classification,
        'loads': {m: loads[m] for m in PRIMARY},
        'pair_sufficiency': {m: suff_pair[m] for m in PRIMARY},
        'pair_necessity': {m: nec_pair[m] for m in PRIMARY},
        'I_S': {m: interaction_s[m] for m in PRIMARY},
        'I_N': {m: interaction_n[m] for m in PRIMARY},
        'pair_minus_b2_S': {m: pair_minus_b2_s[m] for m in PRIMARY},
        'stop_parameter_decomposition': summary['decision']['parameter_decomposition_must_stop'],
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
