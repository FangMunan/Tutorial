#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D6A.3 — W2–b2 suppressor interaction test.

Eval-only terminal causal interaction audit. No training or learner modification.
The only new scientific pair evaluations are H_W2_b2, B_minus_W2_b2, and
A_star_W2_b2. Frozen D6A.2 per-image arrays provide the single-block effects.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch

import gate_d_d3e_runtime as d3r
import gate_d_d6a as d6a
import gate_d_d6a1 as d6a1
import gate_d_d6a2 as d6a2
import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
import r6b3_decoder_operating_point_audit as coarse
import r6b3_decoder_spatial_refinement_audit as spatial
from gate_d_d3e_eval import BASE_GEN_SEED, eval_masked, rollout_js
from r6b4_full_forward_plasticity import FORWARD_PARAMETER_KEYS

PRIMARY = ('F61', 'D_non61')
ALL_METRICS = ('F61', 'D_non61', 'C61', 'delta_P61', 'CE')
PAIR_FIELDS = ('W2', 'b2')
TOL = 1e-7

F_LABEL = 'F_EM'
A_LABEL = 'F_EM_bo_oracle'
B_LABEL = 'F_EM_WO_zero_bo_oracle'


def clone(x):
    return pickle.loads(pickle.dumps(x, pickle.HIGHEST_PROTOCOL))


def locate(root: Path, name: str) -> Path:
    return d3r.locate(root, name)


def hybrid(background, donor, fields):
    s = clone(background)
    for k in fields:
        setattr(s, k, np.array(getattr(donor, k), copy=True))
    return s


def arr_exact(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return bool(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b))


def gate_eval(name, system, schedules, gt, shared):
    return d6a.gate_eval(name, system, schedules, gt, shared)


def boot(vec, seed, nboot):
    d = d6a.bootstrap_mean(np.asarray(vec, dtype=np.float64), seed, nboot)
    lo, hi = d['ci95']
    d['ci_entirely_positive'] = bool(lo is not None and lo > 0)
    d['ci_entirely_negative'] = bool(hi is not None and hi < 0)
    d['ci_includes_zero'] = bool(lo is not None and hi is not None and lo <= 0 <= hi)
    return d


def effect(stored, label, metric):
    return np.asarray(stored[f'{label}__effect__{metric}'], dtype=np.float64)


def calibrate_effects(ev, stored, label, tol=TOL):
    out = {'effects': {}}
    for m in ALL_METRICS:
        out['effects'][m] = d6a.nanaware_max_abs(ev['effects'][m], stored[f'{label}__effect__{m}'])
    out['pass'] = bool(max(out['effects'].values()) < tol)
    return out


def purity_pair(system, background, pair_donor, bo_donor, mode):
    """Exact forward-field source audit for the three new pair hybrids."""
    rows = {}; ok = True
    for k in FORWARD_PARAMETER_KEYS:
        if k == 'bo':
            donor = bo_donor; source = 'bo_B'
        elif k in PAIR_FIELDS:
            donor = pair_donor; source = 'pair_donor'
        else:
            donor = background; source = 'background'
        eq = arr_exact(getattr(system, k), getattr(donor, k))
        rows[k] = {'expected_source': source, 'exact': bool(eq)}
        ok = ok and eq
    return {'pass': bool(ok), 'mode': mode, 'fields': rows}


def secondary_eval(system, va, real_p, real_pos_p, real_pair, seed, eval_images, gen_samples):
    masked = eval_masked(system, va, seed, eval_images)
    rollout = rollout_js(system, real_p, real_pos_p, real_pair, BASE_GEN_SEED, gen_samples)
    if not rollout.get('all_trace_audits_pass', False):
        raise RuntimeError('D6A.3 rollout trace audit failed')
    return {'masked': masked, 'rollout': rollout}


def sign_class(d):
    if d['ci_entirely_negative']:
        return 'negative'
    if d['ci_entirely_positive']:
        return 'positive'
    return 'includes_zero'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--d6a-artifact', type=Path, required=True)
    ap.add_argument('--d6a1-artifact', type=Path, required=True)
    ap.add_argument('--d6a2-artifact', type=Path, required=True)
    ap.add_argument('--anchor-artifact', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--eval-images', type=int, default=256)
    ap.add_argument('--gen-samples', type=int, default=256)
    ap.add_argument('--encode-batch', type=int, default=256)
    ap.add_argument('--bootstrap', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=7311)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    if args.eval_images != 256:
        raise ValueError('formal D6A.3 requires exact 256-image Gate-A evaluation')

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads); device = torch.device('cpu')

    # Frozen prerequisite chain.
    s6 = json.loads(locate(args.d6a_artifact, 'D6A_SUMMARY.json').read_text(encoding='utf-8'))
    s61 = json.loads(locate(args.d6a1_artifact, 'D6A1_SUMMARY.json').read_text(encoding='utf-8'))
    s62 = json.loads(locate(args.d6a2_artifact, 'D6A2_SUMMARY.json').read_text(encoding='utf-8'))
    if s6.get('classification') != 'R0_NO_REPAIR_SUFFICIENCY':
        raise RuntimeError('D6A prerequisite is not frozen R0')
    if s61.get('classification') != 'I3_NO_INTRINSIC_BO_REPAIR_DEGRADATION':
        raise RuntimeError('D6A.1 prerequisite is not frozen I3')
    if s62.get('classification') != 'S1_DOMINANT_SINGLE_SUPPRESSOR':
        raise RuntimeError('D6A.2 prerequisite is not frozen S1')
    if not {'W2','b2'}.issubset(set(s62.get('bidirectional_primary_blocks', []))):
        raise RuntimeError('D6A.2 did not certify both W2 and b2 as bidirectional primary suppressors')
    if s62.get('decision', {}).get('pair_search_performed', True):
        raise RuntimeError('D6A.2 unexpectedly reports prior pair search')
    if not s62.get('closure', {}).get('pass', False):
        raise RuntimeError('D6A.2 forward closure prerequisite failed')

    stored62 = np.load(locate(args.d6a2_artifact, 'D6A2_EFFECT_ARRAYS.npz'), allow_pickle=False)
    schedules = json.loads(locate(args.anchor_artifact, 'HISTORICAL_R6B3_R3_T8_REFERENCE_TRACE.json').read_text(encoding='utf-8'))
    gt = np.asarray(stored62['gt_tokens'], dtype=np.int64)
    shared = np.asarray(stored62['shared_uniform'], dtype=np.float64)
    validation_ids = np.asarray(stored62['validation_ids'], dtype=np.int64)
    if len(gt) != 256 or len(schedules) != 256:
        raise RuntimeError('D6A.3 frozen Gate-A coordinate system is not 256 images')

    # Reconstruct frozen terminal states from D6A.
    F = d6a1.load_d6a_terminal(locate(args.d6a_artifact, 'D6A_F_EM_terminal.pkl'))
    A = d6a1.load_d6a_terminal(locate(args.d6a_artifact, 'D6A_F_EM_bo_oracle_terminal.pkl'))
    B = d6a1.load_d6a_terminal(locate(args.d6a_artifact, 'D6A_F_EM_WO_zero_bo_oracle_terminal.pkl'))
    H = hybrid(F, B, ('bo',))
    A_star = hybrid(A, B, ('bo',))

    # Reconstruct all frozen single-block states used in the algebra.
    H_W2 = hybrid(H, B, ('W2',))
    H_b2 = hybrid(H, B, ('b2',))
    B_mW2 = hybrid(B, F, ('W2',))
    B_mb2 = hybrid(B, F, ('b2',))
    Astar_W2 = hybrid(A_star, B, ('W2',))
    Astar_b2 = hybrid(A_star, B, ('b2',))

    # The only new pair states.
    H_pair = hybrid(H, B, PAIR_FIELDS)
    B_mpair = hybrid(B, F, PAIR_FIELDS)
    Astar_pair = hybrid(A_star, B, PAIR_FIELDS)

    purity = {
        'H_pair': purity_pair(H_pair, H, B, B, 'sufficiency_pair'),
        'B_minus_pair': purity_pair(B_mpair, B, F, B, 'necessity_pair'),
        'Astar_pair': purity_pair(Astar_pair, A_star, B, B, 'cross_background_pair'),
    }
    if not all(v['pass'] for v in purity.values()):
        raise RuntimeError({'D6A3_pair_purity_failed': purity})

    # A0 reconstruction calibration to frozen D6A.2 arrays.
    recon = {
        'H': H, 'H_plus_W2': H_W2, 'H_plus_b2': H_b2,
        'B': B, 'B_minus_W2': B_mW2, 'B_minus_b2': B_mb2,
        'A_star': A_star, 'A_star_plus_W2': Astar_W2, 'A_star_plus_b2': Astar_b2,
    }
    calibration = {}
    for label, system in recon.items():
        ev = gate_eval(f'cal_{label}', system, schedules, gt, shared)
        calibration[label] = calibrate_effects(ev, stored62, label)
        print('D6A3_CAL', json.dumps({'label': label, 'pass': calibration[label]['pass']}), flush=True)
    if not all(v['pass'] for v in calibration.values()):
        raise RuntimeError({'D6A3_D6A2_reconstruction_failed': calibration})

    # New pair evaluations only.
    pair_eval = {
        'H_plus_W2_b2': gate_eval('H_plus_W2_b2', H_pair, schedules, gt, shared),
        'B_minus_W2_b2': gate_eval('B_minus_W2_b2', B_mpair, schedules, gt, shared),
        'A_star_plus_W2_b2': gate_eval('A_star_plus_W2_b2', Astar_pair, schedules, gt, shared),
    }

    interactions = {'sufficiency': {}, 'necessity': {}, 'A_star_supportive': {}}
    descriptive = {}
    raw = {
        'validation_ids': validation_ids,
        'gt_tokens': gt.astype(np.int16),
        'shared_uniform': shared,
    }

    for mi, m in enumerate(ALL_METRICS):
        h = effect(stored62, 'H', m)
        hw = effect(stored62, 'H_plus_W2', m)
        hb = effect(stored62, 'H_plus_b2', m)
        b = effect(stored62, 'B', m)
        bmw = effect(stored62, 'B_minus_W2', m)
        bmb = effect(stored62, 'B_minus_b2', m)
        ast = effect(stored62, 'A_star', m)
        astw = effect(stored62, 'A_star_plus_W2', m)
        astb = effect(stored62, 'A_star_plus_b2', m)
        hp = np.asarray(pair_eval['H_plus_W2_b2']['effects'][m], dtype=np.float64)
        bmp = np.asarray(pair_eval['B_minus_W2_b2']['effects'][m], dtype=np.float64)
        astp = np.asarray(pair_eval['A_star_plus_W2_b2']['effects'][m], dtype=np.float64)

        S_w = hw - h; S_b = hb - h; S_12 = hp - h
        N_w = b - bmw; N_b = b - bmb; N_12 = b - bmp
        IS = hp - hw - hb + h
        IN = bmw + bmb - bmp - b
        IA = astp - astw - astb + ast

        interactions['sufficiency'][m] = boot(IS, 1300000 + 10*mi, args.bootstrap)
        interactions['necessity'][m] = boot(IN, 1310000 + 10*mi, args.bootstrap)
        interactions['A_star_supportive'][m] = {
            'pair_minus_Astar': boot(astp - ast, 1320000 + 20*mi, args.bootstrap),
            'pair_interaction': boot(IA, 1320001 + 20*mi, args.bootstrap),
        }

        sw, sb, s12 = float(np.mean(S_w)), float(np.mean(S_b)), float(np.mean(S_12))
        nw, nb, n12 = float(np.mean(N_w)), float(np.mean(N_b)), float(np.mean(N_12))
        descriptive[m] = {
            'S_W2': sw, 'S_b2': sb, 'S_12': s12,
            'R_S': None if sw + sb <= 0 else float(s12 / (sw + sb)),
            'N_W2': nw, 'N_b2': nb, 'N_12': n12,
            'R_N': None if nw + nb <= 0 else float(n12 / (nw + nb)),
            'pair_vs_W2_sufficiency_ratio': None if abs(sw) <= 1e-30 else float(s12 / sw),
            'pair_vs_W2_necessity_ratio': None if abs(nw) <= 1e-30 else float(n12 / nw),
        }
        raw[f'H_plus_W2_b2__effect__{m}'] = hp
        raw[f'B_minus_W2_b2__effect__{m}'] = bmp
        raw[f'A_star_plus_W2_b2__effect__{m}'] = astp
        raw[f'I_S__{m}'] = IS
        raw[f'I_N__{m}'] = IN
        raw[f'I_Astar__{m}'] = IA

    # Conservative formal classifier: all four primary interaction CIs must agree.
    primary_signs = []
    for side in ('sufficiency','necessity'):
        for m in PRIMARY:
            primary_signs.append(sign_class(interactions[side][m]))
    if all(x == 'negative' for x in primary_signs):
        classification = 'P1_REDUNDANT_W2_DOMINANT'
    elif all(x == 'includes_zero' for x in primary_signs):
        classification = 'P2_ADDITIVE'
    elif all(x == 'positive' for x in primary_signs):
        classification = 'P3_SYNERGISTIC_FFN_SUPPRESSOR'
    else:
        classification = 'P4_MIXED_BACKGROUND_DEPENDENT'

    # Secondary readout only for the three new pair systems, common CRN/mask seed.
    xall, yall, _xte, _yte = core.load(args.data)
    x012, y012 = core.subset012(xall, yall)
    _xtr, _ytr, xva, _yva = core.split(x012, y012, .1, 7)
    toknet, _ = base.load_frozen_tokenizer(args.tokenizer, device)
    va, _ = base.encode_snn(toknet, xva, device, args.encode_batch)
    if not np.array_equal(np.asarray(va[validation_ids], dtype=np.int64), gt):
        raise RuntimeError('D6A.3 validation tokens mismatch frozen Gate-A ground truth')
    V = int(F.c.V); N = int(F.c.N)
    _rc, real_p = coarse.probability_from_ids(va, V)
    _pc, real_pos_p = spatial.position_probabilities(va, V, N)
    real_pair = spatial.pair_reference(va, V, N, 7)
    sec_seed = args.seed + 73000
    secondary = {
        'H_plus_W2_b2': secondary_eval(H_pair, va, real_p, real_pos_p, real_pair, sec_seed, args.eval_images, args.gen_samples),
        'B_minus_W2_b2': secondary_eval(B_mpair, va, real_p, real_pos_p, real_pair, sec_seed, args.eval_images, args.gen_samples),
        'A_star_plus_W2_b2': secondary_eval(Astar_pair, va, real_p, real_pos_p, real_pair, sec_seed, args.eval_images, args.gen_samples),
    }

    np.savez_compressed(args.out/'D6A3_EFFECT_ARRAYS.npz', **raw)
    summary = {
        'stage': 'GATE_D6A3_W2_B2_SUPPRESSOR_INTERACTION',
        'classification': classification,
        'source_D6A_run': 39687028646,
        'source_D6A1_run': 39777693838,
        'source_D6A2_run': 39806081064,
        'learner_modified': False,
        'training_performed': False,
        'expanded_block_search_performed': False,
        'pair_tested': ['W2','b2'],
        'bo_fixed_to_B': True,
        'purity': purity,
        'D6A2_reconstruction_calibration': calibration,
        'interactions': interactions,
        'descriptive_overlap': descriptive,
        'primary_interaction_signs': {
            'I_S_F61': sign_class(interactions['sufficiency']['F61']),
            'I_S_D_non61': sign_class(interactions['sufficiency']['D_non61']),
            'I_N_F61': sign_class(interactions['necessity']['F61']),
            'I_N_D_non61': sign_class(interactions['necessity']['D_non61']),
        },
        'secondary_new_pair_systems': secondary,
        'interpretation_boundary': (
            'P1/P2/P3 require agreement of all four primary paired-interaction bootstrap CIs. '
            'R_S and R_N are descriptive only. D6A.3 does not establish a final plasticity rule and does not test FF1 or any broader factorial interaction.'
        ),
    }
    (args.out/'D6A3_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    manifest = {
        'classification': classification,
        'eval_images': args.eval_images,
        'gen_samples': args.gen_samples,
        'bootstrap': args.bootstrap,
        'learner_modified': False,
        'training_performed': False,
        'pair_search_scope': 'W2+b2 only',
        'protected_baseline_modified': False,
        'no_PR_no_merge': True,
    }
    (args.out/'D6A3_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('D6A3_RESULT', json.dumps({
        'classification': classification,
        'primary_signs': summary['primary_interaction_signs'],
        'primary_overlap': {m: descriptive[m] for m in PRIMARY},
        'A_star_pair': {m: interactions['A_star_supportive'][m] for m in PRIMARY},
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
