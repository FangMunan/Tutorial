#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D8A — parameterization-invariant functional-mode identification.

Identification-only analysis of the frozen R/C/P/Q terminal systems after
D7A.2=A1.  Uses an independent discovery set, common teacher-forced probe
contexts, uncentered functional displacement spectra, pair-to-full subspace
capture, reciprocal pair/release alignment, and split-half stability.

No training, learner modification, activation intervention, or new parameter
search occurs here.  Only I1 authorizes a later D8B causal intervention.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
import gate_d_d7a1 as d71
import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base


LAYERS = ('z', 'u', 'h', 'y', 'logits')
K_REPORT = (1, 2, 3, 5, 10)
DISCOVERY_SEED = 88031
MASK_SEED = 88091
TOL = 1e-7
TAIL_TOL = 1e-10


def locate(root: Path, name: str) -> Path:
    return d3r.locate(root, name)


def clone(x):
    return pickle.loads(pickle.dumps(x, pickle.HIGHEST_PROTOCOL))


def sha_arrays(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        x = np.ascontiguousarray(np.asarray(a))
        h.update(str(x.dtype).encode('utf-8'))
        h.update(str(tuple(x.shape)).encode('utf-8'))
        h.update(x.tobytes())
    return h.hexdigest()


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


def select_discovery_ids(labels, excluded, n, seed):
    labels = np.asarray(labels, dtype=np.int64)
    excluded = set(int(x) for x in np.asarray(excluded, dtype=np.int64))
    if n != 256:
        raise ValueError('formal D8A uses exactly 256 discovery images')
    quotas = {0: 86, 1: 85, 2: 85}
    rng = np.random.default_rng(seed)
    selected = []
    for digit in (0, 1, 2):
        pool = np.asarray([i for i in np.flatnonzero(labels == digit) if int(i) not in excluded], dtype=np.int64)
        if len(pool) < quotas[digit]:
            raise RuntimeError({'D8A_discovery_pool_too_small': {'digit': digit, 'available': int(len(pool))}})
        pool = pool[rng.permutation(len(pool))]
        selected.extend(int(x) for x in pool[:quotas[digit]])
    selected = np.asarray(selected, dtype=np.int64)
    selected = selected[rng.permutation(len(selected))]
    if len(set(selected.tolist())) != n or any(int(x) in excluded for x in selected):
        raise RuntimeError('D8A discovery selection overlap/duplicate failure')
    split = np.zeros(n, dtype=np.int8)
    split[n // 2:] = 1
    return selected, split


def teacher_contexts(gt, image_id, N, mask_token):
    gt = np.asarray(gt, dtype=np.int64).reshape(-1)
    if len(gt) != N:
        raise ValueError('token length mismatch')
    rng = np.random.default_rng(MASK_SEED + int(image_id) * 7919)
    perm = rng.permutation(N)
    visible_counts = (0, N // 3, (2 * N) // 3)
    out = []
    for depth, vis in enumerate(visible_counts):
        cur = np.full(N, int(mask_token), dtype=np.int64)
        if vis:
            idx = perm[:vis]
            cur[idx] = gt[idx]
        masked = np.flatnonzero(cur == int(mask_token)).astype(np.int64)
        out.append((depth, int(vis), cur, masked))
    return out


def forward_probe(system, cur, masked, do_tail_check=False):
    xn, xraw = system.context_from_arr(np.asarray(cur, dtype=np.int64))
    system.rebuild(xn)
    z_rows, u_rows, h_rows, y_rows, l_rows = [], [], [], [], []
    max_tail_error = 0.0
    for qi, t0 in enumerate(np.asarray(masked, dtype=np.int64)):
        t = int(t0)
        a, _q, _num, _den = system._read(t)
        z = np.asarray(xraw[t], dtype=np.float64) + np.asarray(system.bo, dtype=np.float64) + np.asarray(system.WO, dtype=np.float64) @ np.asarray(a, dtype=np.float64)
        mu = float(np.mean(z))
        inv = 1.0 / math.sqrt(float(np.mean((z - mu) ** 2)) + float(system.eps2))
        zhat = (z - mu) * inv
        u = zhat * np.asarray(system.gamma2, dtype=np.float64) + np.asarray(system.beta2, dtype=np.float64)
        h = np.maximum(0.0, np.asarray(system.W1, dtype=np.float64) @ u + np.asarray(system.b1, dtype=np.float64) - float(system.theta))
        y = z + np.asarray(system.W2, dtype=np.float64) @ h + np.asarray(system.b2, dtype=np.float64)
        rms = math.sqrt(float(np.mean(y * y))) + 1e-8
        rf = np.empty(system.c.D + 1, dtype=np.float64)
        rf[:-1] = 0.5 * y / rms
        rf[-1] = 0.5
        logits = np.asarray(system.WR, dtype=np.float64) @ rf

        if do_tail_check and qi < 3:
            u0, h0, rf0, p0 = system._tail(np.asarray(a, dtype=np.float64), np.asarray(xraw[t], dtype=np.float64))
            p = core.softmax_numba(np.asarray(logits, dtype=np.float64))
            errs = [
                float(np.max(np.abs(np.asarray(u0, dtype=np.float64) - u))),
                float(np.max(np.abs(np.asarray(h0, dtype=np.float64) - h))),
                float(np.max(np.abs(np.asarray(rf0, dtype=np.float64) - rf))),
                float(np.max(np.abs(np.asarray(p0, dtype=np.float64) - np.asarray(p, dtype=np.float64)))),
            ]
            max_tail_error = max(max_tail_error, max(errs))

        z_rows.append(z)
        u_rows.append(u)
        h_rows.append(h)
        y_rows.append(y)
        l_rows.append(logits)

    return {
        'z': np.asarray(z_rows, dtype=np.float64),
        'u': np.asarray(u_rows, dtype=np.float64),
        'h': np.asarray(h_rows, dtype=np.float64),
        'y': np.asarray(y_rows, dtype=np.float64),
        'logits': np.asarray(l_rows, dtype=np.float64),
    }, float(max_tail_error)


def spectral(X, max_modes=10, centered=False):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError('spectral matrix must be 2-D with >=2 rows')
    mean = np.mean(X, axis=0)
    A = X - mean[None, :] if centered else X
    G = np.asarray(A.T @ A, dtype=np.float64)
    G = 0.5 * (G + G.T)
    vals, vecs = np.linalg.eigh(G)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(0.0, vals[order])
    vecs = vecs[:, order]
    total = float(np.sum(vals))
    if total <= 1e-30:
        ev = np.zeros_like(vals)
        erank = 0.0
        pr = 0.0
    else:
        p = vals / total
        ev = np.cumsum(p)
        nz = p[p > 1e-30]
        erank = float(np.exp(-np.sum(nz * np.log(nz))))
        pr = float(1.0 / np.sum(p * p))
    m = min(int(max_modes), vecs.shape[1])
    raw_energy = float(np.sum(X * X))
    mean_energy = float(X.shape[0] * np.sum(mean * mean))
    summary = {
        'n_rows': int(X.shape[0]),
        'dim': int(X.shape[1]),
        'total_energy': raw_energy,
        'analysis_energy': total,
        'effective_rank': erank,
        'participation_ratio': pr,
        'mean_displacement_energy_fraction': None if raw_energy <= 1e-30 else float(mean_energy / raw_energy),
        'EV': {str(k): float(ev[min(k, len(ev)) - 1]) for k in K_REPORT if k <= len(ev)},
        'singular_values': [float(math.sqrt(v)) for v in vals[:m]],
    }
    return summary, vecs[:, :m], vals[:m], mean


def principal_cos(Va, Vb, k):
    k = int(k)
    if Va.shape[1] < k or Vb.shape[1] < k:
        return []
    s = np.linalg.svd(np.asarray(Va[:, :k].T @ Vb[:, :k], dtype=np.float64), compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return [float(x) for x in s]


def frob_cos(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    den = math.sqrt(float(np.sum(A * A)) * float(np.sum(B * B)))
    return None if den <= 1e-30 else float(np.sum(A * B) / den)


def gamma_capture(full, Vpair, k):
    full = np.asarray(full, dtype=np.float64)
    den = float(np.sum(full * full))
    if den <= 1e-30 or Vpair.shape[1] < k:
        return None
    proj = full @ Vpair[:, :k]
    return float(np.sum(proj * proj) / den)


def energy_ratio(A, B):
    den = float(np.sum(np.asarray(B, dtype=np.float64) ** 2))
    return None if den <= 1e-30 else float(np.sum(np.asarray(A, dtype=np.float64) ** 2) / den)


def summarize_layer(full, pair, release, residual, row_split, row_depth):
    mats = {'full': full, 'pair': pair, 'release': release, 'residual': residual}
    specs = {}
    modes = {}
    evals = {}
    means = {}
    centered = {}
    for name, X in mats.items():
        specs[name], modes[name], evals[name], means[name] = spectral(X, 10, centered=False)
        centered[name], _v, _e, _m = spectral(X, 10, centered=True)

    cross = {
        'frob_cos_full_pair': frob_cos(full, pair),
        'pair_energy_over_full': energy_ratio(pair, full),
        'release_energy_over_full': energy_ratio(release, full),
        'residual_energy_over_full': energy_ratio(residual, full),
        'Gamma': {},
        'principal_full_pair': {},
        'principal_pair_release': {},
    }
    maxk = min(10, full.shape[1], pair.shape[1], release.shape[1])
    for k in K_REPORT:
        if k <= maxk:
            cross['Gamma'][str(k)] = gamma_capture(full, modes['pair'], k)
            cross['principal_full_pair'][str(k)] = principal_cos(modes['full'], modes['pair'], k)
            cross['principal_pair_release'][str(k)] = principal_cos(modes['pair'], modes['release'], k)

    split = {}
    split_modes = {}
    for half in (0, 1):
        take = np.asarray(row_split) == half
        split[str(half)] = {}
        split_modes[str(half)] = {}
        for name in ('full', 'pair'):
            ss, vv, _ee, _mm = spectral(mats[name][take], 10, centered=False)
            split[str(half)][name] = ss
            split_modes[str(half)][name] = vv
    split_alignment = {'full': {}, 'pair': {}}
    for k in K_REPORT:
        if k <= maxk:
            split_alignment['full'][str(k)] = principal_cos(split_modes['0']['full'], split_modes['1']['full'], k)
            split_alignment['pair'][str(k)] = principal_cos(split_modes['0']['pair'], split_modes['1']['pair'], k)

    depth = {}
    for d in sorted(set(int(x) for x in np.asarray(row_depth))):
        take = np.asarray(row_depth) == d
        sf, vf, _ef, _mf = spectral(full[take], 10, centered=False)
        sp, vp, _ep, _mp = spectral(pair[take], 10, centered=False)
        sr, vr, _er, _mr = spectral(release[take], 10, centered=False)
        kk = min(3, vf.shape[1], vp.shape[1], vr.shape[1])
        depth[str(d)] = {
            'full_EV3': sf['EV'].get(str(kk)),
            'pair_EV3': sp['EV'].get(str(kk)),
            'Gamma3': gamma_capture(full[take], vp, kk),
            'principal_full_pair_k': principal_cos(vf, vp, kk),
            'principal_pair_release_k': principal_cos(vp, vr, kk),
            'frob_cos_full_pair': frob_cos(full[take], pair[take]),
        }

    return {
        'spectra': specs,
        'centered_spectra_supportive': centered,
        'cross': cross,
        'split_half': split,
        'split_half_alignment': split_alignment,
        'mask_depth_supportive': depth,
    }, modes, evals, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--d6a-artifact', type=Path, required=True)
    ap.add_argument('--d7a-artifact', type=Path, required=True)
    ap.add_argument('--d7a2-artifact', type=Path, required=True)
    ap.add_argument('--anchor-artifact', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--discover-images', type=int, default=256)
    ap.add_argument('--encode-batch', type=int, default=256)
    ap.add_argument('--seed', type=int, default=7311)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.discover_images != 256:
        raise ValueError('formal D8A requires 256 discovery images')

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    device = torch.device('cpu')

    s72 = json.loads(locate(a.d7a2_artifact, 'D7A2_SUMMARY.json').read_text(encoding='utf-8'))
    if s72.get('classification') != 'A1_COMPACT_TWO_BLOCK_COMPENSATION':
        raise RuntimeError('D8A prerequisite failed: D7A.2 is not frozen A1')
    stored72 = np.load(locate(a.d7a2_artifact, 'D7A2_EFFECT_ARRAYS.npz'), allow_pickle=False)
    schedules = json.loads(locate(a.anchor_artifact, 'HISTORICAL_R6B3_R3_T8_REFERENCE_TRACE.json').read_text(encoding='utf-8'))

    # Reconstruct the exact D7A.2 terminal contrast.
    B = d6a1.load_d6a_terminal(locate(a.d6a_artifact, 'D6A_F_EM_WO_zero_bo_oracle_terminal.pkl'))
    Fsafe = d71.load_terminal(locate(a.d7a_artifact, 'D7A_F_terminal.pkl'))
    C = d71.load_terminal(locate(a.d7a_artifact, 'D7A_C_W2_safe_clamp_terminal.pkl'))
    R = d71.hybrid(B, Fsafe, ('W2',))
    P = d71.hybrid(R, C, ('b2', 'bo'))
    Q = d71.hybrid(C, R, ('b2', 'bo'))

    invariants = {
        'W2_R_equals_C': d71.arr_exact(R.W2, C.W2),
        'W2_R_C_max_abs': d71.max_abs(R.W2, C.W2),
        'WO_R_equals_C': d71.arr_exact(R.WO, C.WO),
        'WO_R_C_max_abs': d71.max_abs(R.WO, C.WO),
    }
    if not (invariants['W2_R_equals_C'] and invariants['W2_R_C_max_abs'] == 0.0
            and invariants['WO_R_equals_C'] and invariants['WO_R_C_max_abs'] == 0.0):
        raise RuntimeError({'D8A_R_C_invariant_failure': invariants})

    # Frozen Gate-A state calibration. These images are used only for provenance,
    # never for mode discovery.
    gt_gate = np.asarray(stored72['gt_tokens'], dtype=np.int64)
    shared_gate = np.asarray(stored72['shared_uniform'], dtype=np.float64)
    validation_ids = np.asarray(stored72['validation_ids'], dtype=np.int64)
    gate_systems = {'R': R, 'C': C, 'R_plus_C_b2_bO': P, 'C_minus_b2_bO': Q}
    gate_calibration = {}
    for label, system in gate_systems.items():
        ev = d71.gate_eval('D8A_CAL_' + label, system, schedules, gt_gate, shared_gate)
        gate_calibration[label] = d71.calibrate_full(ev, stored72, label, TOL)
    if not all(v['pass'] for v in gate_calibration.values()):
        raise RuntimeError({'D8A_D7A2_state_calibration_failure': gate_calibration})

    # Rebuild the frozen validation-token coordinate system and certify Gate-A IDs.
    xall, yall, _xte, _yte = core.load(a.data)
    x012, y012 = core.subset012(xall, yall)
    _xtr, _ytr, xva, yva = core.split(x012, y012, .1, 7)
    toknet, _ = base.load_frozen_tokenizer(a.tokenizer, device)
    va, _ = base.encode_snn(toknet, xva, device, a.encode_batch)
    va = np.asarray(va, dtype=np.int64)
    yva = np.asarray(yva, dtype=np.int64)
    if not np.array_equal(va[validation_ids], gt_gate):
        raise RuntimeError('D8A historical Gate-A tokens do not reproduce frozen coordinate system')

    discover_ids, discover_split = select_discovery_ids(yva, validation_ids, a.discover_images, DISCOVERY_SEED)
    discover_tokens = np.asarray(va[discover_ids], dtype=np.int64)
    discover_labels = np.asarray(yva[discover_ids], dtype=np.int64)
    if np.intersect1d(discover_ids, validation_ids).size:
        raise RuntimeError('D8A discovery/Gate-A leakage detected')
    provenance_hash = sha_arrays(discover_ids, discover_labels, discover_tokens,
                                  np.asarray([DISCOVERY_SEED, MASK_SEED], dtype=np.int64))

    id_rows = []
    for i, did in enumerate(discover_ids):
        id_rows.append({
            'discover_order': int(i),
            'validation_id': int(did),
            'digit': int(discover_labels[i]),
            'split_half': int(discover_split[i]),
        })
    write_csv(a.out / 'D8A_DISCOVERY_IDS.csv', id_rows)

    systems = {'R': R, 'C': C, 'P': P, 'Q': Q}
    diff_lists = {kind: {layer: [] for layer in LAYERS}
                  for kind in ('full', 'pair', 'release', 'residual')}
    row_split, row_depth, row_image, row_pos = [], [], [], []
    max_tail_error = 0.0

    for ii, did in enumerate(discover_ids):
        gt = discover_tokens[ii]
        for depth, _vis, cur, masked in teacher_contexts(gt, int(did), R.c.N, R.c.V):
            acts = {}
            for label, system in systems.items():
                aa, terr = forward_probe(system, cur, masked,
                                         do_tail_check=(ii < 2 and depth == 0))
                acts[label] = aa
                max_tail_error = max(max_tail_error, terr)
            nrow = len(masked)
            for layer in LAYERS:
                full = acts['C'][layer] - acts['R'][layer]
                pair = acts['P'][layer] - acts['R'][layer]
                release = acts['C'][layer] - acts['Q'][layer]
                residual = acts['C'][layer] - acts['P'][layer]
                if not np.allclose(full - pair, residual, rtol=0.0, atol=2e-12):
                    raise RuntimeError({'D8A_residual_identity_failure': {'layer': layer, 'image': int(did), 'depth': int(depth)}})
                diff_lists['full'][layer].append(full)
                diff_lists['pair'][layer].append(pair)
                diff_lists['release'][layer].append(release)
                diff_lists['residual'][layer].append(residual)
            row_split.extend([int(discover_split[ii])] * nrow)
            row_depth.extend([int(depth)] * nrow)
            row_image.extend([int(did)] * nrow)
            row_pos.extend([int(x) for x in masked])

    if max_tail_error >= TAIL_TOL:
        raise RuntimeError({'D8A_activation_extractor_tail_mismatch': max_tail_error})

    row_split = np.asarray(row_split, dtype=np.int8)
    row_depth = np.asarray(row_depth, dtype=np.int8)
    row_image = np.asarray(row_image, dtype=np.int64)
    row_pos = np.asarray(row_pos, dtype=np.int16)

    analysis = {}
    modes_out = {
        'discover_ids': discover_ids,
        'discover_labels': discover_labels,
        'discover_split': discover_split,
        'row_split': row_split,
        'row_depth': row_depth,
        'row_image': row_image,
        'row_pos': row_pos,
    }
    layer_rows = []
    depth_rows = []

    for layer in LAYERS:
        full = np.vstack(diff_lists['full'][layer])
        pair = np.vstack(diff_lists['pair'][layer])
        release = np.vstack(diff_lists['release'][layer])
        residual = np.vstack(diff_lists['residual'][layer])
        out, modes, evals, means = summarize_layer(full, pair, release, residual, row_split, row_depth)
        analysis[layer] = out
        for kind in ('full', 'pair', 'release', 'residual'):
            modes_out[f'{layer}__{kind}__V'] = modes[kind]
            modes_out[f'{layer}__{kind}__evals'] = evals[kind]
            modes_out[f'{layer}__{kind}__mean'] = means[kind]

        row = {
            'layer': layer,
            'dim': out['spectra']['full']['dim'],
            'rows': out['spectra']['full']['n_rows'],
            'full_effective_rank': out['spectra']['full']['effective_rank'],
            'pair_effective_rank': out['spectra']['pair']['effective_rank'],
            'full_mean_energy_fraction': out['spectra']['full']['mean_displacement_energy_fraction'],
            'pair_mean_energy_fraction': out['spectra']['pair']['mean_displacement_energy_fraction'],
            'frob_cos_full_pair': out['cross']['frob_cos_full_pair'],
            'pair_energy_over_full': out['cross']['pair_energy_over_full'],
            'residual_energy_over_full': out['cross']['residual_energy_over_full'],
        }
        for k in (1, 2, 3):
            if str(k) in out['cross']['Gamma']:
                row[f'full_EV{k}'] = out['spectra']['full']['EV'][str(k)]
                row[f'pair_EV{k}'] = out['spectra']['pair']['EV'][str(k)]
                row[f'Gamma{k}'] = out['cross']['Gamma'][str(k)]
                row[f'full_pair_min_cos{k}'] = min(out['cross']['principal_full_pair'][str(k)])
                row[f'pair_release_min_cos{k}'] = min(out['cross']['principal_pair_release'][str(k)])
        layer_rows.append(row)

        for d, dd in out['mask_depth_supportive'].items():
            depth_rows.append({
                'layer': layer,
                'depth': int(d),
                'full_EV3': dd['full_EV3'],
                'pair_EV3': dd['pair_EV3'],
                'Gamma3': dd['Gamma3'],
                'full_pair_min_cos3': min(dd['principal_full_pair_k']) if dd['principal_full_pair_k'] else None,
                'pair_release_min_cos3': min(dd['principal_pair_release_k']) if dd['principal_pair_release_k'] else None,
                'frob_cos_full_pair': dd['frob_cos_full_pair'],
            })

    yres = analysis['y']
    selected_k = None
    pooled_checks = {}
    for k in (1, 2, 3):
        key = str(k)
        checks = {
            'full_EV_ge_0p70': bool(yres['spectra']['full']['EV'][key] >= 0.70),
            'pair_EV_ge_0p80': bool(yres['spectra']['pair']['EV'][key] >= 0.80),
            'Gamma_ge_0p50': bool(yres['cross']['Gamma'][key] is not None and yres['cross']['Gamma'][key] >= 0.50),
            'full_pair_min_cos_ge_0p80': bool(min(yres['cross']['principal_full_pair'][key]) >= 0.80),
            'pair_release_min_cos_ge_0p80': bool(min(yres['cross']['principal_pair_release'][key]) >= 0.80),
        }
        checks['all'] = bool(all(checks.values()))
        pooled_checks[key] = checks
        if selected_k is None and checks['all']:
            selected_k = k

    stability = None
    stable = False
    if selected_k is not None:
        key = str(selected_k)
        full_cos = yres['split_half_alignment']['full'][key]
        pair_cos = yres['split_half_alignment']['pair'][key]
        stability = {
            'k': int(selected_k),
            'full_split_half_principal_cos': full_cos,
            'pair_split_half_principal_cos': pair_cos,
            'full_min_cos_ge_0p70': bool(min(full_cos) >= 0.70),
            'pair_min_cos_ge_0p70': bool(min(pair_cos) >= 0.70),
        }
        stable = bool(stability['full_min_cos_ge_0p70'] and stability['pair_min_cos_ge_0p70'])

    lowdim_but_not_shared = bool(
        yres['spectra']['full']['EV']['3'] >= 0.70
        and yres['spectra']['pair']['EV']['3'] >= 0.80
    )

    if selected_k is not None and stable:
        classification = 'I1_SHARED_LOW_DIMENSIONAL_FUNCTIONAL_CANDIDATE'
    elif selected_k is not None:
        classification = 'I4_UNSTABLE_SHARED_MODE'
    elif lowdim_but_not_shared:
        classification = 'I2_LOW_DIMENSIONAL_BUT_NOT_SHARED'
    else:
        classification = 'I3_HIGH_DIMENSIONAL_FUNCTIONAL_DISPLACEMENT'

    logits_confirmation = None
    if selected_k is not None:
        key = str(selected_k)
        lr = analysis['logits']
        logits_confirmation = {
            'k': int(selected_k),
            'full_EV': lr['spectra']['full']['EV'].get(key),
            'pair_EV': lr['spectra']['pair']['EV'].get(key),
            'Gamma': lr['cross']['Gamma'].get(key),
            'full_pair_principal_cos': lr['cross']['principal_full_pair'].get(key),
            'pair_release_principal_cos': lr['cross']['principal_pair_release'].get(key),
        }

    if classification == 'I1_SHARED_LOW_DIMENSIONAL_FUNCTIONAL_CANDIDATE':
        modes_out['D8B_frozen_y_pair_basis'] = modes_out['y__pair__V'][:, :selected_k]
    else:
        modes_out['D8B_frozen_y_pair_basis'] = np.empty((R.c.D, 0), dtype=np.float64)

    np.savez_compressed(a.out / 'D8A_MODES.npz', **modes_out)
    write_csv(a.out / 'D8A_LAYER_SUMMARY.csv', layer_rows)
    write_csv(a.out / 'D8A_MASK_DEPTH_SUMMARY.csv', depth_rows)

    summary = {
        'stage': 'GATE_D8A_PARAMETERIZATION_INVARIANT_FUNCTIONAL_MODE_IDENTIFICATION',
        'classification': classification,
        'source_runs': {
            'D7A2': 39942308903,
            'D7A': 39871827217,
            'D6A': 39687028646,
        },
        'training_performed': False,
        'learner_modified': False,
        'activation_intervention_performed': False,
        'new_parameter_search_performed': False,
        'gateA_used_for_provenance_only_not_discovery': True,
        'invariants': invariants,
        'gateA_state_calibration_to_D7A2': gate_calibration,
        'activation_extractor_max_tail_error': max_tail_error,
        'discovery': {
            'n_images': int(len(discover_ids)),
            'gateA_overlap_count': int(np.intersect1d(discover_ids, validation_ids).size),
            'split_half_counts': {str(i): int(np.sum(discover_split == i)) for i in (0, 1)},
            'digit_counts': {str(i): int(np.sum(discover_labels == i)) for i in (0, 1, 2)},
            'discovery_seed': DISCOVERY_SEED,
            'mask_seed': MASK_SEED,
            'provenance_sha256': provenance_hash,
            'common_input_teacher_forced': True,
            'visible_counts': [0, int(R.c.N // 3), int((2 * R.c.N) // 3)],
            'rows_total': int(len(row_split)),
        },
        'analysis': analysis,
        'primary_y_gate': {
            'pooled_checks_by_k': pooled_checks,
            'selected_k': None if selected_k is None else int(selected_k),
            'stability': stability,
            'lowdim_full_pair_at_k3': lowdim_but_not_shared,
            'basis_frozen_for_D8B_if_I1': 'top-k right-singular feature basis of DeltaY_pair = Y_P - Y_R',
        },
        'logits_downstream_confirmation': logits_confirmation,
        'decision': {
            'proceed_to_D8B': bool(classification == 'I1_SHARED_LOW_DIMENSIONAL_FUNCTIONAL_CANDIDATE'),
            'D8B_basis_source': 'pair_y_discovery_topk' if classification == 'I1_SHARED_LOW_DIMENSIONAL_FUNCTIONAL_CANDIDATE' else None,
            'D8B_must_use_historical_GateA_256_only': bool(classification == 'I1_SHARED_LOW_DIMENSIONAL_FUNCTIONAL_CANDIDATE'),
        },
        'interpretation_boundary': (
            'D8A is descriptive functional-mode identification on a discovery set disjoint from Gate-A. '
            'It cannot establish negative deployment utility, causal harmful-mode status, trajectory emergence, or a final admissibility rule. '
            'Only I1 freezes the pair-derived y-space basis and authorizes a separate D8B bidirectional causal activation intervention.'
        ),
    }
    (a.out / 'D8A_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (a.out / 'D8A_MANIFEST.json').write_text(json.dumps({
        'classification': classification,
        'discover_images': int(len(discover_ids)),
        'selected_k': None if selected_k is None else int(selected_k),
        'protected_baseline_modified': False,
        'no_PR_no_merge': True,
        'training_performed': False,
        'activation_intervention_performed': False,
        'discovery_gateA_disjoint': True,
    }, indent=2), encoding='utf-8')

    print('D8A_RESULT', json.dumps({
        'classification': classification,
        'selected_k': selected_k,
        'y_full_EV3': yres['spectra']['full']['EV']['3'],
        'y_pair_EV3': yres['spectra']['pair']['EV']['3'],
        'y_Gamma3': yres['cross']['Gamma']['3'],
        'y_full_pair_min_cos3': min(yres['cross']['principal_full_pair']['3']),
        'y_pair_release_min_cos3': min(yres['cross']['principal_pair_release']['3']),
        'stability': stability,
        'logits_confirmation': logits_confirmation,
        'proceed_to_D8B': summary['decision']['proceed_to_D8B'],
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
