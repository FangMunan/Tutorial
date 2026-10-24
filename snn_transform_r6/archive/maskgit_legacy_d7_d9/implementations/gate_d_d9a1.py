#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D9A.1 — exact same-state plastic functional velocity audit.

Pure diagnostic replay of the frozen D7A F/B/C trajectories.  No learner state
is modified by the audit.  For each context, y is reconstructed algebraically
from the exact pre- and post-update forward-parameter snapshots on the same
frozen discrete context.  No recurrent/Izhikevich/E/M state is advanced by the
velocity measurement.
"""
from __future__ import annotations

import argparse
import copy
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
import gate_d_d7a as d7a
import gate_d_plasticity_audit as gd
import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
import r6b3_context_epoch_training as trainer
from maskgit_input_representation import build_input_embeddings, context_from_arr_unconditional
from r6b4_full_forward_plasticity import FORWARD_PARAMETER_KEYS, upgrade_r6b3_system

CHECKPOINTS = (64, 128, 256, 512, 768, 1024)
PRIMARY_ARMS = ('F', 'B', 'C')
ALL_VEL_ARMS = ('F', 'B', 'C', 'C_native')
DIAG_IMAGES = 128
DIAG_SEED = 2902601
DIAG_MASK_SEED = 2903611
EPS = 1e-30
PARITY_TOL = 3e-11
GATE_TOL = 1e-7


def locate(root: Path, name: str) -> Path:
    return d3r.locate(root, name)


def write_csv(path: Path, rows):
    if not rows:
        path.write_text('', encoding='utf-8'); return
    keys = []; seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def clone(x):
    return pickle.loads(pickle.dumps(x, pickle.HIGHEST_PROTOCOL))


def load_terminal(path: Path):
    with path.open('rb') as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or 'system' not in obj:
        raise RuntimeError(f'bad terminal schema: {path}')
    return obj['system']


def snapshot_forward(system):
    out = {}
    for k in FORWARD_PARAMETER_KEYS:
        v = getattr(system, k)
        out[k] = np.array(v, copy=True) if isinstance(v, np.ndarray) else copy.deepcopy(v)
    # Fixed forward quantities required by the stateless reconstruction.
    out['theta'] = np.array(system.theta, copy=True)
    out['omega'] = np.array(system.omega, copy=True)
    out['eps2'] = float(system.eps2)
    return out


def forward_update_norm(a, b):
    ss = 0.0
    for k in FORWARD_PARAMETER_KEYS:
        x = np.asarray(a[k], dtype=np.float64)
        y = np.asarray(b[k], dtype=np.float64)
        d = y - x
        ss += float(np.dot(d.reshape(-1), d.reshape(-1)))
    return math.sqrt(ss)


def stateless_y(snapshot, cfg, arr, masked):
    """Exact algebraic y on one frozen context; no learner/dynamical state advances."""
    arr = np.asarray(arr, dtype=np.int64)
    masked = np.asarray(masked, dtype=np.int64)
    xn, xraw = context_from_arr_unconditional(arr, np.asarray(snapshot['tok']), np.asarray(snapshot['pos']))
    N, D, H = cfg.N, cfg.D, cfg.H
    hd = D // H
    WQ = np.asarray(snapshot['WQ'], dtype=np.float64)
    WK = np.asarray(snapshot['WK'], dtype=np.float64)
    WV = np.asarray(snapshot['WV'], dtype=np.float64)
    q = (xn @ WQ.T + np.asarray(snapshot['bq'])).reshape(N, H, hd)
    k = (xn @ WK.T + np.asarray(snapshot['bk'])).reshape(N, H, hd)
    vv = (xn @ WV.T + np.asarray(snapshot['bv'])).reshape(N, H, hd)
    omega = np.asarray(snapshot['omega'], dtype=np.float64)

    aa = np.empty((len(masked), D), dtype=np.float64)
    for h in range(H):
        P = np.vstack([core.phi_k(k[n, h], omega[h]) for n in range(N)])
        g = np.sum(P, axis=0)
        Hs = P.T @ vv[:, h, :]
        PQ = np.vstack([core.phi_q(q[int(t), h], omega[h]) for t in masked])
        den = np.maximum(1e-8, PQ @ g)
        aa[:, h * hd:(h + 1) * hd] = (PQ @ Hs) / den[:, None]

    z = xraw[masked] + np.asarray(snapshot['bo']) + aa @ np.asarray(snapshot['WO']).T
    mu = np.mean(z, axis=1, keepdims=True)
    inv = 1.0 / np.sqrt(np.mean((z - mu) ** 2, axis=1, keepdims=True) + float(snapshot['eps2']))
    zhat = (z - mu) * inv
    u = zhat * np.asarray(snapshot['gamma2'])[None, :] + np.asarray(snapshot['beta2'])[None, :]
    h = np.maximum(0.0, u @ np.asarray(snapshot['W1']).T + np.asarray(snapshot['b1'])[None, :] - np.asarray(snapshot['theta'])[None, :])
    y = z + h @ np.asarray(snapshot['W2']).T + np.asarray(snapshot['b2'])[None, :]
    return np.asarray(y, dtype=np.float64)


def ordinary_y(system, arr, masked):
    """Read-only parity path on a clone of the system."""
    s = clone(system)
    arr = np.asarray(arr, dtype=np.int64)
    masked = np.asarray(masked, dtype=np.int64)
    xn, xraw = s.context_from_arr(arr)
    s.rebuild(xn)
    out = []
    for t0 in masked:
        t = int(t0)
        a, _q, _n, _d = s._read(t)
        z = np.asarray(xraw[t], dtype=np.float64) + np.asarray(s.bo) + np.asarray(s.WO) @ np.asarray(a)
        mu = float(np.mean(z))
        inv = 1.0 / math.sqrt(float(np.mean((z - mu) ** 2)) + float(s.eps2))
        zh = (z - mu) * inv
        u = zh * np.asarray(s.gamma2) + np.asarray(s.beta2)
        h = np.maximum(0.0, np.asarray(s.W1) @ u + np.asarray(s.b1) - np.asarray(s.theta))
        y = z + np.asarray(s.W2) @ h + np.asarray(s.b2)
        out.append(y)
    return np.asarray(out, dtype=np.float64)


def parity_check(system, arr, masked):
    snap = snapshot_forward(system)
    a = stateless_y(snap, system.c, arr, masked)
    b = ordinary_y(system, arr, masked)
    err = float(np.max(np.abs(a - b))) if a.size else 0.0
    return {'max_abs': err, 'pass': bool(err < PARITY_TOL), 'n_rows': int(len(masked))}


def velocity(pre, post, cfg, arr, masked, v):
    y0 = stateless_y(pre, cfg, arr, masked)
    y1 = stateless_y(post, cfg, arr, masked)
    dy = y1 - y0
    mean_dy = np.mean(dy, axis=0) if len(dy) else np.zeros(cfg.D, dtype=np.float64)
    dq = float(np.dot(np.asarray(v, dtype=np.float64), mean_dy))
    dth = float(forward_update_norm(pre, post))
    dyn = float(np.linalg.norm(mean_dy))
    row_proj = dy @ np.asarray(v, dtype=np.float64) if len(dy) else np.empty(0)
    return {
        'dq': dq,
        'positive_flux': max(0.0, dq),
        'negative_flux': max(0.0, -dq),
        'abs_flux': abs(dq),
        'forward_update_norm': dth,
        'mean_dy_norm': dyn,
        'e_pos_per_param_norm': float(max(0.0, dq) / (dth + EPS)),
        'r_directional': float(dq / (dyn + EPS)),
        'row_dq_mean': float(np.mean(row_proj)) if len(row_proj) else 0.0,
        'row_dq_std': float(np.std(row_proj)) if len(row_proj) else 0.0,
        'row_positive_fraction': float(np.mean(row_proj > 0)) if len(row_proj) else 0.0,
    }


def aggregate(rows, arm):
    rr = [r for r in rows if r['arm'] == arm]
    dq = np.asarray([r['dq'] for r in rr], dtype=np.float64)
    pos = np.maximum(dq, 0.0); neg = np.maximum(-dq, 0.0)
    e = np.asarray([r['e_pos_per_param_norm'] for r in rr], dtype=np.float64)
    rv = np.asarray([r['r_directional'] for r in rr], dtype=np.float64)
    return {
        'contexts': int(len(rr)),
        'Q_net': float(np.sum(dq)),
        'Q_plus': float(np.sum(pos)),
        'Q_minus': float(np.sum(neg)),
        'Q_abs': float(np.sum(np.abs(dq))),
        'mean_dq': float(np.mean(dq)),
        'mean_positive_flux': float(np.mean(pos)),
        'fraction_dq_positive': float(np.mean(dq > 0)),
        'harm_fraction_Qplus_over_Qabs': float(np.sum(pos) / (np.sum(np.abs(dq)) + EPS)),
        'mean_e_pos_per_param_norm': float(np.mean(e)),
        'median_e_pos_per_param_norm': float(np.median(e)),
        'mean_r_directional': float(np.mean(rv)),
    }


def paired_bootstrap(x, y, seed, nboot):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError('paired shape mismatch')
    return d6a.bootstrap_mean(x - y, seed, nboot)


def select_diag_ids(yva, excluded, n=DIAG_IMAGES, seed=DIAG_SEED):
    excluded = set(int(x) for x in np.asarray(excluded, dtype=np.int64).tolist())
    rng = np.random.default_rng(seed)
    quotas = {0: n // 3, 1: n // 3, 2: n // 3}
    for d in range(n - sum(quotas.values())):
        quotas[d] += 1
    ids = []
    for digit in (0, 1, 2):
        pool = np.asarray([i for i in np.flatnonzero(np.asarray(yva) == digit) if int(i) not in excluded], dtype=np.int64)
        pool = pool[rng.permutation(len(pool))]
        ids.extend(int(x) for x in pool[:quotas[digit]])
    ids = np.asarray(ids, dtype=np.int64)
    ids = ids[rng.permutation(len(ids))]
    if len(ids) != n or len(np.unique(ids)) != n or any(int(x) in excluded for x in ids):
        raise RuntimeError('diagnostic ID selection failure')
    return ids


def diagnostic_contexts(gt, image_id, N, mask_token):
    gt = np.asarray(gt, dtype=np.int64)
    rng = np.random.default_rng(DIAG_MASK_SEED + int(image_id) * 7919)
    perm = rng.permutation(N)
    out = []
    for depth, vis in enumerate((0, N // 3, (2 * N) // 3)):
        cur = np.full(N, int(mask_token), dtype=np.int64)
        if vis:
            idx = perm[:vis]; cur[idx] = gt[idx]
        masked = np.flatnonzero(cur == int(mask_token)).astype(np.int64)
        out.append((depth, cur, masked))
    return out


def checkpoint_drift(F, B, C, va, diag_ids, v):
    sf, sb, sc = snapshot_forward(F), snapshot_forward(B), snapshot_forward(C)
    bvals = []; cvals = []
    for did in diag_ids:
        gt = np.asarray(va[int(did)], dtype=np.int64)
        for _depth, cur, masked in diagnostic_contexts(gt, int(did), F.c.N, F.c.V):
            yf = stateless_y(sf, F.c, cur, masked)
            yb = stateless_y(sb, F.c, cur, masked)
            yc = stateless_y(sc, F.c, cur, masked)
            bvals.extend(((yb - yf) @ v).tolist())
            cvals.extend(((yc - yf) @ v).tolist())
    bvals = np.asarray(bvals, dtype=np.float64); cvals = np.asarray(cvals, dtype=np.float64)
    return {
        'B_minus_F_mean_projection': float(np.mean(bvals)),
        'B_minus_F_std_projection': float(np.std(bvals)),
        'C_minus_F_mean_projection': float(np.mean(cvals)),
        'C_minus_F_std_projection': float(np.std(cvals)),
        'rows': int(len(bvals)),
    }


def cumulative_at(rows, arm):
    q = np.asarray([r['dq'] for r in rows if r['arm'] == arm], dtype=np.float64)
    return {
        'Q_net': float(np.sum(q)), 'Q_plus': float(np.sum(np.maximum(q, 0.0))),
        'Q_minus': float(np.sum(np.maximum(-q, 0.0))), 'Q_abs': float(np.sum(np.abs(q))),
    }


def sha_arrays(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        x = np.ascontiguousarray(a)
        h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()


def run_audit(start, tokens, indices, rng_state, v, va, diag_ids):
    F = d6a.make_d6a(start, d7a.F_LABEL)
    B = d6a.make_d6a(start, d7a.B_LABEL)
    C = d6a.make_d6a(start, d7a.B_LABEL)
    rngs = {k: gd.clone_rng(rng_state) for k in PRIMARY_ARMS}
    streams = {k: [] for k in PRIMARY_ARMS}
    rows = []; checkpoints = []; parity = []
    max_post_clamp = 0.0; cumulative_clamp = 0.0

    for j, image_index in enumerate(indices, start=1):
        if d7a.max_abs(C.W2, F.W2) != 0.0:
            raise RuntimeError({'D9A1_precontext_W2_not_safe': j})

        for s, r in ((F, rngs['F']), (B, rngs['B']), (C, rngs['C'])):
            before = d3r.jstate(r); d3r.reset_boundary(s, 'F_EM')
            if d3r.jstate(r) != before:
                raise RuntimeError('reset consumed external RNG')

        pre = {k: snapshot_forward(s) for k, s in (('F', F), ('B', B), ('C', C))}
        truth = np.asarray(tokens[int(image_index)], dtype=np.int64)
        recF, evF = d6a.run_context(F, j, int(image_index), truth, rngs['F'])
        recB, evB = d6a.run_context(B, j, int(image_index), truth, rngs['B'])
        recC, evC = d6a.run_context(C, j, int(image_index), truth, rngs['C'])

        sr = {k: d6a.stream_row(j, int(image_index), ev) for k, ev in (('F', evF), ('B', evB), ('C', evC))}
        if not (sr['F'] == sr['B'] == sr['C']):
            raise RuntimeError({'D9A1_stream_divergence': j})
        if not (d3r.jstate(rngs['F']) == d3r.jstate(rngs['B']) == d3r.jstate(rngs['C'])):
            raise RuntimeError({'D9A1_external_rng_divergence': j})
        for k in PRIMARY_ARMS: streams[k].append(sr[k])

        masked = np.asarray(evF['masked'], dtype=np.int64)
        arr = truth.copy(); arr[masked] = core.MASK
        if j == 1 or j in CHECKPOINTS:
            for label, system in (('F', F), ('B', B), ('C_native', C)):
                pc = parity_check(system, arr, masked)
                pc.update({'context': int(j), 'arm': label}); parity.append(pc)
                if not pc['pass']:
                    raise RuntimeError({'D9A1_stateless_forward_parity_failure': pc})

        postF = snapshot_forward(F); postB = snapshot_forward(B); postCnative = snapshot_forward(C)
        for label, p0, p1 in (('F', pre['F'], postF), ('B', pre['B'], postB), ('C_native', pre['C'], postCnative)):
            m = velocity(p0, p1, F.c, arr, masked, v)
            rows.append({'context': int(j), 'image_index': int(image_index), 'arm': label, 'nmask': int(len(masked)), **m})

        # D7A intervention: actual C trajectory ends the context on time-matched safe W2.
        correction = np.asarray(F.W2, dtype=np.float64) - np.asarray(C.W2, dtype=np.float64)
        cumulative_clamp += float(np.linalg.norm(correction))
        C.W2 = np.array(F.W2, copy=True)
        err = d7a.max_abs(C.W2, F.W2); max_post_clamp = max(max_post_clamp, err)
        if err != 0.0:
            raise RuntimeError({'D9A1_postcontext_W2_clamp_failure': j, 'err': err})
        postC = snapshot_forward(C)
        m = velocity(pre['C'], postC, F.c, arr, masked, v)
        rows.append({'context': int(j), 'image_index': int(image_index), 'arm': 'C', 'nmask': int(len(masked)), **m})

        if j in CHECKPOINTS:
            drift = checkpoint_drift(F, B, C, va, diag_ids, v)
            tr = {'context': int(j), **drift,
                  'C_cumulative_W2_clamp_correction': float(cumulative_clamp)}
            for arm in ALL_VEL_ARMS:
                for kk, vv in cumulative_at(rows, arm).items(): tr[f'{arm}_{kk}'] = vv
            checkpoints.append(tr)
            print('D9A1_PROGRESS', json.dumps(tr), flush=True)

    return {
        'F': F, 'B': B, 'C': C, 'rows': rows, 'checkpoints': checkpoints,
        'parity': parity, 'streams': streams,
        'rng_after': {k: d3r.jstate(r) for k, r in rngs.items()},
        'max_post_clamp_W2_abs_error': max_post_clamp,
        'cumulative_W2_clamp_correction': cumulative_clamp,
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
    ap.add_argument('--anchor-artifact', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--contexts', type=int, default=1024)
    ap.add_argument('--bootstrap', type=int, default=10000)
    ap.add_argument('--encode-batch', type=int, default=256)
    ap.add_argument('--seed', type=int, default=7311)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if a.contexts != 1024: raise ValueError('formal D9A.1 requires 1024 contexts')

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads)
    device = torch.device('cpu')

    s7 = json.loads(locate(a.d7a_artifact, 'D7A_SUMMARY.json').read_text(encoding='utf-8'))
    s8a = json.loads(locate(a.d8a_artifact, 'D8A_SUMMARY.json').read_text(encoding='utf-8'))
    s8b = json.loads(locate(a.d8b_artifact, 'D8B_SUMMARY.json').read_text(encoding='utf-8'))
    s9 = json.loads(locate(a.d9a_artifact, 'D9A_SUMMARY.json').read_text(encoding='utf-8'))
    if s7.get('classification') != 'T0_NO_W2_TRAJECTORY_RESCUE': raise RuntimeError('D7A prerequisite mismatch')
    if s8a.get('classification') != 'I1_SHARED_LOW_DIMENSIONAL_FUNCTIONAL_CANDIDATE' or int(s8a.get('selected_k', -1)) != 1:
        raise RuntimeError('D8A prerequisite mismatch')
    if s8b.get('classification') != 'B1_BIDIRECTIONAL_CAUSAL_HARMFUL_FUNCTIONAL_MODE': raise RuntimeError('D8B prerequisite mismatch')
    if s9.get('classification') != 'R0_NO_HARD_FUNCTIONAL_TRAJECTORY_RESCUE': raise RuntimeError('D9A prerequisite mismatch')

    d8m = np.load(locate(a.d8a_artifact, 'D8A_MODES.npz'), allow_pickle=False)
    d8be = np.load(locate(a.d8b_artifact, 'D8B_EFFECT_ARRAYS.npz'), allow_pickle=False)
    basis = np.asarray(d8m['D8B_frozen_y_pair_basis'], dtype=np.float64)
    if basis.shape != (96, 1): raise RuntimeError(f'bad frozen basis shape {basis.shape}')
    v = basis[:, 0].copy(); n = float(np.linalg.norm(v)); v /= n
    if abs(n - 1.0) > 1e-10: raise RuntimeError('D8A basis not normalized')
    if not np.allclose(np.asarray(d8be['frozen_y_basis'])[:, 0], v, rtol=0.0, atol=1e-10):
        raise RuntimeError('D8A/D8B frozen basis mismatch')

    stored7 = np.load(locate(a.d7a_artifact, 'D7A_EFFECT_ARRAYS.npz'), allow_pickle=False)
    validation_ids = np.asarray(stored7['validation_ids'], dtype=np.int64)
    gt_gate = np.asarray(stored7['gt_tokens'], dtype=np.int64)
    shared_gate = np.asarray(stored7['shared_uniform'], dtype=np.float64)
    schedules = json.loads(locate(a.anchor_artifact, 'HISTORICAL_R6B3_R3_T8_REFERENCE_TRACE.json').read_text(encoding='utf-8'))

    checkpoint_sha = d3r.sha_file(a.checkpoint)
    source, meta, train_rng, schedule = trainer.load_full_checkpoint(a.checkpoint)
    if checkpoint_sha != d6a.EXPECTED_CHECKPOINT_SHA256 or abs(float(meta.get('context_epoch_equivalent', -1.0)) - 2.0) > 1e-12:
        raise RuntimeError('D9A1 start checkpoint identity failed')
    offset = int(schedule['offset'])
    indices = np.asarray(schedule['permutation'], dtype=np.int64)[offset:offset + a.contexts]
    rng_state = copy.deepcopy(train_rng.bit_generator.state)
    if len(indices) != 1024: raise RuntimeError('checkpoint schedule lacks 1024 contexts')

    xall, yall, _xte, _yte = core.load(a.data)
    x012, y012 = core.subset012(xall, yall)
    xtr, _ytr, xva, yva = core.split(x012, y012, .1, 7)
    toknet, _ = base.load_frozen_tokenizer(a.tokenizer, device)
    tr, _ = base.encode_snn(toknet, xtr, device, a.encode_batch)
    va, _ = base.encode_snn(toknet, xva, device, a.encode_batch)
    tr = np.asarray(tr, dtype=np.int64); va = np.asarray(va, dtype=np.int64); yva = np.asarray(yva, dtype=np.int64)
    if not np.array_equal(va[validation_ids], gt_gate): raise RuntimeError('Gate-A coordinate mismatch')
    tok0, pos0, _ = build_input_embeddings(toknet, source.c.D, grid=7)
    start = clone(source); upgrade_r6b3_system(start, tok0, pos0, aux_lr_scale=1.0)

    discover_ids = np.asarray(d8m['discover_ids'], dtype=np.int64)
    excluded = np.unique(np.concatenate([validation_ids, discover_ids]))
    diag_ids = select_diag_ids(yva, excluded)
    if np.intersect1d(diag_ids, validation_ids).size or np.intersect1d(diag_ids, discover_ids).size:
        raise RuntimeError('D9A1 diagnostic leakage')
    diag_labels = yva[diag_ids].astype(np.int8); diag_tokens = va[diag_ids].astype(np.int16)
    diag_hash = sha_arrays(diag_ids, diag_labels, diag_tokens, np.asarray([DIAG_SEED, DIAG_MASK_SEED], dtype=np.int64))
    write_csv(a.out / 'D9A1_DIAGNOSTIC_IDS.csv', [
        {'order': i, 'validation_id': int(x), 'digit': int(diag_labels[i])} for i, x in enumerate(diag_ids)
    ])

    run = run_audit(start, tr, indices, rng_state, v, va, diag_ids)
    F, B, C = run['F'], run['B'], run['C']
    if not (run['streams']['F'] == run['streams']['B'] == run['streams']['C']): raise RuntimeError('stream exactness failed')
    if not (run['rng_after']['F'] == run['rng_after']['B'] == run['rng_after']['C']): raise RuntimeError('external RNG exactness failed')
    if run['max_post_clamp_W2_abs_error'] != 0.0 or not np.array_equal(C.W2, F.W2): raise RuntimeError('C safe-W2 identity failed')

    oldF = load_terminal(locate(a.d7a_artifact, 'D7A_F_terminal.pkl'))
    oldB = load_terminal(locate(a.d7a_artifact, 'D7A_B_terminal.pkl'))
    oldC = load_terminal(locate(a.d7a_artifact, 'D7A_C_W2_safe_clamp_terminal.pkl'))
    model_replay = {
        'F': d7a.model_field_compare_tolerant(F, oldF),
        'B': d7a.model_field_compare_tolerant(B, oldB),
        'C': d7a.model_field_compare_tolerant(C, oldC),
    }
    if not all(x['pass'] for x in model_replay.values()): raise RuntimeError({'D9A1_model_replay_failed': model_replay})

    gate = {
        'F': d6a.gate_eval('D9A1_F', F, schedules, gt_gate, shared_gate),
        'B': d6a.gate_eval('D9A1_B', B, schedules, gt_gate, shared_gate),
        'C': d6a.gate_eval('D9A1_C', C, schedules, gt_gate, shared_gate),
    }
    gate_replay = {k: d7a.gate_calibration(gate[k], stored7, k, GATE_TOL) for k in PRIMARY_ARMS}
    if not all(x['pass'] for x in gate_replay.values()): raise RuntimeError({'D9A1_GateA_replay_failed': gate_replay})

    rows = run['rows']
    by = {arm: [r for r in rows if r['arm'] == arm] for arm in ALL_VEL_ARMS}
    arrays = {}
    for arm in ALL_VEL_ARMS:
        arrays[arm] = {
            'dq': np.asarray([r['dq'] for r in by[arm]], dtype=np.float64),
            'positive': np.asarray([r['positive_flux'] for r in by[arm]], dtype=np.float64),
            'e': np.asarray([r['e_pos_per_param_norm'] for r in by[arm]], dtype=np.float64),
            'r': np.asarray([r['r_directional'] for r in by[arm]], dtype=np.float64),
            'param_norm': np.asarray([r['forward_update_norm'] for r in by[arm]], dtype=np.float64),
        }
    contrasts = {
        'B_minus_F_positive_flux': paired_bootstrap(arrays['B']['positive'], arrays['F']['positive'], 1910001, a.bootstrap),
        'C_minus_F_positive_flux': paired_bootstrap(arrays['C']['positive'], arrays['F']['positive'], 1910002, a.bootstrap),
        'B_minus_F_normalized_e': paired_bootstrap(arrays['B']['e'], arrays['F']['e'], 1910003, a.bootstrap),
        'C_minus_F_normalized_e': paired_bootstrap(arrays['C']['e'], arrays['F']['e'], 1910004, a.bootstrap),
        'C_native_minus_C_actual_positive_flux': paired_bootstrap(arrays['C_native']['positive'], arrays['C']['positive'], 1910005, a.bootstrap),
    }
    rawB = bool(contrasts['B_minus_F_positive_flux']['ci_entirely_positive'])
    rawC = bool(contrasts['C_minus_F_positive_flux']['ci_entirely_positive'])
    normB = bool(contrasts['B_minus_F_normalized_e']['ci_entirely_positive'])
    normC = bool(contrasts['C_minus_F_normalized_e']['ci_entirely_positive'])
    if rawB and rawC and normB and normC:
        classification = 'V1_STRONG_PLASTIC_FUNCTIONAL_VELOCITY_SUPPORT'
    elif rawB and rawC:
        classification = 'V2_RAW_EXCESS_WITH_SIZE_CONFOUND'
    elif rawB or rawC:
        classification = 'V3_PARTIAL_ARM_SUPPORT'
    else:
        classification = 'V0_NO_HARMFUL_PLASTIC_VELOCITY_EXCESS'

    aggregates = {arm: aggregate(rows, arm) for arm in ALL_VEL_ARMS}
    write_csv(a.out / 'D9A1_CONTEXT_VELOCITY.csv', rows)
    write_csv(a.out / 'D9A1_CHECKPOINT_DRIFT.csv', run['checkpoints'])

    npz = {
        'basis_v': v,
        'diagnostic_ids': diag_ids,
        'diagnostic_labels': diag_labels,
        'diagnostic_tokens': diag_tokens,
        'validation_ids': validation_ids,
    }
    for arm in ALL_VEL_ARMS:
        for key, val in arrays[arm].items(): npz[f'{arm}__{key}'] = val
    np.savez_compressed(a.out / 'D9A1_VELOCITY_ARRAYS.npz', **npz)

    summary = {
        'stage': 'GATE_D9A1_EXACT_SAME_STATE_PLASTIC_FUNCTIONAL_VELOCITY_AUDIT',
        'classification': classification,
        'source_runs': {'D7A': 39871827217, 'D8A': 39979543621, 'D8B': 40021397096, 'D9A': 40037197274},
        'training_modified': False,
        'learner_modified': False,
        'velocity_intervention_performed': False,
        'basis_source': 'frozen D8A/D8B y-space rank-1 basis; D8B harmful orientation',
        'same_state_definition': 'finite-step pre/post forward-parameter snapshots on identical discrete context; algebraic forward only; no recurrent/Izh/E/M state advancement',
        'C_primary_definition': 'actual post-W2-clamp update; C_native is supportive pre-clamp proposal',
        'checkpoint_sha256': checkpoint_sha,
        'diagnostic_probe': {
            'images': int(len(diag_ids)), 'seed': DIAG_SEED, 'mask_seed': DIAG_MASK_SEED,
            'GateA_overlap_count': int(np.intersect1d(diag_ids, validation_ids).size),
            'D8A_discovery_overlap_count': int(np.intersect1d(diag_ids, discover_ids).size),
            'provenance_sha256': diag_hash,
        },
        'stateless_forward_parity': {
            'all_pass': bool(all(x['pass'] for x in run['parity'])),
            'max_abs': float(max(x['max_abs'] for x in run['parity'])),
            'checks': run['parity'],
        },
        'trajectory_provenance': {
            'all_arm_stream_exact': True, 'all_arm_external_rng_exact': True,
            'model_replay_to_D7A': model_replay, 'GateA_replay_to_D7A': gate_replay,
            'C_max_postclamp_W2_abs_error': run['max_post_clamp_W2_abs_error'],
            'C_cumulative_W2_clamp_correction': run['cumulative_W2_clamp_correction'],
        },
        'aggregates': aggregates,
        'paired_contrasts': contrasts,
        'checkpoint_drift': run['checkpoints'],
        'decision': {
            'B_raw_positive_flux_excess': rawB,
            'C_raw_positive_flux_excess': rawC,
            'B_normalized_e_excess': normB,
            'C_normalized_e_excess': normC,
            'D9A2_authorized': bool(classification == 'V1_STRONG_PLASTIC_FUNCTIONAL_VELOCITY_SUPPORT'),
        },
        'interpretation_boundary': (
            'V1 would establish excess harmful-oriented finite-step plastic functional velocity in both B and the actual safe-W2-clamped C trajectory, beyond overall forward-update size. '
            'It would not prove that every positive context update is individually harmful or establish a final local/MSDL/SNN rule. Non-V1 outcomes do not authorize a plastic-velocity clamp.'
        ),
    }
    (a.out / 'D9A1_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    manifest = {
        'classification': classification, 'contexts': 1024, 'bootstrap': int(a.bootstrap),
        'diagnostic_images': int(len(diag_ids)), 'training_modified': False,
        'learner_modified': False, 'velocity_intervention_performed': False,
        'protected_baseline_modified': False, 'no_PR_no_merge': True,
    }
    (a.out / 'D9A1_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print('D9A1_RESULT', json.dumps({
        'classification': classification,
        'aggregates': {k: aggregates[k] for k in ALL_VEL_ARMS},
        'contrasts': contrasts,
        'decision': summary['decision'],
        'C_clamp_correction': run['cumulative_W2_clamp_correction'],
        'parity_max_abs': summary['stateless_forward_parity']['max_abs'],
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
