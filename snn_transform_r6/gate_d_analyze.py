#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline mathematical analysis for Gate D audit artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

TRACK = ('WQ', 'WK', 'WV', 'WO', 'W1', 'W2', 'WR')
ARMS = ('T0', 'T1', 'T2')


def read_csv(path: Path) -> List[Dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    out = []
    with path.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rr = dict(r)
            for k in ('context', 'rank', 'position', 'target', 'pred', 'image_index', 'nmask', 'exposure_n', 'mirror_seed'):
                if k in rr and rr[k] != '': rr[k] = int(rr[k])
            for k in ('p_target', 'p61', 'u_norm', 'g_norm', 'signed_descent', 'efficiency', 'cosine', 'A_M', 'A_E',
                      'w2_native_update_identity_max_abs', 'w2_actual_step_norm', 'w2_oracle_mean_norm',
                      'context_signed_descent', 'context_efficiency', 'context_cosine'):
                if k in rr and rr[k] != '': rr[k] = float(rr[k])
            for k in ('correct', 'false61', 'gt61', 'pred61'):
                if k in rr: rr[k] = str(rr[k]).lower() == 'true'
            out.append(rr)
    return out


def finite(x):
    a = np.asarray(list(x), dtype=np.float64)
    return a[np.isfinite(a)]


def ci_boot(values, stat='mean', seed=1, B=2000):
    x = finite(values)
    if len(x) == 0: return [float('nan'), float('nan')]
    rng = np.random.default_rng(seed); vals = np.empty(B, dtype=np.float64)
    for b in range(B):
        y = x[rng.integers(0, len(x), size=len(x))]
        vals[b] = np.mean(y) if stat == 'mean' else np.median(y)
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def clustered_event_summary(rows: List[Dict], selector, seed: int):
    rr = [r for r in rows if selector(r) and np.isfinite(float(r['cosine']))]
    if not rr: return {'N_events': 0, 'N_contexts': 0}
    by = {}
    for r in rr: by.setdefault(int(r['context']), []).append(r)
    ctx = sorted(by); rng = np.random.default_rng(seed)

    def pooled(sampled_ctx, key):
        return np.asarray([float(r[key]) for c in sampled_ctx for r in by[c] if np.isfinite(float(r[key]))], dtype=np.float64)

    def boot_stat(key, fn, B=2000):
        vals = []
        for _ in range(B):
            sc = [ctx[int(i)] for i in rng.integers(0, len(ctx), size=len(ctx))]
            x = pooled(sc, key); vals.append(fn(x) if len(x) else np.nan)
        a = finite(vals)
        return [float(np.quantile(a, .025)), float(np.quantile(a, .975))] if len(a) else [float('nan')]*2

    cosv = pooled(ctx, 'cosine'); sd = pooled(ctx, 'signed_descent'); eff = pooled(ctx, 'efficiency')
    am = pooled(ctx, 'A_M'); ae = pooled(ctx, 'A_E')
    return {
        'N_events': int(len(rr)), 'N_contexts': int(len(ctx)),
        'cosine_median': float(np.median(cosv)),
        'cosine_median_context_bootstrap_95ci': boot_stat('cosine', np.median),
        'cosine_mean': float(np.mean(cosv)),
        'cosine_mean_context_bootstrap_95ci': boot_stat('cosine', np.mean),
        'signed_descent_mean': float(np.mean(sd)),
        'signed_descent_context_bootstrap_95ci': boot_stat('signed_descent', np.mean),
        'efficiency_median': float(np.median(eff)),
        'p_cosine_negative': float(np.mean(cosv < 0.0)),
        'p_negative_context_bootstrap_95ci': boot_stat('cosine', lambda x: np.mean(x < 0.0)),
        'A_M_median': float(np.median(am)) if len(am) else float('nan'),
        'A_E_median': float(np.median(ae)) if len(ae) else float('nan'),
    }


def context_summary(rows: List[Dict], seed: int):
    if not rows: return {}
    cosv = finite(float(r['context_cosine']) for r in rows)
    sd = finite(float(r['context_signed_descent']) for r in rows)
    eff = finite(float(r['context_efficiency']) for r in rows)
    ident = finite(float(r['w2_native_update_identity_max_abs']) for r in rows)
    return {
        'N_contexts': int(len(rows)), 'cosine_median': float(np.median(cosv)),
        'cosine_median_95ci': ci_boot(cosv, 'median', seed),
        'cosine_mean': float(np.mean(cosv)), 'cosine_mean_95ci': ci_boot(cosv, 'mean', seed + 1),
        'p_cosine_negative': float(np.mean(cosv < 0.0)),
        'signed_descent_mean': float(np.mean(sd)), 'efficiency_median': float(np.median(eff)),
        'native_update_identity_max_abs_max': float(np.max(ident)),
    }


def participation_from_singular(s):
    e = np.asarray(s, dtype=np.float64) ** 2
    if not np.any(e > 0): return 0.0
    return float((e.sum() ** 2) / np.sum(e * e))


def svd_summary(X: np.ndarray, max_values=32):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or min(X.shape) == 0: return {}
    s_unc = np.linalg.svd(X, compute_uv=False, full_matrices=False)
    Xc = X - np.mean(X, axis=0, keepdims=True)
    s_ctr = np.linalg.svd(Xc, compute_uv=False, full_matrices=False)
    return {
        'shape': list(X.shape),
        'effective_dimension_uncentered': participation_from_singular(s_unc),
        'effective_dimension_centered': participation_from_singular(s_ctr),
        'singular_values_uncentered': s_unc[:max_values].tolist(),
        'singular_values_centered': s_ctr[:max_values].tolist(),
        'energy_top1_fraction_uncentered': float(s_unc[0]**2 / np.sum(s_unc**2)) if len(s_unc) else float('nan'),
        'energy_top4_fraction_uncentered': float(np.sum(s_unc[:4]**2) / np.sum(s_unc**2)) if len(s_unc) else float('nan'),
    }


def temporal_mode_overlap(A: np.ndarray, B: np.ndarray, rank=4):
    A = A - A.mean(0, keepdims=True); B = B - B.mean(0, keepdims=True)
    Ua, _, _ = np.linalg.svd(A, full_matrices=False); Ub, _, _ = np.linalg.svd(B, full_matrices=False)
    r = min(rank, Ua.shape[1], Ub.shape[1])
    if r == 0: return []
    return np.linalg.svd(Ua[:, :r].T @ Ub[:, :r], compute_uv=False).tolist()


def lag_corr(x, lag):
    x = np.asarray(x, dtype=np.float64); x = x - np.mean(x)
    if len(x) <= lag or np.std(x) <= 1e-30: return float('nan')
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def d2_summary(npz_path: Path):
    z = np.load(npz_path)
    blocks = {k: np.asarray(z[f'{k}__sketch'], dtype=np.float64) for k in TRACK}
    norms = {k: np.asarray(z[f'{k}__norm'], dtype=np.float64) for k in TRACK}
    raw = np.concatenate([blocks[k] for k in TRACK], axis=1)
    normalized_blocks = []
    for k in TRACK:
        scale = math.sqrt(float(np.mean(np.sum(blocks[k]**2, axis=1)))) + 1e-30
        normalized_blocks.append(blocks[k] / scale)
    eq = np.concatenate(normalized_blocks, axis=1)
    group = {k: svd_summary(blocks[k]) for k in TRACK}
    speed = {k: {'mean_step_norm': float(np.mean(norms[k])), 'median_step_norm': float(np.median(norms[k])),
                 'p95_step_norm': float(np.quantile(norms[k], .95))} for k in TRACK}
    overlaps = {}
    for i, aa in enumerate(TRACK):
        for bb in TRACK[i+1:]: overlaps[f'{aa}__{bb}'] = temporal_mode_overlap(blocks[aa], blocks[bb], rank=4)
    Xc = raw - raw.mean(0, keepdims=True); U, s, _ = np.linalg.svd(Xc, full_matrices=False)
    pc1 = U[:, 0] * s[0] if len(s) else np.zeros(len(raw))
    ac = {str(l): lag_corr(pc1, l) for l in (1, 2, 4, 8, 16, 32, 64) if l < len(pc1)}
    return {
        'group_svd': group, 'joint_raw': svd_summary(raw), 'joint_equal_group': svd_summary(eq),
        'plastic_speed': speed, 'cross_group_temporal_subspace_cosines_rank4': overlaps,
        'exploratory_joint_pc1_autocorrelation': ac,
        'interpretation_boundary': ('CountSketch/SVD establishes observed effective dimensionality only. '
                                    'Low rank or slow autocorrelation is not evidence of an invariant spectral submanifold.'),
    }


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--in-dir', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    event_classes = {
        'all': lambda r: True, 'false61': lambda r: bool(r['false61']),
        'nonfalse61': lambda r: not bool(r['false61']), 'correct': lambda r: bool(r['correct']),
        'incorrect': lambda r: not bool(r['correct']), 'gt61': lambda r: bool(r['gt61']),
        'gt_non61': lambda r: not bool(r['gt61']),
    }
    result = {'stage': 'GATE_D_OFFLINE_MATHEMATICAL_ANALYSIS', 'arms': {}}
    for ai, arm in enumerate(ARMS):
        ep = a.in_dir / f'{arm}_D1_EVENTS.csv'; cp = a.in_dir / f'{arm}_D1_CONTEXTS.csv'; sp = a.in_dir / f'{arm}_D2_SKETCHES.npz'
        if not ep.exists() or not cp.exists() or not sp.exists(): continue
        er = read_csv(ep); cr = read_csv(cp)
        evsum = {name: clustered_event_summary(er, fn, seed=9021 + ai*100 + j) for j, (name, fn) in enumerate(event_classes.items())}
        result['arms'][arm] = {'context_level_W2_alignment': context_summary(cr, 12001 + ai),
                               'event_level_W2_alignment': evsum, 'update_spectrum': d2_summary(sp)}
    result['decision_logic'] = {
        'credit_failure': 'context/event W2 alignment becomes negative, especially in false61 or corrupted contexts',
        'temporal_regulation_candidate': 'alignment remains positive but path/net tortuosity and temporal oscillation are large',
        'coadaptation_candidate': 'local alignment is positive and stable but selected pair interventions show strong non-additivity',
        'slow_manifold_candidate_only_if': 'low effective dimension + reproducible spectral gap + transverse return/invariance evidence',
    }
    (a.out / 'GATE_D_ANALYSIS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
