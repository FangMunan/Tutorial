#!/usr/bin/env python3
"""R6B.0: endogenous context-salience diagnostic for the MaskGIT image route.

Scientific question
-------------------
Can the *forward dynamical state itself*, before any target/mismatch is injected,
separate informative/non-dominant masked targets from the dominant background
class using only visible context?  This is the prerequisite for replacing the
explicit R6A.2 class-balancing diagnostic with an endogenous SNN/activity gate.

This stage is diagnostic only:
  * no learning and no parameter update;
  * no target token is used to construct activity;
  * the true target is used only after the forward pass to group observations;
  * spatial-shuffle controls preserve the exact mask and per-image visible-token
    multiset while destroying spatial context;
  * all-MASK is a no-visible-context control;
  * intact, R6A-damaged, and R6A.2-CB-recovered states are probed.

Important boundary
------------------
The R6A image bridge uses the H/g Transformer-equivalent forward model and
forward-perturbation recovery, not the final P3 Izhikevich E/M plasticity core.
Therefore this experiment tests whether a usable *pre-mismatch activity signal*
exists in the image forward path.  It does not yet prove spike-gated P3 MSDL.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'snn_transform_e29'))
sys.path.insert(0, str(ROOT / 'snn_transform_r4'))
sys.path.insert(0, str(ROOT / 'snn_transform_r5'))
sys.path.insert(0, str(ROOT / 'snn_transform_r6'))

import maskgit_b0_mnist012 as b0
import e29_forward_equivalence as e29
import r4_single_module_recovery as r4
import r5c_two_block_recovery as r5c
import r6a_four_block_recovery as r6a


PRIMARY_METRICS = ('read_rms', 'ffn_drive_rms', 'delta_rms')
ALL_METRICS = PRIMARY_METRICS + ('q_rms', 'k_rms', 'v_rms')
STATES = ('intact', 'damaged', 'cb_recovered')
CONTEXTS = ('real', 'shuffled')


def rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x.float().square(), dim=-1) + 1e-30)


@torch.no_grad()
def forward_activity(model, inp: torch.Tensor, labels: torch.Tensor):
    """Return per-position activity proxies without using targets/mismatch."""
    z = model.tok(inp) + model.pos + model.cls(labels)[:, None]
    out = []
    for block in model.blocks:
        x0 = z
        y2, cache = r5c.injected_block(block, x0, cache=True)
        q = block.attn.WQ(x0)
        k = block.attn.WK(x0)
        v = block.attn.WV(x0)
        ffn_drive = block.f2(cache['h'])
        out.append({
            'read_rms': rms(cache['read']),
            'ffn_drive_rms': rms(ffn_drive),
            'delta_rms': rms(y2 - x0),
            'q_rms': rms(q),
            'k_rms': rms(k),
            'v_rms': rms(v),
        })
        z = y2
    return out


def shuffle_visible(inp: torch.Tensor, mask: torch.Tensor, seed: int) -> torch.Tensor:
    """Permute only visible token values within each image; MASK positions stay fixed."""
    out = inp.clone()
    g = torch.Generator(device='cpu')
    g.manual_seed(int(seed))
    for b in range(len(out)):
        vis = torch.nonzero(~mask[b], as_tuple=False).flatten()
        if len(vis) > 1:
            perm = vis[torch.randperm(len(vis), generator=g)]
            out[b, vis] = inp[b, perm]
    return out


def average_ranks(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float('nan')
    rx, ry = average_ranks(x), average_ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def auc_binary(score: np.ndarray, positive: np.ndarray) -> float:
    """Tie-corrected Mann-Whitney AUC; high score predicts positive=True."""
    score = np.asarray(score, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    n1 = int(positive.sum()); n0 = int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    ranks = average_ranks(score)
    u1 = float(ranks[positive].sum() - n1 * (n1 + 1) / 2.0)
    return u1 / (n1 * n0)


def activity_stats(values: np.ndarray, targets: np.ndarray, dominant: int,
                   train_counts: np.ndarray, min_token_obs: int = 20) -> dict:
    values = np.asarray(values, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    pos = targets != int(dominant)
    a = values[pos]; b = values[~pos]
    ma = float(a.mean()) if len(a) else float('nan')
    mb = float(b.mean()) if len(b) else float('nan')
    va = float(a.var(ddof=1)) if len(a) > 1 else 0.0
    vb = float(b.var(ddof=1)) if len(b) > 1 else 0.0
    denom = math.sqrt(max(0.5 * (va + vb), 1e-30))
    d = (ma - mb) / denom

    s1 = float(values.sum())
    s2 = float(np.square(values).sum())
    participation = (s1 * s1) / (len(values) * s2 + 1e-30)
    n_top = max(1, int(math.ceil(0.20 * len(values))))
    top20 = float(np.partition(values, len(values) - n_top)[-n_top:].sum() / (s1 + 1e-30))

    token_n = np.bincount(targets, minlength=len(train_counts))
    token_sum = np.bincount(targets, weights=values, minlength=len(train_counts))
    good = token_n >= int(min_token_obs)
    token_mean = token_sum[good] / np.maximum(token_n[good], 1)
    freq = np.asarray(train_counts, dtype=np.float64)[good]
    freq_rho = spearman(np.log1p(freq), token_mean)

    return {
        'n': int(len(values)),
        'n_non_dominant': int(pos.sum()),
        'mean_non_dominant': ma,
        'mean_dominant': mb,
        'mean_ratio_non_dom_over_dom': ma / max(mb, 1e-30),
        'cohen_d_non_dom_minus_dom': float(d),
        'auc_non_dom_from_activity': float(auc_binary(values, pos)),
        'activity_participation_fraction': float(participation),
        'top20_activity_mass_fraction': top20,
        'token_frequency_spearman': freq_rho,
        'tokens_in_frequency_correlation': int(good.sum()),
    }


def build_model(checkpoint: Path, features: int, rf_kind: str, rf_seed: int):
    exact = b0.Model(128, 4, 4, 512, .05)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    exact.load_state_dict(ck['model'])
    exact.eval()
    model = e29.HgB0Model(features, rf_kind, rf_seed)
    e29.transfer_b0(exact, model)
    model.eval()
    r4.freeze_all(model)
    return model


def apply_state(model, state: str, recovered_checkpoint: Path):
    if state == 'intact':
        return
    if state == 'damaged':
        for bi in r6a.BLOCKS:
            r5c.damage_one(model, bi)
        return
    if state == 'cb_recovered':
        ck = torch.load(recovered_checkpoint, map_location='cpu', weights_only=False)
        src = ck['params']
        dst = r6a.target_params(model)
        if set(src) != set(dst):
            missing = sorted(set(dst) - set(src)); extra = sorted(set(src) - set(dst))
            raise RuntimeError(f'R6A2 checkpoint parameter mismatch missing={missing} extra={extra}')
        for name, p in dst.items():
            p.copy_(src[name])
        return
    raise ValueError(state)


def ce50(model, te: np.ndarray, yte: np.ndarray) -> float:
    return float(r4.eval_mask_ratios(model, te, yte, (.5,), batch=256)['0.5']['ce'])


def probe_state(model, state: str, te: np.ndarray, yte: np.ndarray,
                train_counts: np.ndarray, dominant: int, ratios: tuple[float, ...],
                batch: int, limit: int, seed: int):
    te = te[:limit]; yte = yte[:limit]
    T = torch.from_numpy(te).long(); Y = torch.from_numpy(yte).long()
    rows = []
    fullmask_identity_max = 0.0

    for ratio in ratios:
        store = {
            ctx: {(bi, metric): {'v': [], 't': []}
                  for bi in range(4) for metric in ALL_METRICS}
            for ctx in CONTEXTS
        }
        for i in range(0, len(T), batch):
            t = T[i:i+batch]
            y = Y[i:i+batch]
            mask_seed = seed + int(round(1000 * ratio)) * 10000 + i
            inp, mask = b0.fixed_mask_batch(t, ratio, seed=mask_seed)
            shuf = shuffle_visible(inp, mask, seed=mask_seed + 777777)
            if ratio >= 1.0 - 1e-12:
                fullmask_identity_max = max(fullmask_identity_max, float((inp != shuf).float().max()))
            targets = t[mask].cpu().numpy().astype(np.int64)
            for ctx, x in (('real', inp), ('shuffled', shuf)):
                acts = forward_activity(model, x, y)
                for bi in range(4):
                    for metric in ALL_METRICS:
                        vals = acts[bi][metric][mask].cpu().numpy().astype(np.float32)
                        store[ctx][(bi, metric)]['v'].append(vals)
                        store[ctx][(bi, metric)]['t'].append(targets)

        stats = {}
        for ctx in CONTEXTS:
            stats[ctx] = {}
            for bi in range(4):
                stats[ctx][bi] = {}
                for metric in ALL_METRICS:
                    rec = store[ctx][(bi, metric)]
                    vals = np.concatenate(rec['v'])
                    targets = np.concatenate(rec['t'])
                    stats[ctx][bi][metric] = activity_stats(vals, targets, dominant, train_counts)

        for bi in range(4):
            for metric in ALL_METRICS:
                rr = stats['real'][bi][metric]
                ss = stats['shuffled'][bi][metric]
                rows.append({
                    'state': state,
                    'mask_ratio': ratio,
                    'block': bi,
                    'metric': metric,
                    **{f'real_{k}': v for k, v in rr.items()},
                    **{f'shuffled_{k}': v for k, v in ss.items()},
                    'context_auc_gain': rr['auc_non_dom_from_activity'] - ss['auc_non_dom_from_activity'],
                    'context_cohen_d_gain': rr['cohen_d_non_dom_minus_dom'] - ss['cohen_d_non_dom_minus_dom'],
                    'context_mean_ratio_gain': rr['mean_ratio_non_dom_over_dom'] - ss['mean_ratio_non_dom_over_dom'],
                    'context_frequency_anticorr_gain': ss['token_frequency_spearman'] - rr['token_frequency_spearman'],
                })
    return rows, fullmask_identity_max


def state_gate(rows: list[dict], state: str) -> dict:
    primary = [r for r in rows
               if r['state'] == state and r['metric'] in PRIMARY_METRICS
               and r['mask_ratio'] in (0.25, 0.5)]
    gains = np.asarray([r['context_auc_gain'] for r in primary], dtype=np.float64)
    dgains = np.asarray([r['context_cohen_d_gain'] for r in primary], dtype=np.float64)
    positive = int(np.sum(gains >= 0.01))
    return {
        'state': state,
        'primary_cells': int(len(primary)),
        'median_context_auc_gain_25_50': float(np.nanmedian(gains)),
        'mean_context_auc_gain_25_50': float(np.nanmean(gains)),
        'cells_auc_gain_ge_0p01': positive,
        'median_context_cohen_d_gain_25_50': float(np.nanmedian(dgains)),
        'supports_context_salience': bool(np.nanmedian(gains) >= 0.02 and positive >= 12),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--codebook', type=Path, required=True)
    ap.add_argument('--recovered-checkpoint', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--features', type=int, default=256)
    ap.add_argument('--rf-kind', default='p3norm')
    ap.add_argument('--rf-seed', type=int, default=29257)
    ap.add_argument('--ratios', default='0.25,0.5,0.9,1.0')
    ap.add_argument('--probe-limit', type=int, default=1536)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--seed', type=int, default=6101)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads); torch.set_grad_enabled(False)
    a.out.mkdir(parents=True, exist_ok=True)

    xall, yall, xte0, yte0 = b0.load(a.data)
    x012, y012 = b0.subset012(xall, yall)
    xte, yte = b0.subset012(xte0, yte0)
    xtr, ytr, _, _ = b0.split(x012, y012, .1, 7)
    cb = np.load(a.codebook)
    tr = b0.encode(xtr, cb)
    te = b0.encode(xte, cb)
    counts = np.bincount(tr.reshape(-1), minlength=b0.K).astype(np.int64)
    dominant = int(np.argmax(counts))
    ratios = tuple(float(x) for x in a.ratios.split(','))

    rec_ck = torch.load(a.recovered_checkpoint, map_location='cpu', weights_only=False)
    rec_diag = rec_ck.get('metrics', {}).get('r6a2_repair', {})
    if rec_diag:
        expected_dom = int(rec_diag.get('dominant_token_id', dominant))
        if expected_dom != dominant:
            raise RuntimeError(f'dominant token mismatch current={dominant} artifact={expected_dom}')

    all_rows = []
    ce_audit = {}
    fullmask_checks = {}
    expected_ce = {
        'intact': 1.0376534208934725,
        'damaged': 1.528645943236697,
        'cb_recovered': 1.3307809699403699,
    }
    for si, state in enumerate(STATES):
        model = build_model(a.checkpoint, a.features, a.rf_kind, a.rf_seed)
        apply_state(model, state, a.recovered_checkpoint)
        ce = ce50(model, te, yte)
        ce_audit[state] = {
            'measured_ce50': ce,
            'r6a2_reference_ce50': expected_ce[state],
            'abs_error': abs(ce - expected_ce[state]),
        }
        if abs(ce - expected_ce[state]) > 2e-5:
            raise RuntimeError(f'{state} reconstruction mismatch: {ce} vs {expected_ce[state]}')
        rows, fm = probe_state(
            model, state, te, yte, counts, dominant, ratios,
            a.batch, min(a.probe_limit, len(te)), a.seed + si * 1000000
        )
        all_rows.extend(rows); fullmask_checks[state] = fm
        print(json.dumps({'state': state, 'ce50': ce, 'rows': len(rows)}), flush=True)

    gates = {s: state_gate(all_rows, s) for s in STATES}
    fullmask_pass = all(v == 0.0 for v in fullmask_checks.values())
    conclusion = {
        'damaged_has_usable_endogenous_context_salience': gates['damaged']['supports_context_salience'],
        'intact_has_endogenous_context_salience': gates['intact']['supports_context_salience'],
        'cb_recovered_has_endogenous_context_salience': gates['cb_recovered']['supports_context_salience'],
        'fullmask_shuffle_identity': fullmask_pass,
    }
    if conclusion['damaged_has_usable_endogenous_context_salience']:
        next_step = 'R6B.1: test target-free activity-gated plasticity against RAW and CB positive control.'
    elif conclusion['intact_has_endogenous_context_salience'] or conclusion['cb_recovered_has_endogenous_context_salience']:
        next_step = 'R6B.1-bootstrap: salience exists only after structure is present; test a bounded activity/homeostatic bootstrap before replacing CB.'
    else:
        next_step = 'Do not add an activity gate yet; identify a different target-free salience variable or reintroduce exact P3 E/M dynamics first.'

    out = {
        'stage': 'R6B.0 endogenous context-salience diagnostic',
        'learning_enabled': False,
        'target_used_to_construct_activity': False,
        'target_usage': 'post-hoc grouping/statistics only',
        'forward_system': 'R6A H/g Transformer-equivalent image bridge',
        'scientific_boundary': 'diagnostic of pre-mismatch forward activity; not yet proof of exact P3 Izhikevich E/M spike-gated plasticity',
        'probe_limit': min(a.probe_limit, len(te)),
        'mask_ratios': list(ratios),
        'dominant_token_id': dominant,
        'dominant_train_fraction': float(counts[dominant] / counts.sum()),
        'token_counts': counts.tolist(),
        'ce50_reconstruction_audit': ce_audit,
        'fullmask_shuffle_difference_max': fullmask_checks,
        'state_gates': gates,
        'conclusion': conclusion,
        'next_step': next_step,
        'primary_metrics': list(PRIMARY_METRICS),
        'gate_definition': {
            'ratios': [0.25, 0.5],
            'cells': '4 blocks x 3 primary metrics x 2 ratios = 24',
            'criterion': 'median context AUC gain >= 0.02 and at least 12/24 cells have AUC gain >= 0.01',
            'context_auc_gain': 'AUC(real visible context) - AUC(spatially shuffled visible context)',
        },
    }
    (a.out / 'summary.json').write_text(json.dumps(out, indent=2))
    if all_rows:
        with (a.out / 'salience_cells.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)
    print('R6B0_FINAL ' + json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
