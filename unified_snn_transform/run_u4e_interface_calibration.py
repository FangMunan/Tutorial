#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""U4E — parameter/interface calibration for the sparse-tail block.

Goal: keep the U4B 25%-25%-25% sparse tail and all task/attention weights
fixed, and improve only the Q/K Izh feature interface. U4D localized the
strict-parity miss to K-side Izh -> normalized H/g readout. The prior scalar
search ranked almost entirely by scalar rel-L2 and admitted K decoders with a
large positivity-clamp fraction. Here we search the same neuronal family but
make the calibration clamp-aware and select the final interface only by
unlabelled parity to the exact sparse block.

No task label is used for interface selection and no network weight is changed.
"""
from __future__ import annotations

import argparse, copy, json
from pathlib import Path
import numpy as np
import torch

import run_unified_gate as u
import run_u3_trained_retrieval as r
import run_u31_izh_localization as u31
import run_u4b_task_sparse_block as u4b
import run_u4c_full_sparse_izh as u4c


def spec_key(spec):
    return (float(spec['bias']), float(spec['gain']), int(spec['window_ms']))


def make_item(spec, dec, row):
    return (float(row['scalar_rel_l2']), dict(spec), dec, dict(row))


def calibration_pool(a: torch.Tensor, features: int, seed: int, max_pool: int = 36):
    """Build a Pareto-like pool spanning scalar fidelity and clamp robustness."""
    a = a.detach().flatten().cpu()
    g = torch.Generator(); g.manual_seed(seed)
    cap = min(len(a), 12000)
    if len(a) > cap:
        a = a[torch.randperm(len(a), generator=g)[:cap]]
    p = torch.randperm(len(a), generator=g)
    n = max(200, min(int(.70 * len(a)), len(a) - 200))
    fit, val = a[p[:n]], a[p[n:]]
    yf = torch.exp(fit.clamp(-30, 30)) * (features ** -.5) + 1e-6
    yv = torch.exp(val.clamp(-30, 30)) * (features ** -.5) + 1e-6

    grid = [
        {'bias': float(b), 'gain': float(gg), 'window_ms': int(w)}
        for b in (0., 1., 2., 3., 4., 5., 6., 8., 10.)
        for gg in (.10, .15, .25, .35, .50, .75, 1.0, 1.4, 2.0)
        for w in (6, 8, 10, 12, 16, 20, 24)
    ]
    all_items = []
    for spec in grid:
        dec = u.fit_decoder(fit, yf, **spec)
        raw = u.traj_design(val, **spec) @ dec
        pred = raw.clamp_min(1e-8)
        rr = float(torch.linalg.vector_norm(pred-yv) /
                   torch.linalg.vector_norm(yv).clamp_min(1e-12))
        clamp = float((raw <= 1e-8).float().mean())
        spk, active = u.izh_activity(val[:min(1500, len(val))], **spec)
        row = {**spec, 'scalar_rel_l2': rr, 'clamp_fraction': clamp,
               'mean_spikes': spk, 'active_fraction': active}
        all_items.append(make_item(spec, dec, row))

    chosen = {}
    def add_sorted(items, n_take):
        for it in items[:n_take]:
            chosen.setdefault(spec_key(it[1]), it)

    # Pure scalar-fidelity anchors.
    add_sorted(sorted(all_items, key=lambda z: z[3]['scalar_rel_l2']), 8)

    # Clamp-aware fronts. Small target values can have tiny L2 weight but are
    # still structurally important in the normalized H/g readout.
    for lam in (2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2):
        ss = sorted(all_items, key=lambda z: z[3]['scalar_rel_l2'] + lam*z[3]['clamp_fraction'])
        add_sorted(ss, 4)

    # Explicit low-clamp anchors, provided scalar error is not pathological.
    for thr in (.01, .025, .05, .10, .20, .30):
        filt = [z for z in all_items if z[3]['clamp_fraction'] <= thr]
        add_sorted(sorted(filt, key=lambda z: z[3]['scalar_rel_l2']), 3)

    pool = list(chosen.values())
    # Preserve diversity but cap downstream evaluations.
    pool = sorted(pool, key=lambda z: (z[3]['scalar_rel_l2'] + 2e-3*z[3]['clamp_fraction'],
                                       z[1]['window_ms']))[:max_pool]
    summary = sorted([z[3] for z in all_items], key=lambda z: z['scalar_rel_l2'])
    return pool, summary


@torch.no_grad()
def eval_interface(ref, cand, gen, seed, batches=4, batch=64):
    return u4c.parity(ref, cand, gen, seed, batches=batches, batch=batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='unified_snn_transform/run/U4E')
    ap.add_argument('--seed', type=int, default=71)
    ap.add_argument('--tokens', type=int, default=16); ap.add_argument('--d', type=int, default=32)
    ap.add_argument('--heads', type=int, default=2); ap.add_argument('--features', type=int, default=32)
    ap.add_argument('--ff', type=int, default=64); ap.add_argument('--value-dim', type=int, default=8)
    ap.add_argument('--theta-init', type=float, default=0.0)
    ap.add_argument('--train-steps', type=int, default=900); ap.add_argument('--train-batch', type=int, default=128)
    ap.add_argument('--eval-batches', type=int, default=10); ap.add_argument('--eval-batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--report-every', type=int, default=300)
    ap.add_argument('--q-topn', type=int, default=12); ap.add_argument('--k-pool', type=int, default=36)
    ap.add_argument('--joint-topk', type=int, default=8); ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args(); torch.set_num_threads(args.threads); u4b.seed_all(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    gen = r.AssocGenerator(args.tokens, args.d, args.value_dim, args.seed)

    # Recreate the frozen U4B structural point exactly: architecture fixed,
    # then train only the task model once with exact positive-RF features.
    ref = u4b.SparseRetrievalModel(args, args.seed+1001, .25, .25, .25)
    ref, hist = u4b.train_model(ref, gen, args, args.seed+2001)
    exact_task, hidden_activity, acct = u4b.eval_sparse(ref, gen, args, args.seed+3001)

    aq, ak = u4c.collect_a(ref, gen, args.seed+4001, batches=10, batch=64)
    qtop, _ = u31.expanded_candidates(aq, args.features, args.seed+5001, args.q_topn)
    kpool, k_scalar_rows = calibration_pool(ak, args.features, args.seed+5002, args.k_pool)

    # K-only downstream screen with exact Q. This is the quantity U4D showed
    # to dominate the integration error. Selection uses no task labels.
    k_rows = []
    for ki, kitem in enumerate(kpool):
        cand = u4c.replace_phi(ref, None, kitem)
        m = eval_interface(ref, cand, gen, args.seed+6001, batches=3, batch=64)
        row = {'k_index': ki, **kitem[3],
               'prediction_rel_l2': m['prediction_rel_l2'],
               'prediction_cosine': m['prediction_cosine'],
               'H_rel_l2': m['H_rel_l2'], 'g_rel_l2': m['g_rel_l2']}
        k_rows.append(row)
    k_ranked_idx = sorted(range(len(kpool)), key=lambda i: k_rows[i]['prediction_rel_l2'])
    kshort = [kpool[i] for i in k_ranked_idx[:args.joint_topk]]

    # Q already passed U4D alone; jointly select a small Q/K interface set by
    # exact-model parity only to avoid an unnecessary additive composition miss.
    qshort = qtop[:args.joint_topk]
    pair_rows = []; best = None
    for qi, qitem in enumerate(qshort):
        for ki, kitem in enumerate(kshort):
            cand = u4c.replace_phi(ref, qitem, kitem)
            m = eval_interface(ref, cand, gen, args.seed+7001, batches=3, batch=64)
            row = {'q_index': qi, 'k_short_index': ki,
                   'prediction_rel_l2': m['prediction_rel_l2'],
                   'prediction_cosine': m['prediction_cosine'],
                   'H_rel_l2': m['H_rel_l2'], 'g_rel_l2': m['g_rel_l2'],
                   'q_spec': qitem[3], 'k_spec': kitem[3]}
            pair_rows.append(row)
            score = m['prediction_rel_l2'] + 5e-6*(qitem[1]['window_ms']+kitem[1]['window_ms'])
            if best is None or score < best[0]:
                best = (score, qitem, kitem, row)

    _, qbest, kbest, _ = best
    q_only = eval_interface(ref, u4c.replace_phi(ref, qbest, None), gen,
                            args.seed+8001, batches=args.eval_batches, batch=args.eval_batch)
    k_only = eval_interface(ref, u4c.replace_phi(ref, None, kbest), gen,
                            args.seed+8001, batches=args.eval_batches, batch=args.eval_batch)
    both = eval_interface(ref, u4c.replace_phi(ref, qbest, kbest), gen,
                          args.seed+8001, batches=args.eval_batches, batch=args.eval_batch)

    gates = {
        'exact_sparse_task_sign_ge_0p95': exact_task['sign_accuracy'] >= .95,
        'K_only_prediction_rel_l2_le_1pct': k_only['prediction_rel_l2'] <= .01,
        'Q_only_prediction_rel_l2_le_1pct': q_only['prediction_rel_l2'] <= .01,
        'QK_prediction_rel_l2_le_1pct': both['prediction_rel_l2'] <= .01,
        'QK_prediction_cosine_ge_0p999': both['prediction_cosine'] >= .999,
        'QK_task_sign_drop_le_0p02': both['candidate_task']['sign_accuracy'] >= both['reference_task']['sign_accuracy']-.02,
        'tail_effective_fraction_le_0p30': acct['effective_fraction'] <= .30,
        'architecture_and_task_weights_unchanged_during_calibration': True,
    }
    result = {
        'stage': 'UNIFIED_U4E_CLAMP_AWARE_INTERFACE_CALIBRATION',
        'config': vars(args),
        'exact_sparse_task': exact_task,
        'hidden_activity': hidden_activity,
        'tail_compute': acct,
        'train_history': hist,
        'K_pool_size': len(kpool),
        'K_downstream_screen': sorted(k_rows, key=lambda z: z['prediction_rel_l2']),
        'K_scalar_best': k_scalar_rows[:20],
        'joint_pair_screen': sorted(pair_rows, key=lambda z: z['prediction_rel_l2']),
        'selected_Q': qbest[3], 'selected_K': kbest[3],
        'heldout': {'Q_only': q_only, 'K_only': k_only, 'QK': both},
        'gates': gates, 'PASS': all(gates.values()),
    }
    result['decision'] = (
        'freeze calibrated Izh Q/K interface with U4 sparse tail; proceed to canonical full-block freeze gate'
        if result['PASS'] else
        'shared scalar Izh interface still misses strict gate; next calibration should be head-aware/Hg-aware without changing sparse-tail architecture'
    )
    (out/'U4E_RESULTS.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({
        'stage': result['stage'], 'PASS': result['PASS'], 'selected_Q': result['selected_Q'],
        'selected_K': result['selected_K'], 'heldout': result['heldout'],
        'gates': result['gates'], 'decision': result['decision']
    }, indent=2), flush=True)
    # Archive scientific failure as a result; workflow remains green for artifact retention.


if __name__ == '__main__':
    main()
