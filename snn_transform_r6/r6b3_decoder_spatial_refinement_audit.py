#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R6B.3 inference-only decoder refinement with spatial token diagnostics.

The completed 2.0-context-epoch dynamical checkpoint is frozen.  This script
performs NO learning and changes NO neural/MSDL/reciprocal state.  It refines
only the scheduled-parallel MaskGIT operating point over:

  num_iter in {3,4,5,6}
  choice_temperature in {5.0,6.0,7.0,8.0}

Each setting uses the same generation seed and 512 samples by default.  In
addition to the global token marginal used in the first decoder audit, this
stage tests spatial structure directly in token space:

  1) positionwise token distribution
       JS_spatial = mean_i JS[p_real(z_i), p_gen(z_i)] over all 49 positions;
  2) ordered nearest-neighbour token-pair distributions, aggregated separately
       over horizontal and vertical 7x7-grid edges.

The report exposes independent rankings rather than an arbitrary weighted
composite score.  MNIST labels remain external post-hoc diagnostics only.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

import r6b3_context_epoch_training as trainer
import r6b3_decoder_operating_point_audit as coarse
from r6b3_snn_encoder_maskgit_msdl import (
    generation_tokens_maskgit,
    posthoc_generation_metric,
    save_grid_unlabeled,
)


def copy_model_dict(model):
    return {
        k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else copy.deepcopy(v))
        for k, v in model.items()
    }


def parse_int_list(text):
    vals = [int(x.strip()) for x in text.split(',') if x.strip()]
    if not vals or any(x < 2 for x in vals):
        raise ValueError('round list must contain integers >=2')
    return vals


def parse_float_list(text):
    vals = [float(x.strip()) for x in text.split(',') if x.strip()]
    if not vals or any(x < 0 for x in vals):
        raise ValueError('temperature list must contain non-negative values')
    return vals


def ensure_token_matrix(ids, n_positions):
    a = np.asarray(ids, dtype=np.int64)
    if a.ndim != 2 or a.shape[1] != int(n_positions):
        raise ValueError(f'expected token matrix [B,{n_positions}], got {a.shape}')
    return a


def position_probabilities(ids, vocab_size, n_positions):
    a = ensure_token_matrix(ids, n_positions)
    counts = np.zeros((n_positions, vocab_size), dtype=np.float64)
    for i in range(n_positions):
        counts[i] = np.bincount(a[:, i], minlength=vocab_size)
    denom = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    return counts, counts / denom


def positionwise_metrics(ids, real_position_p, vocab_size, n_positions):
    counts, p = position_probabilities(ids, vocab_size, n_positions)
    js = np.asarray([
        coarse.js_nats(p[i], real_position_p[i]) for i in range(n_positions)
    ], dtype=np.float64)
    tv = 0.5 * np.abs(p - real_position_p).sum(axis=1)
    kl_g2r = np.asarray([
        coarse.smoothed_kl_nats(p[i], real_position_p[i]) for i in range(n_positions)
    ], dtype=np.float64)
    kl_r2g = np.asarray([
        coarse.smoothed_kl_nats(real_position_p[i], p[i]) for i in range(n_positions)
    ], dtype=np.float64)
    order = np.argsort(-js)
    return {
        'mean_js_nats': float(js.mean()),
        'median_js_nats': float(np.median(js)),
        'max_js_nats': float(js.max()),
        'mean_tv': float(tv.mean()),
        'mean_kl_generated_to_real_smoothed_nats': float(kl_g2r.mean()),
        'mean_kl_real_to_generated_smoothed_nats': float(kl_r2g.mean()),
        'active_codes_per_position_mean': float(np.mean(np.sum(counts > 0, axis=1))),
        'js_by_position': js.tolist(),
        'tv_by_position': tv.tolist(),
        'worst_five_positions_by_js': [int(x) for x in order[:5]],
    }


def edge_index_sets(grid):
    horizontal_left = []
    horizontal_right = []
    vertical_top = []
    vertical_bottom = []
    for r in range(grid):
        for c in range(grid - 1):
            horizontal_left.append(r * grid + c)
            horizontal_right.append(r * grid + c + 1)
    for r in range(grid - 1):
        for c in range(grid):
            vertical_top.append(r * grid + c)
            vertical_bottom.append((r + 1) * grid + c)
    return (
        np.asarray(horizontal_left, dtype=np.int64),
        np.asarray(horizontal_right, dtype=np.int64),
        np.asarray(vertical_top, dtype=np.int64),
        np.asarray(vertical_bottom, dtype=np.int64),
    )


def ordered_pair_probability(ids, left_index, right_index, vocab_size, n_positions):
    a = ensure_token_matrix(ids, n_positions)
    first = a[:, left_index].reshape(-1)
    second = a[:, right_index].reshape(-1)
    packed = first * int(vocab_size) + second
    counts = np.bincount(packed, minlength=int(vocab_size) ** 2).astype(np.float64)
    p = counts / max(float(counts.sum()), 1.0)
    return counts, p


def pair_reference(ids, vocab_size, n_positions, grid):
    hl, hr, vt, vb = edge_index_sets(grid)
    hc, hp = ordered_pair_probability(ids, hl, hr, vocab_size, n_positions)
    vc, vp = ordered_pair_probability(ids, vt, vb, vocab_size, n_positions)
    return {
        'horizontal_counts': hc,
        'horizontal_p': hp,
        'vertical_counts': vc,
        'vertical_p': vp,
        'horizontal_edges_per_image': int(len(hl)),
        'vertical_edges_per_image': int(len(vt)),
    }


def pair_metrics(ids, real_pair_ref, vocab_size, n_positions, grid):
    gen = pair_reference(ids, vocab_size, n_positions, grid)
    hj = coarse.js_nats(gen['horizontal_p'], real_pair_ref['horizontal_p'])
    vj = coarse.js_nats(gen['vertical_p'], real_pair_ref['vertical_p'])
    ht = 0.5 * float(np.abs(gen['horizontal_p'] - real_pair_ref['horizontal_p']).sum())
    vt = 0.5 * float(np.abs(gen['vertical_p'] - real_pair_ref['vertical_p']).sum())
    return {
        'horizontal_js_nats': float(hj),
        'vertical_js_nats': float(vj),
        'mean_orientation_js_nats': float(0.5 * (hj + vj)),
        'horizontal_tv': ht,
        'vertical_tv': vt,
        'mean_orientation_tv': float(0.5 * (ht + vt)),
        'horizontal_active_pairs': int(np.sum(gen['horizontal_counts'] > 0)),
        'vertical_active_pairs': int(np.sum(gen['vertical_counts'] > 0)),
        'horizontal_pair_perplexity': float(np.exp(coarse.entropy_nats(gen['horizontal_p']))),
        'vertical_pair_perplexity': float(np.exp(coarse.entropy_nats(gen['vertical_p']))),
    }


def pareto_front(rows, metric_names):
    """Return keys not dominated on all listed lower-is-better metrics."""
    out = []
    for i, row in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            le = all(float(other[m]) <= float(row[m]) for m in metric_names)
            lt = any(float(other[m]) < float(row[m]) for m in metric_names)
            if le and lt:
                dominated = True
                break
        if not dominated:
            out.append(row['key'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--rounds', type=str, default='3,4,5,6')
    ap.add_argument('--temperatures', type=str, default='5.0,6.0,7.0,8.0')
    ap.add_argument('--samples', type=int, default=512)
    ap.add_argument('--grid-samples', type=int, default=36)
    ap.add_argument('--encode-batch', type=int, default=256)
    ap.add_argument('--seed', type=int, default=7311)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()

    rounds = parse_int_list(a.rounds)
    temperatures = parse_float_list(a.temperatures)
    if a.samples < 1:
        raise ValueError('--samples must be positive')
    a.out.mkdir(parents=True, exist_ok=True)

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    device = torch.device('cpu')

    controller_audit = trainer.controller_self_test()
    if not controller_audit['pass']:
        raise RuntimeError(f'MaskGIT controller self-test failed: {controller_audit}')

    system, checkpoint_meta, _train_rng, schedule = trainer.load_full_checkpoint(a.checkpoint)
    if abs(float(checkpoint_meta.get('context_epoch_equivalent', -1.0)) - 2.0) > 1e-9:
        raise ValueError('this audit requires the exact 2.0-epoch checkpoint')
    if int(schedule.get('total_contexts', -1)) != int(checkpoint_meta['n_train_contexts_per_epoch']) * 2:
        raise ValueError('checkpoint schedule does not correspond to two full context epochs')
    frozen_reference = pickle.loads(pickle.dumps(system, pickle.HIGHEST_PROTOCOL))
    cfg = system.c
    model = trainer.model_dict_from_system(system)
    if int(cfg.N) != 49:
        raise ValueError(f'expected 7x7 token grid (N=49), got N={cfg.N}')
    grid = 7

    xall, yall, xte0, yte0 = trainer.core.load(a.data)
    x012, y012 = trainer.core.subset012(xall, yall)
    xtr, ytr, xva, yva = trainer.core.split(x012, y012, .1, 7)
    tokenizer, tok_meta = trainer.base.load_frozen_tokenizer(a.tokenizer, device)
    va, sp_va = trainer.base.encode_snn(tokenizer, xva, device, a.encode_batch)
    tok, pos, input_audit = trainer.build_input_embeddings(tokenizer, cfg.D, grid=grid)
    rep_audit = trainer.representation_self_test(tokenizer, cfg.D, grid=grid)
    if not rep_audit['pass']:
        raise RuntimeError('input representation self-test failed')

    real_counts, real_p = coarse.probability_from_ids(va, cfg.V)
    real_pos_counts, real_pos_p = position_probabilities(va, cfg.V, cfg.N)
    real_pair = pair_reference(va, cfg.V, cfg.N, grid)
    real_stats = {
        'split': 'held-out validation 0/1/2 subset; labels not used in token/spatial statistics',
        'n_images': int(len(va)),
        'n_tokens': int(np.asarray(va).size),
        'global_active_codes': int(np.sum(real_counts > 0)),
        'global_perplexity': float(np.exp(coarse.entropy_nats(real_p))),
        'global_top1_fraction': float(real_p.max()),
        'position_active_codes_mean': float(np.mean(np.sum(real_pos_counts > 0, axis=1))),
        'horizontal_active_pairs': int(np.sum(real_pair['horizontal_counts'] > 0)),
        'vertical_active_pairs': int(np.sum(real_pair['vertical_counts'] > 0)),
        'horizontal_pair_perplexity': float(np.exp(coarse.entropy_nats(real_pair['horizontal_p']))),
        'vertical_pair_perplexity': float(np.exp(coarse.entropy_nats(real_pair['vertical_p']))),
        'horizontal_edges_per_image': real_pair['horizontal_edges_per_image'],
        'vertical_edges_per_image': real_pair['vertical_edges_per_image'],
    }

    clf = LogisticRegression(max_iter=250, solver='lbfgs', random_state=7).fit(
        xtr.reshape(len(xtr), -1), ytr)

    generation_seed = int(a.seed + 2000)
    rows = []
    details = {}
    for temp in temperatures:
        for n_rounds in rounds:
            key = f'r{n_rounds}_t{temp:g}'
            print('SPATIAL_AUDIT_SETTING_START', json.dumps({
                'key': key,
                'rounds': n_rounds,
                'choice_temperature': temp,
                'samples': a.samples,
                'generation_seed': generation_seed,
            }), flush=True)

            ids, traces, audits = generation_tokens_maskgit(
                copy_model_dict(model), cfg, a.samples, tok, pos, generation_seed,
                num_iter=n_rounds, choice_temperature=temp)
            if not all(x['pass'] for x in audits):
                raise RuntimeError(f'MaskGIT trace audit failed at {key}')

            imgs = trainer.base.decode_snn(tokenizer, ids, device)
            posthoc = posthoc_generation_metric(clf, imgs)
            global_dist = coarse.distribution_metrics(ids, real_p, cfg.V)
            spatial = positionwise_metrics(ids, real_pos_p, cfg.V, cfg.N)
            pairs = pair_metrics(ids, real_pair, cfg.V, cfg.N, grid)

            n_grid = min(int(a.grid_samples), len(imgs))
            save_grid_unlabeled(imgs[:n_grid], a.out / f'{key}_grid.png')

            row = {
                'key': key,
                'rounds': int(n_rounds),
                'choice_temperature': float(temp),
                'samples': int(a.samples),
                'global_js_nats': float(global_dist['js_to_real_nats']),
                'global_tv': float(global_dist['tv_to_real']),
                'global_perplexity': float(global_dist['perplexity']),
                'global_top1_fraction': float(global_dist['top1_fraction']),
                'global_active_codes': int(global_dist['active_codes']),
                'spatial_mean_js_nats': float(spatial['mean_js_nats']),
                'spatial_median_js_nats': float(spatial['median_js_nats']),
                'spatial_max_js_nats': float(spatial['max_js_nats']),
                'spatial_mean_tv': float(spatial['mean_tv']),
                'pair_horizontal_js_nats': float(pairs['horizontal_js_nats']),
                'pair_vertical_js_nats': float(pairs['vertical_js_nats']),
                'pair_mean_js_nats': float(pairs['mean_orientation_js_nats']),
                'pair_mean_tv': float(pairs['mean_orientation_tv']),
                'posthoc_class_entropy_nats': float(posthoc['posthoc_pred_entropy_nats']),
                'posthoc_classes_present': int(posthoc['posthoc_pred_classes_present']),
                'posthoc_mean_classifier_confidence': float(posthoc['posthoc_mean_classifier_confidence']),
                'posthoc_pred_counts_012': posthoc['posthoc_pred_counts_012'],
                'all_trace_audits_pass': True,
            }
            rows.append(row)
            details[key] = {
                'summary': row,
                'global_token_distribution': global_dist,
                'positionwise_token_distribution': spatial,
                'adjacent_pair_distribution': pairs,
                'posthoc': posthoc,
                'first_four_traces': traces[:4],
                'first_four_trace_audits': audits[:4],
            }
            print('SPATIAL_AUDIT_SETTING_RESULT', json.dumps(row), flush=True)

    rank_global = sorted(rows, key=lambda x: (x['global_js_nats'], x['global_tv']))
    rank_spatial = sorted(rows, key=lambda x: (x['spatial_mean_js_nats'], x['spatial_mean_tv']))
    rank_pair = sorted(rows, key=lambda x: (x['pair_mean_js_nats'], x['pair_mean_tv']))
    rank_class = sorted(rows, key=lambda x: (-x['posthoc_class_entropy_nats'], x['spatial_mean_js_nats']))
    pareto = pareto_front(rows, ('global_js_nats', 'spatial_mean_js_nats', 'pair_mean_js_nats'))

    unchanged = trainer.compare_systems(frozen_reference, system)
    if not unchanged['pass']:
        raise RuntimeError(f'inference-only audit mutated system state: {unchanged}')

    report = {
        'stage': 'R6B.3 two-epoch inference-only decoder spatial refinement audit',
        'architecture_changed': False,
        'learning_performed': False,
        'checkpoint_state_mutated': False,
        'system_state_unchanged_audit': unchanged,
        'checkpoint_meta': checkpoint_meta,
        'controller_self_test': controller_audit,
        'input_representation_audit': rep_audit,
        'input_embedding_build_audit': input_audit,
        'tokenizer_metadata': tok_meta,
        'tokenizer_validation_activity': sp_va,
        'labels': {
            'enter_neural_input': False,
            'enter_msdl_learning': False,
            'enter_generation': False,
            'used_by_external_posthoc_classifier_only': True,
        },
        'generation_seed_shared_across_all_settings': generation_seed,
        'rounds': rounds,
        'choice_temperatures': temperatures,
        'samples_per_setting': int(a.samples),
        'real_spatial_reference': real_stats,
        'metric_definitions': {
            'spatial_mean_js_nats': 'mean over 49 positions of JS[p_real(z_i),p_gen(z_i)]',
            'pair_horizontal_js_nats': 'JS of ordered token pairs aggregated over all horizontal nearest-neighbour edges',
            'pair_vertical_js_nats': 'JS of ordered token pairs aggregated over all vertical nearest-neighbour edges',
            'pair_mean_js_nats': 'mean of horizontal and vertical pair JS',
        },
        'settings': details,
        'ranking_by_global_token_js': [x['key'] for x in rank_global],
        'ranking_by_spatial_position_js': [x['key'] for x in rank_spatial],
        'ranking_by_adjacent_pair_js': [x['key'] for x in rank_pair],
        'ranking_by_posthoc_class_entropy': [x['key'] for x in rank_class],
        'pareto_front_global_spatial_pair_js': pareto,
        'best_by_global_token_js': rank_global[0],
        'best_by_spatial_position_js': rank_spatial[0],
        'best_by_adjacent_pair_js': rank_pair[0],
        'best_by_class_entropy': rank_class[0],
    }
    (a.out / 'DECODER_SPATIAL_REFINEMENT_RESULTS.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')

    fields = [
        'key','rounds','choice_temperature','samples',
        'global_js_nats','global_tv','global_perplexity','global_top1_fraction','global_active_codes',
        'spatial_mean_js_nats','spatial_median_js_nats','spatial_max_js_nats','spatial_mean_tv',
        'pair_horizontal_js_nats','pair_vertical_js_nats','pair_mean_js_nats','pair_mean_tv',
        'posthoc_class_entropy_nats','posthoc_classes_present','posthoc_mean_classifier_confidence',
        'posthoc_pred_counts_012','all_trace_audits_pass',
    ]
    with (a.out / 'DECODER_SPATIAL_REFINEMENT_TABLE.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            rr = dict(row)
            rr['posthoc_pred_counts_012'] = json.dumps(rr['posthoc_pred_counts_012'])
            w.writerow(rr)

    concise = {
        'real_spatial_reference': real_stats,
        'best_by_global_token_js': rank_global[0],
        'best_by_spatial_position_js': rank_spatial[0],
        'best_by_adjacent_pair_js': rank_pair[0],
        'best_by_class_entropy': rank_class[0],
        'pareto_front_global_spatial_pair_js': pareto,
        'r4_t6': details.get('r4_t6', {}).get('summary'),
        'r6_t6': details.get('r6_t6', {}).get('summary'),
        'system_state_unchanged': unchanged['pass'],
    }
    print('DECODER_SPATIAL_REFINEMENT_DONE', json.dumps(concise, indent=2), flush=True)


if __name__ == '__main__':
    main()
