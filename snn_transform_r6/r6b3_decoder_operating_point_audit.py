#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R6B.3 inference-only MaskGIT decoder operating-point audit.

This audit freezes a completed full-dynamical checkpoint and changes NO neural
weights, MSDL states, reciprocal states, tokenizer parameters, or input
representation.  It only sweeps scheduled-parallel decoding hyperparameters:

  num_iter in {4,6,8,10,12,16}
  choice_temperature in {3.0,4.5,6.0}

For every operating point it generates the same number of unconditional samples
from the same checkpoint and generation seed, then measures:
  * generated token marginal vs held-out real SNN-VQ token marginal (JS/TV/KL),
  * token perplexity / top-1 fraction / active codes,
  * external post-hoc 0/1/2 classifier class coverage and confidence,
  * MaskGIT trace audit pass/fail.

Labels remain external diagnostics only.  No label enters generation.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

import r6b3_context_epoch_training as trainer
from r6b3_snn_encoder_maskgit_msdl import (
    generation_tokens_maskgit,
    posthoc_generation_metric,
    save_grid_unlabeled,
)


def probability_from_ids(ids, vocab_size):
    a = np.asarray(ids, dtype=np.int64).reshape(-1)
    counts = np.bincount(a, minlength=int(vocab_size)).astype(np.float64)
    total = float(counts.sum())
    p = counts / max(total, 1.0)
    return counts, p


def entropy_nats(p):
    p = np.asarray(p, dtype=np.float64)
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def js_nats(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    out = 0.0
    mp = p > 0
    mq = q > 0
    if np.any(mp):
        out += 0.5 * float(np.sum(p[mp] * np.log(p[mp] / m[mp])))
    if np.any(mq):
        out += 0.5 * float(np.sum(q[mq] * np.log(q[mq] / m[mq])))
    return out


def smoothed_kl_nats(p, q, eps=1e-9):
    p = np.asarray(p, dtype=np.float64) + float(eps)
    q = np.asarray(q, dtype=np.float64) + float(eps)
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def distribution_metrics(ids, real_p, vocab_size):
    counts, p = probability_from_ids(ids, vocab_size)
    h = entropy_nats(p)
    return {
        'active_codes': int(np.sum(counts > 0)),
        'perplexity': float(np.exp(h)),
        'top1_fraction': float(p.max()),
        'entropy_nats': h,
        'js_to_real_nats': js_nats(p, real_p),
        'tv_to_real': float(0.5 * np.abs(p - real_p).sum()),
        'kl_generated_to_real_smoothed_nats': smoothed_kl_nats(p, real_p),
        'kl_real_to_generated_smoothed_nats': smoothed_kl_nats(real_p, p),
        'counts': counts.astype(np.int64).tolist(),
    }


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--rounds', type=str, default='4,6,8,10,12,16')
    ap.add_argument('--temperatures', type=str, default='3.0,4.5,6.0')
    ap.add_argument('--samples', type=int, default=128)
    ap.add_argument('--grid-samples', type=int, default=36,
                    help='number of generated images saved in each visual grid')
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

    # Structural decoder self-test before the sweep.
    controller_audit = trainer.controller_self_test()
    if not controller_audit['pass']:
        raise RuntimeError(f'MaskGIT controller self-test failed: {controller_audit}')

    # Restore the exact 2-epoch system.  The returned train RNG/schedule are not
    # consumed because this audit performs no learning.
    system, checkpoint_meta, _train_rng, schedule = trainer.load_full_checkpoint(a.checkpoint)
    if abs(float(checkpoint_meta.get('context_epoch_equivalent', -1.0)) - 2.0) > 1e-9:
        raise ValueError(f'expected a 2.0-epoch checkpoint, got {checkpoint_meta.get("context_epoch_equivalent")}')
    if int(schedule.get('total_contexts', -1)) != int(checkpoint_meta['n_train_contexts_per_epoch']) * 2:
        raise ValueError('checkpoint schedule does not correspond to exactly two full context epochs')
    frozen_reference = pickle.loads(pickle.dumps(system, pickle.HIGHEST_PROTOCOL))
    cfg = system.c
    model = trainer.model_dict_from_system(system)

    # Recreate exactly the frozen tokenizer/input representation and held-out
    # validation split used by formal context-epoch training.
    xall, yall, xte0, yte0 = trainer.core.load(a.data)
    x012, y012 = trainer.core.subset012(xall, yall)
    xtr, ytr, xva, yva = trainer.core.split(x012, y012, .1, 7)
    tokenizer, tok_meta = trainer.base.load_frozen_tokenizer(a.tokenizer, device)
    va, sp_va = trainer.base.encode_snn(tokenizer, xva, device, a.encode_batch)
    tok, pos, input_audit = trainer.build_input_embeddings(tokenizer, cfg.D, grid=7)
    rep_audit = trainer.representation_self_test(tokenizer, cfg.D, grid=7)
    if not rep_audit['pass']:
        raise RuntimeError('input representation self-test failed')

    # Real token reference is held-out validation data, label-free at this stage.
    real_counts, real_p = probability_from_ids(va, cfg.V)
    real_stats = {
        'split': 'held-out validation 0/1/2 subset; labels not used in token statistics',
        'n_images': int(len(va)),
        'n_tokens': int(np.asarray(va).size),
        'active_codes': int(np.sum(real_counts > 0)),
        'perplexity': float(np.exp(entropy_nats(real_p))),
        'top1_fraction': float(real_p.max()),
        'entropy_nats': entropy_nats(real_p),
        'counts': real_counts.astype(np.int64).tolist(),
        'probabilities': real_p.tolist(),
    }

    # External image classifier is diagnostic only and never feeds generation.
    clf = LogisticRegression(max_iter=250, solver='lbfgs', random_state=7).fit(
        xtr.reshape(len(xtr), -1), ytr)

    generation_seed = int(a.seed + 2000)  # matches formal trainer's final-eval seed
    rows = []
    details = {}
    for temp in temperatures:
        for n_rounds in rounds:
            key = f'r{n_rounds}_t{temp:g}'
            print('AUDIT_SETTING_START', json.dumps({
                'key': key, 'rounds': n_rounds, 'choice_temperature': temp,
                'samples': a.samples, 'generation_seed': generation_seed,
            }), flush=True)

            ids, traces, audits = generation_tokens_maskgit(
                copy_model_dict(model), cfg, a.samples, tok, pos, generation_seed,
                num_iter=n_rounds, choice_temperature=temp)
            imgs = trainer.base.decode_snn(tokenizer, ids, device)
            posthoc = posthoc_generation_metric(clf, imgs)
            dist = distribution_metrics(ids, real_p, cfg.V)
            all_audits_pass = bool(all(x['pass'] for x in audits))
            if not all_audits_pass:
                raise RuntimeError(f'MaskGIT trace audit failed at {key}')

            # Save only a compact visual subset; all 128 samples still enter metrics.
            n_grid = min(int(a.grid_samples), len(imgs))
            save_grid_unlabeled(imgs[:n_grid], a.out / f'{key}_grid.png')

            row = {
                'key': key,
                'rounds': int(n_rounds),
                'choice_temperature': float(temp),
                'samples': int(a.samples),
                'active_codes': dist['active_codes'],
                'perplexity': dist['perplexity'],
                'top1_fraction': dist['top1_fraction'],
                'js_to_real_nats': dist['js_to_real_nats'],
                'tv_to_real': dist['tv_to_real'],
                'kl_generated_to_real_smoothed_nats': dist['kl_generated_to_real_smoothed_nats'],
                'kl_real_to_generated_smoothed_nats': dist['kl_real_to_generated_smoothed_nats'],
                'posthoc_class_entropy_nats': float(posthoc['posthoc_pred_entropy_nats']),
                'posthoc_classes_present': int(posthoc['posthoc_pred_classes_present']),
                'posthoc_mean_classifier_confidence': float(posthoc['posthoc_mean_classifier_confidence']),
                'posthoc_pred_counts_012': posthoc['posthoc_pred_counts_012'],
                'all_trace_audits_pass': all_audits_pass,
            }
            rows.append(row)
            details[key] = {
                'summary': row,
                'token_distribution': dist,
                'posthoc': posthoc,
                'first_four_traces': traces[:4],
                'first_four_trace_audits': audits[:4],
            }
            print('AUDIT_SETTING_RESULT', json.dumps(row), flush=True)

    # No arbitrary composite score: expose independent rankings.
    rank_js = sorted(rows, key=lambda x: (x['js_to_real_nats'], x['tv_to_real']))
    rank_class_entropy = sorted(rows, key=lambda x: (-x['posthoc_class_entropy_nats'], x['js_to_real_nats']))
    rank_confidence = sorted(rows, key=lambda x: (-x['posthoc_mean_classifier_confidence'], x['js_to_real_nats']))

    # Strong audit that inference did not mutate the live dynamical system.
    unchanged = trainer.compare_systems(frozen_reference, system)
    if not unchanged['pass']:
        raise RuntimeError(f'inference-only audit mutated system state: {unchanged}')

    report = {
        'stage': 'R6B.3 two-epoch inference-only decoder operating-point audit',
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
        'real_token_reference': real_stats,
        'settings': details,
        'ranking_by_token_js_to_real': [x['key'] for x in rank_js],
        'ranking_by_posthoc_class_entropy': [x['key'] for x in rank_class_entropy],
        'ranking_by_posthoc_classifier_confidence': [x['key'] for x in rank_confidence],
        'best_by_token_js': rank_js[0],
        'best_by_class_entropy': rank_class_entropy[0],
    }
    (a.out / 'DECODER_AUDIT_RESULTS.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    csv_fields = [
        'key','rounds','choice_temperature','samples','active_codes','perplexity','top1_fraction',
        'js_to_real_nats','tv_to_real','kl_generated_to_real_smoothed_nats',
        'kl_real_to_generated_smoothed_nats','posthoc_class_entropy_nats',
        'posthoc_classes_present','posthoc_mean_classifier_confidence','posthoc_pred_counts_012',
        'all_trace_audits_pass',
    ]
    with (a.out / 'DECODER_AUDIT_TABLE.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for row in rows:
            rr = dict(row)
            rr['posthoc_pred_counts_012'] = json.dumps(rr['posthoc_pred_counts_012'])
            w.writerow(rr)

    concise = {
        'real_token_reference': {k:real_stats[k] for k in ('n_images','active_codes','perplexity','top1_fraction')},
        'best_by_token_js': rank_js[0],
        'best_by_class_entropy': rank_class_entropy[0],
        'default_r8_t4.5': details.get('r8_t4.5',{}).get('summary'),
        'system_state_unchanged': unchanged['pass'],
    }
    print('DECODER_AUDIT_DONE', json.dumps(concise, indent=2), flush=True)


if __name__ == '__main__':
    main()
