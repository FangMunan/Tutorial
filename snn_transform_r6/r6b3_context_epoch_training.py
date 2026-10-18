#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R6B.3 formal context-epoch trainer.

This runner intentionally keeps the frozen R6B.3 model architecture unchanged.
It changes only training infrastructure:

1. exposure is counted in shuffled MaskGIT contexts/epochs rather than target events;
2. checkpoints serialize the complete MaskGITP3System Python state so a resumed run
   continues the same persistent dynamical learner rather than restoring weights only;
3. each checkpoint can be evaluated at fixed mask ratios and decoded with 8/16
   scheduled-parallel MaskGIT refinement rounds.

No P1-G neuronal Q/K reintegration is performed here.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'snn_transform_p3_image'))
sys.path.insert(0, str(HERE))

import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
from maskgit_input_representation import build_input_embeddings, representation_self_test
from maskgit_msdl_training import sample_mask_set
from maskgit_msdl_training_unconditional import (
    MaskGITP3System,
    build_maskgit_scaffold_contexts_unconditional,
    eval_masked_unconditional,
    masking_self_test,
)
from maskgit_scheduled_parallel import controller_self_test
from r6b3_snn_encoder_maskgit_msdl import (
    generation_tokens_maskgit,
    posthoc_generation_metric,
    save_grid_unlabeled,
)


def _system_state_keys(system):
    """Return a compact inventory of persistent state carried by the learner."""
    inventory = {}
    for name, value in vars(system).items():
        if isinstance(value, np.ndarray):
            inventory[name] = {'type': 'ndarray', 'shape': list(value.shape), 'dtype': str(value.dtype)}
        elif isinstance(value, (int, float, bool, str, type(None))):
            inventory[name] = {'type': type(value).__name__, 'value': value}
        else:
            inventory[name] = {'type': type(value).__name__}
    return inventory


def save_full_checkpoint(path: Path, *, system, meta: dict, train_rng, py_rng_state, torch_rng_state):
    """Serialize the full live Python learner, including persistent E/M states.

    Pickle is deliberately used instead of a weights-only NPZ because the accepted
    MaskGITP3System contains many persistent neuronal state arrays inherited from
    P3ImageSystem.  A companion JSON inventory makes the checkpoint auditable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': 'R6B3_FULL_DYNAMICAL_CHECKPOINT_V1',
        'meta': meta,
        'system': system,
        'train_rng_state': train_rng.bit_generator.state,
        'python_random_state': py_rng_state,
        'torch_rng_state': torch_rng_state.cpu().numpy(),
    }
    with path.open('wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    inv = {
        'format': payload['format'],
        'meta': meta,
        'system_state_inventory': _system_state_keys(system),
    }
    path.with_suffix('.inventory.json').write_text(json.dumps(inv, indent=2), encoding='utf-8')


def load_full_checkpoint(path: Path):
    with path.open('rb') as f:
        payload = pickle.load(f)
    if payload.get('format') != 'R6B3_FULL_DYNAMICAL_CHECKPOINT_V1':
        raise ValueError(f'unsupported checkpoint format: {payload.get("format")}')
    rng = np.random.default_rng()
    rng.bit_generator.state = payload['train_rng_state']
    random.setstate(payload['python_random_state'])
    torch.set_rng_state(torch.from_numpy(payload['torch_rng_state']))
    return payload['system'], payload['meta'], rng


def train_contexts(system, cfg, tokens, tok, pos, ones, indices, rng, start_context, report_every=500):
    """Train on an explicit ordered list of image indices, one MaskGIT context each."""
    history = []
    target_events = 0
    t0 = time.time()
    for local_i, image_index in enumerate(indices, start=1):
        arr = np.asarray(tokens[int(image_index)], dtype=np.int64)
        mask_pos, mask_meta = sample_mask_set(cfg.N, rng)
        # MaskGITP3System owns the accepted same-context/cached-forward batch semantics.
        batch = system.prepare_mask_batch(arr, mask_pos, tok, pos)
        order = np.arange(len(batch['targets']), dtype=np.int64)
        rng.shuffle(order)
        for oi in order:
            system.accumulate_cached_target(batch, int(oi), ones)
        system.apply_update(max(1, len(order)))
        target_events += int(len(order))

        global_context = int(start_context + local_i)
        if local_i == 1 or local_i == len(indices) or (report_every > 0 and local_i % report_every == 0):
            nll = float(batch.get('mean_nll', np.nan))
            row = {
                'context': global_context,
                'image_index': int(image_index),
                'masked_targets': int(len(order)),
                'target_events_segment': int(target_events),
                'mask_ratio_sampled': float(mask_meta.get('mask_ratio', np.nan)),
                'online_mean_nll': nll,
                'WR_norm': float(np.linalg.norm(system.WR)),
                'BR_mean_abs': float(np.mean(np.abs(system.BR))),
                'BO_mean_abs': float(np.mean(np.abs(system.BO))),
                'B2_mean_abs': float(np.mean(np.abs(system.B2))),
                'elapsed_s': float(time.time() - t0),
            }
            history.append(row)
            print('CONTEXT_PROGRESS', json.dumps(row), flush=True)
    return history, target_events


def token_distribution_stats(ids, vocab_size):
    a = np.asarray(ids, dtype=np.int64).reshape(-1)
    counts = np.bincount(a, minlength=vocab_size).astype(np.float64)
    q = counts / max(1.0, counts.sum())
    nz = q[q > 0]
    return {
        'active_codes': int(np.sum(counts > 0)),
        'perplexity': float(np.exp(-(nz * np.log(nz)).sum())),
        'top1_fraction': float(q.max()) if len(q) else 0.0,
    }


def evaluate_checkpoint(*, name, model, cfg, te, tok, pos, tokenizer, device, clf,
                        out, seed, eval_images, gen_samples, choice_temperature):
    ev = {
        str(r): eval_masked_unconditional(model, cfg, te, tok, pos, r,
                                          seed + 1000 + int(100*r), eval_images)
        for r in (.25, .5, .9, 1.0)
    }
    generations = {}
    for iters in (8, 16):
        ids, traces, audits = generation_tokens_maskgit(
            model, cfg, gen_samples, tok, pos, seed + 2000,
            num_iter=iters, choice_temperature=choice_temperature)
        imgs = base.decode_snn(tokenizer, ids, device)
        save_grid_unlabeled(imgs, out / f'{name}_generation_{iters}rounds.png')
        generations[str(iters)] = {
            'posthoc': posthoc_generation_metric(clf, imgs),
            'token_stats': token_distribution_stats(ids, cfg.V),
            'all_trace_audits_pass': bool(all(x['pass'] for x in audits)),
            'audits': audits,
        }
        (out / f'{name}_trace_{iters}rounds.json').write_text(json.dumps(traces, indent=2), encoding='utf-8')
    return {'masked_eval': ev, 'generation': generations}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fractional-epoch', type=float, default=0.25)
    ap.add_argument('--resume', type=Path, default=None)
    ap.add_argument('--D', type=int, default=96); ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--features', type=int, default=24); ap.add_argument('--ff', type=int, default=192); ap.add_argument('--R', type=int, default=7)
    ap.add_argument('--scaffold-contexts', type=int, default=2000); ap.add_argument('--scaffold-paths', type=int, default=2500); ap.add_argument('--scaffold-lr', type=float, default=.6)
    ap.add_argument('--internal-lr-scale', type=float, default=.5)
    ap.add_argument('--eval-images', type=int, default=128); ap.add_argument('--gen-samples', type=int, default=36)
    ap.add_argument('--choice-temperature', type=float, default=4.5)
    ap.add_argument('--encode-batch', type=int, default=256)
    ap.add_argument('--report-every-contexts', type=int, default=500)
    ap.add_argument('--seed', type=int, default=7311); ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()

    if not (0 < a.fractional_epoch <= 1.0):
        raise ValueError('--fractional-epoch must be in (0,1] for one runner invocation')

    a.out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads)
    device = torch.device('cpu')

    controller_audit = controller_self_test()
    masking_audit = masking_self_test(n_tokens=49, vocab_size=64, trials=128, seed=a.seed+17)
    if not controller_audit['pass'] or not masking_audit['pass']:
        raise RuntimeError('MaskGIT self-test failed before training')

    xall,yall,xte0,yte0 = core.load(a.data)
    x012,y012 = core.subset012(xall,yall); xte,yte = core.subset012(xte0,yte0)
    xtr,ytr,xva,yva = core.split(x012,y012,.1,7)

    tokenizer,tok_meta = base.load_frozen_tokenizer(a.tokenizer,device)
    tr,sp_tr = base.encode_snn(tokenizer,xtr,device,a.encode_batch)
    va,sp_va = base.encode_snn(tokenizer,xva,device,a.encode_batch)
    te,sp_te = base.encode_snn(tokenizer,xte,device,a.encode_batch)

    cfg = core.Config(D=a.D,H=a.heads,M=a.features,FF=a.ff,R=a.R)
    tok,pos,input_audit = build_input_embeddings(tokenizer,cfg.D,grid=7)
    rep_audit = representation_self_test(tokenizer,cfg.D,grid=7)
    if not rep_audit['pass']:
        raise RuntimeError('input representation self-test failed')

    ones = np.ones(cfg.V,np.float64)
    start_context = 0
    completed_epochs = 0.0

    if a.resume is None:
        model = core.init_model(cfg,a.seed)
        for k in ('lrq','lrk','lrv','lro','lr1','lr2'):
            model[k] *= a.internal_lr_scale
        F,T,scaffold_audit = build_maskgit_scaffold_contexts_unconditional(
            model,cfg,tr,tok,pos,a.scaffold_contexts,a.seed+101)
        model['WR'] = core.train_readout_msdl(F,T,ones,a.scaffold_paths,a.scaffold_lr,a.seed+202)
        system = MaskGITP3System(cfg,model,a.seed+303)
        train_rng = np.random.default_rng(a.seed+404)
    else:
        system, ckmeta, train_rng = load_full_checkpoint(a.resume)
        scaffold_audit = ckmeta.get('scaffold_audit', {'restored': True})
        start_context = int(ckmeta['contexts_completed'])
        completed_epochs = float(ckmeta['context_epochs_completed'])

    n_train = len(tr)
    n_contexts = max(1, int(math.ceil(a.fractional_epoch*n_train)))
    permutation = train_rng.permutation(n_train)
    indices = permutation[:n_contexts]

    initial_weights = {k: system.weights()[k].copy() for k in ('WQ','WK','WV','WO','W1','W2','WR')}
    hist, target_events = train_contexts(
        system,cfg,tr,tok,pos,ones,indices,train_rng,start_context,
        report_every=a.report_every_contexts)

    final = system.weights()
    checkpoint_name = f'checkpoint_{completed_epochs+a.fractional_epoch:.2f}epoch'
    meta = {
        'seed': a.seed,
        'contexts_completed': int(start_context+n_contexts),
        'context_epochs_completed': float(completed_epochs+a.fractional_epoch),
        'fractional_epoch_this_run': float(a.fractional_epoch),
        'n_train_contexts_per_epoch': int(n_train),
        'target_events_this_run': int(target_events),
        'architecture': {'D':cfg.D,'H':cfg.H,'M':cfg.M,'FF':cfg.FF,'R':cfg.R,'V':cfg.V,'N':cfg.N},
        'scaffold_audit': scaffold_audit,
        'tokenizer_metadata': tok_meta,
    }
    ckpt = a.out / f'{checkpoint_name}.pkl'
    save_full_checkpoint(
        ckpt, system=system, meta=meta, train_rng=train_rng,
        py_rng_state=random.getstate(), torch_rng_state=torch.get_rng_state())

    clf = LogisticRegression(max_iter=250,solver='lbfgs',random_state=7).fit(xtr.reshape(len(xtr),-1),ytr)
    evaluation = evaluate_checkpoint(
        name=checkpoint_name, model=final, cfg=cfg, te=te, tok=tok, pos=pos,
        tokenizer=tokenizer, device=device, clf=clf, out=a.out, seed=a.seed,
        eval_images=a.eval_images, gen_samples=a.gen_samples,
        choice_temperature=a.choice_temperature)

    moves = {
        k: float(np.linalg.norm(final[k]-initial_weights[k])/(np.linalg.norm(initial_weights[k])+1e-12))
        for k in ('WQ','WK','WV','WO','W1','W2','WR')
    }
    finite = bool(all(np.all(np.isfinite(v)) for v in final.values() if isinstance(v,np.ndarray)))
    result = {
        'stage': 'R6B.3 G0 context-epoch training',
        'architecture_changed': False,
        'p1g_reintroduced': False,
        'labels_enter_neural_path': False,
        'context_epoch_definition': 'one shuffled visit per training image with one fresh MaskGIT mask',
        'meta': meta,
        'controller_self_test': controller_audit,
        'masking_self_test': masking_audit,
        'input_representation_audit': rep_audit,
        'tokenizer_activity': {'train':sp_tr,'val':sp_va,'test':sp_te},
        'history': hist,
        'relative_weight_move_this_run': moves,
        'mean_abs_reciprocal': {
            'BR': float(np.mean(np.abs(system.BR))),
            'BO': float(np.mean(np.abs(system.BO))),
            'B2': float(np.mean(np.abs(system.B2))),
        },
        'all_forward_weights_finite': finite,
        'evaluation': evaluation,
        'checkpoint': str(ckpt),
    }
    (a.out/'G0_RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('G0_RESULT',json.dumps({
        'contexts':meta['contexts_completed'],
        'epoch':meta['context_epochs_completed'],
        'target_events':target_events,
        'finite':finite,
        'ce50':evaluation['masked_eval']['0.5']['ce'],
        'acc50':evaluation['masked_eval']['0.5']['acc'],
        'checkpoint':str(ckpt),
    }),flush=True)


if __name__ == '__main__':
    main()
