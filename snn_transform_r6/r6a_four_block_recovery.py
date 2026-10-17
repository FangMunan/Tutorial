#!/usr/bin/env python3
"""R6A: joint recovery of all four H/g Transformer blocks.

R5C established that blocks 2 and 3 can recover jointly using only antithetic
forward perturbations and dual-timescale three-factor updates. R6A extends the
same causal test to the complete four-block stack while keeping token/position/
class embeddings and the prediction head fixed at the E29 transferred solution.

This remains a controlled recovery experiment, not from-scratch training and
not the final class-free system.

Credit semantics
----------------
* Blocks 0, 1 and 2: every perturbation uses per-image masked CE modulation,
  because one local token can be redistributed by downstream attention.
* Final block 3: Q/O/FFN/LN use token-local masked CE; K/V use per-image masked
  CE because each sender key/value can influence all queries.
* No gradient backpropagation, torch optimizer, learned reciprocal matrix, or
  auxiliary trainable credit network is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'snn_transform_e29'))
sys.path.insert(0, str(ROOT / 'snn_transform_r4'))
sys.path.insert(0, str(ROOT / 'snn_transform_r5'))

import maskgit_b0_mnist012 as b0
import e29_forward_equivalence as e29
import r4_single_module_recovery as r4
import r5c_two_block_recovery as r5c


BLOCKS = (0, 1, 2, 3)


def target_params(model):
    out = {}
    for bi in BLOCKS:
        out.update(r5c.block_params(model, bi))
    return out


def lr_scale_for_block(bi, a):
    return {0: a.scale_b0, 1: a.scale_b1, 2: a.scale_b2, 3: 1.0}[bi]


def lrs_for(params, a):
    out = {}
    for name in params:
        if '.WQ.' in name or '.WK.' in name or '.WV.' in name:
            lr = a.lr_qkv
        elif '.WO.' in name:
            lr = a.lr_wo
        elif '.f1.' in name:
            lr = a.lr_f1
        elif '.f2.' in name:
            lr = a.lr_f2
        else:
            lr = a.lr_ln
        bi = int(name.split('.')[1])
        lr *= lr_scale_for_block(bi, a)
        if name.endswith('.bias'):
            lr *= 0.5
        out[name] = lr
    return out


@torch.no_grad()
def block_gradset(model, bi, t, y, inp, mask, sigma, generator, clip):
    """Forward-perturbative gradient estimate for one block.

    Every non-final block uses image-level modulation because downstream
    attention can mix a perturbed token into every receiver. The final block
    retains the R5B/R5C mixed rule.
    """
    x0 = r5c.prefix_to(model, inp, y, bi)
    _, cache = r5c.logits_from_target(model, x0, bi, cache=True)
    B, N, D = x0.shape
    FF = e29.FF
    deep = bi < 3
    modes = {
        'q': 'image' if deep else 'token',
        'k': 'image',
        'v': 'image',
        'wo': 'image' if deep else 'token',
        'n1': 'image' if deep else 'token',
        'f1': 'image' if deep else 'token',
        'f2': 'image' if deep else 'token',
        'n2': 'image' if deep else 'token',
    }
    specs = [
        ('WQ', 'q', (B, N, D), cache['x0'], False, None),
        ('WK', 'k', (B, N, D), cache['x0'], False, None),
        ('WV', 'v', (B, N, D), cache['x0'], False, None),
        ('WO', 'wo', (B, N, D), cache['read'], False, None),
        ('n1', 'n1', (B, N, D), None, True, cache['r1']),
        ('f1', 'f1', (B, N, FF), cache['y1'], False, None),
        ('f2', 'f2', (B, N, D), cache['h'], False, None),
        ('n2', 'n2', (B, N, D), None, True, cache['r2']),
    ]
    grads, ces, norms = {}, [], {}
    pfx = f'blocks.{bi}'
    for label, kind, shape, pre, is_ln, normed in specs:
        (gw, gb), ce = r5c.probe(
            model, x0, bi, t, mask, kind, shape, sigma, pre, generator,
            modes[kind], is_ln, normed
        )
        ces.append(ce)
        base = f'{pfx}.attn.{label}' if label in ('WQ', 'WK', 'WV', 'WO') else f'{pfx}.{label}'
        for suffix, g in (('weight', gw), ('bias', gb)):
            name = f'{base}.{suffix}'
            norms[name] = r5c.clip_(g, clip)
            grads[name] = g
    return grads, float(np.mean(ces)), norms


def weight_diag_group(params, gold):
    return {
        k: r4.weight_diag(p, gold[k])
        for k, p in params.items()
        if k.endswith('.weight')
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--codebook', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--features', type=int, default=256)
    ap.add_argument('--rf-kind', default='p3norm')
    ap.add_argument('--rf-seed', type=int, default=29257)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--sigma', type=float, default=.05)
    ap.add_argument('--lr-qkv', type=float, default=.020)
    ap.add_argument('--lr-wo', type=float, default=.040)
    ap.add_argument('--lr-f1', type=float, default=.055)
    ap.add_argument('--lr-f2', type=float, default=.080)
    ap.add_argument('--lr-ln', type=float, default=.015)
    ap.add_argument('--scale-b0', type=float, default=.35)
    ap.add_argument('--scale-b1', type=float, default=.50)
    ap.add_argument('--scale-b2', type=float, default=.70)
    ap.add_argument('--rho-fast', type=float, default=.60)
    ap.add_argument('--rho-slow', type=float, default=.95)
    ap.add_argument('--mix-fast', type=float, default=.70)
    ap.add_argument('--clip', type=float, default=.60)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--seed', type=int, default=5801)
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    a.out.mkdir(parents=True, exist_ok=True)

    xall, yall, xte0, yte0 = b0.load(a.data)
    x012, y012 = b0.subset012(xall, yall)
    xte, yte = b0.subset012(xte0, yte0)
    xtr, ytr, xva, yva = b0.split(x012, y012, .1, 7)
    cb = np.load(a.codebook)
    tr = b0.encode(xtr, cb)
    va = b0.encode(xva, cb)
    te = b0.encode(xte, cb)

    exact = b0.Model(128, 4, 4, 512, .05)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    exact.load_state_dict(ck['model'])
    exact.eval()
    model = e29.HgB0Model(a.features, a.rf_kind, a.rf_seed)
    e29.transfer_b0(exact, model)
    model.eval()
    r4.freeze_all(model)

    params = target_params(model)
    allowed = set(params)
    gold = {k: p.clone() for k, p in params.items()}
    hash_intact = r4.model_hashes(model)
    names = {bi: {k for k in params if k.startswith(f'blocks.{bi}.')} for bi in BLOCKS}

    ratios = (.25, .5, .9, 1.0)
    intact = r4.eval_mask_ratios(model, te, yte, ratios)
    clf = r4.external_classifier(xtr, ytr)
    gen_i = r4.generate_snapshot(model, cb, clf, a.out, 'intact', a.seed + 10)

    # Cumulative lesion ladder: B3 -> B2+B3 -> B1+B2+B3 -> all four blocks.
    ladder = {}
    for bi in (3, 2, 1, 0):
        r5c.damage_one(model, bi)
        key = '_'.join(str(j) for j in range(bi, 4))
        ladder[key] = r4.eval_mask_ratios(model, te, yte, (.5,), batch=256)['0.5']['ce']

    damaged_state = {k: p.clone() for k, p in params.items()}
    damaged = r4.eval_mask_ratios(model, te, yte, ratios)
    gen_d = r4.generate_snapshot(model, cb, clf, a.out, 'damaged_four_blocks', a.seed + 10)

    hash_damaged = r4.model_hashes(model)
    changed_damage = {n for n in hash_intact if hash_intact[n] != hash_damaged[n]}
    required = {
        f'blocks.{bi}.{suffix}'
        for bi in BLOCKS
        for suffix in (
            'attn.WQ.weight', 'attn.WK.weight', 'attn.WV.weight', 'attn.WO.weight',
            'f1.weight', 'f2.weight', 'n1.weight', 'n2.weight'
        )
    }
    if not changed_damage.issubset(allowed) or not required.issubset(changed_damage):
        raise RuntimeError('four-block damage isolation failed')

    states = {k: (torch.zeros_like(p), torch.zeros_like(p)) for k, p in params.items()}
    lr = lrs_for(params, a)
    rng = np.random.default_rng(a.seed + 77)
    tg = torch.Generator()
    tg.manual_seed(a.seed + 99)

    hist = []
    best = float('inf')
    best_state = {k: p.clone() for k, p in params.items()}
    for ep in range(1, a.epochs + 1):
        order = rng.permutation(len(tr))
        probe_ce = {bi: [] for bi in BLOCKS}
        grad_norm = {bi: [] for bi in BLOCKS}
        for s in range(0, len(order), a.batch):
            idx = order[s:s + a.batch]
            t, y, inp, mask = r4.masked_random_batch(tr, ytr, idx)
            all_grads = {}
            # All probes are measured before any parameter update on this batch.
            for bi in BLOCKS:
                grads, ce, norms = block_gradset(model, bi, t, y, inp, mask, a.sigma, tg, a.clip)
                all_grads.update(grads)
                probe_ce[bi].append(ce)
                grad_norm[bi].append(float(np.mean(list(norms.values()))))
            for name, g in all_grads.items():
                fast, slow = states[name]
                fast.mul_(a.rho_fast).add_(g, alpha=1 - a.rho_fast)
                slow.mul_(a.rho_slow).add_(g, alpha=1 - a.rho_slow)
                u = a.mix_fast * fast + (1 - a.mix_fast) * slow
                params[name].add_(u, alpha=-lr[name])

        val = r4.eval_mask_ratios(model, va, yva, (.5,), batch=256)['0.5']['ce']
        test50 = r4.eval_mask_ratios(model, te, yte, (.5,), batch=256, limit=1024)['0.5']['ce']
        rec = {'epoch': ep}
        for bi in BLOCKS:
            rec[f'mean_probe_ce_block{bi}'] = float(np.mean(probe_ce[bi]))
            rec[f'mean_grad_norm_block{bi}'] = float(np.mean(grad_norm[bi]))
        rec['val_ce_50'] = val
        rec['test_ce_50_first1024'] = test50
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        if val < best:
            best = val
            best_state = {k: p.clone() for k, p in params.items()}

    r5c.copy_group(params, best_state, set(params))
    recovered = r4.eval_mask_ratios(model, te, yte, ratios)
    gen_r = r4.generate_snapshot(model, cb, clf, a.out, 'recovered_four_blocks', a.seed + 10)
    diag = weight_diag_group(params, gold)

    # Functional-contribution audit: revert one recovered block at a time to its damaged state.
    recovered_state = {k: p.clone() for k, p in params.items()}
    revert_ce = {}
    for bi in BLOCKS:
        r5c.copy_group(params, damaged_state, names[bi])
        revert_ce[str(bi)] = r4.eval_mask_ratios(model, te, yte, (.5,), batch=256)['0.5']['ce']
        r5c.copy_group(params, recovered_state, names[bi])

    hash_final = r4.model_hashes(model)
    changed_final = {n for n in hash_intact if hash_intact[n] != hash_final[n]}
    if not changed_final.issubset(allowed):
        raise RuntimeError(f'outside four-block learning: {sorted(changed_final - allowed)}')
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('autograd gradient exists in R6A')

    def frac(r):
        ci = intact[r]['ce']; cd = damaged[r]['ce']; cr = recovered[r]['ce']
        return float((cd - cr) / (cd - ci)) if abs(cd - ci) > 1e-12 else float('nan')

    contributions = {bi: revert_ce[str(bi)] - recovered['0.5']['ce'] for bi in BLOCKS}
    ladder_increments = {
        'block2_added_after_block3': ladder['2_3'] - ladder['3'],
        'block1_added_after_blocks2_3': ladder['1_2_3'] - ladder['2_3'],
        'block0_added_after_blocks1_2_3': ladder['0_1_2_3'] - ladder['1_2_3'],
    }
    gate = {
        'damage_isolated_to_blocks_0_3': changed_damage.issubset(allowed) and required.issubset(changed_damage),
        'learning_isolated_to_blocks_0_3': changed_final.issubset(allowed),
        'no_autograd_gradients': all(p.grad is None for p in model.parameters()),
        'damage_delta_ce_50': damaged['0.5']['ce'] - intact['0.5']['ce'],
        'block2_damage_increment_ce_50': ladder_increments['block2_added_after_block3'],
        'block1_damage_increment_ce_50': ladder_increments['block1_added_after_blocks2_3'],
        'block0_damage_increment_ce_50': ladder_increments['block0_added_after_blocks1_2_3'],
        'recovery_fraction_ce_25': frac('0.25'),
        'recovery_fraction_ce_50': frac('0.5'),
        'recovery_fraction_ce_90': frac('0.9'),
        'recovered_better_25': recovered['0.25']['ce'] < damaged['0.25']['ce'],
        'recovered_better_50': recovered['0.5']['ce'] < damaged['0.5']['ce'],
        'recovered_better_90': recovered['0.9']['ce'] < damaged['0.9']['ce'],
        'block0_recovery_functional_contribution_ce_50': contributions[0],
        'block1_recovery_functional_contribution_ce_50': contributions[1],
        'block2_recovery_functional_contribution_ce_50': contributions[2],
        'block3_recovery_functional_contribution_ce_50': contributions[3],
        'generation_condition_match_recovered': gen_r['condition_match'],
        'generation_mean_confidence_recovered': gen_r['mean_confidence'],
    }
    gate['pass'] = bool(
        gate['damage_delta_ce_50'] > .20
        and gate['block1_damage_increment_ce_50'] > .005
        and gate['block0_damage_increment_ce_50'] > .005
        and gate['recovery_fraction_ce_50'] > .50
        and gate['recovered_better_25']
        and gate['recovered_better_50']
        and gate['recovered_better_90']
        and gate['block0_recovery_functional_contribution_ce_50'] > .002
        and gate['block1_recovery_functional_contribution_ce_50'] > .002
        and gate['no_autograd_gradients']
        and gate['learning_isolated_to_blocks_0_3']
    )

    out = {
        'stage': 'R6A all-four-block H/g joint recovery',
        'scope': 'blocks 0-3 plastic; token/position/class embeddings and readout fixed',
        'credit_rule': 'blocks0-2 all image-level; block3 Q/O/FFN/LN token-local and K/V image-level',
        'backprop_used': False,
        'torch_optimizer_used': False,
        'learned_feedback_network_used': False,
        'class_conditioning_retained': True,
        'from_scratch_training': False,
        'rf_kind': a.rf_kind,
        'rf_features': a.features,
        'depth_lr_scales': {'block0': a.scale_b0, 'block1': a.scale_b1, 'block2': a.scale_b2, 'block3': 1.0},
        'intact_eval': intact,
        'damage_ladder_ce_50': ladder,
        'damage_ladder_increments_ce_50': ladder_increments,
        'damaged_eval': damaged,
        'recovered_eval': recovered,
        'generation_intact': gen_i,
        'generation_damaged': gen_d,
        'generation_recovered': gen_r,
        'best_validation_ce_50': best,
        'single_block_revert_ce_50': revert_ce,
        'gate': gate,
        'recovered_weight_diag': diag,
        'history': hist,
    }
    (a.out / 'metrics.json').write_text(json.dumps(out, indent=2))
    with (a.out / 'history.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=hist[0].keys())
        w.writeheader(); w.writerows(hist)
    torch.save({'params': {k: p.clone() for k, p in params.items()}, 'metrics': out}, a.out / 'recovery_checkpoint.pt')
    print('FINAL ' + json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
