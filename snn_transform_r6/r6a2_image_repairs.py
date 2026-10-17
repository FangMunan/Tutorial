#!/usr/bin/env python3
"""R6A.2: literature-guided image-training repair diagnostics.

This is deliberately a *diagnostic bridge*, not a replacement for the P3 MSDL
scientific core.  It reuses the validated R6A four-block recovery experiment to
isolate two concrete failure mechanisms seen in the image branch:

1) visual-token class imbalance (dominant background token), and
2) depth-dependent learning-signal scale/conditioning.

Variants
--------
cb   : class-balanced token modulation only
norm : stateful learning-signal RMS normalization only
both : both repairs

The forward model, lesion, masks, H/g attention, no-backprop constraint, and
R6A recovery protocol are otherwise unchanged. Evaluation remains *unweighted*
so improvements cannot be manufactured by changing the metric.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import r6a_four_block_recovery as base


# -----------------------------------------------------------------------------
# Variant state
# -----------------------------------------------------------------------------
USE_CB = False
USE_NORM = False
CB_BETA = 0.9999
SIGNAL_RHO = 0.98
TOKEN_WEIGHTS = torch.ones(base.b0.K, dtype=torch.float32)
TOKEN_COUNTS = np.ones(base.b0.K, dtype=np.int64)
DOMINANT_TOKEN = 0
SIGNAL_RMS2: dict[str, float] = {}
SIGNAL_OBS: dict[str, int] = {}


def _get_arg(argv, flag: str) -> str:
    try:
        i = argv.index(flag)
    except ValueError as e:
        raise SystemExit(f"required argument {flag} is missing") from e
    if i + 1 >= len(argv):
        raise SystemExit(f"argument {flag} requires a value")
    return argv[i + 1]


def class_balanced_weights(counts: np.ndarray, beta: float) -> np.ndarray:
    """Effective-number class balancing, rescaled to unit empirical mean.

    Shape follows Cui et al. CVPR 2019: (1-beta)/(1-beta**n_k).  The final scalar
    normalization preserves E_data[w(target)] = 1, so this experiment changes
    relative token credit rather than the overall learning-rate scale.
    """
    c = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    raw = (1.0 - beta) / np.maximum(1.0 - np.power(beta, c), 1e-30)
    p = c / c.sum()
    raw /= float(np.sum(p * raw))
    return raw


def _normalize_signal(key: str, m: torch.Tensor, selected: torch.Tensor | None = None) -> torch.Tensor:
    if not USE_NORM:
        return m
    with torch.no_grad():
        vals = m if selected is None else m[selected]
        if vals.numel() == 0:
            return m
        batch_m2 = float(vals.square().mean())
        if key not in SIGNAL_RMS2:
            SIGNAL_RMS2[key] = max(batch_m2, 1e-12)
            SIGNAL_OBS[key] = 1
        else:
            SIGNAL_RMS2[key] = SIGNAL_RHO * SIGNAL_RMS2[key] + (1.0 - SIGNAL_RHO) * batch_m2
            SIGNAL_OBS[key] += 1
        scale = math.sqrt(max(SIGNAL_RMS2[key], 1e-12))
        return m / scale


@torch.no_grad()
def probe_repair(model, x0, bi, t, mask, kind, shape, sigma, pre, generator,
                 mode, is_ln=False, r=None):
    """R5C/R6A node perturbation with controlled token weighting and signal gain."""
    xi = torch.randn(shape, generator=generator)
    lp = base.r5c.logits_from_target(model, x0, bi, kind, sigma * xi)
    lm = base.r5c.logits_from_target(model, x0, bi, kind, -sigma * xi)

    cp0 = F.cross_entropy(lp.transpose(1, 2), t, reduction='none')
    cm0 = F.cross_entropy(lm.transpose(1, 2), t, reduction='none')
    if USE_CB:
        w = TOKEN_WEIGHTS.to(t.device)[t]
        cp = cp0 * w
        cm = cm0 * w
    else:
        cp, cm = cp0, cm0

    key = f'b{bi}:{kind}:{mode}'
    if mode == 'token':
        m = (cp - cm) / (2.0 * sigma)
        m = _normalize_signal(key, m, mask)
        ce = float(0.5 * (cp0[mask].mean() + cm0[mask].mean()))
        grad = (base.r5c.ln_token_grad(m, mask, xi, r) if is_ln
                else base.r5c.linear_token_grad(m, mask, xi, pre))
    else:
        cnt = mask.sum(1).clamp_min(1)
        a = (cp * mask).sum(1) / cnt
        b = (cm * mask).sum(1) / cnt
        m = (a - b) / (2.0 * sigma)
        m = _normalize_signal(key, m)
        a0 = (cp0 * mask).sum(1) / cnt
        b0 = (cm0 * mask).sum(1) / cnt
        ce = float(0.5 * (a0.mean() + b0.mean()))
        grad = (base.r5c.ln_image_grad(m, xi, r) if is_ln
                else base.r5c.linear_image_grad(m, xi, pre))
    return grad, ce


@torch.no_grad()
def block_gradset_repair(model, bi, t, y, inp, mask, sigma, generator, clip):
    x0 = base.r5c.prefix_to(model, inp, y, bi)
    _, cache = base.r5c.logits_from_target(model, x0, bi, cache=True)
    B, N, D = x0.shape
    FF = base.e29.FF
    deep = bi < 3
    modes = {
        'q': 'image' if deep else 'token',
        'k': 'image', 'v': 'image',
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
        (gw, gb), ce = probe_repair(
            model, x0, bi, t, mask, kind, shape, sigma, pre, generator,
            modes[kind], is_ln, normed
        )
        ces.append(ce)
        base_name = (f'{pfx}.attn.{label}' if label in ('WQ', 'WK', 'WV', 'WO')
                     else f'{pfx}.{label}')
        for suffix, g in (('weight', gw), ('bias', gb)):
            name = f'{base_name}.{suffix}'
            norms[name] = base.r5c.clip_(g, clip)
            grads[name] = g
    return grads, float(np.mean(ces)), norms


@torch.no_grad()
def eval_mask_ratios_diag(model, tok_np, y_np, ratios, batch=256, limit=None):
    """Original unweighted evaluation plus dominant-vs-foreground diagnostics."""
    model.eval()
    if limit is not None:
        tok_np = tok_np[:limit]
        y_np = y_np[:limit]
    T = torch.from_numpy(tok_np).long()
    Y = torch.from_numpy(y_np).long()
    out = {}
    nondom = torch.ones(base.b0.K, dtype=torch.bool)
    nondom[DOMINANT_TOKEN] = False
    for ratio in ratios:
        loss_sum = acc_sum = n_mask = 0.0
        fg_loss = fg_acc = fg_mass = fg_n = 0.0
        bg_n = 0.0
        for i in range(0, len(T), batch):
            t = T[i:i+batch]
            y = Y[i:i+batch]
            inp, mask = base.b0.fixed_mask_batch(t, ratio, seed=41000 + i + int(ratio * 1000))
            logits = model(inp, y)
            n = int(mask.sum())
            loss_sum += float(F.cross_entropy(logits[mask], t[mask], reduction='sum'))
            acc_sum += float((logits[mask].argmax(-1) == t[mask]).sum())
            n_mask += n
            fg = mask & t.ne(DOMINANT_TOKEN)
            nf = int(fg.sum())
            if nf:
                lf = logits[fg]
                tf = t[fg]
                fg_loss += float(F.cross_entropy(lf, tf, reduction='sum'))
                fg_acc += float((lf.argmax(-1) == tf).sum())
                prob = lf.softmax(-1)
                fg_mass += float(prob[:, nondom].sum())
                fg_n += nf
            bg_n += float((mask & t.eq(DOMINANT_TOKEN)).sum())
        out[str(ratio)] = {
            'ce': loss_sum / max(1.0, n_mask),
            'masked_token_acc': acc_sum / max(1.0, n_mask),
            'masked_tokens': int(n_mask),
            'dominant_token_id': int(DOMINANT_TOKEN),
            'dominant_target_fraction': bg_n / max(1.0, n_mask),
            'foreground_ce': fg_loss / max(1.0, fg_n),
            'foreground_acc': fg_acc / max(1.0, fg_n),
            'foreground_non_dominant_mass': fg_mass / max(1.0, fg_n),
            'foreground_tokens': int(fg_n),
        }
    return out


def main():
    global USE_CB, USE_NORM, CB_BETA, SIGNAL_RHO
    global TOKEN_WEIGHTS, TOKEN_COUNTS, DOMINANT_TOKEN

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--variant', choices=('cb', 'norm', 'both'), required=True)
    pre.add_argument('--cb-beta', type=float, default=0.9999)
    pre.add_argument('--signal-rho', type=float, default=0.98)
    cfg, remaining = pre.parse_known_args()
    USE_CB = cfg.variant in ('cb', 'both')
    USE_NORM = cfg.variant in ('norm', 'both')
    CB_BETA = cfg.cb_beta
    SIGNAL_RHO = cfg.signal_rho

    # Use exactly the R6A training split/tokenizer to compute label-free token counts.
    data_path = Path(_get_arg(remaining, '--data'))
    codebook_path = Path(_get_arg(remaining, '--codebook'))
    xall, yall, _, _ = base.b0.load(data_path)
    x012, y012 = base.b0.subset012(xall, yall)
    xtr, ytr, _, _ = base.b0.split(x012, y012, .1, 7)
    cb = np.load(codebook_path)
    tr = base.b0.encode(xtr, cb)
    TOKEN_COUNTS = np.bincount(tr.reshape(-1), minlength=base.b0.K).astype(np.int64)
    DOMINANT_TOKEN = int(np.argmax(TOKEN_COUNTS))
    weights = class_balanced_weights(TOKEN_COUNTS, CB_BETA) if USE_CB else np.ones(base.b0.K)
    TOKEN_WEIGHTS = torch.tensor(weights, dtype=torch.float32)

    # Patch only the two diagnostic interfaces. The R6A scaffold/protocol stays intact.
    base.block_gradset = block_gradset_repair
    base.r4.eval_mask_ratios = eval_mask_ratios_diag

    # Remove wrapper-only flags and execute the frozen R6A protocol.
    sys.argv = [sys.argv[0]] + remaining
    base.main()

    # Append mechanism diagnostics to the generated metrics without changing any gate.
    outdir = Path(_get_arg(remaining, '--out'))
    mp = outdir / 'metrics.json'
    result = json.loads(mp.read_text())
    result['r6a2_repair'] = {
        'variant': cfg.variant,
        'class_balanced_modulation': USE_CB,
        'class_balance_beta': CB_BETA if USE_CB else None,
        'class_balance_formula': '(1-beta)/(1-beta**count), rescaled to empirical mean weight 1',
        'signal_rms_normalization': USE_NORM,
        'signal_rho': SIGNAL_RHO if USE_NORM else None,
        'signal_normalization_scope': 'per block x perturbation-node x token/image mode; modulation only',
        'token_counts': TOKEN_COUNTS.tolist(),
        'token_weights': TOKEN_WEIGHTS.tolist(),
        'dominant_token_id': DOMINANT_TOKEN,
        'dominant_token_fraction': float(TOKEN_COUNTS[DOMINANT_TOKEN] / TOKEN_COUNTS.sum()),
        'signal_rms2_final': SIGNAL_RMS2,
        'signal_observations': SIGNAL_OBS,
        'evaluation_loss_is_unweighted': True,
        'scientific_boundary': 'diagnostic bridge only; does not redefine P3 exact E/M Izhikevich MSDL core',
    }
    mp.write_text(json.dumps(result, indent=2))
    print('R6A2_DIAGNOSTIC ' + json.dumps(result['r6a2_repair']), flush=True)


if __name__ == '__main__':
    main()
