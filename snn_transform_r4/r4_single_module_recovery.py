#!/usr/bin/env python3
"""R4: single-module recovery gate for SNN_Transform.

Purpose
-------
Start from the E29 solution that already generates correctly after B0 -> H/g
weight transfer. Damage exactly one *existing forward parameter matrix* and test
whether a forward-perturbative, three-factor MSDL rule can recover function.

This first R4 gate deliberately chooses the final block W2 matrix. This is the
cleanest causal case because after W2 there is no further cross-token mixing:
for token i, a perturbation of W2's output affects only token i's masked-token
loss. Therefore the token-local antithetic perturbation signal is well defined.

NO backpropagation is used. NO optimizer is used. NO learned feedback matrix is
introduced. Only blocks[3].f2.weight may change.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "snn_transform_e29"))

import maskgit_b0_mnist012 as b0
import e29_forward_equivalence as e29


@torch.no_grad()
def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None


def sha_tensor(t: torch.Tensor) -> str:
    a = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def model_hashes(model):
    return {n: sha_tensor(p) for n, p in model.named_parameters()}


@torch.no_grad()
def final_block_prefix(model: e29.HgB0Model, tokens: torch.Tensor, labels: torch.Tensor):
    """Return x_attn and h=GELU(W1 x_attn) before final W2."""
    z = model.tok(tokens) + model.pos + model.cls(labels)[:, None]
    for block in model.blocks[:-1]:
        z = block(z)
    block = model.blocks[-1]
    x_attn = block.n1(z + block.attn(z))
    h = F.gelu(block.f1(x_attn))
    return x_attn, h


@torch.no_grad()
def final_block_tail(model: e29.HgB0Model, x_attn: torch.Tensor, h: torch.Tensor,
                     perturb: torch.Tensor | None = None):
    """Finish the network from final W2 onward, optionally perturbing W2 output."""
    block = model.blocks[-1]
    u = block.f2(h)
    if perturb is not None:
        u = u + perturb
    z = block.n2(x_attn + u)
    z = model.ln(F.gelu(model.dense(z)))
    return F.linear(z, model.tok.weight[:b0.K], model.bias)


@torch.no_grad()
def eval_mask_ratios(model, tok_np, y_np, ratios, batch=256, limit=None):
    model.eval()
    if limit is not None:
        tok_np = tok_np[:limit]
        y_np = y_np[:limit]
    T = torch.from_numpy(tok_np).long()
    Y = torch.from_numpy(y_np).long()
    out = {}
    for ratio in ratios:
        loss_sum = acc_sum = n_mask = 0.0
        for i in range(0, len(T), batch):
            t = T[i:i+batch]
            y = Y[i:i+batch]
            inp, mask = b0.fixed_mask_batch(t, ratio, seed=41000 + i + int(ratio * 1000))
            logits = model(inp, y)
            n = int(mask.sum())
            loss_sum += float(F.cross_entropy(logits[mask], t[mask], reduction="sum"))
            acc_sum += float((logits[mask].argmax(-1) == t[mask]).sum())
            n_mask += n
        out[str(ratio)] = {
            "ce": loss_sum / max(1.0, n_mask),
            "masked_token_acc": acc_sum / max(1.0, n_mask),
            "masked_tokens": int(n_mask),
        }
    return out


@torch.no_grad()
def weight_diag(w, target):
    a = w.flatten(); b = target.flatten()
    rel = float((a-b).norm() / b.norm().clamp_min(1e-12))
    cos = float(F.cosine_similarity(a[None], b[None]).item())
    return {"relative_distance_to_gold": rel, "cosine_to_gold": cos,
            "rms": float(w.square().mean().sqrt())}


def save_grid(img, lab, path, title):
    n = len(img); cols = 15; rows = math.ceil(n / cols)
    fig, ax = plt.subplots(rows, cols, figsize=(cols, rows * 1.15))
    ax = np.asarray(ax).reshape(-1)
    for i, a in enumerate(ax):
        a.axis("off")
        if i < n:
            a.imshow(img[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            a.set_title(str(int(lab[i])), fontsize=7)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def external_classifier(xtr, ytr):
    return LogisticRegression(max_iter=500, solver="lbfgs", random_state=7).fit(
        xtr.reshape(len(xtr), -1), ytr)


def gen_metrics(clf, img, labels):
    pred = clf.predict(img.reshape(len(img), -1))
    p = clf.predict_proba(img.reshape(len(img), -1))
    return {
        "condition_match": float((pred == labels).mean()),
        "mean_confidence": float(p.max(1).mean()),
        "pred_counts": np.bincount(pred, minlength=3).tolist(),
    }


@torch.no_grad()
def generate_snapshot(model, codebook, clf, outdir, tag, seed, n_per_class=12):
    labels_np = np.tile(np.arange(3, dtype=np.int64), n_per_class)
    labels = torch.from_numpy(labels_np)
    toks = e29.hg_generate(model, labels, torch.device("cpu"), steps=16,
                           seed=seed, greedy=False)
    imgs = b0.decode(toks, codebook)
    save_grid(imgs, labels_np, outdir / f"generation_{tag}.png", tag)
    return gen_metrics(clf, imgs, labels_np)


def masked_random_batch(tok_np, y_np, indices):
    t = torch.from_numpy(tok_np[indices]).long()
    y = torch.from_numpy(y_np[indices]).long()
    inp, mask, _ = b0.mask_batch(t)
    return t, y, inp, mask


@torch.no_grad()
def perturbative_msdl_step(model, t, y, inp, mask, sigma, fast_state, slow_state,
                           lr, rho_fast, rho_slow, mix_fast, grad_clip, generator):
    """One forward-perturbative MSDL update of final W2 only.

    For each masked token i:
      directional modulation = (ell_i(+sigma xi_i)-ell_i(-sigma xi_i))/(2 sigma)
      eligibility            = xi_i h_i^T
      local update estimate  = modulation * eligibility

    The fast/slow states are non-learned optimizer/learning dynamics, not model
    parameters. They are retained only to match the multistate-learning premise.
    """
    x_attn, h = final_block_prefix(model, inp, y)
    xi = torch.randn((len(t), b0.L, e29.D), generator=generator)
    logits_p = final_block_tail(model, x_attn, h, sigma * xi)
    logits_m = final_block_tail(model, x_attn, h, -sigma * xi)

    # Per-position CE; final W2 is after all token mixing, so token i's
    # perturbation is paired only with token i's loss.
    loss_p = F.cross_entropy(logits_p.transpose(1, 2), t, reduction="none")
    loss_m = F.cross_entropy(logits_m.transpose(1, 2), t, reduction="none")
    modulation = (loss_p - loss_m) / (2.0 * sigma)

    sel = mask.float()
    n = sel.sum().clamp_min(1.0)
    g = torch.einsum("bn,bnd,bnf->df", modulation * sel, xi, h) / n

    gnorm = float(g.norm())
    if grad_clip > 0 and gnorm > grad_clip:
        g.mul_(grad_clip / (gnorm + 1e-12))
        gnorm = grad_clip

    fast_state.mul_(rho_fast).add_(g, alpha=1.0-rho_fast)
    slow_state.mul_(rho_slow).add_(g, alpha=1.0-rho_slow)
    update = mix_fast * fast_state + (1.0-mix_fast) * slow_state
    model.blocks[-1].f2.weight.add_(update, alpha=-lr)

    masked_loss_mid = 0.5 * (loss_p[mask].mean() + loss_m[mask].mean())
    return float(masked_loss_mid), gnorm, float(update.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--codebook", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--features", type=int, default=256)
    ap.add_argument("--rf-kind", type=str, default="p3norm")
    ap.add_argument("--rf-seed", type=int, default=29257,
                    help="E29 p3norm-m256 construction seed (29001+256)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--rho-fast", type=float, default=0.60)
    ap.add_argument("--rho-slow", type=float, default=0.95)
    ap.add_argument("--mix-fast", type=float, default=0.70)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=4401)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    a.out.mkdir(parents=True, exist_ok=True)

    xall, yall, xte0, yte0 = b0.load(a.data)
    x012, y012 = b0.subset012(xall, yall)
    xte, yte = b0.subset012(xte0, yte0)
    xtr, ytr, xva, yva = b0.split(x012, y012, .1, 7)
    codebook = np.load(a.codebook)
    tr = b0.encode(xtr, codebook)
    va = b0.encode(xva, codebook)
    te = b0.encode(xte, codebook)

    exact = b0.Model(128, 4, 4, 512, .05)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    exact.load_state_dict(ck["model"]); exact.eval()

    model = e29.HgB0Model(a.features, a.rf_kind, a.rf_seed)
    e29.transfer_b0(exact, model); model.eval(); freeze_all(model)

    target_name = "blocks.3.f2.weight"
    target = model.blocks[-1].f2.weight
    gold_w = target.detach().clone()
    hashes_intact = model_hashes(model)

    ratios = (.25, .5, .9, 1.0)
    intact_eval = eval_mask_ratios(model, te, yte, ratios)
    clf = external_classifier(xtr, ytr)
    gen_intact = generate_snapshot(model, codebook, clf, a.out, "intact", a.seed+10)

    # Controlled lesion: erase exactly the final W2 matrix, leave its bias and
    # every other forward parameter at the E29-transferred value.
    target.zero_()
    damaged_eval = eval_mask_ratios(model, te, yte, ratios)
    gen_damaged = generate_snapshot(model, codebook, clf, a.out, "damaged_W2_zero", a.seed+10)
    damaged_diag = weight_diag(target, gold_w)

    hashes_damaged = model_hashes(model)
    changed_by_damage = [n for n in hashes_intact if hashes_intact[n] != hashes_damaged[n]]
    if changed_by_damage != [target_name]:
        raise RuntimeError(f"Damage isolation failure: changed={changed_by_damage}")

    fast = torch.zeros_like(target)
    slow = torch.zeros_like(target)
    rng = np.random.default_rng(a.seed + 77)
    tg = torch.Generator(device="cpu"); tg.manual_seed(a.seed + 99)

    history = []
    best_val = float("inf")
    best_w = target.clone()

    for ep in range(1, a.epochs + 1):
        order = rng.permutation(len(tr))
        losses = []; gnorms = []; unorms = []
        for s in range(0, len(order), a.batch):
            idx = order[s:s+a.batch]
            t, y, inp, mask = masked_random_batch(tr, ytr, idx)
            ml, gn, un = perturbative_msdl_step(
                model, t, y, inp, mask,
                a.sigma, fast, slow, a.lr,
                a.rho_fast, a.rho_slow, a.mix_fast,
                a.grad_clip, tg)
            losses.append(ml); gnorms.append(gn); unorms.append(un)

        val = eval_mask_ratios(model, va, yva, (.5,), batch=256)["0.5"]["ce"]
        test50 = eval_mask_ratios(model, te, yte, (.5,), batch=256, limit=1024)["0.5"]["ce"]
        wd = weight_diag(target, gold_w)
        rec = {
            "epoch": ep,
            "mean_train_perturbed_ce": float(np.mean(losses)),
            "mean_gradient_estimate_norm": float(np.mean(gnorms)),
            "mean_update_state_norm": float(np.mean(unorms)),
            "val_ce_50": val,
            "test_ce_50_first1024": test50,
            **wd,
        }
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if val < best_val:
            best_val = val
            best_w = target.clone()

    # Validation-selected recovery checkpoint; no gold weight was used by learning.
    target.copy_(best_w)
    recovered_eval = eval_mask_ratios(model, te, yte, ratios)
    recovered_diag = weight_diag(target, gold_w)
    gen_recovered = generate_snapshot(model, codebook, clf, a.out, "recovered", a.seed+10)

    hashes_final = model_hashes(model)
    changed_total = [n for n in hashes_intact if hashes_intact[n] != hashes_final[n]]
    if changed_total != [target_name]:
        raise RuntimeError(f"Learning isolation failure: changed={changed_total}")
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("A gradient tensor exists; R4 must be gradient-free")

    def recovery_fraction(ratio="0.5"):
        ci = intact_eval[ratio]["ce"]
        cd = damaged_eval[ratio]["ce"]
        cr = recovered_eval[ratio]["ce"]
        den = cd-ci
        return float((cd-cr)/den) if abs(den) > 1e-12 else float("nan")

    gate = {
        "damage_isolated_to_target": changed_by_damage == [target_name],
        "learning_isolated_to_target": changed_total == [target_name],
        "no_autograd_gradients": all(p.grad is None for p in model.parameters()),
        "damage_delta_ce_50": damaged_eval["0.5"]["ce"] - intact_eval["0.5"]["ce"],
        "recovery_fraction_ce_50": recovery_fraction("0.5"),
        "recovered_better_than_damaged_25": recovered_eval["0.25"]["ce"] < damaged_eval["0.25"]["ce"],
        "recovered_better_than_damaged_50": recovered_eval["0.5"]["ce"] < damaged_eval["0.5"]["ce"],
        "recovered_better_than_damaged_90": recovered_eval["0.9"]["ce"] < damaged_eval["0.9"]["ce"],
    }
    gate["pass"] = bool(
        gate["damage_isolated_to_target"] and
        gate["learning_isolated_to_target"] and
        gate["no_autograd_gradients"] and
        gate["damage_delta_ce_50"] > 0.03 and
        gate["recovery_fraction_ce_50"] > 0.50 and
        gate["recovered_better_than_damaged_25"] and
        gate["recovered_better_than_damaged_50"] and
        gate["recovered_better_than_damaged_90"]
    )

    results = {
        "stage": "R4 single-module recovery gate",
        "scientific_scope": "E29 H/g model; class conditioning retained as causal control",
        "target_parameter": target_name,
        "damage": "set W2 weight to zero; bias unchanged",
        "learning": "token-local antithetic forward perturbation + fast/slow MSDL state",
        "backprop_used": False,
        "optimizer_used": False,
        "learned_feedback_network_used": False,
        "rf_kind": a.rf_kind,
        "rf_features": a.features,
        "intact_eval": intact_eval,
        "damaged_eval": damaged_eval,
        "recovered_eval": recovered_eval,
        "damaged_weight_diag": damaged_diag,
        "recovered_weight_diag": recovered_diag,
        "generation_intact": gen_intact,
        "generation_damaged": gen_damaged,
        "generation_recovered": gen_recovered,
        "best_validation_ce_50": best_val,
        "gate": gate,
        "history": history,
    }
    (a.out / "metrics.json").write_text(json.dumps(results, indent=2))
    with (a.out / "history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader(); w.writerows(history)
    torch.save({"recovered_W2": target.clone(), "gold_W2_for_diagnostic_only": gold_w,
                "metrics": results}, a.out / "recovery_checkpoint.pt")
    print("FINAL " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
