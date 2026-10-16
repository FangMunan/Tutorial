#!/usr/bin/env python3
"""R5A: joint two-matrix FFN recovery gate for SNN_Transform.

R4 established that a single existing forward matrix (final-block W2) can be
functionally recovered without backpropagation or a learned feedback network.
R5A increases the credit-assignment depth by one local nonlinear stage:

    x_attn -> W1 -> GELU -> W2 -> residual/LN -> head

Only the two existing final-block FFN weight matrices are plastic.  All other
forward parameters remain at the E29 B0->H/g transferred solution.

Controlled lesion:
    W1 <- 0
    W2 <- 0.5 * W2_gold

W2 is deliberately not set to zero: if W1 and W2 are both exactly zero, the
multiplicative path has a dead symmetry and W1 perturbations cannot influence
loss.  The lesion is therefore severe but identifiable.

Learning uses local antithetic forward perturbations and fast/slow synaptic
states. No autograd backward call, torch optimizer, or learned inverse network
is used.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "snn_transform_e29"))
sys.path.insert(0, str(ROOT / "snn_transform_r4"))

import maskgit_b0_mnist012 as b0
import e29_forward_equivalence as e29
import r4_single_module_recovery as r4


@torch.no_grad()
def final_xattn(model: e29.HgB0Model, tokens: torch.Tensor, labels: torch.Tensor):
    z = model.tok(tokens) + model.pos + model.cls(labels)[:, None]
    for block in model.blocks[:-1]:
        z = block(z)
    block = model.blocks[-1]
    return block.n1(z + block.attn(z))


@torch.no_grad()
def tail_from_xattn(model: e29.HgB0Model, x_attn: torch.Tensor,
                    pre1_perturb: torch.Tensor | None = None,
                    w2out_perturb: torch.Tensor | None = None):
    block = model.blocks[-1]
    pre1 = block.f1(x_attn)
    if pre1_perturb is not None:
        pre1 = pre1 + pre1_perturb
    h = F.gelu(pre1)
    u = block.f2(h)
    if w2out_perturb is not None:
        u = u + w2out_perturb
    z = block.n2(x_attn + u)
    z = model.ln(F.gelu(model.dense(z)))
    logits = F.linear(z, model.tok.weight[:b0.K], model.bias)
    return logits, h


def clip_inplace(g: torch.Tensor, limit: float):
    n = float(g.norm())
    if limit > 0 and n > limit:
        g.mul_(limit / (n + 1e-12))
        n = limit
    return n


@torch.no_grad()
def r5a_step(model, t, y, inp, mask,
             sigma1, sigma2,
             f1_fast, f1_slow, f2_fast, f2_slow,
             lr1, lr2, rho_fast, rho_slow, mix_fast,
             clip1, clip2, generator):
    """Update final-block W1 and W2 from two local node-perturbation probes."""
    block = model.blocks[-1]
    x = final_xattn(model, inp, y)

    # ---- W1 probe: perturb hidden preactivation before GELU ----
    xi1 = torch.randn((len(t), b0.L, e29.FF), generator=generator)
    lp, _ = tail_from_xattn(model, x, pre1_perturb=sigma1 * xi1)
    lm, _ = tail_from_xattn(model, x, pre1_perturb=-sigma1 * xi1)
    ce1p = F.cross_entropy(lp.transpose(1, 2), t, reduction="none")
    ce1m = F.cross_entropy(lm.transpose(1, 2), t, reduction="none")
    m1 = (ce1p - ce1m) / (2.0 * sigma1)

    sel = mask.float()
    n = sel.sum().clamp_min(1.0)
    # local eligibility xi_hidden x x_attn^T
    g1 = torch.einsum("bn,bnf,bnd->fd", m1 * sel, xi1, x) / n
    gn1 = clip_inplace(g1, clip1)

    # ---- W2 probe: perturb FFN output after W2 ----
    base_logits, h = tail_from_xattn(model, x)
    del base_logits
    xi2 = torch.randn((len(t), b0.L, e29.D), generator=generator)
    lp2, _ = tail_from_xattn(model, x, w2out_perturb=sigma2 * xi2)
    lm2, _ = tail_from_xattn(model, x, w2out_perturb=-sigma2 * xi2)
    ce2p = F.cross_entropy(lp2.transpose(1, 2), t, reduction="none")
    ce2m = F.cross_entropy(lm2.transpose(1, 2), t, reduction="none")
    m2 = (ce2p - ce2m) / (2.0 * sigma2)
    g2 = torch.einsum("bn,bnd,bnf->df", m2 * sel, xi2, h) / n
    gn2 = clip_inplace(g2, clip2)

    # Multistate local learning dynamics. These are state variables, not a
    # second neural network.
    f1_fast.mul_(rho_fast).add_(g1, alpha=1.0-rho_fast)
    f1_slow.mul_(rho_slow).add_(g1, alpha=1.0-rho_slow)
    f2_fast.mul_(rho_fast).add_(g2, alpha=1.0-rho_fast)
    f2_slow.mul_(rho_slow).add_(g2, alpha=1.0-rho_slow)

    u1 = mix_fast * f1_fast + (1.0-mix_fast) * f1_slow
    u2 = mix_fast * f2_fast + (1.0-mix_fast) * f2_slow
    block.f1.weight.add_(u1, alpha=-lr1)
    block.f2.weight.add_(u2, alpha=-lr2)

    pert_ce = 0.25 * (
        ce1p[mask].mean() + ce1m[mask].mean() +
        ce2p[mask].mean() + ce2m[mask].mean()
    )
    return float(pert_ce), gn1, gn2, float(u1.norm()), float(u2.norm())


@torch.no_grad()
def pair_diag(model, gold1, gold2):
    return {
        "W1": r4.weight_diag(model.blocks[-1].f1.weight, gold1),
        "W2": r4.weight_diag(model.blocks[-1].f2.weight, gold2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--codebook", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--features", type=int, default=256)
    ap.add_argument("--rf-kind", type=str, default="p3norm")
    ap.add_argument("--rf-seed", type=int, default=29257)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr1", type=float, default=0.10)
    ap.add_argument("--lr2", type=float, default=0.16)
    ap.add_argument("--sigma1", type=float, default=0.05)
    ap.add_argument("--sigma2", type=float, default=0.05)
    ap.add_argument("--rho-fast", type=float, default=0.60)
    ap.add_argument("--rho-slow", type=float, default=0.95)
    ap.add_argument("--mix-fast", type=float, default=0.70)
    ap.add_argument("--clip1", type=float, default=1.0)
    ap.add_argument("--clip2", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=5501)
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
    e29.transfer_b0(exact, model); model.eval(); r4.freeze_all(model)

    n1 = "blocks.3.f1.weight"
    n2 = "blocks.3.f2.weight"
    allowed = {n1, n2}
    gold1 = model.blocks[-1].f1.weight.detach().clone()
    gold2 = model.blocks[-1].f2.weight.detach().clone()
    hashes_intact = r4.model_hashes(model)

    ratios = (.25, .5, .9, 1.0)
    intact_eval = r4.eval_mask_ratios(model, te, yte, ratios)
    clf = r4.external_classifier(xtr, ytr)
    gen_intact = r4.generate_snapshot(model, codebook, clf, a.out, "intact", a.seed+10)

    # Severe but identifiable joint lesion.
    model.blocks[-1].f1.weight.zero_()
    model.blocks[-1].f2.weight.mul_(0.5)
    damaged_eval = r4.eval_mask_ratios(model, te, yte, ratios)
    gen_damaged = r4.generate_snapshot(model, codebook, clf, a.out, "damaged_W1zero_W2half", a.seed+10)
    damaged_diag = pair_diag(model, gold1, gold2)

    hashes_damaged = r4.model_hashes(model)
    changed_damage = {n for n in hashes_intact if hashes_intact[n] != hashes_damaged[n]}
    if changed_damage != allowed:
        raise RuntimeError(f"Damage isolation failure: {sorted(changed_damage)}")

    f1_fast = torch.zeros_like(model.blocks[-1].f1.weight)
    f1_slow = torch.zeros_like(f1_fast)
    f2_fast = torch.zeros_like(model.blocks[-1].f2.weight)
    f2_slow = torch.zeros_like(f2_fast)
    rng = np.random.default_rng(a.seed + 77)
    tg = torch.Generator(device="cpu"); tg.manual_seed(a.seed + 99)

    history = []
    best_val = float("inf")
    best1 = model.blocks[-1].f1.weight.clone()
    best2 = model.blocks[-1].f2.weight.clone()

    for ep in range(1, a.epochs + 1):
        order = rng.permutation(len(tr))
        losses=[]; g1s=[]; g2s=[]; u1s=[]; u2s=[]
        for s in range(0, len(order), a.batch):
            idx = order[s:s+a.batch]
            t, y, inp, mask = r4.masked_random_batch(tr, ytr, idx)
            z = r5a_step(
                model, t, y, inp, mask,
                a.sigma1, a.sigma2,
                f1_fast, f1_slow, f2_fast, f2_slow,
                a.lr1, a.lr2, a.rho_fast, a.rho_slow, a.mix_fast,
                a.clip1, a.clip2, tg)
            losses.append(z[0]); g1s.append(z[1]); g2s.append(z[2]); u1s.append(z[3]); u2s.append(z[4])

        val = r4.eval_mask_ratios(model, va, yva, (.5,), batch=256)["0.5"]["ce"]
        test50 = r4.eval_mask_ratios(model, te, yte, (.5,), batch=256, limit=1024)["0.5"]["ce"]
        d = pair_diag(model, gold1, gold2)
        rec = {
            "epoch": ep,
            "mean_train_perturbed_ce": float(np.mean(losses)),
            "mean_g1_norm": float(np.mean(g1s)),
            "mean_g2_norm": float(np.mean(g2s)),
            "mean_u1_norm": float(np.mean(u1s)),
            "mean_u2_norm": float(np.mean(u2s)),
            "val_ce_50": val,
            "test_ce_50_first1024": test50,
            "W1_rel_to_gold": d["W1"]["relative_distance_to_gold"],
            "W1_cos_to_gold": d["W1"]["cosine_to_gold"],
            "W2_rel_to_gold": d["W2"]["relative_distance_to_gold"],
            "W2_cos_to_gold": d["W2"]["cosine_to_gold"],
        }
        history.append(rec); print(json.dumps(rec), flush=True)
        if val < best_val:
            best_val = val
            best1 = model.blocks[-1].f1.weight.clone()
            best2 = model.blocks[-1].f2.weight.clone()

    model.blocks[-1].f1.weight.copy_(best1)
    model.blocks[-1].f2.weight.copy_(best2)
    recovered_eval = r4.eval_mask_ratios(model, te, yte, ratios)
    recovered_diag = pair_diag(model, gold1, gold2)
    gen_recovered = r4.generate_snapshot(model, codebook, clf, a.out, "recovered", a.seed+10)

    hashes_final = r4.model_hashes(model)
    changed_final = {n for n in hashes_intact if hashes_intact[n] != hashes_final[n]}
    if changed_final != allowed:
        raise RuntimeError(f"Learning isolation failure: {sorted(changed_final)}")
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Gradient tensor exists in gradient-free R5A")

    def frac(r):
        ci=intact_eval[r]["ce"]; cd=damaged_eval[r]["ce"]; cr=recovered_eval[r]["ce"]
        return float((cd-cr)/(cd-ci)) if abs(cd-ci)>1e-12 else float("nan")

    gate = {
        "damage_isolated_to_W1_W2": changed_damage == allowed,
        "learning_isolated_to_W1_W2": changed_final == allowed,
        "no_autograd_gradients": all(p.grad is None for p in model.parameters()),
        "damage_delta_ce_50": damaged_eval["0.5"]["ce"]-intact_eval["0.5"]["ce"],
        "recovery_fraction_ce_25": frac("0.25"),
        "recovery_fraction_ce_50": frac("0.5"),
        "recovery_fraction_ce_90": frac("0.9"),
        "recovered_better_than_damaged_25": recovered_eval["0.25"]["ce"] < damaged_eval["0.25"]["ce"],
        "recovered_better_than_damaged_50": recovered_eval["0.5"]["ce"] < damaged_eval["0.5"]["ce"],
        "recovered_better_than_damaged_90": recovered_eval["0.9"]["ce"] < damaged_eval["0.9"]["ce"],
        "W1_became_nonzero": float(model.blocks[-1].f1.weight.norm()) > 1e-6,
        "W2_changed_from_half_gold": not torch.equal(model.blocks[-1].f2.weight, 0.5*gold2),
    }
    gate["pass"] = bool(
        gate["damage_isolated_to_W1_W2"] and gate["learning_isolated_to_W1_W2"] and
        gate["no_autograd_gradients"] and gate["damage_delta_ce_50"] > 0.05 and
        gate["recovery_fraction_ce_50"] > 0.50 and
        gate["recovered_better_than_damaged_25"] and gate["recovered_better_than_damaged_50"] and
        gate["recovered_better_than_damaged_90"] and gate["W1_became_nonzero"] and gate["W2_changed_from_half_gold"]
    )

    results = {
        "stage": "R5A joint final-block FFN matrix recovery",
        "scientific_scope": "E29 H/g model; class condition retained; only final-block W1/W2 matrices plastic",
        "damage": {"W1":"zero", "W2":"0.5 * gold", "biases":"unchanged"},
        "learning": "two local antithetic node perturbations + fast/slow multistate synaptic dynamics",
        "backprop_used": False,
        "torch_optimizer_used": False,
        "learned_feedback_network_used": False,
        "rf_kind": a.rf_kind,
        "rf_features": a.features,
        "intact_eval": intact_eval,
        "damaged_eval": damaged_eval,
        "recovered_eval": recovered_eval,
        "damaged_diag": damaged_diag,
        "recovered_diag": recovered_diag,
        "generation_intact": gen_intact,
        "generation_damaged": gen_damaged,
        "generation_recovered": gen_recovered,
        "best_validation_ce_50": best_val,
        "gate": gate,
        "history": history,
    }
    (a.out/"metrics.json").write_text(json.dumps(results, indent=2))
    with (a.out/"history.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=history[0].keys()); w.writeheader(); w.writerows(history)
    torch.save({"recovered_W1":model.blocks[-1].f1.weight.clone(),
                "recovered_W2":model.blocks[-1].f2.weight.clone(),
                "gold_W1_diagnostic_only":gold1,"gold_W2_diagnostic_only":gold2,
                "metrics":results}, a.out/"recovery_checkpoint.pt")
    print("FINAL "+json.dumps(results), flush=True)

if __name__ == "__main__":
    main()
