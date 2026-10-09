#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H1-B2.0 — error localization for the current Izh dynamical feature map.

Purpose
-------
Do NOT optimize anything yet. Reuse the frozen H1-B checkpoint and selected Izh
mapping, then localize where the remaining approximation error comes from.

Audits:
  1) per layer / head / Q-or-K / generation-step scalar phi error and clamp rate;
  2) one-layer-at-a-time Izh replacement, fresh and persistent selective logits;
  3) Q-only versus K-only Izh replacement across all layers;
  4) full-Izh control to verify consistency with H1-B.

The output is diagnostic only. No architecture or parameter changes are made.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

import h0_base as h0
import h1_sparse_fusion as h1
import h1b_izh_fusion as h1b


class SideSelectivePhi(nn.Module):
    """Use Izh only on Q or only on K; exact feature map on the other side."""
    def __init__(self, exact_phi: h0.PositiveRandomFeatures, spec: Dict,
                 decoder: torch.Tensor, use_q: bool, use_k: bool):
        super().__init__()
        self.exact = copy.deepcopy(exact_phi)
        self.izh = h1b.IzhDynamicFeatures(copy.deepcopy(exact_phi),
                                          bias=spec["bias"], gain=spec["gain"],
                                          window_ms=spec["window_ms"], decoder=decoder)
        self.use_q = bool(use_q)
        self.use_k = bool(use_k)
        self.features = int(exact_phi.features)
        self.head_dim = int(exact_phi.head_dim)
        self.register_buffer("omega", exact_phi.omega.detach().clone())

    def forward(self, x: torch.Tensor, *, is_query: bool) -> torch.Tensor:
        if (is_query and self.use_q) or ((not is_query) and self.use_k):
            return self.izh(x, is_query=is_query)
        return self.exact(x, is_query=is_query)


def load_reused_models(args, device):
    reuse = args.reuse
    if not (reuse / "linear_base" / "best.pt").exists():
        raise FileNotFoundError(f"Missing {reuse/'linear_base'/'best.pt'}")
    if not (reuse / "selective_gate.pt").exists():
        raise FileNotFoundError(f"Missing {reuse/'selective_gate.pt'}")
    if not (reuse / "izh_feature_map.pt").exists():
        raise FileNotFoundError(f"Missing {reuse/'izh_feature_map.pt'}")

    base = h1b.build_linear(args, device, False)
    ck = torch.load(reuse / "linear_base" / "best.pt", map_location=device, weights_only=False)
    base.load_state_dict(ck["model"])
    base.eval()

    dense_sel = h1b.build_linear(args, device, True)
    sg = torch.load(reuse / "selective_gate.pt", map_location=device, weights_only=False)
    dense_sel.load_state_dict(sg["model"])
    dense_sel.eval()

    fmap = torch.load(reuse / "izh_feature_map.pt", map_location=device, weights_only=False)
    spec = {k: float(v) if k != "window_ms" else int(v) for k, v in fmap["spec"].items()}
    decoder = fmap["decoder"].to(device).float()
    return base, dense_sel, spec, decoder


def load_validation_tokens(args):
    if args.data is not None:
        xall, yall, _, _ = h0.load_mnist_npz(args.data)
    else:
        xall, yall, _, _ = h0.load_mnist_keras()
    x012, y012 = h0.filter_classes(xall, yall)
    x012, y012 = h0.subset_per_class(x012, y012, args.n_per_class, args.seed)
    _, _, xva, yva = h0.stratified_split(x012, y012, .1, args.seed)
    yva = h0.map_labels_012(yva)
    cent = np.load(args.reuse / "codebook.npy")
    va = h0.encode_tokens(xva, cent)
    return va, yva


@torch.no_grad()
def group_scalar_diagnostics(model, tokens, labels, device, spec, decoder,
                             seed=17001, steps=6, n=24, samples_per_group=1000):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tokens), size=min(n, len(tokens)), replace=False)
    tok = torch.from_numpy(tokens[idx]).to(device)
    y = torch.from_numpy(labels[idx]).to(device)
    gen = torch.Generator(device=device); gen.manual_seed(seed + 17)
    canvases = h0.teacher_mask_sequence(tok, steps, gen)

    rows = []
    for ti, cur in enumerate(canvases):
        z = model.embed(cur, y)
        for li, block in enumerate(model.blocks):
            attn = block.attn
            for side, isq in (("Q", True), ("K", False)):
                aa = h1b.exact_stable(attn, z, isq)  # [B,H,N,M]
                for hi in range(model.heads):
                    flat = aa[:, hi].reshape(-1)
                    g = torch.Generator(device=device)
                    g.manual_seed(seed + 100000*ti + 10000*li + 1000*hi + (1 if isq else 2))
                    ns = min(int(samples_per_group), int(flat.numel()))
                    take = torch.randint(0, flat.numel(), (ns,), generator=g, device=device)
                    a = flat[take]
                    target = h1b.target_phi(a, model.features)
                    met = h1b.scalar_metrics(a, target, decoder, **spec)
                    qs = torch.quantile(a.float(), torch.tensor([.01, .5, .99], device=device))
                    I = float(spec["bias"]) + float(spec["gain"]) * a.float()
                    iq = torch.quantile(I, torch.tensor([.01, .5, .99], device=device))
                    rows.append({
                        "step": ti, "layer": li, "head": hi, "side": side,
                        "a_q01": float(qs[0]), "a_median": float(qs[1]), "a_q99": float(qs[2]),
                        "I_q01": float(iq[0]), "I_median": float(iq[1]), "I_q99": float(iq[2]),
                        **met,
                    })
            z = block(z)
    return rows


def install_layer_izh(model, layer: int, spec: Dict, decoder: torch.Tensor):
    old = model.blocks[layer].attn.phi
    model.blocks[layer].attn.phi = h1b.IzhDynamicFeatures(
        old, bias=spec["bias"], gain=spec["gain"],
        window_ms=spec["window_ms"], decoder=decoder
    ).to(next(model.parameters()).device)
    return model


def install_side_izh_all(model, spec: Dict, decoder: torch.Tensor, use_q: bool, use_k: bool):
    for block in model.blocks:
        old = block.attn.phi
        block.attn.phi = SideSelectivePhi(old, spec, decoder, use_q, use_k).to(next(model.parameters()).device)
    return model


def summarize_group_rows(rows):
    # Aggregate by layer/side and identify worst individual groups.
    agg = {}
    for r in rows:
        key = (r["layer"], r["side"])
        agg.setdefault(key, []).append(r)
    layer_side = []
    for (li, side), rr in sorted(agg.items()):
        layer_side.append({
            "layer": li, "side": side,
            "scalar_rel_l2_mean": float(np.mean([x["scalar_rel_l2"] for x in rr])),
            "scalar_rel_l2_max": float(np.max([x["scalar_rel_l2"] for x in rr])),
            "clamp_fraction_mean": float(np.mean([x["clamp_fraction"] for x in rr])),
            "clamp_fraction_max": float(np.max([x["clamp_fraction"] for x in rr])),
        })
    worst = sorted(rows, key=lambda x: x["scalar_rel_l2"], reverse=True)[:12]
    return layer_side, worst


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("H1B2_DIAGNOSTICS"))
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--n-per-class", type=int, default=2000)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=512)
    ap.add_argument("--features", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    h0.seed_all(args.seed); torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    va, yva = load_validation_tokens(args)
    base, dense_sel, spec, decoder = load_reused_models(args, device)

    # 1) Scalar localization across layer/head/Q-K/generation step.
    group_rows = group_scalar_diagnostics(base, va, yva, device, spec, decoder,
                                          seed=args.seed+17001, steps=6, n=24,
                                          samples_per_group=1000)
    write_csv(args.out / "scalar_error_by_group.csv", group_rows)
    layer_side, worst_groups = summarize_group_rows(group_rows)
    write_csv(args.out / "scalar_error_by_layer_side.csv", layer_side)

    # 2) One layer at a time: fresh + persistent selective network sensitivity.
    layer_probes = []
    exact_sparse = copy.deepcopy(dense_sel); h1.enable_sparse_block1(exact_sparse)
    for li in range(args.layers):
        cand_f = copy.deepcopy(base); install_layer_izh(cand_f, li, spec, decoder)
        fresh = h1b.fresh_logit_error(base, cand_f, va, yva, device,
                                      args.seed+18000+li, n=32)
        cand_p = copy.deepcopy(dense_sel); h1.enable_sparse_block1(cand_p)
        install_layer_izh(cand_p, li, spec, decoder)
        pers = h1b.persistent_logit_error(exact_sparse, cand_p, va, yva, device,
                                         args.seed+18100+li, steps=6, n=16, mode="selective")
        row = {
            "layer": li,
            "fresh_rel_l2": fresh["rel_l2"],
            "fresh_argmax_agreement": fresh["argmax_agreement"],
            "fresh_max_abs": fresh["max_abs"],
            "persistent_max_rel_l2": pers["max_rel_l2"],
            "persistent_min_argmax_agreement": pers["min_argmax_agreement"],
        }
        layer_probes.append(row)
        print("LAYER_PROBE", json.dumps(row), flush=True)
    write_csv(args.out / "one_layer_replacement.csv", layer_probes)

    # 3) Q-only and K-only across all layers.
    side_probes = {}
    for name, uq, uk in (("Q_only", True, False), ("K_only", False, True), ("QK_all", True, True)):
        cand_f = copy.deepcopy(base); install_side_izh_all(cand_f, spec, decoder, uq, uk)
        fresh = h1b.fresh_logit_error(base, cand_f, va, yva, device,
                                      args.seed+19000+(1 if uq else 0)+(2 if uk else 0), n=32)
        cand_p = copy.deepcopy(dense_sel); h1.enable_sparse_block1(cand_p)
        install_side_izh_all(cand_p, spec, decoder, uq, uk)
        pers = h1b.persistent_logit_error(exact_sparse, cand_p, va, yva, device,
                                         args.seed+19100+(1 if uq else 0)+(2 if uk else 0),
                                         steps=6, n=16, mode="selective")
        side_probes[name] = {"fresh": fresh, "persistent": pers}
        print("SIDE_PROBE", name, json.dumps(side_probes[name]), flush=True)

    # 4) Compact decision diagnostics, not an optimization decision yet.
    worst_layer = max(layer_probes, key=lambda x: x["fresh_rel_l2"])
    best_layer = min(layer_probes, key=lambda x: x["fresh_rel_l2"])
    qerr = side_probes["Q_only"]["fresh"]["rel_l2"]
    kerr = side_probes["K_only"]["fresh"]["rel_l2"]
    side_ratio = max(qerr, kerr) / max(1e-12, min(qerr, kerr))
    layer_ratio = worst_layer["fresh_rel_l2"] / max(1e-12, best_layer["fresh_rel_l2"])

    results = {
        "stage": "H1B2_0_ERROR_LOCALIZATION",
        "device": str(device),
        "reused_h1b": str(args.reuse),
        "selected_izh_spec": spec,
        "layer_side_scalar_summary": layer_side,
        "worst_scalar_groups": worst_groups,
        "one_layer_replacement": layer_probes,
        "side_replacement": side_probes,
        "diagnostic_ratios": {
            "Q_vs_K_fresh_error_ratio": float(side_ratio),
            "worst_vs_best_layer_fresh_error_ratio": float(layer_ratio),
            "worst_layer": int(worst_layer["layer"]),
            "best_layer": int(best_layer["layer"]),
        },
        "interpretation_rule": {
            "QK_split_candidate_if_ratio_ge": 1.5,
            "layer_specific_candidate_if_ratio_ge": 1.5,
            "note": "These thresholds only choose the next experiment; they do not establish significance."
        }
    }
    h0.json_dump(args.out / "H1B2_DIAGNOSTICS.json", results)
    print("H1B2_DIAGNOSTICS_DONE", json.dumps({
        "diagnostic_ratios": results["diagnostic_ratios"],
        "Q_only": side_probes["Q_only"]["fresh"],
        "K_only": side_probes["K_only"]["fresh"],
        "QK_all": side_probes["QK_all"]["fresh"],
        "one_layer": layer_probes,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
