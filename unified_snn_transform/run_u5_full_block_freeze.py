#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""U5 — canonical full-block freeze/integration gate.

Builds one reusable SNNTransformBlock from the accepted U4 operators, then
verifies in one run:
  1) exact canonical packaging reproduces the trained sparse reference;
  2) per-head Izh Q/K interface preserves held-out task output;
  3) event-cached Q/K/V (including Q) matches a dense-projection reference;
  4) actual token-projection evaluation reduction is measured;
  5) state carry/reset is deterministic;
  6) sparse-tail connectivity/activity accounting is preserved;
  7) one reusable checkpoint is emitted.

Mamba-like selective retention remains parked; v1 uses fixed retention=1.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import run_unified_gate as u
import run_u3_trained_retrieval as r
import run_u4b_task_sparse_block as u4b
import run_u4e_interface_calibration as u4e
import run_u4f_headaware_interface as u4f
import run_u4g_perhead_search as u4g
from snn_transform_block import SNNTransformBlock


def rel(a, b):
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12))


def cos(a, b):
    aa = a.reshape(-1)
    bb = b.reshape(-1)
    return float(F.cosine_similarity(aa[None], bb[None]).item())


def build_exact_from_reference(ref, args):
    block = SNNTransformBlock(
        d=args.d, heads=args.heads, features=args.features, ff=args.ff,
        out_density=.25, ff1_density=.25, ff2_density=.25,
        theta_init=args.theta_init, retention_alpha=1.0, seed=args.seed + 1001,
    )
    incompatible = block.load_state_dict(ref.block.state_dict(), strict=False)
    allowed_unexpected = {"gate.theta0", "gate.theta_event"}
    bad_unexpected = [x for x in incompatible.unexpected_keys if x not in allowed_unexpected]
    if incompatible.missing_keys or bad_unexpected:
        raise RuntimeError({"missing": incompatible.missing_keys, "unexpected": bad_unexpected})
    return block.eval()


@torch.no_grad()
def collect_head_calibration(ref, gen, args):
    aq, ak, pq0, pk0, v, y0 = u4f.collect_exact(
        ref, gen, args.seed + 4101, batches=5, batch=48
    )
    qitems, kitems, reports = [], [], []
    for h in range(args.heads):
        qp, _ = u4e.calibration_pool(aq[:, h], args.features, args.seed + 5000 + 10 * h, args.pool)
        kp, _ = u4e.calibration_pool(ak[:, h], args.features, args.seed + 5001 + 10 * h, args.pool)
        qrows, krows = [], []
        for i, it in enumerate(qp):
            p = u4g.phi_item(aq[:, h], it)
            yy = u4g.head_attention(p, pk0[:, h], v[:, h])
            qrows.append({"idx": i, "rel_l2": rel(yy, y0[:, h]), **it[3]})
        for i, it in enumerate(kp):
            p = u4g.phi_item(ak[:, h], it)
            yy = u4g.head_attention(pq0[:, h], p, v[:, h])
            krows.append({"idx": i, "rel_l2": rel(yy, y0[:, h]), **it[3]})
        qord = sorted(range(len(qp)), key=lambda i: qrows[i]["rel_l2"])[:args.pair_top]
        kord = sorted(range(len(kp)), key=lambda i: krows[i]["rel_l2"])[:args.pair_top]
        best = None
        pairs = []
        for qi in qord:
            pqi = u4g.phi_item(aq[:, h], qp[qi])
            for ki in kord:
                pki = u4g.phi_item(ak[:, h], kp[ki])
                yy = u4g.head_attention(pqi, pki, v[:, h])
                rr = rel(yy, y0[:, h])
                row = {"q_idx": qi, "k_idx": ki, "rel_l2": rr,
                       "q_spec": qp[qi][3], "k_spec": kp[ki][3]}
                pairs.append(row)
                score = rr + 5e-6 * (qp[qi][1]["window_ms"] + kp[ki][1]["window_ms"])
                if best is None or score < best[0]:
                    best = (score, qp[qi], kp[ki], row)
        _, qb, kb, selected = best
        qitems.append(qb)
        kitems.append(kb)
        reports.append({"head": h, "selected": selected,
                        "Q_screen": sorted(qrows, key=lambda z: z["rel_l2"])[:8],
                        "K_screen": sorted(krows, key=lambda z: z["rel_l2"])[:8]})
    return qitems, kitems, reports


@torch.no_grad()
def packaging_gate(ref, exact, gen, args, seed):
    g = torch.Generator(); g.manual_seed(seed)
    yr, yc, pr, pc = [], [], [], []
    for _ in range(args.eval_batches):
        x, _, _, _ = gen.sample(args.eval_batch, g)
        changed = torch.ones(x.shape[:2], dtype=torch.bool)
        a, _, _ = ref.block.step(x, None, changed)
        b, _, _ = exact.step(x, None, changed)
        yr.append(a); yc.append(b)
        pr.append(ref.readout(a[:, -1])); pc.append(ref.readout(b[:, -1]))
    yr, yc = torch.cat(yr), torch.cat(yc)
    pr, pc = torch.cat(pr), torch.cat(pc)
    return {"block_rel_l2": rel(yc, yr), "block_cosine": cos(yc, yr),
            "prediction_rel_l2": rel(pc, pr), "prediction_cosine": cos(pc, pr)}


@torch.no_grad()
def izh_gate(ref, exact, cand, gen, args, seed):
    g = torch.Generator(); g.manual_seed(seed)
    be, bi, pe, pi, targets = [], [], [], [], []
    Hnum = Hden = gnum = gden = 0.0
    acts = []
    for _ in range(args.eval_batches):
        x, tgt, _, _ = gen.sample(args.eval_batch, g)
        changed = torch.ones(x.shape[:2], dtype=torch.bool)
        ye, se, _ = exact.step(x, None, changed)
        yi, si, st = cand.step(x, None, changed)
        be.append(ye); bi.append(yi)
        pe.append(ref.readout(ye[:, -1])); pi.append(ref.readout(yi[:, -1])); targets.append(tgt)
        Hnum += float(torch.sum((si.H - se.H) ** 2)); Hden += float(torch.sum(se.H ** 2))
        gnum += float(torch.sum((si.g - se.g) ** 2)); gden += float(torch.sum(se.g ** 2))
        acts.append(st["hidden_activity"])
    be, bi = torch.cat(be), torch.cat(bi)
    pe, pi, targets = torch.cat(pe), torch.cat(pi), torch.cat(targets)
    return {
        "block_rel_l2": rel(bi, be), "block_cosine": cos(bi, be),
        "prediction_rel_l2": rel(pi, pe), "prediction_cosine": cos(pi, pe),
        "reference_task": r.metrics(pe, targets), "candidate_task": r.metrics(pi, targets),
        "H_rel_l2": float(np.sqrt(Hnum / max(Hden, 1e-30))),
        "g_rel_l2": float(np.sqrt(gnum / max(gden, 1e-30))),
        "hidden_activity": float(np.mean(acts)),
    }


@torch.no_grad()
def event_cache_gate(block, args, seed):
    xs, masks = u.sequence(seed, args.event_batch, args.tokens, args.d,
                           args.event_steps, args.change_frac)
    event = copy.deepcopy(block).eval()
    dense = copy.deepcopy(block).eval()
    se = sd = None
    max_out = max_H = max_g = max_pq = max_pk = max_v = 0.0
    event_evals = {"q": 0, "k": 0, "v": 0}
    dense_evals = {"q": 0, "k": 0, "v": 0}
    for x, m in zip(xs, masks):
        ye, se, ste = event.step(x, se, m, force_project_all=False)
        yd, sd, std = dense.step(x, sd, m, force_project_all=True)
        max_out = max(max_out, rel(ye, yd))
        max_H = max(max_H, rel(se.H, sd.H)); max_g = max(max_g, rel(se.g, sd.g))
        max_pq = max(max_pq, rel(se.pq, sd.pq)); max_pk = max(max_pk, rel(se.pk, sd.pk)); max_v = max(max_v, rel(se.v, sd.v))
        for key, statkey in (("q", "q_token_evals"), ("k", "k_token_evals"), ("v", "v_token_evals")):
            event_evals[key] += int(ste[statkey]); dense_evals[key] += int(std[statkey])
    red = {k: 1.0 - event_evals[k] / max(1, dense_evals[k]) for k in event_evals}
    return {"max_output_rel_l2": max_out, "max_H_rel_l2": max_H, "max_g_rel_l2": max_g,
            "max_pq_rel_l2": max_pq, "max_pk_rel_l2": max_pk, "max_v_rel_l2": max_v,
            "event_token_evals": event_evals, "dense_token_evals": dense_evals,
            "reduction": red}


@torch.no_grad()
def state_api_gate(block, gen, args, seed):
    g = torch.Generator(); g.manual_seed(seed)
    x0, _, _, _ = gen.sample(args.state_batch, g)
    x1 = x0.clone()
    nchange = max(1, round(args.tokens * args.change_frac))
    idx = torch.randperm(args.tokens, generator=g)[:nchange]
    replacement, _, _, _ = gen.sample(args.state_batch, g)
    x1[:, idx] = replacement[:, idx]

    y0a, s0a, _ = block.step(x0, None)
    y0b, s0b, _ = block.step(x0, None)
    fresh_repeat = rel(y0a, y0b)
    ycarry, scarry, stcarry = block.step(x1, s0a)
    yreset, sreset, _ = block.step(x1, s0a, reset=True)
    yfresh, sfresh, _ = block.step(x1, None)
    reset_match = rel(yreset, yfresh)
    reset_H_match = rel(sreset.H, sfresh.H)
    inferred_changed = stcarry["changed_tokens"]
    expected_changed = args.state_batch * nchange
    carry_state_change = rel(scarry.H, s0a.H)
    return {"fresh_repeat_rel_l2": fresh_repeat,
            "reset_vs_fresh_rel_l2": reset_match,
            "reset_H_vs_fresh_rel_l2": reset_H_match,
            "inferred_changed_tokens": inferred_changed,
            "expected_changed_tokens": expected_changed,
            "carry_H_change_rel_l2": carry_state_change,
            "state_shapes": {"H": list(scarry.H.shape), "g": list(scarry.g.shape),
                             "pq": list(scarry.pq.shape), "pk": list(scarry.pk.shape),
                             "v": list(scarry.v.shape), "weight": list(scarry.weight.shape)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="unified_snn_transform/run/U5")
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--tokens", type=int, default=16); ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--heads", type=int, default=2); ap.add_argument("--features", type=int, default=32)
    ap.add_argument("--ff", type=int, default=64); ap.add_argument("--value-dim", type=int, default=8)
    ap.add_argument("--theta-init", type=float, default=0.0)
    ap.add_argument("--train-steps", type=int, default=900); ap.add_argument("--train-batch", type=int, default=128)
    ap.add_argument("--eval-batches", type=int, default=12); ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--report-every", type=int, default=300)
    ap.add_argument("--pool", type=int, default=28); ap.add_argument("--pair-top", type=int, default=5)
    ap.add_argument("--event-batch", type=int, default=8); ap.add_argument("--event-steps", type=int, default=6)
    ap.add_argument("--change-frac", type=float, default=.25); ap.add_argument("--state-batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(); torch.set_num_threads(args.threads); u4b.seed_all(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    gen = r.AssocGenerator(args.tokens, args.d, args.value_dim, args.seed)

    # Train the accepted exact sparse architecture once.
    ref = u4b.SparseRetrievalModel(args, args.seed + 1001, .25, .25, .25)
    ref, train_history = u4b.train_model(ref, gen, args, args.seed + 2001)
    exact_task, exact_act, exact_compute = u4b.eval_sparse(ref, gen, args, args.seed + 3001)
    shuffled = u4b.shuffled_control(ref, gen, args, args.seed + 3001)

    # Package exact trained weights/masks into the reusable canonical class.
    exact = build_exact_from_reference(ref, args)
    packaging = packaging_gate(ref, exact, gen, args, args.seed + 3501)

    # Reproduce the accepted stable per-head Izh calibration without changing task weights.
    qitems, kitems, head_reports = collect_head_calibration(ref, gen, args)
    cand = copy.deepcopy(exact)
    q_specs = [dict(x[1]) for x in qitems]; q_decoders = [x[2].detach().clone() for x in qitems]
    k_specs = [dict(x[1]) for x in kitems]; k_decoders = [x[2].detach().clone() for x in kitems]
    cand.set_izh_interface(q_specs, q_decoders, k_specs, k_decoders)
    cand.eval()

    izh = izh_gate(ref, exact, cand, gen, args, args.seed + 8001)
    event_cache = event_cache_gate(exact, args, args.seed + 9001)
    state_api = state_api_gate(exact, gen, args, args.seed + 10001)
    tail = cand.synapse_account(izh["hidden_activity"])

    gates = {
        "canonical_packaging_block_rel_l2_le_1e6": packaging["block_rel_l2"] <= 1e-6,
        "canonical_packaging_prediction_rel_l2_le_1e6": packaging["prediction_rel_l2"] <= 1e-6,
        "izh_prediction_rel_l2_le_1pct": izh["prediction_rel_l2"] <= .01,
        "izh_prediction_cosine_ge_0p999": izh["prediction_cosine"] >= .999,
        "izh_task_sign_drop_le_0p02": izh["candidate_task"]["sign_accuracy"] >= izh["reference_task"]["sign_accuracy"] - .02,
        "izh_H_rel_l2_le_1pct": izh["H_rel_l2"] <= .01,
        "izh_g_rel_l2_le_1pct": izh["g_rel_l2"] <= .01,
        "event_cache_output_rel_l2_le_1e5": event_cache["max_output_rel_l2"] <= 1e-5,
        "event_cache_Q_reduction_ge_0p50": event_cache["reduction"]["q"] >= .50,
        "event_cache_K_reduction_ge_0p50": event_cache["reduction"]["k"] >= .50,
        "event_cache_V_reduction_ge_0p50": event_cache["reduction"]["v"] >= .50,
        "reset_matches_fresh_le_1e7": state_api["reset_vs_fresh_rel_l2"] <= 1e-7,
        "changed_inference_exact": state_api["inferred_changed_tokens"] == state_api["expected_changed_tokens"],
        "tail_effective_fraction_le_0p30": tail["effective_fraction"] <= .30,
        "task_sign_accuracy_ge_0p95": exact_task["sign_accuracy"] >= .95,
        "binding_control_drop_ge_0p20": exact_task["sign_accuracy"] - shuffled["sign_accuracy"] >= .20,
    }
    PASS = all(gates.values())

    checkpoint = {
        "format": "SNNTransformBlock-v1",
        "config": {"d": args.d, "heads": args.heads, "features": args.features, "ff": args.ff,
                   "out_density": .25, "ff1_density": .25, "ff2_density": .25,
                   "theta_init": args.theta_init, "retention_alpha": 1.0, "seed": args.seed + 1001},
        "model_state_dict": cand.state_dict(),
        "q_specs": q_specs, "q_decoders": q_decoders,
        "k_specs": k_specs, "k_decoders": k_decoders,
        "readout_state_dict": ref.readout.state_dict(),
        "scientific_scope": {
            "selective_retention": "parked; fixed retention only in v1",
            "hardware": "cloud architectural freeze; no SpiNNaker timing/energy claim",
            "izh_interface": "per-head Izh membrane-trajectory readout; not strict spike-only communication",
        },
    }
    torch.save(checkpoint, out / "SNNTransformBlock_v1.pt")

    result = {
        "stage": "UNIFIED_U5_CANONICAL_FULL_BLOCK_FREEZE",
        "PASS": PASS,
        "config": vars(args),
        "architecture": {
            "api": "y, state = SNNTransformBlock(x, state)",
            "QKV": "event-cached token projections",
            "feature_map": "per-head calibrated Izh Q/K dynamics",
            "attention": "persistent normalized H/g linear attention",
            "output_projection_density": .25,
            "ff1_density": .25, "ff2_density": .25,
            "hidden_population": "thresholded exact-zero ReLU",
            "retention": "fixed alpha=1; selective/Mamba-like module parked",
        },
        "exact_task": exact_task, "shuffled_control": shuffled,
        "binding_sign_drop": exact_task["sign_accuracy"] - shuffled["sign_accuracy"],
        "packaging": packaging, "izh_integration": izh,
        "event_cache": event_cache, "state_api": state_api, "tail_compute": tail,
        "selected_Q": [x[3] for x in qitems], "selected_K": [x[3] for x in kitems],
        "head_reports": head_reports, "train_history": train_history,
        "gates": gates,
        "checkpoint": "SNNTransformBlock_v1.pt",
        "decision": ("freeze SNNTransformBlock-v1 and move to downstream task plug-in gate" if PASS
                     else "do not freeze; localize the failing full-block gate before scaling"),
    }
    (out / "U5_RESULTS.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"stage": result["stage"], "PASS": PASS, "packaging": packaging,
                      "izh_integration": izh, "event_cache": event_cache,
                      "state_api": state_api, "tail_compute": tail,
                      "gates": gates, "decision": result["decision"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
