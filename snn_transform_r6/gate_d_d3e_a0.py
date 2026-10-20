#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D3E-A0 — exact Gate-A provenance recovery smoke test.

This diagnostic does *not* run the four D3E intervention arms.  It separates
historical reference-generation provenance from the historical Gate-A causal
common-random-number (CRN) replay and hard-fails unless all frozen provenance
checks reproduce.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "snn_transform_p3_image"))
sys.path.insert(0, str(HERE))

import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
import r6b3_context_epoch_training as trainer
import r6b4_full_forward_effectiveness_test as effectiveness
from maskgit_input_representation import build_input_embeddings, context_from_arr_unconditional

MASK = 64
V = 64
DOM = 61
ROUNDS = 3
BASE_GEN_SEED = 27311
GATE_A_TEMP = 8.0
VALIDATION_SEED = 11553
CAUSAL_UNIFORM_SEED = 9917311
CAUSAL_GUMBEL_SEED = 9917312
TOL = 1e-7

EXPECTED = {
    "validation_ids_sha256": "42b3f11d7ed11c773e268fdef52634e4ebb30b07b8a87af3c3c2ba2dd9c7d942",
    "gt_tokens_sha256": "4d4603155cdd25c18fb9d93eb9f3afff6e20123dfba8b5ae008ebebc013c57d1",
    "shared_uniform_sha256": "00231d5473188090945c594a09bb535e297a339a5a02b8fe7657c5fa8c88e8c0",
    "shared_gumbel_sha256": "98d034538d1e9e5c3c53a896c979c2732160807e7eafdb6dc6e242ea0132b5e9",
    "terminal_ndarray_state_sha256": "5b69d8fa4a8b1fef2939230014d002731bf4814aa55bfc542eee9edcddc20dd1",
    "effects": {
        "F61": 0.021631765465585838,
        "D_non61": 0.12395075840895527,
        "C61": -0.00265625,
        "CE": 0.17817003568548723,
        "delta_P61": -0.0010512298084766059,
    },
}

METRIC_NAMES = (
    "ce", "acc", "ce61", "acc61", "ce_non61", "acc_non61", "mean_p61",
    "p61_on_gt61", "p61_on_non61", "proposal61_fraction", "context61_fraction",
    "sampled_accuracy", "sampled_confidence", "brier",
)
IX = {k: i for i, k in enumerate(METRIC_NAMES)}


def raw_array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ndarray_state_sha256(obj) -> str:
    """Canonical hash of ndarray-valued system state only."""
    h = hashlib.sha256()
    for k in sorted(vars(obj)):
        v = getattr(obj, k)
        if not isinstance(v, np.ndarray):
            continue
        a = np.ascontiguousarray(v)
        h.update(k.encode("utf-8") + b"\0")
        h.update(str(a.dtype).encode("utf-8"))
        h.update(str(a.shape).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


def inverse_cdf_rows(p: np.ndarray, u: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    c = np.cumsum(p, axis=1)
    c[:, -1] = 1.0
    return np.minimum(
        np.sum(c < np.asarray(u, dtype=np.float64)[:, None], axis=1),
        p.shape[1] - 1,
    ).astype(np.int64)


def model_dict(s):
    return trainer.model_dict_from_system(s)


def predict_round_causal(pred, cur, tok, pos, masked):
    xn, xraw = context_from_arr_unconditional(np.asarray(cur, dtype=np.int64), tok, pos)
    pred.rebuild(xn)
    p = np.empty((len(masked), V), dtype=np.float64)
    for qi, t0 in enumerate(np.asarray(masked, dtype=np.int64)):
        t = int(t0)
        a, _q, _num, _den = pred._read(t)
        _u, _h, _rf, pp = pred._tail(a, np.asarray(xraw[t], dtype=np.float64))
        p[qi] = pp
    p /= p.sum(axis=1, keepdims=True)
    return p


def one_metrics(p, gt, sampled, cur):
    p = np.asarray(p, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.int64)
    sampled = np.asarray(sampled, dtype=np.int64)
    truep = p[np.arange(len(gt)), gt]
    ce = -np.log(np.clip(truep, 1e-12, 1.0))
    arg = np.argmax(p, axis=1)
    acc = arg == gt
    is61 = gt == DOM
    non = ~is61
    sconf = p[np.arange(len(sampled)), sampled]
    scorr = sampled == gt
    one = np.zeros_like(p)
    one[np.arange(len(gt)), gt] = 1.0
    vis = np.asarray(cur) != MASK

    def m(x):
        return float(np.mean(x)) if np.size(x) else np.nan

    return np.asarray([
        m(ce), m(acc), m(ce[is61]), m(acc[is61]), m(ce[non]), m(acc[non]),
        m(p[:, DOM]), m(p[is61, DOM]), m(p[non, DOM]), m(sampled == DOM),
        m(np.asarray(cur)[vis] == DOM), m(scorr), m(sconf),
        m(np.sum((p - one) ** 2, axis=1)),
    ], dtype=np.float64)


def round3_effects(self_metrics: np.ndarray, teacher_metrics: np.ndarray):
    s = np.asarray(self_metrics)[:, 2]
    t = np.asarray(teacher_metrics)[:, 2]
    arrays = {
        "CE": s[:, IX["ce"]] - t[:, IX["ce"]],
        "D_non61": s[:, IX["ce_non61"]] - t[:, IX["ce_non61"]],
        "F61": s[:, IX["p61_on_non61"]] - t[:, IX["p61_on_non61"]],
        "C61": s[:, IX["context61_fraction"]] - t[:, IX["context61_fraction"]],
        "delta_P61": s[:, IX["mean_p61"]] - t[:, IX["mean_p61"]],
    }
    return {k: float(np.mean(v[np.isfinite(v)])) for k, v in arrays.items()}


def reconstruct_baseline(source, tokens, indices, tok0, pos0, train_rng_state):
    baseline = pickle.loads(pickle.dumps(source, pickle.HIGHEST_PROTOCOL))
    ones = np.ones(source.c.V, dtype=np.float64)
    result = effectiveness.train_arm(
        "r6b3_frozen_new_params", baseline, "baseline", tokens, indices,
        tok0, pos0, ones, train_rng_state, report_every=256,
    )
    return baseline, result


def validate_trace_schedule(traces, n_images: int, n_tokens: int):
    if len(traces) != n_images:
        raise AssertionError(f"historical trace image count {len(traces)} != {n_images}")
    for i, tr in enumerate(traces):
        if len(tr) != ROUNDS:
            raise AssertionError(f"historical trace {i} has {len(tr)} rounds, expected {ROUNDS}")
        cur = np.full(n_tokens, MASK, dtype=np.int64)
        for r, row in enumerate(tr):
            masked = np.asarray(row["masked_positions_before"], dtype=np.int64)
            remask = np.asarray(row["remasked_positions"], dtype=np.int64)
            if not np.array_equal(np.flatnonzero(cur == MASK), masked):
                raise AssertionError(f"schedule position mismatch image={i} round={r+1}")
            sampled = np.asarray(row["sampled_ids"], dtype=np.int64)
            proposal = cur.copy()
            proposal[masked] = sampled
            cur = proposal
            cur[remask] = MASK
    return True


def reference_predictor_reconstruction(terminal, tok, pos, traces):
    max_abs = 0.0
    n_probs = 0
    position_pass = True
    image_count_pass = True
    for i, tr in enumerate(traces):
        pred = core.P3ImageSystem(terminal.c, model_dict(terminal), BASE_GEN_SEED + 100000 + i)
        cur = np.full(terminal.c.N, MASK, dtype=np.int64)
        for row in tr:
            masked = np.asarray(row["masked_positions_before"], dtype=np.int64)
            if not np.array_equal(np.flatnonzero(cur == MASK), masked):
                position_pass = False
            xn, xraw = context_from_arr_unconditional(cur, tok, pos)
            p = np.empty((len(masked), terminal.c.V), dtype=np.float64)
            for qi, t0 in enumerate(masked):
                t = int(t0)
                p[qi] = pred.forward_only(xn, xraw[t], t, new_image=(qi == 0), changed=None)
            p /= p.sum(axis=1, keepdims=True)
            sampled = np.asarray(row["sampled_ids"], dtype=np.int64)
            stored = np.asarray(row["sampled_confidence"], dtype=np.float64)
            got = p[np.arange(len(sampled)), sampled]
            if len(got):
                max_abs = max(max_abs, float(np.max(np.abs(got - stored))))
                n_probs += len(got)
            proposal = cur.copy()
            proposal[masked] = sampled
            cur = proposal
            cur[np.asarray(row["remasked_positions"], dtype=np.int64)] = MASK
        if pred.image_count != ROUNDS:
            image_count_pass = False
    return {
        "max_abs_probability_error": float(max_abs),
        "n_compared_probabilities": int(n_probs),
        "position_schedule_exact": bool(position_pass),
        "image_count_after_three_rounds_exact": bool(image_count_pass),
        "pass": bool(position_pass and image_count_pass and max_abs < TOL),
        "tolerance": TOL,
    }


def causal_gate_a_replay(terminal, tok, pos, traces, gt, shared_u):
    n = len(gt)
    sm = np.full((n, ROUNDS, len(METRIC_NAMES)), np.nan, dtype=np.float64)
    tm = np.full_like(sm, np.nan)
    gates = {
        "round1_probability_bitwise_parity": True,
        "round1_sample_exact_parity": True,
        "committed_position_parity": True,
        "intervention_purity": True,
        "same_shared_uniform_field": True,
        "image_count_after_three_rounds_exact": True,
    }
    for i in range(n):
        ps = core.P3ImageSystem(terminal.c, model_dict(terminal), BASE_GEN_SEED + 100000 + i)
        pt = core.P3ImageSystem(terminal.c, model_dict(terminal), BASE_GEN_SEED + 100000 + i)
        cs = np.full(terminal.c.N, MASK, dtype=np.int64)
        ct = np.full(terminal.c.N, MASK, dtype=np.int64)
        for r in range(ROUNDS):
            row = traces[i][r]
            masked = np.asarray(row["masked_positions_before"], dtype=np.int64)
            remask = np.asarray(row["remasked_positions"], dtype=np.int64)
            if not np.array_equal(np.flatnonzero(cs == MASK), masked):
                gates["committed_position_parity"] = False
            if not np.array_equal(np.flatnonzero(ct == MASK), masked):
                gates["committed_position_parity"] = False
            if r > 0:
                vis = np.flatnonzero(ct != MASK)
                if len(vis) and not np.array_equal(ct[vis], gt[i, vis]):
                    gates["intervention_purity"] = False
            p1 = predict_round_causal(ps, cs, tok, pos, masked)
            p2 = predict_round_causal(pt, ct, tok, pos, masked)
            u = np.asarray(shared_u[i, r, masked], dtype=np.float64)
            z1 = inverse_cdf_rows(p1, u)
            z2 = inverse_cdf_rows(p2, u)
            if r == 0 and not np.array_equal(p1, p2):
                gates["round1_probability_bitwise_parity"] = False
            if r == 0 and not np.array_equal(z1, z2):
                gates["round1_sample_exact_parity"] = False
            sm[i, r] = one_metrics(p1, gt[i, masked], z1, cs)
            tm[i, r] = one_metrics(p2, gt[i, masked], z2, ct)
            ns = cs.copy()
            ns[masked] = z1
            ns[remask] = MASK
            nt = ct.copy()
            new = np.setdiff1d(masked, remask, assume_unique=True)
            nt[new] = gt[i, new]
            nt[remask] = MASK
            old = np.flatnonzero(ct != MASK)
            nt[old] = gt[i, old]
            cs, ct = ns, nt
        if ps.image_count != ROUNDS or pt.image_count != ROUNDS:
            gates["image_count_after_three_rounds_exact"] = False
    gates["all_pass"] = bool(all(gates.values()))
    effects = round3_effects(sm, tm)
    abs_diff = {k: abs(float(effects[k]) - float(EXPECTED["effects"][k])) for k in EXPECTED["effects"]}
    effect_pass = bool(all(v < TOL for v in abs_diff.values()))
    return {
        "effects": effects,
        "expected_effects": EXPECTED["effects"],
        "abs_diff": abs_diff,
        "structural_gates": gates,
        "effect_parity_pass": effect_pass,
        "pass": bool(gates["all_pass"] and effect_pass),
        "tolerance": TOL,
        "causal_trace_sample_equality_required": False,
        "causal_uniform_seed": CAUSAL_UNIFORM_SEED,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--historical-traces", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--contexts", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=7311)
    ap.add_argument("--encode-batch", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads)
    device = torch.device("cpu")

    source, meta, train_rng, schedule = trainer.load_full_checkpoint(a.checkpoint)
    if abs(float(meta.get("context_epoch_equivalent", -1.0)) - 2.0) > 1e-12:
        raise ValueError("A0 requires the exact 2.0-epoch R6B3 checkpoint")
    offset = int(schedule["offset"])
    indices = np.asarray(schedule["permutation"][offset:offset + a.contexts], dtype=np.int64)
    train_rng_state = copy.deepcopy(train_rng.bit_generator.state)

    xall, yall, _xte, _yte = core.load(a.data)
    x012, y012 = core.subset012(xall, yall)
    xtr, _ytr, xva, _yva = core.split(x012, y012, .1, 7)
    tokenizer, tok_meta = base.load_frozen_tokenizer(a.tokenizer, device)
    tr, _sp_tr = base.encode_snn(tokenizer, xtr, device, a.encode_batch)
    va, _sp_va = base.encode_snn(tokenizer, xva, device, a.encode_batch)
    tok0, pos0, input_audit = build_input_embeddings(tokenizer, source.c.D, grid=7)

    validation_ids = np.random.default_rng(VALIDATION_SEED).choice(1862, 256, replace=False).astype(np.int64)
    gt = np.asarray(va[validation_ids], dtype=np.int64)
    shared_u = np.random.default_rng(CAUSAL_UNIFORM_SEED).random((256, ROUNDS, source.c.N))
    shared_g = np.random.default_rng(CAUSAL_GUMBEL_SEED).gumbel(size=(256, ROUNDS, source.c.N))
    observed_hashes = {
        "validation_ids_sha256": raw_array_sha256(validation_ids),
        "gt_tokens_sha256": raw_array_sha256(gt),
        "shared_uniform_sha256": raw_array_sha256(shared_u),
        "shared_gumbel_sha256": raw_array_sha256(shared_g),
    }
    a01 = {"observed": observed_hashes, "expected": {k: EXPECTED[k] for k in observed_hashes},
           "pass": bool(all(observed_hashes[k] == EXPECTED[k] for k in observed_hashes)),
           "validation_seed": VALIDATION_SEED, "uniform_seed": CAUSAL_UNIFORM_SEED,
           "gumbel_seed": CAUSAL_GUMBEL_SEED, "validation_ids_first12": validation_ids[:12].astype(int).tolist()}
    print("D3E_A0_1_SHARED_INPUTS", json.dumps(a01, sort_keys=True), flush=True)

    terminal, train_result = reconstruct_baseline(source, tr, indices, tok0, pos0, train_rng_state)
    terminal_hash = ndarray_state_sha256(terminal)
    a02 = {"observed_ndarray_state_sha256": terminal_hash,
           "expected_ndarray_state_sha256": EXPECTED["terminal_ndarray_state_sha256"],
           "pass": bool(terminal_hash == EXPECTED["terminal_ndarray_state_sha256"]),
           "contexts": int(a.contexts), "context_signature_sha256": train_result["context_signature_sha256"]}
    print("D3E_A0_2_TERMINAL_STATE", json.dumps(a02, sort_keys=True), flush=True)

    traces = json.loads(a.historical_traces.read_text(encoding="utf-8"))
    schedule_ok = validate_trace_schedule(traces, 256, source.c.N)
    a03 = {"historical_trace_path": str(a.historical_traces),
           "historical_trace_file_sha256": file_sha256(a.historical_traces),
           "images": int(len(traces)), "rounds": int(ROUNDS), "schedule_structure_exact": bool(schedule_ok),
           "source_run": 39230496177, "source_artifact": "SNN_TRANSFORM_R6B4_FULL_FORWARD_EFFECTIVENESS_TEST",
           "source_member": "result/r6b3_frozen_new_params_r3_t8_traces.json", "pass": bool(schedule_ok)}
    print("D3E_A0_3_HISTORICAL_SCHEDULE", json.dumps(a03, sort_keys=True), flush=True)

    a04 = reference_predictor_reconstruction(terminal, tok0, pos0, traces)
    print("D3E_A0_4_REFERENCE_PREDICTOR", json.dumps(a04, sort_keys=True), flush=True)
    a05 = causal_gate_a_replay(terminal, tok0, pos0, traces, gt, shared_u)
    print("D3E_A0_5_CAUSAL_GATE_A", json.dumps(a05, sort_keys=True), flush=True)

    all_pass = bool(a01["pass"] and a02["pass"] and a03["pass"] and a04["pass"] and a05["pass"])
    report = {"stage": "D3E_A0_GATE_A_PROVENANCE_RECOVERY", "scientific_experiment_run": False,
              "four_d3e_arms_run": False, "original_d3e_modified": False,
              "provenance_bug_under_test": "historical reference-generation RNG must be separated from Gate-A causal CRN sampling",
              "A0_1_shared_input_provenance": a01, "A0_2_terminal_r6b3_state": a02,
              "A0_3_historical_schedule": a03, "A0_4_reference_predictor_reconstruction": a04,
              "A0_5_causal_gate_a_replay": a05, "tokenizer_metadata": tok_meta, "input_audit": input_audit,
              "all_pass": all_pass}
    (a.out / "D3E_A0_PROVENANCE_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("D3E_A0_FINAL", json.dumps({"all_pass": all_pass, "terminal_hash_pass": a02["pass"],
          "reference_predictor_pass": a04["pass"], "causal_gate_a_pass": a05["pass"],
          "causal_abs_diff": a05["abs_diff"]}, sort_keys=True), flush=True)
    if not all_pass:
        raise RuntimeError("D3E-A0 provenance recovery failed; do not run D3E intervention arms")


if __name__ == "__main__":
    main()
