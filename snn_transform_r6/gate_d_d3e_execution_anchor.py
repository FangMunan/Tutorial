#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze the D3E R6B4-native execution anchor.

This is a one-time harness/provenance operation, not a D3E intervention.
It establishes a cross-run functional anchor for the R6B4 aux1.0 learner while
keeping historical R6B3 Gate-A evaluator identity as a separate provenance gate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch

import gate_d_d3e as d3e
import gate_d_d3e_a0 as a0
import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
import r6b3_context_epoch_training as trainer
import r6b3_decoder_operating_point_audit as coarse
import r6b3_decoder_spatial_refinement_audit as spatial
import r6b4_full_forward_effectiveness_test as eff
from maskgit_input_representation import build_input_embeddings
from r6b3_snn_encoder_maskgit_msdl import generation_tokens_maskgit
from r6b4_full_forward_plasticity import upgrade_r6b3_system

EXPECTED_CHECKPOINT_SHA256 = "4a42e60ada31f6ec5167cafd2bd860dd8ca937f7970665c5b95bf8fb2dfc89d0"
EXPECTED_EFFECTIVENESS = {
    "online_nll": 1.599690036154988,
    "context_signature_sha256": "1150e17b370771fe5227012703663b920c1bea01cbedb4df0ab6588deafc7402",
    "mask1_ce": 1.596839748626027,
    "r3_t8_global_js": 0.00477388532350873,
    "r3_t8_spatial_js": 0.03368588417096242,
    "r3_t8_pair_js": 0.09252765679317349,
}
EFFECTIVENESS_TOL = 1e-7
GATE_A_TOL = 1e-7


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def locate(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {hits}")
    return hits[0]


def schedule_projection(traces):
    out = []
    for image in traces:
        rr = []
        for row in image:
            rr.append({
                "masked_positions_before": [int(x) for x in row["masked_positions_before"]],
                "remasked_positions": [int(x) for x in row["remasked_positions"]],
            })
        out.append(rr)
    return out


def json_sha(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def observed_effectiveness(system, va, seed, gen_samples):
    masked = eff.masked_eval(system, va, system.tok, system.pos, seed + 10000, 512)
    _rc, real_p = coarse.probability_from_ids(va, system.c.V)
    _pc, real_pos_p = spatial.position_probabilities(va, system.c.V, system.c.N)
    real_pair = spatial.pair_reference(va, system.c.V, system.c.N, 7)
    ids, _traces, audits = generation_tokens_maskgit(
        trainer.model_dict_from_system(system), system.c, gen_samples,
        system.tok, system.pos, seed + 20000,
        num_iter=3, choice_temperature=8.0)
    gl = coarse.distribution_metrics(ids, real_p, system.c.V)
    sp = spatial.positionwise_metrics(ids, real_pos_p, system.c.V, system.c.N)
    pa = spatial.pair_metrics(ids, real_pair, system.c.V, system.c.N, 7)
    return {
        "mask1_ce": float(masked["1.0"]["ce"]),
        "r3_t8_global_js": float(gl["js_to_real_nats"]),
        "r3_t8_spatial_js": float(sp["mean_js_nats"]),
        "r3_t8_pair_js": float(pa["mean_orientation_js_nats"]),
        "generation_trace_audits_pass": bool(all(x["pass"] for x in audits)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--effectiveness", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7311)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    device = torch.device("cpu")

    checkpoint_sha = sha_file(a.checkpoint)
    source, meta, train_rng, schedule = trainer.load_full_checkpoint(a.checkpoint)
    p1 = bool(
        checkpoint_sha == EXPECTED_CHECKPOINT_SHA256
        and abs(float(meta.get("context_epoch_equivalent", -1.0)) - 2.0) <= 1e-12
    )
    if not p1:
        raise RuntimeError({"P1": False, "checkpoint_sha256": checkpoint_sha, "meta": meta})

    offset = int(schedule["offset"])
    indices = np.asarray(schedule["permutation"], dtype=np.int64)[offset:offset + 1024]
    if len(indices) != 1024:
        raise RuntimeError("saved epoch-3 schedule does not contain 1024 continuation contexts")
    rng_state = copy.deepcopy(train_rng.bit_generator.state)

    xall, yall, _xte, _yte = core.load(a.data)
    x012, y012 = core.subset012(xall, yall)
    xtr, _ytr, xva, _yva = core.split(x012, y012, .1, 7)
    tokenizer, tok_meta = base.load_frozen_tokenizer(a.tokenizer, device)
    tr, sp_tr = base.encode_snn(tokenizer, xtr, device, 256)
    va, sp_va = base.encode_snn(tokenizer, xva, device, 256)
    tok0, pos0, input_audit = build_input_embeddings(tokenizer, source.c.D, grid=7)

    validation_ids = np.random.default_rng(a0.SELECTION_SEED).choice(
        len(va), size=a0.N_IMAGES, replace=False).astype(np.int64)
    gt = np.asarray(va[validation_ids], dtype=np.int16)
    shared_uniform = np.random.default_rng(a0.UNIFORM_SEED).random(
        (a0.N_IMAGES, a0.ROUNDS, source.c.N))
    shared_gumbel = np.random.default_rng(a0.GUMBEL_SEED).gumbel(
        size=(a0.N_IMAGES, a0.ROUNDS, source.c.N))
    p0_hashes = {
        "validation_ids_sha256": a0.arr_sha(validation_ids),
        "gt_tokens_sha256": a0.arr_sha(gt),
        "shared_uniform_sha256": a0.arr_sha(shared_uniform),
        "shared_gumbel_sha256": a0.arr_sha(shared_gumbel),
    }
    p0 = bool(all(p0_hashes[k] == a0.EXPECTED[k] for k in p0_hashes))
    if not p0:
        raise RuntimeError({"P0": False, "hashes": p0_hashes})

    trace_file = locate(a.effectiveness, "r6b3_frozen_new_params_r3_t8_traces.json")
    historical_traces = json.loads(trace_file.read_text(encoding="utf-8"))
    if len(historical_traces) != a0.N_IMAGES:
        raise RuntimeError("historical Gate-A trace does not contain 256 images")
    projected_schedule = schedule_projection(historical_traces)
    schedule_hash = json_sha(projected_schedule)

    r6b3, r6b3_stream_hash, _ = a0.reconstruct_r6b3(
        source, tr, indices, rng_state, tok0, pos0)
    p2_detail = a0.reference_trace_gate(r6b3, tok0, pos0, historical_traces)
    p2 = bool(
        p2_detail["trace_mask_context_exact"]
        and p2_detail["reference_probability_confidence_pass"]
        and p2_detail["image_count_after_three_rounds_exact"]
    )
    r6b3_gate = d3e.eval_gate_a(
        "historical_r6b3_control", r6b3.c, trainer.model_dict_from_system(r6b3),
        tok0, pos0, historical_traces, gt, shared_uniform)
    p3_diff = {
        k: float(abs(r6b3_gate["summary"][k]["mean"] - a0.EXPECTED["effects"][k]))
        for k in a0.EXPECTED["effects"]
    }
    p3 = bool(r6b3_gate["gates"]["all_pass"] and max(p3_diff.values()) < GATE_A_TOL)
    if not (p2 and p3):
        raise RuntimeError({"P2": p2, "P2_detail": p2_detail, "P3": p3, "P3_diff": p3_diff})

    native = pickle.loads(pickle.dumps(source, pickle.HIGHEST_PROTOCOL))
    upgrade_r6b3_system(native, tok0, pos0, aux_lr_scale=1.0)
    ones = np.ones(native.c.V, dtype=np.float64)
    train = eff.train_arm(
        "r6b4_aux_1p0_execution_anchor", native, "full", tr, indices,
        tok0, pos0, ones, rng_state, 256)
    obs_eff = observed_effectiveness(native, va, a.seed, 256)
    obs_eff["online_nll"] = float(train["mean_online_nll"])
    obs_eff["context_signature_sha256"] = str(train["context_signature_sha256"])
    eff_diff = {
        k: float(abs(obs_eff[k] - EXPECTED_EFFECTIVENESS[k]))
        for k in ("online_nll", "mask1_ce", "r3_t8_global_js", "r3_t8_spatial_js", "r3_t8_pair_js")
    }
    a4 = bool(
        obs_eff["context_signature_sha256"] == EXPECTED_EFFECTIVENESS["context_signature_sha256"]
        and obs_eff["generation_trace_audits_pass"]
        and max(eff_diff.values()) < EFFECTIVENESS_TOL
    )
    if not a4:
        raise RuntimeError({"A4_historical_R6B4_semantic_anchor": False,
                            "observed": obs_eff, "diff": eff_diff})

    native_gate = d3e.eval_gate_a(
        "d3e_r6b4_native_execution_anchor", native.c,
        trainer.model_dict_from_system(native), native.tok, native.pos,
        historical_traces, gt, shared_uniform)
    if not native_gate["gates"]["all_pass"]:
        raise RuntimeError({"R6B4_anchor_gate_structural_failure": native_gate["gates"]})

    arrays = {
        "validation_ids": validation_ids,
        "gt_tokens": gt,
        "shared_uniform": shared_uniform,
        "shared_gumbel": shared_gumbel,
        "self_metrics": native_gate["self_metrics"],
        "teacher_metrics": native_gate["teacher_metrics"],
        "self_proposal": native_gate["self_proposal"],
        "teacher_proposal": native_gate["teacher_proposal"],
    }
    for k, v in native_gate["effects"].items():
        arrays[f"effect__{k}"] = np.asarray(v)
    npz_path = a.out / "R6B4_NATIVE_GATE_A_EXECUTION_ANCHOR.npz"
    np.savez_compressed(npz_path, **arrays)

    trace_out = a.out / "HISTORICAL_R6B3_R3_T8_REFERENCE_TRACE.json"
    trace_out.write_text(json.dumps(historical_traces, separators=(",", ":")), encoding="utf-8")
    schedule_out = a.out / "D3E_GATE_A_SCHEDULE.json"
    schedule_out.write_text(json.dumps(projected_schedule, separators=(",", ":")), encoding="utf-8")

    array_hashes = {k: a0.arr_sha(np.asarray(v)) for k, v in arrays.items()}
    aggregate = {k: float(v["mean"]) for k, v in native_gate["summary"].items()}
    report = {
        "stage": "D3E_R6B4_NATIVE_EXECUTION_ANCHOR_FREEZE",
        "anchor_role": "frozen D3E R6B4-native execution anchor",
        "not_historical_gateA_R6B4_reference": True,
        "d3e_intervention_arms_run": False,
        "P0_shared_provenance_exact": p0,
        "P1_exact_2epoch_checkpoint_identity": p1,
        "P2_historical_R6B3_reference_predictor_parity": p2,
        "P3_historical_R6B3_GateA_causal_parity": p3,
        "A4_historical_R6B4_effectiveness_semantic_identity": a4,
        "all_freeze_gates_pass": bool(p0 and p1 and p2 and p3 and a4),
        "checkpoint_sha256": checkpoint_sha,
        "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "shared_provenance": p0_hashes,
        "schedule_sha256": schedule_hash,
        "schedule_definition": "hash over masked_positions_before + remasked_positions only; proposal content excluded",
        "historical_trace_sha256": sha_file(trace_out),
        "historical_R6B3_reconstruction_stream_sha256": r6b3_stream_hash,
        "P2_detail": p2_detail,
        "P3_effect_abs_diff": p3_diff,
        "historical_R6B4_effectiveness_expected": EXPECTED_EFFECTIVENESS,
        "historical_R6B4_effectiveness_observed": obs_eff,
        "historical_R6B4_effectiveness_abs_diff": eff_diff,
        "effectiveness_tolerance": EFFECTIVENESS_TOL,
        "gateA_tolerance": GATE_A_TOL,
        "r6b4_native_gateA_aggregate_means": aggregate,
        "r6b4_native_gateA_structural_gates": native_gate["gates"],
        "anchor_array_sha256": array_hashes,
        "anchor_npz_sha256": sha_file(npz_path),
        "tokenizer_metadata": tok_meta,
        "input_audit": input_audit,
        "train_tokenizer_activity": sp_tr,
        "validation_tokenizer_activity": sp_va,
        "github_sha": os.environ.get("GITHUB_SHA"),
        "scientific_boundary": (
            "The historical R6B3 byte-identical +1024 pickle is not recovered and is not an execution dependency. "
            "P0-P3 validate the historical evaluator/provenance. A4 validates R6B4 aux1.0 learner identity against "
            "historical effectiveness observables. The newly frozen causal arrays define a cross-run functional "
            "execution anchor, not a byte-identical terminal hidden-state authority."
        ),
    }
    (a.out / "D3E_R6B4_NATIVE_EXECUTION_ANCHOR.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    if not report["all_freeze_gates_pass"]:
        raise RuntimeError("execution-anchor freeze gates failed")
    print("D3E_EXECUTION_ANCHOR_FROZEN", json.dumps({
        "all_freeze_gates_pass": True,
        "schedule_sha256": schedule_hash,
        "anchor_npz_sha256": report["anchor_npz_sha256"],
        "gateA_means": aggregate,
        "effectiveness_abs_diff": eff_diff,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
