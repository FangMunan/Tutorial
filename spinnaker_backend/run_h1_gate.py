#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .h1_fixed_point_emulator import (
    F_ACT,
    F_PHI,
    FixedPointBlockRuntime,
    compile_hir,
    deq,
    float_step,
    integer_hir_only,
    make_representative_params,
    qround,
)


def rel(a, b):
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(aa - bb) / max(np.linalg.norm(bb), 1e-12))


def sequence(seed=123, B=4, N=16, D=32, steps=6, change_frac=0.25):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.6, size=(B, N, D))
    xs, masks = [x.copy()], [None]
    nchange = max(1, round(N * change_frac))
    for _ in range(1, steps):
        x = x.copy()
        mask = np.zeros((B, N), dtype=bool)
        for b in range(B):
            idx = rng.choice(N, size=nchange, replace=False)
            mask[b, idx] = True
            x[b, idx] = rng.normal(0, 0.6, size=(nchange, D))
        xs.append(x)
        masks.append(mask)
    return xs, masks, nchange


def main():
    out_dir = Path("spinnaker_backend/run/H1")
    out_dir.mkdir(parents=True, exist_ok=True)

    p = make_representative_params(seed=71)
    hir = compile_hir(p)
    runtime = FixedPointBlockRuntime(hir)

    xs, masks, nchange = sequence(D=p.d)
    fs = qs = None
    rows = []
    max_output = max_H = max_g = max_pq = max_pk = max_v = 0.0

    for step, (x, changed) in enumerate(zip(xs, masks)):
        yf, fs = float_step(p, x, fs, changed)
        yq, qs = runtime.step(qround(x, F_ACT), qs, changed)

        output_rel = rel(deq(yq, F_ACT), yf)
        H_rel = rel(deq(qs.H, F_PHI + F_ACT), fs.H)
        g_rel = rel(deq(qs.g, F_PHI), fs.g)
        pq_rel = rel(deq(qs.pq, F_PHI), fs.pq)
        pk_rel = rel(deq(qs.pk, F_PHI), fs.pk)
        v_rel = rel(deq(qs.v, F_ACT), fs.v)

        max_output = max(max_output, output_rel)
        max_H = max(max_H, H_rel)
        max_g = max(max_g, g_rel)
        max_pq = max(max_pq, pq_rel)
        max_pk = max(max_pk, pk_rel)
        max_v = max(max_v, v_rel)
        rows.append({
            "step": step,
            "output_rel_l2": output_rel,
            "H_rel_l2": H_rel,
            "g_rel_l2": g_rel,
            "pq_rel_l2": pq_rel,
            "pk_rel_l2": pk_rel,
            "v_rel_l2": v_rel,
        })

    B, N, _ = xs[0].shape
    dense_evals = len(xs) * B * N
    expected_event_evals = B * N + (len(xs) - 1) * B * nchange
    measured = runtime.stats.q_token_evals
    event_reduction = 1.0 - measured / dense_evals

    # H is the largest resident integer array in this representative gate.
    max_abs = {
        "H": int(np.max(np.abs(qs.H))),
        "g": int(np.max(np.abs(qs.g))),
        "pq": int(np.max(np.abs(qs.pq))),
        "pk": int(np.max(np.abs(qs.pk))),
        "v": int(np.max(np.abs(qs.v))),
    }
    int32_safe_state = all(v < (1 << 31) for v in max_abs.values())

    dense_tail = p.d * p.d + p.ff * p.d + p.d * p.ff
    sparse_tail = hir.Wo.nnz + hir.W1.nnz + hir.W2.nnz
    sparse_fraction = sparse_tail / dense_tail

    gates = {
        "integer_only_HIR": integer_hir_only(hir),
        "full_block_output_rel_l2_le_0p5pct": max_output <= 0.005,
        "H_rel_l2_le_0p5pct": max_H <= 0.005,
        "g_rel_l2_le_0p5pct": max_g <= 0.005,
        "pq_rel_l2_le_1pct": max_pq <= 0.01,
        "pk_rel_l2_le_1pct": max_pk <= 0.01,
        "v_rel_l2_le_0p5pct": max_v <= 0.005,
        "event_eval_count_exact": measured == expected_event_evals,
        "qkv_event_counts_equal": (
            runtime.stats.q_token_evals
            == runtime.stats.k_token_evals
            == runtime.stats.v_token_evals
        ),
        "event_reduction_ge_0p60": event_reduction >= 0.60,
        "tail_static_fraction_eq_0p25": abs(sparse_fraction - 0.25) < 1e-12,
        "representative_resident_state_int32_safe": int32_safe_state,
        "minimal_state_fields": set(qs.__dict__.keys()) == {"H", "g", "pq", "pk", "v"},
    }
    PASS = all(gates.values())

    result = {
        "stage": "H1_CPU_INTEGER_FIXED_POINT_EMULATOR",
        "PASS": PASS,
        "scientific_scope": {
            "architecture": "frozen v1.1 operator graph unchanged",
            "runtime": "NumPy integer-only HIR execution; no PyTorch operator in runtime",
            "parameter_scope": "deterministic representative v1.1-shaped parameters; not yet the frozen trained checkpoint",
            "board_claim": "none; this is the pre-APLX numerical/backend gate",
        },
        "formats": {
            "main_activation_frac_bits": 14,
            "phi_frac_bits": 12,
            "izh_membrane_frac_bits": 20,
            "decoder_frac_bits": 18,
            "accumulator": "int64",
        },
        "max_rel_l2": {
            "output": max_output,
            "H": max_H,
            "g": max_g,
            "pq": max_pq,
            "pk": max_pk,
            "v": max_v,
        },
        "trajectory": rows,
        "event_execution": {
            "dense_token_evals_per_QKV": dense_evals,
            "event_token_evals_per_QKV": measured,
            "expected_event_token_evals": expected_event_evals,
            "reduction": event_reduction,
        },
        "sparse_tail": {
            "dense_connections": dense_tail,
            "sparse_connections": sparse_tail,
            "static_fraction": sparse_fraction,
        },
        "state_integer_max_abs": max_abs,
        "gates": gates,
        "decision": (
            "H1 PASS: freeze the HIR/fixed-point numerical contract and proceed to frozen-checkpoint export + APLX kernels"
            if PASS else
            "H1 FAIL: localize numerical/backend mismatch before writing APLX"
        ),
    }
    (out_dir / "H1_RESULTS.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not PASS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
