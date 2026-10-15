#!/usr/bin/env python3
"""Prepare one H3A single-core scaling package for token count N.

This is a hardware microbenchmark, not a task benchmark.  The frozen HIR and
all mathematical operators remain unchanged; only N changes.  A deterministic
generic Q20 trajectory avoids the N<=24 constraint of the old associative-task
sequence generator.  Before a package is emitted, NumPy and host-C composed
integer runtimes must be bit-exact at every step and in all resident states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from spinnaker_backend.h2_aplx.test_h2_actual_hir import ActualHIRRuntime, load_hir
from spinnaker_backend.h2_aplx.test_h2_kernels import build
from spinnaker_backend.h2f_aplx.h2f_contract import build_hir_blob, save_board_package
from spinnaker_backend.h3a_single_core.h3a_contract import (
    F_ACT, build_header, arena_bytes, trajectory_word_count, output_word_count)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_scaling_sequence(tokens: int, d: int, *, steps: int, change_frac: float, seed: int):
    if not (0.0 < change_frac <= 1.0):
        raise ValueError("change_frac must be in (0,1]")
    rng = np.random.default_rng(seed)
    # Inputs are deliberately generic rather than task-coded.  Bounded Q20
    # values keep the frozen integer operators in their established range.
    lim = 1 << (F_ACT - 2)  # +/-0.25 in Q20

    def sample() -> np.ndarray:
        return rng.integers(-lim, lim + 1, size=(tokens, d), dtype=np.int32)

    xs = [sample()]
    masks = [np.ones(tokens, dtype=bool)]
    nchange = max(1, int(round(tokens * change_frac)))
    for _ in range(1, steps):
        fresh = sample()
        nxt = xs[-1].copy()
        idx = rng.permutation(tokens)[:nchange]
        nxt[idx] = fresh[idx]
        mask = np.zeros(tokens, dtype=bool)
        mask[idx] = True
        xs.append(nxt)
        masks.append(mask)
    return xs, masks


def assert_equal_state(a, b, *, step: int) -> None:
    for name in ("H", "g", "pq", "pk", "v"):
        aa = np.asarray(getattr(a, name))
        bb = np.asarray(getattr(b, name))
        if not np.array_equal(aa, bb):
            diff = int(np.max(np.abs(aa.astype(np.int64) - bb.astype(np.int64))))
            raise AssertionError(f"host-C parity fail step={step} state={name} max_abs={diff}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--change-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=27171)
    args = ap.parse_args()

    meta, arrays = load_hir(args.hir)
    if meta.get("format") != "SNNTransformBlock-v1.1-HIR-H1.1-v2":
        raise ValueError(f"unexpected HIR format {meta.get('format')!r}")
    d, heads, features, ff = (int(meta[k]) for k in ("d", "heads", "features", "ff"))
    if (d, heads, features, ff) != (32, 2, 32, 64):
        raise ValueError("H3A-v1 fixes D32,H2,M32,FF64 and varies only N")

    xs, masks = make_scaling_sequence(
        args.tokens, d, steps=args.steps, change_frac=args.change_frac, seed=args.seed)

    lib = build()
    rt_np = ActualHIRRuntime(lib, meta, arrays, use_c=False)
    rt_c = ActualHIRRuntime(lib, meta, arrays, use_c=True)
    sn = sc = None
    yn = yc = None
    trajectory_rows = []
    for step, (x, mask) in enumerate(zip(xs, masks)):
        changed = None if step == 0 else mask
        yn, sn = rt_np.step(x, sn, changed)
        yc, sc = rt_c.step(x, sc, changed)
        if not np.array_equal(yn, yc):
            diff = int(np.max(np.abs(yn.astype(np.int64) - yc.astype(np.int64))))
            raise AssertionError(f"host-C parity fail step={step} y max_abs={diff}")
        assert_equal_state(sn, sc, step=step)
        trajectory_rows.append({"step": step, "changed_tokens": int(mask.sum())})
    assert yn is not None and sn is not None

    xs_i32 = np.ascontiguousarray(np.stack(xs), dtype="<i4")
    masks_u32 = np.ascontiguousarray(np.stack(masks).astype(np.uint32), dtype="<u4")
    hir_blob_u32, desc = build_hir_blob(arrays)
    header_u32 = build_header(meta, desc, tokens=args.tokens, steps=args.steps)
    trajectory_u32 = np.concatenate([
        xs_i32.view("<u4").reshape(-1), masks_u32.reshape(-1)
    ]).astype("<u4", copy=False)

    expected_traj = trajectory_word_count(args.tokens, d, args.steps)
    if trajectory_u32.size != expected_traj:
        raise AssertionError((trajectory_u32.size, expected_traj))

    oracle = {
        "y": np.asarray(yn, dtype="<i4"),
        "H": np.asarray(sn.H, dtype="<i4"),
        "g": np.asarray(sn.g, dtype="<i4"),
        "pq": np.asarray(sn.pq, dtype="<i4"),
        "pk": np.asarray(sn.pk, dtype="<i4"),
        "v": np.asarray(sn.v, dtype="<i4"),
    }
    out_words = output_word_count(args.tokens, d, heads, features)
    package_meta = {
        "format": "SNNTransform-H3A-single-core-v1",
        "stage": "H3A_SINGLE_CORE",
        "contract": "H1.1-functional-v9",
        "source_hir_format": meta["format"],
        "source_hir_sha256": sha256_file(args.hir),
        "source_checkpoint_sha256": meta["source_checkpoint_sha256"],
        "tokens": args.tokens,
        "d": d,
        "heads": heads,
        "features": features,
        "ff": ff,
        "steps": args.steps,
        "change_frac": args.change_frac,
        "seed": args.seed,
        "arena_bytes_requested": arena_bytes(args.tokens, d, heads, features, ff),
        "region_words": {
            "config": int(header_u32.size),
            "hir": int(hir_blob_u32.size),
            "trajectory": int(trajectory_u32.size),
            "output": int(out_words),
        },
        "trajectory": trajectory_rows,
        "host_integer_gate": {
            "numpy_vs_composed_C": "bit-exact",
            "all_steps": True,
            "final_states": ["y", "H", "g", "pq", "pk", "v"],
        },
        "scientific_contract": {
            "varied_axis": "tokens_N_only",
            "fixed": {"D": d, "H": heads, "M": features, "FF": ff, "L": 1},
            "host_intermediate_compute": 0,
            "host_hidden_state_roundtrips": 0,
            "board_resident_state": ["H", "g", "pq", "pk", "v"],
            "acceptance_if_allocated": "bit-exact final y,H,g,pq,pk,v versus host integer oracle",
            "capacity_failure": "spin1_malloc arena allocation failure rc=20",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_board_package(
        args.out,
        header_u32=header_u32,
        hir_blob_u32=hir_blob_u32,
        trajectory_u32=trajectory_u32,
        oracle=oracle,
        metadata=package_meta,
    )
    print(json.dumps({"PASS": True, "output": str(args.out), **package_meta}, indent=2))


if __name__ == "__main__":
    main()
