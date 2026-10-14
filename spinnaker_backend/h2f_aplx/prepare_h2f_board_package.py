#!/usr/bin/env python3
"""Prepare the frozen H2F SDRAM image and final-state oracle.

Input is the accepted H1.1-functional-v9 HIR.  The generated package contains:

* static HIR blob;
* six complete Q20 input snapshots plus explicit changed-token masks;
* a fixed header with quantised Izh operating points and array descriptors;
* final H2D integer oracle: y, H, g, pq, pk, v.

No training, calibration or architectural change occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from spinnaker_backend.h2_aplx.test_h2_actual_hir import (
    ActualHIRRuntime, load_hir, make_sequence)
from spinnaker_backend.h2_aplx.test_h2_kernels import build
from spinnaker_backend.h2f_aplx.h2f_contract import (
    F_ACT, F_PQ, F_PK, F_H, F_MEM, F_DEC,
    build_header, build_hir_blob, output_word_count, save_board_package)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--change-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=17171)
    args = ap.parse_args()

    meta, arrays = load_hir(args.hir)
    if meta.get("format") != "SNNTransformBlock-v1.1-HIR-H1.1-v2":
        raise ValueError(f"unexpected HIR format {meta.get('format')!r}")
    expected_formats = {
        "F_ACT": F_ACT, "F_PQ": F_PQ, "F_PK": F_PK,
        "F_H": F_H, "F_DEC": F_DEC,
    }
    for key, value in expected_formats.items():
        if int(meta[key]) != value:
            raise ValueError(f"{key}: HIR={meta[key]} board-contract={value}")

    tokens = 16
    d = int(meta["d"])
    heads = int(meta["heads"])
    features = int(meta["features"])
    ff = int(meta["ff"])
    if (d, heads, features, ff) != (32, 2, 32, 64):
        raise ValueError("H2F-v1 is frozen to N16,D32,H2,M32,FF64")

    xs, masks = make_sequence(
        tokens, d, steps=args.steps,
        change_frac=args.change_frac, seed=args.seed)
    xs_i32 = np.ascontiguousarray(np.stack(xs), dtype="<i4")
    masks_u32 = np.ascontiguousarray(np.stack(masks).astype(np.uint32), dtype="<u4")

    # Oracle deliberately uses the independent NumPy integer reference.  H2D
    # already established bit-exact host-C equivalence to this path.
    lib = build()
    rt = ActualHIRRuntime(lib, meta, arrays, use_c=False)
    state = None
    y = None
    trajectory_rows = []
    for step, (x, mask) in enumerate(zip(xs, masks)):
        y, state = rt.step(x, state, None if state is None else mask)
        trajectory_rows.append({
            "step": step,
            "changed_tokens": int(np.asarray(mask, dtype=bool).sum()),
        })
    assert y is not None and state is not None

    hir_blob_u32, desc = build_hir_blob(arrays)
    header_u32 = build_header(meta, desc, steps=args.steps)
    x_words = xs_i32.view("<u4").reshape(-1)
    trajectory_u32 = np.concatenate([x_words, masks_u32.reshape(-1)]).astype("<u4", copy=False)

    oracle = {
        "y": np.asarray(y, dtype="<i4"),
        "H": np.asarray(state.H, dtype="<i4"),
        "g": np.asarray(state.g, dtype="<i4"),
        "pq": np.asarray(state.pq, dtype="<i4"),
        "pk": np.asarray(state.pk, dtype="<i4"),
        "v": np.asarray(state.v, dtype="<i4"),
    }
    output_words = output_word_count(tokens, d, heads, features)
    package_meta = {
        "format": "SNNTransform-H2F-board-package-v1",
        "contract": "H1.1-functional-v9",
        "source_hir_format": meta["format"],
        "source_hir_sha256": sha256_file(args.hir),
        "source_checkpoint_sha256": meta["source_checkpoint_sha256"],
        "tokens": tokens,
        "d": d,
        "heads": heads,
        "features": features,
        "ff": ff,
        "steps": args.steps,
        "change_frac": args.change_frac,
        "seed": args.seed,
        "formats": {
            "F_ACT": F_ACT, "F_PQ": F_PQ, "F_PK": F_PK,
            "F_H": F_H, "F_MEM": F_MEM, "F_DEC": F_DEC,
        },
        "region_words": {
            "config": int(header_u32.size),
            "hir": int(hir_blob_u32.size),
            "trajectory": int(trajectory_u32.size),
            "output": int(output_words),
        },
        "trajectory": trajectory_rows,
        "scientific_contract": {
            "host_intermediate_compute": 0,
            "host_hidden_state_roundtrips": 0,
            "input_uploads": 1,
            "result_readbacks": 1,
            "board_resident_state": ["H", "g", "pq", "pk", "v"],
            "acceptance": "bit-exact final y,H,g,pq,pk,v versus H2D integer oracle",
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
    print(json.dumps({
        "stage": "H2F_PACKAGE_PREP",
        "PASS": True,
        "output": str(args.out),
        **package_meta,
    }, indent=2))


if __name__ == "__main__":
    main()
