#!/usr/bin/env python3
"""Prepare an H3B single-chip distributed board package.

The package is emitted only after a host-side sharded execution proves exact
integer parity with the global frozen H2D reference.  This tests the *planned*
worker reduction arithmetic independently of board multicast/routing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from spinnaker_backend.h2_aplx.test_h2_actual_hir import (
    ActualHIRRuntime, State, load_hir,
)
from spinnaker_backend.h2_aplx.test_h2_kernels import build, rshift_round
from spinnaker_backend.h2f_aplx.h2f_contract import (
    build_hir_blob, package_metadata_bytes,
)
from spinnaker_backend.h3a_single_core.prepare_h3a_package import make_scaling_sequence
from spinnaker_backend.h3b_single_chip.h3b_contract import (
    F_ACT, F_PK, F_H,
    H3B_MAX_WORKERS,
    arena_bytes,
    build_base_header,
    bcast_words,
    init_gather_words,
    local_trajectory_words,
    output_word_count,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tail(rt: ActualHIRRuntime, x: np.ndarray, st: State) -> np.ndarray:
    attended = rt.read(st).reshape(x.shape[0], rt.d)
    ao = rt.sparse(attended, "Wo", rt.d)
    z64 = x.astype(np.int64) + ao.astype(np.int64)
    if np.max(np.abs(z64)) > np.iinfo(np.int32).max:
        raise OverflowError("H3B host gate z overflow")
    z = z64.astype(np.int32)
    n2 = rt.ln(z, "ln2")
    pre = rt.sparse(n2, "W1", rt.d)
    hidden = np.maximum(pre, 0).astype(np.int32)
    ff2 = rt.sparse(hidden, "W2", rt.ff)
    y64 = z.astype(np.int64) + ff2.astype(np.int64)
    if np.max(np.abs(y64)) > np.iinfo(np.int32).max:
        raise OverflowError("H3B host gate y overflow")
    return y64.astype(np.int32)


def local_delta(old_pk: np.ndarray, old_v: np.ndarray,
                new_pk: np.ndarray, new_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact sum of H2 token-delta contributions, kept in int64 until reduce."""
    shift = F_PK + F_ACT - F_H
    old_prod = old_pk[:, :, None].astype(np.int64) * old_v[:, None, :].astype(np.int64)
    new_prod = new_pk[:, :, None].astype(np.int64) * new_v[:, None, :].astype(np.int64)
    dH = -rshift_round(old_prod, shift) + rshift_round(new_prod, shift)
    dg = new_pk.astype(np.int64) - old_pk.astype(np.int64)
    return dH, dg


def distributed_host_gate(meta: dict, arrays: dict[str, np.ndarray],
                          xs: list[np.ndarray], masks: list[np.ndarray],
                          *, workers: int, local_tokens: int) -> tuple[dict, dict[str, np.ndarray]]:
    lib = build()
    ref = ActualHIRRuntime(lib, meta, arrays, use_c=False)
    shard_rt = ActualHIRRuntime(lib, meta, arrays, use_c=False)

    ref_state = None
    local_states: list[State] = []
    last_y = None
    rows = []

    for step, (x, mask) in enumerate(zip(xs, masks)):
        changed = None if step == 0 else mask
        y_ref, ref_state = ref.step(x, ref_state, changed)

        if step == 0:
            pqs, pks, vs = [], [], []
            for rank in range(workers):
                a = rank * local_tokens
                b = a + local_tokens
                pq, pk, v = shard_rt.features(x[a:b])
                pqs.append(pq); pks.append(pk); vs.append(v)
            # Crucially: concatenate before init_state so H is rounded once
            # after the global token sum, exactly like H2D/H2F.
            pq_all = np.concatenate(pqs, axis=0)
            pk_all = np.concatenate(pks, axis=0)
            v_all = np.concatenate(vs, axis=0)
            global_init = shard_rt.init_state(pq_all, pk_all, v_all)
            local_states = [
                State(H=global_init.H.copy(), g=global_init.g.copy(),
                      pq=pqs[r].copy(), pk=pks[r].copy(), v=vs[r].copy())
                for r in range(workers)
            ]
        else:
            H0 = local_states[0].H.copy()
            g0 = local_states[0].g.copy()
            dH_total = np.zeros_like(H0, dtype=np.int64)
            dg_total = np.zeros_like(g0, dtype=np.int64)
            worker_changes = []
            for rank in range(workers):
                a = rank * local_tokens
                b = a + local_tokens
                local_mask = mask[a:b]
                idx = np.flatnonzero(local_mask)
                worker_changes.append(int(idx.size))
                if idx.size == 0:
                    continue
                pq_new, pk_new, v_new = shard_rt.features(x[a:b][idx])
                st = local_states[rank]
                dHr = np.zeros_like(H0, dtype=np.int64)
                dgr = np.zeros_like(g0, dtype=np.int64)
                for j, li in enumerate(idx.tolist()):
                    dh, dg = local_delta(st.pk[li], st.v[li], pk_new[j], v_new[j])
                    dHr += dh
                    dgr += dg
                    st.pq[li] = pq_new[j]
                    st.pk[li] = pk_new[j]
                    st.v[li] = v_new[j]
                dH_total += dHr
                dg_total += dgr

            H64 = H0.astype(np.int64) + dH_total
            g64 = g0.astype(np.int64) + dg_total
            if (np.max(H64) > np.iinfo(np.int32).max or
                    np.min(H64) < np.iinfo(np.int32).min or
                    np.max(g64) > np.iinfo(np.int32).max or
                    np.min(g64) < np.iinfo(np.int32).min):
                raise OverflowError("H3B reduced H/g leaves int32 range")
            Hn = H64.astype(np.int32)
            gn = g64.astype(np.int32)
            for st in local_states:
                st.H[...] = Hn
                st.g[...] = gn

        y_parts = []
        for rank, st in enumerate(local_states):
            a = rank * local_tokens
            b = a + local_tokens
            y_parts.append(tail(shard_rt, x[a:b], st))
        y_dist = np.concatenate(y_parts, axis=0)

        checks = {
            "y": bool(np.array_equal(y_dist, y_ref)),
            "H": bool(np.array_equal(local_states[0].H, ref_state.H)),
            "g": bool(np.array_equal(local_states[0].g, ref_state.g)),
            "pq": bool(np.array_equal(np.concatenate([s.pq for s in local_states], 0), ref_state.pq)),
            "pk": bool(np.array_equal(np.concatenate([s.pk for s in local_states], 0), ref_state.pk)),
            "v": bool(np.array_equal(np.concatenate([s.v for s in local_states], 0), ref_state.v)),
        }
        if not all(checks.values()):
            raise AssertionError(f"H3B distributed host gate failed step={step}: {checks}")
        rows.append({
            "step": step,
            "changed_tokens": int(mask.sum()),
            "worker_changed_tokens": [int(mask[r*local_tokens:(r+1)*local_tokens].sum()) for r in range(workers)],
            "all_arrays_bit_exact": True,
        })
        last_y = y_ref

    assert ref_state is not None and last_y is not None
    oracle = {
        "y": np.asarray(last_y, dtype="<i4"),
        "H": np.asarray(ref_state.H, dtype="<i4"),
        "g": np.asarray(ref_state.g, dtype="<i4"),
        "pq": np.asarray(ref_state.pq, dtype="<i4"),
        "pk": np.asarray(ref_state.pk, dtype="<i4"),
        "v": np.asarray(ref_state.v, dtype="<i4"),
    }
    gate = {"PASS": True, "rows": rows, "final_arrays_bit_exact": True}
    return gate, oracle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--local-tokens", type=int, default=32)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--change-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=37171)
    ap.add_argument("--pacing-us", type=int, default=1)
    args = ap.parse_args()

    if not (2 <= args.workers <= H3B_MAX_WORKERS):
        raise ValueError("invalid worker count")
    total_tokens = args.workers * args.local_tokens
    meta, arrays = load_hir(args.hir)
    if meta.get("format") != "SNNTransformBlock-v1.1-HIR-H1.1-v2":
        raise ValueError(f"unexpected HIR format {meta.get('format')!r}")
    d, heads, features, ff = (int(meta[k]) for k in ("d", "heads", "features", "ff"))
    if (d, heads, features, ff) != (32, 2, 32, 64):
        raise ValueError("H3B-v1 fixes D32,H2,M32,FF64")

    xs, masks = make_scaling_sequence(
        total_tokens, d, steps=args.steps,
        change_frac=args.change_frac, seed=args.seed)
    gate, oracle = distributed_host_gate(
        meta, arrays, xs, masks,
        workers=args.workers, local_tokens=args.local_tokens)

    hir_blob_u32, desc = build_hir_blob(arrays)
    base_header_u32 = build_base_header(
        meta, desc, local_tokens=args.local_tokens, workers=args.workers,
        steps=args.steps, pacing_us=args.pacing_us)
    xs_i32 = np.ascontiguousarray(np.stack(xs), dtype="<i4")
    masks_u32 = np.ascontiguousarray(np.stack(masks).astype(np.uint32), dtype="<u4")

    per_worker_changes = []
    expected_evals = []
    for rank in range(args.workers):
        a = rank * args.local_tokens
        b = a + args.local_tokens
        counts = [int(m[a:b].sum()) for m in masks]
        per_worker_changes.append(counts)
        expected_evals.append(sum(counts))

    package_meta = {
        "format": "SNNTransform-H3B-single-chip-v1",
        "stage": "H3B0_DISTRIBUTED_CORRECTNESS",
        "contract": "H1.1-functional-v9",
        "source_hir_sha256": sha256_file(args.hir),
        "source_checkpoint_sha256": meta["source_checkpoint_sha256"],
        "workers": args.workers,
        "local_tokens": args.local_tokens,
        "global_tokens": total_tokens,
        "d": d, "heads": heads, "features": features, "ff": ff,
        "steps": args.steps, "change_frac": args.change_frac,
        "seed": args.seed, "pacing_us": args.pacing_us,
        "worker_arena_bytes_requested": arena_bytes(args.local_tokens, d, heads, features, ff),
        "local_trajectory_words": local_trajectory_words(args.local_tokens, d, args.steps),
        "output_words_per_worker": output_word_count(args.local_tokens, d, heads, features),
        "init_gather_payload_words_per_nonzero_worker": init_gather_words(heads, features, d),
        "hg_broadcast_payload_words": bcast_words(heads, features, d),
        "per_worker_changed_tokens": per_worker_changes,
        "expected_qkv_evals_per_worker": expected_evals,
        "host_distributed_gate": gate,
        "scientific_contract": {
            "single_global_block": True,
            "same_chip_required": True,
            "token_sharding": "contiguous equal shards",
            "global_state": "replicated H,g",
            "coordinator": "worker0 communication/reduction only",
            "host_intermediate_compute": 0,
            "host_hidden_state_roundtrips": 0,
            "acceptance": "concatenated y,pq,pk,v and every H,g replica bit-exact to global H2D integer oracle",
        },
    }

    payload = {
        "base_header_u32": base_header_u32,
        "hir_blob_u32": np.ascontiguousarray(hir_blob_u32, dtype="<u4"),
        "trajectory_x_i32": xs_i32,
        "trajectory_masks_u32": masks_u32,
        "package_metadata_json_utf8": package_metadata_bytes(package_meta),
    }
    for name, value in oracle.items():
        payload[f"oracle_{name}"] = np.ascontiguousarray(value, dtype="<i4")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(json.dumps({"PASS": True, "output": str(args.out), **package_meta}, indent=2))


if __name__ == "__main__":
    main()
