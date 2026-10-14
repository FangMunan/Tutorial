#!/usr/bin/env python3
"""H1.1 — export the frozen U5 block into the H1 integer HIR and validate parity.

Scientific scope
----------------
The frozen SNNTransformBlock-v1.1 operator graph is unchanged.  PyTorch is used
only on the offline/reference side to load the accepted checkpoint and to
measure parity.  The candidate execution path is the integer-only H1 runtime in
``h1_fixed_point_emulator.py``.

This gate closes the gap left by H1, which used representative parameters.  H1.1
uses the actual U5 trained weights, masks, LayerNorm parameters, random-feature
matrix, and per-head Izh decoders/specifications.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from SNNTransformBlock_v1_1.SNNTransformBlock_v1_1 import (
    PerHeadIzhPhi,
    _stable_a,
    load_v1_checkpoint_as_v1_1,
)
from spinnaker_backend.h1_fixed_point_emulator import (
    F_ACT,
    F_DEC,
    F_PHI,
    FixedPointBlockRuntime,
    HIR,
    SparseRowsQ,
    deq,
    qround,
)


# -----------------------------------------------------------------------------
# Offline checkpoint -> hardware IR compiler
# -----------------------------------------------------------------------------

def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float64, copy=True)


def _rows_from_sparse_linear(layer) -> SparseRowsQ:
    W = _np(layer.weight)
    mask = _np(layer.mask) > 0.5
    b = _np(layer.bias) if layer.bias is not None else np.zeros(W.shape[0], dtype=np.float64)
    row_ptr = [0]
    col_idx: List[int] = []
    vals: List[float] = []
    for o in range(W.shape[0]):
        cols = np.nonzero(mask[o])[0]
        col_idx.extend(int(c) for c in cols)
        vals.extend(float(W[o, c]) for c in cols)
        row_ptr.append(len(col_idx))
    return SparseRowsQ(
        row_ptr=np.asarray(row_ptr, dtype=np.int32),
        col_idx=np.asarray(col_idx, dtype=np.int32),
        values=qround(np.asarray(vals, dtype=np.float64), F_ACT),
        bias=qround(b, F_ACT),
        out_features=W.shape[0],
        in_features=W.shape[1],
    )


def export_hir(block) -> HIR:
    if not isinstance(block.attn.phi_q, PerHeadIzhPhi) or not isinstance(block.attn.phi_k, PerHeadIzhPhi):
        raise TypeError("H1.1 requires the frozen per-head Izh interface")
    oq = _np(block.attn.phi_q.omega)
    ok = _np(block.attn.phi_k.omega)
    if not np.array_equal(oq, ok):
        raise ValueError("frozen v1.1 expects tied Q/K random-feature omega")

    q_specs = [
        (float(s["bias"]), float(s["gain"]), int(s["window_ms"]))
        for s in block.attn.phi_q.specs
    ]
    k_specs = [
        (float(s["bias"]), float(s["gain"]), int(s["window_ms"]))
        for s in block.attn.phi_k.specs
    ]
    q_decoders = [
        qround(_np(getattr(block.attn.phi_q, f"decoder_{h}")), F_DEC)
        for h in range(block.heads)
    ]
    k_decoders = [
        qround(_np(getattr(block.attn.phi_k, f"decoder_{h}")), F_DEC)
        for h in range(block.heads)
    ]

    return HIR(
        d=block.d,
        heads=block.heads,
        features=block.attn.features,
        ff=block.ff,
        ln1_g=qround(_np(block.n1.weight), F_ACT),
        ln1_b=qround(_np(block.n1.bias), F_ACT),
        ln2_g=qround(_np(block.n2.weight), F_ACT),
        ln2_b=qround(_np(block.n2.bias), F_ACT),
        Wq=qround(_np(block.attn.q.weight), F_ACT),
        bq=qround(_np(block.attn.q.bias), F_ACT),
        Wk=qround(_np(block.attn.k.weight), F_ACT),
        bk=qround(_np(block.attn.k.bias), F_ACT),
        Wv=qround(_np(block.attn.v.weight), F_ACT),
        bv=qround(_np(block.attn.v.bias), F_ACT),
        omega=qround(oq, F_ACT),
        q_specs=q_specs,
        k_specs=k_specs,
        q_decoders=q_decoders,
        k_decoders=k_decoders,
        Wo=_rows_from_sparse_linear(block.attn.out),
        W1=_rows_from_sparse_linear(block.f1),
        W2=_rows_from_sparse_linear(block.f2),
    )


def save_hir_npz(h: HIR, path: Path, source_sha256: str) -> None:
    arrays: Dict[str, np.ndarray] = {
        "ln1_g": h.ln1_g, "ln1_b": h.ln1_b,
        "ln2_g": h.ln2_g, "ln2_b": h.ln2_b,
        "Wq": h.Wq, "bq": h.bq, "Wk": h.Wk, "bk": h.bk,
        "Wv": h.Wv, "bv": h.bv, "omega": h.omega,
    }
    for prefix, rows in (("Wo", h.Wo), ("W1", h.W1), ("W2", h.W2)):
        arrays[f"{prefix}_row_ptr"] = rows.row_ptr
        arrays[f"{prefix}_col_idx"] = rows.col_idx
        arrays[f"{prefix}_values"] = rows.values
        arrays[f"{prefix}_bias"] = rows.bias
    for i, d in enumerate(h.q_decoders):
        arrays[f"q_decoder_{i}"] = d
    for i, d in enumerate(h.k_decoders):
        arrays[f"k_decoder_{i}"] = d
    meta = {
        "format": "SNNTransformBlock-v1.1-HIR-H1.1",
        "source_checkpoint_sha256": source_sha256,
        "d": h.d, "heads": h.heads, "features": h.features, "ff": h.ff,
        "F_ACT": F_ACT, "F_PHI": F_PHI, "F_DEC": F_DEC,
        "q_specs": h.q_specs, "k_specs": h.k_specs,
        "Wo_nnz": h.Wo.nnz, "W1_nnz": h.W1.nnz, "W2_nnz": h.W2.nnz,
    }
    arrays["metadata_json_utf8"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, **arrays)


# -----------------------------------------------------------------------------
# U5 associative-distribution generator (copied as a small dependency-free
# specification so the H1.1 compiler does not import historical experiment code)
# -----------------------------------------------------------------------------

class AssocGenerator:
    def __init__(self, tokens: int, d: int, value_dim: int):
        self.tokens = tokens
        self.slots = tokens - 1
        self.d = d
        self.value_dim = value_dim
        self.key_dim = self.slots
        if self.key_dim + value_dim + 1 > d:
            raise ValueError("d too small for U5 associative probe")
        self.role_idx = self.key_dim + value_dim
        self.keys = torch.eye(self.slots, dtype=torch.float32)

    def sample(self, batch: int, g: torch.Generator):
        B, S, D, kd, vd = batch, self.slots, self.d, self.key_dim, self.value_dim
        x = torch.zeros(B, self.tokens, D)
        x[:, :S, :kd] = self.keys[None]
        vals = torch.randint(0, 2, (B, S, vd), generator=g).float() * 2.0 - 1.0
        x[:, :S, kd:kd + vd] = vals
        x[:, :S, self.role_idx] = 1.0
        qidx = torch.randint(0, S, (B,), generator=g)
        x[:, -1, :kd] = self.keys[qidx]
        x[:, -1, self.role_idx] = -1.0
        target = vals[torch.arange(B), qidx]
        return x, target


def make_mutable_sequence(gen: AssocGenerator, *, batch: int, steps: int, change_frac: float, seed: int):
    g = torch.Generator(); g.manual_seed(seed)
    x, target0 = gen.sample(batch, g)
    xs = [x]
    masks = [torch.ones(batch, gen.tokens, dtype=torch.bool)]
    targets = [target0]
    nchange = max(1, int(round(gen.tokens * change_frac)))
    for _ in range(1, steps):
        fresh, tgt = gen.sample(batch, g)
        nxt = xs[-1].clone()
        mask = torch.zeros(batch, gen.tokens, dtype=torch.bool)
        for b in range(batch):
            idx = torch.randperm(gen.tokens, generator=g)[:nchange]
            nxt[b, idx] = fresh[b, idx]
            mask[b, idx] = True
        xs.append(nxt); masks.append(mask); targets.append(tgt)
    return xs, masks, targets


# -----------------------------------------------------------------------------
# Range and parity helpers
# -----------------------------------------------------------------------------

def rel_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def _mx(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().abs().max().cpu())
    x = np.asarray(x)
    return float(np.max(np.abs(x))) if x.size else 0.0


def profile_float_step(block, x: torch.Tensor, state, changed) -> Dict[str, float]:
    # Offline diagnostic only; it mirrors the frozen equations but does not
    # participate in the integer candidate execution path.
    with torch.no_grad():
        xn = block.n1(x)
        q = block.attn._split(block.attn.q(xn))
        k = block.attn._split(block.attn.k(xn))
        v = block.attn._split(block.attn.v(xn))
        aq = _stable_a(q, block.attn.phi_q.omega, block.attn.head_dim, True)
        ak = _stable_a(k, block.attn.phi_k.omega, block.attn.head_dim, False)
        pq = block.attn.phi_q(q, is_query=True)
        pk = block.attn.phi_k(k, is_query=False)
        y, st = block(x, state, changed)
        num = torch.einsum("bhnm,bhmd->bhnd", st.pq, st.H)
        den = torch.einsum("bhnm,bhm->bhn", st.pq, st.g).unsqueeze(-1).clamp_min(1e-8)
        attended = num / den
        ao = block.attn.out(attended.transpose(1, 2).reshape(x.shape[0], x.shape[1], block.d))
        z = x + ao
        n2 = block.n2(z)
        pre = block.f1(n2)
        hidden = F.relu(pre)
        ff2 = block.f2(hidden)
        return {
            "x": _mx(x), "ln1": _mx(xn), "q": _mx(q), "k": _mx(k), "v_projection": _mx(v),
            "a_q": _mx(aq), "a_k": _mx(ak), "pq": _mx(pq), "pk": _mx(pk),
            "H": _mx(st.H), "g": _mx(st.g), "attended": _mx(attended),
            "attn_out": _mx(ao), "z": _mx(z), "ln2": _mx(n2), "ff1_pre": _mx(pre),
            "ff_hidden": _mx(hidden), "ff2": _mx(ff2), "y": _mx(y),
        }


def merge_max(dst: Dict[str, float], src: Dict[str, float]):
    for k, v in src.items():
        dst[k] = max(float(dst.get(k, 0.0)), float(v))


def state_integer_max(st) -> Dict[str, int]:
    return {k: int(np.max(np.abs(getattr(st, k)))) for k in ("H", "g", "pq", "pk", "v")}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("spinnaker_backend/run/H1_1"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--change-frac", type=float, default=0.25)
    ap.add_argument("--value-dim", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17171)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    block, checkpoint = load_v1_checkpoint_as_v1_1(args.checkpoint, device="cpu", eval_mode=True)
    source_sha = sha256_file(args.checkpoint)
    hir = export_hir(block)
    save_hir_npz(hir, args.out / "SNNTransformBlock_v1_1_HIR_H1_1.npz", source_sha)

    gen = AssocGenerator(tokens=16, d=block.d, value_dim=args.value_dim)
    xs, masks, targets = make_mutable_sequence(
        gen, batch=args.batch, steps=args.steps, change_frac=args.change_frac, seed=args.seed
    )

    runtime = FixedPointBlockRuntime(hir)
    float_state = None
    int_state = None
    rows = []
    ranges: Dict[str, float] = {}
    int_max = {k: 0 for k in ("H", "g", "pq", "pk", "v")}

    readout = torch.nn.Linear(block.d, args.value_dim)
    readout.load_state_dict(checkpoint["readout_state_dict"])
    readout.eval()
    pred_float0 = pred_int0 = None

    with torch.no_grad():
        for t, (x, changed) in enumerate(zip(xs, masks)):
            prof = profile_float_step(block, x, float_state, None if float_state is None else changed)
            merge_max(ranges, prof)
            yf, float_state = block(x, float_state, None if float_state is None else changed)

            yi_q, int_state = runtime.step(
                qround(x.numpy(), F_ACT), int_state,
                None if int_state is None else changed.numpy(),
            )
            yi = deq(yi_q, F_ACT)
            state_float = {
                "H": float_state.H.detach().numpy(),
                "g": float_state.g.detach().numpy(),
                "pq": float_state.pq.detach().numpy(),
                "pk": float_state.pk.detach().numpy(),
                "v": float_state.v.detach().numpy(),
            }
            state_int = {
                "H": deq(int_state.H, F_PHI + F_ACT),
                "g": deq(int_state.g, F_PHI),
                "pq": deq(int_state.pq, F_PHI),
                "pk": deq(int_state.pk, F_PHI),
                "v": deq(int_state.v, F_ACT),
            }
            im = state_integer_max(int_state)
            for k, v in im.items(): int_max[k] = max(int_max[k], v)

            row = {
                "step": t,
                "output_rel_l2": rel_np(yi, yf.detach().numpy()),
                "H_rel_l2": rel_np(state_int["H"], state_float["H"]),
                "g_rel_l2": rel_np(state_int["g"], state_float["g"]),
                "pq_rel_l2": rel_np(state_int["pq"], state_float["pq"]),
                "pk_rel_l2": rel_np(state_int["pk"], state_float["pk"]),
                "v_rel_l2": rel_np(state_int["v"], state_float["v"]),
                "changed_tokens": int(changed.sum()),
            }
            rows.append(row)

            if t == 0:
                pf = readout(yf[:, -1])
                pi = readout(torch.from_numpy(yi).float()[:, -1])
                pred_float0 = pf
                pred_int0 = pi
                target0 = targets[0]

    maxima = {
        key: max(r[f"{key}_rel_l2"] for r in rows)
        for key in ("output", "H", "g", "pq", "pk", "v")
    }
    dense = args.batch * 16 * args.steps
    event_expected = args.batch * 16 + (args.steps - 1) * args.batch * max(1, round(16 * args.change_frac))
    event_stats = {
        "q": runtime.stats.q_token_evals,
        "k": runtime.stats.k_token_evals,
        "v": runtime.stats.v_token_evals,
        "dense_per_projection": dense,
        "expected_event_per_projection": event_expected,
        "reduction": 1.0 - runtime.stats.q_token_evals / dense,
    }

    pred_rel = rel_np(pred_int0.detach().numpy(), pred_float0.detach().numpy())
    float_sign = float(((pred_float0 >= 0) == (target0 >= 0)).float().mean())
    int_sign = float(((pred_int0 >= 0) == (target0 >= 0)).float().mean())

    int32_lim = (1 << 31) - 1
    gates = {
        "actual_frozen_checkpoint_loaded": checkpoint.get("format") == "SNNTransformBlock-v1",
        "full_block_output_rel_l2_le_1pct": maxima["output"] <= 0.01,
        "H_rel_l2_le_1pct": maxima["H"] <= 0.01,
        "g_rel_l2_le_1pct": maxima["g"] <= 0.01,
        "pq_rel_l2_le_2pct": maxima["pq"] <= 0.02,
        "pk_rel_l2_le_1pct": maxima["pk"] <= 0.01,
        "v_rel_l2_le_0p5pct": maxima["v"] <= 0.005,
        "readout_prediction_rel_l2_le_1pct": pred_rel <= 0.01,
        "task_sign_drop_le_0p02": int_sign >= float_sign - 0.02,
        "event_count_exact": runtime.stats.q_token_evals == event_expected == runtime.stats.k_token_evals == runtime.stats.v_token_evals,
        "event_reduction_ge_0p60": event_stats["reduction"] >= 0.60,
        "tail_nnz_exact_25pct": (hir.Wo.nnz + hir.W1.nnz + hir.W2.nnz) == 1280,
        "resident_state_int32_safe": max(int_max.values()) <= int32_lim,
    }

    result = {
        "stage": "H1_1_FROZEN_CHECKPOINT_TO_HIR",
        "PASS": all(gates.values()),
        "source": {
            "artifact_run_id": 38538624735,
            "artifact_name": "SNN_TRANSFORM_BLOCK_V1",
            "checkpoint_name": args.checkpoint.name,
            "checkpoint_sha256": source_sha,
            "checkpoint_format": checkpoint.get("format"),
        },
        "scientific_scope": {
            "architecture": "frozen v1.1 operator graph unchanged",
            "parameters": "actual trained U5 checkpoint, migrated exactly to clean v1.1",
            "candidate_runtime": "H1 NumPy integer-only HIR; PyTorch only reference/compiler side",
            "board_claim": "none; still pre-APLX",
        },
        "formats": {"F_ACT": F_ACT, "F_PHI": F_PHI, "F_DEC": F_DEC, "H_frac_bits": F_ACT + F_PHI},
        "max_rel_l2": maxima,
        "trajectory": rows,
        "task_probe_step0": {
            "prediction_rel_l2": pred_rel,
            "float_sign_accuracy": float_sign,
            "integer_sign_accuracy": int_sign,
        },
        "real_trained_float_max_abs": ranges,
        "resident_integer_max_abs": int_max,
        "event_execution": event_stats,
        "sparse_tail": {
            "Wo_nnz": hir.Wo.nnz, "W1_nnz": hir.W1.nnz, "W2_nnz": hir.W2.nnz,
            "total_nnz": hir.Wo.nnz + hir.W1.nnz + hir.W2.nnz,
            "dense_total": 5120,
            "static_fraction": (hir.Wo.nnz + hir.W1.nnz + hir.W2.nnz) / 5120.0,
        },
        "gates": gates,
        "decision": (
            "H1.1 PASS: freeze actual-checkpoint HIR and proceed to H2 APLX kernels"
            if all(gates.values()) else
            "H1.1 FAIL: localize numerical/range failure before writing APLX"
        ),
    }
    (args.out / "H1_1_RESULTS.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if not result["PASS"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
