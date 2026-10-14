#!/usr/bin/env python3
"""H2D: compose the C kernels on the actual frozen H1.1 functional-v9 HIR.

The test executes a six-step, 25%-changed-token U5-shaped sequence for one
resident sample.  It compares the composed C runtime bit-for-bit against an
independent NumPy integer reference using the frozen HIR arrays.

This is still a host C gate.  It does not claim APLX cross-compilation or board
execution.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spinnaker_backend.h2_aplx.test_h2_kernels import (
    F_ACT, F_PQ, F_PK, F_H, F_MEM,
    build, ptr, rshift_round, div_round, layernorm_ref, stable_a_ref, izh_ref,
)


@dataclass
class State:
    H: np.ndarray
    g: np.ndarray
    pq: np.ndarray
    pk: np.ndarray
    v: np.ndarray


def i32(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.int32))


def u32(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.uint32))


def u16(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.uint16))


def dense_ref(x, W, b):
    acc = np.asarray(x, np.int64) @ np.asarray(W, np.int64).T
    return (rshift_round(acc, F_ACT) + np.asarray(b, np.int64)).astype(np.int32)


def sparse_ref(x, row_ptr, col_idx, values, bias, out_features):
    x = np.asarray(x, np.int64)
    out = np.empty((x.shape[0], out_features), dtype=np.int32)
    for o in range(out_features):
        a, b = int(row_ptr[o]), int(row_ptr[o + 1])
        acc = np.sum(x[:, col_idx[a:b]] * values[a:b][None, :], axis=1, dtype=np.int64)
        out[:, o] = (rshift_round(acc, F_ACT) + int(bias[o])).astype(np.int32)
    return out


def state_init_ref(pk, v):
    tokens, heads, features = pk.shape
    dh = v.shape[-1]
    H = np.empty((heads, features, dh), dtype=np.int32)
    g = np.sum(pk.astype(np.int64), axis=0).astype(np.int32)
    for h in range(heads):
        for m in range(features):
            for d in range(dh):
                raw = np.sum(
                    pk[:, h, m].astype(np.int64) * v[:, h, d].astype(np.int64),
                    dtype=np.int64,
                )
                H[h, m, d] = int(rshift_round(raw, F_PK + F_ACT - F_H))
    return H, g


def read_ref(pq, H, g):
    tokens, heads, features = pq.shape
    dh = H.shape[-1]
    out = np.empty((tokens, heads, dh), dtype=np.int32)
    for n in range(tokens):
        for h in range(heads):
            den = sum(int(pq[n, h, m]) * int(g[h, m]) for m in range(features))
            if den <= 0:
                raise AssertionError("non-positive frozen H/g denominator")
            for d in range(dh):
                num = sum(int(pq[n, h, m]) * int(H[h, m, d]) for m in range(features))
                out[n, h, d] = div_round(num << (F_ACT + F_PK - F_H), den)
    return out


def token_delta_ref(H, g, old_pk, old_v, new_pk, new_v):
    shift = F_PK + F_ACT - F_H
    H -= rshift_round(old_pk[:, :, None].astype(np.int64) * old_v[:, None, :].astype(np.int64), shift).astype(np.int32)
    H += rshift_round(new_pk[:, :, None].astype(np.int64) * new_v[:, None, :].astype(np.int64), shift).astype(np.int32)
    g -= old_pk
    g += new_pk


def load_hir(path: Path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z["metadata_json_utf8"].tolist()).decode("utf-8"))
    arrays = {k: np.asarray(z[k]) for k in z.files if k != "metadata_json_utf8"}
    return meta, arrays


def make_sequence(tokens: int, d: int, *, steps: int = 6, change_frac: float = 0.25, seed: int = 17171):
    rng = np.random.default_rng(seed)
    slots = tokens - 1
    value_dim = 8
    key_dim = slots
    role_idx = key_dim + value_dim
    assert role_idx < d

    def sample():
        x = np.zeros((tokens, d), dtype=np.int32)
        x[:slots, :key_dim] = np.eye(slots, dtype=np.int32) * (1 << F_ACT)
        vals = rng.integers(0, 2, size=(slots, value_dim), dtype=np.int32) * 2 - 1
        x[:slots, key_dim:key_dim + value_dim] = vals * (1 << F_ACT)
        x[:slots, role_idx] = 1 << F_ACT
        qidx = int(rng.integers(0, slots))
        x[-1, qidx] = 1 << F_ACT
        x[-1, role_idx] = -(1 << F_ACT)
        return x

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


class ActualHIRRuntime:
    def __init__(self, lib, meta, A, *, use_c: bool):
        self.lib = lib
        self.meta = meta
        self.A = A
        self.use_c = use_c
        self.d = int(meta["d"])
        self.heads = int(meta["heads"])
        self.features = int(meta["features"])
        self.ff = int(meta["ff"])
        self.dh = self.d // self.heads
        self.scale_q20 = int(round((self.dh ** -0.25) * (1 << F_ACT)))
        self.q_specs = meta["q_specs"]
        self.k_specs = meta["k_specs"]
        self._configure_ctypes()

    def _configure_ctypes(self):
        L = self.lib
        L.h2_layernorm_q20.argtypes = [ct.POINTER(ct.c_int32), ct.c_uint, ct.c_uint, ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32)]
        L.h2_dense_linear_q20.argtypes = [ct.POINTER(ct.c_int32), ct.c_uint, ct.c_uint, ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_uint, ct.POINTER(ct.c_int32)]
        L.h2_sparse_linear_q20.argtypes = [ct.POINTER(ct.c_int32), ct.c_uint, ct.c_uint, ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_uint16), ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_uint, ct.POINTER(ct.c_int32)]
        L.h2_stable_a_q20.argtypes = [ct.POINTER(ct.c_int32), ct.c_uint, ct.c_uint, ct.c_uint, ct.POINTER(ct.c_int32), ct.c_uint, ct.c_int32, ct.c_int, ct.POINTER(ct.c_int32)]
        L.h2_izh_decode_q20.argtypes = [ct.POINTER(ct.c_int32), ct.c_uint, ct.c_int32, ct.c_int32, ct.c_uint, ct.POINTER(ct.c_int32), ct.c_uint, ct.POINTER(ct.c_int32)]
        L.h2_attention_state_init.argtypes = [ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_uint, ct.c_uint, ct.c_uint, ct.c_uint, ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32)]
        L.h2_attention_token_delta.argtypes = [ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_uint, ct.c_uint, ct.c_uint]
        L.h2_attention_read.argtypes = [ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_uint, ct.c_uint, ct.c_uint, ct.c_uint, ct.POINTER(ct.c_int32)]

    def ln(self, x, prefix):
        gamma, beta = i32(self.A[prefix + "_g"]), i32(self.A[prefix + "_b"])
        if not self.use_c:
            return layernorm_ref(x, gamma, beta)
        out = np.empty_like(x, dtype=np.int32)
        self.lib.h2_layernorm_q20(ptr(i32(x), ct.c_int32), x.shape[0], x.shape[1], ptr(gamma, ct.c_int32), ptr(beta, ct.c_int32), ptr(out, ct.c_int32))
        return out

    def dense(self, x, Wn, bn):
        W, b = i32(self.A[Wn]), i32(self.A[bn])
        if not self.use_c:
            return dense_ref(x, W, b)
        out = np.empty((x.shape[0], W.shape[0]), dtype=np.int32)
        self.lib.h2_dense_linear_q20(ptr(i32(x), ct.c_int32), x.shape[0], x.shape[1], ptr(W, ct.c_int32), ptr(b, ct.c_int32), W.shape[0], ptr(out, ct.c_int32))
        return out

    def sparse(self, x, prefix, in_features):
        rp = u32(self.A[prefix + "_row_ptr"])
        ci = u16(self.A[prefix + "_col_idx"])
        val = i32(self.A[prefix + "_values"])
        bias = i32(self.A[prefix + "_bias"])
        out_features = bias.size
        if not self.use_c:
            return sparse_ref(x, rp, ci.astype(np.int64), val.astype(np.int64), bias, out_features)
        out = np.empty((x.shape[0], out_features), dtype=np.int32)
        self.lib.h2_sparse_linear_q20(ptr(i32(x), ct.c_int32), x.shape[0], in_features, ptr(rp, ct.c_uint32), ptr(ci, ct.c_uint16), ptr(val, ct.c_int32), ptr(bias, ct.c_int32), out_features, ptr(out, ct.c_int32))
        return out

    def stable(self, projected, query):
        shaped = i32(projected.reshape(projected.shape[0], self.heads, self.dh))
        omega = i32(self.A["omega"])
        if not self.use_c:
            return stable_a_ref(shaped, omega, self.scale_q20, query)
        out = np.empty((shaped.shape[0], self.heads, self.features), dtype=np.int32)
        self.lib.h2_stable_a_q20(ptr(shaped, ct.c_int32), shaped.shape[0], self.heads, self.dh, ptr(omega, ct.c_int32), self.features, self.scale_q20, int(query), ptr(out, ct.c_int32))
        return out

    def phi(self, a, query):
        specs = self.q_specs if query else self.k_specs
        out_frac = F_PQ if query else F_PK
        out = np.empty_like(a, dtype=np.int32)
        for h, spec in enumerate(specs):
            bias, gain, T = float(spec[0]), float(spec[1]), int(spec[2])
            dec = i32(self.A[("q" if query else "k") + f"_decoder_{h}"])
            ah = i32(a[:, h, :].reshape(-1))
            if not self.use_c:
                ph = izh_ref(ah, bias, gain, T, dec, out_frac)
            else:
                ph = np.empty(ah.size, dtype=np.int32)
                self.lib.h2_izh_decode_q20(
                    ptr(ah, ct.c_int32), ah.size,
                    int(round(bias * (1 << F_MEM))), int(round(gain * (1 << F_MEM))), T,
                    ptr(dec, ct.c_int32), out_frac, ptr(ph, ct.c_int32),
                )
            out[:, h, :] = ph.reshape(a.shape[0], self.features)
        return out

    def features(self, xq):
        xn = self.ln(xq, "ln1")
        q = self.dense(xn, "Wq", "bq")
        k = self.dense(xn, "Wk", "bk")
        v = self.dense(xn, "Wv", "bv").reshape(xq.shape[0], self.heads, self.dh)
        pq = self.phi(self.stable(q, True), True)
        pk = self.phi(self.stable(k, False), False)
        return pq, pk, i32(v)

    def init_state(self, pq, pk, v):
        if not self.use_c:
            H, g = state_init_ref(pk, v)
        else:
            H = np.empty((self.heads, self.features, self.dh), dtype=np.int32)
            g = np.empty((self.heads, self.features), dtype=np.int32)
            self.lib.h2_attention_state_init(ptr(i32(pk), ct.c_int32), ptr(i32(v), ct.c_int32), pk.shape[0], self.heads, self.features, self.dh, ptr(H, ct.c_int32), ptr(g, ct.c_int32))
        return State(H=H, g=g, pq=pq.copy(), pk=pk.copy(), v=v.copy())

    def update_token(self, st, idx, new_pq, new_pk, new_v):
        old_pk, old_v = st.pk[idx].copy(), st.v[idx].copy()
        if not self.use_c:
            token_delta_ref(st.H, st.g, old_pk, old_v, new_pk, new_v)
        else:
            self.lib.h2_attention_token_delta(ptr(st.H, ct.c_int32), ptr(st.g, ct.c_int32), ptr(i32(old_pk), ct.c_int32), ptr(i32(old_v), ct.c_int32), ptr(i32(new_pk), ct.c_int32), ptr(i32(new_v), ct.c_int32), self.heads, self.features, self.dh)
        st.pq[idx] = new_pq
        st.pk[idx] = new_pk
        st.v[idx] = new_v

    def read(self, st):
        if not self.use_c:
            return read_ref(st.pq, st.H, st.g)
        out = np.empty((st.pq.shape[0], self.heads, self.dh), dtype=np.int32)
        self.lib.h2_attention_read(ptr(i32(st.pq), ct.c_int32), ptr(st.H, ct.c_int32), ptr(st.g, ct.c_int32), st.pq.shape[0], self.heads, self.features, self.dh, ptr(out, ct.c_int32))
        return out

    def step(self, xq, state, changed):
        if state is None:
            pq, pk, v = self.features(xq)
            state = self.init_state(pq, pk, v)
        else:
            idx = np.flatnonzero(changed)
            if idx.size:
                pq, pk, v = self.features(xq[idx])
                for j, token in enumerate(idx.tolist()):
                    self.update_token(state, token, pq[j], pk[j], v[j])

        attended = self.read(state).reshape(xq.shape[0], self.d)
        ao = self.sparse(attended, "Wo", self.d)
        z64 = xq.astype(np.int64) + ao.astype(np.int64)
        assert np.max(np.abs(z64)) <= np.iinfo(np.int32).max
        z = z64.astype(np.int32)
        n2 = self.ln(z, "ln2")
        pre = self.sparse(n2, "W1", self.d)
        hidden = np.maximum(pre, 0).astype(np.int32)
        ff2 = self.sparse(hidden, "W2", self.ff)
        y64 = z.astype(np.int64) + ff2.astype(np.int64)
        assert np.max(np.abs(y64)) <= np.iinfo(np.int32).max
        return y64.astype(np.int32), state


def assert_state_equal(a: State, b: State, step: int):
    for name in ("H", "g", "pq", "pk", "v"):
        aa, bb = getattr(a, name), getattr(b, name)
        if not np.array_equal(aa, bb):
            diff = int(np.max(np.abs(aa.astype(np.int64) - bb.astype(np.int64))))
            raise AssertionError(f"step {step} state {name} mismatch max_abs={diff}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hir", type=Path, required=True)
    args = ap.parse_args()

    meta, A = load_hir(args.hir)
    assert meta["format"] == "SNNTransformBlock-v1.1-HIR-H1.1-v2"
    assert int(meta["F_ACT"]) == F_ACT and int(meta["F_PQ"]) == F_PQ
    assert int(meta["F_PK"]) == F_PK and int(meta["F_H"]) == F_H

    lib = build()
    ref = ActualHIRRuntime(lib, meta, A, use_c=False)
    cand = ActualHIRRuntime(lib, meta, A, use_c=True)
    xs, masks = make_sequence(16, int(meta["d"]), steps=6, change_frac=0.25, seed=17171)

    sr = sc = None
    rows = []
    for step, (x, mask) in enumerate(zip(xs, masks)):
        yr, sr = ref.step(x, sr, mask)
        yc, sc = cand.step(x, sc, mask)
        max_abs = int(np.max(np.abs(yr.astype(np.int64) - yc.astype(np.int64))))
        if max_abs != 0:
            raise AssertionError(f"step {step} output mismatch max_abs={max_abs}")
        assert_state_equal(sr, sc, step)
        rows.append({"step": step, "changed_tokens": int(mask.sum()), "output_max_abs_diff": max_abs})

    print(json.dumps({
        "H2D_ACTUAL_HIR_FULL_STEP_GATE": "PASS",
        "contract": "H1.1-functional-v9",
        "source_checkpoint_sha256": meta["source_checkpoint_sha256"],
        "steps": rows,
        "final_output_max_abs_diff": 0,
        "final_state_bit_exact": True,
        "claim_boundary": "host C parity only; no APLX/board claim",
    }, indent=2))


if __name__ == "__main__":
    main()
