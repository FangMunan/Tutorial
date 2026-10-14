#!/usr/bin/env python3
"""Host-native bit-exact gate for the H2 C/APLX arithmetic kernels."""
from __future__ import annotations

import ctypes as ct
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SO = ROOT / "libh2kernels.so"
F_ACT, F_PQ, F_PK, F_H = 20, 24, 16, 20


def rshift_round(x, shift):
    x = np.asarray(x, dtype=np.int64)
    if shift <= 0:
        return x << (-shift)
    add = 1 << (shift - 1)
    return np.where(x >= 0, (x + add) >> shift, -((-x + add) >> shift))


def div_round(n: int, d: int) -> int:
    if d <= 0:
        return 0
    if n >= 0:
        return (n + d // 2) // d
    return -((-n + d // 2) // d)


def build() -> ct.CDLL:
    subprocess.run([
        "gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
        "-shared", "-fPIC", str(ROOT / "h2_kernels.c"), "-o", str(SO),
    ], check=True)
    return ct.CDLL(str(SO))


def ptr(a, ctype):
    return a.ctypes.data_as(ct.POINTER(ctype))


def test_scaled_div(lib):
    lib.h2_div_scaled_round_i64.argtypes = [ct.c_int64, ct.c_int64, ct.c_uint]
    lib.h2_div_scaled_round_i64.restype = ct.c_int32
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(5000):
        den = int(rng.integers(10**9, 10**15, dtype=np.int64))
        num = int(rng.integers(-5 * 10**16, 5 * 10**16, dtype=np.int64))
        ref = div_round(num << 16, den)
        if not (-(1 << 31) <= ref <= (1 << 31) - 1):
            continue
        got = int(lib.h2_div_scaled_round_i64(num, den, 16))
        assert got == ref, (num, den, got, ref)
        checked += 1
    assert checked > 1000
    return checked


def test_outer_and_read(lib):
    lib.h2_attention_token_delta.argtypes = [
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
        ct.c_uint, ct.c_uint, ct.c_uint,
    ]
    lib.h2_attention_read.argtypes = [
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
        ct.c_uint, ct.c_uint, ct.c_uint, ct.c_uint, ct.POINTER(ct.c_int32),
    ]

    rng = np.random.default_rng(22)
    heads, features, dh, tokens = 2, 32, 16, 16

    # Build a realistic resident state from 16 positive K features and Q20 V.
    pk_tokens = rng.integers(100, 2_000_000, size=(tokens, heads, features), dtype=np.int32)
    v_tokens = rng.integers(-3_000_000, 3_000_000, size=(tokens, heads, dh), dtype=np.int32)
    H64 = np.zeros((heads, features, dh), dtype=np.int64)
    g64 = np.zeros((heads, features), dtype=np.int64)
    for n in range(tokens):
        prod = pk_tokens[n, :, :, None].astype(np.int64) * v_tokens[n, :, None, :].astype(np.int64)
        H64 += rshift_round(prod, F_PK + F_ACT - F_H)
        g64 += pk_tokens[n]
    H = H64.astype(np.int32)
    g = g64.astype(np.int32)

    old_pk = pk_tokens[3].copy()
    old_v = v_tokens[3].copy()
    new_pk = rng.integers(100, 2_000_000, size=(heads, features), dtype=np.int32)
    new_v = rng.integers(-3_000_000, 3_000_000, size=(heads, dh), dtype=np.int32)

    H_ref = H.astype(np.int64)
    g_ref = g.astype(np.int64)
    H_ref -= rshift_round(old_pk[:, :, None].astype(np.int64) * old_v[:, None, :].astype(np.int64), 16)
    H_ref += rshift_round(new_pk[:, :, None].astype(np.int64) * new_v[:, None, :].astype(np.int64), 16)
    g_ref -= old_pk.astype(np.int64)
    g_ref += new_pk.astype(np.int64)

    H_c = np.ascontiguousarray(H.copy())
    g_c = np.ascontiguousarray(g.copy())
    lib.h2_attention_token_delta(
        ptr(H_c, ct.c_int32), ptr(g_c, ct.c_int32),
        ptr(np.ascontiguousarray(old_pk), ct.c_int32), ptr(np.ascontiguousarray(old_v), ct.c_int32),
        ptr(np.ascontiguousarray(new_pk), ct.c_int32), ptr(np.ascontiguousarray(new_v), ct.c_int32),
        heads, features, dh,
    )
    np.testing.assert_array_equal(H_c, H_ref.astype(np.int32))
    np.testing.assert_array_equal(g_c, g_ref.astype(np.int32))

    pq = rng.integers(1, 3_000_000, size=(tokens, heads, features), dtype=np.int32)
    ref = np.empty((tokens, heads, dh), dtype=np.int32)
    for n in range(tokens):
        for h in range(heads):
            den = sum(int(pq[n, h, m]) * int(g_c[h, m]) for m in range(features))
            assert den > 0
            for d in range(dh):
                num = sum(int(pq[n, h, m]) * int(H_c[h, m, d]) for m in range(features))
                val = div_round(num << (F_ACT + F_PK - F_H), den)
                assert -(1 << 31) <= val <= (1 << 31) - 1
                ref[n, h, d] = val

    got = np.empty_like(ref)
    lib.h2_attention_read(
        ptr(np.ascontiguousarray(pq), ct.c_int32),
        ptr(H_c, ct.c_int32), ptr(g_c, ct.c_int32),
        tokens, heads, features, dh, ptr(got, ct.c_int32),
    )
    np.testing.assert_array_equal(got, ref)
    return int(np.max(np.abs(got.astype(np.int64) - ref.astype(np.int64))))


def test_sparse(lib):
    lib.h2_sparse_linear_q20.argtypes = [
        ct.POINTER(ct.c_int32), ct.c_uint, ct.c_uint,
        ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_uint16),
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
        ct.c_uint, ct.POINTER(ct.c_int32),
    ]
    rng = np.random.default_rng(33)
    rows, in_f, out_f, fanin = 17, 32, 64, 8
    x = rng.integers(-2_000_000, 2_000_000, size=(rows, in_f), dtype=np.int32)
    row_ptr = np.arange(0, (out_f + 1) * fanin, fanin, dtype=np.uint32)
    col_idx = np.empty(out_f * fanin, dtype=np.uint16)
    values = rng.integers(-300_000, 300_000, size=out_f * fanin, dtype=np.int32)
    bias = rng.integers(-100_000, 100_000, size=out_f, dtype=np.int32)
    for o in range(out_f):
        col_idx[o * fanin:(o + 1) * fanin] = np.sort(rng.choice(in_f, fanin, replace=False)).astype(np.uint16)

    ref = np.empty((rows, out_f), dtype=np.int32)
    for r in range(rows):
        for o in range(out_f):
            a, b = int(row_ptr[o]), int(row_ptr[o + 1])
            acc = np.sum(x[r, col_idx[a:b]].astype(np.int64) * values[a:b].astype(np.int64), dtype=np.int64)
            ref[r, o] = int(rshift_round(acc, F_ACT)) + int(bias[o])

    got = np.empty_like(ref)
    lib.h2_sparse_linear_q20(
        ptr(np.ascontiguousarray(x), ct.c_int32), rows, in_f,
        ptr(row_ptr, ct.c_uint32), ptr(col_idx, ct.c_uint16),
        ptr(values, ct.c_int32), ptr(bias, ct.c_int32), out_f,
        ptr(got, ct.c_int32),
    )
    np.testing.assert_array_equal(got, ref)
    return int(values.size)


def main():
    lib = build()
    ndiv = test_scaled_div(lib)
    read_diff = test_outer_and_read(lib)
    nnz = test_sparse(lib)
    print({
        "H2_C_KERNEL_GATE": "PASS",
        "scaled_div_cases": ndiv,
        "attention_read_max_abs_diff": read_diff,
        "sparse_test_nnz": nnz,
        "contract": "H1.1-functional-v9",
    })


if __name__ == "__main__":
    main()
