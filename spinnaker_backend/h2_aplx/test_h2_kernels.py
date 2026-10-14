#!/usr/bin/env python3
"""Host-native bit-exact gate for the H2 C/APLX arithmetic kernels."""
from __future__ import annotations

import ctypes as ct
import math
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SO = ROOT / "libh2kernels.so"
F_ACT, F_PQ, F_PK, F_H, F_MEM, F_DEC = 20, 24, 16, 20, 22, 20


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


def layernorm_ref(x, gamma, beta):
    x = np.asarray(x, dtype=np.int64)
    rows, width = x.shape
    out = np.empty_like(x)
    eps_q40 = int(round(1e-5 * (1 << (2 * F_ACT))))
    for r in range(rows):
        row = x[r]
        mean = int(round(int(row.sum()) / width))
        c = row - mean
        var_q40 = int(np.sum(c * c, dtype=np.int64) // width) + eps_q40
        denom = math.isqrt(max(var_q40, 1))
        norm = np.asarray([div_round(int(v) << F_ACT, denom) for v in c], dtype=np.int64)
        out[r] = rshift_round(norm * gamma.astype(np.int64), F_ACT) + beta
    return out.astype(np.int32)


def test_layernorm(lib):
    lib.h2_layernorm_q20.argtypes = [
        ct.POINTER(ct.c_int32), ct.c_uint, ct.c_uint,
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
    ]
    rng = np.random.default_rng(44)
    rows, width = 37, 32
    x = rng.integers(-5_000_000, 5_000_000, size=(rows, width), dtype=np.int32)
    gamma = rng.integers(700_000, 1_400_000, size=width, dtype=np.int32)
    beta = rng.integers(-100_000, 100_000, size=width, dtype=np.int32)
    ref = layernorm_ref(x, gamma, beta)
    got = np.empty_like(ref)
    lib.h2_layernorm_q20(
        ptr(np.ascontiguousarray(x), ct.c_int32), rows, width,
        ptr(np.ascontiguousarray(gamma), ct.c_int32),
        ptr(np.ascontiguousarray(beta), ct.c_int32),
        ptr(got, ct.c_int32),
    )
    np.testing.assert_array_equal(got, ref)
    return int(np.max(np.abs(got.astype(np.int64) - ref.astype(np.int64))))


def stable_a_ref(x, omega, scale_q20, is_query):
    x = np.asarray(x, dtype=np.int64)
    omega = np.asarray(omega, dtype=np.int64)
    tokens, heads, dh = x.shape
    features = omega.shape[1]
    out = np.empty((tokens, heads, features), dtype=np.int64)
    xx = rshift_round(x * int(scale_q20), F_ACT)
    for n in range(tokens):
        for h in range(heads):
            dash = rshift_round(xx[n, h] @ omega[h].T, F_ACT)
            diag = rshift_round(np.sum(xx[n, h] * xx[n, h], dtype=np.int64), F_ACT + 1)
            a = dash - diag
            if is_query:
                a = np.clip(a - np.max(a), -30 * (1 << F_ACT), 0)
            else:
                a = np.clip(a, -30 * (1 << F_ACT), 30 * (1 << F_ACT))
            out[n, h] = a
    return out.astype(np.int32)


def test_stable_a(lib):
    lib.h2_stable_a_q20.argtypes = [
        ct.POINTER(ct.c_int32), ct.c_uint, ct.c_uint, ct.c_uint,
        ct.POINTER(ct.c_int32), ct.c_uint, ct.c_int32, ct.c_int,
        ct.POINTER(ct.c_int32),
    ]
    rng = np.random.default_rng(55)
    tokens, heads, dh, features = 23, 2, 16, 32
    x = rng.integers(-3_000_000, 3_000_000, size=(tokens, heads, dh), dtype=np.int32)
    omega = rng.integers(-3_000_000, 3_000_000, size=(heads, features, dh), dtype=np.int32)
    scale_q20 = int(round((dh ** -0.25) * (1 << F_ACT)))
    maxdiff = 0
    for query in (0, 1):
        ref = stable_a_ref(x, omega, scale_q20, bool(query))
        got = np.empty_like(ref)
        lib.h2_stable_a_q20(
            ptr(np.ascontiguousarray(x), ct.c_int32), tokens, heads, dh,
            ptr(np.ascontiguousarray(omega), ct.c_int32), features,
            scale_q20, query, ptr(got, ct.c_int32),
        )
        np.testing.assert_array_equal(got, ref)
        maxdiff = max(maxdiff, int(np.max(np.abs(got.astype(np.int64) - ref.astype(np.int64)))))
    return maxdiff


def izh_ref(a, bias, gain, T, decoder_q, out_frac):
    a = np.asarray(a, dtype=np.int64)
    decoder_q = np.asarray(decoder_q, dtype=np.int64)
    S = 1 << F_MEM
    c02v = int(round(0.2 * S))
    c004 = int(round(0.04 * S))
    c02 = int(round(0.02 * S))
    bias_q = int(round(bias * S))
    gain_q = int(round(gain * S))
    out = np.empty(a.size, dtype=np.int64)
    for i, aq in enumerate(a.reshape(-1)):
        a_mem = int(aq) << (F_MEM - F_ACT)
        v = -65 * S
        u = int(rshift_round(v * c02v, F_MEM))
        current = bias_q + int(rshift_round(a_mem * gain_q, F_MEM))
        acc = int(decoder_q[0]) << F_MEM
        for t in range(T):
            v2 = int(rshift_round(v * v, F_MEM))
            dv = int(rshift_round(v2 * c004, F_MEM)) + 5 * v + 140 * S - u + current
            v += int(rshift_round(dv, 1))
            v2 = int(rshift_round(v * v, F_MEM))
            dv = int(rshift_round(v2 * c004, F_MEM)) + 5 * v + 140 * S - u + current
            v += int(rshift_round(dv, 1))
            du = int(rshift_round((int(rshift_round(v * c02v, F_MEM)) - u) * c02, F_MEM))
            u += du
            if v >= 30 * S:
                v = -65 * S
                u += 8 * S
            acc += v * int(decoder_q[t + 1])
        raw = int(rshift_round(acc, F_MEM + F_DEC - out_frac))
        out[i] = max(raw, 1)
    return out.astype(np.int32)


def test_izh(lib):
    lib.h2_izh_decode_q20.argtypes = [
        ct.POINTER(ct.c_int32), ct.c_uint, ct.c_int32, ct.c_int32,
        ct.c_uint, ct.POINTER(ct.c_int32), ct.c_uint, ct.POINTER(ct.c_int32),
    ]
    rng = np.random.default_rng(66)
    cases = [
        (4.0, 0.5, 12, F_PQ, -30.0, 0.0),
        (2.0, 0.75, 20, F_PQ, -30.0, 0.0),
        (3.0, 0.25, 24, F_PK, -8.0, 6.0),
    ]
    maxdiff = 0
    total = 0
    for bias, gain, T, out_frac, lo, hi in cases:
        af = rng.uniform(lo, hi, size=257)
        a = np.rint(af * (1 << F_ACT)).astype(np.int32)
        decoder_q = rng.integers(-200_000, 200_000, size=T + 1, dtype=np.int32)
        bias_q = int(round(bias * (1 << F_MEM)))
        gain_q = int(round(gain * (1 << F_MEM)))
        ref = izh_ref(a, bias, gain, T, decoder_q, out_frac)
        got = np.empty_like(ref)
        lib.h2_izh_decode_q20(
            ptr(np.ascontiguousarray(a), ct.c_int32), a.size,
            bias_q, gain_q, T,
            ptr(np.ascontiguousarray(decoder_q), ct.c_int32), out_frac,
            ptr(got, ct.c_int32),
        )
        np.testing.assert_array_equal(got, ref)
        maxdiff = max(maxdiff, int(np.max(np.abs(got.astype(np.int64) - ref.astype(np.int64)))))
        total += a.size
    return total, maxdiff


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
        ptr(np.ascontiguousarray(pq), ct.c_int32), ptr(H_c, ct.c_int32), ptr(g_c, ct.c_int32),
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
        ptr(values, ct.c_int32), ptr(bias, ct.c_int32), out_f, ptr(got, ct.c_int32),
    )
    np.testing.assert_array_equal(got, ref)
    return int(values.size)


def main():
    lib = build()
    ndiv = test_scaled_div(lib)
    ln_diff = test_layernorm(lib)
    stable_diff = test_stable_a(lib)
    izh_cases, izh_diff = test_izh(lib)
    read_diff = test_outer_and_read(lib)
    nnz = test_sparse(lib)
    print({
        "H2_C_KERNEL_GATE": "PASS",
        "scaled_div_cases": ndiv,
        "layernorm_max_abs_diff": ln_diff,
        "stable_a_max_abs_diff": stable_diff,
        "izh_scalar_features_checked": izh_cases,
        "izh_max_abs_diff": izh_diff,
        "attention_read_max_abs_diff": read_diff,
        "sparse_test_nnz": nnz,
        "contract": "H1.1-functional-v9",
    })


if __name__ == "__main__":
    main()
