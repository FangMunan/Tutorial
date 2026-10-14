#!/usr/bin/env python3
"""H1 CPU integer/fixed-point emulator for the frozen SNNTransformBlock-v1.1 math.

Purpose
-------
This is NOT a PyTorch implementation.  It is a hardware-oriented numerical
reference whose runtime path uses integer arrays, integer MACs, integer sqrt,
integer division and compact sparse rows.  Floating point is used only by the
offline compiler/reference side to create a representative v1.1-shaped HIR and
to measure parity.

The runtime operator is unchanged:
LN1 -> event Q/K/V -> stable_a -> Izh+online decoder -> H/g -> sparse WO
-> residual -> LN2 -> sparse W1 -> ReLU -> sparse W2 -> residual.

H1 is intentionally a backend/numerical proof on a deterministic representative
parameter realization.  Export of the frozen trained v1.1 checkpoint into this
same HIR is a separate handoff gate before board deployment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Fixed-point contract chosen for H1
# -----------------------------------------------------------------------------
# Main token/vector state: signed Q*.14 in int32-compatible range.
# Feature map phi:          signed Q*.12.
# Izh membrane state:       signed Q*.20 (fits the observed -65..30 mV range).
# Decoder coefficients:     signed Q*.18.
# Products/accumulators use int64.  H is kept at natural Q(phi+act)=Q*.26.

F_ACT = 14
F_PHI = 12
F_MEM = 20
F_DEC = 18
S_ACT = 1 << F_ACT
S_PHI = 1 << F_PHI
INT32_MAX = (1 << 31) - 1


def qround(x, frac: int) -> np.ndarray:
    return np.rint(np.asarray(x, dtype=np.float64) * (1 << frac)).astype(np.int64)


def deq(x, frac: int) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) / float(1 << frac)


def rshift_round(x, shift: int) -> np.ndarray:
    """Symmetric rounded arithmetic shift for integer arrays/scalars."""
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


# -----------------------------------------------------------------------------
# HIR structures
# -----------------------------------------------------------------------------

@dataclass
class SparseRowsQ:
    row_ptr: np.ndarray
    col_idx: np.ndarray
    values: np.ndarray
    bias: np.ndarray
    out_features: int
    in_features: int

    @property
    def nnz(self) -> int:
        return int(self.values.size)


@dataclass
class HIR:
    d: int
    heads: int
    features: int
    ff: int
    ln1_g: np.ndarray
    ln1_b: np.ndarray
    ln2_g: np.ndarray
    ln2_b: np.ndarray
    Wq: np.ndarray
    bq: np.ndarray
    Wk: np.ndarray
    bk: np.ndarray
    Wv: np.ndarray
    bv: np.ndarray
    omega: np.ndarray
    q_specs: List[Tuple[float, float, int]]
    k_specs: List[Tuple[float, float, int]]
    q_decoders: List[np.ndarray]
    k_decoders: List[np.ndarray]
    Wo: SparseRowsQ
    W1: SparseRowsQ
    W2: SparseRowsQ


@dataclass
class QState:
    H: np.ndarray
    g: np.ndarray
    pq: np.ndarray
    pk: np.ndarray
    v: np.ndarray


@dataclass
class RuntimeStats:
    q_token_evals: int = 0
    k_token_evals: int = 0
    v_token_evals: int = 0


# -----------------------------------------------------------------------------
# Integer runtime kernels — this is the part intended to map to C/APLX.
# -----------------------------------------------------------------------------

def dense_linear_q(xq: np.ndarray, wq: np.ndarray, bq: np.ndarray) -> np.ndarray:
    acc = np.asarray(xq, dtype=np.int64) @ np.asarray(wq, dtype=np.int64).T
    return rshift_round(acc, F_ACT) + bq


def sparse_linear_q(xq: np.ndarray, rows: SparseRowsQ) -> np.ndarray:
    xq = np.asarray(xq, dtype=np.int64)
    flat = xq.reshape(-1, rows.in_features)
    out = np.empty((flat.shape[0], rows.out_features), dtype=np.int64)
    for o in range(rows.out_features):
        a, b = int(rows.row_ptr[o]), int(rows.row_ptr[o + 1])
        cols = rows.col_idx[a:b]
        vals = rows.values[a:b]
        if cols.size:
            acc = np.sum(flat[:, cols] * vals[None, :], axis=1, dtype=np.int64)
            out[:, o] = rshift_round(acc, F_ACT) + rows.bias[o]
        else:
            out[:, o] = rows.bias[o]
    return out.reshape(*xq.shape[:-1], rows.out_features)


def layernorm_q(xq: np.ndarray, gammaq: np.ndarray, betaq: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """LayerNorm with integer mean/variance and integer sqrt; output Q*.14."""
    xq = np.asarray(xq, dtype=np.int64)
    D = xq.shape[-1]
    out = np.empty_like(xq)
    eps_q2 = int(round(eps * (1 << (2 * F_ACT))))
    src = xq.reshape(-1, D)
    dst = out.reshape(-1, D)
    for i, row in enumerate(src):
        mean = int(round(int(row.sum()) / D))
        centered = row - mean
        var_q2 = int(np.sum(centered * centered, dtype=np.int64) // D) + eps_q2
        denom_q = math.isqrt(max(var_q2, 1))
        norm = np.array(
            [div_round(int(v) << F_ACT, denom_q) for v in centered],
            dtype=np.int64,
        )
        dst[i] = rshift_round(norm * gammaq, F_ACT) + betaq
    return out


def split_heads(x: np.ndarray, heads: int) -> np.ndarray:
    B, N, D = x.shape
    return x.reshape(B, N, heads, D // heads).transpose(0, 2, 1, 3)


def stable_a_q(xq: np.ndarray, omegaq: np.ndarray, is_query: bool) -> np.ndarray:
    """Integer version of the frozen stable_a formula; returns Q*.14."""
    head_dim = xq.shape[-1]
    scale_q = int(round((head_dim ** -0.25) * S_ACT))
    xx = rshift_round(np.asarray(xq, dtype=np.int64) * scale_q, F_ACT)
    B, H, N, _ = xx.shape
    M = omegaq.shape[1]
    out = np.empty((B, H, N, M), dtype=np.int64)
    for b in range(B):
        for h in range(H):
            dash = rshift_round(xx[b, h] @ omegaq[h].T, F_ACT)
            diag = rshift_round(
                np.sum(xx[b, h] * xx[b, h], axis=-1, keepdims=True, dtype=np.int64),
                F_ACT + 1,
            )
            a = dash - diag
            if is_query:
                a = a - np.max(a, axis=-1, keepdims=True)
                out[b, h] = np.clip(a, -30 * S_ACT, 0)
            else:
                out[b, h] = np.clip(a, -30 * S_ACT, 30 * S_ACT)
    return out


def izh_decode_q(aq: np.ndarray, spec: Tuple[float, float, int], decoder_q: np.ndarray) -> np.ndarray:
    """Fused fixed-point Izh RS substeps + online decoder; returns phi Q*.12."""
    bias, gain, T = spec
    T = int(T)
    if decoder_q.size != T + 1:
        raise ValueError("decoder length != window+1")

    aq = np.asarray(aq, dtype=np.int64)
    # Convert stable_a from Q14 to membrane Q20.
    a_mem = aq << (F_MEM - F_ACT)
    S_MEM = 1 << F_MEM
    v = np.full_like(aq, -65 * S_MEM)
    c02v = int(round(0.2 * S_MEM))
    u = rshift_round(v * c02v, F_MEM)
    bias_q = int(round(float(bias) * S_MEM))
    gain_q = int(round(float(gain) * S_MEM))
    current = bias_q + rshift_round(a_mem * gain_q, F_MEM)

    c004 = int(round(0.04 * S_MEM))
    c02 = int(round(0.02 * S_MEM))
    c140 = 140 * S_MEM

    # Natural scale of accumulator is Q(F_MEM+F_DEC).
    acc = np.full_like(aq, int(decoder_q[0]) << F_MEM, dtype=np.int64)
    for t in range(T):
        # Rescale v^2 before multiplying by 0.04 to avoid int64 overflow.
        v2 = rshift_round(v * v, F_MEM)
        term004 = rshift_round(v2 * c004, F_MEM)
        dv = term004 + 5 * v + c140 - u + current
        v = v + rshift_round(dv, 1)

        v2 = rshift_round(v * v, F_MEM)
        term004 = rshift_round(v2 * c004, F_MEM)
        dv = term004 + 5 * v + c140 - u + current
        v = v + rshift_round(dv, 1)

        du = rshift_round((rshift_round(v * c02v, F_MEM) - u) * c02, F_MEM)
        u = u + du
        fired = v >= 30 * S_MEM
        v = np.where(fired, -65 * S_MEM, v)
        u = np.where(fired, u + 8 * S_MEM, u)
        acc = acc + v * int(decoder_q[t + 1])

    raw = rshift_round(acc, F_MEM + F_DEC - F_PHI)
    return np.maximum(raw, 1)


def phi_q(xq: np.ndarray, omegaq: np.ndarray, specs, decoders, is_query: bool) -> np.ndarray:
    a = stable_a_q(xq, omegaq, is_query)
    out = np.empty_like(a)
    for h in range(a.shape[1]):
        out[:, h] = izh_decode_q(a[:, h], specs[h], decoders[h])
    return out


class FixedPointBlockRuntime:
    """Integer-only execution path corresponding to one v1.1 block."""

    def __init__(self, hir: HIR):
        self.hir = hir
        self.stats = RuntimeStats()

    def step(
        self,
        xq: np.ndarray,
        state: Optional[QState] = None,
        changed: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, QState]:
        p = self.hir
        xq = np.asarray(xq, dtype=np.int64)
        B, N, D = xq.shape
        xn = layernorm_q(xq, p.ln1_g, p.ln1_b)

        if state is None:
            q = split_heads(dense_linear_q(xn, p.Wq, p.bq), p.heads)
            k = split_heads(dense_linear_q(xn, p.Wk, p.bk), p.heads)
            v = split_heads(dense_linear_q(xn, p.Wv, p.bv), p.heads)
            pq = phi_q(q, p.omega, p.q_specs, p.q_decoders, True)
            pk = phi_q(k, p.omega, p.k_specs, p.k_decoders, False)
            H = np.einsum("bhnm,bhnd->bhmd", pk, v, dtype=np.int64)
            g = pk.sum(axis=2, dtype=np.int64)
            self.stats.q_token_evals += B * N
            self.stats.k_token_evals += B * N
            self.stats.v_token_evals += B * N
        else:
            if changed is None:
                raise ValueError("changed [B,N] required for persistent execution")
            changed = np.asarray(changed, dtype=bool)
            if changed.shape != (B, N):
                raise ValueError("changed shape mismatch")
            H, g = state.H.copy(), state.g.copy()
            pq, pk, v = state.pq.copy(), state.pk.copy(), state.v.copy()

            for b in range(B):
                idx = np.nonzero(changed[b])[0]
                if not idx.size:
                    continue
                xb = xn[b:b + 1, idx]
                nq = split_heads(dense_linear_q(xb, p.Wq, p.bq), p.heads)
                nk = split_heads(dense_linear_q(xb, p.Wk, p.bk), p.heads)
                nv = split_heads(dense_linear_q(xb, p.Wv, p.bv), p.heads)
                npq = phi_q(nq, p.omega, p.q_specs, p.q_decoders, True)
                npk = phi_q(nk, p.omega, p.k_specs, p.k_decoders, False)

                old_pk = state.pk[b:b + 1, :, idx, :]
                old_v = state.v[b:b + 1, :, idx, :]
                H[b:b + 1] -= np.einsum("bhnm,bhnd->bhmd", old_pk, old_v, dtype=np.int64)
                H[b:b + 1] += np.einsum("bhnm,bhnd->bhmd", npk, nv, dtype=np.int64)
                g[b:b + 1] -= old_pk.sum(axis=2, dtype=np.int64)
                g[b:b + 1] += npk.sum(axis=2, dtype=np.int64)
                pq[b][:, idx, :] = npq[0]
                pk[b][:, idx, :] = npk[0]
                v[b][:, idx, :] = nv[0]

                n = int(idx.size)
                self.stats.q_token_evals += n
                self.stats.k_token_evals += n
                self.stats.v_token_evals += n

        # H/g normalized read.  Natural fixed-point factors cancel in num/den,
        # so rounded integer division directly returns a value in Q*.14 (V scale).
        dh = D // p.heads
        attended = np.empty((B, p.heads, N, dh), dtype=np.int64)
        for b in range(B):
            for h in range(p.heads):
                for n in range(N):
                    den = sum(int(pq[b, h, n, m]) * int(g[b, h, m]) for m in range(p.features))
                    for d in range(dh):
                        num = sum(
                            int(pq[b, h, n, m]) * int(H[b, h, m, d])
                            for m in range(p.features)
                        )
                        attended[b, h, n, d] = div_round(num, den)

        attended = attended.transpose(0, 2, 1, 3).reshape(B, N, D)
        a = sparse_linear_q(attended, p.Wo)
        z = xq + a
        zn = layernorm_q(z, p.ln2_g, p.ln2_b)
        hidden = np.maximum(sparse_linear_q(zn, p.W1), 0)
        y = z + sparse_linear_q(hidden, p.W2)
        return y, QState(H=H, g=g, pq=pq, pk=pk, v=v)


# -----------------------------------------------------------------------------
# Offline float reference/compiler.  These functions are not board runtime.
# -----------------------------------------------------------------------------

@dataclass
class FloatParams:
    d: int
    heads: int
    features: int
    ff: int
    ln1_g: np.ndarray
    ln1_b: np.ndarray
    ln2_g: np.ndarray
    ln2_b: np.ndarray
    Wq: np.ndarray
    bq: np.ndarray
    Wk: np.ndarray
    bk: np.ndarray
    Wv: np.ndarray
    bv: np.ndarray
    omega: np.ndarray
    q_specs: List[Tuple[float, float, int]]
    k_specs: List[Tuple[float, float, int]]
    q_decoders: List[np.ndarray]
    k_decoders: List[np.ndarray]
    Wo: np.ndarray
    bo: np.ndarray
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray


@dataclass
class FloatState:
    H: np.ndarray
    g: np.ndarray
    pq: np.ndarray
    pk: np.ndarray
    v: np.ndarray


def _fit_decoder(spec, features: int, lo: float, hi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.uniform(lo, hi, size=5000)
    target = np.exp(np.clip(a, -30, 30)) / math.sqrt(features) + 1e-6
    bias, gain, T = spec
    v = np.full_like(a, -65.0)
    u = 0.2 * v
    I = bias + gain * a
    cols = [np.ones_like(a)]
    for _ in range(int(T)):
        dv = 0.04 * v * v + 5 * v + 140 - u + I
        v = v + 0.5 * dv
        dv = 0.04 * v * v + 5 * v + 140 - u + I
        v = v + 0.5 * dv
        u = u + 0.02 * (0.2 * v - u)
        fired = v >= 30
        v = np.where(fired, -65.0, v)
        u = np.where(fired, u + 8.0, u)
        cols.append(v.copy())
    X = np.stack(cols, axis=1)
    ridge = 1e-5
    A = X.T @ X + ridge * np.eye(X.shape[1])
    A[0, 0] -= ridge
    return np.linalg.solve(A, X.T @ target)


def _mask(out_features: int, in_features: int, density: float, rng) -> np.ndarray:
    k = max(1, round(density * in_features))
    mask = np.zeros((out_features, in_features), dtype=bool)
    for o in range(out_features):
        mask[o, rng.choice(in_features, size=k, replace=False)] = True
    return mask


def make_representative_params(seed: int = 71, d: int = 32, heads: int = 2, features: int = 32, ff: int = 64) -> FloatParams:
    rng = np.random.default_rng(seed)
    dh = d // heads

    def linear(out_f, in_f, sd=0.10):
        return rng.normal(0, sd, size=(out_f, in_f)), rng.normal(0, 0.02, size=out_f)

    ln1_g = 1 + rng.normal(0, 0.03, d)
    ln1_b = rng.normal(0, 0.02, d)
    ln2_g = 1 + rng.normal(0, 0.03, d)
    ln2_b = rng.normal(0, 0.02, d)
    Wq, bq = linear(d, d)
    Wk, bk = linear(d, d)
    Wv, bv = linear(d, d)

    omega = rng.normal(size=(heads, features, dh))
    omega = omega / np.linalg.norm(omega, axis=-1, keepdims=True) * math.sqrt(dh)

    # Frozen U4G-style operating points; decoder coefficients are fitted offline
    # for this representative numerical gate only.
    q_specs = [(5.0, 0.75, 24), (4.0, 0.75, 20)]
    k_specs = [(3.0, 0.25, 24), (2.0, 0.25, 20)]
    q_decoders = [_fit_decoder(q_specs[h], features, -7, 0, seed + 100 + h) for h in range(heads)]
    k_decoders = [_fit_decoder(k_specs[h], features, -7, 4, seed + 200 + h) for h in range(heads)]

    Wo, bo = linear(d, d, 0.12)
    W1, b1 = linear(ff, d, 0.12)
    W2, b2 = linear(d, ff, 0.12)
    Wo *= _mask(d, d, 0.25, rng)
    W1 *= _mask(ff, d, 0.25, rng)
    W2 *= _mask(d, ff, 0.25, rng)

    return FloatParams(
        d, heads, features, ff,
        ln1_g, ln1_b, ln2_g, ln2_b,
        Wq, bq, Wk, bk, Wv, bv, omega,
        q_specs, k_specs, q_decoders, k_decoders,
        Wo, bo, W1, b1, W2, b2,
    )


def _compile_sparse(W: np.ndarray, b: np.ndarray) -> SparseRowsQ:
    row_ptr = [0]
    cols, vals = [], []
    for o in range(W.shape[0]):
        idx = np.nonzero(W[o] != 0)[0]
        cols.extend(idx.tolist())
        vals.extend(qround(W[o, idx], F_ACT).tolist())
        row_ptr.append(len(cols))
    return SparseRowsQ(
        np.asarray(row_ptr, dtype=np.int32),
        np.asarray(cols, dtype=np.int32),
        np.asarray(vals, dtype=np.int64),
        qround(b, F_ACT),
        W.shape[0], W.shape[1],
    )


def compile_hir(p: FloatParams) -> HIR:
    return HIR(
        p.d, p.heads, p.features, p.ff,
        qround(p.ln1_g, F_ACT), qround(p.ln1_b, F_ACT),
        qround(p.ln2_g, F_ACT), qround(p.ln2_b, F_ACT),
        qround(p.Wq, F_ACT), qround(p.bq, F_ACT),
        qround(p.Wk, F_ACT), qround(p.bk, F_ACT),
        qround(p.Wv, F_ACT), qround(p.bv, F_ACT),
        qround(p.omega, F_ACT),
        list(p.q_specs), list(p.k_specs),
        [qround(x, F_DEC) for x in p.q_decoders],
        [qround(x, F_DEC) for x in p.k_decoders],
        _compile_sparse(p.Wo, p.bo),
        _compile_sparse(p.W1, p.b1),
        _compile_sparse(p.W2, p.b2),
    )


def _ln_float(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def _stable_a_float(x, omega, is_query):
    dh = x.shape[-1]
    xx = x * (dh ** -0.25)
    dash = np.einsum("bhnd,hmd->bhnm", xx, omega)
    diag = 0.5 * np.sum(xx * xx, axis=-1, keepdims=True)
    a = dash - diag
    if is_query:
        return np.clip(a - np.max(a, axis=-1, keepdims=True), -30, 0)
    return np.clip(a, -30, 30)


def _izh_float(a, spec, decoder):
    bias, gain, T = spec
    v = np.full_like(a, -65.0)
    u = 0.2 * v
    I = bias + gain * a
    out = np.full_like(a, float(decoder[0]))
    for t in range(int(T)):
        dv = 0.04 * v * v + 5 * v + 140 - u + I
        v = v + 0.5 * dv
        dv = 0.04 * v * v + 5 * v + 140 - u + I
        v = v + 0.5 * dv
        u = u + 0.02 * (0.2 * v - u)
        fired = v >= 30
        v = np.where(fired, -65.0, v)
        u = np.where(fired, u + 8.0, u)
        out += v * decoder[t + 1]
    return np.maximum(out, 1e-8)


def _phi_float(x, omega, specs, decoders, is_query):
    a = _stable_a_float(x, omega, is_query)
    out = np.empty_like(a)
    for h in range(a.shape[1]):
        out[:, h] = _izh_float(a[:, h], specs[h], decoders[h])
    return out


def _split_float(x, heads):
    B, N, D = x.shape
    return x.reshape(B, N, heads, D // heads).transpose(0, 2, 1, 3)


def float_step(p: FloatParams, x: np.ndarray, state: Optional[FloatState] = None, changed=None):
    B, N, D = x.shape
    xn = _ln_float(x, p.ln1_g, p.ln1_b)
    if state is None:
        q = _split_float(xn @ p.Wq.T + p.bq, p.heads)
        k = _split_float(xn @ p.Wk.T + p.bk, p.heads)
        v = _split_float(xn @ p.Wv.T + p.bv, p.heads)
        pq = _phi_float(q, p.omega, p.q_specs, p.q_decoders, True)
        pk = _phi_float(k, p.omega, p.k_specs, p.k_decoders, False)
        H = np.einsum("bhnm,bhnd->bhmd", pk, v)
        g = pk.sum(axis=2)
    else:
        H, g = state.H.copy(), state.g.copy()
        pq, pk, v = state.pq.copy(), state.pk.copy(), state.v.copy()
        for b in range(B):
            idx = np.nonzero(changed[b])[0]
            if not idx.size:
                continue
            xb = xn[b:b + 1, idx]
            q = _split_float(xb @ p.Wq.T + p.bq, p.heads)
            k = _split_float(xb @ p.Wk.T + p.bk, p.heads)
            nv = _split_float(xb @ p.Wv.T + p.bv, p.heads)
            npq = _phi_float(q, p.omega, p.q_specs, p.q_decoders, True)
            npk = _phi_float(k, p.omega, p.k_specs, p.k_decoders, False)
            opk = state.pk[b:b + 1, :, idx, :]
            ov = state.v[b:b + 1, :, idx, :]
            H[b:b + 1] -= np.einsum("bhnm,bhnd->bhmd", opk, ov)
            H[b:b + 1] += np.einsum("bhnm,bhnd->bhmd", npk, nv)
            g[b:b + 1] -= opk.sum(axis=2)
            g[b:b + 1] += npk.sum(axis=2)
            pq[b][:, idx, :] = npq[0]
            pk[b][:, idx, :] = npk[0]
            v[b][:, idx, :] = nv[0]

    num = np.einsum("bhnm,bhmd->bhnd", pq, H)
    den = np.einsum("bhnm,bhm->bhn", pq, g)[..., None]
    attended = num / np.maximum(den, 1e-8)
    attended = attended.transpose(0, 2, 1, 3).reshape(B, N, D)
    z = x + attended @ p.Wo.T + p.bo
    hidden = np.maximum(_ln_float(z, p.ln2_g, p.ln2_b) @ p.W1.T + p.b1, 0)
    y = z + hidden @ p.W2.T + p.b2
    return y, FloatState(H, g, pq, pk, v)


def integer_hir_only(hir: HIR) -> bool:
    arrays = [
        hir.ln1_g, hir.ln1_b, hir.ln2_g, hir.ln2_b,
        hir.Wq, hir.bq, hir.Wk, hir.bk, hir.Wv, hir.bv, hir.omega,
        *hir.q_decoders, *hir.k_decoders,
        hir.Wo.row_ptr, hir.Wo.col_idx, hir.Wo.values, hir.Wo.bias,
        hir.W1.row_ptr, hir.W1.col_idx, hir.W1.values, hir.W1.bias,
        hir.W2.row_ptr, hir.W2.col_idx, hir.W2.values, hir.W2.bias,
    ]
    return all(np.issubdtype(a.dtype, np.integer) for a in arrays)
