#!/usr/bin/env python3
"""Host regression test for the exact arithmetic vector used by H2E-board.

The first physical-board attempt returned rc=5 because the APLX self-test had
an incorrect expected H-code (1020 instead of 1320).  The fixed-point kernel was
correct.  This gate mirrors the board vector through the compiled C kernels so
future APLX self-test constants cannot drift away from the frozen Q-format.
"""
from __future__ import annotations

import ctypes as ct
import numpy as np

from test_h2_kernels import build, ptr, F_ACT, F_PQ, F_PK, F_H


def main() -> None:
    lib = build()
    lib.h2_attention_token_delta.argtypes = [
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
        ct.c_uint, ct.c_uint, ct.c_uint,
    ]
    lib.h2_attention_read.argtypes = [
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
        ct.c_uint, ct.c_uint, ct.c_uint, ct.c_uint,
        ct.POINTER(ct.c_int32),
    ]
    lib.h2_div_scaled_round_i64.argtypes = [ct.c_int64, ct.c_int64, ct.c_uint]
    lib.h2_div_scaled_round_i64.restype = ct.c_int32
    lib.h2_sparse_linear_q20.argtypes = [
        ct.POINTER(ct.c_int32), ct.c_uint, ct.c_uint,
        ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_uint16),
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
        ct.c_uint, ct.POINTER(ct.c_int32),
    ]

    H = np.asarray([1000], dtype=np.int32)
    g = np.asarray([500], dtype=np.int32)
    old_pk = np.asarray([100], dtype=np.int32)
    old_v = np.asarray([1 << F_ACT], dtype=np.int32)
    new_pk = np.asarray([120], dtype=np.int32)
    new_v = np.asarray([1 << F_ACT], dtype=np.int32)

    lib.h2_attention_token_delta(
        ptr(H, ct.c_int32), ptr(g, ct.c_int32),
        ptr(old_pk, ct.c_int32), ptr(old_v, ct.c_int32),
        ptr(new_pk, ct.c_int32), ptr(new_v, ct.c_int32),
        1, 1, 1,
    )

    # Q16 pk -> Q20 H with v=1.0 implies a factor 2^(20-16)=16 in code space.
    assert int(g[0]) == 520
    assert int(H[0]) == 1320, int(H[0])

    pq = np.asarray([1 << F_PQ], dtype=np.int32)
    attended = np.zeros(1, dtype=np.int32)
    lib.h2_attention_read(
        ptr(pq, ct.c_int32), ptr(H, ct.c_int32), ptr(g, ct.c_int32),
        1, 1, 1, 1, ptr(attended, ct.c_int32),
    )
    expected = int(lib.h2_div_scaled_round_i64(
        int(H[0]), int(g[0]), F_ACT + F_PK - F_H))
    assert int(attended[0]) == expected

    row_ptr = np.asarray([0, 1], dtype=np.uint32)
    col_idx = np.asarray([0], dtype=np.uint16)
    values = np.asarray([1 << F_ACT], dtype=np.int32)
    bias = np.asarray([0], dtype=np.int32)
    x = np.asarray([123456], dtype=np.int32)
    y = np.zeros(1, dtype=np.int32)
    lib.h2_sparse_linear_q20(
        ptr(x, ct.c_int32), 1, 1,
        ptr(row_ptr, ct.c_uint32), ptr(col_idx, ct.c_uint16),
        ptr(values, ct.c_int32), ptr(bias, ct.c_int32),
        1, ptr(y, ct.c_int32),
    )
    assert int(y[0]) == int(x[0])

    print({
        "H2E_BOARD_CONTRACT_GATE": "PASS",
        "H_after_delta": int(H[0]),
        "g_after_delta": int(g[0]),
        "attention_read": int(attended[0]),
        "sparse_identity": int(y[0]),
    })


if __name__ == "__main__":
    main()
