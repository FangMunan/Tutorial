#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility launcher for Gate D8A vector-threshold forward probing.

The first D8A attempt incorrectly coerced system.theta to a Python float in the
standalone activation probe. In the frozen learner theta is a per-hidden-unit
array and the native forward path uses elementwise subtraction. This launcher
patches only D8A.forward_probe to reproduce that frozen vector-threshold path.
All discovery IDs, R/C/P/Q states, SVD analysis, thresholds, and I1-I4 logic are
unchanged.
"""
from __future__ import annotations

import math
import numpy as np

import gate_d_d8a as impl
import p3i_cb_exact_msdl as core


def forward_probe_vector_theta(system, cur, masked, do_tail_check=False):
    xn, xraw = system.context_from_arr(np.asarray(cur, dtype=np.int64))
    system.rebuild(xn)
    z_rows, u_rows, h_rows, y_rows, l_rows = [], [], [], [], []
    max_tail_error = 0.0
    theta = np.asarray(system.theta, dtype=np.float64)

    for qi, t0 in enumerate(np.asarray(masked, dtype=np.int64)):
        t = int(t0)
        a, _q, _num, _den = system._read(t)
        a = np.asarray(a, dtype=np.float64)
        xr = np.asarray(xraw[t], dtype=np.float64)

        z = xr + np.asarray(system.bo, dtype=np.float64) + np.asarray(system.WO, dtype=np.float64) @ a
        mu = float(np.mean(z))
        inv = 1.0 / math.sqrt(float(np.mean((z - mu) ** 2)) + float(system.eps2))
        zhat = (z - mu) * inv
        u = zhat * np.asarray(system.gamma2, dtype=np.float64) + np.asarray(system.beta2, dtype=np.float64)
        h = np.maximum(0.0, np.asarray(system.W1, dtype=np.float64) @ u + np.asarray(system.b1, dtype=np.float64) - theta)
        y = z + np.asarray(system.W2, dtype=np.float64) @ h + np.asarray(system.b2, dtype=np.float64)
        rms = math.sqrt(float(np.mean(y * y))) + 1e-8
        rf = np.empty(system.c.D + 1, dtype=np.float64)
        rf[:-1] = 0.5 * y / rms
        rf[-1] = 0.5
        logits = np.asarray(system.WR, dtype=np.float64) @ rf

        if do_tail_check and qi < 3:
            u0, h0, rf0, p0 = system._tail(a, xr)
            p = core.softmax_numba(np.asarray(logits, dtype=np.float64))
            errs = [
                float(np.max(np.abs(np.asarray(u0, dtype=np.float64) - u))),
                float(np.max(np.abs(np.asarray(h0, dtype=np.float64) - h))),
                float(np.max(np.abs(np.asarray(rf0, dtype=np.float64) - rf))),
                float(np.max(np.abs(np.asarray(p0, dtype=np.float64) - np.asarray(p, dtype=np.float64)))),
            ]
            max_tail_error = max(max_tail_error, max(errs))

        z_rows.append(z)
        u_rows.append(u)
        h_rows.append(h)
        y_rows.append(y)
        l_rows.append(logits)

    return {
        'z': np.asarray(z_rows, dtype=np.float64),
        'u': np.asarray(u_rows, dtype=np.float64),
        'h': np.asarray(h_rows, dtype=np.float64),
        'y': np.asarray(y_rows, dtype=np.float64),
        'logits': np.asarray(l_rows, dtype=np.float64),
    }, float(max_tail_error)


def main():
    impl.forward_probe = forward_probe_vector_theta
    print('D8A_VECTOR_THETA_FIX active: elementwise theta path; scientific protocol unchanged', flush=True)
    impl.main()


if __name__ == '__main__':
    main()
