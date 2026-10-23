#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility launcher for D9A.

The current branch's shared D8A forward_probe helper still contains the stale
scalar-theta cast from the first failed D8A attempt.  This launcher replaces
only that helper with the vector-threshold-equivalent frozen forward used by
R6B4, then calls the preregistered D9A main() unchanged.

No D9A scientific setting, arm, q-reference definition, regulator, oracle,
classification rule, or evaluation changes here.
"""
import math
import numpy as np

import gate_d_d8a as d8a
import gate_d_d9a as impl
import p3i_cb_exact_msdl as core


def forward_probe_vector_theta(system, cur, masked, do_tail_check=False):
    xn, xraw = system.context_from_arr(np.asarray(cur, dtype=np.int64))
    system.rebuild(xn)
    z_rows, u_rows, h_rows, y_rows, l_rows = [], [], [], [], []
    max_tail_error = 0.0
    for qi, t0 in enumerate(np.asarray(masked, dtype=np.int64)):
        t = int(t0)
        a, _q, _num, _den = system._read(t)
        z = (np.asarray(xraw[t], dtype=np.float64)
             + np.asarray(system.bo, dtype=np.float64)
             + np.asarray(system.WO, dtype=np.float64) @ np.asarray(a, dtype=np.float64))
        mu = float(np.mean(z))
        inv = 1.0 / math.sqrt(float(np.mean((z - mu) ** 2)) + float(system.eps2))
        zhat = (z - mu) * inv
        u = zhat * np.asarray(system.gamma2, dtype=np.float64) + np.asarray(system.beta2, dtype=np.float64)
        h = np.maximum(
            0.0,
            np.asarray(system.W1, dtype=np.float64) @ u
            + np.asarray(system.b1, dtype=np.float64)
            - np.asarray(system.theta, dtype=np.float64),
        )
        y = z + np.asarray(system.W2, dtype=np.float64) @ h + np.asarray(system.b2, dtype=np.float64)
        rms = math.sqrt(float(np.mean(y * y))) + 1e-8
        rf = np.empty(system.c.D + 1, dtype=np.float64)
        rf[:-1] = 0.5 * y / rms
        rf[-1] = 0.5
        logits = np.asarray(system.WR, dtype=np.float64) @ rf

        if do_tail_check and qi < 3:
            u0, h0, rf0, p0 = system._tail(
                np.asarray(a, dtype=np.float64), np.asarray(xraw[t], dtype=np.float64)
            )
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


d8a.forward_probe = forward_probe_vector_theta
impl.d8a.forward_probe = forward_probe_vector_theta


if __name__ == '__main__':
    impl.main()
