#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D4A calibration-only fix.

The original D4A calibration used np.max(abs(a-b)) on metric tensors that
legitimately contain paired NaNs when a round/image has no GT-61 samples.
That turns an otherwise exact comparison into NaN and falsely fails the hard
calibration gate.  This wrapper changes only the calibration comparator:
NaN/non-finite locations must match exactly, and max absolute error is computed
only over finite paired entries.  All D4A interventions/statistics are unchanged.
"""
from __future__ import annotations

import numpy as np
import gate_d_d4a as d4


def max_abs_nan_aware(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float('inf')

    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        return float('inf')

    # Any infinities must also occur at the same locations and with same signs.
    inf_a = np.isinf(a)
    inf_b = np.isinf(b)
    if not np.array_equal(inf_a, inf_b):
        return float('inf')
    if np.any(inf_a) and not np.array_equal(a[inf_a], b[inf_b]):
        return float('inf')

    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return 0.0
    return float(np.max(np.abs(a[finite] - b[finite])))


d4.max_abs = max_abs_nan_aware


if __name__ == '__main__':
    d4.main()
