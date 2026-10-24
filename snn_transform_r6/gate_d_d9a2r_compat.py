#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility launcher for Gate D9A.2-R diagnostic provenance hashing.

D9A.1 froze `diagnostic_tokens` as int16 and included dtype in its provenance
hash. D9A.2-R initially reloaded the identical token values as int64 before
recomputing that hash, causing a schema-only mismatch before any trajectory or
geometry calculation. This launcher restores the frozen int16 hashing schema
only for that provenance check. All trajectories, probes, basis, checkpoints,
geometry, Gate-A evaluation, thresholds, and T1-T4 logic are unchanged.
"""
from __future__ import annotations

import numpy as np

import gate_d_d9a2r as impl

_orig_sha_arrays = impl.d91.sha_arrays


def _sha_arrays_d9a1_frozen_schema(*arrs):
    xs = list(arrs)
    if len(xs) == 4:
        # Exact D9A.1 saved-hash schema:
        # diagnostic_ids int64, labels int8, tokens int16, seeds int64.
        xs[0] = np.asarray(xs[0], dtype=np.int64)
        xs[1] = np.asarray(xs[1], dtype=np.int8)
        xs[2] = np.asarray(xs[2], dtype=np.int16)
        xs[3] = np.asarray(xs[3], dtype=np.int64)
    return _orig_sha_arrays(*xs)


def main():
    impl.d91.sha_arrays = _sha_arrays_d9a1_frozen_schema
    print('D9A2R_DIAGNOSTIC_HASH_SCHEMA_FIX active: int16 diagnostic-token provenance only; scientific protocol unchanged', flush=True)
    impl.main()


if __name__ == '__main__':
    main()
