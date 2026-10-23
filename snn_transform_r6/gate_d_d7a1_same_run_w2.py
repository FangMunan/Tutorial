#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D7A.1 provenance compatibility launcher.

The first D7A.1 attempt compared the historical D6A F terminal W2 with the D7A
same-run clamped C terminal W2 using bit-exact equality. D7A itself certified
that its replayed F terminal matches the historical D6A F terminal within the
registered 1e-8 model-field tolerance and that C.W2 == F.W2 exactly in the
same run. This launcher therefore redirects only the D6A F-terminal loader used
by D7A.1 to the D7A same-run F terminal. B remains the historical D6A combined
terminal; all scientific mapping logic, Gate-A evaluation, bootstrap inference,
and C1-C4/CX classification remain unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import gate_d_d3e_runtime as d3r
import gate_d_d6a1 as d6a1
import gate_d_d7a1 as impl


def _arg_path(flag: str) -> Path:
    try:
        i = sys.argv.index(flag)
    except ValueError as exc:
        raise RuntimeError(f'missing required argument {flag}') from exc
    if i + 1 >= len(sys.argv):
        raise RuntimeError(f'missing value after {flag}')
    return Path(sys.argv[i + 1])


def _max_abs(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float('inf')
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def main():
    d6a_root = _arg_path('--d6a-artifact')
    d7a_root = _arg_path('--d7a-artifact')

    historical_f_path = d3r.locate(d6a_root, 'D6A_F_EM_terminal.pkl')
    same_run_f_path = d3r.locate(d7a_root, 'D7A_F_terminal.pkl')
    c_path = d3r.locate(d7a_root, 'D7A_C_W2_safe_clamp_terminal.pkl')
    d7a_summary_path = d3r.locate(d7a_root, 'D7A_SUMMARY.json')

    historical_f = d6a1.load_d6a_terminal(historical_f_path)
    same_run_f = impl.load_terminal(same_run_f_path)
    c = impl.load_terminal(c_path)
    d7a_summary = json.loads(d7a_summary_path.read_text(encoding='utf-8'))

    replay_pass = bool(d7a_summary.get('model_replay_to_D6A', {}).get('F', {}).get('pass', False))
    hist_err = _max_abs(historical_f.W2, same_run_f.W2)
    same_run_exact = bool(
        isinstance(same_run_f.W2, np.ndarray)
        and isinstance(c.W2, np.ndarray)
        and same_run_f.W2.dtype == c.W2.dtype
        and same_run_f.W2.shape == c.W2.shape
        and np.array_equal(same_run_f.W2, c.W2)
    )
    if not replay_pass or hist_err >= 1e-8 or not same_run_exact:
        raise RuntimeError({
            'D7A1_same_run_W2_reference_not_certified': {
                'D7A_F_replay_to_D6A_pass': replay_pass,
                'historical_D6A_F_vs_D7A_F_W2_max_abs': hist_err,
                'D7A_F_equals_C_W2_exact': same_run_exact,
            }
        })

    original_loader = d6a1.load_d6a_terminal

    def patched_loader(path):
        p = Path(path)
        if p.name == 'D6A_F_EM_terminal.pkl':
            return impl.load_terminal(same_run_f_path)
        return original_loader(p)

    d6a1.load_d6a_terminal = patched_loader
    impl.d6a1.load_d6a_terminal = patched_loader

    print('D7A1_SAME_RUN_W2_REFERENCE', json.dumps({
        'historical_D6A_F_vs_D7A_F_W2_max_abs': hist_err,
        'D7A_F_equals_C_W2_exact': same_run_exact,
        'D7A_F_replay_to_D6A_pass': replay_pass,
    }), flush=True)
    impl.main()


if __name__ == '__main__':
    main()
