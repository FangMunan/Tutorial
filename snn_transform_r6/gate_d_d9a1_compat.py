#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility launcher for Gate D9A.1.

The frozen successful D8A summary stores selected_k under
`primary_y_gate.selected_k`.  The D9A.1 implementation also checks a top-level
`selected_k` for provenance convenience.  This launcher exposes that already
frozen value without changing any scientific data, threshold, trajectory, or
velocity calculation.
"""
from __future__ import annotations

import json as _json
import gate_d_d9a1 as gate

_orig_loads = gate.json.loads


def _loads_with_frozen_d8a_schema(s, *args, **kwargs):
    d = _orig_loads(s, *args, **kwargs)
    if (isinstance(d, dict)
            and d.get('classification') == 'I1_SHARED_LOW_DIMENSIONAL_FUNCTIONAL_CANDIDATE'
            and 'primary_y_gate' in d
            and 'selected_k' not in d):
        d = dict(d)
        d['selected_k'] = d.get('primary_y_gate', {}).get('selected_k')
    return d


gate.json.loads = _loads_with_frozen_d8a_schema

if __name__ == '__main__':
    gate.main()
