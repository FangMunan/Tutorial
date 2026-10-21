#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility launcher for Gate D6A terminal provenance certification.

The D6A scientific protocol is unchanged. This launcher only ensures that the
terminal bitwise comparison is evaluated on the native learner field set, not
on D6A/D5B instrumentation bookkeeping fields that are absent from the frozen
D3E terminal pickle.
"""
from __future__ import annotations

import pickle

import gate_d_d3e_runtime as d3r
import gate_d_d6a as d6a

_ORIG_COMPARE = d3r.compare_native_fields_bitwise


def _native_field_template(obj):
    q = pickle.loads(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL))
    for k in list(vars(q)):
        if k.startswith('_d5b_') or k.startswith('_d6a_'):
            delattr(q, k)
    return q


def _compare_native_only(ref, aud):
    # Plain control refs have no D5B/D6A bookkeeping and are unchanged.
    # Instrumented D6A terminal refs are stripped only of diagnostic fields.
    return _ORIG_COMPARE(_native_field_template(ref), aud)


def main():
    d3r.compare_native_fields_bitwise = _compare_native_only
    d6a.main()


if __name__ == '__main__':
    main()
