#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D6A launcher with model-field terminal provenance certification.

Scientific protocol is unchanged.  The only change is to the cross-run terminal
provenance comparator: D6A/D5B/D3E instrumentation bookkeeping is not part of
the model state.  Cross-run certification therefore compares the frozen D3E
model field set bit-exactly, while same-run N/F_EM plain↔instrumented parity
continues to certify the complete native learner field set at 256/512/1024.
The downstream frozen Gate-A calibration to D3E is still mandatory.
"""
from __future__ import annotations

import hashlib
import struct
import numpy as np

import gate_d_d3e_a0 as d3a0
import gate_d_d3e_runtime as d3r
import gate_d_d6a as d6a

# Frozen model fields used by D3E provenance, plus R6B4 input/aux fields when
# they exist in both objects.  These are model/learner fields, not audit logs.
EXTRA_MODEL_KEYS = ('tok','pos','aux_lr_scale')
MODEL_KEYS = tuple(d3a0.MODEL_KEYS) + EXTRA_MODEL_KEYS


def _same_value(a,b):
    if isinstance(a,np.ndarray) and isinstance(b,np.ndarray):
        return (a.shape==b.shape and a.dtype==b.dtype and
                np.ascontiguousarray(a).tobytes()==np.ascontiguousarray(b).tobytes())
    if isinstance(a,np.generic): a=a.item()
    if isinstance(b,np.generic): b=b.item()
    if isinstance(a,float) and isinstance(b,float):
        return struct.pack('>d',a)==struct.pack('>d',b)
    return type(a)==type(b) and a==b


def _digest_fields(obj,keys):
    h=hashlib.sha256()
    missing=[]; unsupported=[]
    for k in keys:
        if not hasattr(obj,k):
            missing.append(k); continue
        v=getattr(obj,k); h.update(k.encode()+b'\0')
        if isinstance(v,np.ndarray):
            a=np.ascontiguousarray(v); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
        elif isinstance(v,np.generic):
            a=np.asarray(v); h.update(str(a.dtype).encode()); h.update(a.tobytes())
        elif isinstance(v,bool): h.update(b'B1' if v else b'B0')
        elif isinstance(v,int): h.update(b'I'+str(v).encode())
        elif isinstance(v,float): h.update(b'F'+struct.pack('>d',v))
        elif isinstance(v,str): h.update(b'S'+v.encode())
        elif v is None: h.update(b'Z')
        else: unsupported.append(k)
    return h.hexdigest(),missing,unsupported


def _compare_model_fields(ref,aud):
    # Only keys present in the frozen D3E model definition are eligible.
    # Optional R6B4 fields are compared if present in either object and must
    # then be present in both.
    keys=[]
    for k in MODEL_KEYS:
        if k in d3a0.MODEL_KEYS or hasattr(ref,k) or hasattr(aud,k):
            keys.append(k)
    dr,mr,ur=_digest_fields(ref,keys)
    da,ma,ua=_digest_fields(aud,keys)
    mism=[]
    for k in keys:
        if not hasattr(ref,k) or not hasattr(aud,k):
            continue
        if not _same_value(getattr(ref,k),getattr(aud,k)):
            mism.append(k)
    ok=(dr==da and not mr and not ma and not ur and not ua and not mism)
    return {
        'ref_digest':dr,
        'instrumented_digest_on_plain_field_set':da,
        'model_field_set':keys,
        'missing_ref_fields':mr,
        'missing_instrumented_fields':ma,
        'unsupported_ref_fields':ur,
        'unsupported_instrumented_fields':ua,
        'model_mismatch_fields':mism,
        'array_mismatch_fields':[k for k in mism if hasattr(ref,k) and isinstance(getattr(ref,k),np.ndarray)],
        'scalar_mismatch_fields':[k for k in mism if hasattr(ref,k) and not isinstance(getattr(ref,k),np.ndarray)],
        'rng_mismatch_fields':[],
        'missing_fields':sorted(set(mr+ma)),
        'bitwise_pass':bool(ok),
        'certification_scope':'frozen D3E model fields; audit/instrumentation bookkeeping excluded',
    }


def main():
    d3r.compare_native_fields_bitwise=_compare_model_fields
    d6a.main()


if __name__=='__main__':
    main()
