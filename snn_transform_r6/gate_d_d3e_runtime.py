#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D3E runtime/provenance helpers for P4-P6 and frozen boundary interventions."""
from __future__ import annotations

import hashlib
import json
import pickle
import struct
from pathlib import Path

import numpy as np

import gate_d_plasticity_audit as gd
import p3i_cb_exact_msdl as core
from gate_d_d15_d16 import RotationAuditSystem
from gate_d_d3e_eval import bootstrap_mean

P6_TOL = 1e-7


def jstate(rng):
    return json.dumps(gd.json_state(rng.bit_generator.state), sort_keys=True, separators=(',', ':'))


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def locate(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f'expected exactly one {name} under {root}, found {hits}')
    return hits[0]


def schedule_projection(traces):
    return [[{
        'masked_positions_before': [int(x) for x in row['masked_positions_before']],
        'remasked_positions': [int(x) for x in row['remasked_positions']],
    } for row in image] for image in traces]


def json_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


class D3EArmSystem(RotationAuditSystem):
    def d3e_init(self):
        self._d3e_prev_g = None
        self._d3e_cug1 = []
        self._d3e_rows = []

    def audit_finish_context(self):
        n = max(1, len(self._audit_ctx['masked']))
        g = np.asarray(self._audit_G, float) / n
        u = np.asarray(self.W2, float) - self._audit_w_before['W2']
        c = gd.cosine(u, self._d3e_prev_g) if self._d3e_prev_g is not None else np.nan
        self._d3e_rows.append({
            'context': int(self._audit_ctx['context']),
            'cosine': gd.cosine(u,g), 'C_ug_1': float(c),
            'u_norm': float(np.linalg.norm(u)), 'g_norm': float(np.linalg.norm(g)),
        })
        if np.isfinite(c):
            self._d3e_cug1.append(float(c))
        self._d3e_prev_g = np.array(g, copy=True)
        super().audit_finish_context()

    def d3e_summary(self):
        r = self.rotation_summary()
        c = np.asarray(self._d3e_cug1, float)
        r['context_alignment'] = {
            'mean': float(np.mean([x['cosine'] for x in self._d3e_rows])),
            'median': float(np.median([x['cosine'] for x in self._d3e_rows])),
        }
        r['C_ug_lag1'] = {
            'N': int(len(c)), 'mean': float(c.mean()), 'median': float(np.median(c)),
            'mean_95ci': bootstrap_mean(c, 720001, 5000)['ci95'],
        }
        return r


def make_arm(start):
    s = pickle.loads(pickle.dumps(start, pickle.HIGHEST_PROTOCOL))
    s.__class__ = D3EArmSystem
    s.audit_init(sketch_dim=64)
    s.rotation_init()
    s.d3e_init()
    return s


def reset_boundary(s, arm):
    # CAUSAL ARM DEFINITION — DO NOT ALTER.
    if arm in ('F_E','F_EM'):
        s.ve2.fill(core.V0); s.ue2.fill(core.U0); s.ze2.fill(0.)
    if arm in ('F_M','F_EM'):
        s.vm2.fill(core.V0); s.um2.fill(core.U0); s.zm2.fill(0.)


def target_order_signature_slice(rows):
    x = [(int(r['rank']), int(r['position']), int(r['target'])) for r in rows]
    return hashlib.sha256(json.dumps(x, separators=(',', ':')).encode()).hexdigest()


def stream_row(j, image_index, ev, target_sig):
    return {
        'context': int(j), 'image_index': int(image_index),
        'masked': [int(x) for x in ev['masked']],
        'E': [int(x) for x in ev['E']],
        'target_order_signature': str(target_sig),
        'mirror_seed': int(ev['mirror_seed']),
    }


def stream_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _hash_native_value(h, v, seen):
    if isinstance(v, np.ndarray):
        a = np.ascontiguousarray(v)
        h.update(b'A'); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes()); return True
    if isinstance(v, np.random.Generator):
        h.update(b'G'); h.update(jstate(v).encode()); return True
    if isinstance(v, np.generic):
        h.update(b'N'); h.update(str(v.dtype).encode()); h.update(np.asarray(v).tobytes()); return True
    if isinstance(v, bool):
        h.update(b'B1' if v else b'B0'); return True
    if isinstance(v, int):
        h.update(b'I'); h.update(str(v).encode()); return True
    if isinstance(v, float):
        h.update(b'F'); h.update(struct.pack('>d', v)); return True
    if isinstance(v, str):
        h.update(b'S'); h.update(v.encode()); return True
    if v is None:
        h.update(b'Z'); return True
    if isinstance(v, (list, tuple)):
        h.update(b'L' if isinstance(v,list) else b'T')
        for x in v:
            if not _hash_native_value(h, x, seen): return False
        return True
    if isinstance(v, dict):
        h.update(b'D')
        for k in sorted(v, key=lambda x: repr(x)):
            h.update(repr(k).encode())
            if not _hash_native_value(h, v[k], seen): return False
        return True
    if hasattr(v, '__dict__'):
        oid = id(v)
        if oid in seen:
            h.update(b'R'); return True
        seen.add(oid); h.update(b'O'); h.update(v.__class__.__name__.encode())
        for k in sorted(vars(v)):
            h.update(k.encode())
            if not _hash_native_value(h, getattr(v,k), seen): return False
        return True
    return False


def digest_native_fields(template, obj):
    h = hashlib.sha256(); unsupported = []
    for k in sorted(vars(template)):
        if not hasattr(obj, k):
            h.update(b'MISSING'); h.update(k.encode()); continue
        h.update(k.encode() + b'\0')
        if not _hash_native_value(h, getattr(obj,k), set()):
            unsupported.append(k)
    return h.hexdigest(), unsupported


def compare_native_fields_bitwise(ref, aud):
    ref_digest, uref = digest_native_fields(ref, ref)
    aud_digest, uaud = digest_native_fields(ref, aud)
    array_mismatch=[]; scalar_mismatch=[]; rng_mismatch=[]; missing=[]
    for k,v in vars(ref).items():
        if not hasattr(aud,k):
            missing.append(k); continue
        w=getattr(aud,k)
        if isinstance(v,np.ndarray) and isinstance(w,np.ndarray):
            same=(v.shape==w.shape and v.dtype==w.dtype and
                  np.ascontiguousarray(v).tobytes()==np.ascontiguousarray(w).tobytes())
            if not same: array_mismatch.append(k)
        elif isinstance(v,np.random.Generator) and isinstance(w,np.random.Generator):
            if jstate(v)!=jstate(w): rng_mismatch.append(k)
        elif isinstance(v,(int,float,bool,str,type(None),np.integer,np.floating,np.bool_)):
            hv=hashlib.sha256(); hw=hashlib.sha256()
            _hash_native_value(hv,v,set()); _hash_native_value(hw,w,set())
            if hv.digest()!=hw.digest(): scalar_mismatch.append(k)
    return {
        'ref_digest':ref_digest,'instrumented_digest_on_plain_field_set':aud_digest,
        'unsupported_ref_fields':sorted(set(uref)),'unsupported_instrumented_fields':sorted(set(uaud)),
        'array_mismatch_fields':array_mismatch,'scalar_mismatch_fields':scalar_mismatch,
        'rng_mismatch_fields':rng_mismatch,'missing_fields':missing,
        'bitwise_pass':bool(ref_digest==aud_digest and not array_mismatch and not scalar_mismatch
                            and not rng_mismatch and not missing and not uref and not uaud),
    }


def run_native_p4_pair(start, tokens, indices, rng_state):
    plain=pickle.loads(pickle.dumps(start,pickle.HIGHEST_PROTOCOL))
    inst=make_arm(start)
    rp=gd.clone_rng(rng_state); ri=gd.clone_rng(rng_state)
    rows=[]; checkpoints={}
    for j,image_index in enumerate(indices,start=1):
        truth=np.asarray(tokens[int(image_index)],int)
        rec_p,ev_p=gd.run_one_context(plain,False,arm='T0',j=j,image_index=int(image_index),truth=truth,rng=rp)
        e0=len(inst._audit_event_rows)
        rec_i,ev_i=gd.run_one_context(inst,True,arm='T0',j=j,image_index=int(image_index),truth=truth,rng=ri)
        evrows=inst._audit_event_rows[e0:]
        if ev_p['masked']!=ev_i['masked'] or ev_p['E']!=ev_i['E'] or ev_p['mirror_seed']!=ev_i['mirror_seed']:
            raise RuntimeError(f'P4 stream divergence at context {j}')
        if jstate(rp)!=jstate(ri): raise RuntimeError(f'P4 external RNG divergence at context {j}')
        if float(rec_p['mean_nll'])!=float(rec_i['mean_nll']): raise RuntimeError(f'P4 mean_nll divergence at context {j}')
        rows.append(stream_row(j,image_index,ev_i,target_order_signature_slice(evrows)))
        if j in (256,512,1024) or j==len(indices):
            cmp=compare_native_fields_bitwise(plain,inst)
            cmp['external_rng_exact']=bool(jstate(rp)==jstate(ri)); cmp['context']=int(j)
            cmp['pass']=bool(cmp['bitwise_pass'] and cmp['external_rng_exact'])
            checkpoints[str(j)]=cmp
            print('D3E_P4_CHECKPOINT',json.dumps({'context':j,'pass':cmp['pass'],
                  'ref_digest':cmp['ref_digest'],'instrumented_digest':cmp['instrumented_digest_on_plain_field_set']},indent=2),flush=True)
    p4=bool(all(v['pass'] for v in checkpoints.values()))
    return inst, {'stream_rows':rows,'stream_sha256':stream_hash(rows),'rng_state_after':jstate(ri),
                  'mechanism':inst.d3e_summary()}, {
        'pass':p4,'checkpoints':checkpoints,
        'requirement':'same-run R6B4 native plain/instrumented state exact on complete plain native field set'}


def train_intervention_arm(name,start,tokens,indices,rng_state):
    if name not in ('F_E','F_M','F_EM'): raise ValueError(name)
    s=make_arm(start); rng=gd.clone_rng(rng_state); rows=[]; online=[]; reset_rng_unchanged=True
    for j,image_index in enumerate(indices,start=1):
        before_reset=jstate(rng); reset_boundary(s,name)
        if jstate(rng)!=before_reset: reset_rng_unchanged=False
        truth=np.asarray(tokens[int(image_index)],int); e0=len(s._audit_event_rows)
        rec,ev=gd.run_one_context(s,True,arm='T0',j=j,image_index=int(image_index),truth=truth,rng=rng)
        evrows=s._audit_event_rows[e0:]
        rows.append(stream_row(j,image_index,ev,target_order_signature_slice(evrows)))
        online.append({'context':j,'image_index':int(image_index),'mean_nll':float(rec['mean_nll']),'nmask':int(ev['nmask'])})
        if j in (256,512,1024) or j==len(indices):
            print('D3E_ARM_PROGRESS',json.dumps({'arm':name,'context':j,'alignment':s._d3e_rows[-1]['cosine']}),flush=True)
    return s, {'stream_rows':rows,'stream_sha256':stream_hash(rows),'rng_state_after':jstate(rng),
               'reset_rng_unchanged':bool(reset_rng_unchanged),'online_rows':online,'mechanism':s.d3e_summary()}


def max_abs_array_diff(a,b):
    a=np.asarray(a); b=np.asarray(b)
    if a.shape!=b.shape: return float('inf')
    if np.issubdtype(a.dtype,np.floating) or np.issubdtype(b.dtype,np.floating):
        na=np.isnan(a); nb=np.isnan(b)
        if not np.array_equal(na,nb): return float('inf')
        mask=~na
        if not np.any(mask): return 0.0
        return float(np.max(np.abs(a[mask].astype(float)-b[mask].astype(float))))
    return 0.0 if np.array_equal(a,b) else float('inf')


def p6_compare(current,anchor_npz,anchor_report,schedules):
    details={}
    for key in ('self_metrics','teacher_metrics'):
        details[f'{key}_max_abs']=max_abs_array_diff(current[key],anchor_npz[key])
    for key in ('self_proposal','teacher_proposal'):
        details[f'{key}_exact']=bool(np.array_equal(current[key],anchor_npz[key]))
    for k in ('CE','D_non61','F61','C61','delta_P61'):
        details[f'effect_{k}_max_abs']=max_abs_array_diff(current['effects'][k],anchor_npz[f'effect__{k}'])
    agg_expected=anchor_report['r6b4_native_gateA_aggregate_means']
    details['aggregate_abs_diff']={k:float(abs(current['summary'][k]['mean']-agg_expected[k])) for k in agg_expected}
    details['schedule_sha256']=json_sha(schedule_projection(schedules))
    details['schedule_exact']=bool(details['schedule_sha256']==anchor_report['schedule_sha256'])
    numeric=[details['self_metrics_max_abs'],details['teacher_metrics_max_abs']]
    numeric += [details[f'effect_{k}_max_abs'] for k in ('CE','D_non61','F61','C61','delta_P61')]
    p=bool(max(numeric)<P6_TOL and max(details['aggregate_abs_diff'].values())<P6_TOL
           and details['self_proposal_exact'] and details['teacher_proposal_exact'] and details['schedule_exact'])
    return p,details
