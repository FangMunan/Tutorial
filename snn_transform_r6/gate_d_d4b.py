#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D4B — O-block decomposition and dose-response attribution.

Eval-only terminal causal mapping. No training, no learner modification, no E/M
state intervention. Decomposes the D4A O=(WO,bo) antagonist, then interpolates
WO, bo, and the joint O block across native, W2-beneficial, and full-F backgrounds.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

# Resolve D3E arm-system pickles before loading terminal objects.
import gate_d_d3e_runtime  # noqa: F401
from gate_d_d3e_eval import gate_for_system
from r6b4_full_forward_plasticity import FORWARD_PARAMETER_KEYS

TOL = 1e-7
PRIMARY = ('F61', 'D_non61')
SUPPORTIVE = ('C61', 'delta_P61')
ALL_METRICS = ('F61', 'D_non61', 'C61', 'delta_P61', 'CE')
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
COMPONENTS = {
    'WO': ('WO',),
    'bo': ('bo',),
    'O_joint': ('WO', 'bo'),
}


def locate(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f'expected exactly one {name} under {root}, found {hits}')
    return hits[0]


def load_terminal(path: Path):
    with path.open('rb') as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or 'system' not in obj:
        raise RuntimeError(f'unexpected terminal pickle schema: {path}')
    return obj['system']


def clone(s):
    return pickle.loads(pickle.dumps(s, pickle.HIGHEST_PROTOCOL))


def arr_equal(a, b) -> bool:
    a = np.asarray(a); b = np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def set_fields(dst, donor, fields: Iterable[str]):
    for k in fields:
        setattr(dst, k, np.array(getattr(donor, k), copy=True))


def hybrid(background, donor, donor_fields: Tuple[str, ...]):
    s = clone(background)
    set_fields(s, donor, donor_fields)
    return s


def interpolated(background, native, fem, fields: Tuple[str, ...], lam: float, *, w2_from_f: bool = False):
    """Clone background and set selected fields to N + lam*(F-N).

    If w2_from_f is True, W2 is first set to F donor. For a full-F background,
    all non-selected fields remain F; selected fields move from N (lam=0) to F
    (lam=1).
    """
    s = clone(background)
    if w2_from_f:
        s.W2 = np.array(fem.W2, copy=True)
    for k in fields:
        a = np.asarray(getattr(native, k), dtype=np.float64)
        b = np.asarray(getattr(fem, k), dtype=np.float64)
        setattr(s, k, np.array(a + float(lam) * (b - a), copy=True))
    return s


def system_eval(name, s, schedules, gt, shared):
    out = gate_for_system(name, s, schedules, gt, shared)
    if not out['gates']['all_pass']:
        raise RuntimeError(f'Gate-A structural gate failed for {name}: {out["gates"]}')
    return out


def nanaware_max_abs(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float('inf')
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return float('inf')
    if not np.array_equal(np.isposinf(a), np.isposinf(b)):
        return float('inf')
    if not np.array_equal(np.isneginf(a), np.isneginf(b)):
        return float('inf')
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return 0.0
    return float(np.max(np.abs(a[finite] - b[finite])))


def calibrate_one(label, ev, stored, prefix):
    d = {
        'self_metrics_max_abs': nanaware_max_abs(ev['self_metrics'], stored[f'{prefix}__self_metrics']),
        'teacher_metrics_max_abs': nanaware_max_abs(ev['teacher_metrics'], stored[f'{prefix}__teacher_metrics']),
        'self_proposal_exact': bool(np.array_equal(ev['self_proposal'], stored[f'{prefix}__self_proposal'])),
        'teacher_proposal_exact': bool(np.array_equal(ev['teacher_proposal'], stored[f'{prefix}__teacher_proposal'])),
        'effects': {},
    }
    for m in ALL_METRICS:
        d['effects'][m] = nanaware_max_abs(ev['effects'][m], stored[f'{prefix}__effect__{m}'])
    d['pass'] = bool(
        d['self_metrics_max_abs'] < TOL
        and d['teacher_metrics_max_abs'] < TOL
        and d['self_proposal_exact']
        and d['teacher_proposal_exact']
        and max(d['effects'].values()) < TOL
    )
    d['label'] = label
    return d


def bootstrap_mean(vec, seed, nboot=10000, chunk=500):
    x = np.asarray(vec, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {'n':0,'mean':None,'ci95':[None,None],'fraction_positive':None,'fraction_negative':None}
    rng = np.random.default_rng(seed)
    vals = np.empty(nboot, dtype=np.float64)
    pos = 0
    while pos < nboot:
        n = min(chunk, nboot-pos)
        idx = rng.integers(0, len(x), size=(n, len(x)))
        vals[pos:pos+n] = x[idx].mean(axis=1)
        pos += n
    q = np.quantile(vals, [0.025, 0.975])
    return {
        'n': int(len(x)), 'mean': float(np.mean(x)),
        'ci95': [float(q[0]), float(q[1])],
        'fraction_positive': float(np.mean(x > 0)),
        'fraction_negative': float(np.mean(x < 0)),
        'ci_entirely_positive': bool(q[0] > 0),
        'ci_entirely_negative': bool(q[1] < 0),
    }


def bootstrap_ratio(delta_num, delta_den, seed, nboot=10000, chunk=500):
    """Paired bootstrap of ratio mean(delta_num)/mean(delta_den)."""
    a = np.asarray(delta_num, dtype=np.float64)
    b = np.asarray(delta_den, dtype=np.float64)
    good = np.isfinite(a) & np.isfinite(b)
    a = a[good]; b = b[good]
    if not len(a):
        return {'n':0,'ratio':None,'ci95':[None,None],'valid_bootstrap_fraction':0.0}
    den = float(np.mean(b))
    point = None if abs(den) < 1e-30 else float(np.mean(a) / den)
    rng = np.random.default_rng(seed)
    vals=[]
    pos=0
    while pos<nboot:
        n=min(chunk,nboot-pos)
        idx=rng.integers(0,len(a),size=(n,len(a)))
        ma=a[idx].mean(axis=1); mb=b[idx].mean(axis=1)
        ok=np.abs(mb)>1e-14
        if np.any(ok):
            vals.extend((ma[ok]/mb[ok]).tolist())
        pos+=n
    if not vals:
        return {'n':int(len(a)),'ratio':point,'ci95':[None,None],'valid_bootstrap_fraction':0.0}
    vals=np.asarray(vals,float)
    q=np.quantile(vals,[.025,.975])
    return {'n':int(len(a)),'ratio':point,'ci95':[float(q[0]),float(q[1])],
            'valid_bootstrap_fraction':float(len(vals)/nboot)}


def mean_effect(ev, m):
    return float(np.nanmean(np.asarray(ev['effects'][m], dtype=np.float64)))


def contrast(ev_a, ev_b, m):
    """Per-image J(a)-J(b). Higher positive means a is worse."""
    return np.asarray(ev_a['effects'][m], float) - np.asarray(ev_b['effects'][m], float)


def check_forward_purity(system, native, fem, source_map: Dict[str, str]):
    """All forward fields must match the declared N/F/interpolated source.

    Only exact N/F source fields are checked here. Interpolated fields are checked
    separately by construction in interpolation_purity().
    """
    rows={}; ok=True
    for k in FORWARD_PARAMETER_KEYS:
        if k not in source_map:
            continue
        donor = native if source_map[k]=='N' else fem
        same = arr_equal(getattr(system,k), getattr(donor,k))
        rows[k]={'source':source_map[k],'exact':bool(same)}
        ok = ok and same
    return bool(ok), rows


def interpolation_purity(system, native, fem, fields, lam, background, w2_from_f=False):
    selected=set(fields); rows={}; ok=True
    for k in FORWARD_PARAMETER_KEYS:
        got=np.asarray(getattr(system,k))
        if k in selected:
            a=np.asarray(getattr(native,k),float); b=np.asarray(getattr(fem,k),float)
            exp=a+float(lam)*(b-a)
            same=(got.shape==exp.shape and np.array_equal(got,exp))
            source=f'interp({lam:g})'
        elif w2_from_f and k=='W2':
            same=arr_equal(got,getattr(fem,k)); source='F'
        else:
            donor=native if background=='N' else fem
            same=arr_equal(got,getattr(donor,k)); source=background
        rows[k]={'source':source,'exact':bool(same)}
        ok=ok and same
    return bool(ok),rows


def write_csv(path: Path, rows):
    if not rows:
        path.write_text('',encoding='utf-8'); return
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--d3e-artifact',type=Path,required=True)
    ap.add_argument('--d4a-artifact',type=Path,required=True)
    ap.add_argument('--anchor-artifact',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--bootstrap',type=int,default=10000)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)

    # Authoritative sources.
    n_path=locate(args.d3e_artifact,'D3E_N_terminal.pkl')
    f_path=locate(args.d3e_artifact,'D3E_F_EM_terminal.pkl')
    d3e_summary=json.loads(locate(args.d3e_artifact,'D3E_SUMMARY.json').read_text(encoding='utf-8'))
    d4a_summary=json.loads(locate(args.d4a_artifact,'D4A_SUMMARY.json').read_text(encoding='utf-8'))
    d4a_arrays=np.load(locate(args.d4a_artifact,'D4A_EFFECT_ARRAYS.npz'),allow_pickle=False)
    schedules=json.loads(locate(args.anchor_artifact,'HISTORICAL_R6B3_R3_T8_REFERENCE_TRACE.json').read_text(encoding='utf-8'))

    if not d3e_summary.get('all_provenance_gates_pass',False):
        raise RuntimeError('source D3E provenance gates did not all pass')
    if not d4a_summary.get('calibration_pass',False):
        raise RuntimeError('source D4A calibration did not pass')
    if d4a_summary.get('ranking',[{}])[0].get('block')!='O':
        raise RuntimeError('D4B requires D4A O block as top-ranked antagonist')
    if d4a_summary['blocks']['O']['metrics']['F61']['bidirectional_CI_positive'] is not True:
        raise RuntimeError('D4B requires O bidirectional F61 evidence')
    if d4a_summary['blocks']['O']['metrics']['D_non61']['bidirectional_CI_positive'] is not True:
        raise RuntimeError('D4B requires O bidirectional D_non61 evidence')

    native=load_terminal(n_path); fem=load_terminal(f_path)
    gt=np.asarray(d4a_arrays['gt_tokens'],dtype=np.int64)
    shared=np.asarray(d4a_arrays['shared_uniform'],dtype=np.float64)
    if len(gt)!=256 or len(schedules)!=256:
        raise RuntimeError('D4B requires exact 256-image Gate-A coordinate system')

    # Direct decomposition systems.
    systems={
        'N':native,
        'F':fem,
        'W2':hybrid(native,fem,('W2',)),
        'WO':hybrid(native,fem,('WO',)),
        'bo':hybrid(native,fem,('bo',)),
        'O':hybrid(native,fem,('WO','bo')),
        'W2+WO':hybrid(native,fem,('W2','WO')),
        'W2+bo':hybrid(native,fem,('W2','bo')),
        'W2+O':hybrid(native,fem,('W2','WO','bo')),
        'F-WO':hybrid(fem,native,('WO',)),
        'F-bo':hybrid(fem,native,('bo',)),
        'F-O':hybrid(fem,native,('WO','bo')),
    }

    # Exact forward-field purity for direct systems.
    direct_specs={
        'W2':{'W2':'F'},
        'WO':{'WO':'F'},
        'bo':{'bo':'F'},
        'O':{'WO':'F','bo':'F'},
        'W2+WO':{'W2':'F','WO':'F'},
        'W2+bo':{'W2':'F','bo':'F'},
        'W2+O':{'W2':'F','WO':'F','bo':'F'},
    }
    direct_purity={}
    for label,spec in direct_specs.items():
        full={k:'N' for k in FORWARD_PARAMETER_KEYS}; full.update(spec)
        ok,detail=check_forward_purity(systems[label],native,fem,full)
        direct_purity[label]={'pass':ok,'detail':detail}
        if not ok: raise RuntimeError(f'direct transplant purity failed: {label}')
    for label,rev in [('F-WO',('WO',)),('F-bo',('bo',)),('F-O',('WO','bo'))]:
        full={k:'F' for k in FORWARD_PARAMETER_KEYS}
        for k in rev: full[k]='N'
        ok,detail=check_forward_purity(systems[label],native,fem,full)
        direct_purity[label]={'pass':ok,'detail':detail}
        if not ok: raise RuntimeError(f'reverse transplant purity failed: {label}')

    evals={label:system_eval(label,s,schedules,gt,shared) for label,s in systems.items()}

    # Hard calibration to successful D4A for the six shared systems.
    calibration={
        'N':calibrate_one('N',evals['N'],d4a_arrays,'N'),
        'F':calibrate_one('F',evals['F'],d4a_arrays,'F'),
        'W2':calibrate_one('W2',evals['W2'],d4a_arrays,'W2'),
        'O':calibrate_one('O',evals['O'],d4a_arrays,'i__O'),
        'W2+O':calibrate_one('W2+O',evals['W2+O'],d4a_arrays,'W2_plus_i__O'),
        'F-O':calibrate_one('F-O',evals['F-O'],d4a_arrays,'F_minus_i__O'),
    }
    calibration_pass=bool(all(x['pass'] for x in calibration.values()))
    if not calibration_pass:
        raise RuntimeError({'D4B_D4A_calibration_failed':calibration})

    # Direct causal decomposition.
    direct_summary={}; direct_rows=[]; seed0=910000
    for mi,m in enumerate(ALL_METRICS):
        S_WO=contrast(evals['W2+WO'],evals['W2'],m)
        S_bo=contrast(evals['W2+bo'],evals['W2'],m)
        N_WO=contrast(evals['F'],evals['F-WO'],m)
        N_bo=contrast(evals['F'],evals['F-bo'],m)
        I_internal=(np.asarray(evals['O']['effects'][m],float)
                    -np.asarray(evals['WO']['effects'][m],float)
                    -np.asarray(evals['bo']['effects'][m],float)
                    +np.asarray(evals['N']['effects'][m],float))
        I_W2_WO=(np.asarray(evals['W2+WO']['effects'][m],float)
                  -np.asarray(evals['W2']['effects'][m],float)
                  -np.asarray(evals['WO']['effects'][m],float)
                  +np.asarray(evals['N']['effects'][m],float))
        I_W2_bo=(np.asarray(evals['W2+bo']['effects'][m],float)
                 -np.asarray(evals['W2']['effects'][m],float)
                 -np.asarray(evals['bo']['effects'][m],float)
                 +np.asarray(evals['N']['effects'][m],float))
        ds={
            'J_N':mean_effect(evals['N'],m),'J_F':mean_effect(evals['F'],m),'J_W2':mean_effect(evals['W2'],m),
            'J_WO':mean_effect(evals['WO'],m),'J_bo':mean_effect(evals['bo'],m),'J_O':mean_effect(evals['O'],m),
            'J_W2_WO':mean_effect(evals['W2+WO'],m),'J_W2_bo':mean_effect(evals['W2+bo'],m),'J_W2_O':mean_effect(evals['W2+O'],m),
            'J_F_minus_WO':mean_effect(evals['F-WO'],m),'J_F_minus_bo':mean_effect(evals['F-bo'],m),'J_F_minus_O':mean_effect(evals['F-O'],m),
            'S_WO':bootstrap_mean(S_WO,seed0+mi*100+1,args.bootstrap),
            'S_bo':bootstrap_mean(S_bo,seed0+mi*100+2,args.bootstrap),
            'N_WO':bootstrap_mean(N_WO,seed0+mi*100+3,args.bootstrap),
            'N_bo':bootstrap_mean(N_bo,seed0+mi*100+4,args.bootstrap),
            'I_WO_bo':bootstrap_mean(I_internal,seed0+mi*100+5,args.bootstrap),
            'I_W2_WO':bootstrap_mean(I_W2_WO,seed0+mi*100+6,args.bootstrap),
            'I_W2_bo':bootstrap_mean(I_W2_bo,seed0+mi*100+7,args.bootstrap),
        }
        direct_summary[m]=ds
        for name in ('S_WO','S_bo','N_WO','N_bo','I_WO_bo','I_W2_WO','I_W2_bo'):
            x=ds[name]
            direct_rows.append({'metric':m,'contrast':name,'mean':x['mean'],'ci_lo':x['ci95'][0],'ci_hi':x['ci95'][1],
                                'ci_entirely_positive':x['ci_entirely_positive'],'ci_entirely_negative':x['ci_entirely_negative']})

    # Dose-response systems; reuse exact endpoint evals where possible.
    endpoint_alias={
        ('WO','N',0.0):'N',('WO','N',1.0):'WO',('WO','W2',0.0):'W2',('WO','W2',1.0):'W2+WO',('WO','F',0.0):'F-WO',('WO','F',1.0):'F',
        ('bo','N',0.0):'N',('bo','N',1.0):'bo',('bo','W2',0.0):'W2',('bo','W2',1.0):'W2+bo',('bo','F',0.0):'F-bo',('bo','F',1.0):'F',
        ('O_joint','N',0.0):'N',('O_joint','N',1.0):'O',('O_joint','W2',0.0):'W2',('O_joint','W2',1.0):'W2+O',('O_joint','F',0.0):'F-O',('O_joint','F',1.0):'F',
    }
    interp_purity={}
    curve_evals={}
    for comp,fields in COMPONENTS.items():
        for bg in ('N','W2','F'):
            for lam in LAMBDAS:
                key=(comp,bg,lam)
                if key in endpoint_alias:
                    curve_evals[key]=evals[endpoint_alias[key]]
                    continue
                if bg=='N':
                    s=interpolated(native,native,fem,fields,lam,w2_from_f=False); background='N'; w2f=False
                elif bg=='W2':
                    s=interpolated(native,native,fem,fields,lam,w2_from_f=True); background='N'; w2f=True
                else:
                    s=interpolated(fem,native,fem,fields,lam,w2_from_f=False); background='F'; w2f=False
                ok,detail=interpolation_purity(s,native,fem,fields,lam,background,w2f)
                label=f'{comp}::{bg}::lambda={lam:.2f}'
                interp_purity[label]={'pass':ok,'detail':detail}
                if not ok: raise RuntimeError(f'interpolation purity failed: {label}')
                curve_evals[key]=system_eval(label,s,schedules,gt,shared)

    # Curve statistics.
    curve_summary={}; curve_rows=[]; interaction_curves={}
    seedc=930000
    for ci,(comp,fields) in enumerate(COMPONENTS.items()):
        curve_summary[comp]={}
        for bi,bg in enumerate(('N','W2','F')):
            curve_summary[comp][bg]={}
            for mi,m in enumerate(ALL_METRICS):
                e0=np.asarray(curve_evals[(comp,bg,0.0)]['effects'][m],float)
                e1=np.asarray(curve_evals[(comp,bg,1.0)]['effects'][m],float)
                means=[]; lambdas_out={}; adjacent=[]
                prev=e0
                for li,lam in enumerate(LAMBDAS):
                    ex=np.asarray(curve_evals[(comp,bg,lam)]['effects'][m],float)
                    delta=ex-e0
                    stat=bootstrap_mean(delta,seedc+ci*10000+bi*2000+mi*200+li*10+1,args.bootstrap)
                    ratio=bootstrap_ratio(delta,e1-e0,seedc+ci*10000+bi*2000+mi*200+li*10+2,args.bootstrap)
                    means.append(float(np.nanmean(ex)))
                    lambdas_out[f'{lam:.2f}']={'J_mean':float(np.nanmean(ex)),'delta_from_lambda0':stat,'normalized_R':ratio}
                    if li>0:
                        ad=bootstrap_mean(ex-prev,seedc+ci*10000+bi*2000+mi*200+li*10+3,args.bootstrap)
                        adjacent.append({'from':float(LAMBDAS[li-1]),'to':float(lam),'contrast':ad})
                    prev=ex
                monotonic=bool(all(means[j+1]>=means[j]-1e-15 for j in range(len(means)-1)))
                endpoint=bootstrap_mean(e1-e0,seedc+ci*10000+bi*2000+mi*200+91,args.bootstrap)
                curve_summary[comp][bg][m]={
                    'lambda':lambdas_out,'adjacent_steps':adjacent,
                    'mean_sequence':means,'monotonic_non_decreasing_means':monotonic,
                    'endpoint_harmful_change':endpoint,
                }
                for lam in LAMBDAS:
                    x=lambdas_out[f'{lam:.2f}']
                    curve_rows.append({'component':comp,'background':bg,'metric':m,'lambda':lam,
                        'J_mean':x['J_mean'],'delta_mean':x['delta_from_lambda0']['mean'],
                        'delta_ci_lo':x['delta_from_lambda0']['ci95'][0],'delta_ci_hi':x['delta_from_lambda0']['ci95'][1],
                        'R':x['normalized_R']['ratio'],'R_ci_lo':x['normalized_R']['ci95'][0],'R_ci_hi':x['normalized_R']['ci95'][1]})

        # W2-vs-N interaction curve at every dose.
        interaction_curves[comp]={}
        for mi,m in enumerate(ALL_METRICS):
            eN0=np.asarray(curve_evals[(comp,'N',0.0)]['effects'][m],float)
            eW0=np.asarray(curve_evals[(comp,'W2',0.0)]['effects'][m],float)
            interaction_curves[comp][m]={}
            for li,lam in enumerate(LAMBDAS):
                eN=np.asarray(curve_evals[(comp,'N',lam)]['effects'][m],float)
                eW=np.asarray(curve_evals[(comp,'W2',lam)]['effects'][m],float)
                dint=(eW-eW0)-(eN-eN0)
                interaction_curves[comp][m][f'{lam:.2f}']=bootstrap_mean(
                    dint,950000+ci*5000+mi*100+li,args.bootstrap)

    # Descriptive preregistered case candidate; interpretation still uses raw curves.
    def both_primary_positive(prefix):
        return all(direct_summary[m][prefix]['ci_entirely_positive'] for m in PRIMARY)
    wo_bidir=both_primary_positive('S_WO') and both_primary_positive('N_WO')
    bo_bidir=both_primary_positive('S_bo') and both_primary_positive('N_bo')
    joint_interaction=all(direct_summary[m]['I_WO_bo']['ci_entirely_positive'] for m in PRIMARY)
    wo_graded=all(curve_summary['WO'][bg][m]['monotonic_non_decreasing_means']
                  for bg in ('N','W2','F') for m in PRIMARY)
    bo_graded=all(curve_summary['bo'][bg][m]['monotonic_non_decreasing_means']
                  for bg in ('N','W2','F') for m in PRIMARY)
    joint_graded=all(curve_summary['O_joint'][bg][m]['monotonic_non_decreasing_means']
                     for bg in ('N','W2','F') for m in PRIMARY)
    if wo_bidir and not bo_bidir and wo_graded:
        case='B1_WO_DOMINANT'
    elif bo_bidir and not wo_bidir and bo_graded:
        case='B2_bo_DOMINANT'
    elif (not wo_bidir) and (not bo_bidir) and joint_interaction and joint_graded:
        case='B3_JOINT_O_BLOCK'
    else:
        case='MIXED_OR_NONMONOTONIC_REQUIRES_INTERPRETATION'

    # Save all per-image arrays for exact future reuse.
    out_npz={
        'validation_ids':d4a_arrays['validation_ids'],'gt_tokens':d4a_arrays['gt_tokens'],
        'shared_uniform':d4a_arrays['shared_uniform'],'shared_gumbel':d4a_arrays['shared_gumbel'],
    }
    for label,ev in evals.items():
        safe=label.replace('+','_plus_').replace('-','_minus_')
        for m,v in ev['effects'].items(): out_npz[f'direct__{safe}__effect__{m}']=v
    for (comp,bg,lam),ev in curve_evals.items():
        safe=f'{comp}__{bg}__lambda_{lam:.2f}'.replace('.','p')
        for m,v in ev['effects'].items(): out_npz[f'curve__{safe}__effect__{m}']=v
    np.savez_compressed(args.out/'D4B_EFFECT_ARRAYS.npz',**out_npz)
    write_csv(args.out/'D4B_DIRECT_CONTRASTS.csv',direct_rows)
    write_csv(args.out/'D4B_DOSE_RESPONSE.csv',curve_rows)

    result={
        'stage':'GATE_D4B_O_BLOCK_DECOMPOSITION_AND_DOSE_RESPONSE',
        'source_D3E_run':39454173518,'source_D4A_run':39483623999,
        'training_performed':False,'learner_modified':False,'E_M_states_intervened':False,
        'primary_metrics':list(PRIMARY),'supportive_metrics':list(SUPPORTIVE),
        'calibration_pass':calibration_pass,'calibration':calibration,
        'direct_transplant_purity_all_pass':bool(all(x['pass'] for x in direct_purity.values())),
        'interpolation_purity_all_pass':bool(all(x['pass'] for x in interp_purity.values())),
        'direct_decomposition':direct_summary,
        'dose_response':curve_summary,
        'W2_vs_native_dose_interaction':interaction_curves,
        'case_candidate':case,
        'case_diagnostics':{
            'WO_bidirectional_both_primary':wo_bidir,'bo_bidirectional_both_primary':bo_bidir,
            'WO_all_three_backgrounds_monotonic_both_primary':wo_graded,
            'bo_all_three_backgrounds_monotonic_both_primary':bo_graded,
            'O_internal_interaction_positive_both_primary':joint_interaction,
            'O_joint_all_three_backgrounds_monotonic_both_primary':joint_graded,
        },
        'interpretation_boundary':(
            'D4B attributes terminal forward-parameter causality only. Endpoint and dose-response hybrids may be '
            'off the training manifold. Smooth interpolation and reverse interpolation reduce but do not eliminate '
            'that objection. A near-zero W2 interaction supports approximate transferability, not proof of complete '
            'mechanistic independence during training.'),
    }
    (args.out/'D4B_SUMMARY.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    manifest={
        'd4b_eval_only':True,'components':{k:list(v) for k,v in COMPONENTS.items()},
        'lambdas':list(LAMBDAS),'backgrounds':['N','W2','F'],'bootstrap_replicates':args.bootstrap,
        'calibration_tolerance':TOL,'source_d4a_top_block':d4a_summary['ranking'][0]['block'],
        'n_direct_systems':len(systems),'n_curve_coordinates':len(curve_evals),
    }
    (args.out/'D4B_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('D4B_COMPLETE',json.dumps({
        'calibration_pass':calibration_pass,'case_candidate':case,
        'primary_direct':{m:{k:direct_summary[m][k] for k in ('S_WO','S_bo','N_WO','N_bo','I_WO_bo','I_W2_WO','I_W2_bo')} for m in PRIMARY},
        'case_diagnostics':result['case_diagnostics']},indent=2),flush=True)


if __name__=='__main__':
    main()
