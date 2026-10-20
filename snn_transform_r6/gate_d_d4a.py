#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D4A — eval-only terminal compensatory co-adaptation mapping.

Uses provenance-correct D3E N and F_EM terminal systems. No training occurs.
For each non-W2 forward block i, evaluates i, W2+i, and F-i and computes
paired per-image sufficiency, necessity, and W2-centered epistasis contrasts.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

# Import runtime so D3EArmSystem pickles resolve.
import gate_d_d3e_runtime  # noqa: F401
from gate_d_d3e_eval import gate_for_system
from r6b4_full_forward_plasticity import FORWARD_PARAMETER_KEYS

TOL = 1e-7
PRIMARY = ('F61', 'D_non61')
SUPPORTIVE = ('C61', 'delta_P61')
ALL_METRICS = ('F61', 'D_non61', 'C61', 'delta_P61', 'CE')
FOCAL = ('W2',)
BLOCKS = {
    'Q': ('WQ','bq'),
    'K': ('WK','bk'),
    'V': ('WV','bv'),
    'O': ('WO','bo'),
    'Norm': ('gamma2','beta2'),
    'FF1': ('W1','b1'),
    'Readout': ('WR',),
    'Embedding': ('tok','pos'),
    'b2': ('b2',),
}


def locate(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f'expected exactly one {name} under {root}, found {hits}')
    return hits[0]


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_terminal(path: Path):
    with path.open('rb') as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or 'system' not in obj:
        raise RuntimeError(f'unexpected terminal pickle schema: {path}')
    return obj['system']


def arr_equal(a, b) -> bool:
    a = np.asarray(a); b = np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def set_fields(dst, donor, fields: Iterable[str]):
    for k in fields:
        setattr(dst, k, np.array(getattr(donor, k), copy=True))


def hybrid(background, donor, donor_fields: Tuple[str,...]):
    s = pickle.loads(pickle.dumps(background, pickle.HIGHEST_PROTOCOL))
    set_fields(s, donor, donor_fields)
    return s


def purity(system, native, fem, selected_from_f, background: str):
    selected = set(selected_from_f)
    rows = {}
    ok = True
    for k in FORWARD_PARAMETER_KEYS:
        got = np.asarray(getattr(system, k))
        expect_f = (k in selected)
        donor = fem if expect_f else (native if background == 'N' else fem)
        # F-i is represented with background F and selected fields reverted to N.
        if background == 'F' and k in selected:
            donor = native
        equal = arr_equal(got, getattr(donor, k))
        rows[k] = {'expected_source': 'F' if donor is fem else 'N', 'exact': bool(equal)}
        ok = ok and equal
    return bool(ok), rows


def system_eval(name, s, schedules, gt, shared):
    out = gate_for_system(name, s, schedules, gt, shared)
    if not out['gates']['all_pass']:
        raise RuntimeError(f'Gate-A structural gate failed for {name}: {out["gates"]}')
    return out


def max_abs(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.shape != b.shape:
        return float('inf')
    return float(np.max(np.abs(a-b))) if a.size else 0.0


def calibrate_one(label, ev, stored, stored_prefix):
    d = {
        'self_metrics_max_abs': max_abs(ev['self_metrics'], stored[f'{stored_prefix}__self_metrics']),
        'teacher_metrics_max_abs': max_abs(ev['teacher_metrics'], stored[f'{stored_prefix}__teacher_metrics']),
        'self_proposal_exact': bool(np.array_equal(ev['self_proposal'], stored[f'{stored_prefix}__self_proposal'])),
        'teacher_proposal_exact': bool(np.array_equal(ev['teacher_proposal'], stored[f'{stored_prefix}__teacher_proposal'])),
        'effects': {},
    }
    for m in ALL_METRICS:
        d['effects'][m] = max_abs(ev['effects'][m], stored[f'{stored_prefix}__effect__{m}'])
    d['pass'] = bool(
        d['self_metrics_max_abs'] < TOL and d['teacher_metrics_max_abs'] < TOL
        and d['self_proposal_exact'] and d['teacher_proposal_exact']
        and max(d['effects'].values()) < TOL)
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
    }


def mean_effect(ev, m):
    x = np.asarray(ev['effects'][m], float)
    return float(np.nanmean(x))


def contrast_summary(vec, seed, load=None):
    d = bootstrap_mean(vec, seed)
    if load is not None and abs(load) > 1e-30:
        d['fraction_of_total_antagonistic_load_nonadditive'] = float(d['mean']/load)
    else:
        d['fraction_of_total_antagonistic_load_nonadditive'] = None
    d['ci_entirely_positive'] = bool(d['ci95'][0] is not None and d['ci95'][0] > 0)
    d['mean_positive'] = bool(d['mean'] is not None and d['mean'] > 0)
    return d


def write_csv(path: Path, rows):
    if not rows:
        path.write_text('', encoding='utf-8'); return
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def field_delta_report(native, fem):
    out={}
    for k in FORWARD_PARAMETER_KEYS:
        a=np.asarray(getattr(native,k),float); b=np.asarray(getattr(fem,k),float); d=b-a
        out[k]={
            'shape':list(a.shape), 'delta_norm':float(np.linalg.norm(d)),
            'max_abs_delta':float(np.max(np.abs(d))) if d.size else 0.0,
            'exact_equal':bool(np.array_equal(a,b)),
        }
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--d3e-artifact',type=Path,required=True)
    ap.add_argument('--anchor-artifact',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--bootstrap',type=int,default=10000)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)

    summary_path=locate(args.d3e_artifact,'D3E_SUMMARY.json')
    manifest_path=locate(args.d3e_artifact,'D3E_MANIFEST.json')
    effects_path=locate(args.d3e_artifact,'D3E_EFFECT_ARRAYS.npz')
    n_path=locate(args.d3e_artifact,'D3E_N_terminal.pkl')
    f_path=locate(args.d3e_artifact,'D3E_F_EM_terminal.pkl')
    trace_path=locate(args.anchor_artifact,'HISTORICAL_R6B3_R3_T8_REFERENCE_TRACE.json')

    d3e_summary=json.loads(summary_path.read_text(encoding='utf-8'))
    d3e_manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if not d3e_summary.get('all_provenance_gates_pass',False):
        raise RuntimeError('source D3E provenance gates did not all pass')
    if d3e_summary.get('classification')!='E2_TRAJECTORY_CLEANER_NO_PRIMARY_PHENOTYPE_RESCUE':
        raise RuntimeError(f'unexpected D3E source classification: {d3e_summary.get("classification")}')

    stored=np.load(effects_path,allow_pickle=False)
    schedules=json.loads(trace_path.read_text(encoding='utf-8'))
    native=load_terminal(n_path); fem=load_terminal(f_path)
    gt=np.asarray(stored['gt_tokens'],dtype=np.int64)
    shared=np.asarray(stored['shared_uniform'],dtype=np.float64)

    # Structural/source checks.
    if len(schedules)!=len(gt) or len(gt)!=256:
        raise RuntimeError('D4A requires exact 256-image Gate-A schedule')
    if set(FORWARD_PARAMETER_KEYS)!={'tok','pos','WQ','WK','WV','WO','W1','W2','WR','bq','bk','bv','bo','b1','b2','gamma2','beta2'}:
        raise RuntimeError('forward parameter key set changed; D4A block map must be re-audited')
    delta_report=field_delta_report(native,fem)
    for block,fields in BLOCKS.items():
        if all(delta_report[k]['exact_equal'] for k in fields):
            raise RuntimeError(f'D4A block {block} has no N-vs-F terminal change')
    if delta_report['W2']['exact_equal']:
        raise RuntimeError('focal W2 did not change between N and F_EM')

    # Build/evaluate cache.
    systems={
        'N':native,
        'F':fem,
        'W2':hybrid(native,fem,FOCAL),
        'b2':hybrid(native,fem,BLOCKS['b2']),
        'W2+b2':hybrid(native,fem,FOCAL+BLOCKS['b2']),
    }
    purity_report={}
    for label,sel,bg in [
        ('W2',FOCAL,'N'),('b2',BLOCKS['b2'],'N'),('W2+b2',FOCAL+BLOCKS['b2'],'N')]:
        purity_report[label]={'pass':purity(systems[label],native,fem,sel,bg)[0]}
        if not purity_report[label]['pass']: raise RuntimeError(f'transplant purity failed: {label}')

    evals={}
    for label,s in systems.items():
        evals[label]=system_eval(label,s,schedules,gt,shared)

    # Hard calibration to provenance-correct D3E stored arrays.
    calibration={
        'N':calibrate_one('N',evals['N'],stored,'N'),
        'F':calibrate_one('F',evals['F'],stored,'F_EM'),
        'W2':calibrate_one('W2',evals['W2'],stored,'med_W2_only'),
        'b2':calibrate_one('b2',evals['b2'],stored,'med_b2_only'),
        'W2+b2':calibrate_one('W2+b2',evals['W2+b2'],stored,'med_W2_b2'),
    }
    calibration_pass=bool(all(x['pass'] for x in calibration.values()))
    if not calibration_pass:
        raise RuntimeError({'D4A_D3E_calibration_failed':calibration})

    # Build all D4A terminal interventions.
    block_purity={}
    for block,fields in BLOCKS.items():
        li=f'i::{block}'; lw=f'W2+i::{block}'; lf=f'F-i::{block}'
        if li not in systems:
            systems[li]=hybrid(native,fem,fields)
        if lw not in systems:
            systems[lw]=hybrid(native,fem,FOCAL+fields)
        if lf not in systems:
            systems[lf]=hybrid(fem,native,fields)
        for label,sel,bg in [(li,fields,'N'),(lw,FOCAL+fields,'N'),(lf,fields,'F')]:
            ok,detail=purity(systems[label],native,fem,sel,bg)
            block_purity[label]={'pass':ok,'detail':detail}
            if not ok: raise RuntimeError(f'transplant purity failed: {label}')
        for label in (li,lw,lf):
            if label not in evals:
                evals[label]=system_eval(label,systems[label],schedules,gt,shared)

    # Baseline antagonistic loads: F minus W2, positive means the full terminal
    # configuration is worse than the beneficial W2-only hybrid.
    loads={m:float(mean_effect(evals['F'],m)-mean_effect(evals['W2'],m)) for m in ALL_METRICS}

    rows=[]; block_summary={}
    seed_base=810000
    for bi,(block,fields) in enumerate(BLOCKS.items()):
        li=f'i::{block}'; lw=f'W2+i::{block}'; lf=f'F-i::{block}'
        block_summary[block]={'fields':list(fields),'metrics':{}}
        for mi,m in enumerate(ALL_METRICS):
            eN=np.asarray(evals['N']['effects'][m],float)
            eF=np.asarray(evals['F']['effects'][m],float)
            eW=np.asarray(evals['W2']['effects'][m],float)
            ei=np.asarray(evals[li]['effects'][m],float)
            eWi=np.asarray(evals[lw]['effects'][m],float)
            eFmi=np.asarray(evals[lf]['effects'][m],float)
            S=eWi-eW
            Nec=eF-eFmi
            Epi=eWi-eW-ei+eN
            ssum=contrast_summary(S,seed_base+bi*1000+mi*10+1,loads[m])
            nsum=contrast_summary(Nec,seed_base+bi*1000+mi*10+2,loads[m])
            isum=contrast_summary(Epi,seed_base+bi*1000+mi*10+3,None)
            metric={
                'J_N':float(np.mean(eN)), 'J_F':float(np.mean(eF)),
                'J_W2':float(np.mean(eW)), 'J_i':float(np.mean(ei)),
                'J_W2_i':float(np.mean(eWi)), 'J_F_minus_i':float(np.mean(eFmi)),
                'total_antagonistic_load_F_minus_W2':loads[m],
                'sufficiency':ssum,'necessity':nsum,'W2_epistasis':isum,
                'bidirectional_mean_positive':bool(ssum['mean_positive'] and nsum['mean_positive']),
                'bidirectional_CI_positive':bool(ssum['ci_entirely_positive'] and nsum['ci_entirely_positive']),
                'specific_epistasis_CI_positive':bool(ssum['ci_entirely_positive'] and nsum['ci_entirely_positive'] and isum['ci_entirely_positive']),
            }
            block_summary[block]['metrics'][m]=metric
            rows.append({
                'block':block,'fields':'+'.join(fields),'metric':m,
                'J_N':metric['J_N'],'J_F':metric['J_F'],'J_W2':metric['J_W2'],
                'J_i':metric['J_i'],'J_W2_i':metric['J_W2_i'],'J_F_minus_i':metric['J_F_minus_i'],
                'antagonistic_load':loads[m],
                'S_mean':ssum['mean'],'S_ci_lo':ssum['ci95'][0],'S_ci_hi':ssum['ci95'][1],
                'S_load_fraction_nonadditive':ssum['fraction_of_total_antagonistic_load_nonadditive'],
                'N_mean':nsum['mean'],'N_ci_lo':nsum['ci95'][0],'N_ci_hi':nsum['ci95'][1],
                'N_load_fraction_nonadditive':nsum['fraction_of_total_antagonistic_load_nonadditive'],
                'I_mean':isum['mean'],'I_ci_lo':isum['ci95'][0],'I_ci_hi':isum['ci95'][1],
                'bidirectional_mean_positive':metric['bidirectional_mean_positive'],
                'bidirectional_CI_positive':metric['bidirectional_CI_positive'],
                'specific_epistasis_CI_positive':metric['specific_epistasis_CI_positive'],
            })

    # Rank blocks descriptively by normalized mean bidirectional evidence over the
    # two preregistered primary endpoints. Fractions are explicitly non-additive.
    ranking=[]
    for block in BLOCKS:
        vals=[]; strong=0; specific=0
        for m in PRIMARY:
            mm=block_summary[block]['metrics'][m]
            L=abs(loads[m])+1e-30
            vals.append(0.5*(mm['sufficiency']['mean']+mm['necessity']['mean'])/L)
            strong+=int(mm['bidirectional_CI_positive'])
            specific+=int(mm['specific_epistasis_CI_positive'])
        ranking.append({'block':block,'descriptive_bidirectional_score_nonadditive':float(np.mean(vals)),
                        'primary_endpoints_bidirectional_CI_positive':strong,
                        'primary_endpoints_specific_epistasis_CI_positive':specific})
    ranking.sort(key=lambda r:(r['primary_endpoints_bidirectional_CI_positive'],r['primary_endpoints_specific_epistasis_CI_positive'],r['descriptive_bidirectional_score_nonadditive']),reverse=True)

    # Save all per-image causal arrays for D4B/D4C reuse.
    out_npz={
        'validation_ids':stored['validation_ids'], 'gt_tokens':stored['gt_tokens'],
        'shared_uniform':stored['shared_uniform'], 'shared_gumbel':stored['shared_gumbel'],
    }
    for label,ev in evals.items():
        safe=label.replace('::','__').replace('+','_plus_').replace('-','_minus_')
        out_npz[f'{safe}__self_metrics']=ev['self_metrics']
        out_npz[f'{safe}__teacher_metrics']=ev['teacher_metrics']
        out_npz[f'{safe}__self_proposal']=ev['self_proposal']
        out_npz[f'{safe}__teacher_proposal']=ev['teacher_proposal']
        for m,v in ev['effects'].items(): out_npz[f'{safe}__effect__{m}']=v
    np.savez_compressed(args.out/'D4A_EFFECT_ARRAYS.npz',**out_npz)
    write_csv(args.out/'D4A_BLOCK_METRICS.csv',rows)

    result={
        'stage':'GATE_D4A_TERMINAL_COMPENSATORY_COADAPTATION_MAPPING',
        'source_D3E_run':39454173518,
        'source_D3E_artifact_sha256':sha_file(effects_path),
        'source_terminal_files':{'N':sha_file(n_path),'F_EM':sha_file(f_path)},
        'source_D3E_classification':d3e_summary['classification'],
        'source_D3E_all_provenance_gates_pass':True,
        'training_performed':False,'E_M_states_intervened':False,'learner_modified':False,
        'calibration_pass':calibration_pass,'calibration':calibration,
        'forward_parameter_delta_report':delta_report,
        'transplant_purity_all_pass':bool(all(x['pass'] for x in block_purity.values())),
        'primary_metrics':list(PRIMARY),'supportive_metrics':list(SUPPORTIVE),
        'total_antagonistic_load_F_minus_W2':loads,
        'blocks':block_summary,'ranking':ranking,
        'interpretation_boundary':(
            'Sufficiency/necessity/load fractions are non-additive causal diagnostics, not a linear decomposition. '
            'W2-only and other hybrids are off-manifold terminal interventions. D4A maps terminal antagonism; '
            'training-time compensatory mechanism requires further evidence. Ordinary masked CE/JS are not used '
            'for D4A block ranking and may be evaluated for top D4B candidates.'),
    }
    (args.out/'D4A_SUMMARY.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    manifest={
        'd3_closed':True,'d4a_eval_only':True,'blocks':{k:list(v) for k,v in BLOCKS.items()},
        'focal_W2_fields':list(FOCAL),'n_blocks':len(BLOCKS),'n_new_terminal_intervention_classes':27,
        'calibration_tolerance':TOL,'bootstrap_replicates':args.bootstrap,
        'source_d3e_manifest_all_provenance_gates_pass':d3e_manifest.get('all_provenance_gates_pass'),
    }
    (args.out/'D4A_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('D4A_COMPLETE',json.dumps({'calibration_pass':calibration_pass,'loads':loads,
          'ranking':ranking[:9]},indent=2),flush=True)


if __name__=='__main__':
    main()
