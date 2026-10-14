#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import torch

import run_unified_gate as legacy
from core_v2 import LinearAttentionCoreV2


def rel(a,b):
    return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-12))


def make_sequence(seed,B,N,D,T):
    g=torch.Generator();g.manual_seed(seed)
    xs=[torch.randn(B,N,D,generator=g)]; ms=[torch.ones(B,N,dtype=torch.bool)]
    for _ in range(1,T):
        x=xs[-1].clone();m=torch.zeros(B,N,dtype=torch.bool)
        for b in range(B):
            idx=torch.randperm(N,generator=g)[:max(1,N//4)]
            x[b,idx]=torch.randn(len(idx),D,generator=g);m[b,idx]=True
        xs.append(x);ms.append(m)
    return xs,ms


def run_core(core,xs,ms,alphas):
    s=None; ys=[]; states=[]
    with torch.no_grad():
        for x,m,a in zip(xs,ms,alphas):
            y,s,_=core.step(x,s,a,m);ys.append(y);states.append(s)
    return ys,states


def batch_vs_single(core,xs,ms,alphas):
    yb,sb=run_core(core,xs,ms,alphas)
    B=xs[0].shape[0]; per=[]
    for b in range(B):
        xc=[x[b:b+1] for x in xs];mc=[m[b:b+1] for m in ms];ac=[a[b:b+1] for a in alphas]
        per.append(run_core(core,xc,mc,ac))
    rows=[]
    for t in range(len(xs)):
        ycat=torch.cat([z[0][t] for z in per],0)
        Hcat=torch.cat([z[1][t].H for z in per],0);gcat=torch.cat([z[1][t].g for z in per],0)
        rows.append({'step':t,'y_rel_l2':rel(yb[t],ycat),'H_rel_l2':rel(sb[t].H,Hcat),'g_rel_l2':rel(sb[t].g,gcat)})
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='unified_snn_transform/run/U21');ap.add_argument('--seed',type=int,default=41)
    ap.add_argument('--batch',type=int,default=4);ap.add_argument('--tokens',type=int,default=16);ap.add_argument('--d',type=int,default=32)
    ap.add_argument('--heads',type=int,default=2);ap.add_argument('--features',type=int,default=32);ap.add_argument('--steps',type=int,default=5)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.manual_seed(a.seed)
    v2=LinearAttentionCoreV2(a.d,a.heads,a.features,a.seed+101).eval()
    old=legacy.LinearAttentionCore(a.d,a.heads,a.features,a.seed+101).eval();old.load_state_dict(copy.deepcopy(v2.state_dict()))
    xs,ms=make_sequence(a.seed+1000,a.batch,a.tokens,a.d,a.steps)
    g=torch.Generator();g.manual_seed(a.seed+2000)
    alphas=[torch.rand(a.batch,a.heads,generator=g)*.08+.90 for _ in range(a.steps)]
    v2rows=batch_vs_single(v2,xs,ms,alphas); oldrows=batch_vs_single(old,xs,ms,alphas)
    vmax=max(max(r['y_rel_l2'],r['H_rel_l2'],r['g_rel_l2']) for r in v2rows)
    omax=max(max(r['y_rel_l2'],r['H_rel_l2'],r['g_rel_l2']) for r in oldrows)
    gates={'v2_batch_isolated_le_1e6':vmax<=1e-6,'legacy_cross_batch_leak_detected':omax>=1e-4}
    res={'stage':'UNIFIED_U2_1_BATCH_ISOLATION','v2_rows':v2rows,'legacy_rows':oldrows,
         'v2_max_rel_l2':vmax,'legacy_max_rel_l2':omax,'gates':gates,'PASS':all(gates.values()),
         'decision':'promote core_v2 and revalidate persistent-state results' if all(gates.values()) else 'do not promote; regression did not isolate the suspected bug'}
    (out/'U21_RESULTS.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2),flush=True)

if __name__=='__main__':main()
