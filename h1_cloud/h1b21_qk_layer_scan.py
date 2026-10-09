#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H1-B2.1 — Q/K × layer-specific hardware-aware Izh operating-point scan.

This stage keeps the H1-B architecture fixed: one Izh neuron trajectory per scalar
feature, linear trajectory decoder, positivity clamp, persistent H/g state, learned
selective retention, and event-sparse K/V execution.

Only calibration is changed. Each attention block gets independent Q and K mappings:

    I_{l,s} = bias_{l,s} + gain_{l,s} * a_{l,s},   s in {Q,K}

with independently fitted linear decoders and window lengths. Candidate selection is
coarse-to-fine: scalar fidelity/clamp/activity first, then end-to-end one-group network
probe. The final combined 8-group mapping is evaluated fresh and through persistent
selective MaskGIT trajectories.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

import h0_base as h0
import h1_sparse_fusion as h1
import h1b_izh_fusion as h1b
import h1b2_error_localization as h1b2


class DualCalibratedPhi(nn.Module):
    """One exact random projection, separate Izh calibrations for Q and K."""
    def __init__(self, exact_phi: h0.PositiveRandomFeatures,
                 q_spec: Dict, q_dec: torch.Tensor,
                 k_spec: Dict, k_dec: torch.Tensor):
        super().__init__()
        self.qmap = h1b.IzhDynamicFeatures(copy.deepcopy(exact_phi),
                                           bias=q_spec['bias'], gain=q_spec['gain'],
                                           window_ms=q_spec['window_ms'], decoder=q_dec)
        self.kmap = h1b.IzhDynamicFeatures(copy.deepcopy(exact_phi),
                                           bias=k_spec['bias'], gain=k_spec['gain'],
                                           window_ms=k_spec['window_ms'], decoder=k_dec)
        self.features = int(exact_phi.features)
        self.head_dim = int(exact_phi.head_dim)
        self.register_buffer('omega', exact_phi.omega.detach().clone())

    def forward(self, x: torch.Tensor, *, is_query: bool) -> torch.Tensor:
        return self.qmap(x, is_query=True) if is_query else self.kmap(x, is_query=False)


class OneSideLayerPhi(nn.Module):
    """Used only for candidate network probing of one (layer, side)."""
    def __init__(self, exact_phi, spec, dec, side: str):
        super().__init__()
        self.exact = copy.deepcopy(exact_phi)
        self.izh = h1b.IzhDynamicFeatures(copy.deepcopy(exact_phi),
                                          bias=spec['bias'], gain=spec['gain'],
                                          window_ms=spec['window_ms'], decoder=dec)
        self.side = side
        self.features = int(exact_phi.features)
        self.head_dim = int(exact_phi.head_dim)
        self.register_buffer('omega', exact_phi.omega.detach().clone())

    def forward(self, x, *, is_query: bool):
        use = (self.side == 'Q' and is_query) or (self.side == 'K' and not is_query)
        return self.izh(x, is_query=is_query) if use else self.exact(x, is_query=is_query)


def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def load_train_val(args):
    if args.data is not None:
        xall, yall, _, _ = h0.load_mnist_npz(args.data)
    else:
        xall, yall, _, _ = h0.load_mnist_keras()
    x012, y012 = h0.filter_classes(xall, yall)
    x012, y012 = h0.subset_per_class(x012, y012, args.n_per_class, args.seed)
    xtr, ytr, xva, yva = h0.stratified_split(x012, y012, .1, args.seed)
    ytr = h0.map_labels_012(ytr); yva = h0.map_labels_012(yva)
    cent = np.load(args.reuse / 'codebook.npy')
    tr = h0.encode_tokens(xtr, cent); va = h0.encode_tokens(xva, cent)
    return tr, ytr, va, yva


@torch.no_grad()
def collect_group_a(model, tokens, labels, device, *, seed: int,
                    batches: int, batch_size: int, per_group: int):
    """Collect empirical preactivations keyed by (layer, Q/K) from training data."""
    rng = np.random.default_rng(seed)
    bags = {(li, side): [] for li in range(model.layers) for side in ('Q','K')}
    for bi in range(batches):
        idx = rng.choice(len(tokens), size=min(batch_size, len(tokens)), replace=False)
        tok = torch.from_numpy(tokens[idx]).to(device)
        lab = torch.from_numpy(labels[idx]).to(device)
        g = torch.Generator(device=device); g.manual_seed(seed + 1009*bi)
        ratio = torch.rand(len(tok), generator=g, device=device) * .94 + .03
        mask = torch.rand(tok.shape, generator=g, device=device) < ratio[:, None]
        cur = tok.clone(); cur[mask] = h0.MASK_ID
        z = model.embed(cur, lab)
        for li, block in enumerate(model.blocks):
            for side, isq in (('Q',True),('K',False)):
                a = h1b.exact_stable(block.attn, z, isq).reshape(-1)
                n = min(per_group, a.numel())
                gg = torch.Generator(device=device)
                gg.manual_seed(seed + bi*10007 + li*101 + (1 if isq else 2))
                take = torch.randint(0, a.numel(), (n,), generator=gg, device=device)
                bags[(li, side)].append(a[take].detach().cpu())
            z = block(z)
    return {k: torch.cat(v).float() for k,v in bags.items()}


@torch.no_grad()
def izh_activity(a: torch.Tensor, bias: float, gain: float, window_ms: int):
    """Mean spikes/feature under the same RS dynamics; used only as hardware cost proxy."""
    x = a.float(); v = torch.full_like(x, -65.0); u = .2*v
    I = float(bias) + float(gain)*x
    spikes = torch.zeros_like(x)
    for _ in range(int(window_ms)):
        dv=.04*v*v+5*v+140-u+I; v=v+.5*dv
        dv=.04*v*v+5*v+140-u+I; v=v+.5*dv
        u=u+.02*(.2*v-u)
        fired=v>=30.0; spikes += fired.float()
        v=torch.where(fired, torch.full_like(v,-65.0), v)
        u=torch.where(fired, u+8.0, u)
    return float(spikes.mean()), float((spikes>0).float().mean())


def coarse_grid():
    biases = [2.0, 4.0, 6.0, 8.0, 10.0]
    gains = [0.20, 0.35, 0.50, 0.75, 1.00, 1.40]
    windows = [6, 8, 10, 12]
    return [{'bias':b,'gain':g,'window_ms':t} for b in biases for g in gains for t in windows]


def candidate_scalar_scan(a_all: torch.Tensor, features: int, device, topk: int=10):
    """Fit decoder for each operating point; return diverse shortlist and full rows."""
    gen = torch.Generator(); gen.manual_seed(1234567 + int(a_all.numel()))
    perm = torch.randperm(a_all.numel(), generator=gen)
    nfit = int(.72*len(perm)); nval = min(5000, len(perm)-nfit)
    ai = a_all[perm[:nfit]].to(device)
    av = a_all[perm[nfit:nfit+nval]].to(device)
    if ai.numel() > 9000: ai = ai[:9000]
    yi = h1b.target_phi(ai, features); yv = h1b.target_phi(av, features)

    rows=[]; payload=[]
    for spec in coarse_grid():
        dec = h1b.fit_decoder(ai, yi, **spec, ridge=1e-5)
        met = h1b.scalar_metrics(av, yv, dec, **spec)
        spk, active = izh_activity(av[:min(2500,av.numel())], **spec)
        # Shortlisting score: fidelity dominates; clamp, duration and activity break ties.
        score = met['scalar_rel_l2'] + .20*met['clamp_fraction'] + .0025*spec['window_ms'] + .01*spk
        row={**spec, **met, 'mean_spikes_per_feature':spk,
             'spiking_feature_fraction':active, 'hardware_scalar_score':float(score)}
        rows.append(row); payload.append((score,spec,dec,row))

    payload.sort(key=lambda x:x[0])
    chosen = payload[:topk]
    # Preserve at least one low-clamp and one shortest-window candidate when distinct.
    low_clamp = min(payload, key=lambda x:x[3]['clamp_fraction'])
    short = min(payload, key=lambda x:(x[1]['window_ms'], x[0]))
    for x in (low_clamp, short):
        if all((x[1]['bias'],x[1]['gain'],x[1]['window_ms']) !=
               (y[1]['bias'],y[1]['gain'],y[1]['window_ms']) for y in chosen):
            chosen.append(x)
    return chosen, rows


def install_one_group(model, layer: int, side: str, spec: Dict, dec: torch.Tensor):
    old = model.blocks[layer].attn.phi
    model.blocks[layer].attn.phi = OneSideLayerPhi(old, spec, dec, side).to(next(model.parameters()).device)
    return model


def independent_network_select(base, va, yva, device, layer, side, shortlist, seed):
    rows=[]; best=None
    for ci, (_,spec,dec,srow) in enumerate(shortlist):
        cand=copy.deepcopy(base); install_one_group(cand,layer,side,spec,dec)
        met=h1b.fresh_logit_error(base,cand,va,yva,device,seed+ci,n=24)
        # Network fidelity is primary. Tiny latency/activity regularizer avoids gratuitously slow ties.
        objective = met['rel_l2'] + .0006*spec['window_ms'] + .002*srow['mean_spikes_per_feature']
        row={**spec,'layer':layer,'side':side,
             'scalar_rel_l2':srow['scalar_rel_l2'],'clamp_fraction':srow['clamp_fraction'],
             'mean_spikes_per_feature':srow['mean_spikes_per_feature'],
             'network_rel_l2':met['rel_l2'],'network_argmax_agreement':met['argmax_agreement'],
             'network_max_abs':met['max_abs'],'selection_objective':float(objective)}
        rows.append(row)
        if best is None or objective < best[0]: best=(objective,spec,dec,row)
    return best, rows


def install_all_calibrated(model, selected):
    for li, block in enumerate(model.blocks):
        old=block.attn.phi
        qs=selected[(li,'Q')]; ks=selected[(li,'K')]
        block.attn.phi=DualCalibratedPhi(old,qs['spec'],qs['decoder'],ks['spec'],ks['decoder']).to(next(model.parameters()).device)
    return model


def compact_selected(selected):
    out={}
    for (li,side),v in selected.items():
        out[f'L{li}_{side}']={'spec':v['spec'],'probe':v['row'],
                              'decoder':v['decoder'].detach().cpu().tolist()}
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reuse',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path('H1B21_SCAN'))
    ap.add_argument('--data',type=Path,default=None)
    ap.add_argument('--n-per-class',type=int,default=2000)
    ap.add_argument('--d',type=int,default=128); ap.add_argument('--layers',type=int,default=4)
    ap.add_argument('--heads',type=int,default=4); ap.add_argument('--ff',type=int,default=512)
    ap.add_argument('--features',type=int,default=128); ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args()

    h0.seed_all(args.seed); torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr,ytr,va,yva=load_train_val(args)
    base,dense_sel,old_spec,old_dec=h1b2.load_reused_models(args,device)

    # Frozen shared-map baseline on fixed evaluation seeds.
    old_f=copy.deepcopy(base); h1b.replace_all_phi(old_f,old_spec,old_dec)
    baseline_fresh=h1b.fresh_logit_error(base,old_f,va,yva,device,args.seed+31001,n=32)
    exact_sparse=copy.deepcopy(dense_sel); h1.enable_sparse_block1(exact_sparse)
    old_p=copy.deepcopy(dense_sel); h1.enable_sparse_block1(old_p); h1b.replace_all_phi(old_p,old_spec,old_dec)
    baseline_persistent=h1b.persistent_logit_error(exact_sparse,old_p,va,yva,device,args.seed+31002,steps=6,n=16,mode='selective')

    groups=collect_group_a(base,tr,ytr,device,seed=args.seed+30001,batches=8,batch_size=24,per_group=1800)
    all_scalar=[]; all_probes=[]; selected={}

    # Optimize dominant Q groups first, then K, while architecture remains unchanged.
    order=[(li,'Q') for li in range(args.layers)] + [(li,'K') for li in range(args.layers)]
    for gi,(li,side) in enumerate(order):
        shortlist,scalar_rows=candidate_scalar_scan(groups[(li,side)],args.features,device,topk=10)
        for r in scalar_rows: all_scalar.append({'layer':li,'side':side,**r})
        best,probe_rows=independent_network_select(base,va,yva,device,li,side,shortlist,args.seed+32000+gi*100)
        all_probes.extend(probe_rows)
        _,spec,dec,row=best
        selected[(li,side)]={'spec':spec,'decoder':dec,'row':row}
        print('B21_SELECTED',json.dumps({'layer':li,'side':side,'spec':spec,'probe':row}),flush=True)

    write_csv(args.out/'scalar_scan.csv',all_scalar)
    write_csv(args.out/'network_probe_shortlist.csv',all_probes)

    # Combined calibrated model.
    cal_f=copy.deepcopy(base); install_all_calibrated(cal_f,selected)
    calibrated_fresh=h1b.fresh_logit_error(base,cal_f,va,yva,device,args.seed+31001,n=32)
    cal_p=copy.deepcopy(dense_sel); h1.enable_sparse_block1(cal_p); install_all_calibrated(cal_p,selected)
    calibrated_persistent=h1b.persistent_logit_error(exact_sparse,cal_p,va,yva,device,args.seed+31002,steps=6,n=16,mode='selective')

    # Measure aggregate clamp after final selection on the empirical calibration pools.
    aggregate=[]
    for (li,side),v in selected.items():
        a=groups[(li,side)][:5000].to(device); y=h1b.target_phi(a,args.features)
        met=h1b.scalar_metrics(a,y,v['decoder'].to(device),**v['spec'])
        spk,active=izh_activity(a[:2500],**v['spec'])
        aggregate.append({'layer':li,'side':side,**v['spec'],**met,
                          'mean_spikes_per_feature':spk,'spiking_feature_fraction':active})
    write_csv(args.out/'selected_group_metrics.csv',aggregate)

    result={
        'stage':'H1B2_1_QK_LAYER_SPECIFIC_SCAN',
        'device':str(device),
        'architecture_changed':False,
        'search_space':{'bias':[2,4,6,8,10],'gain':[.2,.35,.5,.75,1.0,1.4],'window_ms':[6,8,10,12]},
        'old_shared_spec':old_spec,
        'baseline':{'fresh':baseline_fresh,'persistent':baseline_persistent},
        'selected':compact_selected(selected),
        'selected_group_metrics':aggregate,
        'combined':{'fresh':calibrated_fresh,'persistent':calibrated_persistent},
        'improvement':{
            'fresh_rel_l2_absolute':float(baseline_fresh['rel_l2']-calibrated_fresh['rel_l2']),
            'fresh_rel_l2_fraction':float(1-calibrated_fresh['rel_l2']/max(1e-12,baseline_fresh['rel_l2'])),
            'persistent_max_rel_l2_absolute':float(baseline_persistent['max_rel_l2']-calibrated_persistent['max_rel_l2']),
            'persistent_max_rel_l2_fraction':float(1-calibrated_persistent['max_rel_l2']/max(1e-12,baseline_persistent['max_rel_l2'])),
            'mean_selected_clamp':float(np.mean([r['clamp_fraction'] for r in aggregate])),
            'max_selected_window_ms':int(max(r['window_ms'] for r in aggregate)),
            'mean_spikes_per_feature':float(np.mean([r['mean_spikes_per_feature'] for r in aggregate])),
        },
        'decision_targets':{
            'fresh_rel_l2_target':0.05,
            'persistent_max_rel_l2_target':0.08,
            'argmax_agreement_target':0.95,
            'clamp_target':0.02,
            'note':'Targets guide whether B2.2 architecture adjustment is needed; missing a target is not a workflow failure.'
        }
    }
    h0.json_dump(args.out/'H1B21_RESULTS.json',result)
    torch.save({'selected':{f'L{li}_{side}':{'spec':v['spec'],'decoder':v['decoder'].cpu()} for (li,side),v in selected.items()}},
               args.out/'h1b21_calibration.pt')
    print('H1B21_DONE',json.dumps({
        'baseline_fresh':baseline_fresh,
        'calibrated_fresh':calibrated_fresh,
        'baseline_persistent':{'max_rel_l2':baseline_persistent['max_rel_l2'],'min_argmax_agreement':baseline_persistent['min_argmax_agreement']},
        'calibrated_persistent':{'max_rel_l2':calibrated_persistent['max_rel_l2'],'min_argmax_agreement':calibrated_persistent['min_argmax_agreement']},
        'improvement':result['improvement'],
        'selected_specs':{k:v['spec'] for k,v in result['selected'].items()},
    },indent=2),flush=True)


if __name__=='__main__':
    main()
