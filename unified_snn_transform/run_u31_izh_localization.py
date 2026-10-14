#!/usr/bin/env python3
from __future__ import annotations

import argparse, copy, json
from pathlib import Path
import numpy as np
import torch

import run_unified_gate as u
import run_u3_trained_retrieval as u3


def target_phi(a: torch.Tensor, features: int) -> torch.Tensor:
    return torch.exp(a.clamp(-30.,30.)) * (features ** -0.5) + 1e-6


def load_reference(path: str):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    c=ck['config']
    model=u3.RetrievalModel(int(c['d']),int(c['heads']),int(c['features']),int(c['ff']),int(c['value_dim']),int(c['seed'])+101)
    model.load_state_dict(ck['model'],strict=True)
    model.eval()
    gen=u3.AssocGenerator(int(c['tokens']),int(c['d']),int(c['value_dim']),int(c['seed']))
    return model,gen,c


def split_scalar(a: torch.Tensor, features: int, seed: int, cap: int=120000):
    a=a.detach().flatten().cpu()
    g=torch.Generator();g.manual_seed(seed)
    if len(a)>cap:
        a=a[torch.randperm(len(a),generator=g)[:cap]]
    p=torch.randperm(len(a),generator=g)
    n=max(100,min(int(.70*len(a)),len(a)-100))
    fit=a[p[:n]]; val=a[p[n:]]
    return fit,val,target_phi(fit,features),target_phi(val,features)


def expanded_candidates(a: torch.Tensor, features: int, seed: int, topn: int=6):
    fit,val,yf,yv=split_scalar(a,features,seed)
    grid=[{'bias':float(b),'gain':float(g),'window_ms':int(w)}
          for b in (0.,1.,2.,3.,4.,5.,6.,8.,10.)
          for g in (.15,.25,.35,.5,.75,1.,1.4,2.)
          for w in (4,6,8,10,12,16)]
    rows=[]; keep=[]
    for spec in grid:
        dec=u.fit_decoder(fit,yf,**spec)
        raw=u.traj_design(val,**spec)@dec
        pred=raw.clamp_min(1e-8)
        rr=float(torch.linalg.vector_norm(pred-yv)/torch.linalg.vector_norm(yv).clamp_min(1e-12))
        clamp=float((raw<=1e-8).float().mean())
        spk,active=u.izh_activity(val[:min(2500,len(val))],**spec)
        # Ranking is feature fidelity first. Window is only a weak tie/cost penalty.
        rank=rr + 1e-5*spec['window_ms']
        row={**spec,'scalar_rel_l2':rr,'clamp_fraction':clamp,
             'mean_spikes':spk,'active_fraction':active,'rank_score':rank}
        rows.append(row); keep.append((rank,spec,dec,row))
    keep.sort(key=lambda z:z[0])
    return keep[:topn], sorted(rows,key=lambda z:z['rank_score'])


def replace(ref, q_item=None, k_item=None):
    cand=copy.deepcopy(ref)
    if q_item is not None:
        _,spec,dec,_=q_item
        cand.block.attn.phi_q=u.IzhPhi(ref.block.attn.phi_q,spec,dec)
    if k_item is not None:
        _,spec,dec,_=k_item
        cand.block.attn.phi_k=u.IzhPhi(ref.block.attn.phi_k,spec,dec)
    return cand.eval()


@torch.no_grad()
def parity(ref, cand, gen, seed: int, batches: int=4, batch: int=64):
    g=torch.Generator();g.manual_seed(seed)
    pr=[];pc=[];hr=[];hc=[];target=[]
    Hnum=Hden=gnum=gden=0.0
    for _ in range(batches):
        x,t,_,_=gen.sample(batch,g); ch=torch.ones(x.shape[:2],dtype=torch.bool)
        yr,sr,_=ref.block.step(x,None,ch); yc,sc,_=cand.block.step(x,None,ch)
        rr=ref.readout(yr[:,-1]); cc=cand.readout(yc[:,-1])
        pr.append(rr);pc.append(cc);hr.append(yr[:,-1]);hc.append(yc[:,-1]);target.append(t)
        Hnum+=float(torch.sum((sc.H-sr.H)**2));Hden+=float(torch.sum(sr.H**2))
        gnum+=float(torch.sum((sc.g-sr.g)**2));gden+=float(torch.sum(sr.g**2))
    pr=torch.cat(pr);pc=torch.cat(pc);hr=torch.cat(hr);hc=torch.cat(hc);target=torch.cat(target)
    return {
        'prediction_rel_l2':u3.rel_l2(pc,pr),
        'prediction_cosine':u3.cosine(pc,pr),
        'query_hidden_rel_l2':u3.rel_l2(hc,hr),
        'H_rel_l2':float(np.sqrt(Hnum/max(Hden,1e-30))),
        'g_rel_l2':float(np.sqrt(gnum/max(gden,1e-30))),
        'reference_task':u3.metrics(pr,target),
        'candidate_task':u3.metrics(pc,target),
    }


def old_item(a, features):
    spec,dec,row,_=u.calibrate_side(a,features)
    return (row['score'],spec,dec,{**spec,**row})


def concise(item):
    return dict(item[3])


def run(args):
    u.seed_all(args.seed);torch.set_num_threads(args.threads)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    ref,gen,cfg=load_reference(args.checkpoint)
    aq,ak=u3.collect_empirical_a(ref,gen,args.seed+1000,batches=8,batch=64)

    # First localize the existing U3 interface without changing the calibration rule.
    oq=old_item(aq,ref.block.attn.m); ok=old_item(ak,ref.block.attn.m)
    loc={
        'Q_only_old':parity(ref,replace(ref,q_item=oq),gen,args.seed+2000,args.test_batches,args.batch),
        'K_only_old':parity(ref,replace(ref,k_item=ok),gen,args.seed+2000,args.test_batches,args.batch),
        'QK_old':parity(ref,replace(ref,q_item=oq,k_item=ok),gen,args.seed+2000,args.test_batches,args.batch),
        'Q_old_spec':concise(oq),'K_old_spec':concise(ok),
    }

    # Broaden only the scalar neuronal interface. No task/attention weights are changed.
    qtop,qrows=expanded_candidates(aq,ref.block.attn.m,args.seed+3001,args.topn)
    ktop,krows=expanded_candidates(ak,ref.block.attn.m,args.seed+3002,args.topn)

    # Select pair only against exact-model predictions on a held-out parity set; no task label is used.
    pair_rows=[];best=None
    for qi,qitem in enumerate(qtop):
        for ki,kitem in enumerate(ktop):
            cand=replace(ref,qitem,kitem)
            m=parity(ref,cand,gen,args.seed+4000,args.select_batches,args.batch)
            row={'q_rank':qi,'k_rank':ki,'prediction_rel_l2':m['prediction_rel_l2'],
                 'query_hidden_rel_l2':m['query_hidden_rel_l2'],'q_window':qitem[1]['window_ms'],'k_window':kitem[1]['window_ms']}
            pair_rows.append(row)
            # Numeric parity dominates; tiny cost tie-break prevents gratuitous windows.
            score=m['prediction_rel_l2'] + args.window_penalty*(qitem[1]['window_ms']+kitem[1]['window_ms'])
            if best is None or score<best[0]: best=(score,qi,ki,qitem,kitem,row)

    _,bqi,bki,bq,bk,_=best
    final_cand=replace(ref,bq,bk)
    held=parity(ref,final_cand,gen,args.seed+5000,args.test_batches,args.batch)
    qonly_new=parity(ref,replace(ref,q_item=bq),gen,args.seed+5000,args.test_batches,args.batch)
    konly_new=parity(ref,replace(ref,k_item=bk),gen,args.seed+5000,args.test_batches,args.batch)

    # Distribution-shift diagnostic relative to old scalar calibration.
    result={
        'stage':'UNIFIED_U3_1_IZH_TRAINED_DISTRIBUTION',
        'config':vars(args),
        'old_interface_localization':loc,
        'expanded_search':{
            'grid_size_per_side':len(qrows),
            'q_top':[concise(z) for z in qtop],
            'k_top':[concise(z) for z in ktop],
            'pair_selection':sorted(pair_rows,key=lambda z:z['prediction_rel_l2']),
            'selected_q_rank':bqi,'selected_k_rank':bki,
            'selected_Q':concise(bq),'selected_K':concise(bk),
        },
        'new_interface_localization':{'Q_only':qonly_new,'K_only':konly_new},
        'heldout_QK':held,
    }
    gates={
        'heldout_prediction_rel_l2_le_1pct':held['prediction_rel_l2']<=.01,
        'heldout_prediction_cosine_ge_0p999':held['prediction_cosine']>=.999,
        'heldout_task_sign_drop_le_0p02':held['candidate_task']['sign_accuracy']>=held['reference_task']['sign_accuracy']-.02,
        'no_architecture_or_weight_change':True,
    }
    result['gates']=gates;result['PASS']=all(gates.values())
    result['decision']=(
        'freeze trained-distribution Izh interface and proceed to selective-state functional gate' if result['PASS']
        else 'do not proceed to U4; trained-distribution Izh parity still exceeds 1%, localize/alter only the neuronal feature interface'
    )
    (out/'U31_RESULTS.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    # Scientific gate is archived even on failure; workflow continues so diagnostics/artifacts are retained.
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',default='unified_snn_transform/run/U3/U3_REFERENCE.pt')
    ap.add_argument('--out',default='unified_snn_transform/run/U31')
    ap.add_argument('--seed',type=int,default=17);ap.add_argument('--topn',type=int,default=6)
    ap.add_argument('--select-batches',type=int,default=3);ap.add_argument('--test-batches',type=int,default=8)
    ap.add_argument('--batch',type=int,default=64);ap.add_argument('--window-penalty',type=float,default=1e-5)
    ap.add_argument('--threads',type=int,default=4)
    run(ap.parse_args())


if __name__=='__main__': main()
