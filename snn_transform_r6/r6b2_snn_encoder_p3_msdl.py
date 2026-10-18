#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R6B.2 — reconnect the frozen sensory-SNN tokenizer to the exact P3 image MSDL core.

Scientific question
-------------------
Does replacing ONLY the old raw-patch KMeans image front-end with the frozen
R6B.1 sensory SNN encoder improve the already-established plain P3 image-learning
failure mode?

Historical comparator (NOT re-run here): P3I-2 plain_s05, GitHub Actions run
39130766259 / artifact SNN_TRANSFORM_P3I2_plain_s05.  That run used the exact P3
E/M local rule, plain online mismatch, plain startup MSDL scaffold, and uniform
internal forward learning-rate scale 0.5.  Its final generation condition-match
was 0.4722222 and its 50%-mask CE changed 1.827512 -> 1.774896.

Intervention
------------
Only the visual token source is changed:
  old: 28x28 -> fixed 4x4 patches -> KMeans64 -> 7x7 discrete tokens
  new: 28x28 sensory LIF -> spiking conv -> 7x7 latent LIF populations -> VQ64
The R6B.1 checkpoint is frozen.  No gradient/backpropagation/optimizer updates the
tokenizer or the P3 core in this experiment.  The P3 core remains exact local
Izhikevich-derived E/M with dW ~ M E^T.

This is not an official MaskGIT reimplementation.  MaskGIT is used only as the
public architectural reference that separates an image tokenizer from the masked
bidirectional token model.  Cross-tokenizer CE values are reported but are NOT
interpreted as directly comparable because the discrete vocabularies have
changed; image-space generation condition match is the primary historical bridge.
"""
from __future__ import annotations

import argparse, csv, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'snn_transform_p3_image'))
sys.path.insert(0, str(HERE))

import p3i_cb_exact_msdl as core
from r6b1_spiking_tokenizer import SNNVQ, lif_step

HISTORICAL = {
    'run_id': 39130766259,
    'artifact': 'SNN_TRANSFORM_P3I2_plain_s05',
    'front_end': 'raw 4x4 patches -> KMeans64',
    'online_variant': 'plain',
    'plain_scaffold': True,
    'uniform_internal_forward_lr_scale': 0.5,
    'dominant_fraction': 0.6464326199425537,
    'scaffold_ce50': 1.827512065235565,
    'final_ce50': 1.7748955097931385,
    'scaffold_generation_condition_match': 0.5833333333333334,
    'final_generation_condition_match': 0.4722222222222222,
}


def load_frozen_tokenizer(path: Path, device: torch.device):
    ck = torch.load(path, map_location=device, weights_only=False)
    meta = ck.get('metrics', {})
    cfg = meta.get('config', {})
    model = SNNVQ(
        dim=int(cfg.get('dim', 16)),
        steps=int(cfg.get('steps', 8)),
        alpha=float(cfg.get('alpha', .75)),
        sensory_gain=float(cfg.get('sensory_gain', 1.6)),
        beta=float(cfg.get('beta', .85)),
        commitment=float(cfg.get('commitment', .25)),
        entropy_weight=float(cfg.get('entropy_weight', .02)),
        entropy_temp=float(cfg.get('entropy_temp', .5)),
    ).to(device)
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, meta


@torch.no_grad()
def encode_snn(model: SNNVQ, x: np.ndarray, device: torch.device, batch: int = 256):
    """Encode raw 28x28 images to 49 frozen VQ IDs without running the decoder."""
    out=[]; latent_rates=[]; sensory_rates=[]
    xx=torch.from_numpy(x[:,None]).float()
    for st in range(0, len(xx), batch):
        inp=xx[st:st+batch].to(device)
        ms=torch.zeros_like(inp); m1=m2=m3=None
        ss=s3=0.0
        for _ in range(model.steps):
            sp0,ms=lif_step(model.sensory_gain*inp,ms,model.beta)
            cur1=model.b1(model.c1(sp0)); m1=torch.zeros_like(cur1) if m1 is None else m1
            sp1,m1=lif_step(cur1,m1,model.beta)
            cur2=model.b2(model.c2(sp1)); m2=torch.zeros_like(cur2) if m2 is None else m2
            sp2,m2=lif_step(cur2,m2,model.beta)
            cur3=model.b3(model.c3(sp2)); m3=torch.zeros_like(cur3) if m3 is None else m3
            sp3,m3=lif_step(cur3,m3,model.beta)
            ss=ss+sp0; s3=s3+sp3
        rate0=ss/model.steps; rate3=s3/model.steps
        z=model.alpha*rate3 + (1.0-model.alpha)*torch.tanh(m3)
        flat=z.permute(0,2,3,1).contiguous().view(-1, model.vq.dim)
        emb=model.vq.codebook.weight
        dist=flat.square().sum(1,keepdim=True)+emb.square().sum(1)[None]-2*flat@emb.t()
        ids=dist.argmin(1).view(len(inp),7,7)
        out.append(ids.cpu().numpy().reshape(len(inp),49).astype(np.int64))
        sensory_rates.append(float(rate0.mean().cpu())); latent_rates.append(float(rate3.mean().cpu()))
    return np.concatenate(out), {
        'sensory_rate_mean': float(np.mean(sensory_rates)),
        'latent_rate_mean': float(np.mean(latent_rates)),
    }


@torch.no_grad()
def decode_snn(model: SNNVQ, ids: np.ndarray, device: torch.device, batch: int = 256):
    ids=np.asarray(ids,dtype=np.int64).reshape(-1,7,7); out=[]
    emb=model.vq.codebook.weight
    for st in range(0,len(ids),batch):
        ii=torch.from_numpy(ids[st:st+batch]).to(device)
        q=emb[ii].permute(0,3,1,2).contiguous()
        rec=model.dec(q).squeeze(1).clamp(0,1)
        out.append(rec.cpu().numpy().astype(np.float32))
    return np.concatenate(out)


def token_stats(tokens):
    cnt=np.bincount(np.asarray(tokens).reshape(-1),minlength=64).astype(np.float64)
    p=cnt/cnt.sum(); nz=p[p>0]
    return {
        'counts': cnt.tolist(),
        'dominant_token': int(np.argmax(cnt)),
        'dominant_fraction': float(p.max()),
        'active_codes': int((cnt>0).sum()),
        'codebook_usage': float((cnt>0).mean()),
        'perplexity': float(np.exp(-(nz*np.log(nz)).sum())),
    }


def generation_tokens(model,cfg,labels,tok,pos,cls,seed):
    rng=np.random.default_rng(seed); out=[]
    for gi,lab in enumerate(labels):
        s=core.P3ImageSystem(cfg,model,seed+100000+gi)
        arr=np.full(cfg.N,core.MASK,np.int64); order=rng.permutation(cfg.N); changed=None
        for j,t in enumerate(order):
            xn,xraw=core.context_from_arr(arr,int(lab),tok,pos,cls)
            p=s.forward_only(xn,xraw[t],int(t),new_image=(j==0),changed=changed)
            z=int(rng.choice(cfg.V,p=p/p.sum())); arr[t]=z; changed=int(t)
        out.append(arr)
    return np.asarray(out,dtype=np.int64)


def generation_metric(clf, img, labels):
    pred=clf.predict(img.reshape(len(img),-1)); prob=clf.predict_proba(img.reshape(len(img),-1))
    return {
        'condition_match': float(np.mean(pred==labels)),
        'mean_confidence': float(np.mean(prob.max(1))),
        'pred_counts': np.bincount(pred,minlength=3).tolist(),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--D',type=int,default=96); ap.add_argument('--heads',type=int,default=8)
    ap.add_argument('--features',type=int,default=24); ap.add_argument('--ff',type=int,default=192); ap.add_argument('--R',type=int,default=7)
    ap.add_argument('--scaffold-contexts',type=int,default=2000); ap.add_argument('--scaffold-paths',type=int,default=2500); ap.add_argument('--scaffold-lr',type=float,default=.6)
    ap.add_argument('--events',type=int,default=30000); ap.add_argument('--eval-images',type=int,default=128); ap.add_argument('--gen-per-class',type=int,default=12)
    ap.add_argument('--internal-lr-scale',type=float,default=.5); ap.add_argument('--encode-batch',type=int,default=256)
    ap.add_argument('--seed',type=int,default=7311); ap.add_argument('--report-every',type=int,default=5000); ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True,exist_ok=True); device=torch.device('cpu')

    xall,yall,xte0,yte0=core.load(a.data); x012,y012=core.subset012(xall,yall); xte,yte=core.subset012(xte0,yte0)
    xtr,ytr,xva,yva=core.split(x012,y012,.1,7)

    tokenizer,tok_meta=load_frozen_tokenizer(a.tokenizer,device)
    tenc=time.time(); tr,sp_tr=encode_snn(tokenizer,xtr,device,a.encode_batch); va,sp_va=encode_snn(tokenizer,xva,device,a.encode_batch); te,sp_te=encode_snn(tokenizer,xte,device,a.encode_batch)
    print('TOKENIZATION_DONE',json.dumps({'elapsed_s':time.time()-tenc,'train':len(tr),'val':len(va),'test':len(te),'train_activity':sp_tr}),flush=True)
    tstats=token_stats(tr)

    cfg=core.Config(D=a.D,H=a.heads,M=a.features,FF=a.ff,R=a.R)
    model=core.init_model(cfg,a.seed)
    for k in ('lrq','lrk','lrv','lro','lr1','lr2'):
        model[k]*=a.internal_lr_scale
    tok,pos,cls=core.build_embeddings(cfg,a.seed+99)

    ones=np.ones(cfg.V,np.float64)
    F,T=core.build_scaffold_contexts(model,cfg,tr,ytr,tok,pos,cls,a.scaffold_contexts,a.seed+101)
    ts=time.time(); model['WR']=core.train_readout_msdl(F,T,ones,a.scaffold_paths,a.scaffold_lr,a.seed+202)
    print('SCAFFOLD_DONE',json.dumps({'elapsed_s':time.time()-ts,'contexts':a.scaffold_contexts,'paths':a.scaffold_paths}),flush=True)

    scaffold_eval={str(r):core.eval_masked(model,cfg,va,yva,tok,pos,cls,r,a.seed+300+int(100*r),a.eval_images) for r in (.25,.5,.9,1.0)}
    print('SCAFFOLD_EVAL',json.dumps(scaffold_eval),flush=True)

    labs=np.repeat(np.arange(3),a.gen_per_class).astype(np.int64)
    clf=LogisticRegression(max_iter=250,solver='lbfgs',random_state=7).fit(xtr.reshape(len(xtr),-1),ytr)
    stok=generation_tokens(model,cfg,labs,tok,pos,cls,a.seed+606); simg=decode_snn(tokenizer,stok,device); core.save_grid(simg,labs,a.out/'generation_scaffold.png'); gen_scaffold=generation_metric(clf,simg,labs)

    sysm=core.P3ImageSystem(cfg,model,a.seed+303)
    hist=core.train_full(sysm,cfg,tr,ytr,tok,pos,cls,ones,a.events,a.seed+404,a.report_every)
    final=model.copy(); final.update(sysm.weights())
    final_eval={str(r):core.eval_masked(final,cfg,te,yte,tok,pos,cls,r,a.seed+500+int(100*r),a.eval_images) for r in (.25,.5,.9,1.0)}
    gtok=generation_tokens(final,cfg,labs,tok,pos,cls,a.seed+606); gimg=decode_snn(tokenizer,gtok,device); core.save_grid(gimg,labs,a.out/'generation_final.png'); gen_final=generation_metric(clf,gimg,labs)

    moves={k:float(np.linalg.norm(final[k]-model[k])/(np.linalg.norm(model[k])+1e-12)) for k in ('WQ','WK','WV','WO','W1','W2')}
    rec={'BR':float(np.mean(np.abs(sysm.BR))),'BO':float(np.mean(np.abs(sysm.BO))),'B2':float(np.mean(np.abs(sysm.B2)))}
    sc50=scaffold_eval['0.5']; fn50=final_eval['0.5']; hist_gen=HISTORICAL['final_generation_condition_match']
    delta_gen=float(gen_final['condition_match']-hist_gen)
    gates={
        'plain_no_class_balance': True,
        'frozen_snn_tokenizer': True,
        'candidate_backprop_in_R6B2': False,
        'ce50_improves_with_plain_msdl': bool(fn50['ce'] < sc50['ce']),
        'generation_final_match_ge_0p60': bool(gen_final['condition_match'] >= .60),
        'generation_improvement_vs_historical_ge_0p10': bool(delta_gen >= .10),
        'reciprocal_populations_alive': bool(all(v>0 for v in rec.values())),
    }
    causal_support=bool(gates['generation_final_match_ge_0p60'] and gates['generation_improvement_vs_historical_ge_0p10'])
    diag={
        'stage':'R6B.2 SNN encoder reconnect causal test',
        'question':'Does the frozen R6B.1 sensory-SNN tokenizer improve the plain exact-P3 image route without CB?',
        'historical_comparator':HISTORICAL,
        'comparison_rule':{
            'primary':'image-space generation condition_match; historical KMeans route is not re-run',
            'secondary':'within-new-tokenizer masked CE/accuracy before vs after P3 MSDL',
            'cross_tokenizer_ce_warning':'Do not interpret absolute CE differences between KMeans and SNN-VQ vocabularies as a fair causal metric.'
        },
        'tokenizer':{
            'architecture':'28x28 sensory LIF -> 32x14x14 LIF -> 64x7x7 LIF -> 16x7x7 latent LIF -> VQ64',
            'frozen':True,
            'checkpoint_metadata':tok_meta,
            'train_token_stats':tstats,
            'activity_train':sp_tr,'activity_val':sp_va,'activity_test':sp_te,
        },
        'developed_p3_core':{'resident_Hg':True,'resident_Xphi':True,'collective_K_R':cfg.R,'persistent_EM_Izh':True,'seven_forward_groups':True,'adaptive_reciprocal':True},
        'online_variant':'plain','plain_startup_scaffold':True,'class_conditioning_retained':True,
        'internal_forward_lr_scale':a.internal_lr_scale,'events':a.events,
        'scaffold_eval':scaffold_eval,'final_eval':final_eval,
        'generation_scaffold':gen_scaffold,'generation_final':gen_final,
        'generation_delta_vs_historical':delta_gen,
        'slow_updates':sysm.update_count,'mean_abs_reciprocal':rec,'relative_internal_weight_move':moves,
        'gates':gates,'supports_input_bottleneck_hypothesis':causal_support,
        'interpretation_if_false':'No strong evidence that replacing the KMeans front-end by this frozen discrete SNN-VQ front-end is sufficient to solve the downstream plain-MSDL failure. This does not test direct continuous spike/membrane-state coupling into SNN-Transform.',
        'history':hist,
    }
    (a.out/'metrics.json').write_text(json.dumps(diag,indent=2))
    np.savez_compressed(a.out/'final_state.npz',**{k:v for k,v in final.items() if isinstance(v,np.ndarray)},tok=tok,pos=pos,cls=cls)
    np.savetxt(a.out/'generated_tokens_final.csv',gtok,fmt='%d',delimiter=','); np.savetxt(a.out/'generated_tokens_scaffold.csv',stok,fmt='%d',delimiter=',')
    if hist:
        with (a.out/'history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(hist[0].keys())); w.writeheader(); w.writerows(hist)
    print('FINAL',json.dumps(diag),flush=True)

if __name__=='__main__':
    main()
