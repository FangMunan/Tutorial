#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R6B.3 — frozen SNN encoder + exact P3/MSDL + MaskGIT training and decoding.

Purpose
-------
R6B.2 used a sequential reveal/fill protocol that was not MaskGIT.  R6B.3 keeps
the frozen R6B.1 SNN-VQ tokenizer and exact P3 local dynamical learning core, but
aligns BOTH masked-token training contexts and generation with MaskGIT semantics.

Training protocol:
  r ~ Uniform[0,1), gamma(r)=cos(pi*r/2),
  uniformly mask ceil(gamma(r)*N) positions,
  keep that corrupted context fixed,
  let every masked target contribute to MSDL before one forward-weight update,
  never reveal a ground-truth masked token inside the context.

Generation protocol:
  all MASK -> predict every current MASK -> categorical proposals -> sampled-token
  confidence -> cosine schedule + annealed Gumbel random-top-k -> re-mask low
  confidence proposals -> repeat.

The MaskGIT outer controller remains non-neural and small-N, so it can later live
on a SpiNNaker control core.  P3/MSDL remains local/dynamical rather than BP.
"""
from __future__ import annotations

import argparse, csv, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'snn_transform_p3_image'))
sys.path.insert(0,str(HERE))

import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
from maskgit_scheduled_parallel import scheduled_parallel_decode,audit_trace,controller_self_test
from maskgit_msdl_training import (
    MaskGITP3System,
    build_maskgit_scaffold_contexts,
    masking_self_test,
    train_maskgit_msdl,
)


def generation_tokens_maskgit(model,cfg,labels,tok,pos,cls,seed,
                               num_iter=8,choice_temperature=4.5):
    """MaskGIT generation with P3 SNN-Transform as probability predictor.

    The CPU reference loops over receiver queries, but every receiver in a round
    sees the same immutable token context and no proposal is written back until
    all current MASK positions have been queried.
    """
    outputs=[];traces=[];audits=[]
    for gi,lab in enumerate(np.asarray(labels,dtype=np.int64)):
        system=core.P3ImageSystem(cfg,model,seed+100000+gi)
        sample_rng=np.random.default_rng(seed+200000+gi)

        def predict_masked(cur_ids,masked_positions,step):
            cur_ids=np.asarray(cur_ids,dtype=np.int64)
            masked_positions=np.asarray(masked_positions,dtype=np.int64)
            xn,xraw=core.context_from_arr(cur_ids,int(lab),tok,pos,cls)
            probs=np.empty((len(masked_positions),cfg.V),dtype=np.float64)
            for qi,t in enumerate(masked_positions):
                probs[qi]=system.forward_only(
                    xn,xraw[int(t)],int(t),new_image=(qi==0),changed=None)
            return probs

        initial=np.full(cfg.N,core.MASK,dtype=np.int64)
        final_ids,trace=scheduled_parallel_decode(
            initial,mask_token_id=core.MASK,vocab_size=cfg.V,
            predict_masked=predict_masked,rng=sample_rng,
            num_iter=num_iter,choice_temperature=choice_temperature)
        audit=audit_trace(trace,cfg.N,num_iter)
        outputs.append(final_ids);traces.append(trace);audits.append(audit)
    return np.asarray(outputs,dtype=np.int64),traces,audits


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--D',type=int,default=96);ap.add_argument('--heads',type=int,default=8)
    ap.add_argument('--features',type=int,default=24);ap.add_argument('--ff',type=int,default=192);ap.add_argument('--R',type=int,default=7)
    ap.add_argument('--scaffold-contexts',type=int,default=2000);ap.add_argument('--scaffold-paths',type=int,default=2500);ap.add_argument('--scaffold-lr',type=float,default=.6)
    # `events` retains the historical compute-budget meaning: masked-target contributions.
    ap.add_argument('--events',type=int,default=30000)
    ap.add_argument('--eval-images',type=int,default=128);ap.add_argument('--gen-per-class',type=int,default=12)
    ap.add_argument('--internal-lr-scale',type=float,default=.5);ap.add_argument('--encode-batch',type=int,default=256)
    ap.add_argument('--decode-iterations',type=int,default=8);ap.add_argument('--choice-temperature',type=float,default=4.5)
    ap.add_argument('--seed',type=int,default=7311);ap.add_argument('--report-every',type=int,default=5000);ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True,exist_ok=True);device=torch.device('cpu')

    # Pure structural audits, independent of neural quality.
    controller_audit=controller_self_test()
    training_mask_audit=masking_self_test(n_tokens=49,vocab_size=64,trials=128,seed=a.seed+17)
    print('MASKGIT_CONTROLLER_AUDIT',json.dumps(controller_audit),flush=True)
    print('MASKGIT_TRAINING_MASK_AUDIT',json.dumps(training_mask_audit),flush=True)

    xall,yall,xte0,yte0=core.load(a.data);x012,y012=core.subset012(xall,yall);xte,yte=core.subset012(xte0,yte0)
    xtr,ytr,xva,yva=core.split(x012,y012,.1,7)

    tokenizer,tok_meta=base.load_frozen_tokenizer(a.tokenizer,device)
    tenc=time.time();tr,sp_tr=base.encode_snn(tokenizer,xtr,device,a.encode_batch);va,sp_va=base.encode_snn(tokenizer,xva,device,a.encode_batch);te,sp_te=base.encode_snn(tokenizer,xte,device,a.encode_batch)
    print('TOKENIZATION_DONE',json.dumps({'elapsed_s':time.time()-tenc,'train':len(tr),'val':len(va),'test':len(te),'train_activity':sp_tr}),flush=True)
    tstats=base.token_stats(tr)

    cfg=core.Config(D=a.D,H=a.heads,M=a.features,FF=a.ff,R=a.R)
    model=core.init_model(cfg,a.seed)
    for k in ('lrq','lrk','lrv','lro','lr1','lr2'):
        model[k]*=a.internal_lr_scale
    tok,pos,cls=core.build_embeddings(cfg,a.seed+99)

    # Startup WR scaffold now receives MaskGIT-style corrupted contexts as well.
    ones=np.ones(cfg.V,np.float64)
    F,T,scaffold_mask_audit=build_maskgit_scaffold_contexts(
        model,cfg,tr,ytr,tok,pos,cls,a.scaffold_contexts,a.seed+101)
    ts=time.time();model['WR']=core.train_readout_msdl(F,T,ones,a.scaffold_paths,a.scaffold_lr,a.seed+202)
    print('SCAFFOLD_DONE',json.dumps({'elapsed_s':time.time()-ts,'contexts':a.scaffold_contexts,'paths':a.scaffold_paths,'masking':scaffold_mask_audit}),flush=True)

    scaffold_eval={str(r):core.eval_masked(model,cfg,va,yva,tok,pos,cls,r,a.seed+300+int(100*r),a.eval_images) for r in (.25,.5,.9,1.0)}
    print('SCAFFOLD_EVAL',json.dumps(scaffold_eval),flush=True)

    labs=np.repeat(np.arange(3),a.gen_per_class).astype(np.int64)
    clf=LogisticRegression(max_iter=250,solver='lbfgs',random_state=7).fit(xtr.reshape(len(xtr),-1),ytr)
    stok,strace,saudit=generation_tokens_maskgit(model,cfg,labs,tok,pos,cls,a.seed+606,a.decode_iterations,a.choice_temperature)
    simg=base.decode_snn(tokenizer,stok,device);core.save_grid(simg,labs,a.out/'generation_scaffold.png');gen_scaffold=base.generation_metric(clf,simg,labs)

    # Main online P3/MSDL now uses MaskGIT random multi-mask contexts.
    sysm=MaskGITP3System(cfg,model,a.seed+303)
    hist,online_training_audit=train_maskgit_msdl(
        sysm,cfg,tr,ytr,tok,pos,cls,ones,a.events,a.seed+404,a.report_every)
    final=model.copy();final.update(sysm.weights())
    final_eval={str(r):core.eval_masked(final,cfg,te,yte,tok,pos,cls,r,a.seed+500+int(100*r),a.eval_images) for r in (.25,.5,.9,1.0)}
    gtok,gtrace,gaudit=generation_tokens_maskgit(final,cfg,labs,tok,pos,cls,a.seed+606,a.decode_iterations,a.choice_temperature)
    gimg=base.decode_snn(tokenizer,gtok,device);core.save_grid(gimg,labs,a.out/'generation_final.png');gen_final=base.generation_metric(clf,gimg,labs)

    moves={k:float(np.linalg.norm(final[k]-model[k])/(np.linalg.norm(model[k])+1e-12)) for k in ('WQ','WK','WV','WO','W1','W2')}
    rec={'BR':float(np.mean(np.abs(sysm.BR))),'BO':float(np.mean(np.abs(sysm.BO))),'B2':float(np.mean(np.abs(sysm.B2)))}
    sc50=scaffold_eval['0.5'];fn50=final_eval['0.5'];all_decode_audits=saudit+gaudit
    gates={
        'maskgit_training_mask_self_test':bool(training_mask_audit['pass']),
        'maskgit_online_training_protocol_audit':bool(online_training_audit['pass']),
        'maskgit_controller_self_test':bool(controller_audit['pass']),
        'all_generation_traces_pass_maskgit_audit':bool(all(x['pass'] for x in all_decode_audits)),
        'plain_no_class_balance':True,
        'frozen_snn_tokenizer':True,
        'candidate_backprop_in_R6B3':False,
        'ce50_improves_with_plain_msdl':bool(fn50['ce']<sc50['ce']),
        'generation_final_match_ge_0p60':bool(gen_final['condition_match']>=.60),
        'generation_final_not_worse_than_scaffold':bool(gen_final['condition_match']>=gen_scaffold['condition_match']),
        'reciprocal_populations_alive':bool(all(v>0 for v in rec.values())),
    }
    diag={
        'stage':'R6B.3 MaskGIT-aligned training + scheduled-parallel generation',
        'question':'Can frozen SNN-VQ + exact P3/MSDL learn and generate under matched MaskGIT masking semantics?',
        'r6b2_generation_status':'invalid as a MaskGIT benchmark because it used random one-token-at-a-time permanent filling; its masked-prediction diagnostics remain historical only',
        'no_kmeans_rerun':True,
        'maskgit_training_protocol':{
            'ratio_sampling':'r ~ Uniform[0,1)',
            'schedule':'gamma(r)=cos(pi*r/2)',
            'mask_count':'ceil(gamma(r)*N)',
            'mask_positions':'uniform without replacement',
            'context':'one corrupted image is held fixed for every masked target',
            'targets':'all masked positions contribute before a forward-weight update',
            'ground_truth_reveal_inside_context':False,
            'forward_predictions':'cached for all masked receivers before the update boundary',
            'msdl_update':'local E/M contributions accumulated then averaged by P3 apply_update',
            'local_EM_temporal_note':'persistent E/M populations advance sequentially in randomized masked-target order; this is the MSDL physical-learning implementation, not BP',
        },
        'maskgit_decode_protocol':{
            'initial_state':'all 49 image tokens MASK',
            'predict':'all currently masked sites from one immutable context each round',
            'sample':'categorical sample independently at every currently masked site',
            'confidence':'P3 SNN-Transform probability assigned to the sampled token',
            'schedule':'cosine gamma(r)=cos(pi*r/2)',
            'mask_len_rule':'public Google implementation floor(initial_unknown*mask_ratio), clamped to [1,current_unknown-1]',
            'remask':'lowest log(confidence)+annealed-Gumbel scores',
            'choice_temperature':a.choice_temperature,
            'iterations':a.decode_iterations,
            'committed_tokens':'never re-predicted; low-confidence proposals are re-masked before commitment',
            'controller':'non-neural small-N reference controller; candidate for SpiNNaker control core',
        },
        'training_mask_self_test':training_mask_audit,
        'scaffold_mask_audit':scaffold_mask_audit,
        'online_training_audit':online_training_audit,
        'controller_self_test':controller_audit,
        'tokenizer':{
            'architecture':'28x28 sensory LIF -> 32x14x14 LIF -> 64x7x7 LIF -> 16x7x7 latent LIF -> VQ64',
            'frozen':True,'checkpoint_metadata':tok_meta,'train_token_stats':tstats,
            'activity_train':sp_tr,'activity_val':sp_va,'activity_test':sp_te,
        },
        'developed_p3_core':{'resident_Hg':True,'resident_Xphi':True,'collective_K_R':cfg.R,'persistent_EM_Izh':True,'seven_forward_groups':True,'adaptive_reciprocal':True},
        'online_variant':'plain','plain_startup_scaffold':True,'class_conditioning_retained':True,
        'internal_forward_lr_scale':a.internal_lr_scale,'requested_target_events':a.events,
        'scaffold_eval':scaffold_eval,'final_eval':final_eval,
        'generation_scaffold':gen_scaffold,'generation_final':gen_final,
        'slow_updates':sysm.update_count,'mean_abs_reciprocal':rec,'relative_internal_weight_move':moves,
        'gates':gates,'history':hist,
    }
    (a.out/'metrics.json').write_text(json.dumps(diag,indent=2))
    (a.out/'maskgit_trace_scaffold.json').write_text(json.dumps(strace,indent=2))
    (a.out/'maskgit_trace_final.json').write_text(json.dumps(gtrace,indent=2))
    (a.out/'maskgit_audit_scaffold.json').write_text(json.dumps(saudit,indent=2))
    (a.out/'maskgit_audit_final.json').write_text(json.dumps(gaudit,indent=2))
    (a.out/'maskgit_training_audit.json').write_text(json.dumps({'self_test':training_mask_audit,'scaffold':scaffold_mask_audit,'online':online_training_audit},indent=2))
    np.savez_compressed(a.out/'final_state.npz',**{k:v for k,v in final.items() if isinstance(v,np.ndarray)},tok=tok,pos=pos,cls=cls)
    np.savetxt(a.out/'generated_tokens_final.csv',gtok,fmt='%d',delimiter=',');np.savetxt(a.out/'generated_tokens_scaffold.csv',stok,fmt='%d',delimiter=',')
    if hist:
        with (a.out/'history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(hist[0].keys()));w.writeheader();w.writerows(hist)
    print('FINAL',json.dumps(diag),flush=True)


if __name__=='__main__':
    main()
