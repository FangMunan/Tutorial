#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R6B.3 — label-free SNN-VQ + exact P3/MSDL + MaskGIT training/decoding.

R6B.3 aligns MaskGIT masking semantics while removing the historical class-label
input and fixed-random embedding interface.

Neural input:
  raw image -> frozen sensory SNN-VQ -> discrete visual tokens
  visual token -> frozen SNN-VQ codebook geometry -> deterministic P3 embedding
  7x7 coordinate -> deterministic 2-D sinusoidal position embedding
  NO class embedding and NO class label enters the neural predictor.

Training:
  r~Uniform[0,1), gamma(r)=cos(pi*r/2), mask ceil(gamma(r)N) positions,
  keep one corrupted context fixed, let every masked target contribute to local
  MSDL, then apply one forward-weight update.  No target is revealed inside the
  batch.

Generation:
  all MASK -> parallel proposals at all current MASK sites -> sampled-token
  confidence -> cosine schedule + annealed Gumbel random-top-k -> re-mask low
  confidence proposals -> repeat.

MNIST labels are retained only outside the model for selecting the historical
0/1/2 evaluation subset and for an explicitly post-hoc image classifier.  They
never enter token/position embeddings, P3 context construction, learning, or
generation.
"""
from __future__ import annotations

import argparse,csv,json,math,random,sys,time
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
from maskgit_input_representation import (
    build_input_embeddings,
    context_from_arr_unconditional,
    representation_self_test,
)
from maskgit_scheduled_parallel import scheduled_parallel_decode,audit_trace,controller_self_test
from maskgit_msdl_training_unconditional import (
    MaskGITP3System,
    build_maskgit_scaffold_contexts_unconditional,
    eval_masked_unconditional,
    masking_self_test,
    train_maskgit_msdl_unconditional,
)


def generation_tokens_maskgit(model,cfg,n_samples,tok,pos,seed,
                               num_iter=8,choice_temperature=4.5):
    """Unconditional MaskGIT generation using P3 as the token predictor."""
    outputs=[];traces=[];audits=[]
    for gi in range(int(n_samples)):
        system=core.P3ImageSystem(cfg,model,seed+100000+gi)
        sample_rng=np.random.default_rng(seed+200000+gi)

        def predict_masked(cur_ids,masked_positions,step):
            cur_ids=np.asarray(cur_ids,dtype=np.int64)
            masked_positions=np.asarray(masked_positions,dtype=np.int64)
            xn,xraw=context_from_arr_unconditional(cur_ids,tok,pos)
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


def save_grid_unlabeled(img,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n=len(img);cols=12;rows=math.ceil(n/cols)
    fig,ax=plt.subplots(rows,cols,figsize=(cols,rows*1.2));ax=np.asarray(ax).reshape(-1)
    for i,a in enumerate(ax):
        a.axis('off')
        if i<n:
            a.imshow(img[i],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
    plt.tight_layout();fig.savefig(path,dpi=170,bbox_inches='tight');plt.close(fig)


def posthoc_generation_metric(clf,img):
    """External diagnostic only; classifier labels never feed the generator."""
    pred=clf.predict(img.reshape(len(img),-1))
    prob=clf.predict_proba(img.reshape(len(img),-1))
    counts=np.bincount(pred,minlength=3).astype(np.int64)
    q=counts/counts.sum()
    nz=q[q>0]
    entropy=float(-(nz*np.log(nz)).sum())
    return {
        'posthoc_pred_counts_012':counts.tolist(),
        'posthoc_pred_entropy_nats':entropy,
        'posthoc_pred_classes_present':int(np.sum(counts>0)),
        'posthoc_mean_classifier_confidence':float(np.mean(prob.max(1))),
        'labels_used_only_posthoc':True,
        'condition_match_not_defined':True,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--D',type=int,default=96);ap.add_argument('--heads',type=int,default=8)
    ap.add_argument('--features',type=int,default=24);ap.add_argument('--ff',type=int,default=192);ap.add_argument('--R',type=int,default=7)
    ap.add_argument('--scaffold-contexts',type=int,default=2000);ap.add_argument('--scaffold-paths',type=int,default=2500);ap.add_argument('--scaffold-lr',type=float,default=.6)
    ap.add_argument('--events',type=int,default=30000)
    ap.add_argument('--eval-images',type=int,default=128);ap.add_argument('--gen-samples',type=int,default=36)
    ap.add_argument('--internal-lr-scale',type=float,default=.5);ap.add_argument('--encode-batch',type=int,default=256)
    ap.add_argument('--decode-iterations',type=int,default=8);ap.add_argument('--choice-temperature',type=float,default=4.5)
    ap.add_argument('--seed',type=int,default=7311);ap.add_argument('--report-every',type=int,default=5000);ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True,exist_ok=True);device=torch.device('cpu')

    controller_audit=controller_self_test()
    training_mask_audit=masking_self_test(n_tokens=49,vocab_size=64,trials=128,seed=a.seed+17)
    print('MASKGIT_CONTROLLER_AUDIT',json.dumps(controller_audit),flush=True)
    print('MASKGIT_TRAINING_MASK_AUDIT',json.dumps(training_mask_audit),flush=True)

    # Labels are used only to select the historical 0/1/2 subset and to build an
    # external post-hoc evaluator.  They are not passed to the neural path below.
    xall,yall,xte0,yte0=core.load(a.data)
    x012,y012=core.subset012(xall,yall);xte,yte=core.subset012(xte0,yte0)
    xtr,ytr,xva,yva=core.split(x012,y012,.1,7)

    tokenizer,tok_meta=base.load_frozen_tokenizer(a.tokenizer,device)
    tenc=time.time();tr,sp_tr=base.encode_snn(tokenizer,xtr,device,a.encode_batch);va,sp_va=base.encode_snn(tokenizer,xva,device,a.encode_batch);te,sp_te=base.encode_snn(tokenizer,xte,device,a.encode_batch)
    print('TOKENIZATION_DONE',json.dumps({'elapsed_s':time.time()-tenc,'train':len(tr),'val':len(va),'test':len(te),'train_activity':sp_tr}),flush=True)
    tstats=base.token_stats(tr)

    cfg=core.Config(D=a.D,H=a.heads,M=a.features,FF=a.ff,R=a.R)
    model=core.init_model(cfg,a.seed)
    for k in ('lrq','lrk','lrv','lro','lr1','lr2'):
        model[k]*=a.internal_lr_scale

    # Item-2 correction: no fixed-random tok/pos/class tables.
    tok,pos,input_representation_audit=build_input_embeddings(tokenizer,cfg.D,grid=7)
    representation_audit=representation_self_test(tokenizer,cfg.D,grid=7)
    print('INPUT_REPRESENTATION_AUDIT',json.dumps(representation_audit),flush=True)

    # Startup WR scaffold is intentionally retained for now; its necessity is the
    # next planned causal audit.  Its contexts are already label-free MaskGIT masks.
    ones=np.ones(cfg.V,np.float64)
    F,T,scaffold_mask_audit=build_maskgit_scaffold_contexts_unconditional(
        model,cfg,tr,tok,pos,a.scaffold_contexts,a.seed+101)
    ts=time.time();model['WR']=core.train_readout_msdl(F,T,ones,a.scaffold_paths,a.scaffold_lr,a.seed+202)
    print('SCAFFOLD_DONE',json.dumps({'elapsed_s':time.time()-ts,'contexts':a.scaffold_contexts,'paths':a.scaffold_paths,'masking':scaffold_mask_audit}),flush=True)

    scaffold_eval={str(r):eval_masked_unconditional(model,cfg,va,tok,pos,r,a.seed+300+int(100*r),a.eval_images) for r in (.25,.5,.9,1.0)}
    print('SCAFFOLD_EVAL',json.dumps(scaffold_eval),flush=True)

    # External classifier is diagnostic only. It never supplies y to generation.
    posthoc_clf=LogisticRegression(max_iter=250,solver='lbfgs',random_state=7).fit(xtr.reshape(len(xtr),-1),ytr)
    stok,strace,saudit=generation_tokens_maskgit(model,cfg,a.gen_samples,tok,pos,a.seed+606,a.decode_iterations,a.choice_temperature)
    simg=base.decode_snn(tokenizer,stok,device);save_grid_unlabeled(simg,a.out/'generation_scaffold.png');gen_scaffold=posthoc_generation_metric(posthoc_clf,simg)

    sysm=MaskGITP3System(cfg,model,a.seed+303)
    hist,online_training_audit=train_maskgit_msdl_unconditional(
        sysm,cfg,tr,tok,pos,ones,a.events,a.seed+404,a.report_every)
    final=model.copy();final.update(sysm.weights())
    final_eval={str(r):eval_masked_unconditional(final,cfg,te,tok,pos,r,a.seed+500+int(100*r),a.eval_images) for r in (.25,.5,.9,1.0)}
    gtok,gtrace,gaudit=generation_tokens_maskgit(final,cfg,a.gen_samples,tok,pos,a.seed+606,a.decode_iterations,a.choice_temperature)
    gimg=base.decode_snn(tokenizer,gtok,device);save_grid_unlabeled(gimg,a.out/'generation_final.png');gen_final=posthoc_generation_metric(posthoc_clf,gimg)

    moves={k:float(np.linalg.norm(final[k]-model[k])/(np.linalg.norm(model[k])+1e-12)) for k in ('WQ','WK','WV','WO','W1','W2')}
    rec={'BR':float(np.mean(np.abs(sysm.BR))),'BO':float(np.mean(np.abs(sysm.BO))),'B2':float(np.mean(np.abs(sysm.B2)))}
    sc50=scaffold_eval['0.5'];fn50=final_eval['0.5'];all_decode_audits=saudit+gaudit
    gates={
        'input_representation_self_test':bool(representation_audit['pass']),
        'labels_enter_neural_input':False,
        'fixed_random_embedding_present':False,
        'maskgit_training_mask_self_test':bool(training_mask_audit['pass']),
        'maskgit_online_training_protocol_audit':bool(online_training_audit['pass']),
        'maskgit_controller_self_test':bool(controller_audit['pass']),
        'all_generation_traces_pass_maskgit_audit':bool(all(x['pass'] for x in all_decode_audits)),
        'plain_no_class_balance':True,
        'frozen_snn_tokenizer':True,
        'candidate_backprop_in_R6B3':False,
        'ce50_improves_with_plain_msdl':bool(fn50['ce']<sc50['ce']),
        'reciprocal_populations_alive':bool(all(v>0 for v in rec.values())),
    }
    diag={
        'stage':'R6B.3 label-free MaskGIT-aligned training + scheduled-parallel generation',
        'question':'Can frozen SNN-VQ + exact P3/MSDL learn/generate from image structure without class conditioning?',
        'no_kmeans_rerun':True,
        'labels':{
            'enter_neural_input':False,
            'enter_msdl_learning':False,
            'enter_generation':False,
            'used_to_select_012_evaluation_subset':True,
            'used_by_external_posthoc_classifier_only':True,
        },
        'input_representation':input_representation_audit,
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
        'online_variant':'plain','plain_startup_scaffold':True,'class_conditioning_retained':False,
        'internal_forward_lr_scale':a.internal_lr_scale,'requested_target_events':a.events,
        'scaffold_eval':scaffold_eval,'final_eval':final_eval,
        'generation_scaffold_posthoc':gen_scaffold,'generation_final_posthoc':gen_final,
        'slow_updates':sysm.update_count,'mean_abs_reciprocal':rec,'relative_internal_weight_move':moves,
        'gates':gates,'history':hist,
    }
    (a.out/'metrics.json').write_text(json.dumps(diag,indent=2))
    (a.out/'maskgit_trace_scaffold.json').write_text(json.dumps(strace,indent=2))
    (a.out/'maskgit_trace_final.json').write_text(json.dumps(gtrace,indent=2))
    (a.out/'maskgit_audit_scaffold.json').write_text(json.dumps(saudit,indent=2))
    (a.out/'maskgit_audit_final.json').write_text(json.dumps(gaudit,indent=2))
    (a.out/'maskgit_training_audit.json').write_text(json.dumps({'self_test':training_mask_audit,'scaffold':scaffold_mask_audit,'online':online_training_audit,'input_representation':representation_audit},indent=2))
    np.savez_compressed(a.out/'final_state.npz',**{k:v for k,v in final.items() if isinstance(v,np.ndarray)},tok=tok,pos=pos)
    np.savetxt(a.out/'generated_tokens_final.csv',gtok,fmt='%d',delimiter=',');np.savetxt(a.out/'generated_tokens_scaffold.csv',stok,fmt='%d',delimiter=',')
    if hist:
        with (a.out/'history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(hist[0].keys()));w.writeheader();w.writerows(hist)
    print('FINAL',json.dumps(diag),flush=True)


if __name__=='__main__':
    main()
