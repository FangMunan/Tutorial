#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D3E — full-trajectory context-boundary credit-state intervention.

Diagnostic only. Four R6B4 aux1.0 arms share the exact same 1024-context stream:
N=native, F_E=fresh E2 only at context boundaries, F_M=fresh M2 only, F_EM=both.
Within-context persistence is untouched. M2 is shared by W2 and b2 plasticity.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, math, os, pickle, random
from pathlib import Path
import numpy as np
import torch
import gate_d_plasticity_audit as gd
import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
import r6b3_context_epoch_training as trainer
import r6b3_decoder_operating_point_audit as coarse
import r6b3_decoder_spatial_refinement_audit as spatial
from maskgit_input_representation import build_input_embeddings, context_from_arr_unconditional
from maskgit_msdl_training import sample_mask_set
from r6b3_snn_encoder_maskgit_msdl import generation_tokens_maskgit
from r6b4_full_forward_plasticity import upgrade_r6b3_system
from gate_d_d15_d16 import RotationAuditSystem

EPS=1e-30; MASK=64; V=64; DOM=61; ROUNDS=3; BASE_GEN_SEED=27311; GATE_A_TEMP=8.0
ARM_NAMES=('N','F_E','F_M','F_EM')
METRIC_NAMES=('ce','acc','ce61','acc61','ce_non61','acc_non61','mean_p61','p61_on_gt61','p61_on_non61','proposal61_fraction','context61_fraction','sampled_accuracy','sampled_confidence','brier')
IX={k:i for i,k in enumerate(METRIC_NAMES)}
GATE_A_VALIDATION_IDS=np.asarray([923,1574,287,1306,1597,1245,1268,1734,1464,893,945,513,530,1621,1481,824,405,1618,1794,193,635,247,413,1731,1125,1004,1301,421,312,1049,993,605,8,1085,1673,1389,847,1034,412,389,778,1641,1577,597,1071,664,960,1703,1391,404,283,864,1626,1211,306,1170,1538,1105,1243,319,1523,620,242,553,1436,1500,1582,32,329,1431,1340,494,395,1835,892,1683,212,1797,538,1813,61,1765,1624,751,1711,1144,83,429,1589,276,1146,1482,188,1263,629,1107,1805,100,852,1336,1178,593,1057,531,181,658,1264,718,1422,1409,1539,222,1060,1018,1527,987,932,882,1228,1504,1112,600,1594,1824,345,1442,1851,248,196,1052,1392,982,264,1451,301,631,1506,1274,343,103,691,1269,465,738,1475,1242,1283,661,1333,388,328,628,1118,1832,586,799,487,1681,1833,410,1568,781,1061,108,1790,1490,23,1815,1438,1312,1766,1471,1439,390,552,1009,165,1015,1782,1038,1415,361,695,1848,757,1077,1238,372,138,843,1852,392,19,1693,358,956,637,360,1342,1806,472,215,1579,604,1473,1732,1028,708,460,206,1042,1777,1426,808,1429,1587,1192,1491,1688,1610,373,1376,1337,411,894,1677,157,833,1609,74,470,1654,1728,1623,1069,1008,636,1465,630,1039,618,673,986,441,1516,1781,1743,172,815,416,1191,449,1098,1655,86,1555],dtype=np.int64)
EXPECTED_R6B3_GATE_A={'F61':0.021631765465585838,'D_non61':0.12395075840895527,'C61':-0.0026562499999999963,'CE':0.17817003568548723,'delta_P61':-0.0010512298084766059}
GATE_A_PARITY_TOL=1e-7

def jstate(rng): return json.dumps(gd.json_state(rng.bit_generator.state),sort_keys=True,separators=(',',':'))
def bootstrap_mean(x,seed,nboot=10000):
    a=np.asarray(x,float);a=a[np.isfinite(a)]
    if not len(a):return {'n':0,'mean':None,'ci95':[None,None]}
    rng=np.random.default_rng(seed);v=np.empty(nboot)
    for b in range(nboot):v[b]=a[rng.integers(0,len(a),size=len(a))].mean()
    q=np.quantile(v,[.025,.975]);return {'n':int(len(a)),'mean':float(a.mean()),'ci95':[float(q[0]),float(q[1])],'fraction_positive':float(np.mean(a>0)),'fraction_negative':float(np.mean(a<0))}
def model_dict(s):return trainer.model_dict_from_system(s)
def inverse_cdf_rows(p,u):
    c=np.cumsum(np.asarray(p,float),axis=1);c[:,-1]=1.;return np.minimum(np.sum(c<np.asarray(u,float)[:,None],axis=1),p.shape[1]-1).astype(np.int64)
def predict_round(pred,cur,tok,pos,masked):
    xn,xraw=context_from_arr_unconditional(np.asarray(cur,int),tok,pos);pred.rebuild(xn);p=np.empty((len(masked),V),float)
    for qi,t0 in enumerate(np.asarray(masked,int)):
        t=int(t0);a,q,num,den=pred._read(t);u,h,rf,pp=pred._tail(a,np.asarray(xraw[t],float));p[qi]=pp
    p/=p.sum(1,keepdims=True);return p
def one_metrics(p,gt,sampled,cur):
    p=np.asarray(p,float);gt=np.asarray(gt,int);sampled=np.asarray(sampled,int);truep=p[np.arange(len(gt)),gt];ce=-np.log(np.clip(truep,1e-12,1));arg=np.argmax(p,1);acc=arg==gt;is61=gt==DOM;non=~is61;sconf=p[np.arange(len(sampled)),sampled];scorr=sampled==gt;one=np.zeros_like(p);one[np.arange(len(gt)),gt]=1.;vis=np.asarray(cur)!=MASK
    def m(x):return float(np.mean(x)) if np.size(x) else np.nan
    return np.asarray([m(ce),m(acc),m(ce[is61]),m(acc[is61]),m(ce[non]),m(acc[non]),m(p[:,DOM]),m(p[is61,DOM]),m(p[non,DOM]),m(sampled==DOM),m(np.asarray(cur)[vis]==DOM),m(scorr),m(sconf),m(np.sum((p-one)**2,1))],float)
def effect_arrays(sm,tm,r=2):
    s=np.asarray(sm)[:,r];t=np.asarray(tm)[:,r]
    return {'CE':s[:,IX['ce']]-t[:,IX['ce']],'D_non61':s[:,IX['ce_non61']]-t[:,IX['ce_non61']],'F61':s[:,IX['p61_on_non61']]-t[:,IX['p61_on_non61']],'C61':s[:,IX['context61_fraction']]-t[:,IX['context61_fraction']],'delta_P61':s[:,IX['mean_p61']]-t[:,IX['mean_p61']]}
def eval_gate_a(name,cfg,m,tok,pos,schedules,gt,shared_u,trace_sample_check=None):
    n=len(gt);sm=np.full((n,ROUNDS,len(METRIC_NAMES)),np.nan);tm=np.full_like(sm,np.nan);sp=np.full((n,ROUNDS,cfg.N),-1,np.int16);tp=np.full_like(sp,-1)
    gates={'round1_probability_bitwise_parity':True,'round1_sample_exact_parity':True,'committed_position_parity':True,'intervention_purity':True,'trace_sample_reconstruction_exact':True,'image_count_after_three_rounds_exact':True}
    for i in range(n):
        ps=core.P3ImageSystem(cfg,m,BASE_GEN_SEED+100000+i);pt=core.P3ImageSystem(cfg,m,BASE_GEN_SEED+100000+i);cs=np.full(cfg.N,MASK,int);ct=np.full(cfg.N,MASK,int)
        for r in range(ROUNDS):
            tr=schedules[i][r];masked=np.asarray(tr['masked_positions_before'],int);remask=np.asarray(tr['remasked_positions'],int)
            if not np.array_equal(np.flatnonzero(cs==MASK),masked) or not np.array_equal(np.flatnonzero(ct==MASK),masked):gates['committed_position_parity']=False
            if r>0:
                vis=np.flatnonzero(ct!=MASK)
                if len(vis) and not np.array_equal(ct[vis],gt[i,vis]):gates['intervention_purity']=False
            p1=predict_round(ps,cs,tok,pos,masked);p2=predict_round(pt,ct,tok,pos,masked);u=np.asarray(shared_u[i,r,masked],float);z1=inverse_cdf_rows(p1,u);z2=inverse_cdf_rows(p2,u)
            if r==0 and not np.array_equal(p1,p2):gates['round1_probability_bitwise_parity']=False
            if r==0 and not np.array_equal(z1,z2):gates['round1_sample_exact_parity']=False
            if trace_sample_check is not None and not np.array_equal(z1,np.asarray(trace_sample_check[i][r]['sampled_ids'],int)):gates['trace_sample_reconstruction_exact']=False
            sm[i,r]=one_metrics(p1,gt[i,masked],z1,cs);tm[i,r]=one_metrics(p2,gt[i,masked],z2,ct);pr1=cs.copy();pr1[masked]=z1;pr2=ct.copy();pr2[masked]=z2;sp[i,r]=pr1;tp[i,r]=pr2;ns=pr1.copy();ns[remask]=MASK;nt=ct.copy();new=np.setdiff1d(masked,remask,assume_unique=True);nt[new]=gt[i,new];nt[remask]=MASK;old=np.flatnonzero(ct!=MASK);nt[old]=gt[i,old];cs,ct=ns,nt
        if ps.image_count!=ROUNDS or pt.image_count!=ROUNDS:gates['image_count_after_three_rounds_exact']=False
    gates['all_pass']=bool(all(gates.values()));effects=effect_arrays(sm,tm,2);summary={k:bootstrap_mean(v,710000+100*i) for i,(k,v) in enumerate(effects.items())}
    return {'name':name,'self_metrics':sm,'teacher_metrics':tm,'self_proposal':sp,'teacher_proposal':tp,'effects':effects,'summary':summary,'gates':gates}
def regenerate_gate_a_shared_inputs(source,tokens,indices,rng_state,tok0,pos0,gt):
    b=pickle.loads(pickle.dumps(source,pickle.HIGHEST_PROTOCOL));rng=gd.clone_rng(rng_state);ones=np.ones(b.c.V,float);sig=hashlib.sha256()
    for image_index in indices:
        truth=np.asarray(tokens[int(image_index)],int);sm=sample_mask_set(b.c.N,rng);arr=truth.copy();arr[sm.masked_positions]=core.MASK;mirror_seed=int(rng.integers(1,2**31-1));core.seed_numba(mirror_seed);xn,xraw=context_from_arr_unconditional(arr,tok0,pos0);b.learn_mask_batch(xn,xraw,sm.masked_positions,truth,ones,rng);sig.update(np.asarray(sm.masked_positions,np.int64).tobytes());sig.update(np.asarray([mirror_seed],np.int64).tobytes())
    ids,traces,audits=generation_tokens_maskgit(model_dict(b),b.c,len(gt),tok0,pos0,BASE_GEN_SEED,num_iter=ROUNDS,choice_temperature=GATE_A_TEMP)
    if not all(x['pass'] for x in audits):raise RuntimeError('historical R6B3 schedule audit failed')
    shared=np.full((len(gt),ROUNDS,b.c.N),np.nan,float)
    for gi,tr in enumerate(traces):
        rr=np.random.default_rng(BASE_GEN_SEED+200000+gi)
        for r,row in enumerate(tr):
            masked=np.asarray(row['masked_positions_before'],int);shared[gi,r,masked]=rr.random(len(masked));ratio=float(r+1)/ROUNDS;temp=GATE_A_TEMP*(1-ratio)
            if temp!=0.:rr.gumbel(size=len(masked))
    return b,traces,shared,{'context_stream_sha256':sig.hexdigest(),'generated_schedule_audits_pass':True,'n_images':int(len(gt)),'base_generation_seed':BASE_GEN_SEED,'temperature':GATE_A_TEMP}
class D3EArmSystem(RotationAuditSystem):
    def d3e_init(self):self._d3e_prev_g=None;self._d3e_cug1=[];self._d3e_rows=[]
    def audit_finish_context(self):
        n=max(1,len(self._audit_ctx['masked']));g=np.asarray(self._audit_G,float)/n;u=np.asarray(self.W2,float)-self._audit_w_before['W2'];c=gd.cosine(u,self._d3e_prev_g) if self._d3e_prev_g is not None else np.nan;self._d3e_rows.append({'context':int(self._audit_ctx['context']),'cosine':gd.cosine(u,g),'C_ug_1':float(c),'u_norm':float(np.linalg.norm(u)),'g_norm':float(np.linalg.norm(g))});
        if np.isfinite(c):self._d3e_cug1.append(float(c))
        self._d3e_prev_g=np.array(g,copy=True);super().audit_finish_context()
    def d3e_summary(self):
        r=self.rotation_summary();c=np.asarray(self._d3e_cug1,float);r['context_alignment']={'mean':float(np.mean([x['cosine'] for x in self._d3e_rows])),'median':float(np.median([x['cosine'] for x in self._d3e_rows]))};r['C_ug_lag1']={'N':int(len(c)),'mean':float(c.mean()),'median':float(np.median(c)),'mean_95ci':bootstrap_mean(c,720001,5000)['ci95']};return r
def make_arm(start):
    s=pickle.loads(pickle.dumps(start,pickle.HIGHEST_PROTOCOL));s.__class__=D3EArmSystem;s.audit_init(sketch_dim=64);s.rotation_init();s.d3e_init();return s
def reset_boundary(s,arm):
    if arm in ('F_E','F_EM'):s.ve2.fill(core.V0);s.ue2.fill(core.U0);s.ze2.fill(0.)
    if arm in ('F_M','F_EM'):s.vm2.fill(core.V0);s.um2.fill(core.U0);s.zm2.fill(0.)
def train_arm(name,start,tokens,indices,rng_state):
    s=make_arm(start);rng=gd.clone_rng(rng_state);sig=hashlib.sha256();rows=[]
    for j,image_index in enumerate(indices,start=1):
        reset_boundary(s,name);truth=np.asarray(tokens[int(image_index)],int);rec,ev=gd.run_one_context(s,True,arm='T0',j=j,image_index=int(image_index),truth=truth,rng=rng);sig.update(np.asarray(ev['masked'],np.int64).tobytes());sig.update(np.asarray([ev['mirror_seed']],np.int64).tobytes());rows.append({'context':j,'image_index':int(image_index),'mean_nll':float(rec['mean_nll']),'nmask':int(ev['nmask'])})
        if j in (256,512,1024) or j==len(indices):print('D3E_ARM_PROGRESS',json.dumps({'arm':name,'context':j,'alignment':s._d3e_rows[-1]['cosine']}),flush=True)
    return s,{'stream_sha256':sig.hexdigest(),'rng_state_after':jstate(rng),'online_rows':rows,'mechanism':s.d3e_summary()}
def eval_masked(s,tokens,seed,n):return {str(r):trainer.eval_masked_unconditional(model_dict(s),s.c,tokens,s.tok,s.pos,r,seed+int(round(1000*r)),n) for r in (.25,.5,.9,1.0)}
def rollout_js(s,real_p,real_pos_p,real_pair,seed,samples):
    ids,traces,audits=generation_tokens_maskgit(model_dict(s),s.c,samples,s.tok,s.pos,seed,num_iter=3,choice_temperature=8.0);gl=coarse.distribution_metrics(ids,real_p,s.c.V);sp=spatial.positionwise_metrics(ids,real_pos_p,s.c.V,s.c.N);pa=spatial.pair_metrics(ids,real_pair,s.c.V,s.c.N,7);return {'global_js_nats':float(gl['js_to_real_nats']),'spatial_js_nats':float(sp['mean_js_nats']),'pair_js_nats':float(pa['mean_orientation_js_nats']),'all_trace_audits_pass':bool(all(x['pass'] for x in audits))}
def gate_for_system(name,s,schedules,gt,shared):return eval_gate_a(name,s.c,model_dict(s),np.asarray(s.tok),np.asarray(s.pos),schedules,gt,shared)
def transplant(native,fem,which):
    s=pickle.loads(pickle.dumps(native,pickle.HIGHEST_PROTOCOL))
    if which in ('W2','W2_b2'):s.W2=np.array(fem.W2,copy=True)
    if which in ('b2','W2_b2'):s.b2=np.array(fem.b2,copy=True)
    return s
def rescue(native_eval,other_eval,seed0):return {k:bootstrap_mean(np.asarray(native_eval['effects'][k])-np.asarray(other_eval['effects'][k]),seed0+i) for i,k in enumerate(('F61','D_non61','C61','CE','delta_P61'))}
def write_csv(path,rows):
    if not rows:path.write_text('',encoding='utf-8');return
    keys=[];seen=set()
    for r in rows:
        for k in r:
            if k not in seen:seen.add(k);keys.append(k)
    with path.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--tokenizer',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--contexts',type=int,default=1024);ap.add_argument('--eval-images',type=int,default=256);ap.add_argument('--gen-samples',type=int,default=256);ap.add_argument('--encode-batch',type=int,default=256);ap.add_argument('--seed',type=int,default=7311);ap.add_argument('--threads',type=int,default=4);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.contexts!=1024 or a.eval_images!=256:raise ValueError('formal D3E requires 1024 contexts and 256 Gate-A images')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(a.threads);device=torch.device('cpu');ot=gd.oracle_self_test();
    if not ot['pass']:raise RuntimeError(ot)
    source,meta,train_rng,schedule=trainer.load_full_checkpoint(a.checkpoint)
    if abs(float(meta.get('context_epoch_equivalent',-1))-2.)>1e-12:raise ValueError('requires exact R6B3 2.0-epoch checkpoint')
    offset=int(schedule['offset']);indices=np.asarray(schedule['permutation'],np.int64)[offset:offset+a.contexts];rng_state=copy.deepcopy(train_rng.bit_generator.state);xall,yall,_xte,_yte=core.load(a.data);x012,y012=core.subset012(xall,yall);xtr,_ytr,xva,_yva=core.split(x012,y012,.1,7);tokenizer,tok_meta=base.load_frozen_tokenizer(a.tokenizer,device);tr,sp_tr=base.encode_snn(tokenizer,xtr,device,a.encode_batch);va,sp_va=base.encode_snn(tokenizer,xva,device,a.encode_batch);tok0,pos0,input_audit=build_input_embeddings(tokenizer,source.c.D,grid=7);gt=np.asarray(va[GATE_A_VALIDATION_IDS],np.int64);start=pickle.loads(pickle.dumps(source,pickle.HIGHEST_PROTOCOL));upgrade_r6b3_system(start,tok0,pos0,aux_lr_scale=1.0)
    b3,schedules,shared,shared_audit=regenerate_gate_a_shared_inputs(source,tr,indices,rng_state,tok0,pos0,gt);b3gate=eval_gate_a('r6b3_historical_control',b3.c,model_dict(b3),tok0,pos0,schedules,gt,shared,trace_sample_check=schedules);parity={k:float(abs(b3gate['summary'][k]['mean']-EXPECTED_R6B3_GATE_A[k])) for k in EXPECTED_R6B3_GATE_A};control_pass=bool(b3gate['gates']['all_pass'] and max(parity.values())<GATE_A_PARITY_TOL);print('D3E_GATE_A_CONTROL_PARITY',json.dumps({'pass':control_pass,'abs_diff':parity,'gates':b3gate['gates']},indent=2),flush=True)
    if not control_pass:raise RuntimeError('failed frozen Gate-A2 control reproduction')
    systems={};train={}
    for arm in ARM_NAMES:systems[arm],train[arm]=train_arm(arm,start,tr,indices,rng_state)
    streams={arm:train[arm]['stream_sha256'] for arm in ARM_NAMES};rng_after={arm:train[arm]['rng_state_after'] for arm in ARM_NAMES}
    if len(set(streams.values()))!=1 or len(set(rng_after.values()))!=1:raise RuntimeError('arm streams/RNG diverged')
    if abs(float(train['N']['mechanism']['context_alignment']['mean'])-0.7311738063456619)>1e-10:raise RuntimeError('native D3E does not reproduce D3 baseline')
    mech={}
    for arm in ARM_NAMES:
        r=train[arm]['mechanism'];mech[arm]={'T_native':float(r['native_W2']['tortuosity']),'T_oracle_matched':float(r['oracle_matched']['tortuosity']),'T_excess':float(r['excess_tortuosity_native_over_oracle']),'context_alignment_mean':float(r['context_alignment']['mean']),'context_alignment_median':float(r['context_alignment']['median']),'R_perp':float(r['orthogonal_path_fraction']),'C_ug_lag1_mean':float(r['C_ug_lag1']['mean']),'C_ug_lag1_95ci':r['C_ug_lag1']['mean_95ci']}
    gate={};masked={}
    for arm in ARM_NAMES:
        gate[arm]=gate_for_system(arm,systems[arm],schedules,gt,shared);masked[arm]=eval_masked(systems[arm],va,a.seed+31000,a.eval_images)
        if not gate[arm]['gates']['all_pass']:raise RuntimeError(f'Gate-A structural failure {arm}')
    res={arm:rescue(gate['N'],gate[arm],740000+100*i) for i,arm in enumerate(('F_E','F_M','F_EM'),1)}
    meds={'W2_only':transplant(systems['N'],systems['F_EM'],'W2'),'b2_only':transplant(systems['N'],systems['F_EM'],'b2'),'W2_b2':transplant(systems['N'],systems['F_EM'],'W2_b2')};medgate={k:gate_for_system(k,s,schedules,gt,shared) for k,s in meds.items()};medres={k:rescue(gate['N'],v,750000+100*i) for i,(k,v) in enumerate(medgate.items())}
    _rc,real_p=coarse.probability_from_ids(va,source.c.V);_pc,real_pos_p=spatial.position_probabilities(va,source.c.V,source.c.N);real_pair=spatial.pair_reference(va,source.c.V,source.c.N,7);rollout={arm:rollout_js(systems[arm],real_p,real_pos_p,real_pair,BASE_GEN_SEED,a.gen_samples) for arm in ARM_NAMES}
    n=mech['N'];f=mech['F_EM'];mechanism_pass=bool(f['T_excess']<n['T_excess'] and f['R_perp']<n['R_perp'] and f['C_ug_lag1_mean']<n['C_ug_lag1_mean']);ph=res['F_EM'];phenotype_pass=bool(ph['F61']['ci95'][0]>0 or ph['D_non61']['ci95'][0]>0);case='E1_MECHANISM_AND_PHENOTYPE_RESCUE' if mechanism_pass and phenotype_pass else ('E2_TRAJECTORY_CLEANER_NO_PRIMARY_PHENOTYPE_RESCUE' if mechanism_pass else 'E3_NO_SYSTEM_LEVEL_MECHANISM_RESCUE')
    for arm in ARM_NAMES:write_csv(a.out/f'{arm}_CONTEXT_ROWS.csv',systems[arm]._d3e_rows)
    npz={'validation_ids':GATE_A_VALIDATION_IDS,'gt_tokens':gt,'shared_uniform':shared}
    for name,v in {**gate,**{f'med_{k}':v for k,v in medgate.items()},'r6b3_control':b3gate}.items():npz[f'{name}__self_metrics']=v['self_metrics'];npz[f'{name}__teacher_metrics']=v['teacher_metrics'];npz[f'{name}__self_proposal']=v['self_proposal'];npz[f'{name}__teacher_proposal']=v['teacher_proposal']
    np.savez_compressed(a.out/'D3E_EFFECT_ARRAYS.npz',**npz)
    summary={'stage':'GATE_D3E_FULL_TRAJECTORY_CONTEXT_BOUNDARY_CREDIT_STATE_INTERVENTION','classification':case,'mechanism_pass':mechanism_pass,'phenotype_pass':phenotype_pass,'mechanism':mech,'gateA_round3':{arm:gate[arm]['summary'] for arm in ARM_NAMES},'paired_rescue_native_minus_arm':res,'terminal_mediation_rescue_vs_native':medres,'masked_eval':masked,'rollout_js_r3_t8':rollout,'gateA_control_parity':{'pass':control_pass,'abs_diff_vs_frozen_r6b3_comparator':parity,'expected':EXPECTED_R6B3_GATE_A,'reconstructed':b3gate['summary'],'shared_input_audit':shared_audit},'stream_pairing':{'hashes':streams,'all_identical':len(set(streams.values()))==1,'rng_after_identical':len(set(rng_after.values()))==1},'intervention_definition':{'N':'native','F_E':'canonical fresh E2 state (V0,U0,0) at each context boundary; native within-context persistence','F_M':'canonical fresh M2 state (V0,U0,0) at each context boundary; native within-context persistence','F_EM':'canonical fresh E2+M2 at each context boundary; native within-context persistence','M2_shared_with_b2':True},'scientific_boundary':'D3E is a full-trajectory causal diagnostic. F_M/F_EM alter shared M2 used by W2 and b2. Terminal W2/b2 transplants are direct terminal contribution tests, not complete mediation. Positive D3E does not itself establish an endogenous metaplastic mechanism.'}
    (a.out/'D3E_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');manifest={'contexts':a.contexts,'gateA_images':a.eval_images,'gen_samples':a.gen_samples,'start_context_epoch':float(meta['context_epoch_equivalent']),'github_sha':os.environ.get('GITHUB_SHA'),'native_training_rule_changed':False,'candidate_metaplastic_rule_introduced':False,'counterfactual_boundary_resets_are_diagnostic_only':True,'gateA_control_parity_pass':control_pass,'paired_training_stream_pass':True,'classification':case,'tokenizer_metadata':tok_meta,'input_audit':input_audit,'train_activity':sp_tr,'validation_activity':sp_va};(a.out/'D3E_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for arm,s in systems.items():
        with (a.out/f'D3E_{arm}_terminal.pkl').open('wb') as fh:pickle.dump({'system':s},fh,pickle.HIGHEST_PROTOCOL)
    print('D3E_COMPLETE',json.dumps({'classification':case,'mechanism':mech,'F_EM_rescue':res['F_EM'],'mediation':medres,'rollout':rollout,'gateA_control_parity':control_pass},indent=2),flush=True)
if __name__=='__main__':main()
