#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D3E v2 orchestration with separated P0-P6 provenance/execution gates."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch

import gate_d_d3e_a0 as a0
import gate_d_plasticity_audit as gd
import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
import r6b3_context_epoch_training as trainer
import r6b3_decoder_operating_point_audit as coarse
import r6b3_decoder_spatial_refinement_audit as spatial
from maskgit_input_representation import build_input_embeddings
from r6b4_full_forward_plasticity import upgrade_r6b3_system

from gate_d_d3e_eval import (
    BASE_GEN_SEED, eval_gate_a, eval_masked, gate_for_system, model_dict,
    rescue, rollout_js, transplant, write_csv,
)
from gate_d_d3e_runtime import (
    P6_TOL, json_sha, locate, p6_compare, run_native_p4_pair,
    schedule_projection, sha_file, train_intervention_arm,
)

ARM_NAMES=('N','F_E','F_M','F_EM')
EXPECTED_CHECKPOINT_SHA256='4a42e60ada31f6ec5167cafd2bd860dd8ca937f7970665c5b95bf8fb2dfc89d0'
GATE_A_PARITY_TOL=1e-7
HISTORICAL_ALIGNMENT_REFERENCE=0.7311738063456619


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--anchor',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--contexts',type=int,default=1024)
    ap.add_argument('--eval-images',type=int,default=256)
    ap.add_argument('--gen-samples',type=int,default=256)
    ap.add_argument('--encode-batch',type=int,default=256)
    ap.add_argument('--seed',type=int,default=7311)
    ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    if a.contexts!=1024 or a.eval_images!=256:
        raise ValueError('formal D3E requires 1024 contexts and 256 Gate-A images')

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads)
    device=torch.device('cpu')
    ot=gd.oracle_self_test()
    if not ot['pass']: raise RuntimeError(ot)

    # P1 — exact D3E start checkpoint identity.
    checkpoint_sha=sha_file(a.checkpoint)
    source,meta,train_rng,schedule=trainer.load_full_checkpoint(a.checkpoint)
    p1=bool(checkpoint_sha==EXPECTED_CHECKPOINT_SHA256
            and abs(float(meta.get('context_epoch_equivalent',-1))-2.)<=1e-12)
    if not p1:
        raise RuntimeError({'P1_exact_2epoch_checkpoint_identity':False,
                            'checkpoint_sha256':checkpoint_sha,'meta':meta})
    offset=int(schedule['offset'])
    indices=np.asarray(schedule['permutation'],np.int64)[offset:offset+a.contexts]
    if len(indices)!=1024: raise RuntimeError('saved epoch-3 schedule does not provide 1024 contexts')
    rng_state=copy.deepcopy(train_rng.bit_generator.state)

    xall,yall,_xte,_yte=core.load(a.data)
    x012,y012=core.subset012(xall,yall)
    xtr,_ytr,xva,_yva=core.split(x012,y012,.1,7)
    tokenizer,tok_meta=base.load_frozen_tokenizer(a.tokenizer,device)
    tr,sp_tr=base.encode_snn(tokenizer,xtr,device,a.encode_batch)
    va,sp_va=base.encode_snn(tokenizer,xva,device,a.encode_batch)
    tok0,pos0,input_audit=build_input_embeddings(tokenizer,source.c.D,grid=7)

    anchor_json_path=locate(a.anchor,'D3E_R6B4_NATIVE_EXECUTION_ANCHOR.json')
    anchor_npz_path=locate(a.anchor,'R6B4_NATIVE_GATE_A_EXECUTION_ANCHOR.npz')
    trace_path=locate(a.anchor,'HISTORICAL_R6B3_R3_T8_REFERENCE_TRACE.json')
    anchor_report=json.loads(anchor_json_path.read_text(encoding='utf-8'))
    if not anchor_report.get('all_freeze_gates_pass',False):
        raise RuntimeError('frozen execution anchor was not certified')
    anchor_npz=np.load(anchor_npz_path,allow_pickle=False)
    schedules=json.loads(trace_path.read_text(encoding='utf-8'))

    # P0 — canonical shared provenance exact, including frozen schedule identity.
    validation_ids=np.random.default_rng(a0.SELECTION_SEED).choice(
        len(va),size=a0.N_IMAGES,replace=False).astype(np.int64)
    gt16=np.asarray(va[validation_ids],dtype=np.int16)
    gt=gt16.astype(np.int64)
    shared=np.random.default_rng(a0.UNIFORM_SEED).random((a0.N_IMAGES,a0.ROUNDS,source.c.N))
    shared_gumbel=np.random.default_rng(a0.GUMBEL_SEED).gumbel(size=(a0.N_IMAGES,a0.ROUNDS,source.c.N))
    p0_hashes={
        'validation_ids_sha256':a0.arr_sha(validation_ids),
        'gt_tokens_sha256':a0.arr_sha(gt16),
        'shared_uniform_sha256':a0.arr_sha(shared),
        'shared_gumbel_sha256':a0.arr_sha(shared_gumbel),
    }
    p0=bool(all(p0_hashes[k]==a0.EXPECTED[k] for k in p0_hashes)
            and np.array_equal(validation_ids,anchor_npz['validation_ids'])
            and np.array_equal(gt16,anchor_npz['gt_tokens'])
            and np.array_equal(shared,anchor_npz['shared_uniform'])
            and np.array_equal(shared_gumbel,anchor_npz['shared_gumbel'])
            and json_sha(schedule_projection(schedules))==anchor_report['schedule_sha256'])
    if not p0: raise RuntimeError({'P0_shared_provenance_hashes_exact':False,'hashes':p0_hashes})

    # P2/P3 — historical R6B3 evaluator/provenance validation.
    b3,b3_stream_hash,_=a0.reconstruct_r6b3(source,tr,indices,rng_state,tok0,pos0)
    p2_detail=a0.reference_trace_gate(b3,tok0,pos0,schedules)
    p2=bool(p2_detail['trace_mask_context_exact']
            and p2_detail['reference_probability_confidence_pass']
            and p2_detail['image_count_after_three_rounds_exact'])
    b3gate=eval_gate_a('r6b3_historical_control',b3.c,model_dict(b3),tok0,pos0,schedules,gt,shared)
    p3_diff={k:float(abs(b3gate['summary'][k]['mean']-a0.EXPECTED['effects'][k]))
             for k in a0.EXPECTED['effects']}
    p3=bool(b3gate['gates']['all_pass'] and max(p3_diff.values())<GATE_A_PARITY_TOL)
    if not (p2 and p3):
        raise RuntimeError({'P2':p2,'P2_detail':p2_detail,'P3':p3,'P3_diff':p3_diff})

    # Exact R6B4 aux1.0 D3E start. Causal arm definitions are unchanged.
    start=pickle.loads(pickle.dumps(source,pickle.HIGHEST_PROTOCOL))
    upgrade_r6b3_system(start,tok0,pos0,aux_lr_scale=1.0)

    # P4 — within-run native plain/instrumented bitwise parity at 256/512/1024.
    systems={}; train={}
    systems['N'],train['N'],p4_report=run_native_p4_pair(start,tr,indices,rng_state)
    p4=bool(p4_report['pass'])
    (a.out/'P4_NATIVE_INSTRUMENTATION_PARITY.json').write_text(json.dumps(p4_report,indent=2),encoding='utf-8')
    if not p4: raise RuntimeError('P4 failed: instrumentation changed native R6B4 trajectory')

    # P5 — exact four-arm image/mask/target-order/mirror/RNG pairing.
    for arm in ('F_E','F_M','F_EM'):
        systems[arm],train[arm]=train_intervention_arm(arm,start,tr,indices,rng_state)
    ref_rows=train['N']['stream_rows']; ref_rng=train['N']['rng_state_after']
    p5_arm={}
    for arm in ARM_NAMES:
        p5_arm[arm]={
            'stream_rows_exact':bool(train[arm]['stream_rows']==ref_rows),
            'stream_sha256':train[arm]['stream_sha256'],
            'rng_after_exact':bool(train[arm]['rng_state_after']==ref_rng),
            'reset_rng_unchanged':bool(train[arm].get('reset_rng_unchanged',True)),
        }
        p5_arm[arm]['pass']=bool(p5_arm[arm]['stream_rows_exact']
                                 and p5_arm[arm]['rng_after_exact']
                                 and p5_arm[arm]['reset_rng_unchanged'])
    p5=bool(all(v['pass'] for v in p5_arm.values()))
    (a.out/'P5_FOUR_ARM_STREAM_PAIRING.json').write_text(
        json.dumps({'pass':p5,'arms':p5_arm},indent=2),encoding='utf-8')
    if not p5: raise RuntimeError('P5 failed: four-arm stream/RNG pairing diverged')

    # P6 — current instrumented N vs frozen D3E R6B4-native execution anchor.
    gate={'N':gate_for_system('N',systems['N'],schedules,gt,shared)}
    if not gate['N']['gates']['all_pass']: raise RuntimeError('Gate-A structural failure N')
    p6,p6_detail=p6_compare(gate['N'],anchor_npz,anchor_report,schedules)
    (a.out/'P6_R6B4_NATIVE_EXECUTION_ANCHOR_PARITY.json').write_text(
        json.dumps({'pass':p6,'tolerance':P6_TOL,'detail':p6_detail},indent=2),encoding='utf-8')
    if not p6: raise RuntimeError('P6 failed: current R6B4 N does not match frozen execution anchor')

    provenance_gates={
        'P0_shared_provenance_hashes_exact':p0,
        'P1_exact_2epoch_checkpoint_identity':p1,
        'P2_historical_R6B3_reference_predictor_parity':p2,
        'P3_historical_R6B3_GateA_causal_parity':p3,
        'P4_R6B4_N_plain_instrumented_same_run_bitwise':p4,
        'P5_four_R6B4_arms_stream_RNG_exact_pairing':p5,
        'P6_R6B4_N_frozen_execution_anchor_functional_parity':p6,
    }
    all_provenance_pass=bool(all(provenance_gates.values()))
    if not all_provenance_pass: raise RuntimeError('P0-P6 did not all pass')

    # Only after P0...P6 PASS may intervention arms be interpreted.
    mech={}
    for arm in ARM_NAMES:
        r=train[arm]['mechanism']
        mech[arm]={
            'T_native':float(r['native_W2']['tortuosity']),
            'T_oracle_matched':float(r['oracle_matched']['tortuosity']),
            'T_excess':float(r['excess_tortuosity_native_over_oracle']),
            'context_alignment_mean':float(r['context_alignment']['mean']),
            'context_alignment_median':float(r['context_alignment']['median']),
            'R_perp':float(r['orthogonal_path_fraction']),
            'C_ug_lag1_mean':float(r['C_ug_lag1']['mean']),
            'C_ug_lag1_95ci':r['C_ug_lag1']['mean_95ci'],
        }
    alignment_diagnostic={
        'historical_reference':HISTORICAL_ALIGNMENT_REFERENCE,
        'current_N':mech['N']['context_alignment_mean'],
        'current_minus_reference':float(mech['N']['context_alignment_mean']-HISTORICAL_ALIGNMENT_REFERENCE),
        'hard_gate':False,
    }

    masked={'N':eval_masked(systems['N'],va,a.seed+31000,a.eval_images)}
    for arm in ('F_E','F_M','F_EM'):
        gate[arm]=gate_for_system(arm,systems[arm],schedules,gt,shared)
        masked[arm]=eval_masked(systems[arm],va,a.seed+31000,a.eval_images)
        if not gate[arm]['gates']['all_pass']: raise RuntimeError(f'Gate-A structural failure {arm}')

    res={arm:rescue(gate['N'],gate[arm],740000+100*i) for i,arm in enumerate(('F_E','F_M','F_EM'),1)}
    meds={
        'W2_only':transplant(systems['N'],systems['F_EM'],'W2'),
        'b2_only':transplant(systems['N'],systems['F_EM'],'b2'),
        'W2_b2':transplant(systems['N'],systems['F_EM'],'W2_b2'),
    }
    medgate={k:gate_for_system(k,s,schedules,gt,shared) for k,s in meds.items()}
    medres={k:rescue(gate['N'],v,750000+100*i) for i,(k,v) in enumerate(medgate.items())}

    _rc,real_p=coarse.probability_from_ids(va,source.c.V)
    _pc,real_pos_p=spatial.position_probabilities(va,source.c.V,source.c.N)
    real_pair=spatial.pair_reference(va,source.c.V,source.c.N,7)
    rollout={arm:rollout_js(systems[arm],real_p,real_pos_p,real_pair,BASE_GEN_SEED,a.gen_samples) for arm in ARM_NAMES}

    n=mech['N']; f=mech['F_EM']
    mechanism_pass=bool(f['T_excess']<n['T_excess'] and f['R_perp']<n['R_perp']
                        and f['C_ug_lag1_mean']<n['C_ug_lag1_mean'])
    ph=res['F_EM']
    phenotype_pass=bool(ph['F61']['ci95'][0]>0 or ph['D_non61']['ci95'][0]>0)
    case=('E1_MECHANISM_AND_PHENOTYPE_RESCUE' if mechanism_pass and phenotype_pass
          else ('E2_TRAJECTORY_CLEANER_NO_PRIMARY_PHENOTYPE_RESCUE' if mechanism_pass
                else 'E3_NO_SYSTEM_LEVEL_MECHANISM_RESCUE'))

    for arm in ARM_NAMES: write_csv(a.out/f'{arm}_CONTEXT_ROWS.csv',systems[arm]._d3e_rows)
    npz={'validation_ids':validation_ids,'gt_tokens':gt16,'shared_uniform':shared,'shared_gumbel':shared_gumbel}
    for name,v in {**gate,**{f'med_{k}':v for k,v in medgate.items()},'r6b3_control':b3gate}.items():
        npz[f'{name}__self_metrics']=v['self_metrics']; npz[f'{name}__teacher_metrics']=v['teacher_metrics']
        npz[f'{name}__self_proposal']=v['self_proposal']; npz[f'{name}__teacher_proposal']=v['teacher_proposal']
        for ek,ev in v['effects'].items(): npz[f'{name}__effect__{ek}']=ev
    np.savez_compressed(a.out/'D3E_EFFECT_ARRAYS.npz',**npz)

    summary={
        'stage':'GATE_D3E_FULL_TRAJECTORY_CONTEXT_BOUNDARY_CREDIT_STATE_INTERVENTION',
        'classification':case,'mechanism_pass':mechanism_pass,'phenotype_pass':phenotype_pass,
        'provenance_gates':provenance_gates,'all_provenance_gates_pass':all_provenance_pass,
        'P0_shared_provenance':p0_hashes,'P1_checkpoint_sha256':checkpoint_sha,
        'P2_detail':p2_detail,'P3_abs_diff_vs_historical_R6B3':p3_diff,
        'P4_same_run_instrumentation':p4_report,'P5_stream_pairing':p5_arm,
        'P6_execution_anchor_parity':p6_detail,
        'frozen_execution_anchor':{
            'anchor_json_sha256':sha_file(anchor_json_path),'anchor_npz_sha256':sha_file(anchor_npz_path),
            'reported_anchor_npz_sha256':anchor_report.get('anchor_npz_sha256'),
            'role':'frozen D3E R6B4-native execution anchor','not_historical_GateA_R6B4_reference':True},
        'historical_R6B3_reconstruction_stream_sha256':b3_stream_hash,
        'alignment_historical_reference_diagnostic_only':alignment_diagnostic,
        'mechanism':mech,'gateA_round3':{arm:gate[arm]['summary'] for arm in ARM_NAMES},
        'paired_rescue_native_minus_arm':res,'terminal_mediation_rescue_vs_native':medres,
        'masked_eval':masked,'rollout_js_r3_t8':rollout,
        'intervention_definition':{
            'N':'native',
            'F_E':'canonical fresh E2 state (V0,U0,0) at each context boundary; native within-context persistence',
            'F_M':'canonical fresh M2 state (V0,U0,0) at each context boundary; native within-context persistence',
            'F_EM':'canonical fresh E2+M2 at each context boundary; native within-context persistence',
            'M2_shared_with_b2':True},
        'scientific_boundary':(
            'P0-P3 validate historical evaluator/provenance identity; P4-P6 validate current R6B4 D3E learner/execution identity. '
            'The historical byte-identical R6B3 +1024 pickle remains unrecovered and is not an execution dependency. '
            'D3E is a full-trajectory causal diagnostic. F_M/F_EM alter shared M2 used by W2 and b2. Terminal W2/b2 transplants '
            'are direct terminal contribution tests, not complete mediation. Positive D3E does not itself establish an endogenous metaplastic mechanism.'),
    }
    (a.out/'D3E_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    manifest={
        'contexts':a.contexts,'gateA_images':a.eval_images,'gen_samples':a.gen_samples,
        'start_context_epoch':float(meta['context_epoch_equivalent']),'github_sha':os.environ.get('GITHUB_SHA'),
        'provenance_gates':provenance_gates,'all_provenance_gates_pass':all_provenance_pass,
        'native_training_rule_changed':False,'candidate_metaplastic_rule_introduced':False,
        'counterfactual_boundary_resets_are_diagnostic_only':True,'four_causal_arm_definitions_changed':False,
        'classification':case,'tokenizer_metadata':tok_meta,'input_audit':input_audit,
        'train_activity':sp_tr,'validation_activity':sp_va}
    (a.out/'D3E_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for arm,s in systems.items():
        with (a.out/f'D3E_{arm}_terminal.pkl').open('wb') as fh: pickle.dump({'system':s},fh,pickle.HIGHEST_PROTOCOL)
    print('D3E_COMPLETE',json.dumps({
        'P0_P6_PASS':all_provenance_pass,'classification':case,'mechanism':mech,
        'F_EM_rescue':res['F_EM'],'mediation':medres,'rollout':rollout,
        'alignment_diagnostic':alignment_diagnostic},indent=2),flush=True)
