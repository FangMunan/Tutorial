#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Certify and persist the historical R6B3+1024 Gate-A terminal comparator.

This is provenance infrastructure only. It never runs D3E intervention arms.
A candidate terminal object is reconstructed from the exact 2.0-epoch checkpoint,
then certified against frozen shared-input hashes, the frozen ndarray-state hash,
historical r3/t8 reference probabilities, and historical Gate-A causal effects.
The resulting pickle is uploaded only as a candidate historical comparator; the
report states whether its file SHA also matches the previously recorded pickle SHA.
"""
from __future__ import annotations

import argparse, copy, hashlib, json, pickle, random
from pathlib import Path
import numpy as np
import torch

import gate_d_d3e_a0 as a0
import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
import r6b3_context_epoch_training as trainer
from maskgit_input_representation import build_input_embeddings

EXPECTED_FILE_SHA = '2631db2880077b5dea1fef806fc785685f5063c23ab01cbbe4153219e460f565'
OUT_NAME = 'reconstructed_r6b3_plus1024.pkl'


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def resolve_trace(effectiveness: Path) -> Path:
    p = effectiveness/'result'/'r6b3_frozen_new_params_r3_t8_traces.json'
    if not p.exists():
        p = effectiveness/'r6b3_frozen_new_params_r3_t8_traces.json'
    if not p.exists():
        raise FileNotFoundError('historical r3/t8 trace not found')
    return p


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--effectiveness',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--threads',type=int,default=4)
    z=ap.parse_args(); z.out.mkdir(parents=True,exist_ok=True)

    random.seed(7311); np.random.seed(7311); torch.manual_seed(7311); torch.set_num_threads(z.threads)
    device=torch.device('cpu')

    source,meta,train_rng,schedule=trainer.load_full_checkpoint(z.checkpoint)
    if abs(float(meta.get('context_epoch_equivalent',-1.0))-2.0)>1e-12:
        raise RuntimeError('not exact R6B3 2.0-epoch checkpoint')
    offset=int(schedule['offset'])
    indices=np.asarray(schedule['permutation'],dtype=np.int64)[offset:offset+1024]
    if len(indices)!=1024:
        raise RuntimeError('saved continuation does not contain 1024 contexts')
    rng_state=copy.deepcopy(train_rng.bit_generator.state)

    xall,yall,_xte,_yte=core.load(z.data); x012,y012=core.subset012(xall,yall)
    xtr,_ytr,xva,_yva=core.split(x012,y012,.1,7)
    tokenizer,_=base.load_frozen_tokenizer(z.tokenizer,device)
    tr,_=base.encode_snn(tokenizer,xtr,device,256); va,_=base.encode_snn(tokenizer,xva,device,256)
    tok0,pos0,_=build_input_embeddings(tokenizer,source.c.D,grid=7)

    validation_ids=np.random.default_rng(a0.SELECTION_SEED).choice(len(va),size=a0.N_IMAGES,replace=False).astype(np.int64)
    gt=np.asarray(va[validation_ids],dtype=np.int16)
    shared_uniform=np.random.default_rng(a0.UNIFORM_SEED).random((a0.N_IMAGES,a0.ROUNDS,source.c.N))
    shared_gumbel=np.random.default_rng(a0.GUMBEL_SEED).gumbel(size=(a0.N_IMAGES,a0.ROUNDS,source.c.N))
    prov={
      'validation_ids_sha256':a0.arr_sha(validation_ids),
      'gt_tokens_sha256':a0.arr_sha(gt),
      'shared_uniform_sha256':a0.arr_sha(shared_uniform),
      'shared_gumbel_sha256':a0.arr_sha(shared_gumbel),
    }
    p0=all(prov[k]==a0.EXPECTED[k] for k in prov)

    terminal,stream_sha,_=a0.reconstruct_r6b3(source,tr,indices,rng_state,tok0,pos0)
    ndarray_sha=a0.ndarray_state_hash(terminal)
    p1_state=(ndarray_sha==a0.EXPECTED['r6b3_terminal_ndarray_state_sha256'])

    schedules=json.loads(resolve_trace(z.effectiveness).read_text(encoding='utf-8'))[:a0.N_IMAGES]
    ref=a0.reference_trace_gate(terminal,tok0,pos0,schedules)
    effects,gates=a0.eval_causal(terminal,tok0,pos0,schedules,gt.astype(np.int64),shared_uniform)
    diffs={k:abs(effects[k]-a0.EXPECTED['effects'][k]) for k in a0.EXPECTED['effects']}
    p2=bool(ref['trace_mask_context_exact'] and ref['reference_probability_confidence_pass'] and ref['image_count_after_three_rounds_exact'])
    p3=bool(gates['all_pass'] and max(diffs.values())<a0.TOL)

    serializations={}
    chosen=None
    for protocol in sorted(set([4,pickle.HIGHEST_PROTOCOL])):
        b=pickle.dumps(terminal,protocol=protocol)
        s=sha_bytes(b)
        serializations[f'protocol_{protocol}']={'sha256':s,'bytes':len(b)}
        (z.out/f'reconstructed_r6b3_plus1024_protocol{protocol}.pkl').write_bytes(b)
        if s==EXPECTED_FILE_SHA:
            chosen=b
    if chosen is None:
        chosen=pickle.dumps(terminal,protocol=pickle.HIGHEST_PROTOCOL)
    outfile=z.out/OUT_NAME
    outfile.write_bytes(chosen)
    file_sha=sha_bytes(chosen)
    p1_file=(file_sha==EXPECTED_FILE_SHA)

    report={
      'stage':'D3E_HISTORICAL_TERMINAL_CERTIFICATION',
      'd3e_arms_run':False,
      'P0_shared_provenance_exact':bool(p0),
      'shared_provenance':prov,
      'training_stream_sha256':stream_sha,
      'P1_ndarray_state_exact':bool(p1_state),
      'ndarray_state_sha256':ndarray_sha,
      'expected_ndarray_state_sha256':a0.EXPECTED['r6b3_terminal_ndarray_state_sha256'],
      'P1_historical_file_sha_exact':bool(p1_file),
      'historical_pickle_sha256':file_sha,
      'expected_historical_pickle_sha256':EXPECTED_FILE_SHA,
      'serialization_candidates':serializations,
      'P2_reference_predictor_pass':bool(p2),
      'reference_trace_gate':ref,
      'P3_historical_gateA_effects_pass':bool(p3),
      'causal_replay_gates':gates,
      'reconstructed_effects':effects,
      'effect_abs_diff':diffs,
      'certified_exact_historical_pickle':bool(p0 and p1_state and p1_file and p2 and p3),
      'certified_state_equivalent_comparator':bool(p0 and p1_state and p2 and p3),
      'interpretation_boundary':'A file-SHA mismatch does not certify the old pickle byte identity even when ndarray state and functional/causal gates match.'
    }
    (z.out/'D3E_HISTORICAL_TERMINAL_CERTIFICATION.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('D3E_TERMINAL_CERTIFICATION',json.dumps(report,indent=2),flush=True)
    if not report['certified_state_equivalent_comparator']:
        raise RuntimeError('historical terminal state/function certification failed')

if __name__=='__main__':
    main()
