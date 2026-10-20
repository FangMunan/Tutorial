#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D3E-A0-H — diagnose the sole terminal-state hash mismatch without changing D3E.

This imports the authoritative A0 implementation, repeats its exact reconstruction,
computes several deterministic ndarray hash schemas, and deliberately continues to
the reference-trace and causal Gate-A checks even if the frozen full-state hash
constant does not match.  It is diagnostic only and never runs D3E intervention arms.
"""
from __future__ import annotations

import argparse, copy, hashlib, json, random
from pathlib import Path
import numpy as np
import torch

import gate_d_d3e_a0 as a0
import p3i_cb_exact_msdl as core
import r6b2_snn_encoder_p3_msdl as base
import r6b3_context_epoch_training as trainer
from maskgit_input_representation import build_input_embeddings


def h_arrays(obj, *, sep=False, shape_first=True, include_only=None):
    h=hashlib.sha256()
    keys=[]
    for k in sorted(vars(obj)):
        v=getattr(obj,k)
        if not isinstance(v,np.ndarray):
            continue
        if include_only is not None and k not in include_only:
            continue
        keys.append(k)
        x=np.ascontiguousarray(v)
        h.update(k.encode())
        if sep: h.update(b'\0')
        if shape_first:
            h.update(str(x.shape).encode()); h.update(str(x.dtype).encode())
        else:
            h.update(str(x.dtype).encode()); h.update(str(x.shape).encode())
        h.update(x.tobytes())
    return h.hexdigest(),keys


def h_bytes_only(obj, include_only=None):
    h=hashlib.sha256(); keys=[]
    for k in sorted(vars(obj)):
        v=getattr(obj,k)
        if not isinstance(v,np.ndarray): continue
        if include_only is not None and k not in include_only: continue
        keys.append(k); h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest(),keys


def per_array(obj):
    out={}
    for k in sorted(vars(obj)):
        v=getattr(obj,k)
        if isinstance(v,np.ndarray):
            x=np.ascontiguousarray(v)
            out[k]={"shape":list(x.shape),"dtype":str(x.dtype),"sha256":hashlib.sha256(x.tobytes()).hexdigest(),
                    "norm":float(np.linalg.norm(np.asarray(x,dtype=np.float64))) if x.size else 0.0}
    return out


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
    offset=int(schedule['offset']); indices=np.asarray(schedule['permutation'],dtype=np.int64)[offset:offset+1024]
    rng_state=copy.deepcopy(train_rng.bit_generator.state)
    xall,yall,_xte,_yte=core.load(z.data); x012,y012=core.subset012(xall,yall)
    xtr,_ytr,xva,_yva=core.split(x012,y012,.1,7)
    tokenizer,_meta=base.load_frozen_tokenizer(z.tokenizer,device)
    tr,_=base.encode_snn(tokenizer,xtr,device,256); va,_=base.encode_snn(tokenizer,xva,device,256)
    tok0,pos0,_=build_input_embeddings(tokenizer,source.c.D,grid=7)

    validation_ids=np.random.default_rng(a0.SELECTION_SEED).choice(len(va),size=a0.N_IMAGES,replace=False).astype(np.int64)
    gt=np.asarray(va[validation_ids],dtype=np.int16)
    shared_uniform=np.random.default_rng(a0.UNIFORM_SEED).random((a0.N_IMAGES,a0.ROUNDS,source.c.N))
    shared_gumbel=np.random.default_rng(a0.GUMBEL_SEED).gumbel(size=(a0.N_IMAGES,a0.ROUNDS,source.c.N))
    prov={
      'validation_ids_sha256':a0.arr_sha(validation_ids),'gt_tokens_sha256':a0.arr_sha(gt),
      'shared_uniform_sha256':a0.arr_sha(shared_uniform),'shared_gumbel_sha256':a0.arr_sha(shared_gumbel)}
    prov_pass=all(prov[k]==a0.EXPECTED[k] for k in prov)

    terminal,stream_sha,_=a0.reconstruct_r6b3(source,tr,indices,rng_state,tok0,pos0)
    variants={}
    for name,sep,shape_first in [
        ('a0_exact',False,True),('nul_shape_dtype',True,True),
        ('name_dtype_shape',False,False),('nul_dtype_shape',True,False)]:
        variants[name],_=h_arrays(terminal,sep=sep,shape_first=shape_first)
    variants['bytes_only'],all_keys=h_bytes_only(terminal)
    predictor_keys=tuple(k for k in a0.MODEL_KEYS if hasattr(terminal,k) and isinstance(getattr(terminal,k),np.ndarray))
    variants['predictor_a0_exact'],predictor_keys_used=h_arrays(terminal,sep=False,shape_first=True,include_only=predictor_keys)
    variants['predictor_bytes_only'],_=h_bytes_only(terminal,include_only=predictor_keys)
    frozen_expected=a0.EXPECTED['r6b3_terminal_ndarray_state_sha256']
    matching=[k for k,v in variants.items() if v==frozen_expected]

    trace_path=z.effectiveness/'result'/'r6b3_frozen_new_params_r3_t8_traces.json'
    if not trace_path.exists(): trace_path=z.effectiveness/'r6b3_frozen_new_params_r3_t8_traces.json'
    schedules=json.loads(trace_path.read_text(encoding='utf-8'))[:a0.N_IMAGES]
    ref=a0.reference_trace_gate(terminal,tok0,pos0,schedules)
    effects,gates=a0.eval_causal(terminal,tok0,pos0,schedules,gt.astype(np.int64),shared_uniform)
    diffs={k:abs(effects[k]-a0.EXPECTED['effects'][k]) for k in a0.EXPECTED['effects']}

    report={
      'stage':'D3E_A0_H_TERMINAL_HASH_DIAGNOSTIC','d3e_arms_run':False,
      'shared_provenance_pass':bool(prov_pass),'shared_provenance':prov,
      'frozen_expected_terminal_hash':frozen_expected,'hash_variants':variants,
      'variant_matching_frozen_expected':matching,'all_ndarray_keys':all_keys,
      'predictor_ndarray_keys':predictor_keys_used,'per_array':per_array(terminal),
      'training_stream_sha256':stream_sha,'reference_trace_gate':ref,
      'causal_replay_gates':gates,'reconstructed_effects':effects,'effect_abs_diff':diffs,
      'reference_and_causal_pass_despite_full_state_hash':bool(
          ref['trace_mask_context_exact'] and ref['reference_probability_confidence_pass'] and
          ref['image_count_after_three_rounds_exact'] and gates['all_pass'] and max(diffs.values())<a0.TOL),
      'interpretation_boundary':'This diagnostic does not waive the frozen A0 full-state hash requirement; it only identifies whether that requirement is inconsistent with predictor/causal provenance.'
    }
    (z.out/'D3E_A0_HASH_DIAGNOSTIC.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('D3E_A0_HASH_DIAGNOSTIC',json.dumps({
      'shared_provenance_pass':prov_pass,'a0_exact_hash':variants['a0_exact'],
      'expected_hash':frozen_expected,'matching_variants':matching,
      'reference_trace_gate':ref,'causal_gates':gates,'effects':effects,
      'effect_abs_diff':diffs,'reference_and_causal_pass':report['reference_and_causal_pass_despite_full_state_hash']
    },indent=2),flush=True)

if __name__=='__main__': main()
