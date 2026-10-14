#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import torch

import run_u3br_selective_state_repair as r
import run_u3br2_event_state as r2
from core_v2 import LinearAttentionCoreV2


class V2RepairModel(r2.EventRepairModel):
    def __init__(self,d,heads,features,ff,value_dim,seed):
        super().__init__(d,heads,features,ff,value_dim,seed)
        self.attn=LinearAttentionCoreV2(d,heads,features,seed)


def run(args):
    r.seed_all(args.seed);torch.set_num_threads(args.threads);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    task=r2.EventResetTask(args.tokens,args.d,args.value_dim,args.nkeys,args.control_amp)
    base=V2RepairModel(args.d,args.heads,args.features,args.ff,args.value_dim,args.seed+101)
    base_hist=r.train_base(base,task,args)
    base_eval=r.evaluate(base,task,args,'identity',args.seed+2500)
    oracle=r.evaluate(base,task,args,'oracle',args.seed+3000,oracle_reset=0.0)
    identity=r.evaluate(base,task,args,'identity',args.seed+3000)
    base_valid=base_eval['pre_sign_accuracy']>=args.base_gate
    oracle_valid=oracle['pre_sign_accuracy']>=args.task_gate and oracle['post_sign_accuracy']>=args.task_gate
    oracle_gap=oracle['post_sign_accuracy']-identity['post_sign_accuracy']
    result={'stage':'UNIFIED_U3BR3_CORE_V2','config':vars(args),'base_train':base_hist,
            'base_identity_eval':base_eval,'oracle_reset_eval':oracle,'identity_eval':identity}
    if not (base_valid and oracle_valid and oracle_gap>=args.oracle_gap):
        gates={'base_persistent_memory_pretrained':base_valid,'oracle_reset_task_solvable':oracle_valid,
               'identity_reset_control_worse':oracle_gap>=args.oracle_gap}
        result.update({'gates':gates,'PASS':False,
            'decision':'corrected core v2 still does not yield a valid oracle-controlled task; stop before learned selective gate.'})
        (out/'U3BR3_RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True);return result
    selective=copy.deepcopy(base);fixed=copy.deepcopy(base)
    sh=r.train_gate(selective,task,args,'selective',args.seed+4000)
    fh=r.train_gate(fixed,task,args,'fixed',args.seed+5000)
    se=r.evaluate(selective,task,args,'selective',args.seed+6000)
    fe=r.evaluate(fixed,task,args,'fixed',args.seed+6000)
    sid=r.evaluate(selective,task,args,'identity',args.seed+6000)
    selectivity=se['alpha_nonreset']-se['alpha_reset']
    advantage=se['post_sign_accuracy']-max(fe['post_sign_accuracy'],sid['post_sign_accuracy'])
    gates={'base_persistent_memory_pretrained':base_valid,'oracle_reset_task_solvable':oracle_valid,
           'identity_reset_control_worse':oracle_gap>=args.oracle_gap,
           'selective_pre_ge_gate':se['pre_sign_accuracy']>=args.task_gate,
           'selective_post_ge_gate':se['post_sign_accuracy']>=args.task_gate,
           'selective_alpha_gap_ge_0p20':selectivity>=.20,
           'selective_advantage_ge_gate':advantage>=args.selective_gap}
    result.update({'selective_train':sh,'fixed_train':fh,'selective_eval':se,'fixed_eval':fe,
                   'selective_identity_ablation':sid,'alpha_selectivity_gap':selectivity,
                   'post_reset_selective_advantage':advantage,'gates':gates,'PASS':all(gates.values())})
    result['decision']=('selective retention earns a causal functional role on corrected persistent H/g core v2.' if result['PASS']
                        else 'oracle task is valid but learned gate has not earned canonical status; keep core v2 and localize gate optimization only.')
    (out/'U3BR3_RESULTS.json').write_text(json.dumps(result,indent=2));
    torch.save({'base':base.state_dict(),'selective':selective.state_dict(),'fixed':fixed.state_dict(),'config':vars(args)},out/'U3BR3_MODELS.pt')
    print(json.dumps(result,indent=2),flush=True);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='unified_snn_transform/run/U3BR3')
    ap.add_argument('--seed',type=int,default=43);ap.add_argument('--tokens',type=int,default=16);ap.add_argument('--d',type=int,default=32)
    ap.add_argument('--heads',type=int,default=2);ap.add_argument('--features',type=int,default=32);ap.add_argument('--ff',type=int,default=64)
    ap.add_argument('--value-dim',type=int,default=8);ap.add_argument('--nkeys',type=int,default=4);ap.add_argument('--control-amp',type=float,default=1.0)
    ap.add_argument('--base-steps',type=int,default=500);ap.add_argument('--gate-steps',type=int,default=400);ap.add_argument('--batch',type=int,default=96)
    ap.add_argument('--base-lr',type=float,default=2e-3);ap.add_argument('--gate-lr',type=float,default=1e-2);ap.add_argument('--alpha-reg',type=float,default=.05)
    ap.add_argument('--report-every',type=int,default=100);ap.add_argument('--eval-batches',type=int,default=6);ap.add_argument('--eval-batch',type=int,default=128)
    ap.add_argument('--base-gate',type=float,default=.90);ap.add_argument('--task-gate',type=float,default=.90)
    ap.add_argument('--oracle-gap',type=float,default=.15);ap.add_argument('--selective-gap',type=float,default=.10);ap.add_argument('--threads',type=int,default=4)
    run(ap.parse_args())
if __name__=='__main__':main()
