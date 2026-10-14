#!/usr/bin/env python3
from __future__ import annotations

import argparse, copy, json, math, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import run_unified_gate as u


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(b).clamp_min(1e-12)).detach())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa=a.reshape(-1); bb=b.reshape(-1)
    return float(F.cosine_similarity(aa[None], bb[None]).detach().item())


class RetrievalModel(nn.Module):
    """Task wrapper. The reusable object under test remains UnifiedBlock."""
    def __init__(self, d: int, heads: int, features: int, ff: int, value_dim: int, seed: int):
        super().__init__()
        self.block = u.UnifiedBlock(d=d, heads=heads, features=features, ff=ff, seed=seed)
        self.readout = nn.Linear(d, value_dim)

    def fresh(self, x: torch.Tensor):
        changed=torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        y,_,stats=self.block.step(x, None, changed)
        return self.readout(y[:,-1]), stats


class AssocGenerator:
    """Random-value associative recall with fixed orthogonal address keys.

    There is no target shortcut: values are resampled independently every batch.
    The query contains only the address key and must retrieve the matching random value.
    """
    def __init__(self, tokens: int, d: int, value_dim: int, seed: int):
        self.tokens=tokens; self.slots=tokens-1; self.d=d; self.value_dim=value_dim
        self.key_dim=self.slots
        if self.key_dim + value_dim + 1 > d:
            raise ValueError(f"d={d} too small for {self.slots} keys + value_dim={value_dim} + role")
        self.role_idx=self.key_dim + value_dim
        self.keys=torch.eye(self.slots, dtype=torch.float32)
        self.seed=seed

    def sample(self, batch: int, generator: torch.Generator):
        B=batch; S=self.slots; D=self.d; kd=self.key_dim; vd=self.value_dim
        x=torch.zeros(B,self.tokens,D)
        x[:,:S,:kd]=self.keys[None]
        vals=(torch.randint(0,2,(B,S,vd),generator=generator).float()*2.0-1.0)
        x[:,:S,kd:kd+vd]=vals
        x[:,:S,self.role_idx]=1.0
        qidx=torch.randint(0,S,(B,),generator=generator)
        x[:,-1,:kd]=self.keys[qidx]
        x[:,-1,self.role_idx]=-1.0
        target=vals[torch.arange(B),qidx]
        return x, target, qidx, vals

    def progressive_reveal(self, batch: int, steps: int, generator: torch.Generator):
        final,target,qidx,vals=self.sample(batch,generator)
        B,S,D=final.shape[0],self.slots,self.d
        kd,vd=self.key_dim,self.value_dim
        base=final.clone(); base[:,:S,kd:kd+vd]=0.0
        orders=torch.stack([torch.randperm(S,generator=generator) for _ in range(B)])
        xs=[]; changed=[]
        prev=torch.zeros(B,S,dtype=torch.bool)
        for t in range(steps):
            n=round(S*t/max(1,steps-1))
            known=torch.zeros(B,S,dtype=torch.bool)
            if n:
                for b in range(B): known[b,orders[b,:n]]=True
            cur=base.clone()
            cur[:,:S,kd:kd+vd]=torch.where(known[:,:,None], final[:,:S,kd:kd+vd], torch.zeros_like(final[:,:S,kd:kd+vd]))
            if t==0:
                cm=torch.ones(B,self.tokens,dtype=torch.bool)
            else:
                cm=torch.zeros(B,self.tokens,dtype=torch.bool)
                cm[:,:S]=known & (~prev)
            xs.append(cur); changed.append(cm); prev=known
        return xs,changed,target,qidx,final


def metrics(pred: torch.Tensor, target: torch.Tensor):
    mse=float(F.mse_loss(pred,target).detach())
    sign=float(((pred>=0)==(target>=0)).float().mean().detach())
    cos=cosine(pred,target)
    return {'mse':mse,'sign_accuracy':sign,'cosine':cos}


@torch.no_grad()
def eval_model(model: RetrievalModel, gen: AssocGenerator, batches: int, batch: int, seed: int, shuffle_values: bool=False):
    model.eval(); g=torch.Generator(); g.manual_seed(seed)
    ps=[];ts=[]
    for _ in range(batches):
        x,tgt,qidx,vals=gen.sample(batch,g)
        if shuffle_values:
            # Destroy the key->value binding while leaving query/key statistics intact.
            perm=torch.stack([torch.randperm(gen.slots,generator=g) for _ in range(batch)])
            kd,vd=gen.key_dim,gen.value_dim
            sh=vals[torch.arange(batch)[:,None],perm]
            x[:,:gen.slots,kd:kd+vd]=sh
        p,_=model.fresh(x); ps.append(p);ts.append(tgt)
    return metrics(torch.cat(ps),torch.cat(ts))


def train_reference(args, gen: AssocGenerator):
    model=RetrievalModel(args.d,args.heads,args.features,args.ff,args.value_dim,args.seed+101)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(.9,.98),weight_decay=1e-4)
    g=torch.Generator();g.manual_seed(args.seed+1000)
    hist=[]; model.train()
    for st in range(1,args.train_steps+1):
        x,tgt,_,_=gen.sample(args.train_batch,g)
        pred,_=model.fresh(x)
        loss=F.mse_loss(pred,tgt)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if st==1 or st%args.report_every==0 or st==args.train_steps:
            with torch.no_grad():
                m=metrics(pred,tgt)
            row={'step':st,'loss':float(loss.detach()),**m};hist.append(row);print('TRAIN',json.dumps(row),flush=True)
    return model.eval(),hist


@torch.no_grad()
def collect_empirical_a(model: RetrievalModel, gen: AssocGenerator, seed: int, batches: int=8, batch: int=64):
    g=torch.Generator();g.manual_seed(seed);qs=[];ks=[]
    b=model.block
    for _ in range(batches):
        x,_,_,_=gen.sample(batch,g); z=b.n1(x)
        q=b.attn._split(b.attn.q(z)); k=b.attn._split(b.attn.k(z))
        qs.append(b.attn.phi_q.stable_a(q,True).reshape(-1).cpu())
        ks.append(b.attn.phi_k.stable_a(k,False).reshape(-1).cpu())
    return torch.cat(qs),torch.cat(ks)


def make_izh_candidate(model: RetrievalModel, aq, ak):
    cand=copy.deepcopy(model)
    qspec,qdec,qrow,_=u.calibrate_side(aq,cand.block.attn.m)
    kspec,kdec,krow,_=u.calibrate_side(ak,cand.block.attn.m)
    cand.block.attn.phi_q=u.IzhPhi(model.block.attn.phi_q,qspec,qdec)
    cand.block.attn.phi_k=u.IzhPhi(model.block.attn.phi_k,kspec,kdec)
    return cand.eval(), {'Q':qrow,'K':krow}


@torch.no_grad()
def fresh_parity(ref: RetrievalModel, cand: RetrievalModel, gen: AssocGenerator, seed: int, batches: int=4, batch: int=64):
    g=torch.Generator();g.manual_seed(seed);prs=[];pcs=[];ts=[]
    for _ in range(batches):
        x,t,_,_=gen.sample(batch,g); pr,_=ref.fresh(x); pc,_=cand.fresh(x)
        prs.append(pr);pcs.append(pc);ts.append(t)
    pr=torch.cat(prs);pc=torch.cat(pcs);t=torch.cat(ts)
    return {'prediction_rel_l2':rel_l2(pc,pr),'prediction_cosine':cosine(pc,pr),
            'reference_task':metrics(pr,t),'candidate_task':metrics(pc,t)}


def density(a: torch.Tensor,b: torch.Tensor,frac: float):
    d=(a-b).abs(); scale=float(torch.sqrt((a*a).mean()).clamp_min(1e-12)); thr=frac*scale
    feat=float((d>thr).float().mean())
    tok=float((d.reshape(d.shape[0],d.shape[1],-1).amax(-1)>thr).float().mean()) if d.ndim>=3 else float('nan')
    return {'feature_density':feat,'token_density':tok,'threshold':thr,'rel_l2':rel_l2(a,b)}


@torch.no_grad()
def traced_step(block: u.UnifiedBlock, x: torch.Tensor, state, changed: torch.Tensor):
    event_ratio=changed.float().mean(1); alpha=block.gate(event_ratio)
    n1=block.n1(x)
    attn,state,stats=block.attn.step(n1,state,alpha,changed)
    z=x+attn
    ff_in=block.n2(z)
    ff_hidden=F.gelu(block.f1(ff_in))
    ff_out=block.f2(ff_hidden)
    out=z+ff_out
    return out,state,{'n1':n1,'attn_out':attn,'ff_in':ff_in,'ff_hidden':ff_hidden,'ff_out':ff_out,'block_out':out,
                      'alpha_mean':float(alpha.mean()),'event_ratio':float(event_ratio.mean()),**stats}


@torch.no_grad()
def trajectory_audit(model: RetrievalModel, gen: AssocGenerator, args):
    g=torch.Generator();g.manual_seed(args.seed+5000)
    xs,masks,target,qidx,final=gen.progressive_reveal(args.audit_batch,args.trajectory_steps,g)
    state=None; prev=None; rows=[]; query=[]
    fracs=[1e-5,1e-3,1e-2,5e-2,1e-1]
    for t,(x,cm) in enumerate(zip(xs,masks)):
        out,state,tr=traced_step(model.block,x,state,cm)
        pred=model.readout(out[:,-1]); query.append(metrics(pred,target))
        row={'step':t,'event_ratio':tr['event_ratio'],'alpha_mean':tr['alpha_mean'],'kv_token_evals':tr['kv_token_evals']}
        if prev is not None:
            row['delta']={name:{str(f):density(tr[name],prev[name],f) for f in fracs}
                          for name in ('n1','attn_out','ff_in','ff_hidden','ff_out','block_out')}
        rows.append(row); prev=tr
    # summarize mean feature density over noninitial transitions
    summary={}
    for name in ('n1','attn_out','ff_in','ff_hidden','ff_out','block_out'):
        summary[name]={}
        for f in fracs:
            vals=[r['delta'][name][str(f)]['feature_density'] for r in rows[1:]]
            summary[name][str(f)]={'mean_feature_density':float(np.mean(vals)),'max_feature_density':float(np.max(vals))}
    return {'rows':rows,'query_metrics_by_step':query,'density_summary':summary}


def run(args):
    seed_all(args.seed);torch.set_num_threads(args.threads)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    gen=AssocGenerator(args.tokens,args.d,args.value_dim,args.seed)
    ref,hist=train_reference(args,gen)
    ref_task=eval_model(ref,gen,args.eval_batches,args.eval_batch,args.seed+2000,False)
    shuffled=eval_model(ref,gen,args.eval_batches,args.eval_batch,args.seed+2000,True)

    aq,ak=collect_empirical_a(ref,gen,args.seed+3000,batches=8,batch=min(64,args.eval_batch))
    cand,cal=make_izh_candidate(ref,aq,ak)
    parity=fresh_parity(ref,cand,gen,args.seed+4000,batches=4,batch=min(64,args.eval_batch))
    traj=trajectory_audit(ref,gen,args)

    task_drop=ref_task['sign_accuracy']-shuffled['sign_accuracy']
    # Do not force a sparsity conclusion. U3's purpose is to measure the trained block.
    d1=traj['density_summary']['block_out']['0.01']['mean_feature_density']
    ff1=traj['density_summary']['ff_hidden']['0.01']['mean_feature_density']
    decision={
        'task': 'associative retrieval is accepted as a trained lightweight operator probe' if ref_task['sign_accuracy']>=args.task_gate else 'training gate failed; do not interpret density',
        'qkv': 'retain exact event-cached Q/K/V from U2',
        'output_ffn': ('trained downstream deltas remain dense at 1% RMS threshold; do not promote naive delta-FFN/output' if max(d1,ff1)>0.70
                       else 'trained downstream deltas show exploitable effective sparsity; proceed to controlled event/delta output-FFN gate'),
        'strict_spiking': 'Izh feature map remains a neuronal-dynamics interface; spike activity is reported separately and is not assumed.'
    }
    gates={
        'retrieval_sign_accuracy':ref_task['sign_accuracy']>=args.task_gate,
        'binding_control_drop_ge_0p15':task_drop>=0.15,
        'izh_prediction_rel_l2_le_1pct':parity['prediction_rel_l2']<=.01,
        'izh_task_drop_le_0p02':parity['candidate_task']['sign_accuracy']>=parity['reference_task']['sign_accuracy']-.02,
    }
    result={'stage':'UNIFIED_U3_TRAINED_RETRIEVAL','config':vars(args),'train_history':hist,
            'reference_task':ref_task,'shuffled_binding_control':shuffled,'binding_sign_accuracy_drop':task_drop,
            'izh_calibration':cal,'izh_parity':parity,'trajectory_audit':traj,'decision':decision,'gates':gates,'PASS':all(gates.values())}
    (out/'U3_RESULTS.json').write_text(json.dumps(result,indent=2))
    torch.save({'model':ref.state_dict(),'config':vars(args)},out/'U3_REFERENCE.pt')
    print(json.dumps(result,indent=2),flush=True)
    if not result['PASS']:
        raise SystemExit(2)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=Path('unified_snn_transform/run/U3'))
    ap.add_argument('--seed',type=int,default=7);ap.add_argument('--tokens',type=int,default=16)
    ap.add_argument('--d',type=int,default=32);ap.add_argument('--heads',type=int,default=2)
    ap.add_argument('--features',type=int,default=32);ap.add_argument('--ff',type=int,default=64)
    ap.add_argument('--value-dim',type=int,default=8)
    ap.add_argument('--train-steps',type=int,default=800);ap.add_argument('--train-batch',type=int,default=128)
    ap.add_argument('--lr',type=float,default=2e-3);ap.add_argument('--report-every',type=int,default=100)
    ap.add_argument('--eval-batches',type=int,default=8);ap.add_argument('--eval-batch',type=int,default=128)
    ap.add_argument('--trajectory-steps',type=int,default=6);ap.add_argument('--audit-batch',type=int,default=64)
    ap.add_argument('--task-gate',type=float,default=.85);ap.add_argument('--threads',type=int,default=4)
    run(ap.parse_args())


if __name__=='__main__': main()
