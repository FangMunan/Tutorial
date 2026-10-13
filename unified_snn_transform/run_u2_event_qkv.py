#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F
from run_unified_gate import UnifiedBlock, seed_all, sequence, rel_l2


def split_proj(y, heads):
    B,N,D=y.shape; dh=D//heads
    return y.reshape(B,N,heads,dh).transpose(1,2)


@torch.no_grad()
def event_projection_gate(block, xs, masks, which):
    linear=getattr(block.attn,which)
    phi=block.attn.phi_q if which=='q' else (block.attn.phi_k if which=='k' else None)
    cache=None; phi_cache=None; max_proj=0.; max_phi=0.; evals=0; dense=0; rows=[]
    for t,(x,m) in enumerate(zip(xs,masks)):
        z=block.n1(x); dense_y=linear(z); dense+=x.shape[0]*x.shape[1]
        if cache is None:
            cache=dense_y.clone(); evals+=x.shape[0]*x.shape[1]
        else:
            for b in range(x.shape[0]):
                idx=torch.nonzero(m[b],as_tuple=False).flatten()
                if idx.numel():
                    cache[b,idx]=linear(z[b:b+1,idx])[0]; evals+=int(idx.numel())
        pe=rel_l2(cache,dense_y); max_proj=max(max_proj,pe); pr=None
        if phi is not None:
            dense_split=split_proj(dense_y,block.heads)
            dense_phi=phi(dense_split,is_query=(which=='q'))
            if phi_cache is None:
                phi_cache=dense_phi.clone()
            else:
                for b in range(x.shape[0]):
                    idx=torch.nonzero(m[b],as_tuple=False).flatten()
                    if idx.numel():
                        sub=split_proj(cache[b:b+1,idx],block.heads)
                        p=phi(sub,is_query=(which=='q'))
                        phi_cache[b,:,idx,:]=p[0]
            pr=rel_l2(phi_cache,dense_phi); max_phi=max(max_phi,pr)
        rows.append({'step':t,'projection_rel_l2':pe,'phi_rel_l2':pr})
    return {'rows':rows,'max_projection_rel_l2':max_proj,
            'max_phi_rel_l2':max_phi if phi is not None else None,
            'event_token_evals':evals,'dense_token_evals':dense,
            'token_eval_reduction':1-evals/dense}


def density(a,b,eps):
    d=(a-b).abs()
    return {'feature_density':float((d>eps).float().mean()),
            'token_density':float((d.amax(-1)>eps).float().mean()),
            'rel_l2':rel_l2(a,b)}


@torch.no_grad()
def downstream_density(block,xs,masks,eps):
    st=None; prev={}; rows=[]
    for t,(x,m) in enumerate(zip(xs,masks)):
        event_ratio=m.float().mean(1); alpha=block.gate(event_ratio)
        a,st,_=block.attn.step(block.n1(x),st,alpha,m)
        z=x+a; ff_in=block.n2(z); ff_hidden=F.gelu(block.f1(ff_in)); out=z+block.f2(ff_hidden)
        cur={'attn_out':a,'ff_in':ff_in,'ff_hidden':ff_hidden,'block_out':out}
        if t>0:
            r={'step':t}
            for k,v in cur.items(): r[k]=density(v,prev[k],eps)
            rows.append(r)
        prev={k:v.clone() for k,v in cur.items()}
    summary={}
    for k in ('attn_out','ff_in','ff_hidden','block_out'):
        summary[k]={'mean_feature_density':sum(r[k]['feature_density'] for r in rows)/len(rows),
                    'mean_token_density':sum(r[k]['token_density'] for r in rows)/len(rows)}
    return {'eps':eps,'rows':rows,'summary':summary}


def run(args):
    seed_all(args.seed);torch.set_num_threads(args.threads)
    block=UnifiedBlock(args.d,args.heads,args.features,args.ff,args.seed+101).eval()
    xs,masks=sequence(args.seed+2000,args.batch,args.tokens,args.d,args.steps,args.change_frac)
    q=event_projection_gate(block,xs,masks,'q')
    k=event_projection_gate(block,xs,masks,'k')
    v=event_projection_gate(block,xs,masks,'v')
    down=downstream_density(block,xs,masks,args.eps)
    maxerr=max(q['max_projection_rel_l2'],k['max_projection_rel_l2'],v['max_projection_rel_l2'],
               q['max_phi_rel_l2'],k['max_phi_rel_l2'])
    minred=min(q['token_eval_reduction'],k['token_eval_reduction'],v['token_eval_reduction'])
    gates={'event_qkv_exact_le_1e6':maxerr<=1e-6,
           'qkv_token_eval_reduction_ge_50pct':minred>=.5}
    result={'stage':'UNIFIED_U2_EVENT_QKV','config':vars(args),'q':q,'k':k,'v':v,
            'downstream_delta_density':down,
            'decision':{
                'qkv':'promote event-cached token execution into canonical block',
                'output_ffn':'do not assume sparsity; use trained-task downstream densities before U3/U4 conversion'},
            'gates':gates,'PASS':all(gates.values())}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'U2_RESULTS.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='U2_RUN')
    ap.add_argument('--seed',type=int,default=7);ap.add_argument('--tokens',type=int,default=16)
    ap.add_argument('--d',type=int,default=32);ap.add_argument('--heads',type=int,default=2)
    ap.add_argument('--features',type=int,default=32);ap.add_argument('--ff',type=int,default=64)
    ap.add_argument('--steps',type=int,default=6);ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--change-frac',type=float,default=.25);ap.add_argument('--eps',type=float,default=1e-5)
    ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args();r=run(a);raise SystemExit(0 if r['PASS'] else 2)
