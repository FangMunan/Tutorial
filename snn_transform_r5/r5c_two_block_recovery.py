#!/usr/bin/env python3
"""R5C: joint recovery of the final two H/g blocks.

Purpose
-------
R5B showed that every trainable parameter of the final H/g block can recover
without backpropagation, torch.optim, or a learned feedback network. R5C is the
first explicit deep-credit gate: blocks 2 and 3 (zero-based) are damaged and
plastic simultaneously while blocks 0-1, embeddings, class condition and the
readout remain fixed at the E29 transferred solution.

Credit semantics
----------------
* Final block (block 3): Q/O/FFN/LN use token-local masked loss; K/V use
  image-level masked loss, because a sender key/value affects all queries.
* Penultimate block (block 2): ALL local perturbations use image-level masked
  loss. Even a query-local perturbation at block 2 can be redistributed across
  tokens by block 3 attention, so token-local credit would be invalid.

This is deliberately a recovery gate, not from-scratch training.
"""
from __future__ import annotations

import argparse, csv, json, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'snn_transform_e29'))
sys.path.insert(0, str(ROOT / 'snn_transform_r4'))
import maskgit_b0_mnist012 as b0
import e29_forward_equivalence as e29
import r4_single_module_recovery as r4


def clip_(g, limit):
    n = float(g.norm())
    if limit > 0 and n > limit:
        g.mul_(limit / (n + 1e-12)); n = limit
    return n


def block_params(model, bi):
    b = model.blocks[bi]; p = f'blocks.{bi}'
    return {
        f'{p}.attn.WQ.weight': b.attn.WQ.weight, f'{p}.attn.WQ.bias': b.attn.WQ.bias,
        f'{p}.attn.WK.weight': b.attn.WK.weight, f'{p}.attn.WK.bias': b.attn.WK.bias,
        f'{p}.attn.WV.weight': b.attn.WV.weight, f'{p}.attn.WV.bias': b.attn.WV.bias,
        f'{p}.attn.WO.weight': b.attn.WO.weight, f'{p}.attn.WO.bias': b.attn.WO.bias,
        f'{p}.n1.weight': b.n1.weight, f'{p}.n1.bias': b.n1.bias,
        f'{p}.f1.weight': b.f1.weight, f'{p}.f1.bias': b.f1.bias,
        f'{p}.f2.weight': b.f2.weight, f'{p}.f2.bias': b.f2.bias,
        f'{p}.n2.weight': b.n2.weight, f'{p}.n2.bias': b.n2.bias,
    }


def target_params(model):
    d = {}; d.update(block_params(model, 2)); d.update(block_params(model, 3)); return d


@torch.no_grad()
def prefix_to(model, tokens, labels, bi):
    z = model.tok(tokens) + model.pos + model.cls(labels)[:, None]
    for j in range(bi): z = model.blocks[j](z)
    return z


@torch.no_grad()
def injected_block(block, x0, kind=None, delta=None, cache=False):
    B,N,D = x0.shape; H = block.attn.h; hd = block.attn.hd
    q0 = block.attn.WQ(x0); k0 = block.attn.WK(x0); v0 = block.attn.WV(x0)
    if kind == 'q': q0 = q0 + delta
    if kind == 'k': k0 = k0 + delta
    if kind == 'v': v0 = v0 + delta
    def sp(z): return z.reshape(B,N,H,hd).transpose(1,2)
    q,k,v = sp(q0),sp(k0),sp(v0)
    pq,pk = block.attn.rf.q(q), block.attn.rf.k(k)
    g = pk.sum(2); HH = torch.einsum('bhnm,bhnd->bhmd', pk, v)
    num = torch.einsum('bhnm,bhmd->bhnd', pq, HH)
    den = torch.einsum('bhnm,bhm->bhn', pq, g).clamp_min(1e-12)
    read = (num / den[...,None]).transpose(1,2).reshape(B,N,D)
    wo = block.attn.WO(read)
    if kind == 'wo': wo = wo + delta
    pre_n1 = x0 + wo
    r1 = (pre_n1-pre_n1.mean(-1,keepdim=True))/torch.sqrt(pre_n1.var(-1,unbiased=False,keepdim=True)+block.n1.eps)
    y1 = block.n1(pre_n1)
    if kind == 'n1': y1 = y1 + delta
    pre1 = block.f1(y1)
    if kind == 'f1': pre1 = pre1 + delta
    h = F.gelu(pre1)
    u2 = block.f2(h)
    if kind == 'f2': u2 = u2 + delta
    pre_n2 = y1 + u2
    r2 = (pre_n2-pre_n2.mean(-1,keepdim=True))/torch.sqrt(pre_n2.var(-1,unbiased=False,keepdim=True)+block.n2.eps)
    y2 = block.n2(pre_n2)
    if kind == 'n2': y2 = y2 + delta
    if cache: return y2, {'x0':x0,'read':read,'r1':r1,'y1':y1,'h':h,'r2':r2}
    return y2


@torch.no_grad()
def logits_from_target(model, x0, bi, kind=None, delta=None, cache=False):
    if cache:
        z,c = injected_block(model.blocks[bi], x0, kind, delta, True)
    else:
        z = injected_block(model.blocks[bi], x0, kind, delta, False); c = None
    for j in range(bi+1, len(model.blocks)): z = model.blocks[j](z)
    z = model.ln(F.gelu(model.dense(z)))
    logits = F.linear(z, model.tok.weight[:b0.K], model.bias)
    return (logits,c) if cache else logits


def token_mod(lp,lm,t,mask,sigma):
    cp = F.cross_entropy(lp.transpose(1,2), t, reduction='none')
    cm = F.cross_entropy(lm.transpose(1,2), t, reduction='none')
    m = (cp-cm)/(2*sigma)
    return m, float(.5*(cp[mask].mean()+cm[mask].mean()))


def image_mod(lp,lm,t,mask,sigma):
    cp = F.cross_entropy(lp.transpose(1,2), t, reduction='none')
    cm = F.cross_entropy(lm.transpose(1,2), t, reduction='none')
    cnt = mask.sum(1).clamp_min(1)
    a = (cp*mask).sum(1)/cnt; b = (cm*mask).sum(1)/cnt
    return (a-b)/(2*sigma), float(.5*(a.mean()+b.mean()))


def linear_token_grad(m,mask,xi,pre):
    sel = mask.float(); n = sel.sum().clamp_min(1.)
    return (torch.einsum('bn,bno,bni->oi',m*sel,xi,pre)/n,
            torch.einsum('bn,bno->o',m*sel,xi)/n)


def linear_image_grad(m,xi,pre):
    B = max(1,len(pre))
    return (torch.einsum('b,bno,bni->oi',m,xi,pre)/B,
            torch.einsum('b,bno->o',m,xi)/B)


def ln_token_grad(m,mask,xi,r):
    sel=mask.float(); n=sel.sum().clamp_min(1.); gy=(m*sel)[...,None]*xi
    return (gy.mul(r).sum((0,1))/n, gy.sum((0,1))/n)


def ln_image_grad(m,xi,r):
    B=max(1,len(r)); gy=m[:,None,None]*xi
    return (gy.mul(r).sum((0,1))/B, gy.sum((0,1))/B)


@torch.no_grad()
def probe(model,x0,bi,t,mask,kind,shape,sigma,pre,generator,mode,is_ln=False,r=None):
    xi=torch.randn(shape,generator=generator)
    lp=logits_from_target(model,x0,bi,kind,sigma*xi)
    lm=logits_from_target(model,x0,bi,kind,-sigma*xi)
    if mode=='token':
        m,ce=token_mod(lp,lm,t,mask,sigma)
        grad=ln_token_grad(m,mask,xi,r) if is_ln else linear_token_grad(m,mask,xi,pre)
    else:
        m,ce=image_mod(lp,lm,t,mask,sigma)
        grad=ln_image_grad(m,xi,r) if is_ln else linear_image_grad(m,xi,pre)
    return grad,ce


def lrs_for(params,a):
    out={}
    for k in params:
        if '.WQ.' in k or '.WK.' in k or '.WV.' in k: v=a.lr_qkv
        elif '.WO.' in k: v=a.lr_wo
        elif '.f1.' in k: v=a.lr_f1
        elif '.f2.' in k: v=a.lr_f2
        else: v=a.lr_ln
        if k.startswith('blocks.2.'): v *= a.deep_lr_scale
        if k.endswith('.bias'): v *= .5
        out[k]=v
    return out


@torch.no_grad()
def block_gradset(model,bi,t,y,inp,mask,sigma,generator,clip):
    x0=prefix_to(model,inp,y,bi); _,c=logits_from_target(model,x0,bi,cache=True)
    B,N,D=x0.shape; FF=e29.FF
    # Deep block 2 must use image-level modulation for every node because block 3 mixes tokens.
    deep=(bi==2)
    modes={
        'q':'image' if deep else 'token',
        'k':'image','v':'image',
        'wo':'image' if deep else 'token',
        'n1':'image' if deep else 'token',
        'f1':'image' if deep else 'token',
        'f2':'image' if deep else 'token',
        'n2':'image' if deep else 'token',
    }
    specs=[
      ('WQ','q',(B,N,D),c['x0'],False,None),('WK','k',(B,N,D),c['x0'],False,None),('WV','v',(B,N,D),c['x0'],False,None),
      ('WO','wo',(B,N,D),c['read'],False,None),('n1','n1',(B,N,D),None,True,c['r1']),
      ('f1','f1',(B,N,FF),c['y1'],False,None),('f2','f2',(B,N,D),c['h'],False,None),('n2','n2',(B,N,D),None,True,c['r2'])]
    grads={}; ces=[]; norms={}; p=f'blocks.{bi}'
    for label,kind,shape,pre,is_ln,r in specs:
        (gw,gb),ce=probe(model,x0,bi,t,mask,kind,shape,sigma,pre,generator,modes[kind],is_ln,r); ces.append(ce)
        if label in ('WQ','WK','WV','WO'):
            base=f'{p}.attn.{label}'
        else:
            base=f'{p}.{label}'
        for suffix,g in [('weight',gw),('bias',gb)]:
            name=f'{base}.{suffix}'; norms[name]=clip_(g,clip); grads[name]=g
    return grads,float(np.mean(ces)),norms


@torch.no_grad()
def damage_one(model,bi,attn=.60,ffn=.50,ln=.75):
    b=model.blocks[bi]
    for lin in (b.attn.WQ,b.attn.WK,b.attn.WV,b.attn.WO): lin.weight.mul_(attn);lin.bias.mul_(attn)
    for lin in (b.f1,b.f2): lin.weight.mul_(ffn);lin.bias.mul_(ffn)
    b.n1.weight.mul_(ln);b.n1.bias.zero_();b.n2.weight.mul_(ln);b.n2.bias.zero_()


def weight_diag_group(params,gold):
    return {k:r4.weight_diag(p,gold[k]) for k,p in params.items() if k.endswith('.weight')}


def copy_group(dst,src,names):
    for n in names: dst[n].copy_(src[n])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--codebook',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--features',type=int,default=256);ap.add_argument('--rf-kind',default='p3norm');ap.add_argument('--rf-seed',type=int,default=29257)
    ap.add_argument('--epochs',type=int,default=5);ap.add_argument('--batch',type=int,default=128);ap.add_argument('--sigma',type=float,default=.05)
    ap.add_argument('--lr-qkv',type=float,default=.020);ap.add_argument('--lr-wo',type=float,default=.040);ap.add_argument('--lr-f1',type=float,default=.055);ap.add_argument('--lr-f2',type=float,default=.080);ap.add_argument('--lr-ln',type=float,default=.015);ap.add_argument('--deep-lr-scale',type=float,default=.65)
    ap.add_argument('--rho-fast',type=float,default=.60);ap.add_argument('--rho-slow',type=float,default=.95);ap.add_argument('--mix-fast',type=float,default=.70);ap.add_argument('--clip',type=float,default=.60)
    ap.add_argument('--threads',type=int,default=2);ap.add_argument('--seed',type=int,default=5701)
    a=ap.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(a.threads);torch.set_grad_enabled(False);a.out.mkdir(parents=True,exist_ok=True)

    xall,yall,xte0,yte0=b0.load(a.data);x012,y012=b0.subset012(xall,yall);xte,yte=b0.subset012(xte0,yte0);xtr,ytr,xva,yva=b0.split(x012,y012,.1,7);cb=np.load(a.codebook);tr=b0.encode(xtr,cb);va=b0.encode(xva,cb);te=b0.encode(xte,cb)
    exact=b0.Model(128,4,4,512,.05);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);exact.load_state_dict(ck['model']);exact.eval()
    model=e29.HgB0Model(a.features,a.rf_kind,a.rf_seed);e29.transfer_b0(exact,model);model.eval();r4.freeze_all(model)
    params=target_params(model);allowed=set(params);gold={k:p.clone() for k,p in params.items()};hash_i=r4.model_hashes(model)
    names2={k for k in params if k.startswith('blocks.2.')};names3={k for k in params if k.startswith('blocks.3.')}
    ratios=(.25,.5,.9,1.0);intact=r4.eval_mask_ratios(model,te,yte,ratios);clf=r4.external_classifier(xtr,ytr);gen_i=r4.generate_snapshot(model,cb,clf,a.out,'intact',a.seed+10)

    # Control: last-block-only lesion quantifies how much extra damage block 2 adds.
    damage_one(model,3); block3_only=r4.eval_mask_ratios(model,te,yte,ratios)
    copy_group(params,gold,names3)

    # Actual R5C lesion: both final blocks damaged simultaneously.
    damage_one(model,2);damage_one(model,3);damaged_state={k:p.clone() for k,p in params.items()};damaged=r4.eval_mask_ratios(model,te,yte,ratios);gen_d=r4.generate_snapshot(model,cb,clf,a.out,'damaged_two_blocks',a.seed+10)
    hash_d=r4.model_hashes(model);changed_damage={n for n in hash_i if hash_i[n]!=hash_d[n]}
    required={f'blocks.{bi}.{s}' for bi in (2,3) for s in ('attn.WQ.weight','attn.WK.weight','attn.WV.weight','attn.WO.weight','f1.weight','f2.weight','n1.weight','n2.weight')}
    if not changed_damage.issubset(allowed) or not required.issubset(changed_damage): raise RuntimeError('two-block damage isolation failed')

    states={k:(torch.zeros_like(p),torch.zeros_like(p)) for k,p in params.items()};lr=lrs_for(params,a);rng=np.random.default_rng(a.seed+77);tg=torch.Generator();tg.manual_seed(a.seed+99)
    hist=[];best=float('inf');best_state={k:p.clone() for k,p in params.items()}
    for ep in range(1,a.epochs+1):
        order=rng.permutation(len(tr));ces2=[];ces3=[];n2=[];n3=[]
        for s in range(0,len(order),a.batch):
            idx=order[s:s+a.batch];t,y,inp,mask=r4.masked_random_batch(tr,ytr,idx)
            g2,ce2,norm2=block_gradset(model,2,t,y,inp,mask,a.sigma,tg,a.clip)
            g3,ce3,norm3=block_gradset(model,3,t,y,inp,mask,a.sigma,tg,a.clip)
            # Simultaneous update after both blocks' probes were measured at the same parameter state.
            for name,g in {**g2,**g3}.items():
                fast,slow=states[name];fast.mul_(a.rho_fast).add_(g,alpha=1-a.rho_fast);slow.mul_(a.rho_slow).add_(g,alpha=1-a.rho_slow);u=a.mix_fast*fast+(1-a.mix_fast)*slow;params[name].add_(u,alpha=-lr[name])
            ces2.append(ce2);ces3.append(ce3);n2.append(float(np.mean(list(norm2.values()))));n3.append(float(np.mean(list(norm3.values()))))
        val=r4.eval_mask_ratios(model,va,yva,(.5,),batch=256)['0.5']['ce'];test50=r4.eval_mask_ratios(model,te,yte,(.5,),batch=256,limit=1024)['0.5']['ce']
        rec={'epoch':ep,'mean_probe_ce_block2':float(np.mean(ces2)),'mean_probe_ce_block3':float(np.mean(ces3)),'mean_grad_norm_block2':float(np.mean(n2)),'mean_grad_norm_block3':float(np.mean(n3)),'val_ce_50':val,'test_ce_50_first1024':test50};hist.append(rec);print(json.dumps(rec),flush=True)
        if val<best:best=val;best_state={k:p.clone() for k,p in params.items()}
    copy_group(params,best_state,set(params))
    recovered=r4.eval_mask_ratios(model,te,yte,ratios);gen_r=r4.generate_snapshot(model,cb,clf,a.out,'recovered_two_blocks',a.seed+10);diag=weight_diag_group(params,gold)

    # Contribution audit: revert one recovered block at a time to its damaged state.
    recovered_state={k:p.clone() for k,p in params.items()}
    copy_group(params,damaged_state,names2);revert2=r4.eval_mask_ratios(model,te,yte,(.5,),batch=256)['0.5']['ce'];copy_group(params,recovered_state,names2)
    copy_group(params,damaged_state,names3);revert3=r4.eval_mask_ratios(model,te,yte,(.5,),batch=256)['0.5']['ce'];copy_group(params,recovered_state,names3)

    hash_f=r4.model_hashes(model);changed_final={n for n in hash_i if hash_i[n]!=hash_f[n]}
    if not changed_final.issubset(allowed): raise RuntimeError(f'outside two-block learning: {sorted(changed_final-allowed)}')
    if any(p.grad is not None for p in model.parameters()): raise RuntimeError('autograd gradient exists in R5C')
    def frac(r):
        ci=intact[r]['ce'];cd=damaged[r]['ce'];cr=recovered[r]['ce'];return float((cd-cr)/(cd-ci)) if abs(cd-ci)>1e-12 else float('nan')
    deep_increment=damaged['0.5']['ce']-block3_only['0.5']['ce']
    contrib2=revert2-recovered['0.5']['ce'];contrib3=revert3-recovered['0.5']['ce']
    gate={
      'damage_isolated_to_blocks_2_3':changed_damage.issubset(allowed) and required.issubset(changed_damage),
      'learning_isolated_to_blocks_2_3':changed_final.issubset(allowed),
      'no_autograd_gradients':all(p.grad is None for p in model.parameters()),
      'damage_delta_ce_50':damaged['0.5']['ce']-intact['0.5']['ce'],
      'deep_block_damage_increment_ce_50':deep_increment,
      'recovery_fraction_ce_25':frac('0.25'),'recovery_fraction_ce_50':frac('0.5'),'recovery_fraction_ce_90':frac('0.9'),
      'recovered_better_25':recovered['0.25']['ce']<damaged['0.25']['ce'],'recovered_better_50':recovered['0.5']['ce']<damaged['0.5']['ce'],'recovered_better_90':recovered['0.9']['ce']<damaged['0.9']['ce'],
      'block2_recovery_functional_contribution_ce_50':contrib2,'block3_recovery_functional_contribution_ce_50':contrib3,
    }
    gate['pass']=bool(gate['damage_delta_ce_50']>.10 and gate['deep_block_damage_increment_ce_50']>.01 and gate['recovery_fraction_ce_50']>.50 and gate['recovered_better_25'] and gate['recovered_better_50'] and gate['recovered_better_90'] and gate['block2_recovery_functional_contribution_ce_50']>.003 and gate['no_autograd_gradients'] and gate['learning_isolated_to_blocks_2_3'])
    out={'stage':'R5C joint final-two-block H/g recovery','scope':'blocks 2 and 3 plastic; blocks 0-1 and embeddings/readout fixed','credit_rule':'block2 all image-level due downstream attention mixing; block3 Q/O/FFN/LN token-local and K/V image-level','backprop_used':False,'torch_optimizer_used':False,'learned_feedback_network_used':False,'rf_kind':a.rf_kind,'rf_features':a.features,'intact_eval':intact,'block3_only_damaged_eval':block3_only,'damaged_eval':damaged,'recovered_eval':recovered,'generation_intact':gen_i,'generation_damaged':gen_d,'generation_recovered':gen_r,'best_validation_ce_50':best,'gate':gate,'recovered_weight_diag':diag,'history':hist}
    (a.out/'metrics.json').write_text(json.dumps(out,indent=2));
    with (a.out/'history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=hist[0].keys());w.writeheader();w.writerows(hist)
    torch.save({'params':{k:p.clone() for k,p in params.items()},'metrics':out},a.out/'recovery_checkpoint.pt')
    print('FINAL '+json.dumps(out),flush=True)

if __name__=='__main__': main()
