#!/usr/bin/env python3
"""R5B: full final H/g block recovery gate.

R5A established joint local recovery of the final FFN pair (W1,W2). R5B now
allows every trainable parameter of the final H/g block to adapt, while keeping
all earlier blocks, embeddings and readout/head fixed at the E29 transferred
solution.

Plastic final-block parameters:
  WQ,WK,WV,WO (+ biases)
  W1,W2 (+ biases)
  LayerNorm1 gamma/beta
  LayerNorm2 gamma/beta

No backward/autograd credit assignment, torch optimizer, learned inverse matrix,
or extra trainable network is used. Credit is estimated only from antithetic
forward perturbations at each local computational node and converted to local
three-factor parameter updates. Q/O/FFN/LN use token-local masked loss. K/V use
an image-level masked loss because a sender key/value can affect every query.
"""
from __future__ import annotations

import argparse, csv, json, math, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'snn_transform_e29'))
sys.path.insert(0,str(ROOT/'snn_transform_r4'))
import maskgit_b0_mnist012 as b0
import e29_forward_equivalence as e29
import r4_single_module_recovery as r4


def clip_(g,limit):
    n=float(g.norm())
    if limit>0 and n>limit:
        g.mul_(limit/(n+1e-12)); n=limit
    return n

@torch.no_grad()
def prefix(model,tokens,labels):
    z=model.tok(tokens)+model.pos+model.cls(labels)[:,None]
    for block in model.blocks[:-1]: z=block(z)
    return z

@torch.no_grad()
def final_forward(model,x0,kind=None,delta=None,cache=False):
    b=model.blocks[-1]; B,N,D=x0.shape; H=b.attn.h; hd=b.attn.hd
    q0=b.attn.WQ(x0); k0=b.attn.WK(x0); v0=b.attn.WV(x0)
    if kind=='q': q0=q0+delta
    if kind=='k': k0=k0+delta
    if kind=='v': v0=v0+delta
    def sp(z): return z.reshape(B,N,H,hd).transpose(1,2)
    q,k,v=sp(q0),sp(k0),sp(v0)
    pq,pk=b.attn.rf.q(q),b.attn.rf.k(k)
    g=pk.sum(2); HH=torch.einsum('bhnm,bhnd->bhmd',pk,v)
    num=torch.einsum('bhnm,bhmd->bhnd',pq,HH)
    den=torch.einsum('bhnm,bhm->bhn',pq,g).clamp_min(1e-12)
    read=(num/den[...,None]).transpose(1,2).reshape(B,N,D)
    wo=b.attn.WO(read)
    if kind=='wo': wo=wo+delta
    pre_n1=x0+wo
    r1=(pre_n1-pre_n1.mean(-1,keepdim=True))/torch.sqrt(pre_n1.var(-1,unbiased=False,keepdim=True)+b.n1.eps)
    y1=b.n1(pre_n1)
    if kind=='n1': y1=y1+delta
    pre1=b.f1(y1)
    if kind=='f1': pre1=pre1+delta
    h=F.gelu(pre1)
    u2=b.f2(h)
    if kind=='f2': u2=u2+delta
    pre_n2=y1+u2
    r2=(pre_n2-pre_n2.mean(-1,keepdim=True))/torch.sqrt(pre_n2.var(-1,unbiased=False,keepdim=True)+b.n2.eps)
    y2=b.n2(pre_n2)
    if kind=='n2': y2=y2+delta
    z=model.ln(F.gelu(model.dense(y2)))
    logits=F.linear(z,model.tok.weight[:b0.K],model.bias)
    if not cache: return logits
    return logits,{'x0':x0,'read':read,'r1':r1,'y1':y1,'h':h,'r2':r2}


def token_mod(lp,lm,t,mask,sigma):
    cp=F.cross_entropy(lp.transpose(1,2),t,reduction='none')
    cm=F.cross_entropy(lm.transpose(1,2),t,reduction='none')
    m=(cp-cm)/(2*sigma)
    return m, float(0.5*(cp[mask].mean()+cm[mask].mean()))


def image_mod(lp,lm,t,mask,sigma):
    cp=F.cross_entropy(lp.transpose(1,2),t,reduction='none')
    cm=F.cross_entropy(lm.transpose(1,2),t,reduction='none')
    cnt=mask.sum(1).clamp_min(1)
    lp_img=(cp*mask).sum(1)/cnt; lm_img=(cm*mask).sum(1)/cnt
    m=(lp_img-lm_img)/(2*sigma)
    return m, float(0.5*(lp_img.mean()+lm_img.mean()))


def linear_token_grad(m,mask,xi,pre):
    sel=mask.float(); n=sel.sum().clamp_min(1.)
    gw=torch.einsum('bn,bno,bni->oi',m*sel,xi,pre)/n
    gb=torch.einsum('bn,bno->o',m*sel,xi)/n
    return gw,gb


def linear_image_grad(m,xi,pre):
    B=max(1,len(pre))
    gw=torch.einsum('b,bno,bni->oi',m,xi,pre)/B
    gb=torch.einsum('b,bno->o',m,xi)/B
    return gw,gb


def ln_token_grad(m,mask,xi,r):
    sel=mask.float(); n=sel.sum().clamp_min(1.)
    gy=(m*sel)[...,None]*xi
    gg=(gy*r).sum((0,1))/n
    gb=gy.sum((0,1))/n
    return gg,gb

@torch.no_grad()
def probe_token(model,x0,t,mask,kind,shape,sigma,pre,generator,is_ln=False,r=None):
    xi=torch.randn(shape,generator=generator)
    lp=final_forward(model,x0,kind,sigma*xi); lm=final_forward(model,x0,kind,-sigma*xi)
    m,ce=token_mod(lp,lm,t,mask,sigma)
    if is_ln: return ln_token_grad(m,mask,xi,r),ce
    return linear_token_grad(m,mask,xi,pre),ce

@torch.no_grad()
def probe_image(model,x0,t,mask,kind,shape,sigma,pre,generator):
    xi=torch.randn(shape,generator=generator)
    lp=final_forward(model,x0,kind,sigma*xi); lm=final_forward(model,x0,kind,-sigma*xi)
    m,ce=image_mod(lp,lm,t,mask,sigma)
    return linear_image_grad(m,xi,pre),ce


def target_params(model):
    b=model.blocks[-1]
    return {
      'blocks.3.attn.WQ.weight':b.attn.WQ.weight,'blocks.3.attn.WQ.bias':b.attn.WQ.bias,
      'blocks.3.attn.WK.weight':b.attn.WK.weight,'blocks.3.attn.WK.bias':b.attn.WK.bias,
      'blocks.3.attn.WV.weight':b.attn.WV.weight,'blocks.3.attn.WV.bias':b.attn.WV.bias,
      'blocks.3.attn.WO.weight':b.attn.WO.weight,'blocks.3.attn.WO.bias':b.attn.WO.bias,
      'blocks.3.n1.weight':b.n1.weight,'blocks.3.n1.bias':b.n1.bias,
      'blocks.3.f1.weight':b.f1.weight,'blocks.3.f1.bias':b.f1.bias,
      'blocks.3.f2.weight':b.f2.weight,'blocks.3.f2.bias':b.f2.bias,
      'blocks.3.n2.weight':b.n2.weight,'blocks.3.n2.bias':b.n2.bias,
    }

@torch.no_grad()
def damage_block(model):
    b=model.blocks[-1]
    for lin in (b.attn.WQ,b.attn.WK,b.attn.WV,b.attn.WO):
        lin.weight.mul_(0.60); lin.bias.mul_(0.60)
    b.f1.weight.mul_(0.50); b.f1.bias.mul_(0.50)
    b.f2.weight.mul_(0.50); b.f2.bias.mul_(0.50)
    b.n1.weight.mul_(0.75); b.n1.bias.zero_()
    b.n2.weight.mul_(0.75); b.n2.bias.zero_()

@torch.no_grad()
def block_step(model,t,y,inp,mask,states,lr,sigma,rho_fast,rho_slow,mix,clip,generator):
    b=model.blocks[-1]; x0=prefix(model,inp,y)
    _,c=final_forward(model,x0,cache=True)
    B,N,D=x0.shape; FF=e29.FF
    grads={}; ces=[]; norms={}
    # Q is query-local. K/V are sender variables, so use image-level masked loss.
    (gq,gbq),ce=probe_token(model,x0,t,mask,'q',(B,N,D),sigma,c['x0'],generator); ces.append(ce)
    (gk,gbk),ce=probe_image(model,x0,t,mask,'k',(B,N,D),sigma,c['x0'],generator); ces.append(ce)
    (gv,gbv),ce=probe_image(model,x0,t,mask,'v',(B,N,D),sigma,c['x0'],generator); ces.append(ce)
    (go,gbo),ce=probe_token(model,x0,t,mask,'wo',(B,N,D),sigma,c['read'],generator); ces.append(ce)
    (gn1,gbn1),ce=probe_token(model,x0,t,mask,'n1',(B,N,D),sigma,None,generator,True,c['r1']); ces.append(ce)
    (gf1,gbf1),ce=probe_token(model,x0,t,mask,'f1',(B,N,FF),sigma,c['y1'],generator); ces.append(ce)
    (gf2,gbf2),ce=probe_token(model,x0,t,mask,'f2',(B,N,D),sigma,c['h'],generator); ces.append(ce)
    (gn2,gbn2),ce=probe_token(model,x0,t,mask,'n2',(B,N,D),sigma,None,generator,True,c['r2']); ces.append(ce)
    pairs=[
      ('blocks.3.attn.WQ.weight',gq),('blocks.3.attn.WQ.bias',gbq),
      ('blocks.3.attn.WK.weight',gk),('blocks.3.attn.WK.bias',gbk),
      ('blocks.3.attn.WV.weight',gv),('blocks.3.attn.WV.bias',gbv),
      ('blocks.3.attn.WO.weight',go),('blocks.3.attn.WO.bias',gbo),
      ('blocks.3.n1.weight',gn1),('blocks.3.n1.bias',gbn1),
      ('blocks.3.f1.weight',gf1),('blocks.3.f1.bias',gbf1),
      ('blocks.3.f2.weight',gf2),('blocks.3.f2.bias',gbf2),
      ('blocks.3.n2.weight',gn2),('blocks.3.n2.bias',gbn2)]
    params=target_params(model)
    for name,g in pairs:
        lim=clip
        norms[name]=clip_(g,lim); grads[name]=g
    for name,g in grads.items():
        fast,slow=states[name]
        fast.mul_(rho_fast).add_(g,alpha=1-rho_fast)
        slow.mul_(rho_slow).add_(g,alpha=1-rho_slow)
        u=mix*fast+(1-mix)*slow
        params[name].add_(u,alpha=-lr[name])
    return float(np.mean(ces)),norms


def make_lr(params,a):
    lr={}
    for k in params:
        if '.WQ.' in k or '.WK.' in k or '.WV.' in k: lr[k]=a.lr_qkv
        elif '.WO.' in k: lr[k]=a.lr_wo
        elif '.f1.' in k: lr[k]=a.lr_f1
        elif '.f2.' in k: lr[k]=a.lr_f2
        else: lr[k]=a.lr_ln
        if k.endswith('.bias'): lr[k]*=0.5
    return lr


def diag_group(params,gold):
    out={}
    for k,p in params.items():
        if k.endswith('.weight'):
            out[k]=r4.weight_diag(p,gold[k])
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--codebook',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--features',type=int,default=256);ap.add_argument('--rf-kind',default='p3norm');ap.add_argument('--rf-seed',type=int,default=29257)
    ap.add_argument('--epochs',type=int,default=6);ap.add_argument('--batch',type=int,default=128);ap.add_argument('--sigma',type=float,default=.05)
    ap.add_argument('--lr-qkv',type=float,default=.025);ap.add_argument('--lr-wo',type=float,default=.05);ap.add_argument('--lr-f1',type=float,default=.07);ap.add_argument('--lr-f2',type=float,default=.10);ap.add_argument('--lr-ln',type=float,default=.02)
    ap.add_argument('--rho-fast',type=float,default=.60);ap.add_argument('--rho-slow',type=float,default=.95);ap.add_argument('--mix-fast',type=float,default=.70);ap.add_argument('--clip',type=float,default=.75)
    ap.add_argument('--threads',type=int,default=2);ap.add_argument('--seed',type=int,default=5601)
    a=ap.parse_args(); random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(a.threads);torch.set_grad_enabled(False);a.out.mkdir(parents=True,exist_ok=True)
    xall,yall,xte0,yte0=b0.load(a.data);x012,y012=b0.subset012(xall,yall);xte,yte=b0.subset012(xte0,yte0);xtr,ytr,xva,yva=b0.split(x012,y012,.1,7);cb=np.load(a.codebook);tr=b0.encode(xtr,cb);va=b0.encode(xva,cb);te=b0.encode(xte,cb)
    exact=b0.Model(128,4,4,512,.05);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);exact.load_state_dict(ck['model']);exact.eval()
    model=e29.HgB0Model(a.features,a.rf_kind,a.rf_seed);e29.transfer_b0(exact,model);model.eval();r4.freeze_all(model)
    params=target_params(model); allowed=set(params); gold={k:p.clone() for k,p in params.items()}; hashes_intact=r4.model_hashes(model)
    ratios=(.25,.5,.9,1.0); intact=r4.eval_mask_ratios(model,te,yte,ratios); clf=r4.external_classifier(xtr,ytr); gen_i=r4.generate_snapshot(model,cb,clf,a.out,'intact',a.seed+10)
    damage_block(model); damaged=r4.eval_mask_ratios(model,te,yte,ratios); gen_d=r4.generate_snapshot(model,cb,clf,a.out,'damaged_full_block',a.seed+10)
    hd=r4.model_hashes(model); changed_damage={n for n in hashes_intact if hashes_intact[n]!=hd[n]}
    if not changed_damage.issubset(allowed): raise RuntimeError(f'outside-block damage: {sorted(changed_damage-allowed)}')
    required={'blocks.3.attn.WQ.weight','blocks.3.attn.WK.weight','blocks.3.attn.WV.weight','blocks.3.attn.WO.weight','blocks.3.f1.weight','blocks.3.f2.weight','blocks.3.n1.weight','blocks.3.n2.weight'}
    if not required.issubset(changed_damage): raise RuntimeError('required block weights were not damaged')
    states={k:(torch.zeros_like(p),torch.zeros_like(p)) for k,p in params.items()}; lr=make_lr(params,a);rng=np.random.default_rng(a.seed+77);tg=torch.Generator();tg.manual_seed(a.seed+99)
    hist=[];best=float('inf');best_state={k:p.clone() for k,p in params.items()}
    for ep in range(1,a.epochs+1):
        order=rng.permutation(len(tr)); losses=[]; qn=[]; kn=[]; vn=[]; on=[]; f1n=[]; f2n=[]
        for s in range(0,len(order),a.batch):
            idx=order[s:s+a.batch];t,y,inp,mask=r4.masked_random_batch(tr,ytr,idx);loss,norm=block_step(model,t,y,inp,mask,states,lr,a.sigma,a.rho_fast,a.rho_slow,a.mix_fast,a.clip,tg);losses.append(loss)
            qn.append(norm['blocks.3.attn.WQ.weight']);kn.append(norm['blocks.3.attn.WK.weight']);vn.append(norm['blocks.3.attn.WV.weight']);on.append(norm['blocks.3.attn.WO.weight']);f1n.append(norm['blocks.3.f1.weight']);f2n.append(norm['blocks.3.f2.weight'])
        val=r4.eval_mask_ratios(model,va,yva,(.5,),batch=256)['0.5']['ce'];test=r4.eval_mask_ratios(model,te,yte,(.5,),batch=256,limit=1024)['0.5']['ce']
        rec={'epoch':ep,'mean_probe_ce':float(np.mean(losses)),'val_ce_50':val,'test_ce_50_first1024':test,'gQ':float(np.mean(qn)),'gK':float(np.mean(kn)),'gV':float(np.mean(vn)),'gO':float(np.mean(on)),'gF1':float(np.mean(f1n)),'gF2':float(np.mean(f2n))};hist.append(rec);print(json.dumps(rec),flush=True)
        if val<best: best=val;best_state={k:p.clone() for k,p in params.items()}
    for k,p in params.items(): p.copy_(best_state[k])
    recovered=r4.eval_mask_ratios(model,te,yte,ratios);gen_r=r4.generate_snapshot(model,cb,clf,a.out,'recovered_full_block',a.seed+10)
    hf=r4.model_hashes(model);changed_final={n for n in hashes_intact if hashes_intact[n]!=hf[n]}
    if not changed_final.issubset(allowed): raise RuntimeError(f'outside-block learning: {sorted(changed_final-allowed)}')
    if any(p.grad is not None for p in model.parameters()): raise RuntimeError('autograd tensor exists')
    def frac(r):
        ci=intact[r]['ce'];cd=damaged[r]['ce'];cr=recovered[r]['ce'];return float((cd-cr)/(cd-ci)) if abs(cd-ci)>1e-12 else float('nan')
    gate={'damage_isolated_to_final_block':changed_damage.issubset(allowed),'learning_isolated_to_final_block':changed_final.issubset(allowed),'no_autograd_gradients':all(p.grad is None for p in model.parameters()),'damage_delta_ce_50':damaged['0.5']['ce']-intact['0.5']['ce'],'recovery_fraction_ce_25':frac('0.25'),'recovery_fraction_ce_50':frac('0.5'),'recovery_fraction_ce_90':frac('0.9'),'recovered_better_25':recovered['0.25']['ce']<damaged['0.25']['ce'],'recovered_better_50':recovered['0.5']['ce']<damaged['0.5']['ce'],'recovered_better_90':recovered['0.9']['ce']<damaged['0.9']['ce']}
    gate['pass']=bool(gate['damage_delta_ce_50']>.05 and gate['recovery_fraction_ce_50']>.50 and gate['recovered_better_25'] and gate['recovered_better_50'] and gate['recovered_better_90'] and gate['damage_isolated_to_final_block'] and gate['learning_isolated_to_final_block'] and gate['no_autograd_gradients'])
    res={'stage':'R5B full final H/g block recovery','scope':'all trainable parameters of final H/g block plastic; earlier network fixed','learning':'local antithetic forward perturbation + fast/slow synaptic states','backprop_used':False,'torch_optimizer_used':False,'learned_feedback_network_used':False,'rf_kind':a.rf_kind,'rf_features':a.features,'intact_eval':intact,'damaged_eval':damaged,'recovered_eval':recovered,'generation_intact':gen_i,'generation_damaged':gen_d,'generation_recovered':gen_r,'best_validation_ce_50':best,'gate':gate,'recovered_weight_diag':diag_group(params,gold),'history':hist}
    (a.out/'metrics.json').write_text(json.dumps(res,indent=2));
    with (a.out/'history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=hist[0].keys());w.writeheader();w.writerows(hist)
    torch.save({'final_block':{k:p.clone() for k,p in params.items()},'metrics':res},a.out/'recovery_checkpoint.pt');print('FINAL '+json.dumps(res),flush=True)
if __name__=='__main__':main()
