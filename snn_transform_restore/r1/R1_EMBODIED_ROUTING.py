#!/usr/bin/env python3
from dataclasses import dataclass
import ast, inspect, json, math, textwrap
from pathlib import Path
from typing import List
import torch

OUT=Path(__file__).resolve().parent

@dataclass
class Tree:
    k:int; depth:int; D:int; keys:torch.Tensor; values:torch.Tensor
    parent:torch.Tensor; node_depth:torch.Tensor; children:torch.Tensor
    @property
    def N(self): return self.keys.shape[0]


def build_tree(k=3, depth=6, D=32, V=8, alpha=.75, decay=.70, seed=0):
    if D < 1+k*depth: raise ValueError('D too small')
    g=torch.Generator().manual_seed(seed)
    basis,_=torch.linalg.qr(torch.randn(D,1+k*depth,generator=g),mode='reduced')
    raw=[basis[:,0].clone()]; keys=[raw[0]/raw[0].norm()]; parent=[-1]; dep=[0]; rows={}; level=[0]
    for d in range(depth):
        nxt=[]
        for p in level:
            row=[]
            for b in range(k):
                rc=raw[p]+alpha*(decay**d)*basis[:,1+d*k+b]
                idx=len(keys); raw.append(rc); keys.append(rc/(rc.norm()+1e-12)); parent.append(p); dep.append(d+1); row.append(idx); nxt.append(idx)
            perm=torch.randperm(k,generator=g).tolist(); rows[p]=[row[i] for i in perm]
        level=nxt
    ch=torch.full((len(keys),k),-1,dtype=torch.long)
    for p,row in rows.items(): ch[p]=torch.tensor(row)
    gv=torch.Generator().manual_seed(seed+100003)
    vals=torch.randn(len(keys),V,generator=gv)
    return Tree(k,depth,D,torch.stack(keys),vals,torch.tensor(parent),torch.tensor(dep),ch)


def ref_path(t:Tree,target:int)->List[int]:
    p=[]; cur=target
    while cur>=0: p.append(cur); cur=int(t.parent[cur])
    return p[::-1]

@dataclass
class TCfg:
    tmin:float=0.; tmax:float=20.; tau_s:float=3.; tau_m:float=300.

class Matcher:
    def __init__(self,c=TCfg()): self.c=c
    def latency(self,x): return self.c.tmin+(1-(x.clamp(-1,1)+1)/2)*(self.c.tmax-self.c.tmin)
    def one(self,kt,qt):
        D=len(kt); ev=[(kt[d],0,d) for d in range(D)]+[(qt[d],1,d) for d in range(D)]; ev.sort()
        xk=[0.]*D; xq=[0.]*D; tk=[None]*D; tq=[None]*D; m=0.; tm=ev[0][0] if ev else 0.
        for tt,s,d in ev:
            if tt>tm: m*=math.exp(-(tt-tm)/self.c.tau_m); tm=tt
            if s==0:
                if tq[d] is not None: m+=xq[d]*math.exp(-(tt-tq[d])/self.c.tau_s)
                if tk[d] is not None: xk[d]*=math.exp(-(tt-tk[d])/self.c.tau_s)
                xk[d]+=1.; tk[d]=tt
            else:
                if tk[d] is not None: m+=xk[d]*math.exp(-(tt-tk[d])/self.c.tau_s)
                if tq[d] is not None: xq[d]*=math.exp(-(tt-tq[d])/self.c.tau_s)
                xq[d]+=1.; tq[d]=tt
        return m/max(D,1)
    def responses(self,ck,q,valid):
        B,C,D=ck.shape; kt=self.latency(ck).cpu(); qt=self.latency(q).cpu(); va=valid.cpu(); out=torch.zeros(B,C); cnt=torch.zeros(B,C,dtype=torch.long)
        for b in range(B):
            qq=qt[b].tolist()
            for c in range(C):
                if bool(va[b,c]): out[b,c]=self.one(kt[b,c].tolist(),qq); cnt[b,c]=2*D
        return out,cnt

@dataclass
class Cfg:
    k:int=3; depth:int=6; D:int=32; V:int=8; steps:int=160; dt:float=.1; tau:float=1.; beta:float=1.2; thr:float=.34

class Router:
    def __init__(self,c=Cfg()): self.c=c; self.m=Matcher()
    def competition(self,d,v):
        z=torch.zeros_like(d); vf=v.float()
        for _ in range(self.c.steps):
            inh=self.c.beta*(z.sum(1,keepdim=True)-z); target=torch.relu(d-inh)*vf; z+=self.c.dt/self.c.tau*(-z+target); z*=vf
        return (z>=self.c.thr)&v
    @staticmethod
    def observe(s): return (s.long()*torch.arange(s.shape[1])[None,:]).sum(1)
    def route(self,t:Tree,q):
        B,N,K=len(q),t.N,t.k; q=q/(q.norm(dim=1,keepdim=True)+1e-12)
        valid_edge=t.children>=0; send,slot=torch.nonzero(valid_edge,as_tuple=True); recv=t.children[send,slot]; src=send*K+slot
        P=torch.sparse_coo_tensor(torch.stack([recv,src]),torch.ones(len(src)),(N,N*K)).coalesce()
        node_state=torch.zeros(B,N,dtype=torch.bool); node_state[:,0]=1; stopped=torch.zeros_like(node_state)
        path=torch.full((B,t.depth+1),-1,dtype=torch.long); path[:,0]=0; plen=torch.ones(B,dtype=torch.long); sdepth=torch.full((B,),-1,dtype=torch.long)
        cand_cnt=torch.zeros(B,dtype=torch.long); rel_ev=torch.zeros(B,dtype=torch.long); child_sp=torch.zeros(B,dtype=torch.long); stop_sp=torch.zeros(B,dtype=torch.long); syn_ev=torch.zeros(B,dtype=torch.long)
        for _ in range(t.depth+1):
            active=torch.nonzero(node_state,as_tuple=False)
            if active.numel()==0: break
            if not bool(torch.all((node_state.sum(1)==0)|(node_state.sum(1)==1))): raise RuntimeError('non one-hot packet')
            bi,sender=active[:,0],active[:,1]; children=t.children[sender]
            ids=torch.cat([sender[:,None],children.clamp_min(0)],1); v=torch.cat([torch.ones(len(sender),1,dtype=torch.bool),children>=0],1)
            drives,ec=self.m.responses(t.keys[ids],q[bi],v); hard=self.competition(drives,v)
            if not bool(torch.all(hard.sum(1)==1)): raise RuntimeError('local circuit not unique')
            cand_cnt[bi]+=v.sum(1); rel_ev[bi]+=ec.sum(1)
            stop_evt=torch.zeros(B,N,dtype=torch.bool); stop_evt[bi,sender]=hard[:,0]; stopped|=stop_evt; stop_sp+=stop_evt.sum(1)
            ns=stop_evt.any(1)
            if bool(ns.any()): sdepth[ns]=(stop_evt[ns].long()*t.node_depth[None,:]).sum(1)
            sources=torch.zeros(B,N,K); sources[bi,sender,:]=hard[:,1:].float(); child_sp+=sources.sum((1,2)).long()
            flat=sources.reshape(B,N*K); next_state=(torch.sparse.mm(P,flat.T).T>.5); syn_ev+=flat.sum(1).long()
            if not bool(torch.all((next_state.sum(1)==0)|(next_state.sum(1)==1))): raise RuntimeError('projection branched')
            moving=next_state.sum(1)==1
            if bool(moving.any()):
                rows=torch.nonzero(moving).squeeze(1); obs=self.observe(next_state[moving]); pos=plen[rows]; path[rows,pos]=obs; plen[rows]+=1
            node_state=next_state
        if bool(node_state.any()) or not bool(torch.all(stopped.sum(1)==1)): raise RuntimeError('termination failure')
        final=self.observe(stopped); vals=(stopped.float().unsqueeze(-1)*t.values.unsqueeze(0)).sum(1)
        return final,vals,dict(path=path,plen=plen,sdepth=sdepth,cand=cand_cnt,rel_ev=rel_ev,child_sp=child_sp,stop_sp=stop_sp,syn_ev=syn_ev,synapses=len(src))


def source_audit():
    s=textwrap.dedent(inspect.getsource(Router.route)); tr=ast.parse(s); names={n.id for n in ast.walk(tr) if isinstance(n,ast.Name)}; attrs={n.attr for n in ast.walk(tr) if isinstance(n,ast.Attribute)}
    return dict(current='current' in names,gather='gather' in attrs,next_node='next_node' in names,argmax='argmax' in attrs,topk='topk' in attrs,softmax='softmax' in attrs,sparse_mm='torch.sparse.mm' in s,node_state='node_state' in names)


def validate(c,seed,depth,noise=0.):
    t=build_tree(c.k,depth,c.D,c.V,seed=seed); q=t.keys.clone()
    if noise:
        g=torch.Generator().manual_seed(90000+seed+int(noise*1e6)); q+=noise*torch.randn(q.shape,generator=g)
    pred,val,a=Router(c).route(t,q); targets=torch.arange(t.N)
    ok=0
    for i in range(t.N): ok+=int(a['path'][i,:int(a['plen'][i])].tolist()==ref_path(t,i))
    return dict(depth=depth,seed=seed,N=t.N,synapses=a['synapses'],node_acc=float((pred==targets).float().mean()),stop_acc=float((a['sdepth']==t.node_depth).float().mean()),path_acc=ok/t.N,value_err=float((val-t.values).abs().max()),child_depth=bool(torch.equal(a['child_sp'],t.node_depth)),one_stop=bool(torch.all(a['stop_sp']==1)),syn_eq_child=bool(torch.equal(a['syn_ev'],a['child_sp'])),mean_child=float(a['child_sp'].float().mean()),mean_rel_events=float(a['rel_ev'].float().mean()),noise=noise)


def main():
    c=Cfg(); au=source_audit(); source_pass=(not au['current'] and not au['gather'] and not au['next_node'] and not au['argmax'] and not au['topk'] and not au['softmax'] and au['sparse_mm'] and au['node_state'])
    depths=[validate(c,11,d) for d in range(1,7)]; unseen=[validate(c,s,5) for s in [21,22,23,24,25]]; noisy=[validate(c,31,5,n) for n in [0,.001,.003,.005,.01]]
    func=all(r['node_acc']==r['stop_acc']==r['path_acc']==1 and r['value_err']==0 and r['child_depth'] and r['one_stop'] and r['syn_eq_child'] for r in depths)
    un=all(r['path_acc']==1 and r['syn_eq_child'] for r in unseen); no=all(r['path_acc']==1 for r in noisy); status='PASS' if source_pass and func and un and no else 'FAIL'
    res=dict(stage='R1_EMBODIED_EVENT_PROPAGATION',status=status,source_audit=au,depth_validation=depths,unseen_validation=unseen,noise_validation=noisy,claim_boundary={'closed':'Python selected-child frontier traversal removed; child route spikes propagate only through fixed sparse synaptic projection','still_open':['SpiNNaker timing/energy equivalence','graded multi-Value aggregation','learned representation/topology','generic non-hierarchical benchmark']})
    (OUT/'R1_RESULTS.json').write_text(json.dumps(res,indent=2)); print(json.dumps({'status':status,'source_pass':source_pass,'functional_pass':func,'unseen_pass':un,'noise_pass':no},indent=2))
    if status!='PASS': raise SystemExit(2)

if __name__=='__main__': main()
