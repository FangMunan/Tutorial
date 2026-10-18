#!/usr/bin/env python3
"""R6B.1 sensory-tokenizer gate.

Goal: replace the hand-engineered 4x4-patch KMeans front-end with a neural
28x28 -> 7x7 encoder before returning to SNN-Transform/MSDL.

The SNN candidate follows the architecture used by Spiking-Diffusion's VQ-SVAE:
28x28 -> stride-2 spiking conv -> 14x14 -> stride-2 spiking conv -> 7x7,
then a low-dimensional spiking latent map and vector quantization.  We add an
explicit 28x28 sensory LIF sheet so raw pixels first drive neurons rather than
being converted to categorical IDs by preprocessing.

This stage deliberately uses surrogate-gradient training only to test whether
the proposed sensory representation retains enough visual information.  No
SNN-Transform, MSDL or image generation is used here.  Advancement is based on
MaskGIT-tokenizer-style reconstruction metrics against a same-capacity ANN-VQ
reference and the historical KMeans tokenizer.
"""
from __future__ import annotations

import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import MiniBatchKMeans
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

K = 64
LATENT_HW = 7
CLASSES = (0, 1, 2)


def seed_all(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def load_mnist(path: Path):
    d = np.load(path, allow_pickle=False)
    return (d['x_train'].astype(np.float32)/255., d['y_train'].astype(np.int64),
            d['x_test'].astype(np.float32)/255., d['y_test'].astype(np.int64))


def balanced_subset(x, y, n_per_class, seed):
    rng = np.random.default_rng(seed); ids=[]
    for c in CLASSES:
        z=np.where(y==c)[0]; rng.shuffle(z); ids.append(z[:min(n_per_class,len(z))])
    ids=np.concatenate(ids); rng.shuffle(ids); return x[ids], y[ids]


def train_val_split(x,y,val_frac,seed):
    rng=np.random.default_rng(seed); tr=[]; va=[]
    for c in CLASSES:
        z=np.where(y==c)[0]; rng.shuffle(z); nv=max(1,int(round(val_frac*len(z))))
        va.append(z[:nv]); tr.append(z[nv:])
    tr=np.concatenate(tr); va=np.concatenate(va); rng.shuffle(tr); rng.shuffle(va)
    return x[tr],y[tr],x[va],y[va]


class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x); return (x > 0).to(x.dtype)
    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        # smooth fast-sigmoid surrogate derivative
        return grad / (1.0 + 4.0*x.abs()).pow(2)


def spike(x): return SpikeFn.apply(x)


def lif_step(current, mem, beta=0.85, threshold=1.0):
    mem = beta*mem + current
    spk = spike(mem-threshold)
    mem = mem - spk*threshold
    return spk, mem


class VectorQuantizer(nn.Module):
    def __init__(self, dim=16, k=K, commitment=0.25, entropy_weight=0.02, entropy_temp=0.5):
        super().__init__(); self.dim=dim; self.k=k; self.commitment=commitment
        self.entropy_weight=entropy_weight; self.entropy_temp=entropy_temp
        self.codebook=nn.Embedding(k,dim); nn.init.uniform_(self.codebook.weight,-1/k,1/k)
    def forward(self,z):
        # z: B,C,7,7
        flat=z.permute(0,2,3,1).contiguous().view(-1,self.dim)
        emb=self.codebook.weight
        dist=flat.square().sum(1,keepdim=True)+emb.square().sum(1)[None]-2*flat@emb.t()
        ids=dist.argmin(1)
        q=F.embedding(ids,emb)
        qloss=F.mse_loss(q,flat.detach()) + self.commitment*F.mse_loss(flat,q.detach())
        soft=(-dist/self.entropy_temp).softmax(1); p=soft.mean(0)
        entropy=-(p*(p+1e-12).log()).sum()/math.log(self.k)
        usage_loss=self.entropy_weight*(1.0-entropy)
        qst=flat+(q-flat).detach()
        qst=qst.view(z.shape[0],LATENT_HW,LATENT_HW,self.dim).permute(0,3,1,2).contiguous()
        return qst, ids.view(z.shape[0],LATENT_HW,LATENT_HW), qloss+usage_loss, float(entropy.detach())


class Decoder(nn.Module):
    def __init__(self,dim=16):
        super().__init__()
        self.net=nn.Sequential(
            nn.ConvTranspose2d(dim,64,4,2,1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32,1,3,padding=1), nn.Sigmoid())
    def forward(self,z): return self.net(z)


class ANNVQ(nn.Module):
    def __init__(self,dim=16,**vqkw):
        super().__init__(); self.kind='ann'
        self.enc=nn.Sequential(
            nn.Conv2d(1,32,3,2,1,bias=False),nn.BatchNorm2d(32),nn.SiLU(),
            nn.Conv2d(32,64,3,2,1,bias=False),nn.BatchNorm2d(64),nn.SiLU(),
            nn.Conv2d(64,dim,1,bias=False))
        self.vq=VectorQuantizer(dim=dim,**vqkw); self.dec=Decoder(dim)
    def forward(self,x):
        z=self.enc(x); q,ids,vql,ent=self.vq(z); r=self.dec(q)
        return r,ids,vql,ent,{}


class SNNVQ(nn.Module):
    def __init__(self,dim=16,steps=8,alpha=0.5,sensory_gain=1.6,beta=0.85,**vqkw):
        super().__init__(); self.kind='snn'; self.steps=steps; self.alpha=alpha
        self.sensory_gain=sensory_gain; self.beta=beta
        self.c1=nn.Conv2d(1,32,3,2,1,bias=False); self.b1=nn.BatchNorm2d(32)
        self.c2=nn.Conv2d(32,64,3,2,1,bias=False); self.b2=nn.BatchNorm2d(64)
        self.c3=nn.Conv2d(64,dim,1,bias=False); self.b3=nn.BatchNorm2d(dim)
        self.vq=VectorQuantizer(dim=dim,**vqkw); self.dec=Decoder(dim)
    def forward(self,x):
        ms=torch.zeros_like(x); m1=m2=m3=None
        ss=s1=s2=s3=0.0
        for _ in range(self.steps):
            sp0,ms=lif_step(self.sensory_gain*x,ms,self.beta)
            cur1=self.b1(self.c1(sp0)); m1=torch.zeros_like(cur1) if m1 is None else m1
            sp1,m1=lif_step(cur1,m1,self.beta)
            cur2=self.b2(self.c2(sp1)); m2=torch.zeros_like(cur2) if m2 is None else m2
            sp2,m2=lif_step(cur2,m2,self.beta)
            cur3=self.b3(self.c3(sp2)); m3=torch.zeros_like(cur3) if m3 is None else m3
            sp3,m3=lif_step(cur3,m3,self.beta)
            ss=ss+sp0; s1=s1+sp1; s2=s2+sp2; s3=s3+sp3
        rate0=ss/self.steps; rate1=s1/self.steps; rate2=s2/self.steps; rate3=s3/self.steps
        # Spiking-Diffusion-style mixed firing-rate + membrane readout.
        z=self.alpha*rate3 + (1.0-self.alpha)*torch.tanh(m3)
        q,ids,vql,ent=self.vq(z); r=self.dec(q)
        st={'sensory_rate':float(rate0.detach().mean()),'h1_rate':float(rate1.detach().mean()),
            'h2_rate':float(rate2.detach().mean()),'latent_rate':float(rate3.detach().mean())}
        return r,ids,vql,ent,st


def make_model(kind,a):
    kw=dict(dim=a.dim,commitment=a.commitment,entropy_weight=a.entropy_weight,entropy_temp=a.entropy_temp)
    if kind=='ann': return ANNVQ(**kw)
    if kind=='snn': return SNNVQ(steps=a.steps,alpha=a.alpha,sensory_gain=a.sensory_gain,beta=a.beta,**kw)
    raise ValueError(kind)


def patchify(x):
    return x.reshape(len(x),7,4,7,4).transpose(0,1,3,2,4).reshape(-1,16)


def kmeans_reconstruct(xtr,xte,seed):
    km=MiniBatchKMeans(n_clusters=K,batch_size=4096,n_init=5,max_iter=150,random_state=seed,reassignment_ratio=.005)
    km.fit(patchify(xtr)); p=patchify(xte); ids=km.predict(p)
    rec=km.cluster_centers_[ids].reshape(len(xte),7,7,4,4).transpose(0,1,3,2,4).reshape(len(xte),28,28)
    return np.clip(rec,0,1).astype(np.float32),ids.reshape(len(xte),7,7)


class RefCNN(nn.Module):
    def __init__(self):
        super().__init__(); self.c1=nn.Conv2d(1,32,3,padding=1); self.c2=nn.Conv2d(32,64,3,padding=1)
        self.fc1=nn.Linear(64*7*7,64); self.fc2=nn.Linear(64,3)
    def forward(self,x,feat=False):
        x=F.max_pool2d(F.relu(self.c1(x)),2); x=F.max_pool2d(F.relu(self.c2(x)),2)
        f=F.relu(self.fc1(x.flatten(1))); z=self.fc2(f); return (z,f) if feat else z


def train_ref(x,y,device,epochs=4):
    m=RefCNN().to(device); opt=torch.optim.Adam(m.parameters(),1e-3)
    ld=DataLoader(TensorDataset(torch.from_numpy(x[:,None]),torch.from_numpy(y)),batch_size=128,shuffle=True)
    for _ in range(epochs):
        m.train()
        for xb,yb in ld:
            xb=xb.to(device); yb=yb.to(device); z=m(xb); loss=F.cross_entropy(z,yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def ref_features(m,x,device):
    feats=[]; pred=[]; xx=torch.from_numpy(x[:,None]).float()
    for i in range(0,len(xx),256):
        z,f=m(xx[i:i+256].to(device),True); feats.append(f.cpu().numpy()); pred.append(z.argmax(1).cpu().numpy())
    return np.concatenate(feats),np.concatenate(pred)


def feature_fid(a,b):
    ma,mb=a.mean(0),b.mean(0); ca=np.cov(a,rowvar=False); cb=np.cov(b,rowvar=False)
    c=sqrtm(ca@cb); c=np.real(c) if np.iscomplexobj(c) else c
    return float(np.sum((ma-mb)**2)+np.trace(ca+cb-2*c))


def code_stats(ids):
    cnt=np.bincount(np.asarray(ids).reshape(-1),minlength=K).astype(np.float64); p=cnt/cnt.sum(); nz=p[p>0]
    return {'codebook_usage':float((cnt>0).mean()),'active_codes':int((cnt>0).sum()),
            'perplexity':float(np.exp(-(nz*np.log(nz)).sum())),'top1_token_fraction':float(p.max())}


def recon_metrics(orig,rec,ids,ref,device,y,spike_stats=None):
    mse=float(np.mean((orig-rec)**2)); psnr=float(10*np.log10(1/max(mse,1e-12)))
    ssim=float(np.mean([structural_similarity(a,b,data_range=1.0) for a,b in zip(orig,rec)]))
    fo,po=ref_features(ref,orig,device); fr,pr=ref_features(ref,rec,device)
    out={'mse':mse,'psnr':psnr,'ssim':ssim,'feature_rfid':feature_fid(fo,fr),
         'classifier_acc_original':float((po==y).mean()),'classifier_acc_reconstruction':float((pr==y).mean()),
         'classifier_consistency':float((po==pr).mean())}
    out.update(code_stats(ids))
    if spike_stats: out['spike_activity']=spike_stats
    return out


def save_grid(orig,rec,path,n=30):
    n=min(n,len(orig)); fig,ax=plt.subplots(2,n,figsize=(n*.7,1.6))
    for j in range(n):
        for r,img in enumerate((orig[j],rec[j])):
            ax[r,j].imshow(img,cmap='gray',vmin=0,vmax=1,interpolation='nearest'); ax[r,j].axis('off')
    ax[0,0].set_ylabel('real'); ax[1,0].set_ylabel('recon'); plt.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--model',choices=('kmeans','ann','snn'),required=True)
    ap.add_argument('--n-per-class',type=int,default=1200); ap.add_argument('--test-per-class',type=int,default=400)
    ap.add_argument('--epochs',type=int,default=12); ap.add_argument('--batch',type=int,default=96); ap.add_argument('--lr',type=float,default=2e-3)
    ap.add_argument('--dim',type=int,default=16); ap.add_argument('--steps',type=int,default=8); ap.add_argument('--alpha',type=float,default=.5)
    ap.add_argument('--sensory-gain',type=float,default=1.6); ap.add_argument('--beta',type=float,default=.85)
    ap.add_argument('--commitment',type=float,default=.25); ap.add_argument('--entropy-weight',type=float,default=.02); ap.add_argument('--entropy-temp',type=float,default=.5)
    ap.add_argument('--threads',type=int,default=4); ap.add_argument('--seed',type=int,default=6201)
    a=ap.parse_args(); seed_all(a.seed); torch.set_num_threads(a.threads); a.out.mkdir(parents=True,exist_ok=True); device=torch.device('cpu')

    xall,yall,xte0,yte0=load_mnist(a.data); m=np.isin(yall,CLASSES); xall,yall=xall[m],yall[m]; mt=np.isin(yte0,CLASSES); xte0,yte0=xte0[mt],yte0[mt]
    xsub,ysub=balanced_subset(xall,yall,a.n_per_class,a.seed); xtr,ytr,xva,yva=train_val_split(xsub,ysub,.1,a.seed+1)
    xte,yte=balanced_subset(xte0,yte0,a.test_per_class,a.seed+2)
    ref=train_ref(xtr,ytr,device,4)
    t0=time.time()

    if a.model=='kmeans':
        rec,ids=kmeans_reconstruct(xtr,xte,a.seed); metrics=recon_metrics(xte,rec,ids,ref,device,yte)
        result={'stage':'R6B.1 sensory tokenizer gate','model':'kmeans','architecture':'historical 4x4 raw-patch KMeans -> 7x7 token grid',
                'train_n':len(xtr),'test_n':len(xte),'metrics':metrics,'runtime_s':time.time()-t0}
        save_grid(xte,rec,a.out/'reconstruction.png'); (a.out/'metrics.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True); return

    model=make_model(a.model,a).to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    trld=DataLoader(TensorDataset(torch.from_numpy(xtr[:,None]),torch.from_numpy(ytr)),batch_size=a.batch,shuffle=True)
    vald=DataLoader(TensorDataset(torch.from_numpy(xva[:,None]),torch.from_numpy(yva)),batch_size=a.batch)
    best=1e9; best_state=None; hist=[]
    for ep in range(1,a.epochs+1):
        model.train(); tl=[]
        for xb,_ in trld:
            xb=xb.to(device); rec,ids,vql,ent,st=model(xb); loss=F.mse_loss(rec,xb)+vql
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step(); tl.append(float(loss.detach()))
        model.eval(); vl=[]
        with torch.no_grad():
            for xb,_ in vald:
                xb=xb.to(device); rec,ids,vql,ent,st=model(xb); vl.append(float(F.mse_loss(rec,xb)))
        v=float(np.mean(vl)); hist.append({'epoch':ep,'train_loss':float(np.mean(tl)),'val_mse':v})
        if v<best: best=v; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if ep==1 or ep%3==0: print(json.dumps(hist[-1]),flush=True)
    model.load_state_dict(best_state); model.eval(); rs=[]; ids_all=[]; sts=[]
    with torch.no_grad():
        for i in range(0,len(xte),a.batch):
            xb=torch.from_numpy(xte[i:i+a.batch,None]).to(device); rec,ids,vql,ent,st=model(xb); rs.append(rec[:,0].cpu().numpy()); ids_all.append(ids.cpu().numpy());
            if st: sts.append(st)
    rec=np.concatenate(rs); ids=np.concatenate(ids_all)
    spike_stats=None
    if sts:
        spike_stats={k:float(np.mean([s[k] for s in sts])) for k in sts[0]}
    metrics=recon_metrics(xte,rec,ids,ref,device,yte,spike_stats)
    result={'stage':'R6B.1 sensory tokenizer gate','model':a.model,
            'architecture':('28x28 sensory LIF -> 32x14x14 LIF -> 64x7x7 LIF -> 16x7x7 latent LIF -> VQ64' if a.model=='snn' else 'ANN conv VQ reference: 28x28 -> 14x14 -> 7x7 -> VQ64'),
            'train_n':len(xtr),'val_n':len(xva),'test_n':len(xte),'epochs':a.epochs,'best_val_mse':best,
            'config':vars(a)|{'data':str(a.data),'out':str(a.out)},'metrics':metrics,'history':hist,'runtime_s':time.time()-t0,
            'scientific_boundary':'tokenizer-only test; no SNN-Transform, no MSDL, no generation'}
    save_grid(xte,rec,a.out/'reconstruction.png'); torch.save({'model':best_state,'metrics':result},a.out/'tokenizer.pt'); (a.out/'metrics.json').write_text(json.dumps(result,indent=2)); print('FINAL '+json.dumps(result),flush=True)

if __name__=='__main__': main()
