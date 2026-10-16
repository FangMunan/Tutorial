from __future__ import annotations
import argparse,json,math,random,time,urllib.request
from pathlib import Path
import numpy as np
import torch,torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from model import *
from msdl import Trainer
URL='https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)
def ensure(p):
    p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():urllib.request.urlretrieve(URL,p)
def load(p):
    d=np.load(p,allow_pickle=False);return d['x_train'].astype(np.float32)/255.,d['y_train'].astype(np.int64),d['x_test'].astype(np.float32)/255.,d['y_test'].astype(np.int64)
def sub(x,y):m=(y==0)|(y==1);return x[m],y[m]
def split(x,y,vf,s):
    r=np.random.default_rng(s);tr=[];va=[]
    for c in (0,1):
        z=np.where(y==c)[0];r.shuffle(z);n=max(1,int(round(vf*len(z))));va.append(z[:n]);tr.append(z[n:])
    tr=np.concatenate(tr);va=np.concatenate(va);r.shuffle(tr);r.shuffle(va);return x[tr],y[tr],x[va],y[va]
def codebook(x,p,s):
    if p.exists():return np.load(p)
    km=MiniBatchKMeans(n_clusters=K,batch_size=8192,n_init=5,max_iter=200,random_state=s,reassignment_ratio=.005);km.fit(image_patches(x));c=km.cluster_centers_.astype(np.float32);np.save(p,c);return c
def batches(n,b,r):
    z=np.arange(n);r.shuffle(z)
    for i in range(0,n,b):yield z[i:i+b]
def marginal_baseline(tok):
    counts=np.bincount(np.asarray(tok).reshape(-1),minlength=K).astype(np.float64);p=counts/counts.sum();nz=p>0
    ce=float(-(counts[nz]*np.log(p[nz])).sum()/counts.sum());acc=float(p.max())
    return dict(ce=ce,token_accuracy=acc,probs=p.tolist())
@torch.no_grad()
def ratio_eval(m,tok,dev,ratios=(.25,.5,.75,.9,1.),batch=256):
    m.eval();T=torch.from_numpy(tok).long();out={}
    for rr in ratios:
        ls=ac=n=0.;nm=max(1,min(L,int(round(rr*L))))
        for i in range(0,len(T),batch):
            t=T[i:i+batch].to(dev);B=len(t);g=torch.Generator(device=dev);g.manual_seed(12345+i+int(rr*1000));s=torch.rand(B,L,device=dev,generator=g);ma=s.argsort(1).argsort(1)<nm;inp=t.clone();inp[ma]=MASK;z=m(inp);bs=B;ls+=F.cross_entropy(z[ma],t[ma]).item()*bs;ac+=(z[ma].argmax(-1)==t[ma]).float().mean().item()*bs;n+=bs
        out[str(rr)]=dict(ce=ls/n,masked_token_acc=ac/n)
    return out
def grid(im,pred,path,n=100):
    n=min(n,len(im));cols=10;rows=math.ceil(n/cols);fig,ax=plt.subplots(rows,cols,figsize=(cols,rows));ax=np.asarray(ax).reshape(-1)
    for i,a in enumerate(ax):
        a.axis('off')
        if i<n:a.imshow(im[i],cmap='gray',vmin=0,vmax=1,interpolation='nearest');a.set_title(str(int(pred[i])),fontsize=7)
    plt.tight_layout();fig.savefig(path,dpi=170,bbox_inches='tight');plt.close(fig)
def nnrmse(g,r,chunk=64):
    g=g.reshape(len(g),-1).astype(np.float32);r=r.reshape(len(r),-1).astype(np.float32);o=[]
    for i in range(0,len(g),chunk):d=((g[i:i+chunk,None]-r[None])**2).mean(-1);o.extend(np.sqrt(d.min(1)).tolist())
    return np.asarray(o)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('data/mnist.npz'));ap.add_argument('--out',type=Path,default=Path('run_e27'));ap.add_argument('--epochs',type=int,default=20);ap.add_argument('--batch',type=int,default=32);ap.add_argument('--seed',type=int,default=27101);ap.add_argument('--threads',type=int,default=2);ap.add_argument('--lr-scale',type=float,default=1.);ap.add_argument('--qk-start-epoch',type=int,default=2);ap.add_argument('--qk-groups',type=int,default=4);ap.add_argument('--eval-every',type=int,default=2);ap.add_argument('--wall-minutes',type=float,default=55.);ap.add_argument('--max-train-images',type=int,default=0);ap.add_argument('--generate',type=int,default=100);a=ap.parse_args()
    seed_all(a.seed);torch.set_num_threads(a.threads);dev=torch.device('cpu');a.out.mkdir(parents=True,exist_ok=True);ensure(a.data);x,y,xt,yt=load(a.data);x,y=sub(x,y);xt,yt=sub(xt,yt);xtr,ytr,xv,yv=split(x,y,.1,a.seed)
    if a.max_train_images and a.max_train_images<len(xtr):n=a.max_train_images//2;ii=np.r_[np.where(ytr==0)[0][:n],np.where(ytr==1)[0][:n]];xtr,ytr=xtr[ii],ytr[ii]
    cp=a.out/'codebook.npy';c=codebook(xtr,cp,a.seed);tr,va,te=encode(xtr,c),encode(xv,c),encode(xt,c);re=decode(te,c);tm=float(np.mean((re-xt)**2));tp=float(10*np.log10(1/max(tm,1e-12)));mb=marginal_baseline(tr)
    m=Model(a.seed).to(dev);T=Trainer(m,dev,a.seed,a.lr_scale,a.qk_groups);hist=[];r=np.random.default_rng(a.seed+1000);w0=time.time();stop=False;initial=ratio_eval(m,va,dev,(.5,.9,1.))
    setup=dict(stage='P4-E27',train=len(tr),val=len(va),test=len(te),params=sum(p.numel() for p in m.parameters()),tokenizer_mse=tm,tokenizer_psnr_db=tp,marginal_baseline=mb,initial_val_ratio=initial,candidate_backprop=False,labels_used_by_candidate=False,labels_used_only_for_benchmark_partition_and_external_evaluator=True)
    print(json.dumps(setup),flush=True)
    for ep in range(1,a.epochs+1):
        ce=[];ac=[];mr=[];t0=time.time();qk=ep>=a.qk_start_epoch
        for ids in batches(len(tr),a.batch,r):
            z=T.batch(torch.from_numpy(tr[ids]).long().to(dev),qk);ce.append(z['ce']);ac.append(z['acc']);mr.append(z['mask'])
            if (time.time()-w0)/60>=a.wall_minutes:stop=True;break
        rec=dict(epoch=ep,train_ce=float(np.mean(ce)),train_acc=float(np.mean(ac)),mask_ratio=float(np.mean(mr)),qk_enabled=qk,epoch_s=time.time()-t0,elapsed_s=time.time()-w0)
        if ep==1 or ep%a.eval_every==0 or stop:rec['val_ratio']=ratio_eval(m,va,dev,(.5,.9,1.))
        hist.append(rec);print(json.dumps(rec),flush=True);torch.save(dict(model=m.state_dict(),trainer=T.cpu(),epoch=ep,history=hist,setup=setup),a.out/'latest.pt')
        if stop:break
    er=ratio_eval(m,te,dev);gt=generate(m,a.generate,dev,16,0.,a.seed+7000);gi=decode(gt.cpu().numpy(),c);clf=LogisticRegression(max_iter=200,solver='liblinear',random_state=7).fit(xtr.reshape(len(xtr),-1),ytr);pred=clf.predict(gi.reshape(len(gi),-1));conf=clf.predict_proba(gi.reshape(len(gi),-1)).max(1);nr=nnrmse(gi,xt);grid(gi,pred,a.out/'generation.png',a.generate);np.savez_compressed(a.out/'generation.npz',tokens=gt.cpu().numpy(),images=gi,pred=pred,confidence=conf,nn_rmse=nr)
    met=dict(stage='P4-E27 MSDL B0-faithful dynamical MaskGIT',epochs_completed=hist[-1]['epoch'],wall_s=time.time()-w0,stopped_by_wall_budget=stop,candidate_backprop=False,labels_used_by_candidate=False,labels_used_only_for_benchmark_partition_and_external_evaluator=True,train_samples=len(tr),test_samples=len(te),tokenizer_mse=tm,tokenizer_psnr_db=tp,marginal_baseline=mb,initial_val_ratio=initial,ratio_eval=er,generation_n=len(gi),generation_pred_counts=np.bincount(pred,minlength=2).tolist(),generation_mean_eval_confidence=float(conf.mean()),generation_nn_rmse_mean=float(nr.mean()),generation_nn_rmse_median=float(np.median(nr)),unique_token_sequences=int(np.unique(gt.cpu().numpy(),axis=0).shape[0]),history_tail=hist[-5:]);(a.out/'metrics.json').write_text(json.dumps(met,indent=2));print('FINAL '+json.dumps(met),flush=True)
if __name__=='__main__':main()
