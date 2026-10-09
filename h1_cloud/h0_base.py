#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H0 from-scratch reference: MaskGIT + normalized linear attention + persistent selective state.

Design goals
------------
1. Rebuild the frozen MNIST-0/1/2 MaskGIT baseline from raw MNIST.
2. Train a matched Linear-MaskGIT backbone from scratch.
3. Verify the exact A=1 persistent-state control in the FIRST linear-attention block.
4. Sweep fixed alpha without retraining.
5. Optionally train a tiny per-head selective retention gate on teacher-forced mask trajectories.

Important architectural boundary
--------------------------------
Persistence is only in block 1. Blocks 2..L are recomputed fresh every MaskGIT round.
This makes A=1 exactly reducible to the fresh linear-attention model because block-1
K/V are functions of the input embedding only (token + position + class), while deeper
representations drift as the mask pattern changes.

The persistent state is historical dynamical memory, not a deep K/V cache.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import MiniBatchKMeans

# -----------------------------
# Global tokenization constants
# -----------------------------
SIDE = 28
PATCH = 4
GRID = SIDE // PATCH
SEQ = GRID * GRID          # 49
CODEBOOK = 64
MASK_ID = CODEBOOK         # 64
VOCAB = CODEBOOK + 1       # + [MASK]
GEN_CLASSES = (0, 1, 2)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def json_dump(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# ============================================================
# 1. Data and tokenizer
# ============================================================

def _normalize_mnist_arrays(xtr, ytr, xte, yte):
    xtr = np.asarray(xtr)
    xte = np.asarray(xte)
    if xtr.ndim == 3:
        pass
    elif xtr.ndim == 4 and xtr.shape[-1] == 1:
        xtr = xtr[..., 0]
        xte = xte[..., 0]
    else:
        raise ValueError(f"Expected MNIST [N,28,28], got {xtr.shape}")
    xtr = xtr.astype(np.float32)
    xte = xte.astype(np.float32)
    if xtr.max() > 1.5:
        xtr /= 255.0
        xte /= 255.0
    return xtr, np.asarray(ytr, np.int64), xte, np.asarray(yte, np.int64)


def load_mnist_npz(path: Path):
    d = np.load(path, allow_pickle=False)
    keys = set(d.files)
    # Keras naming
    if {"x_train", "y_train", "x_test", "y_test"}.issubset(keys):
        return _normalize_mnist_arrays(d["x_train"], d["y_train"], d["x_test"], d["y_test"])
    raise KeyError(f"MNIST npz must contain x_train,y_train,x_test,y_test; found {sorted(keys)}")



def load_mnist_keras():
    """Download/load canonical MNIST. Try torchvision first, then Keras."""
    errors=[]
    try:
        from torchvision.datasets import MNIST
        root=Path.home()/".cache"/"h0_mnist"
        tr=MNIST(root=str(root), train=True, download=True)
        te=MNIST(root=str(root), train=False, download=True)
        xtr=tr.data.numpy(); ytr=tr.targets.numpy()
        xte=te.data.numpy(); yte=te.targets.numpy()
        return _normalize_mnist_arrays(xtr,ytr,xte,yte)
    except Exception as e:
        errors.append("torchvision="+repr(e))
    try:
        from keras.datasets import mnist
        (xtr,ytr),(xte,yte)=mnist.load_data()
        return _normalize_mnist_arrays(xtr,ytr,xte,yte)
    except Exception as e:
        errors.append("keras="+repr(e))
    raise RuntimeError(
        "Automatic MNIST download failed. Supply --data /path/to/mnist.npz. " + "; ".join(errors)
    )

def load_smoke_digits(seed: int = 7):
    """Offline structural smoke only. Not a scientific MNIST result."""
    from sklearn.datasets import load_digits
    x, y = load_digits(return_X_y=False).images, load_digits(return_X_y=False).target
    x = (x.astype(np.float32) / 16.0)[:, None]
    xt = torch.from_numpy(x)
    xt = F.interpolate(xt, size=(28, 28), mode="bilinear", align_corners=False)
    x = xt[:, 0].numpy()
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in range(10):
        ids = np.where(y == c)[0]
        rng.shuffle(ids)
        cut = max(1, int(round(0.8 * len(ids))))
        tr.append(ids[:cut]); te.append(ids[cut:])
    tr = np.concatenate(tr); te = np.concatenate(te)
    rng.shuffle(tr); rng.shuffle(te)
    return x[tr], y[tr].astype(np.int64), x[te], y[te].astype(np.int64)


def filter_classes(x, y, classes=GEN_CLASSES):
    m = np.isin(y, np.asarray(classes))
    return x[m], y[m]


def subset_per_class(x, y, n_per_class: int, seed: int, classes=GEN_CLASSES):
    if n_per_class <= 0:
        return x, y
    rng = np.random.default_rng(seed)
    ids = []
    for c in classes:
        z = np.where(y == c)[0]
        rng.shuffle(z)
        ids.append(z[:min(n_per_class, len(z))])
    ids = np.concatenate(ids)
    rng.shuffle(ids)
    return x[ids], y[ids]


def stratified_split(x, y, val_frac: float, seed: int, classes=GEN_CLASSES):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in classes:
        z = np.where(y == c)[0]
        rng.shuffle(z)
        nv = max(1, int(round(val_frac * len(z))))
        va.append(z[:nv]); tr.append(z[nv:])
    tr = np.concatenate(tr); va = np.concatenate(va)
    rng.shuffle(tr); rng.shuffle(va)
    return x[tr], y[tr], x[va], y[va]


def patches(x: np.ndarray) -> np.ndarray:
    return (x.reshape(len(x), GRID, PATCH, GRID, PATCH)
              .transpose(0, 1, 3, 2, 4)
              .reshape(-1, PATCH * PATCH))


def fit_codebook(x: np.ndarray, path: Path, seed: int = 7, k: int = CODEBOOK):
    km = MiniBatchKMeans(
        n_clusters=k, batch_size=4096, n_init=3, max_iter=100,
        random_state=seed, reassignment_ratio=0.01,
    )
    km.fit(patches(x))
    c = km.cluster_centers_.astype(np.float32)
    np.save(path, c)
    return c


def encode_tokens(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    p = patches(x)
    c2 = (centroids * centroids).sum(1)
    out = []
    for i in range(0, len(p), 200000):
        q = p[i:i+200000]
        d = (q*q).sum(1, keepdims=True) - 2*q @ centroids.T + c2[None]
        out.append(d.argmin(1).astype(np.int64))
    return np.concatenate(out).reshape(len(x), SEQ)


def decode_tokens(tokens: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    q = centroids[np.asarray(tokens, dtype=np.int64)]
    q = (q.reshape(-1, GRID, GRID, PATCH, PATCH)
          .transpose(0, 1, 3, 2, 4)
          .reshape(-1, SIDE, SIDE))
    return np.clip(q, 0.0, 1.0).astype(np.float32)


# ============================================================
# 2. Attention modules
# ============================================================

class PositiveRandomFeatures(nn.Module):
    """Performer/FAVOR+-style positive random features for exp(q^T k / sqrt(d)).

    A fixed Gaussian projection is used. Query stabilization subtracts a per-query
    constant; key stabilization subtracts one constant per batch/head. These scales
    cancel in normalized attention and preserve the kernel ratios relevant here.
    """
    def __init__(self, heads: int, head_dim: int, features: int, seed: int = 17):
        super().__init__()
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        omega = torch.randn(heads, features, head_dim, generator=g)
        # Normalize each random vector to sqrt(d), reducing outlier norms.
        omega = omega / omega.norm(dim=-1, keepdim=True).clamp_min(1e-8) * math.sqrt(head_dim)
        self.register_buffer("omega", omega)
        self.features = features
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor, *, is_query: bool) -> torch.Tensor:
        # x [B,H,N,D]
        x = x * (self.head_dim ** -0.25)
        dash = torch.einsum("bhnd,hmd->bhnm", x, self.omega)
        diag = 0.5 * (x*x).sum(dim=-1, keepdim=True)
        logits = dash - diag
        if is_query:
            # Query-side scalar rescaling cancels exactly in normalized attention.
            shift = logits.amax(dim=-1, keepdim=True).detach()
            stable = (logits - shift).clamp(min=-30.0, max=0.0)
        else:
            # IMPORTANT for H0 persistence: phi(k_i) must depend only on k_i, not on
            # the other tokens present in the current MaskGIT round. A sequence-global
            # max shift would make a cached committed feature drift when masks change.
            # We therefore use only a fixed elementwise safety clamp on keys.
            stable = logits.clamp(min=-30.0, max=30.0)
        return torch.exp(stable) * (self.features ** -0.5) + 1e-6


@dataclass
class PersistentState:
    H: torch.Tensor       # [B,H,M,Dh]
    g: torch.Tensor       # [B,H,M]
    known: torch.Tensor   # [B,N] bool


class SoftmaxAttention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        assert d % heads == 0
        self.d = d; self.heads = heads; self.dh = d // heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def _split(self, x):
        B,N,D = x.shape
        return x.reshape(B,N,self.heads,self.dh).transpose(1,2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._split(self.q(x)); k = self._split(self.k(x)); v = self._split(self.v(x))
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.out(a.transpose(1,2).reshape(x.shape))


class LinearAttention(nn.Module):
    def __init__(self, d: int, heads: int, features: int, feature_seed: int):
        super().__init__()
        assert d % heads == 0
        self.d = d; self.heads = heads; self.dh = d // heads; self.m = features
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.phi = PositiveRandomFeatures(heads, self.dh, features, seed=feature_seed)

    def _split(self, x):
        B,N,D = x.shape
        return x.reshape(B,N,self.heads,self.dh).transpose(1,2)

    def projected(self, x: torch.Tensor):
        q = self._split(self.q(x)); k = self._split(self.k(x)); v = self._split(self.v(x))
        pq = self.phi(q, is_query=True)
        pk = self.phi(k, is_query=False)
        return pq, pk, v

    def fresh(self, x: torch.Tensor) -> torch.Tensor:
        pq, pk, v = self.projected(x)
        H = torch.einsum("bhnm,bhnd->bhmd", pk, v)
        g = pk.sum(dim=2)
        num = torch.einsum("bhnm,bhmd->bhnd", pq, H)
        den = torch.einsum("bhnm,bhm->bhn", pq, g).unsqueeze(-1).clamp_min(1e-8)
        y = num / den
        return self.out(y.transpose(1,2).reshape(x.shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fresh(x)

    def persistent(
        self,
        x: torch.Tensor,
        known: torch.Tensor,
        state: Optional[PersistentState],
        alpha: torch.Tensor,
    ) -> Tuple[torch.Tensor, PersistentState, Dict[str, float]]:
        """Persistent/transient decomposition.

        known: [B,N], True for committed/unmasked tokens in current MaskGIT canvas.
        state contains only previously committed K/V writes. New committed tokens are
        written once. Current masked tokens form a transient state every round.

        alpha: [B,H] or [B,H,1], applied once per MaskGIT iteration before new writes.
        """
        pq, pk, v = self.projected(x)
        B,H,N,M = pk.shape
        if state is None:
            Hp = torch.zeros(B,H,M,self.dh, device=x.device, dtype=x.dtype)
            gp = torch.zeros(B,H,M, device=x.device, dtype=x.dtype)
            prev_known = torch.zeros(B,N, device=x.device, dtype=torch.bool)
        else:
            Hp, gp, prev_known = state.H, state.g, state.known

        a = alpha.reshape(B,H,1,1)
        Hp = Hp * a
        gp = gp * alpha.reshape(B,H,1)

        new_known = known & (~prev_known)
        masked = ~known
        nk = new_known[:,None,:,None].to(pk.dtype)
        mk = masked[:,None,:,None].to(pk.dtype)

        # Sparse-on-semantics write; dense tensor implementation for cloud reference.
        Hp = Hp + torch.einsum("bhnm,bhnd->bhmd", pk * nk, v)
        gp = gp + (pk * nk).sum(dim=2)

        Hm = torch.einsum("bhnm,bhnd->bhmd", pk * mk, v)
        gm = (pk * mk).sum(dim=2)
        Htot = Hp + Hm
        gtot = gp + gm

        num = torch.einsum("bhnm,bhmd->bhnd", pq, Htot)
        den = torch.einsum("bhnm,bhm->bhn", pq, gtot).unsqueeze(-1).clamp_min(1e-8)
        y = num / den
        out = self.out(y.transpose(1,2).reshape(x.shape))
        ns = PersistentState(H=Hp, g=gp, known=known.clone())
        stats = {
            "new_known_mean": float(new_known.sum(1).float().mean().detach().cpu()),
            "masked_mean": float(masked.sum(1).float().mean().detach().cpu()),
            "state_H_rms": float(Hp.square().mean().sqrt().detach().cpu()),
            "state_g_min": float(gp.min().detach().cpu()),
        }
        return out, ns, stats


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, kind: str, features: int, feature_seed: int):
        super().__init__()
        if kind == "softmax":
            self.attn = SoftmaxAttention(d, heads)
        elif kind == "linear":
            self.attn = LinearAttention(d, heads, features, feature_seed)
        else:
            raise ValueError(kind)
        self.kind = kind
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, ff)
        self.f2 = nn.Linear(ff, d)

    def forward(self, x):
        x = self.n1(x + self.attn(x))
        return self.n2(x + self.f2(F.gelu(self.f1(x))))


class RetentionGate(nn.Module):
    """Tiny per-head selective retention gate driven only by mask trajectory observables."""
    def __init__(self, heads: int, init_alpha: float = 0.99):
        super().__init__()
        # alpha = exp(-softplus(s)); inverse approx: s=log(exp(-log(alpha))-1)
        target = max(1e-5, -math.log(init_alpha))
        s0 = math.log(max(1e-8, math.exp(target) - 1.0))
        self.theta0 = nn.Parameter(torch.full((heads,), s0))
        self.theta_mask = nn.Parameter(torch.zeros(heads))
        self.theta_new = nn.Parameter(torch.zeros(heads))

    def forward(self, mask_ratio: torch.Tensor, new_ratio: torch.Tensor) -> torch.Tensor:
        # ratios [B]
        s = (self.theta0[None]
             + mask_ratio[:,None] * self.theta_mask[None]
             + new_ratio[:,None] * self.theta_new[None])
        return torch.exp(-F.softplus(s)).clamp(1e-4, 1.0)


class MaskGIT(nn.Module):
    def __init__(
        self,
        kind: str,
        d: int = 128,
        layers: int = 4,
        heads: int = 4,
        ff: int = 512,
        features: int = 128,
        nclass: int = 3,
        feature_seed: int = 17,
        selective_gate: bool = False,
    ):
        super().__init__()
        self.kind = kind
        self.d = d; self.layers = layers; self.heads = heads; self.features = features
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Parameter(torch.randn(1, SEQ, d) * 0.02)
        self.cls = nn.Embedding(nclass, d)
        self.blocks = nn.ModuleList([
            Block(d, heads, ff, kind, features, feature_seed + 97*i) for i in range(layers)
        ])
        self.dense = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
        self.bias = nn.Parameter(torch.zeros(CODEBOOK))
        self.gate = RetentionGate(heads) if selective_gate else None

    def embed(self, x, y):
        return self.tok(x) + self.pos + self.cls(y)[:,None]

    def logits_from_hidden(self, z):
        z = self.ln(F.gelu(self.dense(z)))
        return F.linear(z, self.tok.weight[:CODEBOOK], self.bias)

    def forward(self, x, y):
        z = self.embed(x, y)
        for b in self.blocks:
            z = b(z)
        return self.logits_from_hidden(z)

    def forward_persistent(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        state: Optional[PersistentState],
        mode: str,
        fixed_alpha: float = 1.0,
    ):
        if self.kind != "linear":
            raise RuntimeError("Persistent H0 is defined only for the linear backbone")
        z = self.embed(x, y)
        known = x.ne(MASK_ID)
        B = x.shape[0]
        if state is None:
            prev_known = torch.zeros_like(known)
        else:
            prev_known = state.known
        new_known = known & (~prev_known)
        mask_ratio = (~known).sum(1).float() / SEQ
        new_ratio = new_known.sum(1).float() / SEQ

        if mode == "identity":
            alpha = torch.ones(B, self.heads, device=x.device, dtype=z.dtype)
        elif mode == "fixed":
            alpha = torch.full((B,self.heads), float(fixed_alpha), device=x.device, dtype=z.dtype)
        elif mode == "selective":
            if self.gate is None:
                raise RuntimeError("Model was created without selective_gate=True")
            alpha = self.gate(mask_ratio, new_ratio)
        else:
            raise ValueError(mode)

        # block 1 persistent; blocks 2..L fresh
        b0 = self.blocks[0]
        attn_out, ns, stats = b0.attn.persistent(z, known, state, alpha)
        z = b0.n1(z + attn_out)
        z = b0.n2(z + b0.f2(F.gelu(b0.f1(z))))
        for b in self.blocks[1:]:
            z = b(z)
        stats["alpha_mean"] = float(alpha.mean().detach().cpu())
        stats["alpha_min"] = float(alpha.min().detach().cpu())
        stats["alpha_max"] = float(alpha.max().detach().cpu())
        return self.logits_from_hidden(z), ns, stats


# ============================================================
# 3. Training
# ============================================================

def random_mask_batch(tokens: torch.Tensor):
    B,N = tokens.shape
    r = torch.rand(B, device=tokens.device)
    ratio = torch.cos(0.5 * math.pi * r)
    mask = torch.rand(B,N,device=tokens.device) < ratio[:,None]
    mask[mask.sum(1)==0, 0] = True
    inp = tokens.clone(); inp[mask] = MASK_ID
    return inp, mask


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    ls = ac = n = 0.0
    for tokens, labels in loader:
        tokens = tokens.to(device); labels = labels.to(device)
        inp, mask = random_mask_batch(tokens)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(inp, labels)
        loss = F.cross_entropy(logits[mask], tokens[mask])
        acc = (logits[mask].argmax(-1) == tokens[mask]).float().mean()
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        bs = len(tokens); ls += loss.item()*bs; ac += acc.item()*bs; n += bs
    return ls/n, ac/n


def train_backbone(model, tr_tokens, tr_y, va_tokens, va_y, outdir: Path,
                   epochs: int, batch: int, device, lr=3e-4):
    outdir.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(tr_tokens), torch.from_numpy(tr_y)),
                              batch_size=batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(va_tokens), torch.from_numpy(va_y)),
                            batch_size=batch, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.96), weight_decay=0.05)
    best = float("inf"); hist=[]; best_epoch=0
    for ep in range(1, epochs+1):
        t0=time.time()
        tr_ce,tr_acc=run_epoch(model,train_loader,device,opt)
        va_ce,va_acc=run_epoch(model,val_loader,device,None)
        row={"epoch":ep,"train_ce":tr_ce,"train_acc":tr_acc,
             "val_ce":va_ce,"val_acc":va_acc,"seconds":time.time()-t0}
        hist.append(row); print(json.dumps(row),flush=True)
        if va_ce < best:
            best=va_ce; best_epoch=ep
            torch.save({"model":model.state_dict(),"epoch":ep,"best":best},outdir/"best.pt")
        torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"epoch":ep,"best":best},outdir/"latest.pt")
        with (outdir/"history.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(hist[0]));w.writeheader();w.writerows(hist)
    ck=torch.load(outdir/"best.pt",map_location=device,weights_only=False)
    model.load_state_dict(ck["model"])
    return {"best_epoch":best_epoch,"best_val_ce":best,"history":hist}


def teacher_mask_sequence(tokens: torch.Tensor, steps: int, generator: torch.Generator):
    """Ground-truth reveal trajectory for gate-only fine-tuning.

    Returns current canvases BEFORE each step. Reveal order is random, but the number of
    remaining masks follows the same cosine schedule family as MaskGIT decoding.
    """
    B,N=tokens.shape
    noise=torch.rand(B,N,device=tokens.device,generator=generator)
    order=noise.argsort(dim=1)
    rank=torch.empty_like(order)
    ar=torch.arange(N,device=tokens.device)[None].expand(B,-1)
    rank.scatter_(1,order,ar)
    canvases=[]
    for st in range(steps):
        frac=st/max(1,steps-1)
        remain=int(round(N*math.cos(0.5*math.pi*frac)))
        reveal=N-remain
        known=rank < reveal
        cur=torch.full_like(tokens,MASK_ID)
        cur[known]=tokens[known]
        canvases.append(cur)
    return canvases


def train_selective_gate(model: MaskGIT, tr_tokens, tr_y, va_tokens, va_y,
                         outdir: Path, device, epochs=2, batch=64, steps=6, lr=1e-2, seed=7):
    """Freeze backbone; train only the tiny retention gate on teacher-forced trajectories."""
    assert model.gate is not None and model.kind=="linear"
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.gate.parameters(): p.requires_grad_(True)
    opt=torch.optim.Adam(model.gate.parameters(),lr=lr)
    loader=DataLoader(TensorDataset(torch.from_numpy(tr_tokens),torch.from_numpy(tr_y)),batch_size=batch,shuffle=True)
    hist=[]
    for ep in range(1,epochs+1):
        model.train(); tot=n=0.0
        for bi,(tok,y) in enumerate(loader):
            tok=tok.to(device);y=y.to(device)
            gen=torch.Generator(device=device);gen.manual_seed(seed+100003*ep+bi)
            canvases=teacher_mask_sequence(tok,steps,gen)
            state=None;loss=0.0;terms=0
            for cur in canvases:
                masked=cur.eq(MASK_ID)
                if not masked.any(): continue
                logits,state,_=model.forward_persistent(cur,y,state,"selective")
                loss=loss+F.cross_entropy(logits[masked],tok[masked]);terms+=1
            loss=loss/max(1,terms)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            tot+=loss.item()*len(tok);n+=len(tok)
        row={"epoch":ep,"train_traj_ce":tot/n,
             "theta0":model.gate.theta0.detach().cpu().tolist(),
             "theta_mask":model.gate.theta_mask.detach().cpu().tolist(),
             "theta_new":model.gate.theta_new.detach().cpu().tolist()}
        hist.append(row);print("GATE",json.dumps(row),flush=True)
    torch.save({"model":model.state_dict(),"gate_only":True},outdir/"selective_gate.pt")
    json_dump(outdir/"selective_gate_history.json",{"history":hist})
    for p in model.parameters(): p.requires_grad_(True)
    return hist


# ============================================================
# 4. Generation and evaluation
# ============================================================

@torch.no_grad()
def generate(model: MaskGIT, labels: torch.Tensor, device, steps=10, temp0=0.25,
             seed=1007, mode="fresh", fixed_alpha=1.0, record=True):
    model.eval(); labels=labels.to(device);B=len(labels)
    cur=torch.full((B,SEQ),MASK_ID,dtype=torch.long,device=device)
    initial=torch.full((B,),SEQ,dtype=torch.long,device=device)
    gen=torch.Generator(device=device);gen.manual_seed(seed)
    state=None
    rec={"pre":[],"proposal":[],"conf":[],"post":[],"mask_before":[],"mask_after":[],"desired":[],"alpha_mean":[],"state_H_rms":[]} if record else None
    for st in range(steps):
        unk=cur.eq(MASK_ID)
        if record:
            rec["pre"].append(cur.cpu().numpy().astype(np.int16));rec["mask_before"].append(unk.cpu().numpy())
        if mode=="fresh":
            logits=model(cur,labels);stats={"alpha_mean":float("nan"),"state_H_rms":float("nan")}
        else:
            logits,state,stats=model.forward_persistent(cur,labels,state,mode,fixed_alpha)
        p=logits.softmax(-1)
        sampled=torch.multinomial(p.reshape(-1,CODEBOOK),1,generator=gen).reshape(B,SEQ)
        prop=torch.where(unk,sampled,cur)
        sp=p.gather(-1,prop[...,None]).squeeze(-1)
        sp_rec=torch.where(unk,sp,torch.ones_like(sp))
        if st==steps-1:
            cur=prop;desired=torch.zeros(B,dtype=torch.long,device=device)
        else:
            score_base=torch.where(unk,sp,torch.full_like(sp,float("inf")))
            frac=(st+1)/steps
            desired=torch.floor(initial.float()*math.cos(0.5*math.pi*frac)).long()
            desired=torch.minimum(desired,torch.clamp(unk.sum(1)-1,min=0))
            u=torch.rand(score_base.shape,device=device,generator=gen).clamp_(1e-6,1-1e-6)
            gum=-torch.log(-torch.log(u))
            score=torch.log(score_base.clamp_min(1e-12))+temp0*(1-frac)*gum
            score=torch.where(torch.isinf(score_base),torch.full_like(score,float("inf")),score)
            new=prop.clone()
            for bi in range(B):
                k=int(desired[bi])
                if k:
                    new[bi,torch.topk(score[bi],k,largest=False).indices]=MASK_ID
            cur=new
        if record:
            rec["proposal"].append(prop.cpu().numpy().astype(np.int16));rec["conf"].append(sp_rec.cpu().numpy().astype(np.float32));rec["desired"].append(desired.cpu().numpy().astype(np.int16));rec["post"].append(cur.cpu().numpy().astype(np.int16));rec["mask_after"].append(cur.eq(MASK_ID).cpu().numpy());rec["alpha_mean"].append(stats["alpha_mean"]);rec["state_H_rms"].append(stats["state_H_rms"])
    if record:
        for k in ("pre","proposal","conf","post","mask_before","mask_after","desired"):
            rec[k]=np.stack(rec[k])
        rec["alpha_mean"]=np.asarray(rec["alpha_mean"],np.float32)
        rec["state_H_rms"]=np.asarray(rec["state_H_rms"],np.float32)
    return cur.cpu().numpy().astype(np.int16),rec


class RefCNN(nn.Module):
    def __init__(self):
        super().__init__();self.c1=nn.Conv2d(1,16,3,padding=1);self.c2=nn.Conv2d(16,32,3,padding=1);self.fc=nn.Linear(32*7*7,10)
    def forward(self,x):
        x=F.max_pool2d(F.relu(self.c1(x)),2);x=F.max_pool2d(F.relu(self.c2(x)),2);return self.fc(x.flatten(1))


def train_ref_classifier(x,y,device,epochs=2,batch=512):
    m=RefCNN().to(device);o=torch.optim.Adam(m.parameters(),1e-3)
    ld=DataLoader(TensorDataset(torch.from_numpy(x[:,None].astype(np.float32)),torch.from_numpy(y.astype(np.int64))),batch_size=batch,shuffle=True)
    for _ in range(epochs):
        m.train()
        for xb,yb in ld:
            xb=xb.to(device);yb=yb.to(device);z=m(xb);loss=F.cross_entropy(z,yb);o.zero_grad();loss.backward();o.step()
    return m

@torch.no_grad()
def classify(ref,x,device,batch=512):
    ref.eval();xx=torch.from_numpy(x[:,None].astype(np.float32));out=[]
    for i in range(0,len(xx),batch):out.append(ref(xx[i:i+batch].to(device)).softmax(-1).cpu())
    return torch.cat(out).numpy()


def morphology(images: np.ndarray):
    try:
        from scipy import ndimage
    except Exception:
        return {}
    comps=[];largest=[];isolated=[]
    for im in images:
        fg=im>0.5
        lab,n=ndimage.label(fg)
        comps.append(float(n))
        if fg.sum()==0:
            largest.append(0.0);isolated.append(0.0);continue
        counts=np.bincount(lab.ravel())[1:]
        largest.append(float(counts.max()/fg.sum()) if len(counts) else 0.0)
        isolated.append(float((counts==1).sum()))
    return {"mean_connected_components":float(np.mean(comps)),"mean_largest_component_fraction_of_foreground":float(np.mean(largest)),"mean_isolated_single_pixel_components":float(np.mean(isolated))}


def eval_generation(name, model, labels_np, centroids, ref, device, outdir, steps, temp, seed, mode="fresh", fixed_alpha=1.0):
    labels=torch.from_numpy(labels_np.astype(np.int64))
    tokens,rec=generate(model,labels,device,steps,temp,seed,mode,fixed_alpha,True)
    images=decode_tokens(tokens,centroids)
    probs=classify(ref,images,device)
    pred=probs.argmax(1)
    counts=np.bincount(pred,minlength=10)
    q=counts[:3]/max(1,counts[:3].sum())
    tvd=.5*float(np.abs(q-np.ones(3)/3).sum())
    met={
        "name":name,"mode":mode,"fixed_alpha":float(fixed_alpha),"n":len(labels_np),
        "condition_match":float((pred==labels_np).mean()),
        "valid_012_rate":float(np.isin(pred,[0,1,2]).mean()),
        "mean_classifier_confidence":float(probs.max(1).mean()),
        "confidence_ge_095":float((probs.max(1)>=.95).mean()),
        "TVD_to_balanced_012_among_valid":tvd,
        "class_counts":counts.tolist(),
        "unique_token_fraction":float(len(np.unique(tokens,axis=0))/len(tokens)),
        "morphology":morphology(images),
        "mask_counts_first_sample":[int(x) for x in rec["mask_before"].sum(2)[:,0]],
        "alpha_mean_by_step":[None if np.isnan(x) else float(x) for x in rec["alpha_mean"]],
        "state_H_rms_by_step":[None if np.isnan(x) else float(x) for x in rec["state_H_rms"]],
    }
    np.savez_compressed(outdir/f"generation_{name}.npz",labels=labels_np,tokens=tokens,images=images,**{f"traj_{k}":v for k,v in rec.items() if isinstance(v,np.ndarray)})
    json_dump(outdir/f"metrics_{name}.json",met)
    return met


# ============================================================
# 5. Exact identity audit
# ============================================================
@torch.no_grad()
def identity_audit(model: MaskGIT, device, batch=4, rounds=6, seed=123):
    assert model.kind=="linear"
    model.eval();g=torch.Generator(device=device);g.manual_seed(seed)
    # fixed random ground-truth tokens, progressively revealed
    truth=torch.randint(0,CODEBOOK,(batch,SEQ),device=device,generator=g)
    labels=torch.randint(0,3,(batch,),device=device,generator=g)
    perm=torch.argsort(torch.rand(batch,SEQ,device=device,generator=g),dim=1)
    rank=torch.empty_like(perm);rank.scatter_(1,perm,torch.arange(SEQ,device=device)[None].expand(batch,-1))
    state=None;max_abs=0.0;max_rel=0.0;rows=[]
    for r in range(rounds):
        reveal=int(round(SEQ*r/max(1,rounds-1)))
        known=rank<reveal
        cur=torch.full_like(truth,MASK_ID);cur[known]=truth[known]
        fresh=model(cur,labels)
        pers,state,stats=model.forward_persistent(cur,labels,state,"identity",1.0)
        d=(fresh-pers)
        ma=float(d.abs().max().cpu())
        rel=float((d.norm()/fresh.norm().clamp_min(1e-12)).cpu())
        max_abs=max(max_abs,ma);max_rel=max(max_rel,rel)
        rows.append({"round":r,"reveal":reveal,"max_abs":ma,"rel_l2":rel,"state_H_rms":stats["state_H_rms"]})
    return {"max_abs":max_abs,"max_rel_l2":max_rel,
            "pass_fp32":bool(max_abs<1e-4 and max_rel<1e-6),"rounds":rows}


# ============================================================
# 6. Main orchestration
# ============================================================
def build_model(kind,args,device,selective=False):
    return MaskGIT(kind=kind,d=args.d,layers=args.layers,heads=args.heads,ff=args.ff,
                   features=args.features,nclass=3,feature_seed=args.seed+101,selective_gate=selective).to(device)


def map_labels_012(y):
    # labels are already 0,1,2; keep explicit for invariant.
    if not np.isin(y,[0,1,2]).all():raise ValueError("generator labels must be 0/1/2")
    return y.astype(np.int64)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,default=None,help="Keras-style mnist.npz; if omitted, try keras.datasets.mnist download")
    ap.add_argument("--out",type=Path,default=Path("H0_RUN"))
    ap.add_argument("--smoke",action="store_true",help="Use sklearn digits resized to 28x28; structural test only")
    ap.add_argument("--n-per-class",type=int,default=2000)
    ap.add_argument("--epochs-softmax",type=int,default=18)
    ap.add_argument("--epochs-linear",type=int,default=18)
    ap.add_argument("--gate-epochs",type=int,default=2)
    ap.add_argument("--batch",type=int,default=256)
    ap.add_argument("--d",type=int,default=128)
    ap.add_argument("--layers",type=int,default=4)
    ap.add_argument("--heads",type=int,default=4)
    ap.add_argument("--ff",type=int,default=512)
    ap.add_argument("--features",type=int,default=128)
    ap.add_argument("--steps",type=int,default=10)
    ap.add_argument("--temp",type=float,default=.25)
    ap.add_argument("--eval-n",type=int,default=300)
    ap.add_argument("--ref-epochs",type=int,default=2)
    ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--threads",type=int,default=8)
    ap.add_argument("--skip-softmax",action="store_true")
    ap.add_argument("--skip-linear",action="store_true")
    ap.add_argument("--skip-gate",action="store_true")
    ap.add_argument("--alpha",type=str,default="1,0.995,0.99,0.98,0.95")
    args=ap.parse_args()

    seed_all(args.seed);torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke:
        xall,yall,xte,yte=load_smoke_digits(args.seed)
        scientific=False
    else:
        if args.data is not None:
            if not args.data.exists():
                raise FileNotFoundError(args.data)
            xall,yall,xte,yte=load_mnist_npz(args.data)
        else:
            xall,yall,xte,yte=load_mnist_keras()
        scientific=True

    # Generator uses 0/1/2; reference classifier uses all classes.
    x012,y012=filter_classes(xall,yall)
    x012,y012=subset_per_class(x012,y012,args.n_per_class,args.seed)
    xtr,ytr,xva,yva=stratified_split(x012,y012,.1,args.seed)
    ytr=map_labels_012(ytr);yva=map_labels_012(yva)

    codebook_path=args.out/"codebook.npy"
    cent=fit_codebook(xtr,codebook_path,args.seed) if not codebook_path.exists() else np.load(codebook_path)
    tr=encode_tokens(xtr,cent);va=encode_tokens(xva,cent)
    np.savez_compressed(args.out/"tokenized_012.npz",train_tokens=tr,train_labels=ytr,val_tokens=va,val_labels=yva)

    # Tokenizer diagnostic on 0/1/2 test subset.
    tx,ty=filter_classes(xte,yte);tt=encode_tokens(tx,cent);recon=decode_tokens(tt,cent)
    mse=float(np.mean((recon-tx)**2));psnr=float(10*np.log10(1/max(mse,1e-12)))

    manifest={
        "status":"H0_FROM_SCRATCH_INITIALIZED","scientific_mnist":scientific,"device":str(device),
        "generator_classes":[0,1,2],"train_samples":len(xtr),"val_samples":len(xva),
        "token_grid":[7,7],"sequence_length":SEQ,"codebook":CODEBOOK,"patch":[4,4],
        "d":args.d,"layers":args.layers,"heads":args.heads,"ff":args.ff,"random_features":args.features,
        "decode_steps":args.steps,"temperature":args.temp,"tokenizer_test_mse":mse,"tokenizer_test_psnr_db":psnr,
    }
    json_dump(args.out/"manifest.json",manifest)
    print(json.dumps(manifest),flush=True)

    # Reference classifier for generation metrics.
    ref=train_ref_classifier(xall,yall,device,args.ref_epochs)
    ref_test=classify(ref,xte,device)
    ref_acc=float((ref_test.argmax(1)==yte).mean())
    torch.save(ref.state_dict(),args.out/"reference_classifier.pt")

    results={"manifest":manifest,"reference_classifier_test_acc":ref_acc,"models":{},"generation":{}}

    # A. Conventional softmax B0 baseline.
    if not args.skip_softmax:
        sm=build_model("softmax",args,device,False)
        td=train_backbone(sm,tr,ytr,va,yva,args.out/"softmax",args.epochs_softmax,args.batch,device)
        results["models"]["softmax"]={"parameters":sum(p.numel() for p in sm.parameters()),**{k:v for k,v in td.items() if k!="history"}}
        labs=(np.arange(args.eval_n)%3).astype(np.int64)
        results["generation"]["softmax"]=eval_generation("softmax",sm,labs,cent,ref,device,args.out,args.steps,args.temp,args.seed+1000,"fresh")

    # B. Fresh normalized random-feature linear MaskGIT.
    lin=build_model("linear",args,device,False)
    if not args.skip_linear:
        td=train_backbone(lin,tr,ytr,va,yva,args.out/"linear",args.epochs_linear,args.batch,device)
    else:
        ck=args.out/"linear"/"best.pt"
        if not ck.exists(): raise FileNotFoundError(f"--skip-linear requires {ck}")
        d=torch.load(ck,map_location=device,weights_only=False);lin.load_state_dict(d["model"]);td={"best_epoch":d.get("epoch"),"best_val_ce":d.get("best")}
    results["models"]["linear"]={"parameters":sum(p.numel() for p in lin.parameters()),**{k:v for k,v in td.items() if k!="history"}}

    audit=identity_audit(lin, device, batch=4, rounds=6, seed=args.seed+77)
    # Python positional workaround above kept simple below if signature changes.
    results["identity_audit"]=audit
    json_dump(args.out/"identity_audit.json",audit)
    if not audit["pass_fp32"]:
        raise RuntimeError(f"A=1 exactness gate failed: {audit}")

    labs=(np.arange(args.eval_n)%3).astype(np.int64)
    results["generation"]["linear_fresh"]=eval_generation("linear_fresh",lin,labs,cent,ref,device,args.out,args.steps,args.temp,args.seed+1000,"fresh")
    results["generation"]["persistent_identity"]=eval_generation("persistent_identity",lin,labs,cent,ref,device,args.out,args.steps,args.temp,args.seed+1000,"identity",1.0)

    # Fixed-alpha sweep: no retraining.
    for a in [float(x) for x in args.alpha.split(",") if x.strip()]:
        nm=f"fixed_alpha_{a:.4f}".replace(".","p")
        results["generation"][nm]=eval_generation(nm,lin,labs,cent,ref,device,args.out,args.steps,args.temp,args.seed+1000,"fixed",a)

    # C. Tiny selective gate, initialized from linear backbone and trained trajectory-wise.
    if not args.skip_gate:
        sel=build_model("linear",args,device,True)
        # Copy all shared weights. Gate parameters remain at their own initialization.
        base=lin.state_dict(); cur=sel.state_dict()
        for k,v in base.items():
            if k in cur and cur[k].shape==v.shape:cur[k]=v
        sel.load_state_dict(cur)
        train_selective_gate(sel,tr,ytr,va,yva,args.out,device,args.gate_epochs,min(args.batch,64),steps=min(args.steps,6),seed=args.seed)
        results["generation"]["selective"]=eval_generation("selective",sel,labs,cent,ref,device,args.out,args.steps,args.temp,args.seed+1000,"selective")
        results["selective_gate"]={
            "theta0":sel.gate.theta0.detach().cpu().tolist(),
            "theta_mask":sel.gate.theta_mask.detach().cpu().tolist(),
            "theta_new":sel.gate.theta_new.detach().cpu().tolist(),
            "parameters":sum(p.numel() for p in sel.gate.parameters()),
        }

    # Decision summary.
    lf=results["generation"]["linear_fresh"]
    pi=results["generation"]["persistent_identity"]
    results["controls"]={
        "fresh_vs_identity_condition_match_abs":abs(lf["condition_match"]-pi["condition_match"]),
        "fresh_vs_identity_valid_abs":abs(lf["valid_012_rate"]-pi["valid_012_rate"]),
        "same_sampling_seed":True,
        "identity_expected_exact_at_logits":True,
        "identity_generation_expected_identical_if_floating_ties_do_not_change_sampling":True,
    }
    json_dump(args.out/"H0_RESULTS.json",results)
    print("H0_DONE",json.dumps({"out":str(args.out),"identity":audit,"reference_acc":ref_acc},indent=2),flush=True)


if __name__=="__main__":
    main()
