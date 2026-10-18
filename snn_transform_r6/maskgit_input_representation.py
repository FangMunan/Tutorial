#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Label-free input representation for the R6B.3 MaskGIT/P3 route.

This module removes the historical fixed-random token/position/class embedding
interface without inventing a new credit-assignment pathway.

Token identity
--------------
The 64 visual-token embeddings are derived directly from the frozen SNN-VQ
codebook.  The 16-D codebook geometry is centered and embedded isometrically into
P3 dimension D with a deterministic DCT basis.  A single global scale is applied,
so all pairwise codebook distances are preserved up to that common scale.  The
MASK token is the centered codebook origin (zero vector).

Position
--------
The 7x7 spatial positions use deterministic 2-D sinusoidal embeddings.  They are
centered and globally scaled to the same nominal magnitude used by the previous
P3 interface.  No random position table is used.

Class labels
------------
There is deliberately no class embedding and no label argument in the context
constructor.  MNIST labels may still exist outside this module for post-hoc
measurement only; they cannot enter the neural predictor through this interface.

Why not make a new plastic embedding table here?
-------------------------------------------------
A learnable embedding table would need a new local mismatch/credit pathway from
P3 back to every visible token and position.  Adding such a mechanism merely to
imitate BP-trained MaskGIT embeddings would change the MSDL learning theory and
increase future SpiNNaker cost.  This representation therefore removes the
arbitrary-random bottleneck first, while preserving the frozen SNN-VQ geometry
and explicit spatial structure.  Plastic embedding residuals can later be tested
as a separate causal intervention if needed.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch


def _dct_isometry(out_dim: int, in_dim: int) -> np.ndarray:
    """Return P[out_dim,in_dim] with P.T@P ~= I using the first DCT-II modes."""
    if out_dim < in_dim:
        raise ValueError(f'P3 dimension D={out_dim} must be >= code dimension {in_dim}')
    j=np.arange(out_dim,dtype=np.float64)[:,None]
    k=np.arange(in_dim,dtype=np.float64)[None,:]
    P=np.sqrt(2.0/out_dim)*np.cos(math.pi*(j+0.5)*k/out_dim)
    P[:,0]=1.0/math.sqrt(out_dim)
    gram=P.T@P
    if not np.allclose(gram,np.eye(in_dim),rtol=1e-10,atol=1e-10):
        raise AssertionError('deterministic DCT projection is not isometric')
    return P


def _scale_mean_norm(x: np.ndarray, target_mean_norm: float) -> Tuple[np.ndarray,float]:
    norms=np.linalg.norm(x,axis=1)
    mean=float(np.mean(norms))
    if mean <= 1e-12:
        raise ValueError('cannot scale a zero embedding set')
    scale=float(target_mean_norm/mean)
    return x*scale,scale


def token_embeddings_from_snnvq(tokenizer, D: int,
                                token_scale: float = 0.70) -> Tuple[np.ndarray,Dict]:
    """Map frozen SNN-VQ codebook vectors into P3 space without random tables."""
    with torch.no_grad():
        code=tokenizer.vq.codebook.weight.detach().cpu().numpy().astype(np.float64)
    if code.ndim != 2:
        raise ValueError(f'expected [K,d] VQ codebook, got {code.shape}')
    K,d=code.shape
    centered=code-code.mean(axis=0,keepdims=True)
    P=_dct_isometry(int(D),int(d))
    emb=centered@P.T
    emb,scale=_scale_mean_norm(emb,math.sqrt(D)*float(token_scale))

    # Preserve VQ geometry exactly up to one global scale.  Check all pairwise
    # squared distances without relying on a stochastic sample.
    dc=((centered[:,None,:]-centered[None,:,:])**2).sum(-1)
    de=((emb[:,None,:]-emb[None,:,:])**2).sum(-1)
    if not np.allclose(de,dc*(scale**2),rtol=2e-10,atol=2e-10):
        raise AssertionError('token embedding failed the VQ pairwise-distance audit')

    tok=np.zeros((K+1,D),dtype=np.float64)
    tok[:K]=emb
    # MASK is the centered codebook origin: an explicit neutral/unknown state.
    tok[K]=0.0
    audit={
        'visual_vocab':int(K),
        'vq_dim':int(d),
        'p3_dim':int(D),
        'source':'frozen SNN-VQ codebook',
        'projection':'deterministic DCT isometry',
        'pairwise_vq_geometry_preserved_up_to_global_scale':True,
        'global_scale':scale,
        'mean_visual_token_norm':float(np.mean(np.linalg.norm(tok[:K],axis=1))),
        'mask_token':'zero vector at centered VQ origin',
        'mask_norm':float(np.linalg.norm(tok[K])),
        'random_token_embedding':False,
    }
    return tok,audit


def _axis_sinusoid(coords: np.ndarray, dim: int) -> np.ndarray:
    if dim % 2:
        raise ValueError('each 2-D positional axis must receive an even dimension')
    half=dim//2
    freq=np.exp(-math.log(10000.0)*np.arange(half,dtype=np.float64)/max(1,half))
    angle=np.asarray(coords,dtype=np.float64)[:,None]*freq[None,:]
    return np.concatenate([np.sin(angle),np.cos(angle)],axis=1)


def position_embeddings_2d(grid: int, D: int,
                           position_scale: float = 0.35) -> Tuple[np.ndarray,Dict]:
    """Deterministic 2-D sinusoidal encoding for a square token grid."""
    if D % 4:
        raise ValueError('D must be divisible by 4 for equal row/column sin-cos encoding')
    N=int(grid)*int(grid)
    rr=np.repeat(np.arange(grid,dtype=np.float64),grid)
    cc=np.tile(np.arange(grid,dtype=np.float64),grid)
    row=_axis_sinusoid(rr,D//2)
    col=_axis_sinusoid(cc,D//2)
    pos=np.concatenate([row,col],axis=1)
    pos=pos-pos.mean(axis=0,keepdims=True)
    pos,scale=_scale_mean_norm(pos,math.sqrt(D)*float(position_scale))
    audit={
        'grid':f'{grid}x{grid}',
        'positions':N,
        'p3_dim':int(D),
        'source':'deterministic 2-D sinusoidal coordinates',
        'centered':True,
        'global_scale':scale,
        'mean_position_norm':float(np.mean(np.linalg.norm(pos,axis=1))),
        'random_position_embedding':False,
    }
    return pos,audit


def build_input_embeddings(tokenizer, D: int, grid: int = 7,
                           token_scale: float = 0.70,
                           position_scale: float = 0.35):
    tok,ta=token_embeddings_from_snnvq(tokenizer,D,token_scale)
    pos,pa=position_embeddings_2d(grid,D,position_scale)
    if len(pos) != grid*grid:
        raise AssertionError('position table length mismatch')
    audit={
        'token':ta,
        'position':pa,
        'class_embedding_present':False,
        'labels_enter_neural_input':False,
        'fixed_random_embedding_present':False,
    }
    return tok,pos,audit


def context_from_arr_unconditional(arr: np.ndarray, tok: np.ndarray,
                                   pos: np.ndarray):
    """P3 input x_i = token_embedding(z_i) + position_embedding(i), no class label."""
    arr=np.asarray(arr,dtype=np.int64).reshape(-1)
    if len(arr) != len(pos):
        raise ValueError(f'token/position length mismatch: {len(arr)} vs {len(pos)}')
    if np.any(arr<0) or np.any(arr>=len(tok)):
        raise ValueError('token id outside token-embedding table')
    x=tok[arr]+pos
    mu=x.mean(1,keepdims=True)
    var=np.mean((x-mu)**2,1,keepdims=True)
    xn=(x-mu)/np.sqrt(var+1e-5)
    return xn,x


def representation_self_test(tokenizer, D: int = 96, grid: int = 7) -> Dict:
    tok,pos,audit=build_input_embeddings(tokenizer,D,grid)
    arr=np.full(grid*grid,len(tok)-1,dtype=np.int64)
    xn,x=context_from_arr_unconditional(arr,tok,pos)
    if not np.all(np.isfinite(xn)) or not np.all(np.isfinite(x)):
        raise AssertionError('input representation produced non-finite values')
    if audit['class_embedding_present'] or audit['labels_enter_neural_input']:
        raise AssertionError('label leakage into neural input')
    return {'pass':True,**audit}
