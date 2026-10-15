#!/usr/bin/env python3
"""H3B single-chip distributed execution contract.

H3B keeps the frozen H1.1-functional-v9 operators and fixed D32/H2/M32/FF64/L1.
A global token sequence is partitioned into equal contiguous shards.  Each core
holds local pq/pk/v plus replicated global H/g.  Worker 0 coordinates a
single-chip multicast gather/broadcast; it is not a mathematical host fallback.
"""
from __future__ import annotations

import numpy as np

from spinnaker_backend.h2f_aplx.h2f_contract import (
    H2F_HEADER_WORDS,
    F_ACT, F_PQ, F_PK, F_H, F_MEM, F_DEC,
    build_header as build_h2f_header,
)

H3B_MAGIC = 0x48334231          # "H3B1"
H3B_OUTPUT_MAGIC = 0x4833424F   # "H3BO"
H3B_VERSION = 1
H3B_HEADER_WORDS = 128
H3B_OUTPUT_HEADER_WORDS = 24
H3B_MAX_WORKERS = 16
H3B_N_KEYS = 4096
H3B_PARTITION_ID = "H3B_MC"

# Extra words after the unchanged H2F descriptor header.
CFG_RANK = H2F_HEADER_WORDS + 0
CFG_N_WORKERS = H2F_HEADER_WORDS + 1
CFG_GLOBAL_TOKENS = H2F_HEADER_WORDS + 2
CFG_SEND_KEY = H2F_HEADER_WORDS + 3
CFG_PEER_KEYS = H2F_HEADER_WORDS + 4
CFG_PACING_US = H2F_HEADER_WORDS + 4 + H3B_MAX_WORKERS

# Packet-key offsets inside each worker's 4096-key partition.
KEY_READY_BASE = 3000
KEY_DONE_BASE = 3020
KEY_GRANT_BASE = 3040
KEY_BCAST_DONE_BASE = 3060


def build_base_header(meta: dict, desc: dict[str, tuple[int, int]], *,
                      local_tokens: int, workers: int, steps: int,
                      pacing_us: int = 1) -> np.ndarray:
    if not (2 <= workers <= H3B_MAX_WORKERS):
        raise ValueError(f"workers must be in [2,{H3B_MAX_WORKERS}]")
    if local_tokens < 1:
        raise ValueError("local_tokens must be positive")
    h2 = build_h2f_header(meta, desc, steps=steps)
    out = np.zeros(H3B_HEADER_WORDS, dtype="<u4")
    out[:H2F_HEADER_WORDS] = h2
    out[0] = np.uint32(H3B_MAGIC)
    out[1] = np.uint32(H3B_VERSION)
    out[2] = np.uint32(local_tokens)
    out[CFG_RANK] = 0
    out[CFG_N_WORKERS] = np.uint32(workers)
    out[CFG_GLOBAL_TOKENS] = np.uint32(local_tokens * workers)
    out[CFG_SEND_KEY] = 0
    out[CFG_PEER_KEYS:CFG_PEER_KEYS + H3B_MAX_WORKERS] = 0
    out[CFG_PACING_US] = np.uint32(pacing_us)
    return out


def worker_header(base: np.ndarray, *, rank: int, send_key: int,
                  peer_keys: list[int]) -> np.ndarray:
    h = np.ascontiguousarray(base, dtype="<u4").copy()
    workers = int(h[CFG_N_WORKERS])
    if len(peer_keys) != workers:
        raise ValueError((len(peer_keys), workers))
    if not (0 <= rank < workers):
        raise ValueError(rank)
    h[CFG_RANK] = np.uint32(rank)
    h[CFG_SEND_KEY] = np.uint32(send_key)
    h[CFG_PEER_KEYS:CFG_PEER_KEYS + workers] = np.asarray(peer_keys, dtype="<u4")
    return h


def arena_words(local_tokens: int, d: int = 32, heads: int = 2,
                features: int = 32, ff: int = 64) -> int:
    """H3B worker arena; same resident payload as H3A for N_local plus one
    alignment word.  Init/delta int64 reduction reuses attended/z/hidden.
    """
    dh = d // heads
    H_words = heads * features * dh
    g_words = heads * features
    pq_words = local_tokens * heads * features
    pk_words = pq_words
    v_words = local_tokens * d
    token_words = d
    feature_words = heads * features
    base = (
        H_words + g_words + pq_words + pk_words + v_words
        + local_tokens * d          # ln_buf
        + local_tokens * d          # attended; also raw-H workspace first half
        + local_tokens * d          # z_buf; raw-H workspace second half / delta-g
        + local_tokens * ff         # hidden; raw-g workspace during gather
        + local_tokens * d          # y_buf
        + token_words               # proj
        + feature_words             # a_buf
        + feature_words             # new_pq
        + feature_words             # new_pk
        + token_words               # new_v
    )
    return base + 1  # allow 8-byte alignment of the arena payload


def arena_bytes(local_tokens: int, d: int = 32, heads: int = 2,
                features: int = 32, ff: int = 64) -> int:
    return 4 * arena_words(local_tokens, d, heads, features, ff)


def local_trajectory_words(local_tokens: int, d: int, steps: int) -> int:
    return steps * local_tokens * d + steps * local_tokens


def output_word_count(local_tokens: int, d: int, heads: int,
                      features: int) -> int:
    dh = d // heads
    return (
        H3B_OUTPUT_HEADER_WORDS
        + local_tokens * d
        + heads * features * dh
        + heads * features
        + local_tokens * heads * features
        + local_tokens * heads * features
        + local_tokens * heads * dh
    )


def init_gather_words(heads: int = 2, features: int = 32,
                      d: int = 32) -> int:
    dh = d // heads
    # H raw int64 plus g raw int64, each transmitted as two 32-bit payloads.
    return 2 * (heads * features * dh + heads * features)


def bcast_words(heads: int = 2, features: int = 32, d: int = 32) -> int:
    dh = d // heads
    return heads * features * dh + heads * features


__all__ = [
    "H3B_MAGIC", "H3B_OUTPUT_MAGIC", "H3B_VERSION", "H3B_HEADER_WORDS",
    "H3B_OUTPUT_HEADER_WORDS", "H3B_MAX_WORKERS", "H3B_N_KEYS",
    "H3B_PARTITION_ID", "CFG_RANK", "CFG_N_WORKERS", "CFG_GLOBAL_TOKENS",
    "CFG_SEND_KEY", "CFG_PEER_KEYS", "CFG_PACING_US", "KEY_READY_BASE",
    "KEY_DONE_BASE", "KEY_GRANT_BASE", "KEY_BCAST_DONE_BASE",
    "F_ACT", "F_PQ", "F_PK", "F_H", "F_MEM", "F_DEC",
    "build_base_header", "worker_header", "arena_words", "arena_bytes",
    "local_trajectory_words", "output_word_count", "init_gather_words",
    "bcast_words",
]
