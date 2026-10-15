#!/usr/bin/env python3
"""H3A single-core scaling contract.

H3A keeps the frozen H1.1-functional-v9 operators and fixed dimensions
D=32,H=2,M=32,FF=64, but makes token count N variable.  The scientific
question is the capacity frontier of one SpiNNaker ARM core under the same
resident-state execution principle as H2F.
"""
from __future__ import annotations

import numpy as np

from spinnaker_backend.h2f_aplx.h2f_contract import (
    H2F_HEADER_WORDS, H2F_OUTPUT_HEADER_WORDS,
    F_ACT, F_PQ, F_PK, F_H, F_MEM, F_DEC,
    build_header as build_h2f_header,
)

H3A_MAGIC = 0x48334131       # "H3A1"
H3A_OUTPUT_MAGIC = 0x4833414F  # "H3AO"
H3A_VERSION = 1
H3A_HEADER_WORDS = H2F_HEADER_WORDS
H3A_OUTPUT_HEADER_WORDS = 16


def build_header(meta: dict, desc: dict[str, tuple[int, int]], *, tokens: int, steps: int) -> np.ndarray:
    if tokens < 1:
        raise ValueError("tokens must be >=1")
    h = build_h2f_header(meta, desc, steps=steps).copy()
    h[0] = np.uint32(H3A_MAGIC)
    h[1] = np.uint32(H3A_VERSION)
    h[2] = np.uint32(tokens)
    return h


def arena_words(tokens: int, d: int = 32, heads: int = 2, features: int = 32, ff: int = 64) -> int:
    """Exact words requested by the current H2F/H3A resident arena."""
    dh = d // heads
    H_words = heads * features * dh
    g_words = heads * features
    pq_words = tokens * heads * features
    pk_words = pq_words
    v_words = tokens * d
    token_words = d
    feature_words = heads * features
    return (
        H_words + g_words + pq_words + pk_words + v_words
        + tokens * d          # ln_buf
        + tokens * d          # attended
        + tokens * d          # z_buf
        + tokens * ff         # hidden
        + tokens * d          # y_buf
        + token_words         # proj
        + feature_words       # a_buf
        + feature_words       # new_pq
        + feature_words       # new_pk
        + token_words         # new_v
    )


def arena_bytes(tokens: int, d: int = 32, heads: int = 2, features: int = 32, ff: int = 64) -> int:
    return 4 * arena_words(tokens, d, heads, features, ff)


def trajectory_word_count(tokens: int, d: int, steps: int) -> int:
    return steps * tokens * d + steps * tokens


def output_word_count(tokens: int, d: int, heads: int, features: int) -> int:
    dh = d // heads
    return (
        H3A_OUTPUT_HEADER_WORDS
        + tokens * d
        + heads * features * dh
        + heads * features
        + tokens * heads * features
        + tokens * heads * features
        + tokens * heads * dh
    )


__all__ = [
    "H3A_MAGIC", "H3A_OUTPUT_MAGIC", "H3A_VERSION", "H3A_HEADER_WORDS",
    "H3A_OUTPUT_HEADER_WORDS", "F_ACT", "F_PQ", "F_PK", "F_H", "F_MEM", "F_DEC",
    "build_header", "arena_words", "arena_bytes", "trajectory_word_count", "output_word_count",
]
