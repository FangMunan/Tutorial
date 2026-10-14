#!/usr/bin/env python3
"""Shared H2F frozen board-package layout.

H2F keeps the scientific operator fixed and changes only execution/storage:
static HIR arrays + the complete six-step input trajectory are written to SDRAM
once, resident state lives on the ARM core, and only final output/state are read
back after completion.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

H2F_MAGIC = 0x48324631  # "H2F1"
H2F_OUTPUT_MAGIC = 0x4832464F  # "H2FO"
H2F_VERSION = 1
H2F_HEADER_WORDS = 96
H2F_OUTPUT_HEADER_WORDS = 8

# Must match h2f_contract.h exactly.
ARRAY_NAMES = (
    "ln1_g", "ln1_b", "ln2_g", "ln2_b",
    "Wq", "bq", "Wk", "bk", "Wv", "bv", "omega",
    "Wo_row_ptr", "Wo_col_idx", "Wo_values", "Wo_bias",
    "W1_row_ptr", "W1_col_idx", "W1_values", "W1_bias",
    "W2_row_ptr", "W2_col_idx", "W2_values", "W2_bias",
    "q_decoder_0", "q_decoder_1", "k_decoder_0", "k_decoder_1",
)
ARRAY_INDEX = {name: i for i, name in enumerate(ARRAY_NAMES)}
N_ARRAYS = len(ARRAY_NAMES)
DESC_BASE = 28
assert DESC_BASE + 2 * N_ARRAYS <= H2F_HEADER_WORDS

# Frozen v9 numerical contract.
F_ACT = 20
F_PQ = 24
F_PK = 16
F_H = 20
F_MEM = 22
F_DEC = 20


def _hardware_dtype(name: str) -> np.dtype:
    # The H1/H2 NumPy exporters often keep arithmetic arrays as int64 even when
    # the frozen contract requires signed int32 storage.  H2F therefore chooses
    # the target storage type by operator role, not by incidental host dtype.
    if name.endswith("_row_ptr"):
        return np.dtype("<u4")
    if name.endswith("_col_idx"):
        return np.dtype("<u2")
    return np.dtype("<i4")


def _as_le_bytes(name: str, a: np.ndarray) -> bytes:
    a = np.ascontiguousarray(a)
    dt = _hardware_dtype(name)
    cast = a.astype(dt, copy=False)
    # Prevent silent narrowing if an upstream change exceeds the frozen H2F
    # storage range.
    if dt.itemsize == 4 and dt.kind == "i":
        back = cast.astype(np.int64)
        if not np.array_equal(back, a.astype(np.int64)):
            raise OverflowError(f"{name} does not fit signed int32")
    elif dt.itemsize == 4 and dt.kind == "u":
        back = cast.astype(np.uint64)
        if np.any(a.astype(np.int64) < 0) or not np.array_equal(back, a.astype(np.uint64)):
            raise OverflowError(f"{name} does not fit uint32")
    elif dt.itemsize == 2 and dt.kind == "u":
        back = cast.astype(np.uint64)
        if np.any(a.astype(np.int64) < 0) or not np.array_equal(back, a.astype(np.uint64)):
            raise OverflowError(f"{name} does not fit uint16")
    return cast.tobytes(order="C")


def build_hir_blob(arrays: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """Pack HIR arrays as a 4-byte-aligned little-endian SDRAM blob."""
    blob = bytearray()
    desc: Dict[str, Tuple[int, int]] = {}
    for name in ARRAY_NAMES:
        if name not in arrays:
            raise KeyError(f"missing HIR array {name}")
        while len(blob) & 3:
            blob.append(0)
        offset = len(blob)
        raw = _as_le_bytes(name, np.asarray(arrays[name]))
        blob.extend(raw)
        desc[name] = (offset, len(raw))
    while len(blob) & 3:
        blob.append(0)
    words = np.frombuffer(bytes(blob), dtype="<u4").copy()
    return words, desc


def build_header(meta: dict, desc: Dict[str, Tuple[int, int]], *, steps: int) -> np.ndarray:
    """Build the fixed H2F configuration/descriptor header."""
    n = int(meta["d"])
    heads = int(meta["heads"])
    if heads != 2:
        raise ValueError("H2F-v1 board contract is frozen to two heads")
    header = np.zeros(H2F_HEADER_WORDS, dtype="<u4")
    header[0] = H2F_MAGIC
    header[1] = H2F_VERSION
    header[2] = 16  # tokens, frozen U5/H2D contract
    header[3] = n
    header[4] = heads
    header[5] = int(meta["features"])
    header[6] = int(meta["ff"])
    header[7] = int(steps)
    header[8:14] = [F_ACT, F_PQ, F_PK, F_H, F_MEM, F_DEC]
    dh = n // heads
    header[14] = np.uint32(int(round((dh ** -0.25) * (1 << F_ACT))))
    header[15] = N_ARRAYS

    q_specs = meta["q_specs"]
    k_specs = meta["k_specs"]
    if len(q_specs) != 2 or len(k_specs) != 2:
        raise ValueError("expected two Q and two K Izh specs")
    p = 16
    for specs in (q_specs, k_specs):
        for bias, gain, window in specs:
            header[p] = np.uint32(np.int32(round(float(bias) * (1 << F_MEM))))
            header[p + 1] = np.uint32(np.int32(round(float(gain) * (1 << F_MEM))))
            header[p + 2] = np.uint32(int(window))
            p += 3
    assert p == DESC_BASE

    for i, name in enumerate(ARRAY_NAMES):
        offset, nbytes = desc[name]
        header[DESC_BASE + 2 * i] = np.uint32(offset)
        header[DESC_BASE + 2 * i + 1] = np.uint32(nbytes)
    return header


def output_word_count(tokens: int, d: int, heads: int, features: int) -> int:
    dh = d // heads
    y = tokens * d
    H = heads * features * dh
    g = heads * features
    pq = tokens * heads * features
    pk = pq
    v = tokens * heads * dh
    return H2F_OUTPUT_HEADER_WORDS + y + H + g + pq + pk + v


def package_metadata_bytes(meta: dict) -> np.ndarray:
    return np.frombuffer(json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)


def decode_package_metadata(a: np.ndarray) -> dict:
    return json.loads(bytes(np.asarray(a, dtype=np.uint8).tolist()).decode("utf-8"))


def save_board_package(path: Path, *, header_u32: np.ndarray, hir_blob_u32: np.ndarray,
                       trajectory_u32: np.ndarray, oracle: Dict[str, np.ndarray], metadata: dict) -> None:
    payload = {
        "header_u32": np.ascontiguousarray(header_u32, dtype="<u4"),
        "hir_blob_u32": np.ascontiguousarray(hir_blob_u32, dtype="<u4"),
        "trajectory_u32": np.ascontiguousarray(trajectory_u32, dtype="<u4"),
        "package_metadata_json_utf8": package_metadata_bytes(metadata),
    }
    for key, value in oracle.items():
        payload[f"oracle_{key}"] = np.ascontiguousarray(value, dtype="<i4")
    np.savez_compressed(path, **payload)
