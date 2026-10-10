#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export the frozen H2 software reference into one resident-hardware bundle.

The canonical bundle preserves the validated B2.1 execution semantics: calibrated
Q/K Izh feature maps plus the existing H1 sparse persistent first block.  An experimental
resident projection/feature cache can be requested explicitly, but is OFF by default
because its changed GEMM batching produces a small (~1e-5--1e-4 relative) FP32 numerical
difference and has not been frozen as the hardware reference.

Everything is uploaded once per experiment.  The float32 image is a numerical reference;
SpiNNaker-1 fixed-point conversion remains a separate audited mapping stage.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "h1_cloud"))
sys.path.insert(0, str(ROOT / "h2_spinnaker"))

import h0_base as h0
import h1_sparse_fusion as h1
import h1b2_error_localization as h1b2
import h1b21_qk_layer_scan as b21
from h2_block1_cache_audit import enable_cached_block1, load_selected

ALIGN = 64
MAX_DECODER = 32


def align(n: int, a: int = ALIGN) -> int:
    return ((n + a - 1) // a) * a


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_tensor_image(path: Path, state_dict) -> list[dict]:
    manifest, offset = [], 0
    with path.open("wb") as f:
        for name, t in state_dict.items():
            if not torch.is_tensor(t):
                continue
            arr = np.asarray(t.detach().cpu(), dtype="<f4", order="C")
            pad_to = align(offset)
            if pad_to > offset:
                f.write(b"\x00" * (pad_to - offset)); offset = pad_to
            raw = arr.tobytes(order="C"); f.write(raw)
            manifest.append({
                "name": name, "shape": list(arr.shape), "dtype": "float32-le",
                "offset_bytes": offset, "nbytes": len(raw),
            })
            offset += len(raw)
    return manifest


def write_calibration_binary(path: Path, selected) -> list[dict]:
    """record: layer, side, bias, gain, window, decoder_len, decoder[32]."""
    meta = []
    with path.open("wb") as f:
        for li in range(4):
            for side_i, side in enumerate(("Q", "K")):
                v = selected[(li, side)]; spec = v["spec"]
                dec = v["decoder"].detach().cpu().float().numpy()
                if len(dec) > MAX_DECODER:
                    raise ValueError(f"decoder length {len(dec)} exceeds {MAX_DECODER}")
                padded = np.zeros(MAX_DECODER, dtype="<f4"); padded[:len(dec)] = dec
                f.write(struct.pack(
                    "<IIffII", li, side_i, float(spec["bias"]), float(spec["gain"]),
                    int(spec["window_ms"]), int(len(dec))))
                f.write(padded.tobytes(order="C"))
                meta.append({"layer": li, "side": side, **spec, "decoder_len": int(len(dec))})
    return meta


@torch.no_grad()
def reference_trajectory(model, labels, canvases):
    state = None; logits, checks = [], []
    for step, cur in enumerate(canvases):
        z, state, stats = model.forward_persistent(cur, labels, state, "selective")
        logits.append(z.detach().cpu().float().numpy())
        hp = getattr(state, "Hp", getattr(state, "H", None))
        gp = getattr(state, "gp", getattr(state, "g", None))
        checks.append({
            "step": step,
            "logits_sum": float(z.double().sum().cpu()),
            "logits_sq_sum": float((z.double() * z.double()).sum().cpu()),
            "H_sum": None if hp is None else float(hp.double().sum().cpu()),
            "g_sum": None if gp is None else float(gp.double().sum().cpu()),
            "alpha_mean": float(stats.get("alpha_mean", float("nan"))),
        })
    return np.stack(logits), checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1b", type=Path, required=True)
    ap.add_argument("--b21", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("H2_RESIDENT_BUNDLE"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=512)
    ap.add_argument("--features", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--experimental-resident-cache", action="store_true")
    args = ap.parse_args()

    args.reuse = args.h1b; args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, dense_sel, _, _ = h1b2.load_reused_models(args, device)
    selected = load_selected(args.b21 / "H1B21_RESULTS.json", device)

    model = copy.deepcopy(dense_sel)
    if args.experimental_resident_cache:
        enable_cached_block1(model)
    else:
        h1.enable_sparse_block1(model)
    b21.install_all_calibrated(model, selected); model.eval()

    model_path = args.out / "model_f32.bin"
    tensor_manifest = write_tensor_image(model_path, model.state_dict())
    calib_path = args.out / "calibration.bin"
    calib_records = write_calibration_binary(calib_path, selected)

    labels, canvases = h1.make_reveal_canvases(
        args.batch, args.rounds, device, args.seed + 81001)
    ref_logits, ref_checks = reference_trajectory(model, labels, canvases)
    labels_np = labels.detach().cpu().numpy().astype("<i4")
    canv_np = torch.stack(canvases).detach().cpu().numpy().astype("<i2")
    (args.out / "teacher_labels_i32.bin").write_bytes(labels_np.tobytes(order="C"))
    (args.out / "teacher_canvases_i16.bin").write_bytes(canv_np.tobytes(order="C"))
    np.save(args.out / "reference_logits_f32.npy", ref_logits.astype(np.float32))

    selected_meta = {
        f"L{li}_{side}": {"spec": v["spec"], "decoder": v["decoder"].detach().cpu().float().tolist()}
        for (li, side), v in selected.items()
    }
    manifest = {
        "format": "H2_RESIDENT_BUNDLE_V1",
        "numeric_reference": "float32; fixed-point mapping not yet frozen",
        "architecture": {
            "seq": h0.SEQ, "codebook": h0.CODEBOOK, "d": args.d,
            "layers": args.layers, "heads": args.heads, "head_dim": args.d // args.heads,
            "ff": args.ff, "features": args.features,
            "teacher_batch": args.batch, "teacher_rounds": args.rounds,
        },
        "contracts": {
            "persistent_block": 0, "monotonic_reveal": True,
            "Hg_decay_synchronized": True,
            "block1_execution": "experimental_resident_cache" if args.experimental_resident_cache else "validated_H1_sparse",
            "deeper_blocks_fresh_each_round": True,
            "host_stepwise_feedback_required": False,
        },
        "izh_calibration": selected_meta,
        "calibration_binary": {
            "file": calib_path.name, "sha256": sha256(calib_path),
            "max_decoder": MAX_DECODER, "records": calib_records,
        },
        "tensor_image": {
            "file": model_path.name, "sha256": sha256(model_path),
            "alignment_bytes": ALIGN, "tensors": tensor_manifest,
        },
        "teacher_input": {
            "labels_file": "teacher_labels_i32.bin", "labels_shape": list(labels_np.shape),
            "canvases_file": "teacher_canvases_i16.bin", "canvases_shape": list(canv_np.shape),
        },
        "reference": {
            "logits_file": "reference_logits_f32.npy",
            "logits_shape": list(ref_logits.shape), "checks": ref_checks,
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.out / "bundle_header.bin").write_bytes(struct.pack(
        "<8sIIIIIIII", b"H2RESV1\0", 1, h0.SEQ, args.d, args.layers,
        args.heads, args.ff, args.features, args.rounds))

    print(json.dumps({
        "stage": "H2_RESIDENT_BUNDLE_EXPORT", "out": str(args.out),
        "block1_execution": manifest["contracts"]["block1_execution"],
        "model_bytes": model_path.stat().st_size,
        "model_sha256": manifest["tensor_image"]["sha256"],
        "calibration_bytes": calib_path.stat().st_size,
        "teacher_shape": list(canv_np.shape),
        "reference_logits_shape": list(ref_logits.shape),
    }, indent=2))


if __name__ == "__main__":
    main()
