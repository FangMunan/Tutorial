#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3I-2 stability ablation wrapper for the exact developed P3 image gate.

This file deliberately does NOT replace the P3 E/M learning rule.  It imports the
accepted P3I-A implementation and changes only two experimentally isolated gains:

1) optional internal forward-plasticity scale applied uniformly to
   WQ/WK/WV/WO/W1/W2 learning rates after the already-frozen P3 width scaling;
   WR and reciprocal-population rates are left unchanged.
2) optional plain (unweighted) startup MSDL readout scaffold, while preserving the
   selected online mismatch weighting during the uninterrupted seven-weight phase.

The second intervention removes a confound in P3I-A: the first CB run changed the
startup scaffold and the online learner simultaneously.  R6A.2 only supported CB
as an online image-learning repair, so P3I-2 tests it that way.

No backpropagation, autograd, torch optimizer, learned inverse network, replay, or
checkpoint initialization is introduced by this wrapper.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np

import p3i_cb_exact_msdl as core

_ORIG_INIT_MODEL = core.init_model
_ORIG_TRAIN_READOUT = core.train_readout_msdl

INTERNAL_SCALE = float(os.environ.get("P3I_INTERNAL_LR_SCALE", "1.0"))
PLAIN_SCAFFOLD = os.environ.get("P3I_PLAIN_SCAFFOLD", "1").lower() not in ("0", "false", "no")
EXPERIMENT_NAME = os.environ.get("P3I_EXPERIMENT_NAME", "p3i2")


def init_model_scaled(cfg, seed):
    model = _ORIG_INIT_MODEL(cfg, seed)
    # One scalar only: preserve all P3 relative rates and the frozen sqrt(24/D)
    # width normalization; reduce total internal synaptic movement uniformly.
    for key in ("lrq", "lrk", "lrv", "lro", "lr1", "lr2"):
        model[key] *= INTERNAL_SCALE
    return model


def train_readout_scaffold(features, targets, target_weights, n_paths, lr, seed):
    # CB should be tested as an ONLINE mismatch intervention, not silently alter
    # the initial readout scaffold.  The scaffold is still learned by exact MSDL.
    weights = np.ones_like(target_weights) if PLAIN_SCAFFOLD else target_weights
    return _ORIG_TRAIN_READOUT(features, targets, weights, n_paths, lr, seed)


core.init_model = init_model_scaled
core.train_readout_msdl = train_readout_scaffold


def _out_path(argv):
    try:
        i = argv.index("--out")
        return Path(argv[i + 1])
    except (ValueError, IndexError):
        return None


if __name__ == "__main__":
    out = _out_path(sys.argv)
    core.main()
    # Append intervention metadata after the untouched core run has finished.
    if out is not None:
        p = out / "metrics.json"
        if p.exists():
            d = json.loads(p.read_text())
            d["p3i2_stability_ablation"] = {
                "experiment_name": EXPERIMENT_NAME,
                "uniform_internal_forward_lr_scale": INTERNAL_SCALE,
                "plain_msdl_startup_scaffold": PLAIN_SCAFFOLD,
                "online_variant": d.get("variant"),
                "WR_lr_scaled": False,
                "reciprocal_lr_scaled": False,
                "candidate_backprop": False,
                "scientific_boundary": (
                    "Exact P3 E/M dynamics and local ME^T rule unchanged; this is a "
                    "gain/stability and scaffold-confound ablation only."
                ),
            }
            p.write_text(json.dumps(d, indent=2))
            print("P3I2_METADATA", json.dumps(d["p3i2_stability_ablation"]), flush=True)
