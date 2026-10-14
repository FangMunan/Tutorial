#!/usr/bin/env python3
"""H1.1 v8 — keep the decoder that actually won the held-out projective search.

v7 exposed a calibration bug rather than a hardware limitation: the grid search
found Q decoders with held-out projective errors around 3e-4, then discarded
those validated decoders and re-fit on the full highly collinear trajectory
design matrix.  That final unvalidated solve degraded head-0 held-out error to
~2.7e-2 and the block gate failed.

v8 changes only calibration bookkeeping: the exact quantized decoder/spec that
wins held-out selection is frozen into the HIR.  The integer runtime, numerical
formats, scientific operator and downstream gates are unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

import spinnaker_backend.run_h1_1_frozen_checkpoint_v2 as v2
import spinnaker_backend.run_h1_1_frozen_checkpoint_v3 as v3
import spinnaker_backend.run_h1_1_frozen_checkpoint_v7 as v7

REPORT = Path("spinnaker_backend/run/H1_1/H1_1_V8_PROJECTIVE_Q_SEARCH.json")


def search_q_specs_keep_winner(block, hir):
    fit_a, fit_y, _, _ = v7.collect_tokens(block, hir, batches=8, batch=8, seed=424242)
    val_a, val_y, _, _ = v7.collect_tokens(block, hir, batches=4, batch=8, seed=515151)
    biases = [2.0, 3.0, 4.0, 5.0, 6.0]
    gains = [0.25, 0.35, 0.5, 0.75, 1.0, 1.4]
    windows = [12, 16, 20, 24, 28, 32]
    all_rows: List[dict] = []
    selected_specs = []
    selected_decoders = []
    selected_rows = []

    for head in range(block.heads):
        best = None
        for bias in biases:
            for gain in gains:
                for window in windows:
                    spec = (bias, gain, window)
                    try:
                        dq, df = v7.fit_decoder_from_tokens(fit_a[head], fit_y[head], spec)
                        dmax = float(np.max(np.abs(df)))
                        if not np.isfinite(dmax) or dmax > 128.0:
                            continue
                        pred = v7.predict_tokens(val_a[head], spec, dq, v2.F_PQ)
                        raw_rel = v7.base.rel_np(pred, val_y[head])
                        proj_rel = v7.projective_rel(pred, val_y[head])
                        score = proj_rel + 0.02 * raw_rel
                        row = {
                            "head": head, "bias": bias, "gain": gain, "window_ms": window,
                            "raw_rel_l2": raw_rel, "projective_rel_l2": proj_rel,
                            "decoder_max_abs": dmax, "score": score,
                        }
                        all_rows.append(row)
                        if best is None or score < best[0]:
                            # Crucial v8 change: preserve the tested quantized decoder.
                            best = (score, spec, dq.copy(), row)
                    except (np.linalg.LinAlgError, FloatingPointError, OverflowError):
                        continue
        if best is None:
            raise RuntimeError(f"no valid Q operating point for head {head}")
        selected_specs.append(best[1])
        selected_decoders.append(best[2])
        selected_rows.append(best[3])

    report = {
        "method": "held-out projective search; winner decoder preserved exactly (no post-selection refit)",
        "grid": {"bias": biases, "gain": gains, "window_ms": windows},
        "selected": selected_rows,
        "top10": sorted(all_rows, key=lambda r: r["score"])[:10],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({"H1_1_v8_projective_q_search": report}, indent=2), flush=True)
    return selected_specs, selected_decoders


_original_export = v7.base.export_hir

def export_v8(block):
    hir = _original_export(block)
    qspec, qdec = search_q_specs_keep_winner(block, hir)
    kdec, kreport = v7.calibrate_k_decoders(block, hir)
    hir.q_specs = qspec
    hir.q_decoders = qdec
    hir.k_decoders = kdec
    print(json.dumps({"H1_1_v8_k_calibration": kreport}, indent=2), flush=True)
    return hir


if __name__ == "__main__":
    v2.export_hir = export_v8
    v2.ActualCheckpointRuntime = v3.IndexCorrectRuntime
    v2.main()
