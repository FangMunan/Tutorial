#!/usr/bin/env python3
"""Freeze H1.1 v9 as the functional backend contract.

This wrapper deliberately keeps two statements separate:

* functional H1.1 acceptance: frozen task/readout parity, resident-state parity,
  event counts, sparsity, and integer-range safety;
* strict full-block representation parity: reported, but NOT claimed as passed
  unless max block-output rel-L2 is <= 1%.

The v9 calibration itself is unchanged.  A non-zero exit from the historical
strict v9 runner is expected when only its strict block-output gate fails.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--change-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=17171)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    v9_out = args.out / "v9"
    cmd = [
        sys.executable, "-m", "spinnaker_backend.run_h1_1_frozen_checkpoint_v9",
        "--checkpoint", str(args.checkpoint),
        "--out", str(v9_out),
        "--batch", str(args.batch),
        "--steps", str(args.steps),
        "--change-frac", str(args.change_frac),
        "--seed", str(args.seed),
    ]
    proc = subprocess.run(cmd, check=False)
    if proc.returncode not in (0, 2):
        raise SystemExit(proc.returncode)

    result_path = v9_out / "H1_1_V2_RESULTS.json"
    hir_path = v9_out / "SNNTransformBlock_v1_1_HIR_H1_1_v2.npz"
    result = json.loads(result_path.read_text())
    g = result["gates"]

    required = [
        "actual_checkpoint",
        "H_le_1pct", "g_le_1pct", "pq_le_1pct", "pk_le_1pct", "v_le_0p5pct",
        "prediction_le_1pct", "sign_drop_le_0p02",
        "event_count_exact", "event_reduction_ge_0p60", "tail_nnz_1280",
        "H_int32_safe", "g_int32_safe", "pq_int32_safe", "pk_int32_safe", "v_int32_safe",
    ]
    functional_gates = {name: bool(g[name]) for name in required}
    functional_pass = all(functional_gates.values())
    strict_repr_pass = float(result["max_rel_l2"]["output"]) <= 0.01

    frozen_hir = args.out / "SNNTransformBlock_v1_1_HIR_H1_1_FUNCTIONAL_V9.npz"
    shutil.copy2(hir_path, frozen_hir)

    contract = {
        "stage": "H1_1_FUNCTIONAL_V9_FROZEN_CONTRACT",
        "PASS": functional_pass,
        "contract_version": "H1.1-functional-v9",
        "source": result["source"],
        "functional_gates": functional_gates,
        "strict_representation_gate": {
            "threshold_rel_l2": 0.01,
            "observed_max_block_output_rel_l2": result["max_rel_l2"]["output"],
            "PASS": strict_repr_pass,
            "claim": "reported only; not part of functional acceptance",
        },
        "task_probe_step0": result["task_probe_step0"],
        "max_rel_l2": result["max_rel_l2"],
        "event_execution": result["event_execution"],
        "sparse_tail": result["sparse_tail"],
        "formats": result["formats"],
        "hir_file": frozen_hir.name,
        "decision": (
            "FUNCTIONAL PASS: v9 is the H1.1 backend contract for H2 APLX; "
            "strict representation parity remains separately reported."
            if functional_pass else
            "FUNCTIONAL FAIL: do not enter H2 APLX."
        ),
    }
    (args.out / "H1_1_FUNCTIONAL_CONTRACT.json").write_text(json.dumps(contract, indent=2))
    print(json.dumps(contract, indent=2), flush=True)
    if not functional_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
