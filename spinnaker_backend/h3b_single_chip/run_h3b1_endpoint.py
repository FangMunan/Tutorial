#!/usr/bin/env python3
"""Endpoint-first H3B1 single-chip ceiling test.

Runs the already host-gated P=16, N_local=32, N_global=512 point directly.
For this ceiling test advanced monitor support is disabled in the isolated
frontend config by default so frontend support cores do not consume the worker
budget. This does not change the frozen H1.1-functional-v9 arithmetic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from spinnaker_backend.h2f_aplx import run_h2f_spalloc_board as h2fbase
from spinnaker_backend.h3b_single_chip import run_h3b_single_chip as h3b


def _precheck(package_path: Path) -> dict:
    z = np.load(package_path, allow_pickle=False)
    meta = json.loads(bytes(z["package_metadata_json_utf8"].tolist()).decode())
    checks = {
        "stage": meta.get("stage") == "H3B1_WEAK_SCALING",
        "workers_16": meta.get("workers") == 16,
        "local_tokens_32": meta.get("local_tokens") == 32,
        "global_tokens_512": meta.get("global_tokens") == 512,
        "balanced_local_changes": meta.get("balanced_local_changes") is True,
        "qkv_per_worker_72": meta.get("expected_qkv_evals_per_worker") == [72] * 16,
        "host_gate_pass": meta.get("host_distributed_gate", {}).get("PASS") is True,
        "host_final_bit_exact": meta.get("host_distributed_gate", {}).get("final_arrays_bit_exact") is True,
        "single_global_block": meta.get("scientific_contract", {}).get("single_global_block") is True,
        "same_chip_required": meta.get("scientific_contract", {}).get("same_chip_required") is True,
    }
    if not all(checks.values()):
        raise SystemExit("P16/N512 package precheck failed:\n" + json.dumps(checks, indent=2))
    print(json.dumps({
        "stage": "H3B1_N512_PRECHECK",
        "PASS": True,
        "workers": 16,
        "local_tokens": 32,
        "global_tokens": 512,
        "arena_bytes_per_worker": meta["worker_arena_bytes_requested"],
        "expected_qkv_evals_per_worker": 72,
        "checks": checks,
    }, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplx", type=Path, required=True)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--source-config", type=Path, default=Path.home() / ".spynnaker.cfg")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--precheck-only", action="store_true")
    ap.add_argument("--keep-advanced-monitors", action="store_true")
    args = ap.parse_args()

    package = args.package.expanduser().resolve()
    _precheck(package)
    if args.precheck_only:
        return

    # The shared H2F config helper intentionally preserves only a small Machine
    # key whitelist. For the endpoint test, extend the generated config locally
    # without changing the user's ~/.spynnaker.cfg.
    original_machine_cfg_text = h2fbase._machine_cfg_text
    if not args.keep_advanced_monitors:
        def endpoint_machine_cfg_text(cfg: dict[str, str]) -> str:
            text = original_machine_cfg_text(cfg)
            return text + "enable_advanced_monitor_support = False\n"
        h2fbase._machine_cfg_text = endpoint_machine_cfg_text

    argv = [
        "run_h3b_single_chip.py",
        "--aplx", str(args.aplx),
        "--package", str(package),
        "--source-config", str(args.source_config),
    ]
    if args.out is not None:
        argv += ["--out", str(args.out)]
    old_argv = sys.argv
    try:
        sys.argv = argv
        h3b.main()
    finally:
        sys.argv = old_argv
        h2fbase._machine_cfg_text = original_machine_cfg_text


if __name__ == "__main__":
    main()
