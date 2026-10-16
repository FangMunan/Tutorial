#!/usr/bin/env python3
"""Run the H3B1 single-chip weak-scaling sweep on physical SpiNNaker.

Each point is one global frozen block with N_local=32 tokens per worker.  The
point runner is invoked in an isolated output directory so every physical run
retains its own GFE reports, IOBUF, final state and JSON record.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_workers(text: str) -> list[int]:
    vals = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        p = int(item)
        if p < 2 or p > 16:
            raise ValueError(f"worker count outside [2,16]: {p}")
        vals.append(p)
    if not vals:
        raise ValueError("empty worker list")
    if vals != sorted(set(vals)):
        raise ValueError("workers must be unique and ascending")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplx", type=Path, required=True)
    ap.add_argument("--package-dir", type=Path, required=True)
    ap.add_argument("--source-config", type=Path,
                    default=Path.home() / ".spynnaker.cfg")
    ap.add_argument("--workers", default="2,4,8,16")
    ap.add_argument("--local-tokens", type=int, default=32)
    ap.add_argument("--n-boards", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--continue-on-fail", action="store_true")
    args = ap.parse_args()

    aplx = args.aplx.expanduser().resolve()
    package_dir = args.package_dir.expanduser().resolve()
    source_cfg = args.source_config.expanduser().resolve()
    workers = parse_workers(args.workers)
    if not aplx.is_file():
        raise SystemExit(f"missing APLX: {aplx}")
    if not package_dir.is_dir():
        raise SystemExit(f"missing package directory: {package_dir}")
    if not source_cfg.is_file():
        raise SystemExit(f"missing source config: {source_cfg}")

    root = (args.out if args.out else
            Path("spinnaker_backend/run/H3B1_single_chip") / utc_stamp())
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    point_runner = Path(__file__).with_name("run_h3b_single_chip.py").resolve()
    rows = []

    for p in workers:
        n = p * args.local_tokens
        package = package_dir / f"H3B1_P{p}_N{n}.npz"
        point_dir = root / f"P{p}_N{n}"
        point_dir.mkdir(parents=True, exist_ok=True)
        if not package.is_file():
            row = {
                "workers": p, "global_tokens": n, "PASS": False,
                "status": "MISSING_PACKAGE", "package": str(package),
            }
            rows.append(row)
            if not args.continue_on_fail:
                break
            continue

        cmd = [
            sys.executable, str(point_runner),
            "--aplx", str(aplx),
            "--package", str(package),
            "--source-config", str(source_cfg),
            "--n-boards", str(args.n_boards),
            "--out", str(point_dir),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True)
        (point_dir / "runner_stdout.txt").write_text(proc.stdout)
        (point_dir / "runner_stderr.txt").write_text(proc.stderr)

        result_path = point_dir / "H3B_BOARD_RESULT.json"
        if result_path.is_file():
            point = json.loads(result_path.read_text())
            row = {
                "workers": p,
                "global_tokens": n,
                "returncode": proc.returncode,
                "PASS": bool(point.get("PASS", False)),
                "status": point.get("status"),
                "same_chip": point.get("same_chip"),
                "all_arrays_bit_exact": point.get("all_arrays_bit_exact"),
                "max_worker_elapsed_ms": point.get("max_worker_elapsed_ms"),
                "total_q_evals": point.get("total_q_evals"),
                "total_k_evals": point.get("total_k_evals"),
                "total_v_evals": point.get("total_v_evals"),
                "total_tx_packets": point.get("total_tx_packets"),
                "total_rx_packets": point.get("total_rx_packets"),
                "total_tx_retries": point.get("total_tx_retries"),
                "placements": point.get("placements"),
                "exception_type": point.get("exception_type"),
                "exception": point.get("exception"),
                "result_json": str(result_path),
            }
        else:
            row = {
                "workers": p, "global_tokens": n,
                "returncode": proc.returncode, "PASS": False,
                "status": "NO_RESULT_JSON",
                "stderr_tail": proc.stderr[-4000:],
            }
        rows.append(row)
        print(json.dumps(row, indent=2))
        if not row["PASS"] and not args.continue_on_fail:
            break

    passed = [r for r in rows if r.get("PASS")]
    first_fail = next((r for r in rows if not r.get("PASS")), None)
    summary = {
        "stage": "H3B1_SINGLE_CHIP_WEAK_SCALING",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "local_tokens": args.local_tokens,
        "requested_workers": workers,
        "points": rows,
        "largest_pass_workers": max((r["workers"] for r in passed), default=None),
        "largest_pass_global_tokens": max((r["global_tokens"] for r in passed), default=None),
        "first_failure": first_fail,
        "all_requested_pass": len(rows) == len(workers) and all(r.get("PASS") for r in rows),
        "claim_boundary": (
            "H3B1 measures physical single-chip weak scaling for the frozen block "
            "with N_local fixed at 32. A failed point is a frontier candidate only; "
            "its cause must be diagnosed before assigning a resource bottleneck."
        ),
    }
    out = root / "H3B1_SWEEP_RESULT.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[H3B1] sweep record: {out}")


if __name__ == "__main__":
    main()
