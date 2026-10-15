#!/usr/bin/env python3
"""Run the preregistered H3A one-core token-count sweep on physical SpiNNaker.

This wrapper executes each prepared point in ascending N, stops after the first
valid CAPACITY_FAIL, and writes aggregate JSON/CSV summaries.  A capacity fail
is a successful measurement outcome, not an implementation failure.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

POINTS = (16, 24, 32, 36, 38, 40, 42, 44, 48)


def _run_point(root: Path, n: int, source_config: Path, n_boards: int, results_dir: Path) -> dict:
    pkg = root / "packages" / f"H3A_N{n}.npz"
    aplx = root / "snn_transform_h3a.aplx"
    point_out = results_dir / f"N{n}"
    cmd = [
        sys.executable, "-m", "spinnaker_backend.h3a_single_core.run_h3a_single_core_point",
        "--aplx", str(aplx),
        "--package", str(pkg),
        "--source-config", str(source_config),
        "--n-boards", str(n_boards),
        "--out", str(point_out),
    ]
    print("\n[H3A sweep]", " ".join(cmd), flush=True)
    cp = subprocess.run(cmd, cwd=root, text=True)
    result_path = point_out / "H3A_POINT_RESULT.json"
    if not result_path.is_file():
        raise RuntimeError(f"point N={n} produced no {result_path}")
    record = json.loads(result_path.read_text())
    if cp.returncode != 0 and record.get("status") != "CAPACITY_FAIL":
        raise RuntimeError(f"point N={n} runner exit={cp.returncode}, status={record.get('status')}")
    if not record.get("MEASUREMENT_VALID", False):
        raise RuntimeError(f"point N={n} measurement invalid")
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--source-config", type=Path, default=Path.home() / ".spynnaker.cfg")
    ap.add_argument("--n-boards", type=int, default=1)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--points", type=int, nargs="*", default=list(POINTS))
    args = ap.parse_args()

    root = args.bundle_root.expanduser().resolve()
    source_cfg = args.source_config.expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir = (args.results_dir or (root / "H3A_SWEEP_RESULTS" / stamp)).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    first_capacity_fail = None
    for n in args.points:
        rec = _run_point(root, int(n), source_cfg, args.n_boards, results_dir)
        h = rec.get("output_header", {})
        row = {
            "tokens": int(n),
            "status": rec.get("status"),
            "measurement_valid": bool(rec.get("MEASUREMENT_VALID")),
            "exec_pass": bool(rec.get("EXEC_PASS")),
            "arena_bytes": h.get("arena_bytes"),
            "heap_free_before": h.get("heap_free_before"),
            "heap_max_before": h.get("heap_max_before"),
            "heap_free_after": h.get("heap_free_after"),
            "heap_max_after": h.get("heap_max_after"),
            "elapsed_ms": h.get("elapsed_ms"),
            "q_evals": h.get("q_evals"),
            "k_evals": h.get("k_evals"),
            "v_evals": h.get("v_evals"),
            "result_json": str(results_dir / f"N{n}" / "H3A_POINT_RESULT.json"),
        }
        rows.append(row)
        print("[H3A sweep result]", json.dumps(row, indent=2), flush=True)
        if rec.get("status") == "CAPACITY_FAIL":
            first_capacity_fail = int(n)
            break

    pass_points = [r["tokens"] for r in rows if r["status"] == "EXEC_PASS"]
    summary = {
        "stage": "H3A_SINGLE_CORE_SWEEP",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "PASS": bool(rows) and all(r["measurement_valid"] for r in rows),
        "points_requested": [int(x) for x in args.points],
        "points_executed": [r["tokens"] for r in rows],
        "largest_exec_pass_N": max(pass_points) if pass_points else None,
        "first_capacity_fail_N": first_capacity_fail,
        "boundary_bracket": {
            "last_pass": max(pass_points) if pass_points else None,
            "first_fail": first_capacity_fail,
        },
        "rows": rows,
        "claim_boundary": "Baseline one-core capacity/execution sweep only; no chip-, board-, task-, training-, or energy-scaling claim.",
    }
    (results_dir / "H3A_SWEEP_RESULT.json").write_text(json.dumps(summary, indent=2))
    with (results_dir / "H3A_SWEEP_RESULT.csv").open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print("\n" + json.dumps(summary, indent=2))
    print(f"[H3A sweep] summary: {results_dir / 'H3A_SWEEP_RESULT.json'}")


if __name__ == "__main__":
    main()
