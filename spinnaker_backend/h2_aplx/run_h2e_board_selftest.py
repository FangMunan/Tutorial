#!/usr/bin/env python3
"""Run the H2E APLX arithmetic self-test on a directly reachable board.

This is the low-level ybug fallback.  The primary Manchester route is the
spalloc-compatible ``run_h2e_spalloc_board_selftest.py`` runner.

The H2E application waits at the standard SYNC0 barrier, executes the arithmetic
self-test, writes a result to IOBUF + VCPU user registers, and exits.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PASS_TOKEN = "SNN_TRANSFORM_H2_APLX_SELFTEST PASS"
FAIL_TOKEN = "SNN_TRANSFORM_H2_APLX_SELFTEST FAIL"


def _find_ybug(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    for root in (os.environ.get("SPINN_DIRS"), os.environ.get("SPINN_INSTALL_DIR")):
        if root:
            candidates.append(Path(root).expanduser() / "tools" / "ybug")
    on_path = shutil.which("ybug")
    if on_path:
        candidates.append(Path(on_path))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find executable ybug. Set SPINN_DIRS/SPINN_INSTALL_DIR, "
        "put ybug on PATH, or pass --ybug /path/to/ybug."
    )


def _make_commands(aplx_name: str, core: int, app_id: int, boot: bool,
                   keep_app: bool, wait_s: int) -> str:
    lines: list[str] = []
    if boot:
        lines.extend(["boot", "sleep 2"])
    lines.extend([
        f"app_load {aplx_name} . {core} {app_id}",
        f"app_sig . {app_id} sync0",
        f"sleep {wait_s}",
        f"iobuf {core}",
        "ps",
    ])
    if not keep_app:
        lines.append(f"app_stop {app_id}")
    lines.append("quit")
    return "\n".join(lines) + "\n"


def _classify(transcript: str, returncode: int) -> tuple[bool, str]:
    if FAIL_TOKEN in transcript:
        return False, "application_reported_fail"
    if PASS_TOKEN in transcript and returncode == 0:
        return True, "pass_token_seen"
    if PASS_TOKEN in transcript:
        return False, f"pass_token_seen_but_ybug_returncode_{returncode}"
    return False, "pass_token_missing"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    ap.add_argument("--aplx", type=Path,
                    default=Path(__file__).with_name("snn_transform_h2.aplx"))
    ap.add_argument("--ybug", default=None)
    ap.add_argument("--core", type=int, default=1)
    ap.add_argument("--app-id", type=int, default=31)
    ap.add_argument("--wait-s", type=int, default=1)
    ap.add_argument("--boot", action="store_true")
    ap.add_argument("--keep-app", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not 1 <= args.core <= 17:
        raise SystemExit("--core must be in virtual-core range 1..17")
    if not 1 <= args.app_id <= 255:
        raise SystemExit("--app-id must be in 1..255")
    if args.wait_s < 0:
        raise SystemExit("--wait-s must be >= 0")

    aplx = args.aplx.expanduser().resolve()
    if not aplx.is_file() or aplx.stat().st_size == 0:
        raise SystemExit(f"APLX file is missing or empty: {aplx}")
    ybug = _find_ybug(args.ybug)

    if args.out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = Path("spinnaker_backend/run/H2E_board_direct") / stamp
    else:
        out = args.out
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    staged_aplx = out / "snn_transform_h2.aplx"
    if aplx != staged_aplx:
        shutil.copy2(aplx, staged_aplx)

    commands = _make_commands(
        staged_aplx.name, args.core, args.app_id,
        args.boot, args.keep_app, args.wait_s)
    (out / "board_commands.ybug").write_text(commands)

    proc = subprocess.run(
        [str(ybug), args.board], input=commands, text=True, cwd=out,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    transcript = proc.stdout or ""
    (out / "ybug_transcript.txt").write_text(transcript)
    print(transcript, end="" if transcript.endswith("\n") else "\n")

    passed, reason = _classify(transcript, proc.returncode)
    record = {
        "stage": "H2E_BOARD_APLX_SELFTEST_DIRECT_YBUG",
        "PASS": passed,
        "reason": reason,
        "board": args.board,
        "core": args.core,
        "app_id": args.app_id,
        "boot_requested": bool(args.boot),
        "wait_s": args.wait_s,
        "ybug": str(ybug),
        "aplx_source": str(aplx),
        "aplx_size_bytes": staged_aplx.stat().st_size,
        "ybug_returncode": proc.returncode,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "PASS proves only one-core H2 arithmetic self-test execution; "
            "it is not the full frozen-HIR resident block board test."
        ),
    }
    (out / "H2E_BOARD_SELFTEST.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    if not passed:
        print(f"[H2E-board] FAIL; inspect {out / 'ybug_transcript.txt'}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
