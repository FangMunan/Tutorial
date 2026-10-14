#!/usr/bin/env python3
"""H2F physical-board runner v2.

The target deliberately preserves VCPU user0 as the executed data-spec
region-table pointer. SpiNNMan needs that pointer after FINISHED to locate the
output SDRAM region. This wrapper reuses the tested v1 runner implementation,
replaces only the obsolete ``user0_magic`` acceptance check with a non-zero
region-table-pointer check, and leaves all scientific gates unchanged.

The import fallback allows the two runner files to be placed side-by-side in a
flat JupyterLab upload directory as well as inside the repository package.
"""
from __future__ import annotations

import argparse
import json
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from spinnaker_backend.h2f_aplx import run_h2f_spalloc_board as base
except ImportError:
    import run_h2f_spalloc_board as base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplx", type=Path, required=True)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument(
        "--source-config", type=Path,
        default=Path.home() / ".spynnaker.cfg")
    ap.add_argument("--n-boards", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    aplx = args.aplx.expanduser().resolve()
    package_path = args.package.expanduser().resolve()
    source_cfg = args.source_config.expanduser().resolve()
    if not aplx.is_file():
        raise SystemExit(f"missing APLX: {aplx}")
    if not package_path.is_file():
        raise SystemExit(f"missing H2F package: {package_path}")
    package = base._load_package(package_path)

    run_dir = (
        args.out if args.out else
        Path("spinnaker_backend/run/H2F_spalloc_board") / base._utc_stamp()
    ).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    staged_aplx = run_dir / "snn_transform_h2f.aplx"
    shutil.copy2(aplx, staged_aplx)
    staged_package = run_dir / "H2F_BOARD_PACKAGE.npz"
    shutil.copy2(package_path, staged_package)

    machine_cfg = base._read_source_machine_config(source_cfg)
    local_cfg, cfg_home, hidden_cfg = base._prepare_configs(
        run_dir, machine_cfg)

    record: dict[str, Any] = {
        "stage": "H2F_FULL_RESIDENT_BOARD",
        "runner_version": "v2-preserve-dse-user0",
        "PASS": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "aplx_size_bytes": staged_aplx.stat().st_size,
        "package": package["meta"],
        "machine": base._safe_machine_summary(machine_cfg),
        "source_config": str(source_cfg),
        "generated_gfe_config": str(local_cfg),
        "isolated_gfe_home": str(cfg_home),
        "isolated_gfe_user_config": str(hidden_cfg),
        "software_versions": {
            name: base._version(name) for name in (
                "sPyNNaker", "SpiNNakerGraphFrontEnd",
                "SpiNNFrontEndCommon", "SpiNNMan", "SpiNNMachine")
        },
        "claim_boundary": (
            "PASS proves full six-step frozen-HIR resident fixed-point block "
            "execution on physical SpiNNaker for this preregistered trajectory."
        ),
    }

    result_path = run_dir / "H2F_BOARD_RESULT.json"
    try:
        runtime = base._run_frontend(
            run_dir, cfg_home, staged_aplx.name, package, args.n_boards)

        # v1 expected user0 to be an application magic. H2F target v2 keeps
        # user0 as the data-spec pointer so the host can locate output region 3.
        checks = runtime["checks"]
        checks.pop("user0_magic", None)
        checks["user0_region_table_pointer_nonzero"] = (
            int(runtime["user_registers"]["user0"]) != 0)
        runtime["PASS"] = all(bool(x) for x in checks.values())

        iobuf = runtime.pop("iobuf")
        iobuf_path = run_dir / "core_iobuf.txt"
        iobuf_path.write_text(iobuf)
        record.update(runtime)
        record["iobuf_file"] = str(iobuf_path)
    except Exception as exc:
        record.update({
            "PASS": False,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        })
        result_path.write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2))
        raise SystemExit(2) from exc

    result_path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    if not record["PASS"]:
        raise SystemExit(2)
    print(f"[H2F] PASS: {result_path}")


if __name__ == "__main__":
    main()
