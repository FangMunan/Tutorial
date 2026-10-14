#!/usr/bin/env python3
"""H2E-board: run the H2 arithmetic APLX on one real SpiNNaker core via spalloc.

Primary Manchester route.  The runner reuses the machine/spalloc settings from
the already-working ``~/.spynnaker.cfg`` but does *not* require the user to
maintain a second permanent GraphFrontEnd config.

SpiNNakerGraphFrontEnd 7.4.1 has an important configuration rule: before it
reads a run-local ``spiNNakerGraphFrontEnd.cfg``, it insists that a user config
named ``~/.spiNNakerGraphFrontEnd.cfg`` exists.  A local file alone is therefore
insufficient and triggers ``NoConfigFoundException``.  To keep this board test
self-contained and non-invasive, this script creates an isolated temporary
HOME inside the run directory, writes the required hidden user config there,
and temporarily points HOME/XDG_CONFIG_HOME at it only while GraphFrontEnd is
running.  The user's real home configuration is never overwritten.

The custom APLX is represented as one machine vertex with
``ExecutableType.SYNC``.  The binary waits at SYNC0, runs the arithmetic test,
writes an IOBUF PASS/FAIL line plus four VCPU user-register signatures, then
calls ``spin1_exit``.

PASS requires the current-run IOBUF token and matching VCPU signatures:

    user0 = 0x48324531 (H2E1 magic)
    user1 = 0          (self-test return code)
    user2 = chip_id
    user3 = (core_id << 16) | app_id

A PASS proves physical one-core execution of the H2 arithmetic self-test.  It
is not yet the H2F full frozen-HIR resident-block board test.
"""
from __future__ import annotations

import argparse
import configparser
import importlib.metadata
import json
import os
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PASS_TOKEN = "SNN_TRANSFORM_H2_APLX_SELFTEST PASS"
FAIL_TOKEN = "SNN_TRANSFORM_H2_APLX_SELFTEST FAIL"
MAGIC = 0x48324531
GFE_CFG = "spiNNakerGraphFrontEnd.cfg"
HIDDEN_GFE_CFG = f".{GFE_CFG}"

# Only copy options shared by SpiNNMan/GFE.  Copying the whole PyNN config can
# fail GFE's strict local-config validation because PyNN has extra sections.
_MACHINE_KEYS = (
    ("machineName", "machinename"),
    ("version", "version"),
    ("spalloc_server", "spalloc_server"),
    ("spalloc_port", "spalloc_port"),
    ("spalloc_user", "spalloc_user"),
    ("spalloc_group", "spalloc_group"),
    ("virtual_board", "virtual_board"),
    ("width", "width"),
    ("height", "height"),
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _read_source_machine_config(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"SpiNNaker source config not found: {path}. "
            "Pass --source-config if your working config is elsewhere."
        )
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(path)
    if not parser.has_section("Machine"):
        raise ValueError(f"No [Machine] section in {path}")

    # ConfigParser is case-insensitive by default; use lower-case lookups.
    src = {k.lower(): v.strip() for k, v in parser.items("Machine")}
    out: dict[str, str] = {}
    for canonical, lookup in _MACHINE_KEYS:
        if lookup in src:
            out[canonical] = src[lookup]

    server = out.get("spalloc_server", "None")
    machine = out.get("machineName", "None")
    virtual = out.get("virtual_board", "False").lower() in {
        "1", "true", "yes", "on"}
    use_spalloc = server.lower() not in {"", "none"}
    use_direct = machine.lower() not in {"", "none"}
    if virtual:
        raise ValueError("H2E-board requires physical hardware; virtual_board=True")
    if use_spalloc == use_direct:
        raise ValueError(
            "Expected exactly one physical machine source: spalloc_server or "
            "machineName. Current source config is ambiguous/incomplete."
        )
    return out


def _machine_cfg_text(machine_cfg: dict[str, str]) -> str:
    lines = ["[Machine]"]
    for canonical, _ in _MACHINE_KEYS:
        if canonical in machine_cfg:
            lines.append(f"{canonical} = {machine_cfg[canonical]}")
    return "\n".join(lines) + "\n"


def _write_gfe_cfg(run_dir: Path, machine_cfg: dict[str, str]) -> Path:
    # Run-local config.  GraphFrontEnd reads this strictly *after* it has found
    # a user config in HOME, hence the isolated HOME prepared below as well.
    cfg = run_dir / GFE_CFG
    cfg.write_text(_machine_cfg_text(machine_cfg), encoding="utf-8")
    try:
        cfg.chmod(0o600)
    except OSError:
        pass
    return cfg


def _prepare_isolated_gfe_home(
        run_dir: Path, machine_cfg: dict[str, str]) -> tuple[Path, Path]:
    """Create the hidden GFE user config required by GraphFrontEnd 7.4.1.

    The directory is deliberately inside the run output so no permanent file is
    written to the user's real HOME.  XDG_CONFIG_HOME is also redirected while
    the frontend is active so config discovery cannot accidentally pick another
    user-level GraphFrontEnd config.
    """
    cfg_home = run_dir / "_gfe_home"
    xdg_home = cfg_home / ".config"
    xdg_home.mkdir(parents=True, exist_ok=True)
    hidden_cfg = cfg_home / HIDDEN_GFE_CFG
    hidden_cfg.write_text(_machine_cfg_text(machine_cfg), encoding="utf-8")
    try:
        hidden_cfg.chmod(0o600)
    except OSError:
        pass
    return cfg_home, hidden_cfg


def _safe_machine_summary(machine_cfg: dict[str, str]) -> dict[str, Any]:
    server = machine_cfg.get("spalloc_server", "None")
    machine = machine_cfg.get("machineName", "None")
    return {
        "route": (
            "spalloc" if server.lower() not in {"", "none"}
            else "direct_frontend"),
        "spalloc_server": (
            None if server.lower() in {"", "none"} else server),
        "spalloc_port": machine_cfg.get("spalloc_port"),
        "machineName": (
            None if machine.lower() in {"", "none"} else machine),
        "version": machine_cfg.get("version"),
        "spalloc_user_configured": (
            machine_cfg.get("spalloc_user", "None").lower()
            not in {"", "none"}),
        "spalloc_group_configured": (
            machine_cfg.get("spalloc_group", "None").lower()
            not in {"", "none"}),
    }


def _run_frontend(run_dir: Path, n_boards: int, cfg_home: Path) -> dict[str, Any]:
    # Imports are delayed so `python -m py_compile` remains possible on systems
    # without the SpiNNaker Python stack.  The GraphFrontEnd constructor itself
    # resets configuration and then looks for ~/.spiNNakerGraphFrontEnd.cfg, so
    # HOME must be redirected before front_end.setup() is called.
    try:
        from spinn_utilities.overrides import overrides
        from spinn_machine import CoreSubsets
        from spinnman.model.enums import ExecutableType, UserRegister
        from pacman.model.graphs.machine import SimpleMachineVertex
        from pacman.model.resources import ConstantSDRAM
        from spinn_front_end_common.abstract_models import AbstractHasAssociatedBinary
        from spinn_front_end_common.data import FecDataView
        import spinnaker_graph_front_end as front_end
    except ImportError as exc:
        raise RuntimeError(
            "SpiNNakerGraphFrontEnd stack is not available in this Python "
            "environment. Use the same environment as your working sPyNNaker "
            "installation and install the matching SpiNNakerGraphFrontEnd "
            "release; do not upgrade the stack blindly."
        ) from exc

    class H2EBoardVertex(SimpleMachineVertex, AbstractHasAssociatedBinary):
        def __init__(self) -> None:
            super().__init__(ConstantSDRAM(0), label="H2E arithmetic self-test")

        @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
        def get_binary_file_name(self) -> str:
            return "snn_transform_h2.aplx"

        @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
        def get_binary_start_type(self):
            return ExecutableType.SYNC

    old_cwd = Path.cwd()
    old_home = os.environ.get("HOME")
    old_xdg = os.environ.get("XDG_CONFIG_HOME")
    vertex = H2EBoardVertex()
    setup_done = False
    try:
        os.chdir(run_dir)
        os.environ["HOME"] = str(cfg_home)
        os.environ["XDG_CONFIG_HOME"] = str(cfg_home / ".config")

        front_end.setup(
            n_boards_required=n_boards,
            model_binary_folder=str(run_dir),
        )
        setup_done = True
        front_end.add_machine_vertex_instance(vertex)

        # For SYNC executables this waits for SYNC0 readiness, sends the start
        # signal and then waits for CPUState.FINISHED.
        front_end.run_until_complete()

        placement = FecDataView.get_placement_of_vertex(vertex)
        app_id = int(FecDataView.get_app_id())
        txrx = FecDataView.get_transceiver()

        subset = CoreSubsets()
        subset.add_processor(placement.x, placement.y, placement.p)
        buffers = list(txrx.get_iobuf(subset))
        iobuf = "\n".join(buf.iobuf for buf in buffers)

        regs = {
            "user0": int(txrx.read_user(
                placement.x, placement.y, placement.p,
                UserRegister.USER_0)),
            "user1": int(txrx.read_user(
                placement.x, placement.y, placement.p,
                UserRegister.USER_1)),
            "user2": int(txrx.read_user(
                placement.x, placement.y, placement.p,
                UserRegister.USER_2)),
            "user3": int(txrx.read_user(
                placement.x, placement.y, placement.p,
                UserRegister.USER_3)),
        }
        expected_chip = (int(placement.x) << 8) | int(placement.y)
        expected_user3 = (int(placement.p) << 16) | (app_id & 0xFFFF)

        checks = {
            "iobuf_pass_token": PASS_TOKEN in iobuf,
            "iobuf_no_fail_token": FAIL_TOKEN not in iobuf,
            "user0_magic": regs["user0"] == MAGIC,
            "user1_rc_zero": regs["user1"] == 0,
            "user2_chip_match": regs["user2"] == expected_chip,
            "user3_core_app_match": regs["user3"] == expected_user3,
        }
        passed = all(checks.values())

        return {
            "PASS": passed,
            "checks": checks,
            "placement": {
                "x": int(placement.x),
                "y": int(placement.y),
                "p": int(placement.p),
            },
            "app_id": app_id,
            "expected_chip_id": expected_chip,
            "expected_user3": expected_user3,
            "user_registers": regs,
            "iobuf": iobuf,
        }
    finally:
        if setup_done:
            try:
                front_end.stop()
            except Exception:
                # Preserve the primary test exception while still attempting to
                # release a Manchester spalloc allocation.
                traceback.print_exc()
        os.chdir(old_cwd)
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
        if old_xdg is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = old_xdg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--aplx", type=Path,
        default=Path(__file__).with_name("snn_transform_h2.aplx"),
        help="cross-compiled H2E APLX image")
    ap.add_argument(
        "--source-config", type=Path,
        default=Path.home() / ".spynnaker.cfg",
        help=(
            "working sPyNNaker config; only compatible [Machine] options "
            "are copied"))
    ap.add_argument(
        "--n-boards", type=int, default=1,
        help="spalloc allocation size; one board is sufficient")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.n_boards < 1:
        raise SystemExit("--n-boards must be >= 1")

    aplx = args.aplx.expanduser().resolve()
    source_cfg = args.source_config.expanduser().resolve()
    if not aplx.is_file() or aplx.stat().st_size == 0:
        raise SystemExit(f"APLX file is missing or empty: {aplx}")

    if args.out is None:
        run_dir = Path("spinnaker_backend/run/H2E_spalloc_board") / _utc_stamp()
    else:
        run_dir = args.out
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    staged_aplx = run_dir / "snn_transform_h2.aplx"
    if staged_aplx != aplx:
        shutil.copy2(aplx, staged_aplx)

    machine_cfg = _read_source_machine_config(source_cfg)
    local_cfg = _write_gfe_cfg(run_dir, machine_cfg)
    cfg_home, hidden_cfg = _prepare_isolated_gfe_home(run_dir, machine_cfg)
    safe_machine = _safe_machine_summary(machine_cfg)

    base_record: dict[str, Any] = {
        "stage": "H2E_SPALLOC_BOARD_APLX_SELFTEST",
        "PASS": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "aplx_source": str(aplx),
        "aplx_staged": str(staged_aplx),
        "aplx_size_bytes": staged_aplx.stat().st_size,
        "source_config": str(source_cfg),
        "generated_gfe_config": str(local_cfg),
        "isolated_gfe_home": str(cfg_home),
        "isolated_gfe_user_config": str(hidden_cfg),
        "machine": safe_machine,
        "requested_boards": args.n_boards,
        "software_versions": {
            "sPyNNaker": _version("sPyNNaker"),
            "SpiNNakerGraphFrontEnd": _version("SpiNNakerGraphFrontEnd"),
            "SpiNNFrontEndCommon": _version("SpiNNFrontEndCommon"),
            "SpiNNMan": _version("SpiNNMan"),
            "SpiNNMachine": _version("SpiNNMachine"),
            "PACMAN": _version("SpiNNaker-PACMAN") or _version("PACMAN"),
        },
        "claim_boundary": (
            "PASS proves physical execution of the one-core H2 arithmetic "
            "self-test APLX. It does not prove full frozen-HIR resident block "
            "execution; that is H2F."
        ),
    }

    result_path = run_dir / "H2E_SPALLOC_BOARD_SELFTEST.json"
    try:
        runtime = _run_frontend(run_dir, args.n_boards, cfg_home)
        iobuf = runtime.pop("iobuf")
        (run_dir / "core_iobuf.txt").write_text(iobuf, encoding="utf-8")
        base_record.update(runtime)
        base_record["iobuf_file"] = str(run_dir / "core_iobuf.txt")
    except Exception as exc:
        base_record.update({
            "PASS": False,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        })
        result_path.write_text(
            json.dumps(base_record, indent=2), encoding="utf-8")
        print(json.dumps(base_record, indent=2))
        raise SystemExit(2) from exc

    result_path.write_text(
        json.dumps(base_record, indent=2), encoding="utf-8")
    print(json.dumps(base_record, indent=2))
    if not base_record["PASS"]:
        raise SystemExit(2)

    print(f"[H2E-board] PASS: {result_path}")


if __name__ == "__main__":
    main()
