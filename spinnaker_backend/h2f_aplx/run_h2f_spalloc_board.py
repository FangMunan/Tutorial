#!/usr/bin/env python3
"""H2F: execute the full frozen resident block on physical SpiNNaker.

One data-specification load writes the complete static HIR and six-step input
trajectory to SDRAM.  The custom APLX executes all steps autonomously with
resident (H,g,pq,pk,v).  After FINISHED, this runner reads the final output
region once and requires bit-exact equality to the H2D oracle stored in the
package.
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

import numpy as np

H2F_BOARD_MAGIC = 0x48324642
H2F_OUTPUT_MAGIC = 0x4832464F
PASS_TOKEN = "SNN_TRANSFORM_H2F_FULL_BLOCK PASS"
FAIL_TOKEN = "SNN_TRANSFORM_H2F_FULL_BLOCK FAIL"
GFE_CFG = "spiNNakerGraphFrontEnd.cfg"
HIDDEN_GFE_CFG = f".{GFE_CFG}"

_MACHINE_KEYS = (
    ("machineName", "machinename"), ("version", "version"),
    ("spalloc_server", "spalloc_server"), ("spalloc_port", "spalloc_port"),
    ("spalloc_user", "spalloc_user"), ("spalloc_group", "spalloc_group"),
    ("virtual_board", "virtual_board"), ("width", "width"), ("height", "height"),
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _read_source_machine_config(path: Path) -> dict[str, str]:
    parser = configparser.ConfigParser(interpolation=None)
    if not path.is_file():
        raise FileNotFoundError(path)
    parser.read(path)
    if not parser.has_section("Machine"):
        raise ValueError(f"No [Machine] section in {path}")
    src = {k.lower(): v.strip() for k, v in parser.items("Machine")}
    out = {canonical: src[lookup] for canonical, lookup in _MACHINE_KEYS if lookup in src}
    server = out.get("spalloc_server", "None")
    machine = out.get("machineName", "None")
    virtual = out.get("virtual_board", "False").lower() in {"1", "true", "yes", "on"}
    if virtual:
        raise ValueError("H2F requires physical hardware")
    if (server.lower() not in {"", "none"}) == (machine.lower() not in {"", "none"}):
        raise ValueError("Expected exactly one of spalloc_server or machineName")
    return out


def _machine_cfg_text(cfg: dict[str, str]) -> str:
    lines = ["[Machine]"]
    for canonical, _ in _MACHINE_KEYS:
        if canonical in cfg:
            lines.append(f"{canonical} = {cfg[canonical]}")
    return "\n".join(lines) + "\n"


def _prepare_configs(run_dir: Path, machine_cfg: dict[str, str]) -> tuple[Path, Path, Path]:
    text = _machine_cfg_text(machine_cfg)
    local = run_dir / GFE_CFG
    local.write_text(text)
    home = run_dir / "_gfe_home"
    (home / ".config").mkdir(parents=True, exist_ok=True)
    hidden = home / HIDDEN_GFE_CFG
    hidden.write_text(text)
    for p in (local, hidden):
        try: p.chmod(0o600)
        except OSError: pass
    return local, home, hidden


def _safe_machine_summary(cfg: dict[str, str]) -> dict[str, Any]:
    server = cfg.get("spalloc_server", "None")
    machine = cfg.get("machineName", "None")
    return {
        "route": "spalloc" if server.lower() not in {"", "none"} else "direct_frontend",
        "spalloc_server": None if server.lower() in {"", "none"} else server,
        "machineName": None if machine.lower() in {"", "none"} else machine,
        "version": cfg.get("version"),
        "spalloc_user_configured": cfg.get("spalloc_user", "None").lower() not in {"", "none"},
    }


def _load_package(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z["package_metadata_json_utf8"].tolist()).decode("utf-8"))
    if meta.get("format") != "SNNTransform-H2F-board-package-v1":
        raise ValueError(f"unexpected package format {meta.get('format')!r}")
    return {
        "meta": meta,
        "header": np.ascontiguousarray(z["header_u32"], dtype="<u4"),
        "hir": np.ascontiguousarray(z["hir_blob_u32"], dtype="<u4"),
        "trajectory": np.ascontiguousarray(z["trajectory_u32"], dtype="<u4"),
        "oracle": {name: np.ascontiguousarray(z[f"oracle_{name}"], dtype="<i4")
                   for name in ("y", "H", "g", "pq", "pk", "v")},
    }


def _decode_output(words: np.ndarray, meta: dict) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    w = np.asarray(words, dtype="<u4")
    header = {
        "magic": int(w[0]), "rc": int(w[1]), "steps": int(w[2]),
        "q_evals": int(w[3]), "k_evals": int(w[4]), "v_evals": int(w[5]),
        "chip": int(w[6]), "core_app": int(w[7]),
    }
    p = 8
    N, D, H, M = (int(meta[k]) for k in ("tokens", "d", "heads", "features"))
    dh = D // H
    shapes = {
        "y": (N, D), "H": (H, M, dh), "g": (H, M),
        "pq": (N, H, M), "pk": (N, H, M), "v": (N, H, dh),
    }
    arrays: dict[str, np.ndarray] = {}
    for name, shape in shapes.items():
        n = int(np.prod(shape))
        arrays[name] = w[p:p+n].view("<i4").copy().reshape(shape)
        p += n
    return header, arrays


def _run_frontend(run_dir: Path, cfg_home: Path, aplx_name: str,
                  package: dict[str, Any], n_boards: int) -> dict[str, Any]:
    try:
        from spinn_utilities.overrides import overrides
        from spinn_machine import CoreSubsets
        from spinnman.model.enums import ExecutableType, UserRegister
        from pacman.model.graphs.machine import SimpleMachineVertex
        from pacman.model.resources import ConstantSDRAM
        from spinn_front_end_common.abstract_models import (
            AbstractHasAssociatedBinary, AbstractGeneratesDataSpecification)
        from spinn_front_end_common.data import FecDataView
        from spinn_front_end_common.interface.ds import DataSpecificationGenerator
        from spinn_front_end_common.utilities.helpful_functions import locate_memory_region_for_placement
        import spinnaker_graph_front_end as front_end
    except ImportError as exc:
        raise RuntimeError("Use the same SpiNNakerGraphFrontEnd 7.4.1 environment as H2E-board") from exc

    header = package["header"]
    hir = package["hir"]
    trajectory = package["trajectory"]
    meta = package["meta"]
    output_words = int(meta["region_words"]["output"])
    output_bytes = output_words * 4
    total_sdram = (header.nbytes + hir.nbytes + trajectory.nbytes + output_bytes + 4096)

    class H2FVertex(SimpleMachineVertex, AbstractHasAssociatedBinary,
                    AbstractGeneratesDataSpecification):
        def __init__(self) -> None:
            super().__init__(ConstantSDRAM(total_sdram), label="H2F full resident block")

        @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
        def get_binary_file_name(self) -> str:
            return aplx_name

        @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
        def get_binary_start_type(self):
            return ExecutableType.SYNC

        @overrides(AbstractGeneratesDataSpecification.generate_data_specification)
        def generate_data_specification(self, spec: DataSpecificationGenerator, placement) -> None:
            _ = placement
            regions = ((0, header, "config"), (1, hir, "hir"), (2, trajectory, "trajectory"))
            for rid, array, label in regions:
                spec.reserve_memory_region(region=rid, size=int(array.nbytes), label=label)
                spec.switch_write_focus(rid)
                spec.write_array(array)
            zeros = np.zeros(output_words, dtype="<u4")
            spec.reserve_memory_region(region=3, size=output_bytes, label="output")
            spec.switch_write_focus(3)
            spec.write_array(zeros)
            spec.end_specification()

    old_cwd = Path.cwd(); old_home = os.environ.get("HOME"); old_xdg = os.environ.get("XDG_CONFIG_HOME")
    vertex = H2FVertex(); setup_done = False
    try:
        os.chdir(run_dir)
        os.environ["HOME"] = str(cfg_home)
        os.environ["XDG_CONFIG_HOME"] = str(cfg_home / ".config")
        front_end.setup(n_boards_required=n_boards, model_binary_folder=str(run_dir))
        setup_done = True
        front_end.add_machine_vertex_instance(vertex)
        front_end.run_until_complete()

        placement = FecDataView.get_placement_of_vertex(vertex)
        app_id = int(FecDataView.get_app_id())
        txrx = FecDataView.get_transceiver()
        subset = CoreSubsets(); subset.add_processor(placement.x, placement.y, placement.p)
        iobuf = "\n".join(buf.iobuf for buf in txrx.get_iobuf(subset))
        regs = {
            "user0": int(txrx.read_user(placement.x, placement.y, placement.p, UserRegister.USER_0)),
            "user1": int(txrx.read_user(placement.x, placement.y, placement.p, UserRegister.USER_1)),
            "user2": int(txrx.read_user(placement.x, placement.y, placement.p, UserRegister.USER_2)),
            "user3": int(txrx.read_user(placement.x, placement.y, placement.p, UserRegister.USER_3)),
        }

        out_base = locate_memory_region_for_placement(placement, 3)
        raw = txrx.read_memory(placement.x, placement.y, out_base, output_bytes)
        words = np.frombuffer(bytes(raw), dtype="<u4").copy()
        out_header, board = _decode_output(words, meta)
        oracle = package["oracle"]
        array_checks = {name: bool(np.array_equal(board[name], oracle[name])) for name in oracle}
        max_abs = {name: int(np.max(np.abs(board[name].astype(np.int64) - oracle[name].astype(np.int64))))
                   for name in oracle}
        expected_evals = int(meta["tokens"] + (meta["steps"] - 1) * round(meta["tokens"] * meta["change_frac"]))
        expected_core_app = (int(placement.p) << 16) | (app_id & 0xffff)
        checks = {
            "iobuf_pass_token": PASS_TOKEN in iobuf,
            "iobuf_no_fail_token": FAIL_TOKEN not in iobuf,
            "user0_magic": regs["user0"] == H2F_BOARD_MAGIC,
            "user1_rc_zero": regs["user1"] == 0,
            "user2_steps": regs["user2"] == int(meta["steps"]),
            "user3_core_app": regs["user3"] == expected_core_app,
            "output_magic": out_header["magic"] == H2F_OUTPUT_MAGIC,
            "output_rc_zero": out_header["rc"] == 0,
            "output_steps": out_header["steps"] == int(meta["steps"]),
            "qkv_event_count": out_header["q_evals"] == out_header["k_evals"] == out_header["v_evals"] == expected_evals,
            "all_arrays_bit_exact": all(array_checks.values()),
        }
        np.savez_compressed(run_dir / "H2F_BOARD_FINAL_STATE.npz", **board)
        return {
            "PASS": all(checks.values()), "checks": checks,
            "array_bit_exact": array_checks, "array_max_abs_diff": max_abs,
            "output_header": out_header, "user_registers": regs,
            "placement": {"x": int(placement.x), "y": int(placement.y), "p": int(placement.p)},
            "app_id": app_id, "expected_event_evals_per_projection": expected_evals,
            "iobuf": iobuf,
        }
    finally:
        if setup_done:
            try: front_end.stop()
            except Exception: traceback.print_exc()
        os.chdir(old_cwd)
        if old_home is None: os.environ.pop("HOME", None)
        else: os.environ["HOME"] = old_home
        if old_xdg is None: os.environ.pop("XDG_CONFIG_HOME", None)
        else: os.environ["XDG_CONFIG_HOME"] = old_xdg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplx", type=Path, required=True)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--source-config", type=Path, default=Path.home() / ".spynnaker.cfg")
    ap.add_argument("--n-boards", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    aplx = args.aplx.expanduser().resolve(); package_path = args.package.expanduser().resolve()
    source_cfg = args.source_config.expanduser().resolve()
    if not aplx.is_file(): raise SystemExit(f"missing APLX: {aplx}")
    if not package_path.is_file(): raise SystemExit(f"missing H2F package: {package_path}")
    package = _load_package(package_path)

    run_dir = (args.out if args.out else Path("spinnaker_backend/run/H2F_spalloc_board") / _utc_stamp()).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    staged_aplx = run_dir / "snn_transform_h2f.aplx"; shutil.copy2(aplx, staged_aplx)
    staged_package = run_dir / "H2F_BOARD_PACKAGE.npz"; shutil.copy2(package_path, staged_package)
    machine_cfg = _read_source_machine_config(source_cfg)
    local_cfg, cfg_home, hidden_cfg = _prepare_configs(run_dir, machine_cfg)

    record: dict[str, Any] = {
        "stage": "H2F_FULL_RESIDENT_BOARD",
        "PASS": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "aplx_size_bytes": staged_aplx.stat().st_size,
        "package": package["meta"],
        "machine": _safe_machine_summary(machine_cfg),
        "source_config": str(source_cfg), "generated_gfe_config": str(local_cfg),
        "isolated_gfe_home": str(cfg_home), "isolated_gfe_user_config": str(hidden_cfg),
        "software_versions": {name: _version(name) for name in (
            "sPyNNaker", "SpiNNakerGraphFrontEnd", "SpiNNFrontEndCommon",
            "SpiNNMan", "SpiNNMachine")},
        "claim_boundary": "PASS proves full six-step frozen-HIR resident fixed-point block execution on physical SpiNNaker for this preregistered trajectory.",
    }
    result_path = run_dir / "H2F_BOARD_RESULT.json"
    try:
        runtime = _run_frontend(run_dir, cfg_home, staged_aplx.name, package, args.n_boards)
        iobuf = runtime.pop("iobuf")
        (run_dir / "core_iobuf.txt").write_text(iobuf)
        record.update(runtime); record["iobuf_file"] = str(run_dir / "core_iobuf.txt")
    except Exception as exc:
        record.update({"PASS": False, "exception_type": type(exc).__name__, "exception": str(exc), "traceback": traceback.format_exc()})
        result_path.write_text(json.dumps(record, indent=2)); print(json.dumps(record, indent=2)); raise SystemExit(2) from exc
    result_path.write_text(json.dumps(record, indent=2)); print(json.dumps(record, indent=2))
    if not record["PASS"]: raise SystemExit(2)
    print(f"[H2F] PASS: {result_path}")


if __name__ == "__main__":
    main()
