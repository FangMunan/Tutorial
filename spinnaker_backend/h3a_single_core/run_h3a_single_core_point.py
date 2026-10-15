#!/usr/bin/env python3
"""Run one H3A token-count point on one physical SpiNNaker core.

Successful allocation must be bit-exact to the package oracle.  rc=20 is not a
scientific failure: it is the expected single-core DTCM capacity boundary and
is recorded as CAPACITY_FAIL with valid heap diagnostics.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from spinnaker_backend.h2f_aplx import run_h2f_spalloc_board as h2fbase

H3A_OUTPUT_MAGIC = 0x4833414F
PASS_TOKEN = "SNN_TRANSFORM_H3A_SINGLE_CORE PASS"
CAPACITY_TOKEN = "SNN_TRANSFORM_H3A_SINGLE_CORE CAPACITY"
FAIL_TOKEN = "SNN_TRANSFORM_H3A_SINGLE_CORE FAIL"
OUT_HEADER_WORDS = 16


def _load_package(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z["package_metadata_json_utf8"].tolist()).decode("utf-8"))
    if meta.get("format") != "SNNTransform-H3A-single-core-v1":
        raise ValueError(f"unexpected package format {meta.get('format')!r}")
    return {
        "meta": meta,
        "header": np.ascontiguousarray(z["header_u32"], dtype="<u4"),
        "hir": np.ascontiguousarray(z["hir_blob_u32"], dtype="<u4"),
        "trajectory": np.ascontiguousarray(z["trajectory_u32"], dtype="<u4"),
        "oracle": {name: np.ascontiguousarray(z[f"oracle_{name}"], dtype="<i4")
                   for name in ("y", "H", "g", "pq", "pk", "v")},
    }


def _decode_header(words: np.ndarray) -> dict[str, int]:
    w = np.asarray(words, dtype="<u4")
    names = (
        "magic", "rc", "steps", "q_evals", "k_evals", "v_evals",
        "chip", "core_app", "tokens", "arena_bytes", "heap_free_before",
        "heap_max_before", "heap_free_after", "heap_max_after", "elapsed_ms",
        "reserved",
    )
    return {name: int(w[i]) for i, name in enumerate(names)}


def _decode_arrays(words: np.ndarray, meta: dict) -> dict[str, np.ndarray]:
    w = np.asarray(words, dtype="<u4")
    p = OUT_HEADER_WORDS
    N, D, H, M = (int(meta[k]) for k in ("tokens", "d", "heads", "features"))
    dh = D // H
    shapes = {
        "y": (N, D), "H": (H, M, dh), "g": (H, M),
        "pq": (N, H, M), "pk": (N, H, M), "v": (N, H, dh),
    }
    out = {}
    for name, shape in shapes.items():
        n = int(np.prod(shape))
        out[name] = w[p:p+n].view("<i4").copy().reshape(shape)
        p += n
    return out


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
        raise RuntimeError("Use the same SpiNNakerGraphFrontEnd 7.4.1 environment as H2F") from exc

    header, hir, trajectory, meta = (
        package["header"], package["hir"], package["trajectory"], package["meta"])
    output_words = int(meta["region_words"]["output"])
    output_bytes = output_words * 4
    total_sdram = header.nbytes + hir.nbytes + trajectory.nbytes + output_bytes + 4096

    class H3AVertex(SimpleMachineVertex, AbstractHasAssociatedBinary,
                    AbstractGeneratesDataSpecification):
        def __init__(self) -> None:
            super().__init__(ConstantSDRAM(total_sdram), label=f"H3A N={meta['tokens']}")

        @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
        def get_binary_file_name(self) -> str:
            return aplx_name

        @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
        def get_binary_start_type(self):
            return ExecutableType.SYNC

        @overrides(AbstractGeneratesDataSpecification.generate_data_specification)
        def generate_data_specification(self, spec: DataSpecificationGenerator, placement) -> None:
            _ = placement
            for rid, array, label in ((0, header, "config"), (1, hir, "hir"), (2, trajectory, "trajectory")):
                spec.reserve_memory_region(region=rid, size=int(array.nbytes), label=label)
                spec.switch_write_focus(rid)
                spec.write_array(array)
            spec.reserve_memory_region(region=3, size=output_bytes, label="output")
            spec.switch_write_focus(3)
            spec.write_array(np.zeros(output_words, dtype="<u4"))
            spec.end_specification()

    old_cwd = Path.cwd(); old_home = os.environ.get("HOME"); old_xdg = os.environ.get("XDG_CONFIG_HOME")
    vertex = H3AVertex(); setup_done = False
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
        out_header = _decode_header(words)
        expected_core_app = (int(placement.p) << 16) | (app_id & 0xffff)
        common = {
            "user0_region_table_pointer_nonzero": regs["user0"] != 0,
            "user3_core_app": regs["user3"] == expected_core_app,
            "output_magic": out_header["magic"] == H3A_OUTPUT_MAGIC,
            "output_tokens": out_header["tokens"] == int(meta["tokens"]),
            "arena_bytes_match_package": out_header["arena_bytes"] == int(meta["arena_bytes_requested"]),
        }

        if out_header["rc"] == 20:
            checks = {
                **common,
                "iobuf_capacity_token": CAPACITY_TOKEN in iobuf,
                "user1_rc20": regs["user1"] == 20,
                "output_rc20": out_header["rc"] == 20,
                "heap_max_below_request": out_header["heap_max_before"] < out_header["arena_bytes"],
            }
            return {
                "MEASUREMENT_VALID": all(checks.values()), "EXEC_PASS": False,
                "status": "CAPACITY_FAIL", "checks": checks,
                "output_header": out_header, "user_registers": regs,
                "placement": {"x": int(placement.x), "y": int(placement.y), "p": int(placement.p)},
                "app_id": app_id, "iobuf": iobuf,
            }

        board = _decode_arrays(words, meta)
        oracle = package["oracle"]
        array_checks = {name: bool(np.array_equal(board[name], oracle[name])) for name in oracle}
        max_abs = {name: int(np.max(np.abs(board[name].astype(np.int64) - oracle[name].astype(np.int64))))
                   for name in oracle}
        expected_evals = sum(int(row["changed_tokens"]) for row in meta["trajectory"])
        checks = {
            **common,
            "iobuf_pass_token": PASS_TOKEN in iobuf,
            "iobuf_no_fail_token": FAIL_TOKEN not in iobuf,
            "user1_rc_zero": regs["user1"] == 0,
            "user2_steps": regs["user2"] == int(meta["steps"]),
            "output_rc_zero": out_header["rc"] == 0,
            "output_steps": out_header["steps"] == int(meta["steps"]),
            "qkv_event_count": out_header["q_evals"] == out_header["k_evals"] == out_header["v_evals"] == expected_evals,
            "all_arrays_bit_exact": all(array_checks.values()),
            "heap_allocation_consistent": out_header["heap_max_before"] >= out_header["arena_bytes"],
        }
        np.savez_compressed(run_dir / "H3A_BOARD_FINAL_STATE.npz", **board)
        return {
            "MEASUREMENT_VALID": all(checks.values()), "EXEC_PASS": all(checks.values()),
            "status": "EXEC_PASS" if all(checks.values()) else "EXEC_FAIL",
            "checks": checks, "array_bit_exact": array_checks, "array_max_abs_diff": max_abs,
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

    aplx = args.aplx.expanduser().resolve(); pkg_path = args.package.expanduser().resolve()
    source_cfg = args.source_config.expanduser().resolve()
    if not aplx.is_file(): raise SystemExit(f"missing APLX: {aplx}")
    if not pkg_path.is_file(): raise SystemExit(f"missing package: {pkg_path}")
    package = _load_package(pkg_path)

    run_dir = (args.out if args.out else Path("spinnaker_backend/run/H3A_single_core") /
               f"N{package['meta']['tokens']}_{h2fbase._utc_stamp()}").expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    staged_aplx = run_dir / "snn_transform_h3a.aplx"; shutil.copy2(aplx, staged_aplx)
    staged_pkg = run_dir / pkg_path.name; shutil.copy2(pkg_path, staged_pkg)
    machine_cfg = h2fbase._read_source_machine_config(source_cfg)
    local_cfg, cfg_home, hidden_cfg = h2fbase._prepare_configs(run_dir, machine_cfg)

    record: dict[str, Any] = {
        "stage": "H3A_SINGLE_CORE_POINT",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tokens": int(package["meta"]["tokens"]),
        "package": package["meta"],
        "aplx_size_bytes": staged_aplx.stat().st_size,
        "machine": h2fbase._safe_machine_summary(machine_cfg),
        "source_config": str(source_cfg), "generated_gfe_config": str(local_cfg),
        "isolated_gfe_home": str(cfg_home), "isolated_gfe_user_config": str(hidden_cfg),
        "claim_boundary": "H3A measures the current frozen one-core mapping only; it does not claim chip- or board-level scaling.",
    }
    result_path = run_dir / "H3A_POINT_RESULT.json"
    try:
        runtime = _run_frontend(run_dir, cfg_home, staged_aplx.name, package, args.n_boards)
        iobuf = runtime.pop("iobuf")
        (run_dir / "core_iobuf.txt").write_text(iobuf)
        record.update(runtime); record["iobuf_file"] = str(run_dir / "core_iobuf.txt")
    except Exception as exc:
        record.update({"MEASUREMENT_VALID": False, "EXEC_PASS": False, "status": "RUNNER_EXCEPTION",
                       "exception_type": type(exc).__name__, "exception": str(exc),
                       "traceback": traceback.format_exc()})
        result_path.write_text(json.dumps(record, indent=2)); print(json.dumps(record, indent=2)); raise SystemExit(2) from exc

    result_path.write_text(json.dumps(record, indent=2)); print(json.dumps(record, indent=2))
    if not record.get("MEASUREMENT_VALID", False):
        raise SystemExit(2)
    print(f"[H3A] {record['status']}: {result_path}")


if __name__ == "__main__":
    main()
