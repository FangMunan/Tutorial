#!/usr/bin/env python3
"""Run one H3B distributed-block point on a single physical SpiNNaker chip."""
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
from spinnaker_backend.h3b_single_chip.h3b_contract import (
    H3B_MAGIC, H3B_OUTPUT_MAGIC, H3B_VERSION, H3B_HEADER_WORDS,
    H3B_OUTPUT_HEADER_WORDS, H3B_N_KEYS, H3B_PARTITION_ID,
    local_trajectory_words, output_word_count, worker_header,
)

PASS_TOKEN = "SNN_TRANSFORM_H3B_WORKER PASS"
OUT_NAMES = (
    "magic", "rc", "steps", "rank", "n_workers", "global_tokens",
    "local_tokens", "q_evals", "k_evals", "v_evals", "tx_packets",
    "rx_packets", "tx_retries", "protocol_errors", "elapsed_ms", "chip",
    "core_app", "ready_events", "gather_phases", "bcast_phases",
    "arena_bytes", "reserved0", "reserved1", "reserved2",
)


def _load_package(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z["package_metadata_json_utf8"].tolist()).decode("utf-8"))
    if meta.get("format") != "SNNTransform-H3B-single-chip-v1":
        raise ValueError(f"unexpected package format {meta.get('format')!r}")
    return {
        "meta": meta,
        "base_header": np.ascontiguousarray(z["base_header_u32"], dtype="<u4"),
        "hir": np.ascontiguousarray(z["hir_blob_u32"], dtype="<u4"),
        "x": np.ascontiguousarray(z["trajectory_x_i32"], dtype="<i4"),
        "masks": np.ascontiguousarray(z["trajectory_masks_u32"], dtype="<u4"),
        "oracle": {name: np.ascontiguousarray(z[f"oracle_{name}"], dtype="<i4")
                   for name in ("y", "H", "g", "pq", "pk", "v")},
    }


def _decode_output(words: np.ndarray, *, local_tokens: int, d: int,
                   heads: int, features: int) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    w = np.asarray(words, dtype="<u4")
    header = {name: int(w[i]) for i, name in enumerate(OUT_NAMES)}
    p = H3B_OUTPUT_HEADER_WORDS
    dh = d // heads
    shapes = {
        "y": (local_tokens, d),
        "H": (heads, features, dh),
        "g": (heads, features),
        "pq": (local_tokens, heads, features),
        "pk": (local_tokens, heads, features),
        "v": (local_tokens, heads, dh),
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
        from pacman.model.graphs.machine import MachineEdge, SimpleMachineVertex
        from pacman.model.resources import ConstantSDRAM
        from spinn_front_end_common.abstract_models import (
            AbstractHasAssociatedBinary, AbstractGeneratesDataSpecification)
        from spinn_front_end_common.data import FecDataView
        from spinn_front_end_common.interface.ds import DataSpecificationGenerator
        from spinn_front_end_common.utilities.helpful_functions import locate_memory_region_for_placement
        import spinnaker_graph_front_end as front_end
    except ImportError as exc:
        raise RuntimeError("Use the same SpiNNakerGraphFrontEnd 7.4.1 environment as H2F/H3A") from exc

    meta = package["meta"]
    workers = int(meta["workers"])
    local_tokens = int(meta["local_tokens"])
    d = int(meta["d"]); heads = int(meta["heads"]); features = int(meta["features"])
    steps = int(meta["steps"])
    hir = package["hir"]
    base_header = package["base_header"]
    xs = package["x"]
    masks = package["masks"]
    out_words = int(meta["output_words_per_worker"])
    out_bytes = out_words * 4

    vertices: list[Any] = []

    class H3BVertex(SimpleMachineVertex, AbstractHasAssociatedBinary,
                    AbstractGeneratesDataSpecification):
        def __init__(self, rank: int) -> None:
            self.rank = rank
            traj_words = local_trajectory_words(local_tokens, d, steps)
            total_sdram = H3B_HEADER_WORDS * 4 + hir.nbytes + traj_words * 4 + out_bytes + 4096
            super().__init__(ConstantSDRAM(total_sdram), label=f"H3B worker {rank}")
            # Force all workers onto one physical chip; PACMAN chooses the cores.
            self.set_fixed_location(0, 0)

        @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
        def get_binary_file_name(self) -> str:
            return aplx_name

        @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
        def get_binary_start_type(self):
            return ExecutableType.SYNC

        @overrides(SimpleMachineVertex.get_n_keys_for_partition)
        def get_n_keys_for_partition(self, partition_id: str) -> int:
            if partition_id == H3B_PARTITION_ID:
                return H3B_N_KEYS
            return super().get_n_keys_for_partition(partition_id)

        @overrides(AbstractGeneratesDataSpecification.generate_data_specification)
        def generate_data_specification(self, spec: DataSpecificationGenerator, placement) -> None:
            _ = placement
            routing = FecDataView.get_routing_infos()
            peer_keys = [int(routing.get_key_from(v, H3B_PARTITION_ID)) for v in vertices]
            own_key = peer_keys[self.rank]
            header = worker_header(
                base_header, rank=self.rank, send_key=own_key, peer_keys=peer_keys)
            a = self.rank * local_tokens
            b = a + local_tokens
            x_local = np.ascontiguousarray(xs[:, a:b, :], dtype="<i4")
            m_local = np.ascontiguousarray(masks[:, a:b], dtype="<u4")
            trajectory = np.concatenate([
                x_local.view("<u4").reshape(-1), m_local.reshape(-1)
            ]).astype("<u4", copy=False)
            expected = local_trajectory_words(local_tokens, d, steps)
            if trajectory.size != expected:
                raise AssertionError((trajectory.size, expected))

            for rid, arr, label in (
                (0, header, "config"), (1, hir, "hir"), (2, trajectory, "trajectory")):
                spec.reserve_memory_region(region=rid, size=int(arr.nbytes), label=label)
                spec.switch_write_focus(rid)
                spec.write_array(arr)
            spec.reserve_memory_region(region=3, size=out_bytes, label="output")
            spec.switch_write_focus(3)
            spec.write_array(np.zeros(out_words, dtype="<u4"))
            spec.end_specification()

    for rank in range(workers):
        vertices.append(H3BVertex(rank))

    old_cwd = Path.cwd(); old_home = os.environ.get("HOME"); old_xdg = os.environ.get("XDG_CONFIG_HOME")
    setup_done = False
    try:
        os.chdir(run_dir)
        os.environ["HOME"] = str(cfg_home)
        os.environ["XDG_CONFIG_HOME"] = str(cfg_home / ".config")
        front_end.setup(n_boards_required=n_boards, model_binary_folder=str(run_dir))
        setup_done = True
        for v in vertices:
            front_end.add_machine_vertex_instance(v)
        # Star routing: every nonzero worker sends only to rank 0; rank 0
        # broadcasts to every nonzero worker.  All use one 4096-key partition.
        for rank in range(1, workers):
            front_end.add_machine_edge_instance(
                MachineEdge(vertices[rank], vertices[0]), H3B_PARTITION_ID)
            front_end.add_machine_edge_instance(
                MachineEdge(vertices[0], vertices[rank]), H3B_PARTITION_ID)
        front_end.run_until_complete()

        txrx = FecDataView.get_transceiver()
        app_id = int(FecDataView.get_app_id())
        placements = [FecDataView.get_placement_of_vertex(v) for v in vertices]
        same_chip = len({(int(p.x), int(p.y)) for p in placements}) == 1
        placement_rows = [
            {"rank": r, "x": int(p.x), "y": int(p.y), "p": int(p.p)}
            for r, p in enumerate(placements)
        ]

        worker_records = []
        board_arrays = []
        for rank, placement in enumerate(placements):
            subset = CoreSubsets(); subset.add_processor(placement.x, placement.y, placement.p)
            iobuf = "\n".join(buf.iobuf for buf in txrx.get_iobuf(subset))
            regs = {
                "user0": int(txrx.read_user(placement.x, placement.y, placement.p, UserRegister.USER_0)),
                "user1": int(txrx.read_user(placement.x, placement.y, placement.p, UserRegister.USER_1)),
                "user2": int(txrx.read_user(placement.x, placement.y, placement.p, UserRegister.USER_2)),
                "user3": int(txrx.read_user(placement.x, placement.y, placement.p, UserRegister.USER_3)),
            }
            out_base = locate_memory_region_for_placement(placement, 3)
            raw = txrx.read_memory(placement.x, placement.y, out_base, out_bytes)
            words = np.frombuffer(bytes(raw), dtype="<u4").copy()
            header, arrays = _decode_output(
                words, local_tokens=local_tokens, d=d, heads=heads, features=features)
            expected_core_app = (int(placement.p) << 16) | (app_id & 0xffff)
            checks = {
                "pass_token": PASS_TOKEN in iobuf,
                "rc_zero": header["rc"] == 0 and regs["user1"] == 0,
                "steps": header["steps"] == steps and regs["user2"] == steps,
                "rank": header["rank"] == rank,
                "n_workers": header["n_workers"] == workers,
                "global_tokens": header["global_tokens"] == int(meta["global_tokens"]),
                "local_tokens": header["local_tokens"] == local_tokens,
                "protocol_errors_zero": header["protocol_errors"] == 0,
                "core_app": regs["user3"] == expected_core_app,
                "output_magic": header["magic"] == H3B_OUTPUT_MAGIC,
                "qkv_evals": (
                    header["q_evals"] == header["k_evals"] == header["v_evals"] ==
                    int(meta["expected_qkv_evals_per_worker"][rank])),
            }
            (run_dir / f"worker_{rank}_iobuf.txt").write_text(iobuf)
            worker_records.append({
                "rank": rank, "placement": placement_rows[rank], "header": header,
                "registers": regs, "checks": checks,
                "PASS": all(checks.values()),
                "iobuf_file": str(run_dir / f"worker_{rank}_iobuf.txt"),
            })
            board_arrays.append(arrays)

        oracle = package["oracle"]
        combined = {
            "y": np.concatenate([a["y"] for a in board_arrays], axis=0),
            "pq": np.concatenate([a["pq"] for a in board_arrays], axis=0),
            "pk": np.concatenate([a["pk"] for a in board_arrays], axis=0),
            "v": np.concatenate([a["v"] for a in board_arrays], axis=0),
        }
        array_checks = {
            name: bool(np.array_equal(combined[name], oracle[name]))
            for name in ("y", "pq", "pk", "v")
        }
        replica_checks = {
            f"H_rank{r}": bool(np.array_equal(a["H"], oracle["H"]))
            for r, a in enumerate(board_arrays)
        }
        replica_checks.update({
            f"g_rank{r}": bool(np.array_equal(a["g"], oracle["g"]))
            for r, a in enumerate(board_arrays)
        })
        max_abs = {
            name: int(np.max(np.abs(combined[name].astype(np.int64) - oracle[name].astype(np.int64))))
            for name in ("y", "pq", "pk", "v")
        }
        for r, arr in enumerate(board_arrays):
            max_abs[f"H_rank{r}"] = int(np.max(np.abs(arr["H"].astype(np.int64) - oracle["H"].astype(np.int64))))
            max_abs[f"g_rank{r}"] = int(np.max(np.abs(arr["g"].astype(np.int64) - oracle["g"].astype(np.int64))))

        all_exact = all(array_checks.values()) and all(replica_checks.values())
        board_pass = same_chip and all(w["PASS"] for w in worker_records) and all_exact
        np.savez_compressed(
            run_dir / "H3B_BOARD_FINAL_STATE.npz",
            **combined, H=board_arrays[0]["H"], g=board_arrays[0]["g"])
        return {
            "PASS": board_pass,
            "same_chip": same_chip,
            "placements": placement_rows,
            "workers": worker_records,
            "array_bit_exact": array_checks,
            "replica_bit_exact": replica_checks,
            "array_max_abs_diff": max_abs,
            "all_arrays_bit_exact": all_exact,
            "total_q_evals": sum(w["header"]["q_evals"] for w in worker_records),
            "total_k_evals": sum(w["header"]["k_evals"] for w in worker_records),
            "total_v_evals": sum(w["header"]["v_evals"] for w in worker_records),
            "total_tx_packets": sum(w["header"]["tx_packets"] for w in worker_records),
            "total_rx_packets": sum(w["header"]["rx_packets"] for w in worker_records),
            "total_tx_retries": sum(w["header"]["tx_retries"] for w in worker_records),
            "max_worker_elapsed_ms": max(w["header"]["elapsed_ms"] for w in worker_records),
            "app_id": app_id,
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

    aplx = args.aplx.expanduser().resolve()
    pkg_path = args.package.expanduser().resolve()
    source_cfg = args.source_config.expanduser().resolve()
    if not aplx.is_file(): raise SystemExit(f"missing APLX: {aplx}")
    if not pkg_path.is_file(): raise SystemExit(f"missing package: {pkg_path}")
    package = _load_package(pkg_path)

    stamp = h2fbase._utc_stamp()
    run_dir = (args.out if args.out else Path("spinnaker_backend/run/H3B_single_chip") / stamp).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    staged_aplx = run_dir / "snn_transform_h3b_worker.aplx"; shutil.copy2(aplx, staged_aplx)
    staged_pkg = run_dir / pkg_path.name; shutil.copy2(pkg_path, staged_pkg)
    machine_cfg = h2fbase._read_source_machine_config(source_cfg)
    local_cfg, cfg_home, hidden_cfg = h2fbase._prepare_configs(run_dir, machine_cfg)

    record: dict[str, Any] = {
        "stage": "H3B0_SINGLE_CHIP_DISTRIBUTED_CORRECTNESS",
        "PASS": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "package": package["meta"],
        "aplx_size_bytes": staged_aplx.stat().st_size,
        "machine": h2fbase._safe_machine_summary(machine_cfg),
        "source_config": str(source_cfg),
        "generated_gfe_config": str(local_cfg),
        "isolated_gfe_home": str(cfg_home),
        "isolated_gfe_user_config": str(hidden_cfg),
        "claim_boundary": "H3B0 proves only same-chip distributed correctness/communication for one global frozen block; no scaling-efficiency claim until H3B1 sweep.",
    }
    result_path = run_dir / "H3B_BOARD_RESULT.json"
    try:
        runtime = _run_frontend(run_dir, cfg_home, staged_aplx.name, package, args.n_boards)
        record.update(runtime)
    except Exception as exc:
        record.update({
            "PASS": False, "status": "RUNNER_EXCEPTION",
            "exception_type": type(exc).__name__, "exception": str(exc),
            "traceback": traceback.format_exc(),
        })
        result_path.write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2))
        raise SystemExit(2) from exc

    record["status"] = "PASS" if record.get("PASS") else "FAIL"
    result_path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    if not record.get("PASS"):
        raise SystemExit(2)
    print(f"[H3B0] PASS: {result_path}")


if __name__ == "__main__":
    main()
