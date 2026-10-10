#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launch one resident H2 full-model validation on SpiNNaker.

Communication contract:
    setup once -> upload bundle once -> run complete trajectory once -> bulk read once.

No stepwise host feedback is allowed.  Host wall-clock timing is kept separate from
on-chip cycle counters because campus Ethernet / front-end overhead can dominate a fast
resident kernel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import spinnaker_graph_front_end as front_end
from spinn_front_end_common.data import FecDataView

from resident_vertex import H2ResidentVertex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--binary-folder", type=Path,
                    default=Path(__file__).resolve().parent / "c_src")
    ap.add_argument("--out", type=Path, default=Path("H2_SPINNAKER_RUN"))
    ap.add_argument("--mode", choices=("audit", "profile"), default="audit")
    ap.add_argument("--output-bytes", type=int, default=None,
                    help="Override result-buffer bytes; profile defaults small")
    ap.add_argument("--profile-bytes", type=int, default=4096)
    ap.add_argument("--n-chips", type=int, default=1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    aplx = args.binary_folder / "h2_resident.aplx"
    if not aplx.exists():
        raise FileNotFoundError(
            f"Missing {aplx}. Build c_src after entering the campus SpiNNaker toolchain."
        )

    timing = {"mode": args.mode}
    t0 = perf_counter()
    stopped = False
    try:
        ts = perf_counter()
        front_end.setup(
            n_chips_required=args.n_chips,
            model_binary_folder=str(args.binary_folder.resolve()),
        )
        timing["frontend_setup_s"] = perf_counter() - ts

        vertex = H2ResidentVertex(
            args.bundle,
            mode=args.mode,
            output_bytes=args.output_bytes,
            profile_bytes=args.profile_bytes,
        )
        timing["application_payload_bytes_host_to_board"] = vertex.payload_bytes
        timing["requested_result_capacity_bytes"] = vertex.output_bytes + vertex.profile_bytes
        front_end.add_machine_vertex_instance(vertex)

        # This call includes graph mapping/data loading and resident execution.  The C
        # kernel's cycle counters are the authoritative on-chip compute measurement.
        tr = perf_counter()
        front_end.run_until_complete(1)
        timing["run_call_wall_s"] = perf_counter() - tr

        placement = FecDataView.get_placement_of_vertex(vertex)
        timing["placement"] = {
            "x": placement.x, "y": placement.y, "p": placement.p
        }

        tb = perf_counter()
        raw = vertex.read_result()
        timing["bulk_read_s"] = perf_counter() - tb
        timing["result_bytes_received"] = len(raw)
        (args.out / "result.bin").write_bytes(raw)

        te = perf_counter()
        front_end.stop()
        stopped = True
        timing["frontend_stop_s"] = perf_counter() - te
    finally:
        if not stopped:
            try:
                front_end.stop()
            except Exception:
                pass

    timing["total_host_wall_s"] = perf_counter() - t0
    (args.out / "host_timing.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps(timing, indent=2), flush=True)


if __name__ == "__main__":
    main()
