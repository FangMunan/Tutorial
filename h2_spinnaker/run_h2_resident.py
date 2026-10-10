#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launch one resident H2 full-model validation on SpiNNaker.

Communication contract:
    setup once -> upload bundle once -> run complete trajectory once -> bulk read once.

No stepwise host feedback is allowed in this launcher.  That makes wall-clock timing
interpretable and prevents Ethernet / remote-board latency from masquerading as model
compute time.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import spinnaker_graph_front_end as front_end
from spinn_front_end_common.data import FecDataView

from resident_vertex import H2ResidentVertex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--binary-folder", type=Path, default=Path(__file__).resolve().parent / "c_src")
    ap.add_argument("--out", type=Path, default=Path("H2_SPINNAKER_RUN"))
    ap.add_argument("--output-bytes", type=int, default=262144)
    ap.add_argument("--profile-bytes", type=int, default=4096)
    ap.add_argument("--n-chips", type=int, default=1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    aplx = args.binary_folder / "h2_resident.aplx"
    if not aplx.exists():
        raise FileNotFoundError(
            f"Missing {aplx}. Build c_src after entering the campus SpiNNaker toolchain."
        )

    timing = {}
    t0 = perf_counter()
    front_end.setup(
        n_chips_required=args.n_chips,
        model_binary_folder=str(args.binary_folder.resolve()),
    )
    timing["frontend_setup_s"] = perf_counter() - t0

    vertex = H2ResidentVertex(
        args.bundle,
        output_bytes=args.output_bytes,
        profile_bytes=args.profile_bytes,
    )
    front_end.add_machine_vertex_instance(vertex)

    # This single call includes mapping/data loading plus resident execution.  On-chip
    # cycle counters in the result must be used to separate compute from front-end cost.
    t1 = perf_counter()
    front_end.run_until_complete(1)
    timing["run_call_wall_s"] = perf_counter() - t1

    placements = [p for p in FecDataView.iterate_placemements() if p.vertex is vertex]
    timing["placement"] = None if not placements else {
        "x": placements[0].x, "y": placements[0].y, "p": placements[0].p
    }

    t2 = perf_counter()
    raw = vertex.read_result()
    timing["bulk_read_s"] = perf_counter() - t2
    (args.out / "result.bin").write_bytes(raw)

    t3 = perf_counter()
    front_end.stop()
    timing["frontend_stop_s"] = perf_counter() - t3
    timing["total_host_wall_s"] = perf_counter() - t0
    timing["result_bytes"] = len(raw)

    (args.out / "host_timing.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps(timing, indent=2), flush=True)


if __name__ == "__main__":
    main()
