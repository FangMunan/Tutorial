#!/usr/bin/env python3
"""H1.1 v5 — maximal useful signed-32 operand precision.

v4 reduced the trained-checkpoint prediction error from ~15.8% to ~2.46%, but
the normalized read remained sensitive to a ~0.37% Q-feature error.  The actual
trained ranges permit Q*.24 main operands while remaining signed-32-safe
(q_max~25.75 -> ~4.32e8).  v5 therefore tests the highest practical common
activation precision before 32-bit range, while retaining int64 accumulators.

Contract:
- main activations/weights/random projection: Q*.24
- Q feature cache: Q*.28
- K feature cache/g: Q*.18
- H resident state: Q*.21
- Izh membrane: Q*.24 (v=-65 remains signed-32-safe)
- Izh decoder: Q*.20 (higher decoder scale would threaten int64 accumulator
  headroom for K decoders near magnitude 40)
"""
from __future__ import annotations

import spinnaker_backend.h1_fixed_point_emulator as h1
h1.F_ACT = 24
h1.S_ACT = 1 << h1.F_ACT
h1.F_MEM = 24
h1.F_DEC = 20

import spinnaker_backend.run_h1_1_frozen_checkpoint as base
base.F_ACT = 24
base.F_DEC = 20
base.F_PHI = 18

import spinnaker_backend.run_h1_1_frozen_checkpoint_v2 as v2
v2.F_ACT = 24
v2.F_MEM = 24
v2.F_DEC = 20
v2.F_PQ = 28
v2.F_PK = 18
v2.F_H = 21

import spinnaker_backend.run_h1_1_frozen_checkpoint_v3 as v3

if __name__ == "__main__":
    v2.ActualCheckpointRuntime = v3.IndexCorrectRuntime
    v2.main()
