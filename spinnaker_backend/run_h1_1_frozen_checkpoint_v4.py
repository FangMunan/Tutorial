#!/usr/bin/env python3
"""H1.1 v4 — high-precision 32-bit operand contract for the trained checkpoint.

v3 showed that the resident H/g/K/V states were already accurate (<0.1%) and
int32-safe, but the normalized read amplified the ~0.8% Q-feature error into
~25% block-output error.  The frozen trained checkpoint has ample signed-32-bit
headroom in activations/weights, so v4 uses that headroom to reduce upstream
quantization without changing any scientific operator.

Chosen contract (all resident operands remain signed 32-bit):
- main activation/weights: Q*.20
- Q feature cache: Q*.24
- K feature cache/g: Q*.16
- H resident state: Q*.20
- Izh membrane: Q*.22
- Izh decoder: Q*.20
- products/accumulators: signed 64-bit
"""
from __future__ import annotations

# Patch the generic H1 numerical library before importing the compiler/runtime
# modules.  Its functions read these module globals at execution time.
import spinnaker_backend.h1_fixed_point_emulator as h1
h1.F_ACT = 20
h1.S_ACT = 1 << h1.F_ACT
h1.F_MEM = 22
h1.F_DEC = 20

import spinnaker_backend.run_h1_1_frozen_checkpoint as base
base.F_ACT = 20
base.F_DEC = 20
base.F_PHI = 16  # metadata/legacy only; v4 uses asymmetric Q/K feature scales

import spinnaker_backend.run_h1_1_frozen_checkpoint_v2 as v2
v2.F_ACT = 20
v2.F_MEM = 22
v2.F_DEC = 20
v2.F_PQ = 24
v2.F_PK = 16
v2.F_H = 20

# Import only after the constants above are installed so the local F_ACT used by
# IndexCorrectRuntime is captured as 20.
import spinnaker_backend.run_h1_1_frozen_checkpoint_v3 as v3


if __name__ == "__main__":
    v2.ActualCheckpointRuntime = v3.IndexCorrectRuntime
    v2.main()
