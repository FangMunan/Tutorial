# Project status

Current canonical branch: `snn-transform-unified-20261014`

Current gate: **U0/U1 lightweight unified block**.

This commit intentionally triggers the first independent GitHub Actions run after the workflow was installed.

Next action is determined only from `UNIFIED_RESULTS.json`:

- if operator/state parity fails: localize the first divergent operator;
- if parity passes: freeze U1 and move directly to U2 actual delta/event Q/K/V/O execution inside the same block/API;
- do not start MaskGIT or SpiNNaker work before the complete reusable block reaches its defined gates.
