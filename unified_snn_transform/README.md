# Unified SNN-Transform — Lightweight Canonical Route

This branch is the new single mainline for the SNN-Transform project.

## Scientific target

Build a reusable neural-population dynamical block with the interface

```python
y, state = SNNTransformBlock(x, state)
```

that preserves the computational role of a lightweight Linear-Transformer + selective persistent state backbone, while progressively replacing useful computation by neuronal/event-driven dynamics.

The project is not defined by MNIST, MaskGIT, ionic masks, or a specific SpiNNaker implementation. Those are downstream tasks / physical mappings.

## Canonical architecture

Reference block:

```text
x
 -> LayerNorm
 -> Q/K/V projections
 -> positive random-feature Linear Attention
 -> persistent H/g associative state
 -> input-dependent selective retention (Mamba-like)
 -> normalized H/g readout
 -> output projection
 -> residual
 -> LayerNorm
 -> FFN
 -> residual
```

Current candidate keeps the same block but replaces the positive exponential feature map by a calibrated Izhikevich membrane-trajectory decoder and executes K/V updates only for changed token events.

## Inherited evidence; do not repeat from zero

The historical H1/H1-B/H1-B2.1 route already established:

- normalized positive-RF Linear Attention with persistent H/g state;
- selective retention across iterative MaskGIT trajectories;
- event-sparse K/V execution semantics;
- 32-LIF feature-map replacement as an earlier bridge;
- one-Izh trajectory feature map with layer/Q/K-specific calibration;
- H1-B2.1 cloud result: fresh relative-L2 ~6.3e-4 and persistent worst relative-L2 ~8.1e-4 versus the exact feature-map reference.

H7 hardware work is retained as a numerical/fixed-point mapping anchor, not as the scientific architecture definition.

## Lightweight first gate

`run_unified_gate.py` intentionally uses a small, task-independent block:

- N = 16 tokens
- d = 32
- 2 heads
- 32 positive random features
- FFN = 64
- one block
- six persistent steps

It performs three checks in one run:

1. **Operator/state parity** — exact positive-RF block vs Izh-feature candidate over a persistent trajectory.
2. **Event-sparse K/V execution** — only changed tokens recompute K/V after the first step.
3. **Delta-projection audit** — verifies the exact identity `W x_t = W x_(t-1) + W Delta x_t` for Q/K/V and measures delta density.

This is an architecture gate, not a downstream-task benchmark.

## Current boundary

A PASS does **not** mean the final SNN-Transform block is complete. The following remain explicitly open:

- Q/K/V/O channel projections are still dense payload operations;
- H/g readout is still explicit tensor arithmetic in the cloud reference;
- FFN is still dense;
- residual/norm are conventional operations;
- the best Izh feature regime may be subthreshold, so strict spike-only communication is not yet established.

These are tracked rather than hidden.

## Anti-drift rules

1. Keep one canonical `SNNTransformBlock` API.
2. Every change must state which operator it replaces and must report parity + compute cost.
3. Do not optimize image quality before block-level gates pass.
4. Do not use SpiNNaker parity as evidence that an operator is SNN-native.
5. Do not neuralize an operation if it increases total neuromorphic work without adding a required computational function.
6. Scaling is monitored now but optimized only after the complete block is frozen.

## Planned progression

```text
U0  lightweight exact baseline + unified audit
U1  Izh feature dynamics + event-sparse K/V          [current]
U2  actual delta/event Q/K/V/O execution
U3  resident/event H/g read-write realization
U4  event/dynamical FFN + complete reusable block
U5  associative-retrieval + tiny MaskGIT plug-in gates
U6  conventional scaling sweep
U7  final SpiNNaker mapping / hardware efficiency campaign
```

The U-numbering is intentionally short and should replace creation of many unrelated H/B/F branches.
