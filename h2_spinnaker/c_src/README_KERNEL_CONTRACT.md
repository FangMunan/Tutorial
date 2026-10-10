# `h2_resident.aplx` kernel contract

This directory intentionally freezes the **interface and mathematics** before freezing a
C implementation.  The first C source should be written inside the actual campus WSL
SpiNNaker toolchain so API/compiler assumptions can be tested immediately.

## What the first kernel must do

The first implementation is a serial resident correctness baseline on one application
core.  It reads all regions once, executes the complete teacher-forced trajectory, records
compact results/profiling, and exits.  It must not request host computation between
MaskGIT rounds.

For every teacher-forced round:

1. construct token + position + class embeddings;
2. run all four blocks;
3. in block 1 use the validated H1 sparse persistent H/g rule and learned selective alpha;
4. in blocks 2--4 use fresh H/g linear attention;
5. evaluate Q/K dynamic features with the B2.1 layer/side-specific Izh calibration;
6. apply residual, LayerNorm, FFN/GELU, second residual/LayerNorm exactly according to the frozen reference;
7. produce output logits;
8. in audit mode record compact numerical checkpoints; in profile mode record only counters/final summaries.

The initial C baseline may be slow.  Its job is to establish fixed-point numerical
correctness and a resident communication baseline.  Only after this passes should the
computation be partitioned across the application cores of one chip.

## Numeric mapping

Do not use ARM software floating point as the final performance implementation.  The
FP32 bundle is a reference image.  First measure tensor/state ranges and choose a fixed-
point format (or mixed formats).  Then run a software fixed-point preflight against the
same reference trajectory.  Some K-side Izh trajectory decoders contain alternating
coefficients large enough that fixed-point cancellation must be checked explicitly.

If the B2.1 decoder loses fidelity under fixed point, recalibrate the decoder under the
same fixed-point neuron/trajectory arithmetic rather than changing the high-level model.

## DMA and DTCM

Weights remain in chip SDRAM.  Matrix tiles are DMA'd into DTCM.  Do not attempt to copy
the full model or full multi-head H state into one core's local memory.  Instrument DMA
bytes and cycles from the first implementation.

## Later one-chip parallelization

Once the serial reference passes, the likely mapping is to shard attention by head and
FFN/channel blocks across cores while reusing the same worker cores across network layers.
Keep the entire sample on one chip before studying cross-chip model partitioning.  This
provides a clean separation between compute/DMA cost and router/inter-chip cost.

## Runtime counters

At minimum measure cycles for embedding, QKV, dynamic feature evaluation, H/g update/read,
FFN, normalization/output, plus DMA bytes.  Do not infer a bottleneck from Python timing.
The host runner's `run_call_wall_s` includes front-end setup/data movement and therefore
must be reported separately from on-chip cycles.

## Production communication rule

No live membrane-voltage stream.  No per-layer H/g readback.  No generation-step host
callback.  A normal profile run is one upload, one resident execution, one bulk read.
