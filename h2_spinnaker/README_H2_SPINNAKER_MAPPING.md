# H2 — Full Resident SpiNNaker Mapping

## Purpose

This directory is the hand-off point from the validated cloud model to the first
full-architecture SpiNNaker experiment.  The objective is **not** to replay only the
Izhikevich feature map on hardware.  The objective is to keep the core generation
trajectory resident on SpiNNaker long enough to measure the architecture as a system.

The frozen software reference currently contains:

- a 49-token MNIST MaskGIT generator;
- 4 linear-attention blocks, width 128, 4 heads, FFN width 512;
- positive random-feature H/g attention;
- block-1 persistent H/g state with learned selective retention;
- monotonic reveal and event-sparse block-1 execution;
- Q/K- and layer-specific 6-ms Izhikevich dynamical feature calibration.

The H2 hardware experiment must preserve these semantics first.  Hardware-oriented
changes are allowed only when they pass an explicit equivalence audit.

---

## Main hardware principle: resident execution

Host-to-board communication is treated as setup/teardown, not as part of every model
step.  A normal H2 run should therefore be:

1. allocate machine once;
2. upload model image, calibration, inputs and random/trajectory control once;
3. execute all requested generation/teacher-forced steps on the board;
4. write only compact results/profiling counters to SDRAM;
5. bulk-read results once at the end.

Do **not** implement the model as a loop of `run -> get_data -> host compute -> inject -> run`.
Do not stream membrane voltages or H/g to the host in production profiling mode.
A separate debug mode may record selected internal states for one sample.

This choice is important because external live streaming does not scale to the aggregate
internal state bandwidth of SpiNNaker.

---

## First mapping target

The first target is a **teacher-forced resident trajectory**, followed by autonomous
MaskGIT generation.

Teacher-forced validation is deliberately first because it isolates the numerical
mapping from on-chip categorical sampling.  The host uploads the full monotonic canvas
trajectory at the start; the board computes every layer, every attention read/write,
selective alpha, and output logits without host intervention.  A small set of logits
and state checksums is read back once.

After this passes, the same resident kernel can add the MaskGIT sampling controller.

This is full-model inference validation.  It is not yet a claim of on-chip learning.
On-chip training is a separate stage because it requires an explicit local learning
rule and weight-update representation.

---

## Mathematical contract

For a head, normalized linear attention is

    H = sum_i phi(k_i) v_i^T
    g = sum_i phi(k_i)
    y_j = phi(q_j)^T H / (phi(q_j)^T g + eps)

For the persistent first block,

    Hp_t = alpha_t Hp_{t-1} + newly_committed_writes_t
    gp_t = alpha_t gp_{t-1} + newly_committed_key_features_t

and current masked tokens contribute a non-decayed transient Hm/gm.

`alpha_t` is applied synchronously to both Hp and gp.  Never decay H without g.

The generation trajectory is monotonic.  A committed token is not re-masked.  If that
contract changes later, the resident cache must be replaced by explicit replacement or
delta semantics.

---

## Hardware optimization already identified

### 1. Cache all block-1 Q/K/V/phi states across the monotonic reveal trajectory

The original H1 sparse code avoids K/V evaluation for already committed tokens, but it
still recomputes Q for all 49 tokens each round and K/V for all still-masked tokens.
That is unnecessary in block 1 because its input is only

    token embedding + position embedding + class embedding.

A masked position is unchanged until reveal; after reveal its committed embedding is
also unchanged.  Therefore a position changes at most once:

    MASK state -> committed token state.

H2 maintains a resident masked contribution Hm/gm and cached phi(Q).  When a token is
revealed, its old masked contribution is subtracted once, the real token Q/K/V/phi is
evaluated once, and the committed write is added to Hp/gp.  This removes repeated
feature dynamics and projection work without changing the model equation.

`h2_block1_cache_audit.py` is the required FP32 equivalence gate for this optimization.
Do not port the cache to hardware if that audit fails.

### 2. Keep H/g sharded by attention head

For the current dimensions, one full persistent H tensor contains

    4 * 128 * 32 = 16384 real values.

A SpiNNaker-1 core has only 64 KiB DTCM, so the full tensor plus code/scratch is not a
sensible one-core resident object.  Per-head H contains only 4096 values and is a natural
shard.  This also matches the four attention heads and avoids unnecessary head-to-head
communication.

### 3. Store large weights in chip SDRAM and DMA tiles into DTCM

The whole trained network is small relative to a chip's SDRAM but too large for local
DTCM.  Weight matrices should be uploaded once and then tiled through DMA.  Repeated
host uploads during a run are forbidden in the production mapping.

### 4. Prefer one-chip-resident sample pipelines before cross-chip model partitioning

The present model is small enough that its weights and state comfortably fit in one
chip's SDRAM.  A good first H2 implementation therefore uses the application cores of
one chip cooperatively for one sample/trajectory, reusing the same cores across layers.
This minimizes inter-chip traffic and gives a clean single-chip compute baseline.
After that, replicate samples across chips and only then study model partitioning when
the model itself grows beyond one-chip capacity.

---

## Fixed-point requirement

The cloud reference is FP32 PyTorch.  SpiNNaker-1 ARM968 cores do not have hardware
floating point.  Therefore a full resident mapping should not silently rely on software
float and then report the resulting runtime as representative.

The first deployment should use SpiNNaker fixed-point arithmetic (or an explicitly
profiled equivalent) and must run a **fixed-point preflight audit** before hardware claims
are made.  The audit must compare:

- fresh logits;
- persistent trajectory logits;
- alpha values;
- H/g checksums or selected entries;
- final token decisions.

Quantization is a hardware mapping approximation and must be separated from the already
validated Izh feature-map approximation.

---

## Communication modes

### `profile`

Production timing mode.  Read back only:

- final tokens / selected logits;
- per-stage cycle counters;
- DMA bytes;
- multicast packet counters;
- router/provenance counters;
- memory high-water marks;
- error/status flags.

No per-step membrane/state recording.

### `audit`

One or a few samples only.  Additionally record selected states at predefined checkpoints
for comparison with the software reference.  Never use audit-mode communication numbers
as production performance measurements.

---

## Files

- `h2_block1_cache_audit.py` — exact cloud audit for the first runtime optimization.
- `export_h2_resident_bundle.py` — creates a self-contained hardware input bundle from
  the frozen H1-B checkpoint and B2.1 calibration.
- `resident_vertex.py` — GraphFrontEnd vertex/data-spec interface used to upload one
  resident bundle and bulk-read one result block.
- `run_h2_resident.py` — host launcher.  It allocates once, runs once, reads once.
- `c_src/` — on-chip resident kernel to be compiled to `h2_resident.aplx` after the WSL
  SpiNNaker development environment is available.

The Python-side code intentionally isolates site/machine configuration from the model
math.  Codex should adapt only the allocation/build details after inspecting the actual
campus SpiNNaker environment; it should not redesign the mathematical contract merely to
make an API example easier.

---

## Required sequence once WSL has campus access

1. Confirm the installed sPyNNaker / SpiNNakerGraphFrontEnd / spinnaker_tools versions.
2. Run the existing minimal SpiNNaker connectivity test without changing this package.
3. Build a trivial custom GraphFrontEnd application to confirm local `.aplx` compilation.
4. Run the H2 fixed-point/reference preflight locally.
5. Compile `h2_resident.aplx`.
6. Run one sample in `audit` mode and compare against frozen software checkpoints.
7. Run a batch in `profile` mode with no live state streaming.
8. Only after functional equivalence passes, sweep chips/cores/model dimensions for scaling.

Do not jump directly to multi-board scaling before the resident single-chip accounting is
understood.
