# H2 — Full Resident SpiNNaker Mapping

## 1. Purpose

This directory is the hand-off from the validated cloud model to the first **full-architecture** SpiNNaker experiment. The target is not an Izhikevich-only replay. The target is to keep the generation trajectory and the core model computation resident on SpiNNaker long enough to measure the architecture as a system.

Frozen software reference at the hand-off point:

- 49-token MNIST MaskGIT generator;
- width 128, 4 blocks, 4 heads, FFN width 512;
- normalized positive-feature H/g linear attention;
- persistent H/g only in block 1, with learned selective retention;
- monotonic reveal and validated H1 block-1 sparse execution;
- Q/K- and layer-specific 6-ms Izhikevich dynamic-feature calibration;
- B2.1 end-to-end feature replacement error approximately 0.063% fresh and 0.081% worst persistent trajectory.

H2 preserves this computation first. Architecture changes are allowed later, but only after a baseline hardware profile identifies the real bottleneck.

## 2. Important terminology

The immediate H2 experiment is **full-board inference / generation-trajectory validation**, not yet on-chip training. The current learned weights came from cloud training and there is not yet a frozen local on-chip learning rule for the full model. Calling this stage “training” would mix two scientific questions. Full on-chip learning should be introduced as a separate stage after the resident forward architecture is validated.

## 3. Resident execution is the primary communication rule

Normal H2 execution must be:

1. allocate the machine once;
2. upload model/calibration/input batches once;
3. execute all requested generation or teacher-forced steps on board;
4. store compact results and profiling counters in SDRAM;
5. bulk-read once at the end.

Do **not** implement the model as `run -> get_data -> host compute -> inject -> run` for every generation step. Do not stream membrane voltages, H/g matrices or layer activations to the host in production timing mode. A separate audit mode may record selected internal values for one or a few samples.

This is essential because the external Ethernet / remote-board path can dominate wall time even when the ARM cores themselves are fast.

Communication overhead should also be amortized across many inputs. Once the single-trajectory audit passes, profile throughput using a resident batch/queue of many samples under one model upload rather than allocating/reloading the machine for every image. In profile mode, return final decisions plus counters, not full per-token logits for every generation step.

The hardware model exporter additionally deduplicates byte-identical buffers. In the current software wrapper, each attention block contains three identical copies of the random-feature `omega` buffer (`phi`, `qmap`, `kmap`). The hardware image stores one physical copy and aliases the others. Izh decoder vectors are stored once in `calibration.bin` instead of being duplicated in the model image. This reduces host-to-board model bytes without changing model mathematics.

## 4. First validation target

Start with a **teacher-forced resident trajectory**. The full monotonic canvas trajectory is uploaded once. The board computes every layer, dynamic feature map, H/g write/read, selective alpha and output logits without host feedback. Only compact logits/checksums/profiling information are returned once.

After this passes, add the autonomous MaskGIT sampling controller on chip. This ordering separates numerical mapping errors from categorical sampling / RNG differences.

## 5. Mathematical contract

For one head:

    H = sum_i phi(k_i) v_i^T
    g = sum_i phi(k_i)
    y_j = phi(q_j)^T H / (phi(q_j)^T g + eps)

For the persistent first block:

    Hp_t = alpha_t Hp_{t-1} + newly_committed_writes_t
    gp_t = alpha_t gp_{t-1} + newly_committed_key_features_t

Current masked tokens contribute a non-decayed transient Hm/gm. `alpha_t` must be applied synchronously to both Hp and gp. Never decay H without g.

The current generation contract is monotonic reveal. A committed token is not re-masked. If this changes later, the cache/write rule must be replaced by explicit replacement or delta semantics.

## 6. Runtime audit: what is actually likely to dominate

A rough MAC-like accounting for the present dense channel dimensions shows that attention state is not necessarily the largest cost. For one 49-token block, approximate dominant terms are:

    Q/K/V/out projections     ~3.21 M MAC
    two-layer FFN             ~6.42 M MAC
    Q/K random-feature proj   ~1.61 M MAC
    H/g build + read          ~1.63 M MAC-equivalent

Thus the conventional FFN and dense projections remain major costs. The full four-block, ten-round model is on the order of several hundred million MAC-like operations before counting LayerNorm/GELU/Izh updates. Therefore H2 profiling must not assume that the H/g attention core is the bottleneck. It may reveal that dense channel mixing is the next part that needs dynamical/event conversion.

The model image is only a few MB, but the dominant static storage is also informative: the FFN weights account for roughly half of the current FP32 tensor bytes, attention projections for roughly a quarter, and the random-feature matrices for a substantial remaining fraction. This is another reason to profile the conventional dense channel path rather than optimizing only H/g.

## 7. Block-1 resident cache audit

A mathematically valid optimization was identified because the input to block 1 is only token + position + class embedding. Under monotonic reveal each position changes at most once:

    MASK -> committed token.

Therefore repeated Q/K/V projection and repeated 6-ms Izh feature evaluation can in principle be cached. The audit found the following operation-count reduction for block-1 projection/feature evaluations over ten rounds:

    Q evaluations:       -80%
    K/V evaluations:     -66.7%
    combined Q + K/V:    -75%

However, changing PyTorch GEMM batch shapes produced a small floating-point difference even though the algebra is unchanged. The best current audit reached roughly:

    max relative L2 ~2.7e-5
    max absolute logit difference ~2.3e-3

This is tiny relative to the eventual fixed-point mapping error, but it failed the deliberately strict original FP32 gate. Therefore the **canonical first H2 hardware bundle does not enable this cache**. It keeps the already validated H1 sparse semantics. `h2_block1_cache_audit.py` remains an optional performance optimization to reconsider after the baseline hardware mapping is stable.

Do not claim the cache as a zero-error optimization in the manuscript unless the numerical contract is explicitly relaxed and justified.

## 8. Memory / mapping strategy

The complete persistent H state is

    4 * 128 * 32 = 16384 values.

At 32-bit representation this is about 64 KiB before code or scratch space, so it is not a sensible single-core DTCM object on SpiNNaker-1. Per-head H is 4096 values and is a natural later multi-core shard.

The whole trained network is only a few MB at 32-bit precision, so the current model comfortably fits chip SDRAM. Large weights should therefore live in SDRAM and be DMA-tiled into DTCM. They should not be repeatedly uploaded from the host.

For correctness, the first custom kernel may be a serial resident baseline. After it passes, the preferred performance mapping is a one-chip cooperative pipeline before any cross-chip model partitioning. This isolates compute/DMA cost from router and inter-chip cost.

## 9. Fixed-point is a mandatory mapping gate

The cloud reference is FP32 PyTorch. SpiNNaker-1 ARM968 cores do not provide hardware floating point suitable for treating software float as the representative high-performance path. H2 therefore needs a fixed-point preflight before performance claims.

The fixed-point audit must separately compare:

- fresh logits;
- persistent trajectory logits;
- alpha values;
- selected H/g entries or checksums;
- final token decisions.

The B2.1 Izh decoder, especially some K-side decoders with larger alternating coefficients, may be sensitive to fixed-point cancellation. If necessary, recalibrate the decoder directly under the chosen fixed-point arithmetic rather than blindly copying the FP32 decoder coefficients.

## 10. Communication modes

`profile` mode returns only final token/logit summaries, stage cycle counters, DMA-byte counters, packet/provenance counters, memory high-water marks and error flags. No internal trajectory streaming is allowed.

`audit` mode is for one or a few samples and may additionally return selected layer states/checksums. Its communication time must never be reported as production inference timing.

Measure and report host-side setup/upload/read time separately from on-chip cycle counts. A fast core computation plus slow machine allocation/Ethernet transfer is not a slow neural architecture; conversely, excluding communication entirely is not a fair end-to-end deployment metric. Both numbers are needed.

## 11. Files in this hand-off

- `h2_block1_cache_audit.py` — optional runtime optimization audit; currently not canonical.
- `export_h2_resident_bundle.py` — exports the frozen B2.1 model, calibration and deterministic teacher trajectory into one self-contained bundle. Default export uses the validated H1 sparse block-1 path, deduplicates repeated hardware buffers, and emits `model_layout_generated.h` for the C kernel.
- `resident_vertex.py` — custom GraphFrontEnd data-spec / recording interface for a one-upload, one-read resident application.
- `run_h2_resident.py` — host launcher; setup once, run once, bulk-read once.
- `c_src/h2_resident_format.h` — frozen host/kernel binary interface.
- `c_src/README_KERNEL_CONTRACT.md` — mathematical and profiling contract for the on-chip kernel.
- `c_src/h2_resident.aplx` — **not yet supplied**. The on-chip C kernel must be compiled after Codex inspects the exact campus SpiNNaker toolchain/API version. Do not fabricate compatibility before that environment is visible.

The missing `.aplx` is deliberate: the mathematical reference and host communication contract are frozen here, while the hardware-specific C implementation is the next Codex task inside WSL. The launcher fails explicitly if the binary is absent.

## 12. Required sequence after WSL obtains campus access

1. Inspect exact installed `sPyNNaker`, `SpiNNakerGraphFrontEnd`, `spinnaker_tools`, compiler and machine-allocation versions.
2. Re-run the existing minimal campus connectivity test.
3. Compile/run an official trivial custom GraphFrontEnd application to verify local `.aplx` toolchain compatibility.
4. Run the canonical H2 bundle exporter and freeze its SHA256 manifest.
5. Implement the resident C kernel against the frozen mathematical contract and generated tensor layout header.
6. Perform fixed-point preflight locally and, if needed, fixed-point-aware Izh decoder recalibration.
7. Run one sample/trajectory in `audit` mode and compare checkpoints with the reference bundle.
8. Run many resident samples in `profile` mode with no live state streaming, amortizing model upload/allocation overhead.
9. Only after functional equivalence passes, parallelize within a chip and begin scaling sweeps.

The central rule is: first establish a resident, auditable single-chip baseline; then let measured compute/DMA/routing costs determine the next architecture optimization.
