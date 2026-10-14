# H0 — SpiNNaker operator mapping audit for SNNTransformBlock-v1.1

Date: 2026-10-14

Goal: map the frozen `SNNTransformBlock-v1.1` mathematics onto SpiNNaker execution primitives **without changing the scientific operator**.

Frozen source of truth:

```text
Input X + explicit changed-token events
  -> LN1
  -> event Q/K/V updates
  -> per-head Izh Q/K feature dynamics
  -> persistent state S=(H,g,pq,pk,v)
  -> normalized H/g read
  -> sparse WO
  -> residual
  -> LN2
  -> sparse W1
  -> exact-zero ReLU population
  -> sparse W2
  -> residual
  -> Y + state
```

## Classification

- **A — native/direct**: maps naturally to an existing SpiNNaker mechanism.
- **B — custom APLX**: no suitable PyNN primitive, but the same mathematical operator can be implemented resident on ARM cores.
- **C — numerical backend substitution**: same operator, but floating-point reference arithmetic must become fixed/integer arithmetic and pass parity gates.
- **D — scientific change**: changes the operator/representation rather than execution; forbidden in the first board implementation.

## Hardware facts that constrain the mapping

The target SpiNNaker-1 ARM968 cores have no hardware floating point, limited local DTCM/ITCM, integer/DSP multiply-accumulate support, DMA access to SDRAM, and multicast packet routing. The software stack supports custom applications/APLX and fixed-point libraries. Standard sPyNNaker already provides Izhikevich neuron implementations, but the v1.1 Izh interface additionally requires a resident trajectory decoder.

Useful official references:

- SpiNNaker technical documents: https://spinnakermanchester.github.io/docs/
- fixed-point support (`spinn_common`): https://spinnakermanchester.github.io/spinn_common/stdfix-full-iso_8h.html
- SpiNNaker workshop hardware/arithmetic limits: https://spinnakermanchester.github.io/workshops/sixth/proceedings.pdf
- sPyNNaker model limitations / fixed-point parameter representation: https://spinnakermanchester.github.io/spynnaker/6.0.0/SPyNNakerModelsAndLimitations.html

## Operator-by-operator audit

| v1.1 operator/state | H0 class | Board representation | First implementation decision |
|---|---:|---|---|
| explicit changed-token mask | A | multicast event key / event queue | represent only changed token IDs; do not scan a Boolean matrix |
| token vector `x_i in R^d` | B+C | fixed-point vector in DTCM/SDRAM, selected by token event | do **not** introduce rate/latency coding in H1/H2 |
| LN1/LN2 | B+C | integer mean/variance + integer sqrt/inverse scaling | preserve LayerNorm exactly; no RMSNorm substitution |
| dense `WQ/WK/WV` | B+C | event-triggered fixed-point GEMV | run only for changed tokens |
| head reshape/split | A | address/layout convention | zero arithmetic cost other than indexing |
| random projection / `stable_a` | B+C | integer MAC + norm term + query max subtraction + clamp | preserve current formula and clamps |
| Izhikevich RS dynamics | A+B+C | resident discrete Izh steps; fixed-point state | same update equations; may fuse the 20/24 substeps inside one event handler because intermediate states are not externally coupled in v1.1 |
| Izh linear trajectory decoder | B+C | online accumulator `r += d_t v_t` | never record/download the full voltage trajectory |
| cached `pq, pk, v` | A | resident DTCM/SDRAM arrays | preserve mutable-token cache semantics |
| persistent `H, g` | B+C | resident state arrays, int64 accumulators if needed | update by subtract-old/add-new on token events |
| normalized H/g read | B+C | integer MAC + integer division/reciprocal | preserve numerator and denominator; numerical parity required |
| sparse `WO` | A+B+C | compact row/synapse lists + fixed-point weights | compile only existing connections |
| residual add | B+C | fixed-point vector add | direct local arithmetic |
| sparse `W1` | A+B+C | compact sparse rows | direct sparse fixed-point MAC |
| exact-zero ReLU | B | compare-to-zero and suppress zero outputs | hidden nonzero channels can become events for W2 |
| sparse `W2` | A+B+C | event-conditioned sparse fan-out / sparse MAC | exploit both structural and hidden activity sparsity |
| output/state readback | A | SDRAM/output buffer, DMA/SDP only at run boundary | no per-step host round-trip |

## Explicitly forbidden first-stage substitutions (class D)

The following may be interesting later, but they are **not** allowed while proving the SpiNNaker implementation of v1.1:

- LayerNorm -> RMSNorm;
- real-valued token vector -> rate/latency/population spike code;
- normalized H/g attention -> unnormalized or approximate different attention rule;
- Izh feature dynamics -> lookup table that bypasses the Izh state equations;
- sparse FFN -> a different WTA/top-k architecture;
- host-side H/g update, decoder, or FFN inside every simulation step;
- reintroducing the removed `x_cache`, fixed-retention `weight`, or selective/Mamba state into the core.

## H0 execution partition

Recommended first resident decomposition:

```text
changed-token packet
      |
      v
[Token-update kernel]
LN1 -> Q/K/V -> stable_a
      |
      v
[Izh-feature kernel]
Izh substeps + online decoder
      |
      v
[Persistent attention kernel]
pq/pk/v cache + H/g subtract/add + normalized read
      |
      v
[Sparse tail kernel]
WO -> residual -> LN2 -> W1 -> ReLU -> W2 -> residual
      |
      v
output/state buffer
```

This is a logical partition, not yet a claim that each box must occupy one physical core. H2 placement can fuse/split kernels according to DTCM, cycles, and packet pressure while preserving the IR.

## H0 hardware intermediate representation (HIR)

The board backend should consume a hardware-neutral IR rather than PyTorch objects:

```text
BlockConfig
QKV matrices + biases
LN1/LN2 gamma,beta,epsilon
omega per head
Izh per-head {bias,gain,window,decoder}
Sparse WO/W1/W2 row lists
Token state {pq,pk,v}
Global state {H,g}
Fixed-point format metadata
Event stream {token_id, value_location/time}
```

The HIR must be usable by both:

1. H1 CPU integer/fixed-point emulator;
2. H2/H3 C/APLX SpiNNaker backend.

## H0 acceptance criteria

H0 is complete when:

1. every v1.1 operation is assigned to A/B/C/D;
2. no required operation is left as “PyTorch-only”;
3. no class-D substitution is needed for feasibility;
4. the minimal state remains exactly `(H,g,pq,pk,v)`;
5. there is a concrete integer-runtime route for LN, QKV, Izh, H/g normalization, and sparse tail.

**H0 verdict: PASS at the architecture/mapping level.** No v1.1 operator requires a scientific change to map onto SpiNNaker. The hard work is implementation/numerical engineering, concentrated in fixed-point LN, QKV/stable-a, Izh decoding, persistent H/g, normalized readout, and compact sparse execution.
