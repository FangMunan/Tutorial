# SNNTransformBlock-v1.1 SpiNNaker backend — H0/H1 status

Branch: `snn-transform-spinnaker-h0-h1-20261014`

## H0 — operator mapping audit

**PASS.** Every frozen v1.1 operator has a SpiNNaker execution route without changing the scientific operator.

Key decisions:

- explicit changed-token mask -> multicast/event IDs;
- real token vector remains a fixed-point vector; no new spike code is introduced;
- LayerNorm stays LayerNorm;
- Q/K/V become event-triggered fixed-point GEMV;
- Izh equations are preserved and fused with the already-approved online trajectory decoder;
- H/g and token caches remain resident state;
- normalized H/g read remains numerator/denominator computation, implemented in integer arithmetic;
- sparse WO/W1/W2 compile to compact sparse rows/synapses;
- residuals remain vector addition;
- no per-step host round-trip is permitted in the target backend.

See `H0_OPERATOR_MAPPING_AUDIT.md` for the A/B/C/D classification.

## H1 — CPU integer/fixed-point hardware emulator

**PASS** on the deterministic representative v1.1-shaped numerical gate.

The H1 runtime contains no PyTorch operator. Its HIR arrays are integer-only. Float is used only by the offline compiler/reference side.

Chosen first fixed-point contract:

- main token/vector values: 14 fractional bits;
- phi values: 12 fractional bits;
- Izh membrane state: 20 fractional bits;
- Izh decoder coefficients: 18 fractional bits;
- MAC/state products: int64 accumulator where required.

Six-step mutable-token trajectory, B=4, N=16, D=32, 25% tokens changed after the first step:

- max full-block output relative-L2: `0.0004909658542376653` (~0.0491%);
- max H relative-L2: `0.0007905215521152793` (~0.0791%);
- max g relative-L2: `0.000837216628775985` (~0.0837%);
- max pq relative-L2: `0.006557911383955102` (~0.656%);
- max pk relative-L2: `0.0008226346436349562` (~0.0823%);
- max v relative-L2: `0.0002123315621174444` (~0.0212%).

Event Q/K/V execution remained exact at the scheduling level:

- dense token evaluations per Q/K/V: `384`;
- event token evaluations per Q/K/V: `144`;
- reduction: `62.5%`.

Sparse tail:

- dense reference connections: `5120`;
- compiled compact sparse connections: `1280`;
- static fraction: `25%`.

Representative resident integer state stayed inside signed 32-bit range in this gate:

- max |H| = `418519840`;
- max |g| = `36569`;
- max |pq| = `724`;
- max |pk| = `9620`;
- max |v| = `32133`.

All H1 gates passed.

## Critical scope limitation

H1 is a **backend numerical proof**, not yet a board result and not yet a parity test using the frozen trained `SNNTransformBlock-v1.1` checkpoint. The representative HIR uses the same dimensions, operator graph, sparse density, event semantics, and U4G-style Izh operating points, but deterministic synthetic weights/decoder fits.

Therefore the next required gate before APLX implementation is:

### H1.1 — frozen-checkpoint -> HIR export/parity

1. load/migrate the frozen v1/v1.1 checkpoint;
2. export its actual Q/K/V, LayerNorm, omega, Izh decoders, sparse WO/W1/W2 into the same HIR;
3. profile real trained dynamic ranges;
4. choose/freeze fixed-point formats from those real ranges;
5. rerun the same six-step event/parity gate;
6. only if H1.1 passes, begin H2 C/APLX kernels.

## Decision

**Freeze the H0 mapping and H1 integer-runtime structure.** Do not redesign the block for SpiNNaker. The next work is checkpoint export/range validation, followed by direct translation of the already-passing integer kernels into resident C/APLX code.
