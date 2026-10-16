# H3B — single-chip distributed scaling

## Frozen input from H3A

H3A is frozen conservatively at the largest physically verified safe point:

```text
N_single_core_baseline = 38
```

This is **not** claimed as the absolute ARM968/DTCM maximum. It is the largest
current resident mapping point that completed the full six-step block and was
bit-exact in `y,H,g,pq,pk,v`. H3B deliberately uses a smaller per-worker shard
(`N_local=32`) to leave DTCM headroom for inter-core communication state.

## H3B scientific question

Can one *single* frozen SNNTransformBlock grow beyond the one-core resident
limit by distributing token-local state across several ARM cores on one
SpiNNaker chip, while preserving the frozen H1.1-functional-v9 arithmetic?

H3B is not a throughput test of independent blocks. All participating cores
jointly execute one global block.

## H3B0 mapping

The distributed mapping uses contiguous token shards:

```text
worker 0 : tokens [0, 32)
worker 1 : tokens [32, 64)
...
```

Each worker keeps resident only its local `pq,pk,v` plus a replicated global
`H,g` and its local output/tail workspace. Replicating `H,g` costs 4352 bytes
for the frozen D32/H2/M32 configuration and keeps the normalized H/g read local
on every worker.

Worker 0 is a communication/reduction coordinator only; it does not perform
host-side mathematics and does not change the attention equation. Step-0 H is
reduced in unrounded int64 and rounded once after the global token sum. Later
updates sum the exact per-token rounded H/g deltas. The host package gate proves
this sharded arithmetic bit-exact to the global H2D integer reference before a
board package is emitted.

## H3B0 physical result — PASS

Physical run timestamp: `2026-10-16T09:33:39Z`.

Configuration:

```text
workers = 2
N_local = 32
N_global = 64
steps = 6
changed fraction = 0.25
D=32, H=2, M=32, FF=64, L=1
worker arena = 50,436 bytes
APLX size = 21,200 bytes
```

Physical placement:

```text
rank 0 -> chip (0,0), core 3
rank 1 -> chip (0,0), core 4
```

Observed board result:

```text
PASS = true
same_chip = true
all_arrays_bit_exact = true
protocol_errors = 0 on both workers
tx_retries = 0 on both workers
max_worker_elapsed_ms = 924
total Q/K/V evals = 144 / 144 / 144
total tx packets = 19,608
total rx packets = 19,608
```

Final concatenated `y,pq,pk,v` were bit-exact to the global oracle. Both H/g
replicas were bit-exact as well; every reported max absolute difference was 0.
The worker-local Q/K/V evaluation counts were exactly the preregistered counts
(77 and 67 respectively).

Therefore H3B0 is frozen as a physical correctness/communication PASS:

```text
one global N=64 frozen block
= 2 token-sharded ARM cores
= same physical chip
= no host intermediate compute
= no hidden-state host roundtrip
= bit-exact global state/output
```

This result establishes distributed correctness, not scaling efficiency.

## H3B1 — physical weak scaling

H3B1 keeps the per-worker resident workload fixed:

```text
N_local = 32
changed tokens per worker per update = 8
steps = 6
D=32, H=2, M=32, FF=64, L=1
pacing = 1 us/packet
```

and sweeps:

```text
P = 2, 4, 8, 16
N = 64, 128, 256, 512
```

For H3B1 the update masks are balanced per worker, rather than selected only by
a global random mask. This removes the P=2 H3B0 imbalance (11/5, 8/8, ...)
from the weak-scaling timing measurement. Every worker therefore executes 32
Q/K/V projections at step 0 and exactly 8 at each of the next five steps:

```text
expected Q/K/V evals per worker = 32 + 5*8 = 72
```

For every point record:

- same-chip placement and number of cores used;
- bit-exact `y,H,g,pq,pk,v` correctness;
- board-reported maximum worker elapsed time;
- total Q/K/V evaluations;
- multicast TX/RX packets and retries;
- protocol errors;
- first failed point and its actual failure mode.

A failed point is only a **frontier candidate** until its cause is diagnosed.
Do not label a placement, routing, numerical-range, synchronization, DTCM, or
runtime failure as another bottleneck without evidence.

Only after this baseline sweep is known do we consider H/g sharding,
communication-tree reduction, pacing changes, or DTCM buffer-lifetime
optimisation.
