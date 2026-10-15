# H3B — single-chip distributed scaling

## Frozen input from H3A

H3A is frozen conservatively at the largest physically verified safe point:

```text
N_single_core_baseline = 38
```

This is **not** claimed as the absolute ARM968/DTCM maximum.  It is the largest
current resident mapping point that completed the full six-step block and was
bit-exact in `y,H,g,pq,pk,v`.  H3B deliberately uses a smaller per-worker shard
(`N_local=32`) to leave DTCM headroom for inter-core communication state.

## H3B scientific question

Can one *single* frozen SNNTransformBlock grow beyond the one-core resident
limit by distributing token-local state across several ARM cores on one
SpiNNaker chip, while preserving the frozen H1.1-functional-v9 arithmetic?

H3B is not a throughput test of independent blocks.  All participating cores
jointly execute one global block.

## H3B0 mapping

The first distributed mapping uses contiguous token shards:

```text
worker 0 : tokens [0, 32)
worker 1 : tokens [32, 64)
...
```

Each worker keeps resident only its local

```text
pq, pk, v
```

plus a replicated global

```text
H, g
```

and the local output/tail workspace.  Replicating `H,g` costs only 4352 bytes
for the frozen D32/H2/M32 configuration and keeps the normalized H/g read local
on every worker.

## Deterministic single-chip reduction protocol

H3B0 uses worker 0 as a **communication coordinator only**; it does not perform
host-side mathematics and does not change the attention equation.

For step 0:

1. every worker computes local Q/K/V and Izh feature maps;
2. each worker forms an unrounded int64 partial H sum and int64 g sum;
3. workers send those partials to worker 0 using on-chip multicast;
4. worker 0 sums partials and performs the same single final H rounding as the
   original one-core `h2_attention_state_init`;
5. worker 0 multicasts the resulting global H/g back to all workers;
6. each worker performs the attention read and sparse tail for its local tokens.

For later steps:

1. each worker evaluates Q/K/V only for its local changed tokens;
2. it forms an int64 sum of the exact per-token rounded H/g deltas;
3. worker 0 gathers the partial deltas in rank order, updates global H/g and
   broadcasts the new H/g;
4. all workers execute their local read/tail.

The package generator must prove that this sharded reduction is bit-exact to
the global H2D integer reference before a board package is emitted.

## H3B0 first physical gate

The first board point is intentionally small:

```text
workers = 2
N_local = 32
N_global = 64
steps = 6
changed fraction = 0.25
D=32, H=2, M=32, FF=64, L=1
```

Both workers are constrained to chip `(0,0)` but not to a specific processor
ID; PACMAN chooses free application cores on that chip.

Acceptance:

- all worker placements are on the same physical chip;
- all workers finish without protocol error or packet-count mismatch;
- concatenated board `y,pq,pk,v` are bit-exact to the global integer oracle;
- every worker's replicated `H,g` are bit-exact to the same global oracle;
- summed Q/K/V evaluation counts match the explicit global changed-token masks;
- host intermediate compute and hidden-state roundtrips remain zero.

H3B0 is a correctness/communication gate only.  It does not yet establish
single-chip scaling efficiency.

## After H3B0 passes

H3B1 fixes `N_local=32` and increases worker count, nominally

```text
P = 2, 4, 8, ...
N = 32 P
```

until the chip reaches a placement, routing, packet, synchronization, runtime,
or memory frontier.  We will record wall latency, board-reported execution
time, packet counts/retries, cores used and correctness.  Only after that
baseline is known do we consider H/g sharding or communication optimisation.
