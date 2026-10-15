# H3A — Single-core scaling gate

## Hierarchy

The hardware scaling programme is deliberately hierarchical:

1. **H3A — single ARM core limit**
2. **H3B — single SpiNNaker chip limit**
3. **H3C — single physical board limit**

H2F remains the N=16 resident correctness anchor.

## Frozen H3A baseline

The conservative physically verified single-core baseline is now frozen as:

```text
N_single_core_baseline = 38
D=32, H=2, M=32, FF=64, L=1
steps=6
changed fraction=0.25
```

At N=38 the complete resident block executed on one physical ARM core and the
final `y,H,g,pq,pk,v` were all bit-exact to the host integer oracle.  The
current arena request was:

```text
arena_bytes(N=38) = 58,880 B
```

The attempted N=40 point requested 61,696 B and repeatedly failed to complete
with the patched H2F failure path.  This behaviour is consistent with the
available DTCM heap being exhausted, but the patched binary did not provide a
clean post-`spin1_start` capacity signal.  Therefore H3A deliberately does
**not** claim that 38 is the absolute physical maximum.  It is the largest
verified safe point and is used as the baseline for the next scaling level.

## Current resident arena

For the H2F/H3A mapping:

```text
arena_words = 1344 + 352*N
arena_bytes = 5376 + 1408*N
```

This includes resident `H,g,pq,pk,v` plus the full-token work buffers
`ln_buf, attended, z_buf, hidden, y_buf` and token/feature scratch space.

Representative requests:

| N | requested arena bytes | status |
|---:|---:|---|
| 16 | 27,904 | verified pass anchor |
| 24 | 39,168 | verified pass |
| 32 | 50,432 | verified pass |
| 36 | 56,064 | verified pass |
| 38 | 58,880 | verified pass, frozen baseline |
| 40 | 61,696 | probable DTCM frontier; not used as a clean capacity claim |

## Interpretation

H3A identifies local resident memory as the first practical scaling bottleneck
of the current mapping.  It is not an arithmetic-throughput limit of the
ARM968 and not an absolute optimized DTCM limit.  Buffer-lifetime reuse and
other memory-layout optimization are parked as a possible later `H3A-OPT`
stage.

The immediate next stage is **H3B single-chip distributed scaling**.  H3B uses
`N_local=32` per worker rather than 38 so that each core retains DTCM headroom
for multicast/synchronization state.

## Claim boundary

H3A establishes only the safe capacity/execution baseline of the current
single-core resident mapping.  It does not establish single-chip scaling,
router scaling, inter-core communication efficiency, single-board scaling,
task accuracy, training efficiency, or energy efficiency.
