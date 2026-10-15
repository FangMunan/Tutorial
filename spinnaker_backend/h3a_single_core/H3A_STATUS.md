# H3A — Single-core scaling gate

## Hierarchy

The hardware scaling programme is deliberately hierarchical:

1. **H3A — single ARM core limit**
2. **H3B — single SpiNNaker chip limit**
3. **H3C — single physical board limit**

H2F is frozen as the N=16 resident correctness anchor. H3A does not modify H2F.

## H3A scientific question

With the frozen `H1.1-functional-v9` mathematics and fixed
`D=32, heads=2, features=32, FF=64, L=1`, how far can token count `N` grow on
**one physical ARM core** before the current resident mapping reaches its real
DTCM capacity frontier?

Only `N` changes. This is a hardware microbenchmark, not a task benchmark.

## Current resident arena

For the current H2F mapping, the exact requested DTCM arena is

```text
arena_words = 1344 + 352*N
arena_bytes = 5376 + 1408*N
```

This includes resident `H,g,pq,pk,v` plus the current full-token work buffers
`ln_buf, attended, z_buf, hidden, y_buf` and token/feature scratch space.

Predicted requests for the preregistered coarse sweep:

| N | requested arena bytes |
|---:|---:|
| 16 | 27,904 |
| 24 | 39,168 |
| 32 | 50,432 |
| 36 | 56,064 |
| 38 | 58,880 |
| 40 | 61,696 |
| 42 | 64,512 |
| 44 | 67,328 |
| 48 | 72,960 |

These are requested payload bytes, not an assumption about usable DTCM. The
board APLX measures the actual SARK heap immediately before allocation:

- `heap_free_before`
- `heap_max_before` (largest contiguous allocatable payload)
- requested arena bytes
- post-allocation heap values

Therefore the capacity boundary is measured rather than inferred from the
nominal 64-KB DTCM size.

## Acceptance semantics

For an `N` that allocates successfully:

- all six steps execute resident on the same physical core;
- no host intermediate compute or hidden-state roundtrip occurs;
- Q/K/V evaluation count must match the explicit changed-token masks;
- final `y,H,g,pq,pk,v` must be bit-exact to the host integer oracle.

`rc=20` is an expected **CAPACITY_FAIL**, not a scientific failure, if the APLX
reports that the largest free DTCM heap block is smaller than the requested
arena. Other return codes remain failures.

## Why H3A is baseline-first

H3A first measures the frozen H2F memory layout **without buffer-lifetime
optimisation**. Once the baseline boundary is observed, a separately labelled
H3A-OPT stage may reuse work buffers and reduce DTCM pressure. This prevents an
optimised result from hiding the original bottleneck and gives a clean
before/after efficiency result.

## Preboard sweep

Coarse points:

```text
N = 16, 24, 32, 36, 38, 40, 42, 44, 48
```

with fixed:

```text
steps = 6
changed fraction = 0.25
D=32, H=2, M=32, FF=64, L=1
```

After the first successful/failed bracket is observed, perform a fine sweep at
integer N around the boundary and freeze

```text
N_max_single_core_baseline
```

before any H3A-OPT work.

## Claim boundary

H3A can establish only the capacity and execution scaling of the current
**single-core mapping**. It does not establish single-chip scaling, router
scaling, inter-core communication efficiency, single-board scaling, task
accuracy, training efficiency, or energy efficiency.
