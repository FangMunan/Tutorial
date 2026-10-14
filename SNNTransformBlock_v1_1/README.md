# SNNTransformBlock-v1.1 — clean core

This folder contains the mathematically cleaned version of the frozen v1 block.
The purpose of v1.1 is **cleanup, not redesign**.

## What changed from v1

Exact/function-preserving cleanup:

1. `theta` removed from the FFN and absorbed into `f1.bias` during v1 checkpoint migration:

   `f1.bias_new = f1.bias_v1 - theta_v1`

2. fixed-retention `weight` state removed because frozen v1 uses `alpha = 1`.

3. raw `x_cache` removed from the mathematical core. Persistent calls now require an explicit Boolean changed-token mask.

4. `force_project_all`, runtime statistics and validation/accounting code removed from the hot path.

5. Izhikevich trajectory decoding is streamed online instead of materializing the whole membrane trajectory tensor.

The core persistent state is now exactly:

```text
S = (H, g, pq, pk, v)
```

## Canonical structure

```text
Input X + explicit changed-token events
  -> LayerNorm 1
  -> event Q/K/V projection updates
  -> positive-RF or fixed per-head Izh Q/K feature dynamics
  -> persistent H/g normalized linear attention
  -> sparse output projection
  -> residual
  -> LayerNorm 2
  -> sparse FFN input projection
  -> ReLU hidden population
  -> sparse FFN output projection
  -> residual
  -> Y + S
```

## Core API

```python
from SNNTransformBlock_v1_1 import SNNTransformBlock

block = SNNTransformBlock(d=32, heads=2, features=32, ff=64)

# fresh execution: all tokens are evaluated
state = None
y, state = block(x0, state)

# persistent execution: events are explicit
changed = torch.zeros(x1.shape[:2], dtype=torch.bool)
changed[:, [2, 7]] = True
y, state = block(x1, state, changed)
```

If `state` is not `None`, omitting `changed` is an error by design.  This prevents an O(BND) full-input comparison from silently entering the event-driven core.

## Optional convenience wrapper

For generic PyTorch use where an upstream event source is unavailable:

```python
from SNNTransformBlock_v1_1.wrappers import AutoEventWrapper

wrapped = AutoEventWrapper(block)
state = None
y, state = wrapped(x0, state)
y, state = wrapped(x1, state)
```

The wrapper stores `x_cache` and performs full tensor comparison.  Do not use it as the reference for scaling or SpiNNaker execution.

## Load the frozen v1 checkpoint into v1.1

```python
from SNNTransformBlock_v1_1 import load_v1_checkpoint_as_v1_1

block, old_metadata = load_v1_checkpoint_as_v1_1(
    "SNNTransformBlock_v1.pt",
    device="cpu",
)
```

The loader installs the saved per-head Izh interface, absorbs the historical `theta` into the first FFN bias, discards the fixed-alpha retention residue, and loads all remaining weights/masks.

## Save/load native v1.1 checkpoints

```python
from SNNTransformBlock_v1_1 import save_v1_1, load_v1_1

save_v1_1(block, "SNNTransformBlock_v1_1.pt")
block, metadata = load_v1_1("SNNTransformBlock_v1_1.pt")
```

## Important scaling boundary

`SparseLinear` in this portable PyTorch reference still stores a dense weight tensor and dense binary mask for checkpoint compatibility.  This preserves the sparse graph but does **not** claim sparse PyTorch execution.  A scaling/hardware backend should compile only the nonzero mask entries into compact synapse lists.

Likewise, the small Python loop over changed-token sets is a reference implementation.  SpiNNaker should consume event packets directly rather than reproducing this host-side loop.

## Validation

`test_v1_to_v1_1_parity.py` compares v1 and v1.1 for:

- exact PositiveRF execution with nonzero historical `theta`;
- event-updated persistent H/g state;
- neuronal Izh feature execution with streamed decoding;
- native v1.1 checkpoint round-trip;
- absence of `theta`, `weight` and `x_cache` from the core.

The corresponding GitHub Actions workflow is:

```text
.github/workflows/snn_transform_block_v1_1_clean.yml
```
