# SNNTransformBlock-v1 — standalone engineering interface

This folder is the clean engineering entry point for the frozen lightweight block.
It is deliberately separated from the U0–U5 experiment scripts.

## Files

- `SNNTransformBlock_v1.py` — self-contained block implementation; only PyTorch is required.
- frozen checkpoint from U5: `SNNTransformBlock_v1.pt` (currently emitted by the GitHub Actions artifact `SNN_TRANSFORM_BLOCK_V1`).

The experiment history and scientific scope are recorded separately in:

`unified_snn_transform/PROJECT_STATUS.md`

## Canonical architecture

```text
Input [B,N,D]
  -> LayerNorm
  -> event-cached Q/K/V
  -> per-head Izh Q/K feature dynamics
  -> persistent normalized H/g linear attention
  -> 25% structural sparse output projection
  -> residual
  -> LayerNorm
  -> 25% structural sparse FFN input
  -> thresholded exact-zero hidden population
  -> 25% structural sparse FFN output
  -> residual
  -> Output [B,N,D] + persistent state
```

Mamba-like selective retention is parked and is not part of v1.  Default retention is fixed at `alpha=1`.

## 1. Use as a generic trainable block

```python
import torch
from SNNTransformBlock_v1 import SNNTransformBlock

block = SNNTransformBlock(
    d=32,
    heads=2,
    features=32,
    ff=64,
    out_density=0.25,
    ff1_density=0.25,
    ff2_density=0.25,
)

x = torch.randn(8, 16, 32)
state = None

y, state = block(x, state)
```

This constructor starts with the exact positive-RF feature interface.  It is the clean choice when adapting/training the architecture for a new task.

## 2. Persistent/event execution

```python
state = None

y0, state = block(x0, state)
y1, state = block(x1, state)     # only changed tokens need new Q/K/V projections
```

If the upstream application already knows which tokens changed, pass the event mask explicitly:

```python
changed = torch.zeros(B, N, dtype=torch.bool)
changed[:, [2, 7]] = True

y, state = block(x, state, changed=changed)
```

If `changed` is omitted, the block compares the current input against `state.x_cache` and infers the changed tokens.

## 3. Reset

```python
y, state = block(x, state, reset=True)
```

or simply:

```python
state = None
```

## 4. Load the frozen U5 block

After downloading the U5 artifact checkpoint `SNNTransformBlock_v1.pt`:

```python
from SNNTransformBlock_v1 import load_frozen_v1

block, metadata = load_frozen_v1(
    "SNNTransformBlock_v1.pt",
    device="cpu",
)

state = None
y, state = block(x, state)
```

The loader automatically installs the saved per-head Izh operating points and decoders before loading the model state.  A downstream application does not need to repeat U4G calibration.

## 5. Inspect sparsity/event statistics

After a call:

```python
print(block.last_stats)
print(block.synapse_account())
```

`last_stats` includes Q/K/V token evaluations, number of changed tokens and hidden activity.  `synapse_account()` reports the sparse-tail connectivity/effective SynOp proxy.

## Frozen-v1 scope

U5 established the following for the lightweight gate configuration:

- canonical packaging error: 0
- task-output Izh parity: ~0.971% relative-L2
- task sign accuracy: 100% reference and candidate
- event-cached Q/K/V: 62.5% fewer token projections in the tested 25%-change trajectory
- sparse tail: 25% static connectivity and ~21.4% effective dense-tail proxy in that test

Important limitation: full hidden-block representation parity was ~5.29% relative-L2, so v1 is frozen as a functional/architectural block, not as a claim of <=1% equivalence for every internal channel.

## Recommended downstream contract

Treat the block as an operator with the following stable interface:

```python
y, state = block(x, state, changed=None, reset=False)
```

Do not modify the internal architecture for the first downstream validation.  Adapt task-specific input/output adapters around the block instead.  Only reopen block internals if an independent workload exposes a specific failure mode.
