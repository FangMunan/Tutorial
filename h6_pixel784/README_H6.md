# H6 — 784-pixel Ionic SNN_Transform

## Scientific target

H6 is the final software interface before a new SpiNNaker migration. It deliberately replaces H4's 196 lossless 2x2 patch tokens with **784 individual binary pixels**:

- sequence length: `N = 784`
- input alphabet: `{0, 1, MASK}`
- output alphabet: `{0, 1}`
- task: **unconditional MNIST-012 generation**
- generation: monotonic 24-step MaskGIT-style demasking

The architectural core remains the H4 SNN_Transform family:

- positive random-feature normalized linear attention
- associative `H/g` state
- block-1 persistent `H/g` across generation rounds
- per-head selective retention gate
- deeper blocks recomputed fresh
- event-sparse new/changed K/V semantics retained for later hardware implementation

H6 is a **software positive-feature reference**. It makes no Izhikevich, fixed-point, SpiNNaker, latency, or energy claim yet.

## Why 784 pixels

The measured ionic controller addresses pixels. A 196-token 2x2-patch model requires an extra artificial rule mapping one physical pixel event to a 4-pixel token. H6 removes that mismatch:

`one physical scanner position <-> one image pixel <-> one model token`.

This is also a stronger sequence-length test for linear attention: moving 196 -> 784 tokens increases a dense all-pairs attention matrix by 16x, whereas the H/g linear-attention path is linear in token count at fixed feature/hidden dimensions.

## Frozen ionic protocol

The H6 ionic controller is copied from the accepted canonical measured PGI binary protocol. The ionic stream changes **mask/reveal locations only**; it does not control the internal retention gate or modify model weights.

Canonical file contract:

```text
PGI_ionic_binary.npz
  bits_concat       : 1-D uint8, values {0,1}
  sequence_ids      : physical sequence IDs
  sequence_offsets  : boundaries in bits_concat
```

The file is never re-thresholded, ranked, normalized, shuffled, spatially transformed, or joined across independent physical sequence boundaries.

Physical split:

```text
train       ionseq_001, ionseq_002, ionseq_003
validation  ionseq_004
generation  ionseq_005
```

Scanner rule for one image:

```text
pixel cursor: 0,1,...,783,0,1,...
bit 0       : consume event, no transition
bit 1       : transition only if current pixel is eligible
ineligible 1: consume event, no successful transition
stop step   : successful_transitions == quota
```

The pixel cursor resets for each image and persists across steps within that image. The ionic cursor persists through use of the same physical sequence and wraps only inside that sequence.

Forward training uses visible -> MASK transitions. Reverse generation uses MASK -> sampled {0,1} transitions. Exact successful-event quotas preserve the matched cosine mask-coverage schedule.

## H6 stages

### H6-A — standard 784-pixel controller

First establish that the new pixel-token SNN_Transform itself can learn and generate. This is the current cloud run and does **not** require ionic data.

```bash
python h6_pixel784/train_h6_pixel784.py \
  --out H6_PIXEL784_STANDARD \
  --controller standard \
  --epochs 20 --n-per-class 2000 --batch 32 \
  --d 128 --layers 4 --heads 4 --ff 512 --features 128 \
  --gate-epochs 1 --gate-samples 1024 --gate-batch 16 --gate-steps 4 \
  --decode-steps 24 --choice-temperature 4.5 --ngen 300 \
  --threads 4 --seed 7
```

Primary outputs:

- `FINAL_METRICS.json`
- `snn_transform_h6_pixel784_final.pt`
- `generation_standard_fresh_grid.png`
- `generation_standard_selective_grid.png`
- `trajectory_standard_fresh.png`
- `identity_audit.json`

### H6-B — measured ionic controller

Run only with the unchanged canonical measured `PGI_ionic_binary.npz`:

```bash
python h6_pixel784/train_h6_pixel784.py \
  --out H6_PIXEL784_IONIC \
  --controller ionic \
  --ionic /path/to/PGI_ionic_binary.npz \
  --epochs 20 --n-per-class 2000 --batch 32 \
  --d 128 --layers 4 --heads 4 --ff 512 --features 128 \
  --gate-epochs 1 --gate-samples 1024 --gate-batch 16 --gate-steps 4 \
  --decode-steps 24 --choice-temperature 4.5 --ngen 300 \
  --threads 4 --seed 7
```

This trains with ionic forward corruption from the outset and generates using the held-out physical generation sequence. It also generates standard-controller samples from the same trained model to expose controller/training effects.

## Required comparisons before hardware migration

The final software evidence should separate:

1. standard-trained + standard generation
2. ionic-trained + ionic generation
3. ionic-trained + standard generation
4. persistent/selective ON vs fresh read

The key question is not whether ionic masking necessarily improves MNIST accuracy. It is whether the final 784-pixel SNN_Transform remains a competent generator when the digital mask-location process is replaced by the frozen measured physical protocol, and what trajectory/distribution changes result.

Only after this software comparison is frozen should H7 map the 784-pixel target to SpiNNaker using the already validated H2 resident/fixed-point diagnostic framework.
