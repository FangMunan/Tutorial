# P4-E27 — B0-faithful dynamical MaskGIT + MSDL

This gate tests one narrow question: can the frozen MaskGIT-compatible image-generation scaffold be learned with local dynamical MSDL when standard attention is replaced by resident H/g computation?

## Frozen scaffold

- MNIST 28x28.
- 4x4 patches -> 7x7 = 49 visual-token positions.
- unlabeled MiniBatchKMeans tokenizer, K=64; MASK token=64.
- d=128, four blocks, four heads, FF=512, dropout=0.05.
- learned token and position embeddings.
- dense -> GELU -> LayerNorm prediction head.
- output weights tied to the first 64 token embeddings.
- cosine random-mask training protocol.
- 16-step confidence-based cosine MaskGIT decoding.
- no class embedding.

Each attention block uses resident sufficient states

`g_h = sum_i phi_k(k_hi)`

`H_h = sum_i phi_k(k_hi) v_hi^T`

and reads

`a_ht = phi_q(q_ht)^T H_h / (phi_q(q_ht)^T g_h)`.

No N x N attention matrix is constructed.

## Learning

The candidate does not call `backward()` and has no gradient optimizer. Prediction mismatch at every masked position drives a dual-timescale M state; local forward activity drives a dual-timescale E state. Synaptic changes use local M-E outer products. Learned reciprocal populations route mismatch to deeper blocks. Q/K credit is estimated through resident-state perturbations; V credit uses the resident sufficient-state path.

The frozen E/M recurrence is

`Ef <- .62 Ef + .38 x`, `Es <- .92 Es + .08 x`

`Mf <- .62 Mf + .38 m`, `Ms <- .92 Ms + .08 m`

with `E=.70 Ef+.30 Es` and `M=.70 Mf+.30 Ms`.

All masked positions in an image contribute to the learning event. H/g is per-image working state and is rebuilt for each corrupted image; long-term learning is stored in synaptic parameters.

## Labels and tokenizer

Digit labels are used outside the candidate only to define the controlled 0/1 benchmark and to train the final post-hoc evaluator. They are never passed to the generative candidate. The KMeans tokenizer is fitted to patches without digit labels and is frozen for this gate.

## Run

Workflow: `.github/workflows/snn_transform_e27_msdl_cloud.yml`.

Default cloud run uses a 55-minute candidate-training wall budget and an 80-minute workflow timeout. It checkpoints after every epoch and uploads `metrics.json`, `generation.png`, `generation.npz`, `latest.pt`, `codebook.npy`, and `console.log` as artifact `SNN_TRANSFORM_E27_MSDL_CLOUD`.

The run reports both the empirical token-marginal baseline and the untrained model's 50/90/100%-mask validation metrics. This makes improvement attributable to MSDL measurable before judging image quality.

## Gate interpretation

A useful run should show decreasing held-out masked-token CE, especially at 90% and 100% mask, without numerical instability. The primary generation gate is then the fixed-seed all-MASK gallery plus mode diversity and nearest-real-image statistics. If conditional learning improves monotonically but has not converged within the cloud budget, extend training or move the same state equations to SpiNNaker rather than changing the frozen image scaffold first.
