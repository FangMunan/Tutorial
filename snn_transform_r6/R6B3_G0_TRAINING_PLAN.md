# R6B.3 G0 training plan

This file freezes the next training stage without changing the model architecture.

- Keep the current R6B.3 predictor exactly as-is.
- Keep the frozen R6B.1 SNN-VQ tokenizer.
- Keep the startup WR scaffold.
- Keep label-free MaskGIT training and scheduled-parallel decoding.
- Do not reintroduce P1-G Izh Q/K feature dynamics during this stage.

## Training unit

Replace target-event exposure as the primary progress variable with MaskGIT context epochs.

One context epoch means that every training image is visited once in a shuffled order, with one fresh random cosine-schedule mask sampled for that image. All masked targets from that single corrupted image share the same forward context and contribute before the forward-weight update boundary.

## G0 budget

G0 = 0.25 context epoch.

At the end of G0 evaluate fixed validation masks at 25%, 50%, 90%, and 100% mask ratios and generate unconditional samples with both 8 and 16 MaskGIT refinement rounds.

Record:

- CE/accuracy at 25%, 50%, 90%, 100% masks;
- non-dominant-token CE/accuracy/mass;
- WR norm;
- mean absolute BR, BO, B2;
- relative movement of WQ/WK/WV/WO/W1/W2;
- numerical finite/stability checks;
- generation token usage/perplexity/top-1 fraction;
- post-hoc classifier class histogram, entropy and confidence;
- scheduled-parallel trace audits.

## Continuation decision

If G0 remains numerically stable and validation CE trends are consistent with learning, continue the same dynamical trajectory to 1 context epoch. If 1 epoch is still improving, continue to 5 epochs before deciding whether longer training is needed.

## Checkpoint requirement

Formal continuation must restore the complete MaskGITP3System dynamical state, not only forward weights. The checkpoint must therefore include all forward and reciprocal weights, all persistent E/M membrane/adaptation/trace arrays, resident state/caches where relevant, training counters and RNG state required by the learner. Architecture and training semantics are not changed by this checkpoint layer.
