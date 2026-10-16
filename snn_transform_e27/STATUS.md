# P4-E27 status

**State:** cloud MSDL training gate prepared; long run is triggered separately by `RUN_TRIGGER` or manual workflow dispatch.

Frozen candidate:

- B0-faithful 28x28 MNIST scaffold: 4x4 patches, 49 positions, K=64 visual tokens, MASK=64.
- d=128, four H/g dynamical blocks, four heads, FF=512.
- B0 cosine masking and 16-step cosine iterative decoding retained.
- class-conditioning removed.
- no candidate backprop / optimizer.
- local dual-timescale MSDL plus learned reciprocal credit populations.
- evaluator labels are used only to construct the 0/1 benchmark and for post-hoc evaluation; labels are not passed to the candidate.

Default GitHub Actions budget: 55 min of candidate training, with an 80 min workflow timeout so final evaluation and artifact upload have room to complete.

Primary gate: masked-token CE at 50/90/100% mask must improve from initialization and beat the empirical token-marginal baseline; then autonomous all-MASK generation is judged by the saved gallery, diversity, external 0/1 evaluator and nearest-real-image distance.
