# R6B.3 scaffold-removal plan

Status: **S0 STATIC AUDIT PASSED / NO TRAINING RUN**.

## Question

Can exact P3/MSDL bootstrap from its native local dynamics with `WR=0` and without the startup readout scaffold?

The experiment must not introduce class labels, backpropagation, class balancing, or a replacement pretraining mechanism.

## S0 result

Source inspection finds **no algebraic deadlock**. The direct readout pathway can update `WR` from `WR=0` because its local update does not depend on `BR`, while the masked-token mismatch is nonzero under the initial uniform prediction and `rf` contains a guaranteed nonzero bias coordinate.

One important refinement was found relative to the original plan: `BO` and `B2` reciprocal mirrors can move before `BR`, because their mirror-learning rules depend on the already-nonzero forward `WO` and `W2`, whereas `BR` depends on `WR` through `dlogit = WR @ df`.

The precise expected startup sequence is therefore:

1. first pre-batch reciprocal stage: `BO/B2` may move, `BR` remains zero because `WR=0`;
2. first masked-target batch: `WR` can move, while internal forward mismatch remains blocked by `BR=0`;
3. second pre-batch reciprocal stage: the now-nonzero `WR` can drive `BR` learning;
4. second masked-target batch onward: `dy = BR @ err` can activate internal P3 plasticity.

Full source-level reasoning is frozen in `R6B3_S0_BOOTSTRAP_STATIC_AUDIT.md`.

## Why bootstrap is mathematically possible

At initialization, `WR=0`, therefore the categorical readout is uniform:

`p = softmax(0) = 1/V`.

However the direct readout MSDL pathway is not silent. For a masked target, the readout eligibility `ER` is driven by the nonzero forward feature `rf`, while the readout mismatch `MR` is driven by `one_hot(target) - p`. Thus

`Delta WR ~ MR ER^T`

can be nonzero even when `WR=0`.

The reciprocal output mirror `BR` is initially zero and therefore the internal mismatch drive `dy=BR@err` is initially zero. This means the first native update is expected to bootstrap `WR` before the deeper P3 weights can receive useful mismatch drive.

Once `WR != 0`, the existing reciprocal mirror microdynamics use `dlogit = WR @ df`, so `BR` can become nonzero. On subsequent mask batches:

`err -> BR -> dy -> internal MSDL`

can activate WQ/WK/WV/WO/W1/W2. `BO/B2` may already have become nonzero from their independent reciprocal mirror dynamics.

Therefore the scaffold is not obviously required by the equations; it may only accelerate the `WR -> BR -> internal plasticity` startup transient.

## Experimental sequence

### S0 — static bootstrap audit — COMPLETE

No training was run. Verified by source inspection that:

1. `WR=0`, `BR=0` at initialization.
2. `rf` supplies a nonzero readout eligibility drive, including `rf[-1]=0.5`.
3. `MR(one_hot, uniform_prediction)` is nonzero.
4. the direct `WR` accumulator does not depend on `BR`.
5. `BR` mirror learning becomes possible after `WR` becomes nonzero, but only on the following pre-batch reciprocal stage in the current batch ordering.
6. deeper forward MSDL is gated by `BR`, so its expected onset is later than `WR`.
7. `BO/B2` may move before `BR`; their movement alone is not evidence that target mismatch has reached internal forward plasticity.

Result: **PASS BY SOURCE INSPECTION**.

### S1 — minimal bootstrap smoke test

Use the final label-free MaskGIT masking/input protocol, but only a few complete mask batches. Do not evaluate generation quality. Log after every relevant phase/batch:

- `||WR||`
- `mean(|BR|)`, `mean(|BO|)`, `mean(|B2|)`
- relative movement of WQ/WK/WV/WO/W1/W2
- online NLL
- masked-target count

Expected causal order:

`BO/B2 may move during first pre-batch mirror`

`first target batch -> WR movement`

`second pre-batch mirror -> BR movement`

`second target batch onward -> internal-weight movement`.

If `WR` remains exactly zero after the first native update, stop; there is a true bootstrap bug. If `WR` moves but `BR` never moves on the next reciprocal stage, inspect the reciprocal BR mirror path. If `BR` moves but internal weights remain silent, inspect the downstream mismatch routing.

### S2 — controlled scaffold ablation

Only after S1 passes, compare two runs with identical tokenizer, structured token/position representation, MaskGIT mask draws, model seed, target-event budget and generation controller:

- A: current startup WR scaffold + online MSDL
- B: no scaffold; exact native initialization + online MSDL

Primary comparison:

- masked-token CE/accuracy at 25%, 50%, 90%, 100% masking
- online NLL trajectory versus masked-target events
- time/target-events required for WR, BR and internal weights to become active

Secondary comparison:

- unconditional generated-image quality/diversity
- post-hoc external 0/1/2 diagnostics only; labels never enter either learner

### S3 — decision rule

Remove the scaffold permanently if the no-scaffold branch:

1. self-starts without any external target/class conditioning beyond the normal masked-token target;
2. produces nonzero `WR`, then nonzero `BR`, then nonzero internal plasticity in the predicted order;
3. learns masked-token prediction rather than remaining near uniform `log(64)` NLL;
4. reaches comparable useful prediction/generation behavior at an acceptable extra startup cost.

If it fails, do **not** immediately restore a large scaffold. First identify the exact deadlocked local pathway. Any rescue should be the smallest local/homeostatic mechanism that removes that deadlock and must be tested separately.

## SpiNNaker criterion

The preferred final route is no scaffold because it removes a separate pretraining phase. If a bootstrap aid proves necessary, its board cost must be measured explicitly. A rescue that requires host-side pretraining or repeated host/board exchange is disfavored even if it improves cloud accuracy.
