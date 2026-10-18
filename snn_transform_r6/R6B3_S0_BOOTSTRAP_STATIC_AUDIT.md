# R6B.3 S0 — static bootstrap audit

Status: **PASS BY SOURCE INSPECTION / NO TRAINING RUN**.

## Question

Can exact P3/MSDL leave `WR=0` without the startup readout scaffold, or is there an algebraic deadlock that forces `WR`, `BR`, and the internal plasticity pathways to remain identically zero?

## Source-level result

There is **no algebraic deadlock** in the current equations.

### 1. Initialization

`init_model(...)` initializes:

- `WR = 0`
- `BR = 0`
- `BO = 0`
- `B2 = 0`

while `WQ/WK/WV/WO/W1/W2` are nonzero random forward weights.

Therefore the initial categorical readout is exactly

`p = softmax(WR @ rf) = softmax(0) = 1/V`.

For `V=64`, the initial uniform NLL is `log(64)`.

### 2. The direct WR pathway is not gated by BR

For every masked target, P3 computes

`ER = E_dynamics(rf)`

and

`MR = M_dynamics(one_hot(target), p)`

then accumulates

`ar += outer(MR, ER)`.

This update does not contain `BR`, `BO`, or `B2`.

At `WR=0`, `p=1/V`. Therefore the M-drive is nonzero because

`one_hot(target) - 1/V != 0`.

The E-drive is also structurally nonzero: `rf` contains the normalized forward feature and an explicit final bias coordinate `rf[-1]=0.5`. Thus the readout eligibility pathway does not require a pre-existing trained readout.

Consequently `Delta WR ~ MR ER^T` can be nonzero at the first native MaskGIT mask batch.

### 3. First-batch BR remains zero, by construction

The MaskGIT batch wrapper runs reciprocal mirror microsteps **before** accumulating the masked-target local update.

Inside the BR mirror rule:

`dlogit = WR @ df`.

During the first no-scaffold batch, `WR=0` at this pre-batch mirror stage, so `dlogit=0`. Starting from the mirror equilibrium state, the BR eligibility drive is therefore zero and `BR` remains zero during this first reciprocal stage.

This is expected and is not a deadlock: the target batch can still update `WR` because the direct WR rule is independent of BR.

### 4. BO and B2 have a different startup order

Unlike BR, `BO` and `B2` do not depend on WR:

- BO mirror target uses `dz = WO @ da`;
- B2 mirror target uses `dy = W2 @ dh`.

Because `WO` and `W2` are already nonzero at initialization, BO and B2 are algebraically able to become nonzero during the first reciprocal mirror stage even while `WR=BR=0`.

Therefore the precise static causal order is not

`WR -> BR -> BO/B2 -> internal`.

It is instead expected to be approximately:

`first pre-batch mirror: BO/B2 can move; BR stays zero`

`first target batch: WR can move; internal forward weights remain gated because BR=0`

`second pre-batch mirror: new WR allows BR to move`

`second target batch onward: BR@err can drive internal P3 plasticity`.

### 5. Internal forward plasticity is gated by BR

For a target mismatch

`err = one_hot(target) - p`,

the internal mismatch routing starts with

`dy = BR @ err`.

At initial `BR=0`, `dy=0`. Therefore:

- W2 and WO mismatch drives are initially zero;
- `dh = B2 @ dy = 0`, so W1 mismatch drive is zero;
- `da = BO @ dy = 0`, so WV/WK/WQ mismatch drives are zero.

Thus the first native batch should principally bootstrap WR rather than the complete internal network.

Once BR becomes nonzero on a later pre-batch reciprocal stage, `dy` can become nonzero and all downstream local pathways become algebraically available.

### 6. The label-free MaskGIT input does not create a zero-input deadlock

The current input interface uses

`x_i = token_embedding(z_i) + position_embedding(i)`

with no class embedding. The MASK token itself is the zero vector in the centered VQ geometry, but masked positions still carry their deterministic 2-D positional embedding. The context is normalized before entering P3.

Even in an all-MASK context, the spatial representation is therefore not identically zero. Independently, the P3 readout feature contains the fixed bias coordinate `rf[-1]=0.5`, which guarantees a nonzero direct readout eligibility input.

## S0 verdict

**PASS:** there is no source-level algebraic requirement for the startup WR scaffold.

The equations support autonomous bootstrapping through the direct readout pathway.

The strongest static prediction for a no-scaffold smoke test is:

1. after the first complete MaskGIT mask batch, `||WR||` should become nonzero;
2. `BR` is allowed to remain zero through that first batch because reciprocal BR learning occurs before the WR update;
3. on the next pre-batch reciprocal stage, nonzero WR should permit BR growth;
4. internal WQ/WK/WV/WO/W1/W2 movement should begin only after BR provides nonzero `dy`;
5. BO/B2 may become nonzero before BR and therefore must not be used as evidence that target mismatch has already reached the internal forward weights.

## Decision

Proceed to S1 only as a **minimal bootstrap smoke test**, not a training-quality experiment. S1 should stop immediately if the observed activation order contradicts the source-derived sequence above.
