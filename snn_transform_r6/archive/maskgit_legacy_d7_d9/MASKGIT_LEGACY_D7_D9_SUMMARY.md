# MaskGIT legacy D7–D9 scientific summary

This archive preserves the mechanistic evidence from the old MaskGIT/SNN_Transform Gate D7–D9 line. It is historical evidence, not the active development tree and not a final learning rule.

## A. Local objective ≠ autonomous deployment

The D7–D9 line reinforced a central deployment lesson:

`masked/local objective improvement` does not guarantee `autonomous iterative rollout improvement`.

This is directly relevant to the conventional MaskGIT training–sampling mismatch: masked-token CE/NLL is a local conditional objective, whereas autonomous iterative generation repeatedly changes the model's own context distribution.

This archive does not claim that the legacy gates alone establish a universal MaskGIT theorem; it preserves the empirical warning that local repair metrics are insufficient as deployment criteria.

## B. Single-parameter repair can be bypassed

`W2` had strong terminal attribution, but trajectory-level clamping did not close the failure because the plastic network could compensate through other pathways.

Frozen conclusion:

`single-parameter repair is not generally closed under co-adaptation`.

Any future training intervention must therefore be evaluated as a coupled dynamical intervention, not only as a terminal single-parameter correction.

## C. (b2,bO) is an important coupled subsystem, not a final learning rule

D7A.2 froze:

`A1_COMPACT_TWO_BLOCK_COMPENSATION`

with the pair `(b2,bO)`.

D9A.3 froze:

`M1_PAIR_DRIVEN_CONSOLIDATION`.

For the preregistered 128→256 consolidation transition:

`g_pair / g_full = 0.5765163477554687` (~57.65%).

The residual network still contributed same-sign growth, so the supported interpretation is pair-led consolidation with residual reinforcement. This is mechanistic evidence only:

`(b2,bO) importance ≠ validated final learning rule`.

## D. The terminal rank-1 mode is a failure diagnostic, not a validated online training rule

D8B established that the frozen terminal rank-1 y-space mode has bidirectional causal negative deployment utility.

However, the subsequent D9A/D9A.1 trajectory tests did not validate either

`q = v^T y`

as an absolute online homeostat, or

`[Delta q^pl]_+`

as a simple instantaneous admissibility signal for training.

Frozen conclusion:

`terminal failure coordinate ≠ validated online training coordinate`.

The rank-1 mode should therefore remain a diagnostic/failure coordinate unless a new prospective training experiment independently validates an online use.

## Transition to the new mainline

D7–D9 is archived mechanistic evidence. The active research question is now:

`How should a dynamical MaskGIT be trained coarse-to-fine while controlling the masked-training / autonomous-generation gap?`

The new active path is:

`clean conventional MaskGIT baseline`
→ `train/validation/rollout mismatch`
→ `coarse parameter training`
→ `coupled subsystem refinement`
→ `background-dependent fine tuning`
→ `autonomous rollout evaluation`.
