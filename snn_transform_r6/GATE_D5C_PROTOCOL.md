# Gate D5C — Local-descent / self-conditioning gradient-gap mapping

## Frozen scientific question

D5B closed terminal attribution and established a component-level split inside the O block: norm-matched exact local descent in `WO` is locally valid yet deployment-harmful, while exact-gradient `bo` has opposite terminal deployment valence. D5C does **not** modify the learner. It asks where a locally correct descent direction first becomes harmful under self-conditioned deployment.

The narrow question is:

> For `W2`, `WO`, and `bo`, does the exact descent direction of the masked-token objective have a different directional effect on teacher-conditioned versus self-conditioned Gate-A predictions, and is any additional harm created by discrete self-generated branch switching?

D5C is eval-only. No training, learning-rate change, E/M reset change, O-path repair, freezing, or new plasticity rule is introduced.

## Base terminal state

The base model is the successful D5B `F_native` terminal system from run `39570284981`. Its Gate-A outputs must reproduce the stored D5B `F_native` arrays with exact proposal IDs and finite metric error < `1e-7` before D5C is interpreted.

D5B L1 and the component-level interpretation are prerequisites, not re-estimated in D5C.

## Definition of the local descent directions

The symbol `d_i = -grad_i L_masked / ||grad_i L_masked||` must denote one fixed parameter-space direction before Gate-A sensitivity is measured. To avoid using the Gate-A evaluation images to define that direction, D5C uses a separate held-out gradient batch:

- take 256 validation-token images not present in the frozen 256-image Gate-A set;
- select them deterministically from the complement with a fixed seed;
- sample one standard cosine-schedule MaskGIT context per image with a second fixed seed;
- use the D5A exact event oracles and average the exact negative gradients over all masked targets.

For `i in {W2, WO, bo}`:

`G_i = mean_event[- d L_masked / d theta_i]`

`d_i = G_i / ||G_i||`.

`W2` and `WO` are support-matched to their frozen sparse masks. The gradient-image IDs, masks, target count, gradient norms and SHA256 signatures are saved.

The Gate-A images are never used to choose `d_i`.

## Perturbation scale

A unit perturbation is one median native F_EM context-update norm for the corresponding pathway, measured from the successful D5A F_EM trajectory:

`rho_i = median_context ||Delta theta_i^native||`.

The formal dimensionless scales are:

`alpha in {0.25, 0.5, 1.0}`.

For each pathway and scale, construct symmetric perturbations around the same D5B `F_native` terminal state:

`theta_i(plus)  = theta_i + alpha * rho_i * d_i`

`theta_i(minus) = theta_i - alpha * rho_i * d_i`.

The primary small-step frozen-branch sensitivity uses `alpha=0.25`. `alpha=0.5` is a robustness check. The primary full-CRN causal sensitivity uses `alpha=1.0`, corresponding to one median native context step. All three scales are reported to expose nonlinearity rather than permit scale cherry-picking.

For each pathway, the same held-out masked batch used to define `d_i` is re-evaluated at plus/minus perturbations. Plus must decrease its defining masked CE and minus must increase it at the primary small scale; otherwise the direction audit is considered invalid.

## Gate-A endpoints

D5C keeps the frozen D3E/D4/D5 Gate-A coordinate system: 256 images, three historical rounds, same schedule, same CRN field.

Primary deployment-gap endpoints at round 3 are:

`J_D = CE_non61_self - CE_non61_teacher`

`J_F = P61_on_non61_self - P61_on_non61_teacher`.

Lower is better for both self and teacher components.

For a symmetric perturbation at scale `alpha`, the response per unit native step is

`D_i X(alpha) = [X(+alpha) - X(-alpha)] / (2 alpha)`.

The self-conditioning gradient gaps are

`G_i^D = D_i CE_non61_self - D_i CE_non61_teacher`

`G_i^F = D_i P61_non61_self - D_i P61_non61_teacher`.

Positive `G` means the local-descent direction worsens self-conditioned behavior relative to teacher-conditioned behavior.

All responses are formed per image first and summarized by paired bootstrap (10,000 resamples).

## Layer 1 — frozen-branch sensitivity

First construct the exact baseline Gate-A self and teacher contexts under the unperturbed D5B F_native model. Baseline sampled proposals and committed positions are then frozen.

For every perturbed model, probabilities are recomputed on those same baseline contexts, but no sampled token is allowed to alter a later context. This isolates conditional predictor sensitivity from branch switching.

The baseline frozen-context reconstruction must reproduce stored D5B F_native proposals exactly.

## Layer 2 — full-CRN causal sensitivity

The same plus/minus perturbed models are then evaluated by the ordinary frozen Gate-A evaluator with the original CRN field. Perturbations may change sampled tokens and therefore later self-conditioned contexts.

For each perturbation, save:

- self and teacher component responses;
- `G_i^D` and `G_i^F`;
- self-proposal token-switch fraction versus baseline by round;
- fraction of images with any self-proposal switch.

Teacher contexts remain ground-truth committed as in the frozen Gate-A definition.

## Controls

Only three pathways are tested:

- `W2`: exact local descent has beneficial terminal deployment valence;
- `WO`: exact local descent has harmful terminal deployment valence;
- `bo`: exact local direction improves terminal utility relative to native MSDL.

No broad-layer scan is allowed in D5C.

## Preregistered interpretation

### C1 — Self-conditioning mismatch confirmed

At the primary frozen scale (`alpha=0.25`), `WO` must show:

1. teacher `CE_non61` directional response < 0 with bootstrap CI entirely below zero; and
2. `G_WO^D > 0` and/or `G_WO^F > 0` with CI entirely above zero;
3. the same qualitative positive gap must persist in full-CRN at `alpha=1.0`;
4. neither `W2` nor `bo` may satisfy the same complete frozen-branch pattern at the primary scale.

Interpretation: exact `WO` masked-token descent improves clean/teacher prediction while specifically increasing sensitivity to self-generated context.

### C3 — Discrete branch amplification

If `WO` has no significant positive frozen-branch gap at `alpha=0.25` and `0.5`, but full-CRN at `alpha=1.0` has a significant positive `G_WO^D` and/or `G_WO^F` together with nonzero self-proposal switching, classify C3.

Interpretation: the major harmful conversion occurs through discrete sampled-token feedback rather than conditional predictor sensitivity on a fixed branch.

### C2 — Mismatch not localized to the measured self-conditioning gap

If the WO direction audit is valid but neither C1 nor C3 is satisfied, while the prerequisite D5B L1 result remains valid, classify C2.

Interpretation: D5C does not localize the known WO deployment mismatch to the measured round-3 self-vs-teacher gradient gap. Longer-horizon token-distribution or controller interactions remain candidates.

Patterns that fail base calibration or direction validity are technical/inconclusive, not scientific C1-C3 outcomes.

## Interpretation boundary

D5C tests a terminal, pathway-specific directional mechanism around the D5B F_native state. It does not claim that the masked-token objective is globally wrong, nor that gradients are generally unsuitable for generation. A C1 or C3 result would identify the specific conversion route by which a locally correct `WO` descent direction acquires negative autonomous-rollout utility in this model and deployment protocol.

The plasticity spectrum is therefore treated as separating at least `A_i^local` from `U_i^deploy`, with D5C decomposing deployment utility into teacher-conditioned, self-conditioned, and discrete-feedback contributions.