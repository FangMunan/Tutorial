# SNN-Transform Context/Parity Audit v1

## Scope

This is a **read-only diagnostic** of the completed Adaptive Timescale v1 checkpoint from GitHub Actions run `40115465892`, artifact `SNN_TRANSFORM_ADAPTIVE_TIMESCALE_V1` (artifact ID `12569296115`).

No retraining. No parameter update. No timescale feedback update. No MaskGIT rollout modification. No new learning rule.

The audit asks whether the current SNN-Transform forward computation itself implements and uses token-to-token context.

## Operator under audit

The intended attention operator is the existing positive-random-feature / kernelized operator:

\[
H_h=\sum_j \phi(k_{hj})v_{hj}^{\top},\qquad
 g_h=\sum_j\phi(k_{hj})
\]

and, for target token `i`,

\[
a_{hi}^{resident}=\frac{\phi(q_{hi})^{\top}H_h}
{\phi(q_{hi})^{\top}g_h}.
\]

The independent explicit reference is

\[
a_{hi}^{explicit}=\frac{\sum_j [\phi(q_{hi})^{\top}\phi(k_{hj})]v_{hj}}
{\sum_j \phi(q_{hi})^{\top}\phi(k_{hj})}.
\]

This is **not** a softmax-attention equality test. The claim being tested is exact implementation of the designed kernel attention plus functional query-dependent contextual integration.

## Diagnostic sample definition

Use the frozen Adaptive Timescale v1 checkpoint and the same frozen SNN-VQ tokenizer.

- Dataset: MNIST classes 0/1/2, test split.
- Foreground diagnostic target: token ID is not the dominant token used by the parent masked evaluation. This preserves the parent experiment's operational foreground definition.
- Default: 256 foreground target positions + 128 dominant/background target positions.
- A target token is always MASK in every context variant.
- The target image, target position, target truth token, and model parameters are held fixed within each paired comparison.

For each target, pre-sample one ordering of the other 48 positions. Correct context is revealed using nested prefixes of that same order at visible fractions:

`0, 0.10, 0.25, 0.50, 0.75, 0.90`.

The preregistered primary context level is **0.75 visible**.

## Gate A — controlled context perturbation

At the same target position compare:

1. `allmask`: all other positions MASK;
2. `correct`: visible positions use the true tokens from the same image;
3. `shuffled`: the same visible-token multiset is position-shuffled;
4. `wrong`: the same visible positions use tokens from an image of a different MNIST class.

Primary paired quantities at 75% visible:

\[
\Delta P_{true}^{correct-allmask},\quad
\Delta P_{true}^{correct-shuffled},\quad
\Delta P_{true}^{correct-wrong}.
\]

Report paired bootstrap 95% CIs. Also report CE, top-1 accuracy, Jensen-Shannon divergence from all-MASK, and relative change in the attention output vector `a`.

This directly tests whether `p(z_i | z_visible)` depends on the controlled visible context. It does not infer context use indirectly by comparing different random mask evaluations.

## Gate B — explicit vs resident parity

For all-MASK and primary 75%-visible correct contexts, independently recompute Q/K/V, `phi(q)`, `phi(k)`, H, g, attention numerator/denominator, attention weights, and `a` by scanning all 49 tokens.

Compare to the resident implementation after `rebuild()` / `_read()`.

Preregistered numerical tolerance:

\[
\max |\Delta| \le 10^{-10}.
\]

Parity failures are implementation failures and take precedence over all functional interpretation.

## Gate C — attention structure

At 75% correct context, report per head:

- normalized entropy;
- effective token count `exp(H(alpha))`;
- max attention coefficient;
- target self-attention coefficient;
- total mass on visible vs masked positions;
- mean per-token attention on visible vs masked positions.

This is descriptive. No claim is made that a particular entropy value is intrinsically correct.

## Gate D — downstream context contribution

For the same 75%-visible correct context, preserve the same target residual input and downstream network but ablate only the attention vector:

\[
a_i \leftarrow 0.
\]

Compare normal vs no-attention prediction using:

- `P(true)`;
- CE;
- top-1 accuracy;
- Jensen-Shannon divergence;
- \(\|W_O a_i\| / \|x_i+b_O\|\).

This distinguishes "attention is computed" from "the downstream predictor actually uses it".

## Formal classification

Classification precedence:

1. `F0_FORWARD_IMPLEMENTATION_PARITY_FAIL`
   - any preregistered resident-vs-explicit parity quantity exceeds `1e-10`.

2. `F1_CONTEXT_PATH_EFFECTIVELY_INACTIVE`
   - parity passes, but both the attention output and final target distribution are numerically unchanged by correct 75% context relative to all-MASK.

3. `F2_CONTEXT_COMPUTED_BUT_DOWNSTREAM_EFFECT_WEAK`
   - attention state changes with context, but ablating `a` produces essentially no final-distribution change.

4. `F4_CONTEXTUAL_FORWARD_FUNCTION_SUPPORTED`
   - parity passes, context changes the forward state/output, and for foreground targets the paired bootstrap 95% CI is entirely positive for all three primary `P(true)` contrasts:
     - correct - all-MASK;
     - correct - shuffled;
     - correct - wrong-image.

5. `F3_CONTEXT_ACTIVE_BUT_SEMANTIC_BENEFIT_NOT_ESTABLISHED`
   - any other valid result after parity passes and context is active.

The `F4` threshold is deliberately strong: the purpose is to clear the SNN-Transform forward context path before returning to learning/controller analysis.

## Audit integrity

The audit must verify:

- no forward trainable parameter changed;
- `tau_fast`, `tau_medium`, `tau_slow` unchanged;
- controller context count unchanged;
- no call to `learn_mask_batch_arr`, `apply_update`, or controller `observe` is made by the diagnostic.

## Outputs

- `CONTEXT_PARITY_AUDIT_SUMMARY.json`
- `CONTEXT_SENSITIVITY_ROWS.csv`
- `PARITY_ROWS.csv`
- `ATTENTION_STRUCTURE_ROWS.csv`
- `CONTEXT_CONTRIBUTION_ROWS.csv`
- `CONTEXT_PARITY_DIAGNOSTIC_IDS.json`
- `CONTEXT_SENSITIVITY_FOREGROUND.png`
- console log

## Decision after this audit

- `F0`: stop; repair forward implementation.
- `F1/F2`: stop MaskGIT/controller tuning; repair or redesign the context path/readout.
- `F3`: forward path works mechanically but has not learned semantically useful context; next diagnose learning/credit.
- `F4`: forward SNN-Transform context computation is provisionally cleared; next audit confidence calibration / MaskGIT commitment behavior.
