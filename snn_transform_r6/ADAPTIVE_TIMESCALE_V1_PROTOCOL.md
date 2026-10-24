# Adaptive Timescale v1 — first functional training test

## Question

Can the existing full-forward dynamical MaskGIT learner acquire autonomous MNIST-like image generation if its **existing local plastic updates are placed on adaptive heterogeneous timescales**, without changing the architecture or MSDL credit rule?

This is a functional-first test. The historical failed baseline is not rerun.

## Frozen components

Do not modify:

- frozen R6B1 SNN-VQ tokenizer / decoder;
- label-free MaskGIT cosine masking;
- immutable corrupted context within a masked-target batch;
- scheduled-parallel MaskGIT generation;
- P3/MSDL local E/M update directions;
- persistent E/M state equations;
- reciprocal `BR/BO/B2` learning dynamics;
- sparse masks / random feature map / forward architecture;
- existing base learning-rate ratios.

## Only new mechanism

For parameter group `g`:

\[
\Delta\Theta_g(t)=\frac{1}{\tau_g(t)}\Delta\Theta_g^{\mathrm{R6B4}}(t).
\]

The initial grouping is a prior from previous experiments, not a claim of optimality:

- **FAST**: `WR, WQ, WK, WV`, initial `tau=1`.
- **MEDIUM**: `WO, W1, W2`, initial `tau=5`.
- **SLOW**: `bq,bk,bv,bo,b1,b2,gamma2,beta2,tok,pos`, initial `tau=25`.

`BR/BO/B2` are not timescale-modulated in v1 because they are credit machinery, not the forward trainable groups under test.

## Causal feedback law

The current context uses `tau(t)`. Only after that context has completed is its already-observed masked-token NLL supplied to the controller. Therefore:

\[
L_t\rightarrow\tau_{t+1}\rightarrow\Delta\Theta_{t+1}.
\]

There is no future-loss gradient, BP, BPTT, autograd, or retroactive rewrite.

The v1 controller is deliberately minimal and experimental:

1. maintain an EMA of context NLL;
2. every fixed feedback interval compare current EMA with the EMA one trend window earlier;
3. clear improvement -> decrease the active group's `tau` (faster future plasticity);
4. clear worsening -> increase the active group's `tau` (slower future plasticity);
5. sustained plateau -> hand plasticity focus in the fixed order `FAST -> MEDIUM -> SLOW`;
6. clamp all `tau` to positive bounds.

The exact feedback constants are configuration parameters and are not claimed as a final learning law.

## Existing startup retained

The existing MaskGIT startup `WR` scaffold remains. Removing it would introduce a second scientific variable and is outside v1.

After the scaffold, the adaptive full-forward system trains from its normal initial forward state rather than from the historical failed two-epoch checkpoint.

## First-run endpoint

The first run is not a mechanism audit. It asks whether the system learns images.

Required outputs:

- training NLL trace;
- `tau_fast(t), tau_medium(t), tau_slow(t)`;
- Fast/Medium/Slow parameter movement;
- final masked validation metrics;
- **100 autonomous generated samples** by default;
- decoded generation grid;
- generated token/image NPZ;
- final checkpoint for continuation;
- self-test results.

No success classifier is preregistered beyond finite execution and valid MaskGIT traces. The generated images are inspected first; detailed causal analysis is performed only after the result exists.

## Mandatory self-tests

1. Adaptive controller behaves directionally on synthetic decreasing / plateau / increasing loss.
2. With `tau_fast=tau_medium=tau_slow=1` and adaptive feedback disabled, one R6B4 update matches the original full-forward update to tolerance `1e-12`.
3. Existing MaskGIT masking and decode-controller self-tests pass.

## Interpretation boundary

A successful image-generation run supports the **training strategy** as a candidate. It does not yet prove that the chosen group assignment or feedback law is optimal. A failed run triggers analysis of the observed loss/timescale/generation traces before adding further mechanisms.
