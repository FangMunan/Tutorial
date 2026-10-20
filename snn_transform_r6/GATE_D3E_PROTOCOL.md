# Gate D3E — Full-trajectory context-boundary credit-state intervention

D3E is a causal diagnostic intervention, **not** an accepted new MSDL learning rule.

## Question

If only cross-context inheritance of the W2-side `E2/M2` credit states is removed, while native within-context persistence is left completely unchanged, does the full 1024-context trajectory become less cancelling and does the frozen Gate-A self-conditioning phenotype improve?

## Arms

All arms start from the same exact R6B3 2.0-epoch checkpoint upgraded to R6B4 aux1.0 and consume the same image/context stream, masks, target-order RNG, mirror seeds and learning rates.

- `N`: native.
- `F_E`: before every new MaskGIT context set `(ve2,ue2,ze2)=(V0,U0,0)`, then run the native context unchanged.
- `F_M`: before every new context set `(vm2,um2,zm2)=(V0,U0,0)`, then run the native context unchanged.
- `F_EM`: reset both E2 and M2 to the canonical fresh initialization state at the context boundary only.

`(V0,U0,0)` is called the **canonical fresh initialization state**, not an unverified mathematical equilibrium.

`M2` contributes to both `W2` and `b2` plasticity in exact R6B4. Therefore `F_M/F_EM` are W2-side E2/M2 credit-state interventions, not pure-W2 interventions.

## Primary mechanism endpoints

Each arm independently recomputes its own exact instantaneous W2 oracle field and matched-norm oracle trajectory:

- `T_native`
- `T_oracle_matched`
- `T_excess = T_native/T_oracle_matched`
- context-level W2 alignment
- orthogonal path fraction `R_perp`
- lag-1 stale-credit signature `C_ug(1)=cos(u_t,g_{t-1})`

No arm is compared against the historical `T_oracle=29.62` as a fixed denominator.

## Primary phenotype endpoints

The terminal arms use the frozen Gate-A2 protocol: the same 256 validation rows, 3 rounds, R6B3 r3/t8 committed-position schedule, common-random-number categorical sampling, and paired teacher/self contexts. A regenerated historical R6B3 control must reproduce the frozen C1 Gate-A comparator to max absolute mean error `<1e-7` before D3E results are interpreted.

Primary round-3 effects are:

- `F61 = P(61|GT!=61)_self - P(61|GT!=61)_teacher`
- `D_non61 = CE_non61,self - CE_non61,teacher`
- `C61 = Context61_self - Context61_teacher`

Positive rescue is `Native effect - intervention effect`.

## Terminal mediation check

In the native terminal background evaluate frozen-state transplants of only `W2` from `F_EM`, only `b2`, and `W2+b2`. This tests direct terminal contribution only; it is not a complete mediation analysis because the F_EM trajectory may co-adapt other groups.

## Secondary endpoints

Ordinary fixed-mask CE and autonomous r3/t8 global/spatial/pair JS are recorded but are not primary causal endpoints.

## Pre-registered classification

- **E1:** mechanism improves and at least F61 or D_non61 has paired-bootstrap rescue CI entirely above zero -> cross-context inherited E/M state is a causal contributor to fragility; only then consider an endogenous boundary-sensitive metaplastic mechanism.
- **E2:** mechanism improves but primary phenotype does not -> cross-context carry causes plastic inefficiency but is not a major source of late fragility; move to D4.
- **E3:** full-trajectory mechanism does not materially improve -> D3C is a real local effect with limited system-level contribution; move to D4.

No learning-rate sweep, Bayesian optimization, or candidate metaplastic rule is introduced in D3E.
