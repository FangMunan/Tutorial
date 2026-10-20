# Gate D3E — Full-trajectory context-boundary credit-state intervention

D3E is a causal diagnostic intervention, **not** an accepted new MSDL learning rule.

## Question

If only cross-context inheritance of the W2-side `E2/M2` credit states is removed, while native within-context persistence is left completely unchanged, does the full 1024-context trajectory become less cancelling and does the frozen Gate-A self-conditioning phenotype improve?

## Arms — frozen definition

All arms start from the same exact R6B3 2.0-epoch checkpoint, are upgraded identically to R6B4 aux1.0, and consume the same image/context stream, masks, target-order RNG, mirror seeds and learning rates.

- `N`: native.
- `F_E`: before every new MaskGIT context set `(ve2,ue2,ze2)=(V0,U0,0)`, then run the native context unchanged.
- `F_M`: before every new context set `(vm2,um2,zm2)=(V0,U0,0)`, then run the native context unchanged.
- `F_EM`: reset both E2 and M2 to the canonical fresh initialization state at the context boundary only.

`(V0,U0,0)` is the **canonical fresh initialization state**, not an asserted mathematical equilibrium. The boundary reset contains no RNG call. Within-context persistence is unchanged. `M2` contributes to both `W2` and `b2` plasticity in exact R6B4, so `F_M/F_EM` are W2-side E2/M2 credit-state interventions, not pure-W2 interventions.

## Provenance and execution hierarchy

Historical evaluator identity, current learner identity, and instrumentation identity are deliberately separated. D3E intervention results may be interpreted only if all seven gates pass:

- **P0 — shared provenance exact:** canonical validation IDs, GT-token hash, CRN uniform/Gumbel fields and committed-position schedule identity match the frozen provenance.
- **P1 — start checkpoint exact:** the D3E start is the exact R6B3 2.0-epoch checkpoint, including the frozen checkpoint SHA-256.
- **P2 — historical R6B3 predictor parity:** the reconstructed historical R6B3 control reproduces the frozen r3/t8 reference probabilities under the historical committed-position schedule.
- **P3 — historical R6B3 Gate-A causal parity:** the five historical Gate-A causal means reproduce within `1e-7` under the canonical CRN field.
- **P4 — same-run instrumentation parity:** R6B4 aux1.0 native plain and instrumented trajectories start from identical upgraded state, consume cloned RNG streams, and are bitwise identical on the complete plain native field set at contexts 256, 512 and 1024.
- **P5 — four-arm exact pairing:** `N/F_E/F_M/F_EM` have identical per-context image index, masked set, target-order signature and mirror seed; boundary resets consume no RNG; final external RNG states are identical.
- **P6 — frozen D3E R6B4-native execution anchor:** current instrumented `N` reproduces the separately frozen R6B4-native Gate-A execution anchor. Round-wise self/teacher metrics and all five per-image round-3 effect arrays must agree within `1e-7`; self/teacher proposal IDs must match exactly; aggregate means and schedule identity must also pass.

The P6 object is called the **frozen D3E R6B4-native execution anchor**. It is not a “historical Gate-A R6B4 reference”: the historical effectiveness experiment did not save those R6B4 causal arrays. Its learner identity is first certified against the historical R6B4 aux1.0 effectiveness observables and exact paired training-stream signature, after which the validated historical Gate-A evaluator is used to freeze the causal execution anchor.

The original historical byte-identical R6B3 `+1024` pickle is not currently recovered. Its historical metadata hashes remain provenance metadata, but it is not an execution dependency. Cross-run scientific equivalence is established functionally; same-run instrumentation equivalence is tested bitwise.

The historical native W2 alignment value `0.7311738063456619` is retained only as a diagnostic reference and is **not** a cross-run hard gate.

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

The terminal arms use the frozen Gate-A protocol: the same 256 validation rows, 3 rounds, historical R6B3 r3/t8 committed-position schedule, canonical common-random-number categorical sampling, and paired teacher/self contexts.

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

No learning-rate sweep, Bayesian optimization, or candidate metaplastic rule is introduced in D3E. Even an E1 result does not itself establish an endogenous metaplastic mechanism; it establishes a causal contribution of inherited credit state and motivates a later endogenous mechanism test.
