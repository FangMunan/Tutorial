# Gate D — Native Plasticity Organization Audit

## Status

This branch is a diagnostic continuation of `r6b4-full-forward-plasticity`.
It does **not** change the accepted MSDL learner, learning rates, sparse masks,
reciprocal microsteps, persistent E/M dynamics, target ordering, or update
boundary.

The central question is:

> Does the current R6B4 failure arise because native W2 credit points in the
> wrong direction, because persistent plastic dynamics are temporally unstable,
> or because individually sensible local updates co-adapt destructively?

## Scientific order

The analysis is deliberately ordered as:

1. **D0 — instrumentation parity**
2. **D1 — W2 native credit alignment**
3. **D2 — net/path mobility and update-mode spectrum**
4. **Evidence-dependent branch**
   - D1b: source of credit misalignment, if alignment fails;
   - D3: timescale/persistence, if alignment is good but trajectories are tortuous;
   - D4: coupling/epistasis, if local learning is stable but co-adaptation remains harmful.

Hessian/GGN and spectral-submanifold analysis are intentionally deferred.

---

## D0 — Instrumentation parity

The instrumented system is cloned from the same exact R6B4 `aux_lr_scale=1.0`
start state as an uninstrumented reference. Both consume the same saved epoch-3
schedule and cloned training RNG state.

Required checkpoints:

- 256 contexts
- 512 contexts
- 1024 contexts

At every checkpoint the native arrays, scalar state, internal RNG state and
training RNG state must be bitwise identical.

**Hard gate:** `D0 != PASS => STOP`.

---

## D1 — W2 native credit audit

### Native contribution

For event `j`:

```text
u_j = (M2_j E2_j^T) ⊙ mask2
```

The learner is sparse, so the mathematical oracle is restricted to the same
support:

```text
g_j = (-dL_j/dW2) ⊙ mask2
```

Dense off-support gradient components are never included in the alignment.

### Event-level diagnostics

For each event the audit stores:

- cosine(`u_j`, `g_j`)
- signed descent `u_j · g_j`
- efficiency `(u_j · g_j)/(||u_j||^2 + eps)`
- `A_M = cosine(M2_j, -dL/dy)`
- `A_E = cosine(E2_j, h_j)`
- target/prediction class, `p_target`, `p61`
- `false61 = (GT != 61 and argmax(p) == 61)`
- native within-context processing rank

The event contribution is a **realized sequential increment**, not an isolated
SGD sample: persistent E/M states contain previous-event history.

### Context-level primary metric

The actual W2 update occurs once per masked context. Therefore the primary
alignment is context-level:

```text
U_c = actual W2_after - W2_before
G_c = mean_j [(-dL_j/dW2) ⊙ mask2]
A_c = cosine(U_c, G_c)
```

The audit also verifies the native identity

```text
W2_after - W2_before == lr2 * mean_j[(M2_j E2_j^T) ⊙ mask2]
```

up to floating-point roundoff.

### Context conditions

The same audit is run under:

- **T0 / GT**: visible non-masked tokens are ground truth;
- **T1**: a fixed fraction of visible tokens are sampled from the current model;
- **T2**: T1 exposure positions and exact correct/error locations are retained,
  while wrong-token identities are strictly deranged and their global marginal
  is preserved.

T2 therefore tests wrong-token identity association; it does **not** remove the
T1 error-location/difficulty structure.

---

## D2 — Plasticity trajectory spectrum

For every native context update and every main forward group
`WQ, WK, WV, WO, W1, W2, WR`, record:

1. exact step norm `||Delta W_i(t)||_F`;
2. deterministic linear CountSketch of `Delta W_i(t)`.

From exact norms:

```text
mu_net  = ||W_i(T)-W_i(0)|| / ||W_i(0)||
mu_path = sum_t ||Delta W_i(t)|| / ||W_i(0)||
T_i     = mu_path / (mu_net + eps)
```

`T_i` is trajectory tortuosity. Terminal displacement must not be interpreted as
plastic activity.

Offline SVD is performed in two representations:

- **raw magnitude**: preserves real plastic movement;
- **equal-group normalized**: asks whether groups share latent temporal modes
  independently of absolute update magnitude.

Low SVD rank is only evidence for observed low-dimensional update trajectories.
It is **not** evidence for an invariant slow/spectral manifold.

---

## Evidence-dependent branches

### Credit failure

If W2 context alignment becomes negative, especially in T1/T2 or false-61
events, proceed to D1b and localize whether the mismatch side (`BR -> dy -> M2`)
or eligibility side (`h -> E2`) fails.

### Temporal-regulation candidate

If alignment is positive but `mu_path / mu_net` is large and the update modes
oscillate, prioritize E/M persistence, eligibility lifetime and metaplastic
timescales.

### Co-adaptation candidate

If local alignment is positive and trajectories are not excessively tortuous,
run selected pair interventions before broad sweeps:

- W2 x W1
- W2 x WO
- W2 x WQ
- W2 x WR
- Q x K
- Q x V
- K x V

Use normalized non-additivity as well as the raw interaction.

### Slow-manifold / SSM candidate

Only consider SSM after all three are demonstrated:

1. reproducible spectral gap;
2. slow modes reproducible across seeds/contexts;
3. transverse perturbations return toward the candidate low-dimensional set.

---

## Interpretation boundary

This gate uses gradient information only as an offline mathematical microscope.
No gradient, autograd result, adjoint, Hessian or Gauss-Newton quantity enters the
candidate MSDL learner.

A result that W2 should be slower/faster/gated would first apply only to the
current architecture, MSDL implementation and task distribution. Biological
interpretation is a separate hypothesis layer.
