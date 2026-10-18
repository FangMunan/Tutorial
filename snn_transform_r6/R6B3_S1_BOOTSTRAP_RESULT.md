# R6B.3 S1 no-scaffold bootstrap smoke result

Status: **PASS**.

Run: `39152333344`  
Artifact: `SNN_TRANSFORM_R6B3_S1_BOOTSTRAP_SMOKE` (ID `12145687419`)  
Seed: `7311`  
Scope: eight complete MaskGIT mask batches only; no generation and no quality claim.

## Experimental invariants

- No startup WR scaffold.
- `WR=0` and `BR=0` exactly at initialization.
- No class conditioning enters the neural path.
- Structured input representation only: frozen SNN-VQ codebook geometry + deterministic 2-D sinusoidal position encoding.
- MaskGIT random multi-mask batches.
- No generation is executed.

## Observed bootstrap order

| Batch | masked targets | `||WR||` | mean `|BR|` | max internal relative movement | interpretation |
|---:|---:|---:|---:|---:|---|
| 1 | 48 | 3.8256e-4 | 0 | 0 | direct WR self-starts; BR/internal still silent |
| 2 | 10 | 9.2213e-4 | 1.0774e-8 | 2.0501e-9 | BR starts after WR; internal movement begins |
| 3 | 7 | 1.7612e-3 | 1.5516e-8 | 8.7996e-9 | bootstrap continues |
| 4 | 17 | 2.6596e-3 | 3.5089e-8 | 3.1231e-8 | internal mismatch routing strengthens |
| 5 | 46 | 2.9877e-3 | 1.3651e-7 | 8.6479e-8 | continued growth |
| 6 | 30 | 3.3308e-3 | 2.2676e-7 | 2.0217e-7 | continued growth |
| 7 | 47 | 3.6726e-3 | 3.0855e-7 | 3.9127e-7 | continued growth |
| 8 | 45 | 3.9816e-3 | 3.9097e-7 | 5.8111e-7 | continued growth |

`BO` and `B2` moved already in batch 1, as predicted by S0, because their reciprocal mirrors depend on the already-nonzero `WO` and `W2` rather than `WR`.

## Gates

All preregistered S1 bootstrap gates passed:

- WR moves by batch 1: PASS.
- BR remains zero through batch 1: PASS.
- Internal forward weights remain unchanged through batch 1: PASS.
- BR moves only after WR has become nonzero: PASS.
- Internal forward-weight movement appears once BR is available: PASS.
- Overall causal bootstrap chain `WR -> BR -> internal plasticity`: PASS.

The observed first movement batches were:

`WR = 1`, `BR = 2`, `internal = 2`.

## NLL boundary

Online NLL changed only slightly over eight batches, from the uniform baseline `log(64)=4.158883...` at batch 1 to `4.157922` by batch 8. This is expected for this smoke test and is **not** evidence of useful task learning yet. S1 tests only whether the local learning machinery can self-start without the scaffold.

## Conclusion

The startup WR scaffold is **not required to break an algebraic/dynamical startup deadlock in this tested seed/configuration**. Exact P3/MSDL self-started natively from `WR=0` in the predicted causal order. This supports proceeding to the controlled S2 scaffold-vs-no-scaffold ablation, but does not yet justify permanently deleting the scaffold or claim comparable final learning/generation quality.
