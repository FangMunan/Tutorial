# R6B.3 S2G — non-MaskGIT scaffold generality result

Status: **GENERIC BOOTSTRAP SUPPORT** (minimal single-seed diagnostic).

Run: `39154453827`  
Artifact: `SNN_TRANSFORM_R6B3_S2G_NEXTTOKEN_GENERALITY` (ID `12146801628`)  
Artifact digest: `sha256:08706661940f01f070f437a443574d2717222ed1fc3e1896ddbdb6cddfd862e5`  
Seed: `7311`

## Question

Does the WR startup scaffold accelerate P3/MSDL only because of MaskGIT, or does the same acceleration appear in a non-MaskGIT task?

## Task

The frozen R6B.1 SNN-VQ produces the same 49-token image sequences as the main route, but the training objective is ordinary causal next-token prediction:

`z[:t] -> z[t]`, with target positions sampled uniformly from 1..48.

The target and all future tokens are represented as MASK in the fixed 49-position P3 layout. There is no MaskGIT multi-mask objective, no cosine mask schedule, no simultaneous masked reconstruction, no confidence remasking, no iterative generation, and no class conditioning.

The two online branches share the exact same 10,000-event `(image, target-position)` schedule (SHA256 `3152b4e4cde36e86d7a25110c369239814669fad2dbdbb771339677c9731933e`) and the same initial internal P3 weights:

- A: next-token WR scaffold, then full P3/MSDL.
- B: exact native `WR=0` start, then full P3/MSDL.

The scaffold itself uses only causal next-token contexts and trains only WR with the existing local MSDL readout rule. No class label enters the neural path.

## Predeclared decision rule

At the diagnostic horizon, support a generic P3/MSDL bootstrap effect if both are true:

1. `NLL_native - NLL_scaffold > 0.5`.
2. `mean|BR|_scaffold / mean|BR|_native > 5`.

## Learning trajectory

| events | scaffold NLL | native NLL | NLL gap | scaffold mean |BR| | native mean |BR| | BR ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 2k | 1.6708 | 3.6376 | 1.9668 | 0.01893 | 0.000510 | 37.1x |
| 4k | 1.6604 | 3.2579 | 1.5974 | 0.03119 | 0.001711 | 18.2x |
| 6k | 1.6829 | 3.0907 | 1.4078 | 0.03957 | 0.003073 | 12.9x |
| 8k | 1.6813 | 2.9937 | 1.3124 | 0.04454 | 0.004300 | 10.4x |
| 10k | 1.6801 | 2.9402 | 1.2602 | 0.04853 | 0.005608 | 8.65x |

WR at 10k was `13.3435` with scaffold versus `2.1620` native (6.17x ratio).

The predeclared NLL and BR gates are therefore passed throughout the measured 2k–10k trajectory, not only at the final point.

## Important stability caveat

The native branch developed a late dynamical instability between 8k and 10k events: mean `|B2|` increased from `0.00889` at 8k to approximately `1.35e8` at 10k, and the maximum internal relative movement became approximately `1.96e9`. The scaffold branch remained bounded (`mean|B2| = 0.00748` at 10k).

Therefore the 10k native final state must **not** be treated as a clean stable endpoint for detailed model-quality interpretation. However, this does not undermine the generic-bootstrap decision because the predeclared gates already pass strongly at 2k, 4k, 6k and 8k, before the late instability. At 8k, for example, the NLL advantage is `1.3124` (>0.5) and the BR ratio is `10.4x` (>5).

## Evaluation diagnostic

The scaffold final next-token evaluation over 2,048 held-out pairs was CE `1.6128`, accuracy `57.81%`. The native 10k final evaluation was CE `2.7471`, accuracy `27.73%`, but this native endpoint is contaminated by the late B2 instability and should be treated only as a diagnostic, not as a clean performance comparison.

## Conclusion / project decision

The scaffold acceleration is **not specific to MaskGIT** in this minimal control. The same large WR/BR startup advantage appears in ordinary causal next-token prediction with all MaskGIT-specific training and decoding machinery removed. The evidence supports interpreting the scaffold mainly as a generic P3/MSDL readout/reciprocal bootstrap accelerator.

This is intentionally only a single-seed task-level control, not a theorem of universality. For the current project-management question, however, it is sufficient to stop expanding the scaffold investigation and return to the main objective: make label-free MaskGIT + P3/MSDL generation work, then test continual learning. The late no-scaffold instability is recorded as a future stability issue rather than a blocker for the MaskGIT main line.
