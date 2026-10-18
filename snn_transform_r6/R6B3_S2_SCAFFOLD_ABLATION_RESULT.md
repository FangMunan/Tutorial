# R6B.3 S2 controlled startup-scaffold ablation result

Status: **PRIMARY COMPARABILITY GATE FAILS**.

Run: `39153062019`  
Artifact: `SNN_TRANSFORM_R6B3_S2_SCAFFOLD_ABLATION` (ID `12145734062`)  
Artifact digest: `sha256:5e52533687c50c095cbc8d7a96906053dfe153013cfbbbb5af48e61c4a9f653d`  
Seed: `7311`

## Experimental control

Two branches were compared:

- A: label-free MaskGIT WR startup scaffold + online P3/MSDL.
- B: exact native start with `WR=0`, no startup scaffold + online P3/MSDL.

Both branches used the same frozen R6B.1 SNN-VQ tokenizer, the same structured VQ-derived token representation and deterministic 2-D sinusoidal position representation, the same initial internal P3 weights, the same model/system seed, and exactly the same precomputed MaskGIT training schedule.

Shared training schedule:

- requested target events: 30,000
- actual target events: 30,030
- mask batches: 936
- schedule SHA256: `8e9d8b929c0da9f12104774a16d1e74280fb9e8c51a398002a8806e1e83a476a`
- same masked-target-order seeds for both branches: yes

No class label entered either neural path. Labels were used only to select the historical MNIST 0/1/2 subset and in an external post-hoc classifier.

## Online learning trajectories

| target events | scaffold NLL | native no-scaffold NLL | scaffold `||WR||` | native `||WR||` | scaffold mean `|BR|` | native mean `|BR|` |
|---:|---:|---:|---:|---:|---:|---:|
| ~5k | 1.6600 | 4.1309 | 14.3123 | 0.0911 | 3.809e-2 | 7.575e-5 |
| ~10k | 1.6751 | 4.1032 | 14.3120 | 0.1813 | 5.263e-2 | 2.390e-4 |
| ~15k | 1.6747 | 4.0685 | 14.3119 | 0.2765 | 5.773e-2 | 4.351e-4 |
| ~20k | 1.6675 | 4.0194 | 14.3116 | 0.3976 | 5.952e-2 | 6.561e-4 |
| ~25k | 1.6718 | 3.9540 | 14.3115 | 0.5287 | 6.077e-2 | 9.439e-4 |
| ~30k | 1.6714 | 3.8798 | 14.3113 | 0.6700 | 6.065e-2 | 1.244e-3 |

The native branch is clearly learning: its NLL decreases monotonically away from the uniform `log(64)=4.158883...` baseline. However, its WR/BR bootstrap remains much slower than the scaffold branch at the 30k-event horizon.

At 30k events:

- scaffold `||WR|| = 14.3113`
- native `||WR|| = 0.6700`
- scaffold mean `|BR| = 0.06065`
- native mean `|BR| = 0.001244`

Thus the native BR pathway is still about 49x smaller in mean absolute magnitude, while its readout norm is about 21x smaller.

## Final masked-token evaluation

| mask ratio | scaffold CE | native CE | native-scaffold CE | scaffold acc | native acc | native-scaffold acc |
|---:|---:|---:|---:|---:|---:|---:|
| 25% | 1.5736 | 3.3757 | +1.8021 | 58.59% | 28.65% | -29.95 pp |
| 50% | 1.6923 | 3.3634 | +1.6711 | 57.55% | 28.65% | -28.91 pp |
| 90% | 1.5770 | 3.3426 | +1.7655 | 59.64% | 29.43% | -30.21 pp |
| 100% | 1.6027 | 3.3849 | +1.7822 | 58.33% | 26.82% | -31.51 pp |

At 50% masking, the native branch also had non-dominant-token accuracy `0.0`, versus `0.4215` for the scaffold branch. Therefore the no-scaffold branch has not reached comparable conditional-token discrimination within the matched 30k online budget.

Primary predeclared comparability gates:

- native CE50 <= scaffold CE50 + 0.10: **FAIL**
- native Acc50 >= scaffold Acc50 - 0.05: **FAIL**
- native CE50 < uniform `log(64)`: **PASS**

Overall primary comparability gate: **FAIL**.

## Generation diagnostic

Both branches passed the MaskGIT controller trace audit.

Scaffold final generation:

- post-hoc 0/1/2 counts: `[2,18,16]`
- post-hoc classifier confidence: `0.8860`
- active token codes: `27/64`
- token perplexity: `7.35`
- top token fraction: `49.0%`

Native final generation:

- post-hoc 0/1/2 counts: `[5,16,15]`
- post-hoc classifier confidence: `0.7948`
- active token codes: `64/64`
- token perplexity: `36.59`
- top token fraction: `24.3%`

The native branch is more token-diverse, but this is not evidence that its generated images are better. Its masked-token prediction is much poorer, and the post-hoc classifier confidence is lower. No class-condition match is defined because generation is unconditional.

## Interpretation

S1 established that the scaffold is not required to break a mathematical/dynamical deadlock: native P3/MSDL self-starts in the expected `WR -> BR -> internal` order.

S2 establishes a different point: **under the present learning rates and a 30k target-event budget, the scaffold is practically important because it bypasses a very slow WR -> BR startup transient.** The native branch learns, but its output reciprocal pathway remains much weaker and its final masked-token prediction is not comparable.

This does not prove that a scaffold is fundamentally necessary. The current evidence localizes the practical bottleneck to bootstrap timescale, especially the slow growth of `WR` and consequently `BR`.

## Next decision

Do not permanently delete the scaffold yet. Also do not treat the existing large scaffold as theoretically necessary.

The next causal test should determine whether native learning catches up simply with a longer event horizon or whether a minimal local bootstrap/homeostatic mechanism is needed. The priority is to preserve a fully local, label-free route and avoid replacing the scaffold with another hidden pretraining stage.
