# R6B.3 pre-run audit — MaskGIT-aligned training and decoding

Status: **CODE PREPARED / NOT RUN**.

## Scope

R6B.2 is retained only as historical provenance. Its old generation result is not a valid MaskGIT benchmark because it used random one-token-at-a-time permanent filling.

R6B.3 keeps the frozen R6B.1 SNN-VQ tokenizer and exact P3 local dynamical learning machinery, but now aligns **both** the masked-token training task and generation controller with MaskGIT semantics. No KMeans route is re-run.

## A. Training masking: one-to-one check against MaskGIT MVTM

MaskGIT paper rule: sample `r ~ Uniform[0,1)`, compute `gamma(r)`, uniformly mask `ceil(gamma(r) * N)` tokens, and predict every masked token from the same corrupted token sequence.

| MaskGIT training operation | R6B.3 implementation | Audit |
|---|---|---|
| sample ratio `r` uniformly | `rng.random()` | PASS by inspection |
| cosine schedule | `gamma(r)=cos(pi*r/2)` | PASS by inspection |
| mask count | `ceil(gamma(r)*N)` | PASS by inspection |
| choose masked locations uniformly without replacement | `rng.choice(..., replace=False)` | PASS by inspection |
| replace all selected locations by MASK before prediction | one `arr` is constructed once per mask context | PASS by inspection |
| predict/learn all masked positions | every `masked_position` contributes once | PASS by inspection |
| same corrupted context for all masked targets | P3 resident state is rebuilt once from that single `arr` | PASS by inspection |
| no progressive ground-truth reveal | no target token is written back into `arr` | PASS by inspection |
| no forward-weight change before all masked targets contribute | all receiver forward quantities are cached first; local contributions accumulate; one `apply_update()` follows the complete masked set | PASS by inspection |

The historical `core.train_full(...)` sequential-reveal routine is **not called** by R6B.3.

### MSDL-specific implementation boundary

MaskGIT uses backpropagation on the sum/mean of masked-token losses. R6B.3 does not imitate BP. It keeps the accepted P3 local rule: persistent E/M populations and local `M E^T` contributions. The forward context, forward weights and cached predictions are fixed across the entire masked target set. The persistent E/M populations then advance through those masked-target contributions in randomized order, and the accumulated local update is applied once after the complete set.

Therefore task semantics are MaskGIT-style; the physical/local learning mechanism remains intentionally different.

The startup WR scaffold has no direct counterpart in standard MaskGIT. Its **input contexts** are now also generated with the same MaskGIT random multi-mask rule and without progressive reveal. The historical MSDL scaffold optimizer still samples those conditional target rows as its own initialization procedure.

## B. Generation: one-to-one check against public Google MaskGIT inference

Reference files:
- `google-research/maskgit/maskgit/libml/parallel_decode.py`
- `google-research/maskgit/maskgit/libml/mask_schedule.py`

| MaskGIT operation | R6B.3 implementation | Audit |
|---|---|---|
| start from all MASK image tokens | `initial=np.full(cfg.N, core.MASK)` | PASS by inspection |
| predict every currently masked location from one context | one `xn,xraw`; every masked receiver queried before any proposal write | PASS by inspection |
| categorical sample at every masked site | `_sample_rows` | PASS by inspection |
| keep committed tokens unchanged | proposals only write current unknown positions | PASS by inspection |
| confidence = probability of sampled token | `selected_prob=p[row,sampled]` | PASS by inspection |
| cosine schedule | `cos(pi*ratio/2)` | PASS by inspection |
| public Google mask-length rule | floor from initial unknown count + clamp | PASS by inspection |
| annealed random-top-k | `log(prob)+temperature*gumbel` | PASS by inspection |
| re-mask lowest-confidence proposals | ascending score, first `mask_len` | PASS by inspection |
| iterative refinement | default `T=8` | PASS by inspection |
| return final full proposal | stored before the final re-mask state | PASS by inspection |

For N=49 and T=8, the controller asserts post-round MASK counts:

`[48, 45, 40, 34, 27, 18, 9, 1]`

The returned image is the fully sampled proposal of the final round, matching the public Google implementation's stored output convention.

## C. Hardware boundary

Within one generation round, the cloud reference loops over receiver queries in Python, but every receiver sees one immutable context. Thus the semantics are parallel while the reference execution is serial. On SpiNNaker the receiver queries can be dispatched concurrently. The 49-value cosine/ranking/re-mask controller remains ordinary control-core logic; it is not forced into an SNN.

During training, the same principle is used: one resident corrupted context serves all masked receiver targets. Forward quantities are cached before the local update boundary. The persistent E/M learning states are then advanced locally; this is the P3/MSDL implementation rather than a claim of BP equivalence.

## D. Guardrails

- Training-mask self-test verifies cosine/ceil/uniform-without-replacement masking.
- Workflow rejects any call to legacy `core.train_full`.
- Workflow rejects the old `rng.permutation(cfg.N)` sequential reveal/fill pattern in R6B.3.
- Generation controller self-test checks all-MASK start, all-current-MASK proposal, confidence re-mask and exact N=49/T=8 schedule.
- Per-image generation traces and audits are saved.
- Online training audit records mask-batch counts and confirms no in-batch ground-truth reveal.
- Workflow remains `workflow_dispatch` only.

## Boundary

This is a **source-level pre-run audit**, not an experimental result. No R6B.3 training or generation run has been launched.
