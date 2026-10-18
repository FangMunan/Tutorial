# R6B.3 pre-run audit — corrected MaskGIT scheduled-parallel decoding

Status: **CODE PREPARED / NOT RUN**.

## Scope

R6B.2 is retained as historical provenance. Its masked-prediction results remain usable, but its generation result is invalid as a MaskGIT benchmark because generation used random one-token-at-a-time permanent filling.

R6B.3 changes the generation controller only. The frozen R6B.1 SNN-VQ tokenizer, exact P3 SNN-Transform/MSDL core, plain mismatch weighting, class conditioning, internal LR scale 0.5 and 30,000-event training configuration are otherwise retained.

## Static one-to-one check against public Google MaskGIT inference

Reference files:
- `google-research/maskgit/maskgit/libml/parallel_decode.py`
- `google-research/maskgit/maskgit/libml/mask_schedule.py`

| MaskGIT operation | R6B.3 implementation | Audit |
|---|---|---|
| start from all MASK image tokens | `initial=np.full(cfg.N, core.MASK)` | PASS by inspection |
| predict every currently masked location from one context | `predict_masked(...)` builds one `xn,xraw`; every masked receiver is queried before any proposal is written | PASS by inspection |
| categorical sample at every masked site | `_sample_rows` | PASS by inspection |
| keep already committed tokens unchanged | proposals only write `unknown`; assertions protect known positions | PASS by inspection |
| confidence = probability of sampled token | `selected_prob=p[row,sampled]` | PASS by inspection |
| cosine schedule | `cos(pi*ratio/2)` | PASS by inspection |
| mask length from initial unknown count | `floor(initial_unknown*mask_ratio)` then official clamp | PASS by inspection |
| annealed random-top-k | `log(prob)+temperature*gumbel`, temperature=`choice_temperature*(1-ratio)` | PASS by inspection |
| re-mask lowest-confidence proposals | ascending confidence score, first `mask_len` positions | PASS by inspection |
| fixed iterative refinement | default `T=8` | PASS by inspection |
| final output is final full proposal before last re-mask state | `final_proposal` is recorded before re-masking and returned | PASS by inspection |

For N=49 and T=8, the controller asserts the official-code-style post-round MASK counts:

`[48, 45, 40, 34, 27, 18, 9, 1]`

The final returned proposal contains 49 sampled tokens; the remaining one-MASK controller state after round 8 is not returned, matching the official implementation's use of the stored pre-remask sequence.

## P3 predictor audit

Within each MaskGIT round, R6B.3 calls P3 `forward_only` for every currently masked receiver. The same immutable `xn,xraw` context is used for all receiver queries. `forward_only` only rebuilds/reads resident forward state and computes the tail/readout; it does not invoke the MSDL `step` learning path. No sampled proposal is written into the context until all masked receivers have been evaluated.

Therefore the current CPU implementation is **semantically parallel** but computationally serial over receiver queries. This is intentional as a cloud reference. A SpiNNaker implementation should dispatch the receiver populations concurrently within each of the 8 MaskGIT rounds; the small cosine/ranking/re-mask controller can remain conventional control-core code.

## Guardrails added

- Runtime controller self-test before expensive training.
- Per-image, per-round decode trace.
- Assertions that every MASK gets a proposal before filtering.
- Assertions that committed tokens never change.
- Assertions that post-round MASK count equals the scheduled count.
- N=49/T=8 exact schedule assertion.
- GitHub workflow static guard rejects the old `rng.permutation(cfg.N)` sequential-generation pattern in R6B.3.
- Workflow remains `workflow_dispatch` only.

## Boundary

This audit confirms source-level alignment of the **generation/inference loop** with public MaskGIT semantics. It does not claim that the P3/MSDL learning rule is backpropagation-equivalent to MaskGIT training, and it does not establish generation quality until the experiment is explicitly run.
