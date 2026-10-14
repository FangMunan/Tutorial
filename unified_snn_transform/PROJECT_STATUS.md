# SNN-Transform unified project status — U4G

Canonical development line: `snn-transform-u4-sparse-tail-20261014`

## Frozen / accepted results

### U0/U1 — lightweight linear-attention block
PASS. Positive-RF normalized linear attention with persistent H/g is the reference computation. Izh feature replacement is treated as a neuronal-dynamics interface, not as a strict spike-only claim.

### U2 — event-cached Q/K/V
PASS as an algebraic/event-execution gate. On the lightweight trajectory, token projection evaluations fall from 384 to 144 (62.5% reduction) with zero numerical error in the cache test. Integration caveat: the current main `LinearAttentionCore.step` still evaluates Q globally; the Q-cache proof remains to be folded into the canonical implementation.

### U3.1 — trained-distribution Izh interface
PASS on the dense-tail trained retrieval block. Held-out Q+K replacement reached ~0.446% prediction relative-L2 with unchanged 100% task sign accuracy.

### Mamba-like selective retention
PARKED, not rejected. The mechanism learned different retention for reset/non-reset inputs, but U3B did not establish a functional task advantage because the gate task itself remained at chance. Revisit only when a long-context/context-switch workload actually requires selective forgetting.

## U4 — sparse Output Projection + FFN

### U4A — frozen dense random teacher screen
FAIL, as expected for the wrong equivalence question. A sparse network is not expected to reproduce an arbitrary frozen dense random mapping exactly. This result is retained only as a negative control.

### U4B — task-trained sparse architecture
PASS.

Strongest tested point:
- output projection connectivity density = 25%
- FFN input connectivity density = 25%
- FFN output connectivity density = 25%
- task sign accuracy = 100%
- shuffled key/value binding control sign accuracy ~54%
- static tail synapses = 1280 vs 5120 dense (25%)
- hidden activity ~64%
- effective tail SynOp proxy ~21–22% of dense (~78% reduction)

Scientific conclusion: the lightweight retrieval function can be learned by a structurally sparse Output+FFN realization. This establishes U4 sparse-tail feasibility; it is not a claim that an arbitrary trained dense tail can be losslessly pruned.

### U4C/U4D — localization of the composition error
The first sparse-tail + shared-Izh composition kept 100% task accuracy but missed the strict numerical gate (~2.87% prediction relative-L2). U4D localized the miss to the K-side Izh / normalized H-g readout; the sparse Output/FFN tail did not amplify the error.

### U4E — wider/clamp-aware shared scalar calibration
Improved the interface but did not close the gate:
- Q-only ~0.87% rel-L2 — PASS
- K-only ~1.87% rel-L2 — FAIL
- Q+K ~2.40% rel-L2 — FAIL

Conclusion: a single shared Q decoder and a single shared K decoder are too restrictive for this sparse-trained head distribution.

### U4F — gradient attention-aware decoder calibration
NEGATIVE CONTROL / DO NOT USE. The unconstrained gradient calibration was numerically unstable and diverged badly. It is retained to document that downstream fitting of the decoder is not automatically safe.

### U4G — stable per-head operating-point calibration
PASS and now frozen as the neuronal feature interface for this lightweight block.

Each attention head keeps the same Izh RS -> membrane trajectory -> fixed linear decoder interface, but may use its own fixed bias/gain/window and decoder. No task labels, task-model weights, attention weights, or sparse-tail connectivity are changed during interface selection.

Selected operating points:
- Q head 0: bias 5.0, gain 0.75, window 24 ms
- Q head 1: bias 4.0, gain 0.75, window 20 ms
- K head 0: bias 3.0, gain 0.25, window 24 ms
- K head 1: bias 2.0, gain 0.25, window 20 ms

Held-out integrated result:
- prediction rel-L2 = 0.009150 (~0.915%)
- prediction cosine = 0.9999696
- reference task sign accuracy = 100%
- candidate task sign accuracy = 100%
- H rel-L2 ~0.482%
- g rel-L2 ~0.427%
- sparse-tail effective fraction remains <=30% gate

All preregistered U4G gates PASS. The strict <=1% feature-interface integration issue is therefore closed for the lightweight block.

## Current architecture status

```text
Input
  -> event-cached Q/K/V projections                         [U2 proof PASS]
  -> per-head calibrated Izh Q/K feature dynamics          [U4G PASS]
  -> persistent H/g normalized linear attention            [PASS]
  -> 25% structurally sparse output projection             [U4B PASS]
  -> 25% structurally sparse FFN input connections         [U4B PASS]
  -> thresholded hidden population (~64% active)
  -> 25% structurally sparse FFN output connections        [U4B PASS]
  -> residual / normalization
```

## Next gate

Proceed to a canonical full-block freeze/integration gate. The purpose is no longer to redesign individual operators, but to package the accepted operators behind one reusable `SNNTransformBlock` API and verify, in one run:

1. full-block numerical/task parity on held-out data;
2. event-cached Q/K/V actually used in the canonical implementation, including Q;
3. sparse-tail connectivity and hidden activity accounting;
4. persistent-state API and deterministic state reset/carry behavior;
5. one compact artifact/checkpoint that can be reused by downstream tasks.

Scaling and SpiNNaker timing remain later stages; no hardware-speed claim is made from cloud SynOp proxies.
