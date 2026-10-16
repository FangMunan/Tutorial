# P4-E29 — Zero-training B0 -> H/g forward-equivalence gate

This experiment isolates only the attention implementation.

1. Train/load the original class-conditioned B0 MaskGIT reference.
2. Freeze its learned parameters.
3. Copy every learned parameter into an H/g version, including token/position/class embeddings, q/k/v/o, FFN, LayerNorm, prediction head and biases.
4. Replace only exact softmax attention by resident H/g linear-attention dynamics.
5. Perform **zero training** of the H/g model.
6. Sweep RF width and compare layer outputs, logits, masked-token CE/KL, top-1 agreement, and generation.

Class conditioning is intentionally retained in E29. Removing it here would introduce a second change and invalidate the causal test. The final autonomous model remains class-free; that comes after forward-equivalence is established.

RF modes:
- `gaussian`: paper-faithful Eq.14 style standard-normal random projections.
- `p3norm`: previous E26/P3 normalized-row RF control.

Primary decision:
- If B0 generation and conditional predictions survive weight transfer at sufficient RF width, the H/g forward route passes.
- If not, repair the forward approximation before any further MSDL work.
