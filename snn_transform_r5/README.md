# R5 — one-block learning gate, staged

R5 is deliberately split so that failures can be localized.

## R5A — joint FFN matrix recovery

Plastic parameters:
- final block `f1.weight` (W1)
- final block `f2.weight` (W2)

All other E29 H/g parameters stay fixed.

Controlled lesion:
- W1 -> 0
- W2 -> 0.5 * its E29-transferred value
- biases unchanged

W2 is not fully zeroed because simultaneous W1=W2=0 creates a dead multiplicative path: W1 perturbations cannot influence loss. R5A therefore uses a severe but identifiable lesion.

Learning:
- W1: perturb hidden preactivation before GELU; pair each masked token's loss difference with `xi_hidden x x_attn^T`.
- W2: perturb FFN output; pair each masked token's loss difference with `xi_out x h^T`.
- fast/slow synaptic learning states are retained as non-model state variables.
- no `.backward()`, no `torch.optim`, no learned feedback network.

Primary gate:
- measurable 50%-mask lesion (`delta CE > 0.05`)
- >50% functional CE recovery at 50% mask
- improvement over damaged model at 25%, 50%, and 90% mask
- only W1/W2 weights change

## R5B — full final H/g block

Only after R5A passes, extend plasticity to attention projections and normalization:
`WQ, WK, WV, WO, W1, W2, LN`.

R5B is not started by the R5A workflow.
