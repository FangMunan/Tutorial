# R6B.3 pre-run audit — MaskGIT-aligned, label-free route

Status: **CODE PREPARED / NOT RUN**.

## Current scope

R6B.3 now changes four historical interface assumptions before any new training is launched:

1. MaskGIT-style simultaneous random masking for training.
2. MaskGIT scheduled-parallel confidence/remask decoding for generation.
3. No class conditioning: 0/1/2 labels do not enter the neural predictor, MSDL learning, or generation.
4. No fixed-random token/position/class embedding table.

The frozen R6B.1 SNN-VQ tokenizer and exact P3 local dynamical core remain the neural substrate.

## One-to-one protocol audit

| Operation | Standard MaskGIT principle | Current R6B.3 | Status |
|---|---|---|---|
| visual tokenizer | learned VQ tokenizer | frozen sensory-SNN VQ tokenizer | deliberate SNN replacement |
| visual-token representation | learned token embedding | geometry-preserving embedding of frozen SNN-VQ codebook | no fixed-random table |
| position representation | learned position embedding | deterministic 2-D sinusoidal 7x7 representation | no fixed-random table |
| class conditioning | optional class token in class-conditional model | absent | deliberate label-free choice |
| training mask ratio | random ratio with cosine schedule | same task-level rule | aligned |
| positions masked together | multiple random positions | multiple random positions | aligned |
| context during masked-target predictions | fixed corrupted context | fixed corrupted context | aligned |
| ground-truth reveal within masked batch | no | no | aligned |
| predictor learning | BP/CE | local persistent E/M MSDL | deliberate learning-rule replacement |
| generation start | all MASK | all 49 MASK | aligned |
| generation proposal | all current MASK sites | all current MASK sites | aligned |
| sampled-token confidence | yes | yes | aligned |
| cosine confidence remask | yes | yes | aligned |
| iterative refinement | yes | 8 rounds by default | aligned |

## Input representation correction

Historical P3 input was

`x_i = random_token[z_i] + random_position[i] + random_class[y]`.

Current R6B.3 is

`x_i = VQ_geometry[z_i] + sinusoidal_2D_position[i]`.

Token embeddings are obtained by centering the frozen SNN-VQ codebook and applying a deterministic isometric DCT lift from the VQ latent dimension to P3 dimension D. Pairwise VQ-code distances are preserved up to one global scale. MASK is the centered VQ origin (zero vector).

Position embeddings are deterministic 2-D sinusoidal coordinates for the 7x7 grid. There is no class table.

This removes the arbitrary fixed-random bottleneck without adding a new embedding-specific credit-assignment mechanism. Making token/position tables plastic would require a new local mismatch pathway to all visible inputs and is therefore deferred as a separate intervention rather than silently added to P3/MSDL.

## Label boundary

MNIST labels remain available only for:

- selecting the historical 0/1/2 evaluation subset;
- an external post-hoc classifier used to describe generated samples.

They are not passed to tokenizer encoding, token/position context construction, P3/MSDL learning, or MaskGIT generation. Consequently class-conditioned `condition_match` is no longer a valid generation metric and has been removed from the R6B.3 gate.

## Startup scaffold status

The WR startup scaffold is **still present** in current R6B.3. It now sees label-free MaskGIT-style corrupted contexts, but it remains an extra bootstrapping stage not present in standard MaskGIT.

Its necessity is not assumed. A separate staged scaffold-removal plan is stored in `R6B3_SCAFFOLD_REMOVAL_PLAN.md`. The core algebra suggests native bootstrapping may be possible because the direct WR MSDL update can be nonzero even at `WR=0`; after WR moves, the reciprocal BR pathway can activate deeper plasticity. This must be checked causally before deleting the scaffold.

## SpiNNaker boundary

The expensive predictor remains SNN/P3. The MaskGIT scheduler remains a small conventional controller. The new token representation is a fixed lookup derived from the frozen codebook and the position representation is deterministic, so this correction does not introduce a new online optimizer or host-side training requirement.

## Execution status

No R6B.3 training/generation workflow has been launched after these changes. The workflow remains `workflow_dispatch` only.
