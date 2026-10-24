# MaskGIT / SNN_Transform restart

## Current objective

**Coarse-to-fine dynamical MaskGIT training**

Primary scientific question:

> How should a dynamical MaskGIT be trained coarse-to-fine while controlling the masked-training / autonomous-generation gap?

The active mainline no longer asks which single parameter is harmful. Legacy D7-D9 parameter/mechanism audits are archived evidence rather than the current development tree.

## Active baseline plan

1. **Conventional MaskGIT reference.** Establish a clean conventional MaskGIT baseline before introducing dynamical-learning changes.
2. **Measure the training-deployment gap.** Compare train masked CE/NLL, validation masked CE/NLL, and autonomous iterative rollout/generation performance as functions of mask ratio, mask schedule, refinement rounds, confidence policy, and commitment policy.
3. **Coarse parameter training.** Establish robust large-scale/background structure before attempting fine correction.
4. **Coupled subsystem refinement.** Refine parameter groups whose useful behavior depends on coordinated co-adaptation rather than isolated single-parameter repair.
5. **Background-dependent fine tuning.** Resolve parameters whose useful update depends on the state established by coarse and coupled stages.
6. **Autonomous rollout evaluation.** Treat iterative autonomous generation as a mandatory deployment-level endpoint, not as a consequence inferred from masked training loss alone.

The active scientific order is therefore:

`clean MaskGIT baseline`
→ `training / autonomous-generation mismatch`
→ `coarse`
→ `coupled`
→ `fine`
→ `rollout validation`.

## Legacy

Gate D7-D9 is archived mechanistic evidence, not the active development tree.

Start with:

`archive/maskgit_legacy_d7_d9/README_LEGACY_D7_D9.md`

The archive preserves the old protocols, implementations, workflow sources, scientific conclusions, and provenance references. It does **not** define the new training rule.
