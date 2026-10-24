# D7–D9 archive inventory and dependency audit

Audited branch: `gate-d-plasticity-spectrum-audit`.

Decision rule: uncertain or shared files remain active. `DELETE_AFTER_ARCHIVE` means the file is legacy MaskGIT Gate-D forensic/mechanism code, has an exact archive copy, and is not required by the active MaskGIT baseline/retraining dependency graph.

## Legacy candidates

| File | Purpose | MaskGIT-only? | Still referenced? | Action |
|---|---|---:|---|---|
| `GATE_D7A_PROTOCOL.md` | W2 trajectory clamp protocol | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `GATE_D7A1_PROTOCOL.md` | compensation mapping protocol | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `GATE_D7A2_PROTOCOL.md` | b2/bO pair stopping protocol | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `GATE_D8A_PROTOCOL.md` | functional-mode identification protocol | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `GATE_D8B_PROTOCOL.md` | rank-1 causal intervention protocol | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `GATE_D9A_PROTOCOL.md` | trajectory functional-regulation protocol | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `GATE_D9A1_PROTOCOL.md` | plastic functional-velocity audit protocol | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `GATE_D9A2R_PROTOCOL.md` | matched trajectory-relative geometry protocol | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `GATE_D9A3_PROTOCOL.md` | pair-versus-residual consolidation protocol | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `gate_d_d7a.py` | D7A implementation | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `gate_d_d7a1.py` | D7A.1 implementation | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `gate_d_d7a1_same_run_w2.py` | D7A.1 provenance launcher | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `gate_d_d7a2.py` | D7A.2 implementation | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `gate_d_d8a.py` | D8A implementation | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `gate_d_d8a_theta_fix.py` | D8A compatibility launcher | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `gate_d_d8b.py` | D8B implementation | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `gate_d_d9a.py` | D9A implementation | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `gate_d_d9a1.py` | D9A.1 implementation | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `gate_d_d9a1_compat.py` | D9A.1 compatibility launcher | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `gate_d_d9a2r.py` | D9A.2-R implementation | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `gate_d_d9a2r_compat.py` | D9A.2-R provenance launcher | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `gate_d_d9a3.py` | D9A.3 implementation | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `gate_d_d9a_vector_theta_launcher.py` | D9A compatibility launcher | Yes | Legacy D7–D9 only | DELETE_AFTER_ARCHIVE |
| `.github/workflows/snn_transform_gate_d_d7a.yml` | D7A Actions workflow | Yes | Legacy gate only | DELETE_AFTER_ARCHIVE |
| `.github/workflows/snn_transform_gate_d_d7a1.yml` | D7A.1 Actions workflow | Yes | Legacy gate only | DELETE_AFTER_ARCHIVE |
| `.github/workflows/snn_transform_gate_d_d7a2.yml` | D7A.2 Actions workflow | Yes | Legacy gate only | DELETE_AFTER_ARCHIVE |
| `.github/workflows/snn_transform_gate_d_d8a.yml` | D8A Actions workflow | Yes | Legacy gate only | DELETE_AFTER_ARCHIVE |
| `.github/workflows/snn_transform_gate_d_d8b.yml` | D8B Actions workflow | Yes | Legacy gate only | DELETE_AFTER_ARCHIVE |
| `.github/workflows/snn_transform_gate_d_d9a.yml` | D9A Actions workflow | Yes | Legacy gate only | DELETE_AFTER_ARCHIVE |
| `.github/workflows/snn_transform_gate_d_d9a1.yml` | D9A.1 Actions workflow | Yes | Legacy gate only | DELETE_AFTER_ARCHIVE |
| `.github/workflows/snn_transform_gate_d_d9a2r.yml` | D9A.2-R Actions workflow | Yes | Legacy gate only | DELETE_AFTER_ARCHIVE |
| `.github/workflows/snn_transform_gate_d_d9a2r_compat.yml` | D9A.2-R compatibility workflow | Yes | Legacy gate only | DELETE_AFTER_ARCHIVE |
| `.github/workflows/snn_transform_gate_d_d9a3.yml` | D9A.3 Actions workflow | Yes | Legacy gate only | DELETE_AFTER_ARCHIVE |

## Shared / active files deliberately kept

| File | Purpose | Action |
|---|---|---|
| `gate_d_d3e_runtime.py` | shared Gate-D replay/runtime utility | SHARED_KEEP |
| `gate_d_d6a.py` | shared pre-D7 trajectory/gate utility | SHARED_KEEP |
| `gate_d_d6a1.py` | shared pre-D7 evaluation/bootstrap utility | SHARED_KEEP |
| `gate_d_plasticity_audit.py` | shared Gate-D plasticity/provenance utility | SHARED_KEEP |
| `gate_d_analyze.py` | older Gate-D analysis utility; not required for this cleanup and not proven private | UNCERTAIN_KEEP |
| `maskgit_input_representation.py` | label-free token/position representation | KEEP_ACTIVE |
| `maskgit_msdl_training.py` | MaskGIT masked-token local training adapter | KEEP_ACTIVE |
| `maskgit_msdl_training_unconditional.py` | label-free MaskGIT training/evaluation wrappers | KEEP_ACTIVE |
| `maskgit_scheduled_parallel.py` | reusable MaskGIT iterative sampling controller | KEEP_ACTIVE |
| `r6b1_spiking_tokenizer.py` | SNN-VQ tokenizer/encoder-decoder infrastructure | KEEP_ACTIVE |
| `r6b2_snn_encoder_p3_msdl.py` | frozen tokenizer loader and P3 connection | KEEP_ACTIVE |
| `r6b3_context_epoch_training.py` | checkpointed MaskGIT context-epoch trainer | KEEP_ACTIVE |
| `r6b3_snn_encoder_maskgit_msdl.py` | label-free MaskGIT generation/training integration | KEEP_ACTIVE |
| `r6b3_decoder_operating_point_audit.py` | reusable decoder/evaluation infrastructure | KEEP_ACTIVE |
| `r6b3_decoder_spatial_refinement_audit.py` | reusable decoder spatial evaluation | KEEP_ACTIVE |
| `r6b4_full_forward_plasticity.py` | current full-forward dynamical parameter system | KEEP_ACTIVE |
| `r6b4_full_forward_effectiveness_test.py` | reusable effectiveness evaluation | KEEP_ACTIVE |
| `r6b4_full_forward_plasticity_smoke.py` | reusable core smoke/audit | KEEP_ACTIVE |

## Dependency conclusion

The D7–D9 implementation cluster imports backward from shared D3/D6 Gate-D utilities and from R6B/MaskGIT infrastructure. The active MaskGIT baseline/retraining core listed above does not import the D7–D9 implementation cluster. Therefore the old D7–D9 cluster can be removed from the active source/workflow directories after exact archive-copy verification, while shared D3/D6 and R6B/MaskGIT infrastructure remains untouched.

D9A.4 was not present on the audited branch; no D9A.4 file was inferred or fabricated.
