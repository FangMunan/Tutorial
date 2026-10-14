# H2 — APLX resident backend status

Branch: `snn-transform-spinnaker-h0-h1-20261014`

## Frozen input contract

H2 consumes `H1.1-functional-v9`. H1.1 functional parity is accepted; strict
full-block representation parity remains separately reported as failed at
~2.612% max block-output rel-L2.

H2 is not permitted to change the v1.1 scientific operator to improve this
number.

## Implemented APLX-oriented resident kernels

`spinnaker_backend/h2_aplx/` contains the fixed-point implementation of the
complete frozen block arithmetic required for resident execution:

- Q20 dense Q/K/V projection;
- Q20 compact sparse-row WO/W1/W2 projection;
- integer LayerNorm with integer mean/variance/sqrt;
- fixed-point `stable_a` random-feature projection;
- Q22 Izhikevich RS two-half-step integration and online decoder;
- initial H/g construction;
- batch-free changed-token subtract-old/add-new H/g update;
- normalized H/g read with overflow-safe scaled integer division;
- residual, ReLU and sparse FFN composition in the H2D host harness.

## H2A/H2B/H2C primitive gate — PASS

Verified primitive result:

```text
scaled_div_cases: 4997
layernorm_max_abs_diff: 0
stable_a_max_abs_diff: 0
izh_scalar_features_checked: 771
izh_max_abs_diff: 0
attention_read_max_abs_diff: 0
sparse_test_nnz: 512
contract: H1.1-functional-v9
```

## H2D actual frozen-HIR full-step gate — PASS

GitHub Actions run: `38578938548`.

Source HIR:

- H1.1 run `38575839809`;
- artifact `SNN_TRANSFORM_SPINNAKER_H1_1_FUNCTIONAL_V9`;
- HIR `SNNTransformBlock_v1_1_HIR_H1_1_FUNCTIONAL_V9.npz`;
- source checkpoint SHA256
  `e86fdbcef1ed32652cc3547642d2890a6b40166dbb0242fda966e4a1fbc2dcc5`.

Six-step persistent trajectory:

```text
step 0: changed tokens 16, output max-abs diff 0
step 1: changed tokens  4, output max-abs diff 0
step 2: changed tokens  4, output max-abs diff 0
step 3: changed tokens  4, output max-abs diff 0
step 4: changed tokens  4, output max-abs diff 0
step 5: changed tokens  4, output max-abs diff 0
final resident state: bit-exact
```

## H2E actual SpiNNaker APLX cross-compilation — PASS

The board-test application uses the standard `SYNC_WAIT` start barrier. The
callback is queued before `spin1_start(SYNC_WAIT)`; after SYNC0 it executes the
arithmetic checks, writes IOBUF + VCPU signatures, and calls `spin1_exit`.

The first physical run established allocation/placement/load/start/readback but
returned rc=5 because the self-test expected H=1020 instead of the correct Q20
value H=1320.  The expected value was fixed and a host regression gate was added
for the exact board-test vector before cross-compilation.

## H2E-board physical arithmetic acceptance — PASS

Corrected physical Manchester/spalloc run supplied by the user:

```text
stage:     H2E_SPALLOC_BOARD_APLX_SELFTEST
PASS:      true
APLX size: 19540 bytes
placement: x=0, y=0, p=3
app_id:    17
```

All six acceptance checks passed:

```text
iobuf_pass_token       = true
iobuf_no_fail_token    = true
user0_magic             = true
user1_rc_zero           = true
user2_chip_match        = true
user3_core_app_match    = true
```

VCPU signatures:

```text
user0 = 1211254065 = 0x48324531  (H2E magic)
user1 = 0                         (self-test rc)
user2 = 0                         (chip id, matches x=0,y=0)
user3 = 196625                    (core/app signature, matches p=3/app=17)
```

Therefore **H2E-board PASS is frozen**.  This establishes physical execution of
the H2 fixed-point arithmetic self-test on a real SpiNNaker ARM968 core,
including spalloc allocation, placement, APLX loading, SYNC0 start, arithmetic
execution, IOBUF generation, VCPU signature writeback, host readback and clean
completion.

This result does not yet prove that the complete frozen H1.1 HIR has executed on
the board.

## H2 stages

- **H2A — PASS:** fixed-point attention/sparse primitives.
- **H2B — PASS:** integer LayerNorm, event Q/K/V and stable-a.
- **H2C — PASS:** fixed-point Izh RS + online trajectory decoder.
- **H2D — PASS:** complete host-C resident block on actual frozen H1.1 HIR,
  including persistent changed-token execution.
- **H2E — PASS:** synchronized custom application cross-compiles into a genuine
  ARM968/SpiNNaker APLX image.
- **H2E-board — PASS:** corrected arithmetic self-test executed on physical
  SpiNNaker hardware and all six independent host/board checks passed.
- **H2F — NEXT:** SDRAM frozen-HIR loader, persistent resident state,
  six-step changed-token execution, and output/state readback versus H2D.

## H2F acceptance target

The first H2F gate should keep the experiment deliberately narrow:

```text
frozen H1.1 functional-v9 HIR
        -> host packs deterministic board image
        -> SDRAM upload once
        -> one resident APLX run
        -> step 0: 16 token initialization
        -> steps 1..5: 4 changed tokens per step
        -> no hidden host compute/writeback loop
        -> board stores H,g,pq,pk,v persistently
        -> board writes final y,H,g,pq,pk,v result region
        -> host reads result once after completion
        -> compare against the H2D integer oracle
```

Primary scientific H2F gate:

```text
final y              bit-exact to H2D
final H              bit-exact to H2D
final g              bit-exact to H2D
final pq             bit-exact to H2D
final pk             bit-exact to H2D
final v              bit-exact to H2D
changed-token counts = 16,4,4,4,4,4
host mid-run compute/writeback calls = 0
```

If strict bit-exact parity is broken only by a documented board-specific
arithmetic/compiler difference, the discrepancy must be localized before any
tolerance-based functional gate is considered.  Do not weaken H2F to a task-only
gate pre-emptively.

## Claim boundary

Current strongest valid statement:

> The H2 fixed-point arithmetic implementation has now executed successfully on
> a physical SpiNNaker core and passed all board-side arithmetic and host-side
> signature checks.  The complete frozen H1.1 block has not yet been demonstrated
> as a resident board computation; that is the H2F gate.
