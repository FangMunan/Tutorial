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

The board-test version now uses the standard `SYNC_WAIT` start barrier. This is
important because the arithmetic test is short enough that an unsynchronised
application could finish before the host front-end reliably observes the run.
The callback is queued before `spin1_start(SYNC_WAIT)`; after SYNC0 it executes
the arithmetic checks, writes IOBUF + VCPU signatures, and calls `spin1_exit`.

Latest post-SYNC compile:

- commit: `1c543cce2bbbc4236eb2f9ceaa6079aa19c171d1`;
- H2 host-kernel workflow run: `38621223349` — PASS;
- H2E APLX workflow run: `38621223025` — PASS;
- artifact: `SNN_TRANSFORM_H2E_APLX`;
- artifact ID: `11921198226`;
- artifact ZIP SHA256:
  `2b3f746fa5eeca9f76c9a89a5914a691e0b0ff764871ca798e6fd6cde847674a`;
- current APLX size: `19532` bytes.

The bundle contains:

```text
snn_transform_h2.aplx
build/gnu/snn_transform_h2.elf
build/gnu/snn_transform_h2.txt
run_h2e_spalloc_board_selftest.py
run_h2e_board_selftest.py
H2E_BOARD_TEST.md
```

## H2E-board physical test — READY, NOT YET EXECUTED

The primary runner for the Manchester environment is:

```text
spinnaker_backend/h2_aplx/run_h2e_spalloc_board_selftest.py
```

It reuses the working `[Machine]` settings from `~/.spynnaker.cfg`, requests one
physical board through spalloc, places one custom machine vertex, loads the
current APLX, starts it through the SYNC0 barrier, waits for FINISHED, reads the
core's IOBUF, verifies VCPU user registers, and releases the allocation in a
`finally` block.

A PASS requires all of:

```text
iobuf_pass_token       = true
iobuf_no_fail_token    = true
user0_magic             = true   # 0x48324531 / H2E1
user1_rc_zero           = true
user2_chip_match        = true
user3_core_app_match    = true
```

The direct `ybug` runner is retained only as a fallback for directly reachable
Ethernet boards.

## H2 stages

- **H2A — PASS:** fixed-point attention/sparse primitives.
- **H2B — PASS:** integer LayerNorm, event Q/K/V and stable-a.
- **H2C — PASS:** fixed-point Izh RS + online trajectory decoder.
- **H2D — PASS:** complete host-C resident block on actual frozen H1.1 HIR,
  including persistent changed-token execution.
- **H2E — PASS:** latest synchronized custom application cross-compiles into a
  genuine ARM968/SpiNNaker APLX image.
- **H2E-board — READY / PENDING EXECUTION:** physical one-core arithmetic test.
- **H2F — after H2E-board:** SDRAM frozen-HIR loader, changed-token ingress,
  complete resident board execution, and output/state readback versus H2D.

## Claim boundary

Current strongest valid statement:

> The latest synchronized H2 board-test application cross-compiles successfully
> into a genuine SpiNNaker APLX image, and the complete fixed-point C path is
> bit-exact to the frozen H1.1 integer contract on the tested host trajectory.

Do not yet state that the APLX has executed on a physical board. H2E-board is the
next experimental gate.
