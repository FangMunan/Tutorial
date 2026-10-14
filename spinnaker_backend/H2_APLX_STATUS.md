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

## H2E-board first physical execution — EXECUTED, self-test vector bug exposed

The first physical Manchester/spalloc run reached a real SpiNNaker core:

```text
placement: x=0, y=0, p=3
app_id:    17
user0:     0x48324531  (H2E magic)      PASS
user2:     0           (chip id)        PASS
user3:     196625      (core/app id)    PASS
user1:     5           (self-test rc)   FAIL at check 5
```

This proves that allocation, placement, APLX loading, SYNC0 start, ARM execution,
VCPU-register writeback and host readback all occurred on physical hardware.
The scientific H2E-board acceptance gate remains FAIL because the arithmetic
self-test itself returned rc=5.

### rc=5 diagnosis

The failure was in the **self-test expected value**, not in the H/g kernel.
The test used:

```text
H0      = 1000          Q20 code
old_pk  = 100           Q16 code
new_pk  = 120           Q16 code
old_v   = new_v = 1<<20 Q20 code (=1.0)
```

The resident H update converts `pk(Q16) * v(Q20)` to H(Q20), therefore shifts by
16 bits.  With v=1.0:

```text
old outer code = 100 * 2^(20-16) = 1600
new outer code = 120 * 2^(20-16) = 1920
H_new = 1000 - 1600 + 1920 = 1320
```

The original APLX incorrectly expected `1020`, producing rc=5 even though the
kernel computed the correct Q-format result.

The APLX source is corrected to require `H[0] == 1320`.  A new host regression
`test_h2e_board_contract.py` now executes this exact board vector through the C
kernels before every APLX build so this particular contract mismatch cannot
recur silently.

## H2 stages

- **H2A — PASS:** fixed-point attention/sparse primitives.
- **H2B — PASS:** integer LayerNorm, event Q/K/V and stable-a.
- **H2C — PASS:** fixed-point Izh RS + online trajectory decoder.
- **H2D — PASS:** complete host-C resident block on actual frozen H1.1 HIR,
  including persistent changed-token execution.
- **H2E — PASS:** synchronized custom application cross-compiles into a genuine
  ARM968/SpiNNaker APLX image.
- **H2E-board transport/execution path — VERIFIED:** physical APLX execution and
  result readback are established by the first run.
- **H2E-board arithmetic acceptance — PENDING corrected APLX rerun:** first run
  stopped at rc=5 because of the incorrect self-test constant above.
- **H2F — after corrected H2E-board PASS:** SDRAM frozen-HIR loader,
  changed-token ingress, complete resident board execution, and output/state
  readback versus H2D.

## Claim boundary

Current strongest valid statement:

> The H2 APLX has executed on a physical SpiNNaker core and the board/host
> transport path is verified.  The first arithmetic acceptance run failed at
> self-test check 5 because the test vector encoded the wrong Q-format expected
> H value; the kernel itself matched the frozen contract and the expected value
> has been corrected from 1020 to 1320.

Do not yet mark H2E-board arithmetic acceptance as PASS until the corrected APLX
returns rc=0 and the six host-side PASS checks are all true.
