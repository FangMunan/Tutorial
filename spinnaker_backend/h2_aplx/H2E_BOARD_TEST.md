# H2E-board — physical SpiNNaker arithmetic self-test

This gate is deliberately small.  It asks only whether the cross-compiled H2
fixed-point arithmetic application really executes on a physical SpiNNaker
ARM968 core.  It does **not** yet load the full H1.1 frozen HIR; that is H2F.

## Primary Manchester/spalloc route

Use the same Python environment in which the existing sPyNNaker Manchester
spalloc tests work.  The runner reads only the compatible `[Machine]` options
from the existing `~/.spynnaker.cfg`.

SpiNNakerGraphFrontEnd 7.4.1 requires a user config named
`~/.spiNNakerGraphFrontEnd.cfg` to exist before it will read the run-local
`spiNNakerGraphFrontEnd.cfg`.  A run-local file alone therefore raises
`NoConfigFoundException`.  The current runner handles this automatically by
creating an isolated temporary HOME under the run directory, writing the hidden
GraphFrontEnd user config there, and restoring the real HOME/XDG environment
after the test.  It does not overwrite the user's permanent home config.

From the repository root:

```bash
python spinnaker_backend/h2_aplx/run_h2e_spalloc_board_selftest.py \
  --aplx spinnaker_backend/h2_aplx/snn_transform_h2.aplx \
  --source-config ~/.spynnaker.cfg
```

If `SpiNNakerGraphFrontEnd` is not importable, install the release matching the
already-working SpiNNaker Python stack.  Do not blindly upgrade sPyNNaker,
SpiNNMan, PACMAN or SpiNNFrontEndCommon just for this test.

The script requests one board, places one machine vertex, loads
`snn_transform_h2.aplx`, waits for its SYNC0 barrier, starts it, waits for
`FINISHED`, reads IOBUF and VCPU user registers, then releases the allocation in
a `finally` block.

A scientific PASS requires all six checks:

```text
iobuf_pass_token       = true
iobuf_no_fail_token    = true
user0_magic             = true   # 0x48324531 / H2E1
user1_rc_zero           = true
user2_chip_match        = true
user3_core_app_match    = true
```

The result is written under:

```text
spinnaker_backend/run/H2E_spalloc_board/<UTC>/
    H2E_SPALLOC_BOARD_SELFTEST.json
    core_iobuf.txt
    spiNNakerGraphFrontEnd.cfg
    _gfe_home/.spiNNakerGraphFrontEnd.cfg
    snn_transform_h2.aplx
```

The generated configs can contain the copied spalloc user name and are created
with mode `0600` where supported.  The JSON intentionally records only whether
a user/group was configured, not their values.

## Direct-board fallback

Only use this route when a board Ethernet address is directly reachable from
the host:

```bash
python spinnaker_backend/h2_aplx/run_h2e_board_selftest.py \
  --board <BOARD_IP_OR_HOSTNAME> \
  --aplx spinnaker_backend/h2_aplx/snn_transform_h2.aplx
```

The direct route uses official `ybug`, performs `app_load`, sends `SYNC0`, reads
IOBUF and stops the temporary application.  `--boot` is opt-in and should not be
used on a shared/allocated machine.

## Claim boundary

After a PASS the strongest justified statement is:

> The H2 arithmetic self-test APLX executed successfully on a physical
> SpiNNaker core and returned the expected fixed-point kernel signatures.

Do **not** yet claim that the complete SNNTransformBlock/H1.1 frozen HIR has run
on SpiNNaker.  H2F must first load the real HIR into SDRAM, execute the six-step
resident/event trajectory and read back output plus `(H,g,pq,pk,v)` for parity
against the frozen H2D oracle.
