# G0 checkpoint note

The G0 runner serializes the complete live `MaskGITP3System` Python object plus NumPy/Python/PyTorch RNG states. This is intentionally broader than a weights-only checkpoint because the learner carries persistent dynamical state. The companion `*.inventory.json` records every top-level system field and ndarray shape for audit.

Before a continuation run is treated as scientifically valid, the saved checkpoint must pass a round-trip parity test: load immediately, compare all serialized ndarray fields and counters, then compare one deterministic forward/evaluation operation against the pre-save system. A weights-only restore is not acceptable for G1/G2 continuation.
