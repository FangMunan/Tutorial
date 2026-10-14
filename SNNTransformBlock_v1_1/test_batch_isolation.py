#!/usr/bin/env python3
"""Regression test for the v1.1 persistent-state batch-isolation contract."""
from __future__ import annotations

import torch

from SNNTransformBlock_v1_1.SNNTransformBlock_v1_1 import SNNTransformBlock, SNNTransformState


def _slice_state(state: SNNTransformState, b: int) -> SNNTransformState:
    return SNNTransformState(
        H=state.H[b:b + 1].clone(),
        g=state.g[b:b + 1].clone(),
        pq=state.pq[b:b + 1].clone(),
        pk=state.pk[b:b + 1].clone(),
        v=state.v[b:b + 1].clone(),
    )


def main() -> None:
    torch.manual_seed(1701)
    block = SNNTransformBlock(d=32, heads=2, features=16, ff=64, seed=17).eval()

    B, N, D = 3, 8, 32
    x0 = torch.randn(B, N, D)
    with torch.no_grad():
        _, state0 = block(x0)

    # Change only batch item 1.  Other resident H/g states must remain bitwise
    # unchanged; no delta is allowed to broadcast across the batch dimension.
    x1 = x0.clone()
    changed = torch.zeros(B, N, dtype=torch.bool)
    changed[1, [1, 4, 6]] = True
    x1[1, [1, 4, 6]] += torch.randn(3, D) * 0.25

    with torch.no_grad():
        y_batch, state_batch = block(x1, state0, changed)

        changed_one = changed[1:2].clone()
        y_one, state_one = block(x1[1:2], _slice_state(state0, 1), changed_one)

    assert torch.equal(state_batch.H[0], state0.H[0])
    assert torch.equal(state_batch.g[0], state0.g[0])
    assert torch.equal(state_batch.H[2], state0.H[2])
    assert torch.equal(state_batch.g[2], state0.g[2])

    torch.testing.assert_close(state_batch.H[1:2], state_one.H, rtol=0.0, atol=0.0)
    torch.testing.assert_close(state_batch.g[1:2], state_one.g, rtol=0.0, atol=0.0)
    torch.testing.assert_close(state_batch.pq[1:2], state_one.pq, rtol=0.0, atol=0.0)
    torch.testing.assert_close(state_batch.pk[1:2], state_one.pk, rtol=0.0, atol=0.0)
    torch.testing.assert_close(state_batch.v[1:2], state_one.v, rtol=0.0, atol=0.0)
    torch.testing.assert_close(y_batch[1:2], y_one, rtol=0.0, atol=0.0)

    print({
        "batch_isolation": True,
        "changed_batch": 1,
        "changed_tokens": int(changed[1].sum().item()),
        "unaffected_H_bitwise": True,
        "unaffected_g_bitwise": True,
        "isolated_execution_bitwise": True,
    })


if __name__ == "__main__":
    main()
