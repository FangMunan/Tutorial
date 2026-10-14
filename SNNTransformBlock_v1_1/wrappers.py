"""Optional convenience wrappers for SNNTransformBlock-v1.1.

These helpers are deliberately outside the mathematical core.  In particular,
automatic changed-token inference performs an O(BND) tensor comparison and keeps
an O(BND) raw-input cache; event-native applications should instead provide the
changed mask directly to SNNTransformBlock.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .SNNTransformBlock_v1_1 import SNNTransformBlock, SNNTransformState

Tensor = torch.Tensor


@dataclass
class AutoEventState:
    core: SNNTransformState
    x_cache: Tensor


class AutoEventWrapper(nn.Module):
    """PyTorch convenience wrapper that infers changed tokens by comparison."""

    def __init__(self, block: SNNTransformBlock):
        super().__init__()
        self.block = block

    def forward(
        self,
        x: Tensor,
        state: Optional[AutoEventState] = None,
        *,
        reset: bool = False,
    ) -> Tuple[Tensor, AutoEventState]:
        if reset:
            state = None
        if state is None:
            y, core = self.block(x, None, None)
        else:
            if state.x_cache.shape != x.shape:
                raise ValueError("input shape changed; reset wrapper state")
            changed = torch.any(x != state.x_cache, dim=-1)
            y, core = self.block(x, state.core, changed)
        return y, AutoEventState(core=core, x_cache=x.detach().clone())
