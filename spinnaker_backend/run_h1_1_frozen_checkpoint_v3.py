#!/usr/bin/env python3
"""H1.1 v3 — indexing-correct execution of the v2 range-aware contract.

v2 failed before producing scientific metrics because NumPy advanced indexing
reordered the [head, changed-token, feature] axes in cache assignment.  This
file changes indexing only; all fixed-point formats, equations, checkpoint
parameters and gates remain those of v2.
"""
from __future__ import annotations

import numpy as np

import spinnaker_backend.run_h1_1_frozen_checkpoint_v2 as v2
from spinnaker_backend.h1_fixed_point_emulator import (
    F_ACT, dense_linear_q, div_round, layernorm_q, qround,
    sparse_linear_q, split_heads,
)


class IndexCorrectRuntime(v2.ActualCheckpointRuntime):
    def step(self, xq, state, changed):
        p = self.p
        xq = np.asarray(xq, dtype=np.int64)
        B, N, D = xq.shape
        xn = layernorm_q(xq, p.ln1_g, p.ln1_b)

        if state is None:
            q = split_heads(dense_linear_q(xn, p.Wq, p.bq), p.heads)
            k = split_heads(dense_linear_q(xn, p.Wk, p.bk), p.heads)
            value = split_heads(dense_linear_q(xn, p.Wv, p.bv), p.heads)
            pq = v2.phi_scaled(q, p.omega, p.q_specs, p.q_decoders, True, v2.F_PQ)
            pk = v2.phi_scaled(k, p.omega, p.k_specs, p.k_decoders, False, v2.F_PK)
            H = v2.outer_to_H(pk, value)
            g = pk.sum(axis=2, dtype=np.int64)
            self.q_token_evals += B * N
            self.k_token_evals += B * N
            self.v_token_evals += B * N
        else:
            if changed is None:
                raise ValueError("changed required")
            H, g = state.H.copy(), state.g.copy()
            pq, pk, value = state.pq.copy(), state.pk.copy(), state.v.copy()

            for b in range(B):
                idx = np.nonzero(np.asarray(changed[b], dtype=bool))[0]
                if not idx.size:
                    continue
                xb = xn[b:b + 1, idx]
                nq = split_heads(dense_linear_q(xb, p.Wq, p.bq), p.heads)
                nk = split_heads(dense_linear_q(xb, p.Wk, p.bk), p.heads)
                nv = split_heads(dense_linear_q(xb, p.Wv, p.bv), p.heads)
                npq = v2.phi_scaled(nq, p.omega, p.q_specs, p.q_decoders, True, v2.F_PQ)
                npk = v2.phi_scaled(nk, p.omega, p.k_specs, p.k_decoders, False, v2.F_PK)

                # Slice the batch dimension first.  This prevents NumPy advanced
                # indexing from moving the changed-token axis ahead of the head axis.
                oldpk = state.pk[b:b + 1][:, :, idx, :]
                oldv = state.v[b:b + 1][:, :, idx, :]
                H[b:b + 1] -= v2.outer_to_H(oldpk, oldv)
                H[b:b + 1] += v2.outer_to_H(npk, nv)
                g[b:b + 1] -= oldpk.sum(axis=2, dtype=np.int64)
                g[b:b + 1] += npk.sum(axis=2, dtype=np.int64)

                pq[b][:, idx, :] = npq[0]
                pk[b][:, idx, :] = npk[0]
                value[b][:, idx, :] = nv[0]

                n = int(idx.size)
                self.q_token_evals += n
                self.k_token_evals += n
                self.v_token_evals += n

        dh = D // p.heads
        attended = np.empty((B, p.heads, N, dh), dtype=np.int64)
        quotient_shift = F_ACT + v2.F_PK - v2.F_H
        if quotient_shift < 0:
            raise ValueError("chosen H scale would require pre-division right shift")
        for b in range(B):
            for h in range(p.heads):
                for n in range(N):
                    den = sum(
                        int(pq[b, h, n, m]) * int(g[b, h, m])
                        for m in range(p.features)
                    )
                    for d in range(dh):
                        num = sum(
                            int(pq[b, h, n, m]) * int(H[b, h, m, d])
                            for m in range(p.features)
                        )
                        attended[b, h, n, d] = div_round(num << quotient_shift, den)

        attended = attended.transpose(0, 2, 1, 3).reshape(B, N, D)
        a = sparse_linear_q(attended, p.Wo)
        z = xq + a
        zn = layernorm_q(z, p.ln2_g, p.ln2_b)
        hidden = np.maximum(sparse_linear_q(zn, p.W1), 0)
        y = z + sparse_linear_q(hidden, p.W2)
        return y, v2.IntState(H=H, g=g, pq=pq, pk=pk, v=value)


if __name__ == "__main__":
    # v2.main resolves ActualCheckpointRuntime from its module globals at call
    # time, so replacing it here keeps the entire v2 compiler/gates unchanged.
    v2.ActualCheckpointRuntime = IndexCorrectRuntime
    v2.main()
