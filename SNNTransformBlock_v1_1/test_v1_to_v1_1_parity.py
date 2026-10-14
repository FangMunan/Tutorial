#!/usr/bin/env python3
"""Parity gate for the v1 -> v1.1 mathematical cleanup."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from SNNTransformBlock_v1.SNNTransformBlock_v1 import SNNTransformBlock as V1Block
from SNNTransformBlock_v1_1.SNNTransformBlock_v1_1 import (
    SNNTransformBlock as V11Block,
    load_v1_1,
    load_v1_checkpoint_as_v1_1,
    save_v1_1,
)


def rel(a, b):
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12))


def make_v1_checkpoint(block, *, q_specs=None, q_decoders=None, k_specs=None, k_decoders=None):
    ckpt = {
        "format": "SNNTransformBlock-v1",
        "config": block.config_dict(),
        "model_state_dict": block.state_dict(),
    }
    if q_specs is not None:
        ckpt.update({
            "q_specs": q_specs,
            "q_decoders": q_decoders,
            "k_specs": k_specs,
            "k_decoders": k_decoders,
        })
    return ckpt


def sequence(seed=11, B=3, N=13, D=32, steps=6, frac=0.25):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, N, D, generator=g)
    xs, masks = [x.clone()], [torch.ones(B, N, dtype=torch.bool)]
    nchange = max(1, round(N * frac))
    for _ in range(1, steps):
        x = x.clone()
        m = torch.zeros(B, N, dtype=torch.bool)
        for b in range(B):
            idx = torch.randperm(N, generator=g)[:nchange]
            x[b, idx] = torch.randn(nchange, D, generator=g)
            m[b, idx] = True
        xs.append(x.clone())
        masks.append(m)
    return xs, masks


def compare_sequence(old, new, xs, masks):
    so = sn = None
    metrics = {k: 0.0 for k in ("output", "H", "g", "pq", "pk", "v")}
    for t, (x, changed) in enumerate(zip(xs, masks)):
        yo, so, _ = old.step(x, so, changed, retention=1.0)
        yn, sn = new(x, sn, None if t == 0 else changed)
        metrics["output"] = max(metrics["output"], rel(yn, yo))
        for name in ("H", "g", "pq", "pk", "v"):
            metrics[name] = max(metrics[name], rel(getattr(sn, name), getattr(so, name)))
    return metrics, sn


def install_test_izh(block):
    q_specs = [
        {"bias": 5.0, "gain": 0.75, "window_ms": 8},
        {"bias": 4.0, "gain": 0.60, "window_ms": 10},
    ]
    k_specs = [
        {"bias": 3.0, "gain": 0.30, "window_ms": 9},
        {"bias": 2.0, "gain": 0.25, "window_ms": 7},
    ]

    def decoder(T, phase):
        # Positive operating range with nonzero weights on every membrane tick.
        d = torch.linspace(0.00005 + phase, 0.00012 + phase, T + 1)
        d[0] = 1.0 + 0.01 * phase
        return d

    q_dec = [decoder(s["window_ms"], 0.0 + 1e-5 * i) for i, s in enumerate(q_specs)]
    k_dec = [decoder(s["window_ms"], 2e-5 + 1e-5 * i) for i, s in enumerate(k_specs)]
    block.set_izh_interface(q_specs, q_dec, k_specs, k_dec)
    return q_specs, q_dec, k_specs, k_dec


def main():
    torch.manual_seed(1234)
    xs, masks = sequence()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # Gate A: exact PositiveRF path + nonzero theta absorption.
        old_exact = V1Block(d=32, heads=2, features=32, ff=64,
                            out_density=.25, ff1_density=.25, ff2_density=.25,
                            theta_init=0.0, retention_alpha=1.0, seed=91).eval()
        with torch.no_grad():
            old_exact.theta.copy_(torch.linspace(-0.25, 0.20, old_exact.ff))
        p1 = td / "v1_exact.pt"
        torch.save(make_v1_checkpoint(old_exact), p1)
        new_exact, _ = load_v1_checkpoint_as_v1_1(p1)
        exact_metrics, exact_state = compare_sequence(old_exact, new_exact, xs, masks)

        # Gate B: same cleanup with the neuronal Izh interface.  This also tests
        # streaming trajectory decoding against historical materialize+matmul.
        torch.manual_seed(5678)
        old_izh = V1Block(d=32, heads=2, features=32, ff=64,
                          out_density=.25, ff1_density=.25, ff2_density=.25,
                          theta_init=0.0, retention_alpha=1.0, seed=103).eval()
        with torch.no_grad():
            old_izh.theta.copy_(torch.linspace(0.18, -0.12, old_izh.ff))
        q_specs, q_dec, k_specs, k_dec = install_test_izh(old_izh)
        p2 = td / "v1_izh.pt"
        torch.save(make_v1_checkpoint(old_izh, q_specs=q_specs, q_decoders=q_dec,
                                      k_specs=k_specs, k_decoders=k_dec), p2)
        new_izh, _ = load_v1_checkpoint_as_v1_1(p2)
        izh_metrics, izh_state = compare_sequence(old_izh, new_izh, xs, masks)

        # Gate C: v1.1 checkpoint round trip.
        p3 = td / "v1_1.pt"
        save_v1_1(new_izh, p3, metadata={"parity_test": True})
        roundtrip, saved = load_v1_1(p3)
        yr, _ = roundtrip(xs[0], None, None)
        yn, _ = new_izh(xs[0], None, None)
        roundtrip_rel = rel(yr, yn)

        # Gate D: residue really is absent from the production core.
        residue = {
            "theta_parameter_absent": "theta" not in dict(new_exact.named_parameters()),
            "minimal_state_fields": set(vars(exact_state)) == {"H", "g", "pq", "pk", "v"},
            "no_x_cache": not hasattr(exact_state, "x_cache"),
            "no_weight_state": not hasattr(exact_state, "weight"),
            "v1_1_format": saved.get("format") == V11Block.VERSION,
        }
        try:
            new_exact(xs[1], exact_state, None)
            residue["explicit_event_required"] = False
        except ValueError:
            residue["explicit_event_required"] = True

    gates = {
        "exact_output_rel_l2_le_1e7": exact_metrics["output"] <= 1e-7,
        "exact_state_rel_l2_le_1e7": max(exact_metrics[k] for k in ("H", "g", "pq", "pk", "v")) <= 1e-7,
        "izh_output_rel_l2_le_1e5": izh_metrics["output"] <= 1e-5,
        "izh_state_rel_l2_le_1e5": max(izh_metrics[k] for k in ("H", "g", "pq", "pk", "v")) <= 1e-5,
        "checkpoint_roundtrip_rel_l2_le_1e7": roundtrip_rel <= 1e-7,
        "residue_removed": all(residue.values()),
    }
    result = {
        "stage": "SNN_TRANSFORM_BLOCK_V1_1_CLEAN_PARITY",
        "PASS": all(gates.values()),
        "exact_metrics": exact_metrics,
        "izh_metrics": izh_metrics,
        "roundtrip_rel_l2": roundtrip_rel,
        "residue": residue,
        "gates": gates,
        "decision": "freeze clean v1.1 core" if all(gates.values()) else "do not freeze; localize failed cleanup gate",
    }
    print(json.dumps(result, indent=2))
    if not result["PASS"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
