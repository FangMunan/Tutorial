#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""H6-B2: corrected ionic-eligibility vs standard MaskGIT test.

Scientific question
-------------------
Does the frozen measured ionic binary stream preserve generation quality when it
controls ONLY whether a pixel is eligible to be committed, while the frozen
MaskGIT model confidence independently decides which eligible proposals are kept?

This intentionally does NOT use the H6-B hard-commit semantics.  The rule is:

    ionic opens -> confidence ranks -> cosine schedule sets the target quota.

For each non-final generation round, every pixel is physically checked exactly
once against the held-out ionseq_005 stream in row-major order.  bit=1 makes an
UNKNOWN pixel eligible for that round; bit=0 leaves it closed.  Among the opened
UNKNOWN candidates, only the highest standard MaskGIT confidence/Gumbel scores
are committed, up to the standard cosine reveal quota.  If a physical round
opens too few candidates, the deficit remains MASK and the next round attempts
to catch up to the same standard schedule.  The final round consumes additional
physical full sweeps until every pixel has observed at least one bit=1, so no
residual MASK is forced through a closed gate.

The model, A2 checkpoint, 96-step schedule, temperature, confidence definition,
held-out ionic sequence and evaluator are otherwise matched to the standard arm.
No training or model tuning occurs in this script.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import train_h6_pixel784 as h6
import test_h6b_ionic_mask as old

base = h6.base
h0 = h6.h0
NPIX = h6.NPIX
MASK_ID = h6.MASK_ID
DATA_VOCAB = h6.DATA_VOCAB
TEST_ID = h6.TEST_IDS[0]


def dump(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_eligibility_maps(bits: np.ndarray, n_images: int, steps: int):
    """Build physical open/closed maps independent of model confidence.

    Non-final rounds consume exactly one full 784-pixel sweep.  The final round
    consumes as many full sweeps as required for every spatial address to have
    observed at least one physical 1, preserving completion without bypassing
    the binary physical gate.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.ndim != 1 or len(bits) == 0 or not np.all(np.isin(np.unique(bits), [0, 1])):
        raise ValueError("held-out ionic sequence must be non-empty and binary")

    opened = np.zeros((n_images, steps, NPIX), dtype=np.bool_)
    cur = 0
    wraps = 0
    reads_per_image = np.zeros(n_images, dtype=np.int64)
    final_sweeps = np.zeros(n_images, dtype=np.int64)
    ones_per_step = np.zeros((n_images, steps), dtype=np.int16)
    L = len(bits)
    pix = np.arange(NPIX, dtype=np.int64)

    def one_sweep():
        nonlocal cur, wraps
        idx = (cur + pix) % L
        out = bits[idx].astype(np.bool_)
        old_cur = cur
        cur = int((cur + NPIX) % L)
        wraps += int((old_cur + NPIX) // L)
        return out

    for im in range(n_images):
        for st in range(steps - 1):
            m = one_sweep()
            opened[im, st] = m
            ones_per_step[im, st] = int(m.sum())
            reads_per_image[im] += NPIX

        # Physical completion round: keep checking every address until each one
        # has received at least one open event.  This is independent of confidence.
        acc = np.zeros(NPIX, dtype=np.bool_)
        ns = 0
        guard = 0
        while not acc.all():
            m = one_sweep()
            acc |= m
            ns += 1
            reads_per_image[im] += NPIX
            guard += 1
            if guard > 10000:
                raise RuntimeError("final ionic eligibility sweep guard exceeded")
        opened[im, steps - 1] = acc
        ones_per_step[im, steps - 1] = int(acc.sum())
        final_sweeps[im] = ns

    diag = {
        "rule": "one full physical sweep per non-final round; final physical sweeps until every address has opened",
        "sequence_id": TEST_ID,
        "sequence_length": int(L),
        "sequence_one_fraction": float(bits.mean()),
        "final_cursor": int(cur),
        "total_wraps": int(wraps),
        "total_reads": int(reads_per_image.sum()),
        "mean_reads_per_image": float(reads_per_image.mean()),
        "median_reads_per_image": float(np.median(reads_per_image)),
        "mean_open_fraction_nonfinal": float(opened[:, :-1].mean()) if steps > 1 else 1.0,
        "mean_final_sweeps": float(final_sweeps.mean()),
        "max_final_sweeps": int(final_sweeps.max()),
        "final_sweeps": final_sweeps.astype(int).tolist(),
        "reads_per_image": reads_per_image.astype(int).tolist(),
        "mean_open_count_by_step": ones_per_step.mean(axis=0).astype(float).tolist(),
    }
    return opened, diag


@torch.no_grad()
def generate_ionic_eligibility(model, n: int, device, steps: int, temp0: float,
                               seed: int, eligibility: np.ndarray):
    """Frozen MaskGIT confidence rule with an independent binary eligibility gate."""
    if eligibility.shape != (n, steps, NPIX):
        raise ValueError(f"eligibility shape mismatch: {eligibility.shape}")

    model.eval()
    labels = torch.zeros(n, dtype=torch.long, device=device)
    cur = torch.full((n, NPIX), MASK_ID, dtype=torch.long, device=device)
    initial = torch.full((n,), NPIX, dtype=torch.long, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    state = None

    mask_before_rows = []
    mask_after_rows = []
    quota_rows = []
    eligible_rows = []
    committed_rows = []
    shortfall_rows = []
    alpha_rows = []
    h_rows = []

    for st in range(steps):
        unk = cur.eq(MASK_ID)
        before = unk.sum(1)
        mask_before_rows.append(before.detach().cpu().numpy().astype(np.int16))

        logits, state, stats = model.forward_persistent(cur, labels, state, "selective")
        p = logits.softmax(-1)
        sampled = torch.multinomial(
            p.reshape(-1, DATA_VOCAB), 1, generator=gen
        ).reshape(n, NPIX)
        prop = torch.where(unk, sampled, cur)
        sp = p.gather(-1, prop[..., None]).squeeze(-1)

        if st == steps - 1:
            # The final physical eligibility map is all-open by construction after
            # repeated physical sweeps, so this is not a gate bypass.
            cur = prop
            desired = torch.zeros(n, dtype=torch.long, device=device)
            quota = before
            elig_count = before
            committed = before
            shortfall = torch.zeros_like(before)
        else:
            frac = (st + 1) / steps
            desired = torch.floor(initial.float() * math.cos(0.5 * math.pi * frac)).long()
            desired = torch.minimum(desired, torch.clamp(before - 1, min=0))
            quota = before - desired

            # EXACT frozen standard confidence/Gumbel definition.
            score_base = torch.where(unk, sp, torch.full_like(sp, float("inf")))
            u = torch.rand(score_base.shape, device=device, generator=gen).clamp_(1e-6, 1 - 1e-6)
            gum = -torch.log(-torch.log(u))
            score = torch.log(score_base.clamp_min(1e-12)) + temp0 * (1 - frac) * gum
            score = torch.where(torch.isinf(score_base), torch.full_like(score, float("inf")), score)

            opened = torch.from_numpy(eligibility[:, st]).to(device=device) & unk
            elig_count = opened.sum(1)
            new = cur.clone()
            committed = torch.zeros_like(before)
            shortfall = torch.zeros_like(before)

            for bi in range(n):
                q = int(quota[bi].item())
                if q <= 0:
                    continue
                cand = torch.where(opened[bi])[0]
                take = min(q, int(cand.numel()))
                if take:
                    local_scores = score[bi, cand]
                    top = torch.topk(local_scores, k=take, largest=True).indices
                    chosen = cand[top]
                    new[bi, chosen] = prop[bi, chosen]
                    committed[bi] = take
                shortfall[bi] = q - take
            cur = new

        after = cur.eq(MASK_ID).sum(1)
        mask_after_rows.append(after.detach().cpu().numpy().astype(np.int16))
        quota_rows.append(quota.detach().cpu().numpy().astype(np.int16))
        eligible_rows.append(elig_count.detach().cpu().numpy().astype(np.int16))
        committed_rows.append(committed.detach().cpu().numpy().astype(np.int16))
        shortfall_rows.append(shortfall.detach().cpu().numpy().astype(np.int16))
        alpha_rows.append(float(stats["alpha_mean"]))
        h_rows.append(float(stats["state_H_rms"]))

    if cur.eq(MASK_ID).any():
        raise RuntimeError("corrected ionic eligibility generation ended with residual MASK")

    diag = {
        "mask_before": np.stack(mask_before_rows),
        "mask_after": np.stack(mask_after_rows),
        "target_reveal_quota": np.stack(quota_rows),
        "eligible_open_candidates": np.stack(eligible_rows),
        "committed": np.stack(committed_rows),
        "shortfall": np.stack(shortfall_rows),
        "alpha_mean": np.asarray(alpha_rows, np.float32),
        "state_H_rms": np.asarray(h_rows, np.float32),
    }
    return cur.cpu().numpy().astype(np.int64), diag


def summarize_generation_diag(diag):
    short = np.asarray(diag["shortfall"], dtype=np.int64)
    quota = np.asarray(diag["target_reveal_quota"], dtype=np.int64)
    elig = np.asarray(diag["eligible_open_candidates"], dtype=np.int64)
    committed = np.asarray(diag["committed"], dtype=np.int64)
    nonfinal = short[:-1] if len(short) > 1 else short
    return {
        "total_target_commits": int(quota.sum()),
        "total_actual_commits": int(committed.sum()),
        "total_nonfinal_shortfall": int(nonfinal.sum()),
        "fraction_image_rounds_with_shortfall": float((nonfinal > 0).mean()),
        "mean_shortfall_when_positive": float(nonfinal[nonfinal > 0].mean()) if np.any(nonfinal > 0) else 0.0,
        "mean_eligible_candidates_by_step": elig.mean(axis=1).astype(float).tolist(),
        "mean_mask_after_by_step": np.asarray(diag["mask_after"]).mean(axis=1).astype(float).tolist(),
        "alpha_mean": np.asarray(diag["alpha_mean"]).astype(float).tolist(),
        "state_H_rms": np.asarray(diag["state_H_rms"]).astype(float).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--standard-dir", type=Path, required=True)
    ap.add_argument("--bounded-dir", type=Path, required=True)
    ap.add_argument("--ionic", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("H6B2_IONIC_ELIGIBILITY"))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=3.5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    h6.configure_pixel_globals()
    h6.seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    by_id = h6.load_canonical_ionic(args.ionic)
    if TEST_ID not in by_id:
        raise KeyError(f"missing held-out generation sequence {TEST_ID}")
    ionic_meta = {
        "path": str(args.ionic),
        "sha256": sha256_file(args.ionic),
        "test_sequence": TEST_ID,
        "test_length": int(len(by_id[TEST_ID])),
        "test_one_fraction": float(np.asarray(by_id[TEST_ID]).mean()),
    }
    dump(args.out / "IONIC_INPUT_AUDIT.json", ionic_meta)
    print("IONIC_INPUT", json.dumps(ionic_meta), flush=True)

    model, ma, ca, final_ck, a2_ck = old.load_model(args.standard_dir, args.bounded_dir, device)
    xtr, ytr, xva, yva, x012, xte012, xtr_all, ytr_all, xte_all, yte_all = old.load_data(
        args.seed, int(ca.get("n_per_class", 2000))
    )

    print("training matched evaluator", flush=True)
    h6.seed_all(args.seed + 30000)
    ref, refhist = base.train_ref10(xtr_all, ytr_all, xte_all, yte_all, device, epochs=4, batch=256)
    dump(args.out / "ref_history.json", refhist)

    seed_gen = args.seed + 10000

    t0 = time.time()
    tokA = old.generate_standard(model, args.n, device, args.steps, args.temperature, seed_gen)
    metA, rawA = old.full_eval("A_standard_A2", tokA, ref, xte012, x012, device, args.out)
    secA = time.time() - t0
    print("A_STANDARD", json.dumps({**old.metric_core(metA), "seconds": secA}), flush=True)

    eligibility, physical_diag = build_eligibility_maps(
        np.asarray(by_id[TEST_ID], dtype=np.uint8), args.n, args.steps
    )
    dump(args.out / "PHYSICAL_ELIGIBILITY_AUDIT.json", physical_diag)

    t0 = time.time()
    tokB, gen_diag = generate_ionic_eligibility(
        model, args.n, device, args.steps, args.temperature, seed_gen, eligibility
    )
    secB = time.time() - t0
    metB, rawB = old.full_eval("B_A2_ionic_eligibility", tokB, ref, xte012, x012, device, args.out)
    gen_summary = summarize_generation_diag(gen_diag)
    dump(args.out / "B_GENERATION_DIAGNOSTICS.json", gen_summary)

    boot = old.paired_bootstrap_valid(rawA["valid"], rawB["valid"], seed=args.seed + 7101)
    gate = old.noninferiority(metB, metA, boot)

    diff = tokA != tokB
    per_image_hamming = diff.mean(axis=1)
    paired = {
        "pixel_hamming_fraction": float(diff.mean()),
        "exact_image_fraction": float((~diff.any(axis=1)).mean()),
        "mean_per_image_hamming": float(per_image_hamming.mean()),
        "median_per_image_hamming": float(np.median(per_image_hamming)),
        "max_per_image_hamming": float(per_image_hamming.max()),
    }

    report = {
        "status": "H6B2_IONIC_ELIGIBILITY_NONINFERIOR_PASS" if gate["pass"] else "H6B2_IONIC_ELIGIBILITY_NONINFERIOR_FAIL",
        "scientific_question": "Does ionic eligibility preserve generation when standard MaskGIT confidence remains independent and decisive?",
        "controller_semantics": {
            "ionic": "binary physical eligibility only",
            "confidence": "unchanged standard log p(sampled token) + annealed Gumbel score",
            "schedule": "unchanged 96-step cosine target mask count",
            "commit_rule": "top-confidence proposals among physically opened UNKNOWN pixels",
            "hard_commit_H6B_reused": False,
            "training_or_weight_updates": False,
        },
        "ionic_input": ionic_meta,
        "physical_eligibility": physical_diag,
        "generation_diagnostics": gen_summary,
        "generation": {"n": args.n, "steps": args.steps, "temperature": args.temperature, "seed": seed_gen},
        "A_standard_A2": metA,
        "B_ionic_eligibility_A2": metB,
        "noninferiority_gate": gate,
        "paired_output_difference": paired,
        "checkpoint_sources": {"standard_final": str(final_ck), "A2": str(a2_ck)},
        "timing_seconds": {"A_standard": secA, "B_ionic_eligibility": secB},
    }
    dump(args.out / "H6B2_FINAL.json", report)
    np.savez_compressed(
        args.out / "B_generation_diagnostics_arrays.npz",
        **{k: np.asarray(v) for k, v in gen_diag.items()},
    )

    print("B_IONIC_ELIGIBILITY", json.dumps({**old.metric_core(metB), "gate": gate, "paired": paired, "seconds": secB}), flush=True)
    print("FINAL", json.dumps({"status": report["status"], "pass": gate["pass"], "A": old.metric_core(metA), "B": old.metric_core(metB), "paired": paired, "shortfall": {"fraction_image_rounds": gen_summary["fraction_image_rounds_with_shortfall"], "total": gen_summary["total_nonfinal_shortfall"]}}), flush=True)


if __name__ == "__main__":
    main()
