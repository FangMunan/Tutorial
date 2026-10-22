#!/usr/bin/env python3
"""Formal D6A.2 launcher.

Keeps ordinary masked-CE secondary readouts on one common deterministic mask
seed across all evaluated terminal systems. This changes no primary Gate-A
logic, causal hybrid, classifier, or learner state.
"""
import gate_d_d6a2 as d6a2

_orig_secondary_eval = d6a2.secondary_eval
_FIXED_SECONDARY_MASK_SEED = 77311  # formal seed 7311 + 70000


def _secondary_eval_common_crn(label, system, va, real_p, real_pos_p, real_pair,
                               seed, eval_images, gen_samples):
    return _orig_secondary_eval(label, system, va, real_p, real_pos_p, real_pair,
                                _FIXED_SECONDARY_MASK_SEED, eval_images, gen_samples)


d6a2.secondary_eval = _secondary_eval_common_crn

if __name__ == '__main__':
    d6a2.main()
