#!/usr/bin/env python3
"""Hotfix wrapper for E29 audit.

The original audit used sklearn solver='liblinear', which recent sklearn versions
reject for 3-class LogisticRegression. This wrapper changes only the external
generation-evaluation classifier; it does not change B0, H/g, transferred weights,
or any forward-equivalence metric.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
import e29_forward_equivalence as e29


def fixed_classifier(xtr, ytr):
    return LogisticRegression(
        max_iter=500,
        solver='lbfgs',
        random_state=7,
    ).fit(xtr.reshape(len(xtr), -1), ytr)


e29.classifier = fixed_classifier

if __name__ == '__main__':
    e29.main()
