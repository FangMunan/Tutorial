#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility launcher for D5C.

D5B terminal pickles were created while gate_d_d5b.py ran as __main__, so their
class reference is serialized as __main__.D5BSystem.  When D5C is launched as a
different script, plain pickle.load cannot resolve that class.  This wrapper
registers the exact D5BSystem class under __main__ and then calls D5C unchanged.
No scientific intervention, evaluator, statistic, or parameter value is altered.
"""
import __main__
import gate_d_d5b

__main__.D5BSystem = gate_d_d5b.D5BSystem

import gate_d_d5c

if __name__ == '__main__':
    gate_d_d5c.main()
