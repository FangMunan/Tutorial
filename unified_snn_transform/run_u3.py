#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import run_u3_trained_retrieval as core


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='unified_snn_transform/run/U3')
    ap.add_argument('--seed',type=int,default=7);ap.add_argument('--tokens',type=int,default=16)
    ap.add_argument('--d',type=int,default=32);ap.add_argument('--heads',type=int,default=2)
    ap.add_argument('--features',type=int,default=32);ap.add_argument('--ff',type=int,default=64)
    ap.add_argument('--value-dim',type=int,default=8)
    ap.add_argument('--train-steps',type=int,default=800);ap.add_argument('--train-batch',type=int,default=128)
    ap.add_argument('--lr',type=float,default=2e-3);ap.add_argument('--report-every',type=int,default=100)
    ap.add_argument('--eval-batches',type=int,default=8);ap.add_argument('--eval-batch',type=int,default=128)
    ap.add_argument('--trajectory-steps',type=int,default=6);ap.add_argument('--audit-batch',type=int,default=64)
    ap.add_argument('--task-gate',type=float,default=.85);ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args()
    # Core converts the output location with Path(args.out), while keeping args.out
    # itself JSON-serializable in the archived result config.
    core.run(args)


if __name__=='__main__':
    main()
