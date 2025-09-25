#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ah_rag.agent.policy_bc import train_bc


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a simple BC policy (MLP) from trajectories")
    ap.add_argument("--traj", type=str, default="artifacts/rl/trajectories.jsonl")
    ap.add_argument("--out", type=str, default="artifacts/rl/bc_policy.pt")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--actions", type=int, default=6)
    args = ap.parse_args()

    train_bc(args.traj, args.out, epochs=args.epochs, lr=args.lr, n_actions=args.actions)


if __name__ == "__main__":
    main()

