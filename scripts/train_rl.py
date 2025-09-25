#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal RL trainer placeholder (action-frequency policy)")
    ap.add_argument("--traj", type=str, default="artifacts/rl/trajectories.jsonl")
    ap.add_argument("--out", type=str, default="artifacts/rl/policy.json")
    args = ap.parse_args()

    if not os.path.exists(args.traj):
        raise SystemExit(f"Trajectory file not found: {args.traj}")

    counts = Counter()
    total = 0
    with open(args.traj, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            for s in obj.get("steps", []):
                a = s.get("action")
                if isinstance(a, int):
                    counts[a] += 1
                    total += 1

    if total == 0:
        raise SystemExit("No steps found in trajectories")

    # Convert to probability distribution
    policy = {str(a): c / total for a, c in counts.items()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"action_probs": policy}, f, ensure_ascii=False, indent=2)
    print(f"Saved naive policy to {args.out}")


if __name__ == "__main__":
    main()

