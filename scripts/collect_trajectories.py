#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(ROOT, 'src'))

from ah_rag.agent.gym_env import AHRAGGymEnv


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect random/heuristic trajectories for AH-RAG RL")
    ap.add_argument("--dataset", type=str, default="hotpotqa", help="hotpotqa|triviaqa")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--policy", type=str, default="random", help="random")
    ap.add_argument("--out", type=str, default="artifacts/rl/trajectories.jsonl")
    ap.add_argument("--max-steps", type=int, default=6)
    args = ap.parse_args()

    # Minimal data loader (reuse run_benchmark loader for convenience)
    sys.path.append(ROOT)
    from scripts.run_benchmark import load_dataset  # type: ignore
    items: List[Dict[str, Any]] = load_dataset(args.dataset, limit=args.limit)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    env = AHRAGGymEnv(graph_dir="graph", max_steps=args.max_steps)

    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            q = it["question"]
            vec, info = env.reset(q)
            traj: Dict[str, Any] = {
                "query": q,
                "steps": [],
            }
            done = False
            while not done:
                import random
                a = random.randrange(env.action_size)
                nvec, r, done, step_info = env.step(a)
                traj["steps"].append({
                    "action": a,
                    "reward": r,
                    "obs_vec": list(map(float, vec)) if hasattr(vec, "__len__") else [],
                    "obs_aux": step_info.get("aux", {}),
                })
                vec = nvec
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
    print(f"Saved trajectories to {args.out}")


if __name__ == "__main__":
    main()
