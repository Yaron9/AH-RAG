#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(ROOT, 'src'))

from ah_rag.agent.gym_env import AHRAGGymEnv
from ah_rag.agent.policy_ppo import ppo_train, PPOConfig


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a minimal PPO policy on AHRAGGymEnv")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--out", type=str, default="artifacts/rl/ppo_policy.pt")
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--early-patience", type=int, default=4)
    ap.add_argument("--early-min", type=float, default=0.02)
    args = ap.parse_args()

    def env_ctor():
        return AHRAGGymEnv(graph_dir="graph", max_steps=args.max_steps)

    cfg = PPOConfig(epochs=2, gamma=0.99, clip_eps=0.2, entropy_coef=0.01, value_coef=0.5, lr=3e-4, batch_size=256)
    ppo_train(env_ctor,
              total_episodes=args.episodes,
              max_steps=args.max_steps,
              ppo_cfg=cfg,
              save_path=args.out,
              n_envs=args.envs,
              early_stop_patience=args.early_patience,
              early_stop_min_improve=args.early_min)


if __name__ == "__main__":
    main()
