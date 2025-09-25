#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(ROOT, 'src'))

from ah_rag.agent.gym_env import AHRAGGymEnv
from ah_rag.eval.answer_eval import AnswerEvaluator
from ah_rag.answer.generator import AnswerGenerator
from ah_rag.agent.policy_bc import load_bc, act_bc
from ah_rag.agent.policy_ppo import load_ppo, act_ppo


def load_policy(path: str) -> Dict[int, float]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    probs = obj.get("action_probs", {})
    return {int(k): float(v) for k, v in probs.items()}


def sample_action(probs: Dict[int, float], n_actions: int) -> int:
    if not probs:
        return random.randrange(n_actions)
    keys, ps = zip(*sorted(probs.items()))
    # normalize
    s = sum(ps)
    if s <= 0:
        return random.randrange(n_actions)
    psn = [p / s for p in ps]
    r = random.random()
    acc = 0.0
    for k, p in zip(keys, psn):
        acc += p
        if r <= acc:
            return int(k)
    return int(keys[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a simple action-probability policy")
    ap.add_argument("--dataset", type=str, default="hotpotqa")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--policy", type=str, default="artifacts/rl/policy.json")
    ap.add_argument("--bc-model", type=str, default=None, help="Path to BC .pt policy; if set overrides --policy")
    ap.add_argument("--ppo-model", type=str, default=None, help="Path to PPO .pt policy; if set overrides others")
    ap.add_argument("--out", type=str, default="artifacts/rl/eval.json")
    ap.add_argument("--max-steps", type=int, default=6)
    args = ap.parse_args()

    sys.path.append(ROOT)
    from scripts.run_benchmark import load_dataset  # type: ignore
    data: List[Dict[str, Any]] = load_dataset(args.dataset, limit=args.limit)
    env = AHRAGGymEnv(graph_dir="graph", max_steps=args.max_steps)
    policy = load_policy(args.policy) if (not args.bc_model and not args.ppo_model) and os.path.exists(args.policy) else {}
    bc = load_bc(args.bc_model) if args.bc_model and os.path.exists(args.bc_model) else None
    ppo = load_ppo(args.ppo_model) if args.ppo_model and os.path.exists(args.ppo_model) else None

    evaluator = AnswerEvaluator(system_type="ah_rag")
    gen = AnswerGenerator()

    results: List[Dict[str, Any]] = []
    for item in data:
        q = item["question"]
        vec, info = env.reset(q)
        done = False
        while not done:
            # Try to respect action mask from env
            mask = None
            try:
                mask = env.get_action_mask()
            except Exception:
                mask = None
            if ppo is not None:
                a = act_ppo(ppo, vec)
            elif bc is not None:
                a = act_bc(bc, vec)
            else:
                a = sample_action(policy, env.action_size)
            if mask is not None and mask[a] == 0:
                # choose first valid action; fallback to end_episode
                valid = [i for i, v in enumerate(mask) if v == 1]
                a = valid[0] if valid else env.action_size - 1
            vec, r, done, sinfo = env.step(a)

        # Build minimal answer using current context pipeline
        # Note: for evaluation we reuse standard inference components via AnswerGenerator
        # Context/evidence lives in env through committed selections
        # For simplicity, skip full context assembly and rely on downstream evaluator retriever metrics.
        # If needed, upgrade to full inference integration.
        answer_obj = {
            "query": q,
            "answer": "",  # generation skipped in this lightweight eval
            "gold_answers": item.get("answers", []),
            "evidence": {},
            "context": {},
            "retrieved_nodes": list(env.env.selection_set),
            "session_data": env.env.stats,
        }
        from ah_rag.graph.hierarchical_graph import HierarchicalGraph
        hg = HierarchicalGraph.load("graph")
        unified = evaluator.evaluate(answer_obj, graph=hg, config={})
        results.append({
            "id": item.get("id"),
            "scores": unified["scores"],
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"items": results}, f, ensure_ascii=False, indent=2)
    print(f"Saved eval to {args.out}")


if __name__ == "__main__":
    main()
