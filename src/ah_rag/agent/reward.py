from __future__ import annotations

from typing import Any, Dict


def step_reward(prev_obs: Dict[str, Any] | None, cur_obs: Dict[str, Any]) -> float:
    """Dense step reward derived from retriever-oriented signals.
    Heuristics (no LLM):
      +1.0 per unique addition to selection set (proxy via selection_size delta)
      +0.05 per new frontier item (capped)
      -0.05 per step baseline to encourage shorter episodes
    """
    if prev_obs is None:
        return 0.0
    prev_state = prev_obs.get("state") or {}
    cur_state = cur_obs.get("state") or {}
    prev_sel = set(prev_state.get("selection_ids") or [])
    cur_sel = set(cur_state.get("selection_ids") or [])
    prev_frontier = set(prev_state.get("frontier_ids") or [])
    cur_frontier = set(cur_state.get("frontier_ids") or [])

    add_sel = len(cur_sel - prev_sel)
    add_frontier = max(0, len(cur_frontier) - len(prev_frontier))

    reward = 1.0 * add_sel + 0.05 * min(add_frontier, 10) - 0.05
    return float(reward)


def final_reward(metrics: Dict[str, float]) -> float:
    """Combine evaluator outputs into a scalar final reward.
    Inputs can include: f1, faithfulness, answer_relevancy, contextual_recall.
    """
    f1 = float(metrics.get("f1", 0.0))
    faith = float(metrics.get("faithfulness", 0.0))
    ans_rel = float(metrics.get("answer_relevancy", 0.0))
    ctx_recall = float(metrics.get("contextual_recall", 0.0))
    # weight faithful/relevancy more; keep recall as supporting factor
    return 0.4 * f1 + 0.3 * faith + 0.2 * ans_rel + 0.1 * ctx_recall

