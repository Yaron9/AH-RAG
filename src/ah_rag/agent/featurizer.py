from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np


def _node_feats(n: Dict[str, Any]) -> List[float]:
    # One-hot for node_type: entity, summary, other
    nt = str(n.get("node_type") or "")
    t_entity = 1.0 if nt == "entity" else 0.0
    t_summary = 1.0 if nt == "summary" else 0.0
    t_other = 1.0 if (nt not in {"entity", "summary"}) else 0.0
    layer = float(n.get("layer") or 0)
    score = float(n.get("score") or 0.0)
    semantic = float(n.get("semantic") or 0.0)
    judge = float(n.get("judge_overall") or 0.0)
    conf = float(n.get("confidence") or 0.0)
    return [t_entity, t_summary, t_other, layer, score, semantic, judge, conf]


def featurize_observation(obs: Dict[str, Any], k_nodes: int = 10) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Convert environment observation to a fixed-length vector.

    Vector layout (float32):
      - 4 globals: step, selection_size, frontier_size, n_seeds
      - k_nodes blocks of 8 dims for top-k selection entries (padded)
    Returns (vec, aux_info) where aux_info may include selected node_ids for debugging.
    """
    step = float(obs.get("step") or 0)
    state = obs.get("state") or {}
    selection_ids = state.get("selection_ids") or []
    frontier_ids = state.get("frontier_ids") or []
    n_seeds = len(obs.get("seeds") or [])

    # Base features
    feats: List[float] = [
        step,
        float(len(selection_ids)),
        float(len(frontier_ids)),
        float(n_seeds),
    ]

    # Top-k nodes from current selection list
    sel = (obs.get("selection") or [])[:k_nodes]
    node_ids: List[str] = []
    for n in sel:
        feats.extend(_node_feats(n))
        node_ids.append(str(n.get("node_id")))

    # Pad to k_nodes
    pad_blocks = k_nodes - len(sel)
    if pad_blocks > 0:
        feats.extend([0.0] * (8 * pad_blocks))

    vec = np.asarray(feats, dtype=np.float32)
    return vec, {"top_node_ids": node_ids}

