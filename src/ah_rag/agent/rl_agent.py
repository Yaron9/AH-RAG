from __future__ import annotations

from typing import Any, Dict, List

from ah_rag.agent.gym_env import AHRAGGymEnv
from ah_rag.agent.featurizer import featurize_observation
from ah_rag.agent.policy_ppo import load_ppo, act_ppo


class RLPolicyAgent:
    """
    Inference-time agent that uses a trained PPO policy to pick the next action.
    Maps discrete action ids to Environment verbs used by InferenceEngine.
    """

    def __init__(self, env: AHRAGGymEnv | Any, model_path: str) -> None:
        self.env_like = env
        self.model = load_ppo(model_path)

    def _pick_top_ids(self, observation: Dict[str, Any], k: int = 2) -> List[str]:
        ids: List[str] = []
        for n in (observation.get("selection") or [])[:k]:
            nid = n.get("node_id")
            if nid:
                ids.append(nid)
        return ids

    def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        vec, _aux = featurize_observation(observation)
        a = act_ppo(self.model, vec)
        # Map to InferenceEngine actions
        if a == 0:
            return {"action": "expand_parents", "params": {"node_ids": self._pick_top_ids(observation, 2)}}
        if a == 1:
            return {"action": "expand_children", "params": {"node_ids": self._pick_top_ids(observation, 2)}}
        if a == 2:
            return {"action": "expand_related", "params": {"node_ids": self._pick_top_ids(observation, 1)}}
        if a == 3:
            return {"action": "commit_selection", "params": {"node_ids": self._pick_top_ids(observation, 3)}}
        if a == 4:
            ids = self._pick_top_ids(observation, 1)
            return {"action": "query_node_details", "params": {"node_ids": ids}}
        return {"action": "end_episode", "params": {}}

