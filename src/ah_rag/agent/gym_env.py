from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import os

from ah_rag.agent.environment import GraphEnvironment
from ah_rag.agent.featurizer import featurize_observation
from ah_rag.agent.reward import step_reward


class AHRAGGymEnv:
    """
    Minimal Gym-like wrapper without external dependencies.
    Action space (discrete):
      0 expand_parents
      1 expand_children
      2 expand_related
      3 commit_top (commit current top node ids)
      4 query_details_top (first top node)
      5 end_episode
    Observations: fixed-length vector from featurizer + info dict.
    Rewards: dense per-step heuristic; final reward computed externally.
    """

    def __init__(self, graph_dir: str = "graph", max_steps: int = 6, debug: bool = False,
                 enable_action_mask: bool = True, repeat_penalty: float = 0.02) -> None:
        self.env = GraphEnvironment(graph_dir=graph_dir, debug=debug, log_level="off", redact=True)
        self.max_steps = max(1, int(max_steps))
        self._cur_step = 0
        self._last_obs: Optional[Dict[str, Any]] = None
        self._last_raw_obs: Optional[Dict[str, Any]] = None
        self._query: Optional[str] = None
        self.enable_action_mask = bool(enable_action_mask)
        self.repeat_penalty = float(repeat_penalty)
        self._last_action: Optional[int] = None

    @property
    def action_size(self) -> int:
        return 6

    def reset(self, query: str) -> Tuple[Any, Dict[str, Any]]:
        self._query = query
        self._cur_step = 0
        raw_obs, _ = self.env.reset(seed_query=query)
        vec, aux = featurize_observation(raw_obs)
        self._last_obs = {"vec": vec, "aux": aux}
        self._last_raw_obs = raw_obs
        info = {"raw_obs": raw_obs, "aux": aux}
        if self.enable_action_mask:
            info["action_mask"] = self.get_action_mask()
        self._last_action = None
        return vec, info

    def get_action_mask(self) -> List[int]:
        """Compute a binary mask of valid actions given current observation.
        1 = valid, 0 = invalid. Order matches action ids 0..5.
        Heuristics:
          - If no top nodes, only allow end_episode (index 5).
          - Otherwise, allow all except when selection is empty for commit/query.
        """
        mask = [1] * self.action_size
        raw = self._last_raw_obs or {}
        tops = (raw.get("selection") or [])
        has_top = len(tops) > 0
        if not has_top:
            # disallow all except end_episode
            for i in range(self.action_size - 1):
                mask[i] = 0
            mask[5] = 1
            return mask
        # If no tops, commit/query are invalid
        if len(tops) == 0:
            mask[3] = 0  # commit_top
            mask[4] = 0  # query_details_top
        return mask

    def _pick_top_ids(self, raw_obs: Dict[str, Any], k: int = 2) -> List[str]:
        ids = []
        for n in (raw_obs.get("selection") or [])[:k]:
            nid = n.get("node_id")
            if nid:
                ids.append(nid)
        return ids

    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        assert self._last_raw_obs is not None, "Call reset() first"
        self._cur_step += 1
        prev_raw = self._last_raw_obs
        done = False
        info: Dict[str, Any] = {"action": int(action)}

        # Map action id to environment call
        if action == 0:  # expand_parents
            obs, _ = self.env.expand_parents(self._pick_top_ids(prev_raw, 2))
        elif action == 1:  # expand_children
            obs, _ = self.env.expand_children(self._pick_top_ids(prev_raw, 2))
        elif action == 2:  # expand_related
            obs, _ = self.env.expand_related(self._pick_top_ids(prev_raw, 1))
        elif action == 3:  # commit_top
            obs, _ = self.env.commit_selection(self._pick_top_ids(prev_raw, 3))
        elif action == 4:  # query_details_top
            ids = self._pick_top_ids(prev_raw, 1)
            if ids:
                obs, _ = self.env.query_node_details(ids[0])
            else:
                obs = prev_raw
        else:  # end_episode
            done = True
            obs = prev_raw

        # Step reward (only for non-terminal transitions)
        r = 0.0 if done else step_reward(prev_raw, obs)
        # Repeat-action penalty
        if not done and self._last_action is not None and int(action) == int(self._last_action) and self.repeat_penalty > 0:
            r -= self.repeat_penalty
        done = done or (self._cur_step >= self.max_steps)

        vec, aux = featurize_observation(obs)
        self._last_obs = {"vec": vec, "aux": aux}
        self._last_raw_obs = obs
        if self.enable_action_mask:
            info["action_mask"] = self.get_action_mask()
        self._last_action = int(action)
        info.update({"raw_obs": obs, "aux": aux, "step": self._cur_step})
        return vec, float(r), bool(done), info
