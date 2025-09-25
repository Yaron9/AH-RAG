from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import json
import os
import re

from ah_rag.agent.environment import GraphEnvironment

from ah_rag.utils.llm_client import LLMModule, create_chat_completion, is_llm_enabled


class AHRAG_Agent:
    """
    Minimal agent wrapper. V1 chooses next action given observation.
    - Rule-based fallback (no LLM required)
    - Optional LLM JSON decision via Kimi/OpenAI-compatible API
    """

    def __init__(self, env: GraphEnvironment, use_llm: bool = False) -> None:
        self.env = env
        self.use_llm = use_llm and is_llm_enabled(LLMModule.AGENT_DECISION)

    def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if self.use_llm:
            obj = self._llm_decide(observation)
            if obj is not None:
                return obj
        # rule-based fallback
        return self._rule_based(observation)

    def _sanitize(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        action = str(obj.get("action", "noop"))
        params = obj.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        return {"action": action, "params": params}

    def _rule_based(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        selection = observation.get("selection") or []
        state = observation.get("state") or {}
        frontier_ids = state.get("frontier_ids") or []
        # 1) if nothing yet, keep expanding parents of top selection
        if selection:
            top_id = selection[0].get("node_id")
            if top_id:
                return {"action": "expand_parents", "params": {"node_ids": [top_id]}}
        # 2) otherwise, if we have frontier, expand related
        if frontier_ids:
            return {"action": "expand_related", "params": {"node_ids": frontier_ids[:1]}}
        # 3) fallback to no-op end
        return {"action": "end_episode", "params": {}}

    def _build_prompt(self, observation: Dict[str, Any], include_thought: bool = False) -> str:
        # Trimmed observation for low token cost
        selection = (observation.get("selection") or [])[:3]
        trimmed = []
        for s in selection:
            trimmed.append({
                "node_id": s.get("node_id"),
                "node_type": s.get("node_type"),
                "layer": s.get("layer"),
                "title": (s.get("title") or "")[:120],
                "name": (s.get("name") or "")[:120],
                "score": s.get("score"),
            })
        state = observation.get("state") or {}
        counts = observation.get("counts") or {}
        brief = json.dumps({
            "selection": trimmed,
            "frontier_size": len(state.get("frontier_ids") or []),
            "selection_size": len(state.get("selection_ids") or []),
            "step": counts.get("step") or observation.get("step"),
        }, ensure_ascii=False, indent=2)

        actions_desc = (
            "动作说明（何时使用）：\n"
            "- expand_parents: 希望上卷到共同父以建立更抽象的上下文时优先使用。\n"
            "- expand_related: 希望横向探索相近主题时使用。\n"
            "- expand_children: 需要下钻查看组成成员时使用。\n"
            "- semantic_anchor: 仅在需要从新角度重新锚定时使用。\n"
            "- commit_selection: 确认关键节点加入选择集。\n"
            "- query_node_details: 需要细节时使用。\n"
            "- end_episode: 若连续扩展无增益，结束。\n"
        )

        schema_core = (
            "{\n"
            "  \"action\": \"semantic_anchor|expand_parents|expand_children|expand_related|commit_selection|query_node_details|end_episode\",\n"
            "  \"params\": { \"node_ids\": [\"id\"], \"query\": \"...\" }\n"
            "}"
        )
        schema = schema_core if not include_thought else (
            "{\n"
            "  \"action\": \"semantic_anchor|expand_parents|expand_children|expand_related|commit_selection|query_node_details|end_episode\",\n"
            "  \"params\": { \"node_ids\": [\"id\"], \"query\": \"...\" },\n"
            "  \"thought\": \"单句解释动机（≤30字）\"\n"
            "}"
        )
        instructions = (
            "你是检索策略助手。根据当前观测选择下一个动作，仅返回一个严格 JSON。\n"
            "- 优先顺序：expand_parents → expand_related → expand_children → end_episode。\n"
            "- 只返回一个 JSON 对象，不要输出任何额外文字。\n"
        )
        return (
            f"{instructions}\n{actions_desc}\n当前观测(裁剪):\n{brief}\n\n返回 JSON（模式）：\n{schema}"
        )

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    def _llm_decide(self, observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Try 1: normal prompt, temp 0.2
        try:
            prompt = self._build_prompt(observation, include_thought=bool(getattr(self.env, "debug", False)))
            resp = create_chat_completion(
                LLMModule.AGENT_DECISION,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            obj = self._extract_json(resp.choices[0].message.content)
            if obj is not None:
                return self._sanitize(obj)
        except Exception:
            pass
        # Try 2: tighter prompt (no thought), temp 0.0
        try:
            prompt = self._build_prompt(observation, include_thought=False)
            resp = create_chat_completion(
                LLMModule.AGENT_DECISION,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=160
            )
            obj = self._extract_json(resp.choices[0].message.content)
            if obj is not None:
                return self._sanitize(obj)
        except Exception:
            pass
        return None


def run_agent_once(env: GraphEnvironment, agent: AHRAG_Agent, seed_query: str, steps: int = 3) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    obs, info = env.reset(seed_query=seed_query)
    last_info = info
    for _ in range(steps):
        decision = agent.decide(obs)
        action = decision.get("action")
        params = decision.get("params", {})
        if action == "semantic_anchor":
            q = params.get("query") or seed_query
            obs, last_info = env.semantic_anchor(q)
        elif action == "expand_parents":
            obs, last_info = env.expand_parents(params.get("node_ids", []))
        elif action == "expand_children":
            obs, last_info = env.expand_children(params.get("node_ids", []))
        elif action == "expand_related":
            obs, last_info = env.expand_related(params.get("node_ids", []))
        elif action == "commit_selection":
            obs, last_info = env.commit_selection(params.get("node_ids", []))
        elif action == "query_node_details":
            ids = params.get("node_ids", [])
            nid = ids[0] if ids else None
            if nid:
                obs, last_info = env.query_node_details(nid)
        elif action == "end_episode":
            break
        else:
            break
    summary = env.end_episode()
    return obs, summary
