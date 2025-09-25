from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, continue without it

from ah_rag.agent.environment import GraphEnvironment
from ah_rag.agent.agent import AHRAG_Agent
from ah_rag.answer.context_processor import ContextProcessor
from ah_rag.answer.generator import AnswerGenerator
from ah_rag.utils.config import load_config

# OpenAI import removed - now using unified LLM client manager


class InferenceEngine:
    """
    V1 single-turn inference engine: think-act loop, evidence aggregation, answer generation.
    """

    def __init__(self, env: GraphEnvironment, agent: AHRAG_Agent) -> None:
        self.env = env
        self.agent = agent

    def run_inference(self, query: str, steps: int = 4) -> Dict[str, Any]:
        obs, info = self.env.reset(seed_query=query)
        used_actions: List[Dict[str, Any]] = [info]

        # CRITICAL FIX: Immediately commit the top findings from semantic_anchor
        # This ensures we have evidence even if subsequent expand operations return nothing
        initial_top_ids = _pick_top_ids(obs, query)
        if initial_top_ids:
            obs_commit, info_commit = self.env.commit_selection(initial_top_ids)
            used_actions.append(info_commit)

        for _ in range(max(1, steps - 1)):
            decision = self.agent.decide(obs)
            action = decision.get("action")
            params = decision.get("params", {})
            if action == "semantic_anchor":
                q = params.get("query") or query
                obs, info = self.env.semantic_anchor(q)
            elif action == "expand_parents":
                obs, info = self.env.expand_parents(params.get("node_ids", []) or _pick_top_ids(obs, query))
            elif action == "expand_children":
                obs, info = self.env.expand_children(params.get("node_ids", []) or _pick_top_ids(obs, query))
            elif action == "expand_related":
                obs, info = self.env.expand_related(params.get("node_ids", []) or _pick_top_ids(obs, query))
            elif action == "commit_selection":
                ids = params.get("node_ids", []) or _pick_top_ids(obs, query)
                obs, info = self.env.commit_selection(ids)
            elif action == "query_node_details":
                ids = params.get("node_ids", []) or _pick_top_ids(obs, query)
                if ids:
                    obs, info = self.env.query_node_details(ids[0])
            elif action == "end_episode":
                break
            else:
                break
            used_actions.append(info)
            # auto-commit the current top selection to evidence set
            top_ids = _pick_top_ids(obs, query)
            if top_ids:
                obs, info2 = self.env.commit_selection(top_ids)
                used_actions.append(info2)

        evidence = self._collect_evidence(max_summaries=3, max_entities=5)
        # Context assembly + Answer generation
        cfg = load_config()

        # Read from both old and new config locations for compatibility
        answer_cfg = cfg.get("answer", {})
        llm_cfg = cfg.get("llm", {})
        answer_llm_cfg = llm_cfg.get("modules", {}).get("answer_generation", {})

        token_budget = int(answer_cfg.get("total_context_budget", 6000))
        ctx_cfg = {
            "skeleton_ratio": answer_cfg.get("skeleton_ratio", 0.2),
            "reserve_ratio": answer_cfg.get("reserve_ratio", 0.1),
            "enable_kept_spans": answer_cfg.get("enable_kept_spans", True),
            "enable_cache": answer_cfg.get("enable_cache", True),
            "summarizer_model": answer_cfg.get("summarizer_model"),
            "summarizer_max_tokens": answer_cfg.get("summarizer_max_tokens", 256),
            "rank_weights": {"judge": 0.4, "conf": 0.2, "layer": 0.4},
        }
        cp = ContextProcessor()
        context = cp.build_context(evidence, self.env.hg, token_budget, ctx_cfg)

        # Event logging
        try:
            ev_path = os.path.join(self.env.session_path, "events.jsonl")
            os.makedirs(self.env.session_path, exist_ok=True)
            with open(ev_path, "a", encoding="utf-8") as ef:
                ef.write(json.dumps({
                    "event": "context_assembled",
                    "stats": context.get("stats", {}),
                    "used_nodes": context.get("used_nodes", []),
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

        # Answer generation with unified LLM config
        gen = AnswerGenerator()
        gen_cfg = {
            # LLM settings from unified config with fallback to old config
            "use_llm": answer_llm_cfg.get("enabled", llm_cfg.get("enabled", answer_cfg.get("use_llm", True))),
            "model": answer_llm_cfg.get("model", llm_cfg.get("default_model", answer_cfg.get("model", "deepseek-chat"))),
            "temperature": answer_llm_cfg.get("temperature", llm_cfg.get("default_temperature", answer_cfg.get("temperature", 0.1))),
            "max_retries": answer_llm_cfg.get("max_retries", llm_cfg.get("default_max_retries", answer_cfg.get("max_retries", 2))),
        }
        answer = gen.generate(query, context, gen_cfg)
        summary = self.env.end_episode()
        out = {
            "query": query,
            "answer": answer.get("answer"),
            "rationale": answer.get("rationale"),
            "citations": answer.get("citations"),
            "used_actions": used_actions,
            "metrics": summary.get("stats", {}).get("cumulative", {}),
            "session_path": self.env.session_path,
            "evidence": evidence,  # 添加证据信息
            "context": context,    # 添加处理后的上下文
            "retrieved_nodes": list(self.env.selection_set),  # 添加检索节点
        }
        # persist answer.json
        try:
            with open(os.path.join(self.env.session_path, "answer.json"), "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return out

    def _collect_evidence(self, max_summaries: int = 3, max_entities: int = 5) -> Dict[str, Any]:
        hg = self.env.hg
        selection_ids = list(self.env.selection_set)
        # rank selection by simple priority: summaries first (higher level), then entities
        summaries: List[str] = []
        entities: List[str] = []
        for nid in selection_ids:
            d = hg.G.nodes.get(nid, {})
            if d.get("node_type") == "summary":
                summaries.append(nid)
            elif d.get("node_type") == "entity":
                entities.append(nid)
                parents = d.get("l1_parents")
                if parents:
                    try:
                        parent_map = json.loads(parents) if isinstance(parents, str) else parents
                    except Exception:
                        parent_map = {}
                    for tid in parent_map.keys():
                        sid = hg.topic_to_summary_id.get(int(tid))
                        if sid and sid not in summaries:
                            summaries.append(sid)
                            if sid not in selection_ids:
                                selection_ids.append(sid)
        # deduplicate while preserving order
        summaries = list(dict.fromkeys(summaries))
        entities = list(dict.fromkeys(entities))

        # limit
        summaries = summaries[:max_summaries]
        entities = entities[:max_entities]
        # craft concise snippets
        def brief(nid: str) -> Dict[str, Any]:
            d = hg.G.nodes.get(nid, {})
            return {
                "node_id": nid,
                "node_type": d.get("node_type"),
                "title": (d.get("title") or d.get("name") or "")[:120],
                "summary": (d.get("summary_text") or d.get("summary") or d.get("description") or "")[:240],
            }
        summary_briefs = [brief(n) for n in summaries]
        entity_briefs = [brief(n) for n in entities]

        def add_members_from_summary(summary_id: str, depth: int = 0) -> None:
            if depth > 2:
                return
            data = hg.G.nodes.get(summary_id, {})
            members_raw = data.get("members")
            if not members_raw:
                return
            try:
                members = json.loads(members_raw) if isinstance(members_raw, str) else members_raw
            except Exception:
                members = []
            for member in members:
                if len(entity_briefs) >= max_entities:
                    return
                if isinstance(member, str) and member.startswith("sum:"):
                    add_members_from_summary(member, depth + 1)
                    continue
                name = member if isinstance(member, str) else None
                if not name:
                    continue
                ent_id = hg.name_to_entity_id.get(name)
                if ent_id and ent_id not in entities:
                    entity_briefs.append(brief(ent_id))
                    entities.append(ent_id)

        if len(entity_briefs) < max_entities:
            for sid in summaries:
                add_members_from_summary(sid)
                if len(entity_briefs) >= max_entities:
                    break

        return {"summaries": summary_briefs, "entities": entity_briefs}

# Removed old _generate_answer and _build_answer_prompt methods
# Now using unified AnswerGenerator with ContextProcessor


def _pick_top_ids(observation: Dict[str, Any], query: str = "") -> List[str]:
    sel = observation.get("selection") or []
    ids: List[str] = []

    # Enhanced selection logic: try to pick the most relevant entities based on query context
    entity_candidates = []
    summary_candidates = []

    for x in sel:
        if x.get("node_type") == "entity" and x.get("node_id"):
            entity_candidates.append(x)
        elif x.get("node_type") == "summary" and x.get("node_id"):
            summary_candidates.append(x)

    priority_map = {
        "person": 5,
        "position": 4,
        "location": 3,
        "organization": 2,
        "work": 2,
        "event": 1,
        "concept": 1,
        "date": 1,
    }

    def entity_priority(item: Dict[str, Any]) -> tuple[float, float]:
        etype = (item.get("entity_type") or "").lower()
        return (priority_map.get(etype, 0), float(item.get("score", 0.0)))

    entity_candidates.sort(key=entity_priority, reverse=True)

    # If we have multiple entities, try to pick the most contextually relevant
    if len(entity_candidates) > 1:
        # Look for entities that match the query context
        query_lower = query.lower()
        relevant_entities = []

        # If query mentions movies/films, prioritize FILM entities
        if "director" in query_lower or "author" in query_lower or "writer" in query_lower:
            relevant_entities = [x for x in entity_candidates if (x.get("entity_type") or "").lower() in {"person", "position"}]
            if not relevant_entities:
                relevant_entities = [x for x in entity_candidates if (x.get("entity_type") or "").lower() == "work"]
        elif any(keyword in query_lower for keyword in ["movie", "film", "cinema"]):
            relevant_entities = [x for x in entity_candidates if (x.get("entity_type") or "").lower() == "work"]
            if not relevant_entities:
                relevant_entities = [x for x in entity_candidates if any(word in (x.get("name") or "").lower() for word in ["film", "movie"]) ]

        # If query mentions birth/death/time, prioritize DATE entities
        elif any(keyword in query_lower for keyword in ["when", "born", "birth", "died", "death", "date"]):
            relevant_entities = [x for x in entity_candidates if x.get("entity_type") == "DATE"]

        # If query mentions nationality/country, prioritize PERSON entities specifically
        elif any(keyword in query_lower for keyword in ["nationality", "country", "citizen", "where", "location"]):
            # For nationality questions, strongly prefer person entities
            relevant_entities = [x for x in entity_candidates if (x.get("entity_type") or "").lower() == "person"]

            # For comparison nationality questions, try to find entities mentioned in the query
            if "same" in query_lower or "both" in query_lower:
                import re
                # Extract potential names (simple pattern: Capitalized words)
                names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)

                name_matched = []
                for name in names:
                    name_lower = name.lower()
                    for entity in relevant_entities:
                        entity_name = (entity.get("name") or "").lower()
                        if name_lower in entity_name or any(part in entity_name for part in name_lower.split()):
                            if entity not in name_matched:
                                name_matched.append(entity)

                if name_matched:
                    relevant_entities = name_matched

            # Fallback: if no person entities, try related entities
            if not relevant_entities:
                relevant_entities = [x for x in entity_candidates if (x.get("entity_type") or "").lower() in {"work", "organization", "location"}]

        # If we found relevant entities, use them; otherwise use top-scoring entities
        if relevant_entities:
            ids.extend([x["node_id"] for x in relevant_entities[:3]])
        else:
            # Pick the top scoring entities (up to 3) to preserve coverage
            ids.extend([x["node_id"] for x in entity_candidates[:3]])
    elif entity_candidates:
        ids.append(entity_candidates[0]["node_id"])

    # If no entity found, fall back to summary with highest score
    if summary_candidates:
        summary_candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        top_summary_id = summary_candidates[0]["node_id"]
        if top_summary_id not in ids:
            ids.append(top_summary_id)

    return ids


# Removed _extract_json_first - functionality moved to AnswerGenerator
