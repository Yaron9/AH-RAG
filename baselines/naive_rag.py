from __future__ import annotations

from typing import Any, Dict, List
import json


class NaiveRAG:
    """
    Minimal baseline: vector top‑K retrieval + direct generation with shared AnswerGenerator.
    Assumes an existing HierarchicalGraph with vector index and an AnswerGenerator.
    """

    def __init__(self, hg: Any, answer_generator: Any) -> None:
        self.hg = hg
        self.answer_generator = answer_generator

    def run(self, query: str, top_k: int = 5, gen_cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
        # use graph's hybrid search but treat results as flat top‑K evidence
        res = self.hg.search(query, top_k=top_k, return_cluster=False)
        ids: List[str] = [x.get("node_id") for x in (res or []) if x.get("node_id")]
        # build a simple context: concatenate titles/summaries
        skeleton = []
        used = []
        for nid in ids:
            d = self.hg.G.nodes.get(nid, {})
            title = d.get("title") or d.get("name") or ""
            summary = d.get("summary_text") or d.get("summary") or d.get("description") or ""
            skeleton.append(f"- [{nid}] {title} :: {summary[:200]}")
            used.append(nid)
        context = {
            "context_text": "\n".join(skeleton),
            "used_nodes": used,
            "stats": {},
        }
        return self.answer_generator.generate(query, context, gen_cfg or {})


