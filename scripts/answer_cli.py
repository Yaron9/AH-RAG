#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ah_rag.utils.config import load_config
from ah_rag.answer.context_processor import ContextProcessor
from ah_rag.answer.generator import AnswerGenerator


def main() -> None:
    ap = argparse.ArgumentParser(description="Standalone Answer Generation CLI")
    ap.add_argument("query", type=str, help="User query text")
    ap.add_argument("--evidence", type=str, default=None, help="Path to evidence JSON with summaries/entities list")
    ap.add_argument("--graph", type=str, default=None, help="Optional path to load graph if needed (not used if evidence has full texts)")
    ap.add_argument("--out", type=str, default=None, help="Output file path (answer.json). Print to stdout if omitted")
    args = ap.parse_args()

    cfg = load_config()
    answer_cfg = cfg.get("answer") or {}

    # load evidence JSON
    evidence = {"summaries": [], "entities": []}
    if args.evidence and os.path.exists(args.evidence):
        with open(args.evidence, "r", encoding="utf-8") as f:
            evidence = json.load(f)

    # Graph handle is expected via env_cli/agent_cli; for this standalone CLI we assume evidence carries node_ids
    # and the ContextProcessor will use minimal fields from graph if available. For now, require env-driven run.
    try:
        from ah_rag.graph.hierarchical_graph import HierarchicalGraph  # type: ignore
        hg = HierarchicalGraph.load("graph")  # default location
    except Exception:
        hg = None
        if not evidence:
            print("Evidence missing and graph not loadable.", file=sys.stderr)
            sys.exit(2)

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
    context = cp.build_context(evidence, hg, token_budget, ctx_cfg) if hg else {"context_text": "", "used_nodes": [], "stats": {}}

    gen = AnswerGenerator()
    gen_cfg = {
        "use_llm": answer_cfg.get("use_llm", False),
        "model": answer_cfg.get("model", "moonshot-v1-8k"),
        "temperature": answer_cfg.get("temperature", 0.1),
        "max_retries": answer_cfg.get("max_retries", 2),
    }
    out = gen.generate(args.query, context, gen_cfg)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


