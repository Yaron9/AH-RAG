import argparse
import json
import os
import sys

# Ensure 'src' is on the path so 'ah_rag' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ah_rag.agent import GraphEnvironment
from ah_rag.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="AH-RAG Environment CLI (V1/V2/V3)")
    parser.add_argument("query", nargs="?", default=None, help="Seed query for semantic_anchor")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--expand", choices=["children", "parents", "related"], default=None)
    parser.add_argument("--select", nargs="*", help="Commit selection by node ids")
    parser.add_argument("--filters", nargs="*", help="Set filters: judge>=x, conf>=y, type=entity|summary")
    parser.add_argument("--weights", nargs="*", help="Set weights: alpha=,beta=,gamma=,delta=,member_top_m=,top_k=")
    parser.add_argument("--debug", action="store_true", help="Enable extended observation diagnostics")
    parser.add_argument("--log-level", default="normal", help="Logging level: off|normal|debug|trace")
    parser.add_argument("--redact", action="store_true", help="Redact sensitive fields in logs")
    parser.add_argument("--session", default=None, help="Custom session id")
    parser.add_argument("--end", action="store_true", help="End episode and emit summary path")
    args = parser.parse_args()

    cfg = load_config()
    log_level = args.log_level or cfg.get("logging", {}).get("log_level", "normal")
    redact = bool(args.redact if args.redact is not None else cfg.get("logging", {}).get("redact", True))
    env = GraphEnvironment(graph_dir="graph", debug=args.debug, session_id=args.session or None, log_level=log_level, redact=redact)

    # Apply dynamic filters
    if args.filters:
        jf = None
        cf = None
        tf = None
        for tok in args.filters:
            if tok.startswith("judge>="):
                jf = float(tok.split("=", 1)[1])
            elif tok.startswith("conf>="):
                cf = float(tok.split("=", 1)[1])
            elif tok.startswith("type="):
                tf = tok.split("=", 1)[1].split(",")
        env.set_filters(judge_overall_min=jf, confidence_min=cf, type_filter=tf)

    # Apply dynamic weights
    if args.weights:
        w: dict = {}
        for tok in args.weights:
            k, v = tok.split("=", 1)
            if k in {"alpha", "beta", "gamma", "delta"}:
                w[k] = float(v)
            elif k in {"member_top_m", "top_k"}:
                w[k] = int(v)
        env.set_search_weights(**w)

    if args.query:
        obs, info = env.reset(seed_query=args.query, top_k=args.top_k)
        print(json.dumps({"observation": obs, "info": info}, ensure_ascii=False, indent=2))
    else:
        obs, info = env.reset()
        print(json.dumps({"observation": obs, "info": info}, ensure_ascii=False, indent=2))

    # Commit selection if provided
    if args.select:
        obs, info = env.commit_selection(args.select)
        print(json.dumps({"observation": obs, "info": info}, ensure_ascii=False, indent=2))

    # Expansion
    if args.expand and obs.get("selection"):
        target_id = obs["selection"][0]["node_id"]
        if args.expand == "children":
            obs2, info2 = env.expand_children([target_id])
        elif args.expand == "parents":
            obs2, info2 = env.expand_parents([target_id])
        else:
            obs2, info2 = env.expand_related([target_id])
        print(json.dumps({"observation": obs2, "info": info2}, ensure_ascii=False, indent=2))

    if args.end:
        summary = env.end_episode()
        print(json.dumps({"summary": summary, "session_path": env.session_path}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


