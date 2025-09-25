import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ah_rag.agent import GraphEnvironment, AHRAG_Agent, run_agent_once
from ah_rag.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="AH-RAG Agent CLI (Task 2.2 V1)")
    parser.add_argument("query", help="Seed query for the agent loop")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--llm", action="store_true", help="Enable LLM-driven decision if API available")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log-level", default="normal")
    parser.add_argument("--redact", action="store_true")
    parser.add_argument("--session", default=None)
    args = parser.parse_args()

    cfg = load_config()
    log_level = args.log_level or cfg.get("logging", {}).get("log_level", "normal")
    redact = bool(args.redact if args.redact is not None else cfg.get("logging", {}).get("redact", True))
    use_llm = bool(args.llm or cfg.get("agent", {}).get("use_llm", False))
    env = GraphEnvironment(graph_dir="graph", debug=args.debug, session_id=args.session or None, log_level=log_level, redact=redact)
    agent = AHRAG_Agent(env, use_llm=use_llm)
    obs, summary = run_agent_once(env, agent, seed_query=args.query, steps=args.steps)
    print(json.dumps({"final_observation": obs, "summary": summary, "session_path": env.session_path}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


