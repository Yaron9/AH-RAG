#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ah_rag.utils.config import load_config
from ah_rag.eval.answer_eval import AnswerEvaluator
from ah_rag.answer.generator import AnswerGenerator


def load_dataset(name: str, limit: int | None = None) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        raise RuntimeError("Please install 'datasets' to run benchmarks.")
    if name.lower() == "hotpotqa":
        ds = load_dataset("hotpot_qa", "distractor")['validation']
        items: List[Dict[str, Any]] = []
        for idx, i in enumerate(ds):
            _id = str(i.get("_id") or i.get("id") or idx)
            q = i.get("question") or ""
            ans = i.get("answer")
            golds = []
            if isinstance(ans, list):
                golds = [x for x in ans if x]
            elif isinstance(ans, str) and ans:
                golds = [ans]
            items.append({"id": _id, "question": q, "answers": golds, "context": i.get("context")})
    elif name.lower() == "triviaqa":
        ds = load_dataset("trivia_qa", "rc")['validation']
        items = []
        for i in ds:
            _id = str(i.get("question_id") or i.get("id") or i.get("_id") or "")
            q = i.get("question") or ""
            a = i.get("answer") or {}
            golds: List[str] = []
            for k in ("value", "normalized_value"):
                v = a.get(k)
                if isinstance(v, str) and v:
                    golds.append(v)
            for k in ("aliases", "normalized_aliases"):
                vs = a.get(k) or []
                for v in vs:
                    if isinstance(v, str) and v:
                        golds.append(v)
            # deduplicate while preserving order
            seen = set()
            dedup: List[str] = []
            for g in golds:
                if g not in seen:
                    seen.add(g)
                    dedup.append(g)
            items.append({"id": _id, "question": q, "answers": dedup})
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    if limit is not None:
        items = items[: int(limit)]
    return items


def build_hotpotqa_graph(context_data: Dict[str, Any]) -> str:
    """Build temporary knowledge graph from HotpotQA context using existing demo_cli.py"""
    import tempfile
    import subprocess

    # Combine all context sentences into one document
    context_text = ""
    for title, sentences in zip(context_data['title'], context_data['sentences']):
        context_text += f"\n\n=== {title} ===\n"
        for sentence in sentences:
            context_text += sentence + " "

    # Write context to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(context_text.strip())
        temp_file = f.name

    try:
        # Use existing demo_cli.py to build the graph
        print(f"Building knowledge graph from HotpotQA context using demo_cli.py...")
        exe = sys.executable or 'python3'
        result = subprocess.run([
            exe, 'scripts/demo_cli.py', temp_file
        ], cwd=os.path.join(os.path.dirname(__file__), '..'),
           capture_output=True, text=True)

        if result.returncode != 0:
            print(f"demo_cli.py failed: {result.stderr}")
            raise RuntimeError(f"Failed to build graph: {result.stderr}")

        print("Graph built successfully")
        return "graph"  # demo_cli.py builds to the standard 'graph' directory

    finally:
        # Cleanup temp text file
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def run_system(system: str, query: str, cfg: Dict[str, Any], corpus: str, dataset: str, context_data: Dict[str, Any] = None) -> Dict[str, Any]:
    if system == "ah_rag":
        # Use existing inference CLI path via module import if available
        from ah_rag.agent.environment import GraphEnvironment
        from ah_rag.agent.agent import AHRAG_Agent
        from ah_rag.agent.rl_agent import RLPolicyAgent  # type: ignore
        from ah_rag.agent.inference import InferenceEngine
        # If HotpotQA context provided, build temporary graph
        if context_data and dataset.lower() == "hotpotqa":
            graph_dir = build_hotpotqa_graph(context_data)
        else:
            graph_dir = "graph" if corpus == "graph" else f"graph_datasets/{dataset}_distractor"
        env = GraphEnvironment(graph_dir)
        # Align with unified LLM config in config/ah_rag.yaml
        llm_modules = (cfg.get("llm") or {}).get("modules", {})
        agent_rl_cfg = (cfg.get("rl") or {}).get("inference", {})
        use_ppo = bool(agent_rl_cfg.get("use_ppo", False))
        if use_ppo:
            model_path = agent_rl_cfg.get("ppo_model_path", "artifacts/rl/ppo_policy.pt")
            agent = RLPolicyAgent(env, model_path=model_path)
        else:
            agent_llm_cfg = llm_modules.get("agent_decision", {})
            agent_use_llm = bool(agent_llm_cfg.get("enabled", False))
            _agent_model = agent_llm_cfg.get("model", (cfg.get("llm") or {}).get("default_model", "deepseek-chat"))
            agent = AHRAG_Agent(env, use_llm=agent_use_llm)
        # InferenceEngine now reads generation settings from unified config internally
        engine = InferenceEngine(env, agent)
        return engine.run_inference(query, steps=int(cfg.get("inference", {}).get("steps", 4)))
    elif system == "naive":
        from ah_rag.graph.hierarchical_graph import HierarchicalGraph  # type: ignore
        from baselines.naive_rag import NaiveRAG
        # If HotpotQA context provided, build temporary graph
        if context_data and dataset.lower() == "hotpotqa":
            graph_dir = build_hotpotqa_graph(context_data)
        else:
            graph_dir = "graph" if corpus == "graph" else f"graph_datasets/{dataset}_distractor"
        hg = HierarchicalGraph.load(graph_dir)
        gen = AnswerGenerator()
        top_k = int(cfg.get("evaluation", {}).get("naive_rag_top_k", 5))
        return NaiveRAG(hg, gen).run(query, top_k=top_k, gen_cfg=cfg.get("answer", {}))
    else:
        raise ValueError(f"Unknown system: {system}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run standardized benchmark for AH-RAG and baselines")
    ap.add_argument("--dataset", type=str, required=True, help="Dataset name: hotpotqa|triviaqa")
    ap.add_argument("--system", type=str, default="both", help="ah_rag|naive|both")
    ap.add_argument("--corpus", type=str, default="graph", help="graph|dataset (dataset uses graph_datasets/<dataset>_distractor)")
    ap.add_argument("--limit", type=int, default=10, help="Limit number of samples")
    ap.add_argument("--judge-sample", type=float, default=None, help="Override qualitative judge sample ratio")
    ap.add_argument("--out", type=str, default=None, help="Output report path (JSON). Markdown table prints to stdout")
    args = ap.parse_args()

    cfg = load_config()
    data = load_dataset(args.dataset, limit=args.limit)
    systems = ["ah_rag", "naive"] if args.system == "both" else [args.system]

    results: List[Dict[str, Any]] = []

    # determine qualitative sampling
    sample_ratio = args.judge_sample if args.judge_sample is not None else float((cfg.get("evaluation", {}).get("judge", {}) or {}).get("sample_ratio", 0.2))

    for item in data:
        q = item["question"]
        golds = item.get("answers") or []
        # Get context data for HotpotQA
        context_data = item.get("context") if args.dataset.lower() == "hotpotqa" else None
        if context_data:
            print(f"🔍 Using HotpotQA context with {len(context_data['title'])} passages")

        for sys_name in systems:
            # Create system-specific evaluator
            evaluator = AnswerEvaluator(system_type=sys_name)

            ans = run_system(sys_name, q, cfg, corpus=args.corpus, dataset=args.dataset, context_data=context_data)
            pred_text = ans.get("answer", "")

            # Prepare answer object for new unified evaluation
            answer_obj = {
                'query': q,
                'answer': pred_text,
                'rationale': ans.get('rationale', ''),
                'citations': ans.get('citations', []),
                'session_data': ans.get('session_data', {}),
                'gold_answers': golds,
                'evidence': ans.get('evidence', {}),  # 添加证据信息
                'context': ans.get('context', {}),    # 添加上下文信息
                'retrieved_nodes': ans.get('retrieved_nodes', [])  # 添加检索节点
            }

            # Use new unified evaluation framework
            import random
            random.seed(int(cfg.get("evaluation", {}).get("seed", 42)))

            # Configure LLM judge based on sampling
            eval_cfg = cfg.copy()
            use_llm_judge = random.random() < float(sample_ratio)
            eval_cfg.setdefault('evaluation', {}).setdefault('judge', {})['use_llm'] = use_llm_judge

            # Get the knowledge graph for evaluation
            from ah_rag.graph.hierarchical_graph import HierarchicalGraph
            graph_dir = "graph" if args.corpus == "graph" else f"graph_datasets/{args.dataset}_distractor"
            if context_data and args.dataset.lower() == "hotpotqa":
                graph_dir = "graph"  # Use the temporary graph built from HotpotQA context
            hg = HierarchicalGraph.load(graph_dir)

            # Run unified evaluation
            unified_results = evaluator.evaluate(answer_obj, graph=hg, config=eval_cfg)

            # Extract scores for compatibility
            scores = unified_results['scores']
            diagnosis = unified_results['details']['diagnosis']

            results.append({
                "id": item.get("id"),
                "system": sys_name,
                "dataset": args.dataset,

                # Legacy metrics (backward compatibility)
                "f1": scores.get("f1", 0.0),
                "em": scores.get("em", 0.0),
                "judge_overall": scores.get("judge_overall", 0.0),

                # New Claude.md framework metrics
                "contextual_recall": scores.get("contextual_recall", 0.0),
                "contextual_relevancy": scores.get("contextual_relevancy", 0.0),
                "contextual_precision": scores.get("contextual_precision", 0.0),
                "faithfulness": scores.get("faithfulness", 0.0),
                "answer_relevancy": scores.get("answer_relevancy", 0.0),
                "overall_score": scores.get("overall", 0.0),

                # Diagnosis information
                "primary_issue": diagnosis["primary_issue"],
                "diagnosis_reason": diagnosis["reason"],
                "diagnosis_confidence": diagnosis["confidence"]
            })

    # aggregate
    import pandas as pd  # type: ignore
    df = pd.DataFrame(results)

    # Aggregate both legacy and new metrics
    agg_metrics = {
        # Legacy metrics
        "f1": "mean", "em": "mean", "judge_overall": "mean",
        # New Claude.md framework metrics
        "contextual_recall": "mean", "contextual_relevancy": "mean", "contextual_precision": "mean",
        "faithfulness": "mean", "answer_relevancy": "mean", "overall_score": "mean",
        "diagnosis_confidence": "mean"
    }

    agg = df.groupby(["dataset", "system"], as_index=False).agg(agg_metrics)

    # Calculate issue distribution for each system from original data
    issue_dist = df.groupby(["dataset", "system"])["primary_issue"].apply(
        lambda x: "/".join([f"{issue}({count})" for issue, count in x.value_counts().head(2).items()])
    ).to_dict()

    # print markdown table with enhanced metrics
    def to_markdown_agg(agg_df: Any) -> str:
        # Core metrics table
        headers = ["dataset", "system", "overall_score", "f1", "em", "contextual_recall", "faithfulness", "primary_issues"]
        lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]

        for _, row in agg_df.iterrows():
            key = (row['dataset'], row['system'])
            issues = issue_dist.get(key, "none")
            lines.append(
                f"| {row['dataset']} | {row['system']} | {row['overall_score']:.3f} | "
                f"{row['f1']:.3f} | {row['em']:.3f} | {row['contextual_recall']:.3f} | "
                f"{row['faithfulness']:.3f} | {issues} |"
            )
        return "\n".join(lines)

    print("🏆 Unified RAG Evaluation Results (Claude.md Framework)")
    print("=" * 70)
    md = to_markdown_agg(agg)
    print(md)
    print()

    # Additional diagnostics summary
    print("🩺 Diagnosis Summary:")
    print("-" * 30)
    diagnosis_summary = df.groupby(["system", "primary_issue"]).size().unstack(fill_value=0)
    print(diagnosis_summary)
    print()
    report = {"items": results, "aggregate": agg.to_dict(orient="records")}
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
