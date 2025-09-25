import unittest

from ah_rag.graph.hierarchical_graph import HierarchicalGraph
from ah_rag.agent.environment import GraphEnvironment
from ah_rag.answer.context_processor import ContextProcessor
from ah_rag.answer.generator import AnswerGenerator


class TestMiniPipeline(unittest.TestCase):
    def test_context_pipeline_works(self) -> None:
        hg = HierarchicalGraph.load("graph")
        env = GraphEnvironment(graph_dir="graph", logging_enabled=False)
        obs, _ = env.reset(seed_query="Tim Burton", top_k=3)
        evidence = {
            "summaries": [{"node_id": item["node_id"]} for item in obs.get("selection", []) if item.get("node_type") == "summary"],
            "entities": [{"node_id": item["node_id"]} for item in obs.get("selection", []) if item.get("node_type") == "entity"],
        }
        cp = ContextProcessor()
        context = cp.build_context(evidence, hg, token_budget=800, config={"enable_kept_spans": True})
        self.assertIn("Evidence Skeleton", context["context_text"])
        self.assertTrue(context.get("used_nodes"))

        gen = AnswerGenerator()
        result = gen.generate("Who is Tim Burton?", context, {"use_llm": False})
        self.assertTrue(result.get("citations"))
        self.assertIn("answer", result)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
