import unittest

from ah_rag.agent.environment import GraphEnvironment


class TestGraphEnvironment(unittest.TestCase):
    def test_reset_with_seed_query(self) -> None:
        env = GraphEnvironment(graph_dir="graph", logging_enabled=False)
        obs, info = env.reset(seed_query="Tim Burton", top_k=3)
        self.assertEqual(info.get("action"), "semantic_anchor")
        self.assertEqual(info.get("returned"), len(obs.get("selection", [])))
        self.assertGreater(len(obs.get("seeds", [])), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
