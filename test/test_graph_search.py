import unittest

from ah_rag.graph.hierarchical_graph import HierarchicalGraph


class TestGraphSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.hg = HierarchicalGraph.load("graph")

    def test_search_returns_scored_results(self) -> None:
        results = self.hg.search("Tim Burton", top_k=3)
        self.assertGreater(len(results), 0)
        for item in results:
            self.assertIn(item["node_type"], {"entity", "summary", "hyperedge"})
            self.assertGreaterEqual(item["score"], 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
