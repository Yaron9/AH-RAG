import unittest

from ah_rag.answer.generator import AnswerGenerator


class TestAnswerGenerator(unittest.TestCase):
    def test_fallback_structure_without_llm(self) -> None:
        context = {
            "context_text": "- [ent:4983994569] (entity) Tim Burton :: American director",
            "used_nodes": ["ent:4983994569"],
        }
        gen = AnswerGenerator()
        result = gen.generate("Who is Tim Burton?", context, {"use_llm": False})
        self.assertIn("answer", result)
        self.assertIn("rationale", result)
        self.assertIn("citations", result)
        self.assertTrue(result["citations"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
