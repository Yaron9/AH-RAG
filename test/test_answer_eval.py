import unittest

from ah_rag.eval.answer_eval import AnswerEvaluator


class TestAnswerEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = AnswerEvaluator()

    def test_quantitative_perfect_match(self) -> None:
        metrics = self.evaluator.evaluate_quantitative("Tim Burton", ["Tim Burton"])
        self.assertAlmostEqual(metrics["f1"], 100.0)
        self.assertAlmostEqual(metrics["em"], 100.0)

    def test_diagnosis_edge_case_when_metrics_high(self) -> None:
        diagnosis = self.evaluator.apply_diagnosis_formula({
            "faithfulness": 0.8,
            "answer_relevancy": 0.8,
            "contextual_recall": 0.9,
        })
        self.assertEqual(diagnosis["primary_issue"], "edge_case")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
