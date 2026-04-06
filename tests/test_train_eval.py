import unittest

from src.train_eval import search_best_threshold, summarize_prediction_records


class TrainEvalTest(unittest.TestCase):
    def test_search_best_threshold_returns_f1(self) -> None:
        result = search_best_threshold(
            labels=[1, 0, 1, 0],
            probabilities=[0.9, 0.8, 0.4, 0.2],
            grid_size=11,
        )
        self.assertIn("threshold", result)
        self.assertGreaterEqual(result["f1"], 0.0)

    def test_summary_groups_by_question_type(self) -> None:
        records = [
            {
                "example_id": "1",
                "question_type": "bridge",
                "labels": [1, 0],
                "probabilities": [0.8, 0.1],
            },
            {
                "example_id": "2",
                "question_type": "comparison",
                "labels": [0, 1],
                "probabilities": [0.6, 0.9],
            },
        ]
        summary = summarize_prediction_records(records, threshold=0.5)
        self.assertEqual(summary["num_examples"], 2)
        self.assertIn("bridge", summary["by_question_type"])
        self.assertIn("comparison", summary["by_question_type"])


if __name__ == "__main__":
    unittest.main()
