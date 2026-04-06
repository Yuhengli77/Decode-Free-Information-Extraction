import unittest

from src.data import dataset_statistics, prepare_dataset_splits, prepare_hotpotqa_example


class DataPreparationTest(unittest.TestCase):
    def test_prepare_hotpotqa_example_maps_supporting_titles(self) -> None:
        record = {
            "id": "sample-1",
            "question": "Who founded the company behind the iPhone?",
            "type": "bridge",
            "supporting_facts": {"title": ["Apple Inc.", "Steve Jobs"], "sent_id": [0, 0]},
            "context": {
                "title": [f"title-{idx}" for idx in range(8)] + ["Apple Inc.", "Steve Jobs"],
                "sentences": [["dummy sentence"] for _ in range(8)]
                + [["Apple makes phones."]]
                + [["Steve Jobs co-founded Apple."]],
            },
        }

        example = prepare_hotpotqa_example(record)
        self.assertIsNotNone(example)
        self.assertEqual(example["labels"].count(1), 2)
        self.assertEqual(example["question_type"], "bridge")

    def test_dataset_statistics_counts_labels(self) -> None:
        rows = [
            {
                "question": "q1",
                "labels": [1, 0, 0, 1],
                "passages": [{"text": "a"}, {"text": "b"}, {"text": "c"}, {"text": "d"}],
            }
        ]
        stats = dataset_statistics(rows)
        self.assertEqual(stats["num_examples"], 1)
        self.assertEqual(stats["num_positive_labels"], 2)
        self.assertEqual(stats["num_negative_labels"], 2)

    def test_prepare_dataset_splits_caps_train_and_validation(self) -> None:
        dataset = {
            "train": [self._make_record(idx) for idx in range(20)],
            "validation": [self._make_record(100 + idx) for idx in range(8)],
        }

        prepared = prepare_dataset_splits(
            dataset,
            validation_ratio=0.2,
            seed=42,
            max_train_examples=10,
            max_validation_examples=2,
        )

        self.assertEqual(len(prepared["train"]), 10)
        self.assertEqual(len(prepared["validation"]), 2)
        self.assertEqual(len(prepared["test"]), 8)

    def _make_record(self, idx: int) -> dict:
        return {
            "id": f"sample-{idx}",
            "question": f"question-{idx}",
            "type": "bridge" if idx % 2 == 0 else "comparison",
            "supporting_facts": {"title": [f"title-{idx}-0", f"title-{idx}-1"], "sent_id": [0, 0]},
            "context": {
                "title": [f"title-{idx}-{passage_idx}" for passage_idx in range(10)],
                "sentences": [[f"sentence-{idx}-{passage_idx}"] for passage_idx in range(10)],
            },
        }


if __name__ == "__main__":
    unittest.main()
