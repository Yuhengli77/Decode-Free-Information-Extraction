import unittest

from src.modeling import tokenize_evidence_example


class DummyTokenizer:
    eos_token_id = 99
    sep_token_id = None

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [idx + 1 for idx, _ in enumerate(text.split())]


class ModelingTest(unittest.TestCase):
    def test_tokenization_marks_all_passage_end_positions(self) -> None:
        tokenizer = DummyTokenizer()
        example = {
            "example_id": "sample-1",
            "question": "What is the capital of France?",
            "question_type": "bridge",
            "labels": [1] + [0] * 9,
            "passages": [
                {"title": f"title-{idx}", "text": "some text for testing"}
                for idx in range(10)
            ],
        }

        tokenized = tokenize_evidence_example(example, tokenizer=tokenizer, max_length=120)
        self.assertEqual(len(tokenized.passage_end_positions), 10)
        self.assertLessEqual(len(tokenized.input_ids), 120)
        for position in tokenized.passage_end_positions:
            self.assertEqual(tokenized.input_ids[position], tokenizer.eos_token_id)


if __name__ == "__main__":
    unittest.main()
