from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .config import DataConfig
from .utils import ensure_dir, load_jsonl, progress, save_json, save_jsonl

try:
    from datasets import DatasetDict, load_dataset  # type: ignore
except ImportError:
    DatasetDict = Any  # type: ignore
    load_dataset = None


PreparedExample = Dict[str, Any]


def _require_datasets() -> None:
    if load_dataset is None:
        raise ImportError(
            "datasets is required for loading HotpotQA. Install it with `pip install datasets`."
        )


def _normalize_context(raw_context: Any) -> List[Tuple[str, Sequence[str]]]:
    if isinstance(raw_context, dict):
        titles = raw_context.get("title", [])
        sentences = raw_context.get("sentences", [])
        return list(zip(titles, sentences))
    return [(item[0], item[1]) for item in raw_context]


def _normalize_supporting_titles(raw_supporting_facts: Any) -> List[str]:
    if isinstance(raw_supporting_facts, dict):
        return list(raw_supporting_facts.get("title", []))
    return [item[0] for item in raw_supporting_facts]


def prepare_hotpotqa_example(record: Dict[str, Any], num_passages: int = 10) -> PreparedExample | None:
    context_rows = _normalize_context(record.get("context", []))
    if len(context_rows) != num_passages:
        return None

    supporting_titles = set(_normalize_supporting_titles(record.get("supporting_facts", [])))
    passages = []
    labels = []
    for title, sentences in context_rows:
        if isinstance(sentences, str):
            text = sentences
        else:
            text = " ".join(sentence.strip() for sentence in sentences if sentence and sentence.strip())
        is_supporting = title in supporting_titles
        passages.append(
            {
                "title": title,
                "text": text,
                "is_supporting": int(is_supporting),
            }
        )
        labels.append(int(is_supporting))

    return {
        "example_id": str(record.get("id", "")),
        "question": record.get("question", "").strip(),
        "question_type": record.get("type", "unknown"),
        "passages": passages,
        "labels": labels,
    }


def _split_train_validation(train_rows: List[PreparedExample], validation_ratio: float, seed: int) -> Tuple[List[PreparedExample], List[PreparedExample]]:
    rows = list(train_rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    validation_size = max(1, int(len(rows) * validation_ratio))
    return rows[validation_size:], rows[:validation_size]


def _cap_split(rows: List[PreparedExample], limit: int | None, seed: int) -> List[PreparedExample]:
    if limit is None or limit <= 0 or len(rows) <= limit:
        return rows
    capped_rows = list(rows)
    random.Random(seed).shuffle(capped_rows)
    return capped_rows[:limit]


def _cache_paths(cache_dir: Path) -> Dict[str, Path]:
    return {
        "train": cache_dir / "train.jsonl",
        "validation": cache_dir / "validation.jsonl",
        "test": cache_dir / "test.jsonl",
        "metadata": cache_dir / "metadata.json",
    }


def load_or_prepare_hotpotqa(config: DataConfig) -> Dict[str, List[PreparedExample]]:
    cache_dir = ensure_dir(config.cache_dir)
    paths = _cache_paths(cache_dir)
    if all(path.exists() for key, path in paths.items() if key != "metadata"):
        return {
            "train": load_jsonl(paths["train"]),
            "validation": load_jsonl(paths["validation"]),
            "test": load_jsonl(paths["test"]),
        }

    _require_datasets()
    dataset = load_dataset(config.dataset_name, config.dataset_config_name, cache_dir=str(cache_dir / "hf"))
    prepared = prepare_dataset_splits(
        dataset,
        config.validation_ratio,
        config.seed,
        max_train_examples=config.max_train_examples,
        max_validation_examples=config.max_validation_examples,
    )

    for split_name, rows in prepared.items():
        save_jsonl(rows, paths[split_name])

    save_json(
        {
            "dataset_name": config.dataset_name,
            "dataset_config_name": config.dataset_config_name,
            "validation_ratio": config.validation_ratio,
            "max_train_examples": config.max_train_examples,
            "max_validation_examples": config.max_validation_examples,
            "seed": config.seed,
            "split_sizes": {name: len(rows) for name, rows in prepared.items()},
        },
        paths["metadata"],
    )
    return prepared


def prepare_dataset_splits(
    dataset: DatasetDict,
    validation_ratio: float,
    seed: int,
    max_train_examples: int | None = None,
    max_validation_examples: int | None = None,
) -> Dict[str, List[PreparedExample]]:
    prepared_by_split: Dict[str, List[PreparedExample]] = {}
    for split_name, split in dataset.items():
        rows = []
        for record in progress(split, desc=f"Preparing {split_name}", leave=False):
            prepared = prepare_hotpotqa_example(record)
            if prepared is not None:
                rows.append(prepared)
        prepared_by_split[split_name] = rows

    if "validation" in prepared_by_split and "test" not in prepared_by_split:
        train_rows, validation_rows = _split_train_validation(
            prepared_by_split["train"],
            validation_ratio=validation_ratio,
            seed=seed,
        )
        return {
            "train": _cap_split(train_rows, max_train_examples, seed),
            "validation": _cap_split(validation_rows, max_validation_examples, seed + 1),
            "test": prepared_by_split["validation"],
        }

    if all(split in prepared_by_split for split in ("train", "validation", "test")):
        return {
            "train": _cap_split(prepared_by_split["train"], max_train_examples, seed),
            "validation": _cap_split(
                prepared_by_split["validation"],
                max_validation_examples,
                seed + 1,
            ),
            "test": prepared_by_split["test"],
        }

    raise ValueError(
        "Expected dataset splits to include either train+validation or train+validation+test."
    )


def dataset_statistics(rows: Iterable[PreparedExample]) -> Dict[str, Any]:
    rows = list(rows)
    if not rows:
        return {
            "num_examples": 0,
            "num_positive_labels": 0,
            "num_negative_labels": 0,
            "positive_rate": 0.0,
            "avg_question_chars": 0.0,
            "avg_passage_chars": 0.0,
        }

    positive = sum(sum(example["labels"]) for example in rows)
    total = len(rows) * len(rows[0]["labels"])
    negative = total - positive
    avg_question_chars = sum(len(example["question"]) for example in rows) / len(rows)
    avg_passage_chars = (
        sum(len(passage["text"]) for example in rows for passage in example["passages"]) / total
    )

    return {
        "num_examples": len(rows),
        "num_positive_labels": positive,
        "num_negative_labels": negative,
        "positive_rate": positive / total if total else 0.0,
        "avg_question_chars": avg_question_chars,
        "avg_passage_chars": avg_passage_chars,
    }


def estimate_token_length_stats(
    rows: Sequence[PreparedExample],
    tokenizer: Any,
    sample_size: int = 128,
) -> Dict[str, float]:
    if not rows:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}

    limit = min(sample_size, len(rows))
    sampled = rows[:limit]
    lengths = []
    for example in sampled:
        length = len(tokenizer.encode(f"Question: {example['question']}", add_special_tokens=False))
        for idx, passage in enumerate(example["passages"], start=1):
            length += len(
                tokenizer.encode(
                    f"Passage {idx} | Title: {passage['title']}\n{passage['text']}",
                    add_special_tokens=False,
                )
            )
        lengths.append(length)
    return {"mean": sum(lengths) / len(lengths), "min": min(lengths), "max": max(lengths)}
