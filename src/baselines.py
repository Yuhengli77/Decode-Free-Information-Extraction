"""Baseline scoring functions and evaluation harness.

Provides dual-tower (embedding similarity) and cross-encoder baselines,
both generic and HotpotQA in-domain variants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
from tqdm.auto import tqdm

from .modeling import build_passage_text
from .train_eval import search_best_threshold, summarize_prediction_records
from .utils import ensure_dir, save_json


def score_baseline(
    split_rows: Sequence[Dict[str, Any]],
    score_fn: Callable[[str, List[Dict[str, Any]]], List[float]],
    desc: str = "Scoring",
) -> List[Dict[str, Any]]:
    records = []
    for ex in tqdm(split_rows, desc=desc, leave=False):
        question = ex["question"]
        scores = score_fn(question, ex["passages"])
        records.append({
            "example_id": str(ex["example_id"]),
            "question_type": ex["question_type"],
            "labels": list(ex["labels"]),
            "probabilities": scores,
        })
    return records


def run_baseline(
    name: str,
    score_fn: Callable[[str, List[Dict[str, Any]]], List[float]],
    dataset_splits: Dict[str, List[Dict[str, Any]]],
    output_dir: Path | str,
) -> Dict[str, Any]:
    run_dir = ensure_dir(Path(output_dir) / name)

    val_records = score_baseline(dataset_splits["validation"], score_fn, desc=f"{name} validation")
    test_records = score_baseline(dataset_splits["test"], score_fn, desc=f"{name} test")

    flat_labels = [l for r in val_records for l in r["labels"]]
    flat_probs = [p for r in val_records for p in r["probabilities"]]
    best = search_best_threshold(flat_labels, flat_probs)

    test_metrics = summarize_prediction_records(test_records, threshold=best["threshold"])
    save_json(test_records, run_dir / "test_predictions.json")
    save_json(test_metrics, run_dir / "test_metrics_tuned_threshold.json")

    print(f"[{name}] threshold={best['threshold']:.2f}, test F1={test_metrics['overall']['f1']:.4f}")
    return {"records": test_records, "tuned_threshold": test_metrics}


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def make_dual_tower_score(model: Any) -> Callable:
    def score(question: str, passages: List[Dict[str, Any]]) -> List[float]:
        q_emb = model.encode(question, normalize_embeddings=True)
        p_texts = [f"{p['title']}: {p['text']}" for p in passages]
        p_embs = model.encode(p_texts, normalize_embeddings=True)
        return (p_embs @ q_emb).tolist()
    return score


def make_cross_encoder_score(model: Any) -> Callable:
    def score(question: str, passages: List[Dict[str, Any]]) -> List[float]:
        pairs = [(question, f"{p['title']}: {p['text']}") for p in passages]
        scores = model.predict(pairs).tolist()
        return [1 / (1 + np.exp(-s)) for s in scores]
    return score


def make_modernbert_dual_tower_score(model: Any) -> Callable:
    def score(question: str, passages: List[Dict[str, Any]]) -> List[float]:
        p_texts = [f"{p['title']}: {p['text']}" for p in passages]
        if hasattr(model, "encode_query"):
            q_emb = model.encode_query(question, normalize_embeddings=True)
        else:
            q_emb = model.encode(question, normalize_embeddings=True)
        if hasattr(model, "encode_document"):
            p_embs = model.encode_document(p_texts, normalize_embeddings=True)
        else:
            p_embs = model.encode(p_texts, normalize_embeddings=True)
        return (p_embs @ q_emb).tolist()
    return score
