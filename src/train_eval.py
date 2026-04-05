from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .config import TrainConfig
from .utils import ensure_dir, progress, save_json

try:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    AdamW = None
    DataLoader = None


def _require_torch() -> None:
    if torch is None or AdamW is None or DataLoader is None:
        raise ImportError("torch is required for training and evaluation.")


def flatten_labels(rows: Sequence[Dict[str, Any]]) -> List[int]:
    return [label for row in rows for label in row["labels"]]


def estimate_pos_weight(rows: Sequence[Dict[str, Any]]) -> float:
    flattened = flatten_labels(rows)
    positives = sum(flattened)
    negatives = len(flattened) - positives
    if positives == 0:
        return 1.0
    return negatives / positives


def precision_recall_f1(labels: Sequence[int], predictions: Sequence[int]) -> Dict[str, float]:
    tp = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 1)
    fp = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 1)
    fn = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def search_best_threshold(
    labels: Sequence[int],
    probabilities: Sequence[float],
    grid_size: int = 101,
) -> Dict[str, float]:
    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": -1.0}
    for step in range(grid_size):
        threshold = step / (grid_size - 1)
        predictions = [1 if prob >= threshold else 0 for prob in probabilities]
        metrics = precision_recall_f1(labels, predictions)
        candidate = {"threshold": threshold, **metrics}
        if candidate["f1"] > best["f1"]:
            best = candidate
        elif math.isclose(candidate["f1"], best["f1"]) and abs(threshold - 0.5) < abs(
            best["threshold"] - 0.5
        ):
            best = candidate
    return best


def summarize_prediction_records(
    records: Sequence[Dict[str, Any]],
    threshold: float,
) -> Dict[str, Any]:
    flat_labels = [label for record in records for label in record["labels"]]
    flat_predictions = [
        1 if probability >= threshold else 0
        for record in records
        for probability in record["probabilities"]
    ]
    overall = precision_recall_f1(flat_labels, flat_predictions)

    by_question_type: Dict[str, Dict[str, Any]] = {}
    question_types = sorted({record["question_type"] for record in records})
    for question_type in question_types:
        subset = [record for record in records if record["question_type"] == question_type]
        subset_labels = [label for record in subset for label in record["labels"]]
        subset_predictions = [
            1 if probability >= threshold else 0
            for record in subset
            for probability in record["probabilities"]
        ]
        by_question_type[question_type] = precision_recall_f1(subset_labels, subset_predictions)

    return {
        "threshold": threshold,
        "overall": overall,
        "by_question_type": by_question_type,
        "num_examples": len(records),
    }


def predict(model: Any, dataloader: Any, device: str) -> Dict[str, Any]:
    _require_torch()

    model.eval()
    records = []
    losses = []
    with torch.no_grad():
        for batch in progress(dataloader, desc="Predicting", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            passage_end_positions = batch["passage_end_positions"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                passage_end_positions=passage_end_positions,
                labels=labels,
            )
            logits = outputs["logits"].detach().cpu()
            probabilities = torch.sigmoid(logits)
            if "loss" in outputs:
                losses.append(outputs["loss"].item())

            for row_idx in range(logits.size(0)):
                records.append(
                    {
                        "example_id": batch["example_ids"][row_idx],
                        "question_type": batch["question_types"][row_idx],
                        "labels": batch["labels"][row_idx].int().tolist(),
                        "logits": logits[row_idx].tolist(),
                        "probabilities": probabilities[row_idx].tolist(),
                    }
                )

    return {"records": records, "loss": sum(losses) / len(losses) if losses else None}


def train_classifier(
    model: Any,
    train_loader: Any,
    validation_loader: Any,
    train_config: TrainConfig,
    output_dir: Path | str,
    device: str,
) -> Dict[str, Any]:
    _require_torch()

    run_dir = ensure_dir(output_dir)
    optimizer = AdamW(
        model.classifier.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    best_state = None
    best_epoch = -1
    best_summary = None
    patience_left = train_config.patience
    history = []

    model.to(device)
    for epoch in range(1, train_config.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in progress(
            train_loader,
            desc=f"Training epoch {epoch}/{train_config.epochs}",
            leave=False,
        ):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                passage_end_positions=batch["passage_end_positions"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        validation_prediction = predict(model, validation_loader, device)
        flat_labels = [label for row in validation_prediction["records"] for label in row["labels"]]
        flat_probs = [
            prob for row in validation_prediction["records"] for prob in row["probabilities"]
        ]
        threshold_result = search_best_threshold(
            flat_labels,
            flat_probs,
            grid_size=train_config.threshold_grid_size,
        )
        summary = summarize_prediction_records(
            validation_prediction["records"],
            threshold=threshold_result["threshold"],
        )
        summary["loss"] = validation_prediction["loss"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": sum(epoch_losses) / len(epoch_losses),
                "validation": summary,
            }
        )

        if best_summary is None or summary["overall"]["f1"] > best_summary["overall"]["f1"]:
            best_summary = summary
            best_epoch = epoch
            patience_left = train_config.patience
            best_state = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "threshold": threshold_result["threshold"],
                "validation_summary": summary,
            }
            torch.save(best_state, run_dir / "best_model.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None or best_summary is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state["model_state_dict"])
    save_json(
        {
            "best_epoch": best_epoch,
            "history": history,
            "best_validation": best_summary,
            "best_threshold": best_state["threshold"],
        },
        run_dir / "training_history.json",
    )

    return {
        "best_epoch": best_epoch,
        "best_threshold": best_state["threshold"],
        "best_validation": best_summary,
        "checkpoint_path": str(run_dir / "best_model.pt"),
        "history_path": str(run_dir / "training_history.json"),
    }


def train_lora(
    model: Any,
    train_loader: Any,
    validation_loader: Any,
    train_config: "TrainConfig",
    lora_config: Any,  # src.config.LoraConfig
    output_dir: "Path | str",
    device: str,
) -> Dict[str, Any]:
    """Training loop for LoRA fine-tuning.

    Uses two parameter groups with separate learning rates:
      - LoRA adapter params (backbone)  → ``lora_config.backbone_learning_rate``
      - Classifier head params          → ``train_config.learning_rate``

    Gradient clipping (max norm 1.0) is applied to prevent large updates through
    the now-trainable backbone adapters.
    """
    _require_torch()

    run_dir = ensure_dir(output_dir)

    # Separate LoRA backbone params from the linear classifier head.
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    classifier_params = list(model.classifier.parameters())

    optimizer = AdamW(
        [
            {
                "params": lora_params,
                "lr": lora_config.backbone_learning_rate,
                "weight_decay": lora_config.backbone_weight_decay,
            },
            {
                "params": classifier_params,
                "lr": train_config.learning_rate,
                "weight_decay": train_config.weight_decay,
            },
        ]
    )

    best_state = None
    best_epoch = -1
    best_summary = None
    patience_left = train_config.patience
    history = []

    accum_steps = getattr(train_config, "gradient_accumulation_steps", 1)

    model.to(device)
    for epoch in range(1, train_config.epochs + 1):
        model.train()
        epoch_losses = []
        optimizer.zero_grad()
        for step, batch in enumerate(progress(
            train_loader,
            desc=f"LoRA epoch {epoch}/{train_config.epochs}",
            leave=False,
        )):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                passage_end_positions=batch["passage_end_positions"].to(device),
                labels=batch["labels"].to(device),
            )
            # Scale loss so gradients are averaged over accumulation steps.
            loss = outputs["loss"] / accum_steps
            loss.backward()
            epoch_losses.append(loss.item() * accum_steps)  # log unscaled

            if (step + 1) % accum_steps == 0:
                # Clip gradients through LoRA adapters to stabilise training.
                torch.nn.utils.clip_grad_norm_(
                    lora_params + classifier_params, max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()

        # Handle any remaining steps not covered by the last full accumulation.
        if (step + 1) % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                lora_params + classifier_params, max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()

        validation_prediction = predict(model, validation_loader, device)
        flat_labels = [label for row in validation_prediction["records"] for label in row["labels"]]
        flat_probs = [
            prob for row in validation_prediction["records"] for prob in row["probabilities"]
        ]
        threshold_result = search_best_threshold(
            flat_labels,
            flat_probs,
            grid_size=train_config.threshold_grid_size,
        )
        summary = summarize_prediction_records(
            validation_prediction["records"],
            threshold=threshold_result["threshold"],
        )
        summary["loss"] = validation_prediction["loss"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": sum(epoch_losses) / len(epoch_losses),
                "validation": summary,
            }
        )

        if best_summary is None or summary["overall"]["f1"] > best_summary["overall"]["f1"]:
            best_summary = summary
            best_epoch = epoch
            patience_left = train_config.patience

            # Save only the LoRA adapter weights (a few MB) via PEFT's
            # save_pretrained, plus the classifier head separately.
            # Saving model.state_dict() would include all frozen base weights
            # (~1.2 GB) and its keys are incompatible with plain load_state_dict
            # on a PEFT-wrapped model.
            adapter_dir = run_dir / "best_lora_adapter"
            model.backbone.save_pretrained(str(adapter_dir))
            classifier_state = {
                "weight": model.classifier.weight.detach().cpu(),
                "bias": model.classifier.bias.detach().cpu(),
            }
            torch.save(
                {
                    "classifier_state": classifier_state,
                    "epoch": epoch,
                    "threshold": threshold_result["threshold"],
                    "validation_summary": summary,
                },
                run_dir / "best_classifier.pt",
            )
            # Keep an in-memory snapshot for restoring the best epoch at the end.
            best_state = {
                "classifier_state": {
                    "weight": classifier_state["weight"].clone(),
                    "bias": classifier_state["bias"].clone(),
                },
                "epoch": epoch,
                "threshold": threshold_result["threshold"],
                "validation_summary": summary,
            }
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None or best_summary is None:
        raise RuntimeError("LoRA training did not produce a valid checkpoint.")

    # Restore best classifier weights into the live model.
    model.classifier.weight.data.copy_(best_state["classifier_state"]["weight"].to(device))
    model.classifier.bias.data.copy_(best_state["classifier_state"]["bias"].to(device))
    # Note: adapter weights on disk in best_lora_adapter/ already correspond to the
    # best epoch because save_pretrained was called at that checkpoint.

    save_json(
        {
            "best_epoch": best_epoch,
            "history": history,
            "best_validation": best_summary,
            "best_threshold": best_state["threshold"],
        },
        run_dir / "training_history.json",
    )

    return {
        "best_epoch": best_epoch,
        "best_threshold": best_state["threshold"],
        "best_validation": best_summary,
        # Two artefacts to reload the model later:
        #   adapter weights  → best_lora_adapter/  (PeftModel.from_pretrained)
        #   classifier head  → best_classifier.pt  (torch.load)
        "adapter_dir": str(run_dir / "best_lora_adapter"),
        "classifier_path": str(run_dir / "best_classifier.pt"),
        "history_path": str(run_dir / "training_history.json"),
    }



def evaluate_and_save(
    model: Any,
    dataloader: Any,
    device: str,
    output_dir: Path | str,
    split_name: str,
    tuned_threshold: float,
) -> Dict[str, Any]:
    run_dir = ensure_dir(output_dir)
    prediction = predict(model, dataloader, device)
    tuned_metrics = summarize_prediction_records(prediction["records"], threshold=tuned_threshold)

    save_json(prediction["records"], run_dir / f"{split_name}_predictions.json")
    save_json(tuned_metrics, run_dir / f"{split_name}_metrics_tuned_threshold.json")
    return {
        "records": prediction["records"],
        "tuned_threshold": tuned_metrics,
        "prediction_path": str(run_dir / f"{split_name}_predictions.json"),
    }