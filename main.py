"""Run all experiments: frozen backbones, baselines, and LoRA fine-tuning."""

from functools import partial
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from torch.utils.data import DataLoader

from src.baselines import (
    make_cross_encoder_score,
    make_dual_tower_score,
    make_modernbert_dual_tower_score,
    run_baseline,
)
from src.config import ExperimentConfig, LoraConfig
from src.data import dataset_statistics, load_or_prepare_hotpotqa
from src.modeling import (
    HiddenStateEvidenceClassifier,
    collate_batch,
    get_hidden_size,
    load_backbone_and_tokenizer,
    load_backbone_and_tokenizer_lora,
    tokenize_dataset,
)
from src.train_eval import (
    estimate_pos_weight,
    evaluate_and_save,
    train_classifier,
    train_lora,
)
from src.utils import get_device, set_seed
from src.visualize import plot_f1_comparison, plot_training_curves, print_results_table


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

config = ExperimentConfig.default()
config.data.cache_dir = Path("artifacts/cache/hotpotqa_distractor")
config.train.output_dir = Path("artifacts/runs")
config.data.max_train_examples = 10_000
config.data.max_validation_examples = 2_000
config.train.batch_size = 64
config.train.epochs = 5
config.train.patience = 2
config.models["causal"].max_length = 3072
config.models["bidirectional"].max_length = 3072

device = get_device()
set_seed(config.train.seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data():
    dataset_splits = load_or_prepare_hotpotqa(config.data)
    for split_name, rows in dataset_splits.items():
        print(split_name, dataset_statistics(rows))
    return dataset_splits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_loader(rows, tokenizer, model_config, batch_size, shuffle):
    tokenized_rows = tokenize_dataset(rows, tokenizer, model_config)
    loader = DataLoader(
        tokenized_rows,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_batch, pad_token_id=tokenizer.pad_token_id),
    )
    return tokenized_rows, loader


# ---------------------------------------------------------------------------
# Frozen backbone experiments
# ---------------------------------------------------------------------------

def run_frozen_experiment(model_key, dataset_splits):
    model_config = config.models[model_key]
    run_dir = config.train.output_dir / model_key

    backbone, tokenizer = load_backbone_and_tokenizer(model_config)

    _, train_loader = build_loader(
        dataset_splits["train"], tokenizer, model_config, config.train.batch_size, shuffle=True,
    )
    _, validation_loader = build_loader(
        dataset_splits["validation"], tokenizer, model_config, config.train.batch_size, shuffle=False,
    )
    _, test_loader = build_loader(
        dataset_splits["test"], tokenizer, model_config, config.train.batch_size, shuffle=False,
    )

    model = HiddenStateEvidenceClassifier(
        backbone=backbone,
        hidden_size=get_hidden_size(backbone),
        pos_weight=estimate_pos_weight(dataset_splits["train"]),
    )
    training_summary = train_classifier(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        train_config=config.train,
        output_dir=run_dir,
        device=device,
    )
    test_results = evaluate_and_save(
        model=model,
        dataloader=test_loader,
        device=device,
        output_dir=run_dir,
        split_name="test",
        tuned_threshold=training_summary["best_threshold"],
    )
    return {"model_config": model_config, "training": training_summary, "test": test_results}


# ---------------------------------------------------------------------------
# LoRA experiments
# ---------------------------------------------------------------------------

def run_lora_experiment(model_key, dataset_splits):
    model_config = config.models[model_key]
    assert model_config.lora is not None, f"{model_key} has no LoraConfig set."
    run_dir = config.train.output_dir / model_key

    backbone, tokenizer = load_backbone_and_tokenizer_lora(model_config)

    _, train_loader = build_loader(
        dataset_splits["train"], tokenizer, model_config, config.train.batch_size, shuffle=True,
    )
    _, validation_loader = build_loader(
        dataset_splits["validation"], tokenizer, model_config, config.train.batch_size, shuffle=False,
    )
    _, test_loader = build_loader(
        dataset_splits["test"], tokenizer, model_config, config.train.batch_size, shuffle=False,
    )

    model = HiddenStateEvidenceClassifier(
        backbone=backbone,
        hidden_size=get_hidden_size(backbone),
        pos_weight=estimate_pos_weight(dataset_splits["train"]),
    )
    training_summary = train_lora(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        train_config=config.train,
        lora_config=model_config.lora,
        output_dir=run_dir,
        device=device,
    )
    test_results = evaluate_and_save(
        model=model,
        dataloader=test_loader,
        device=device,
        output_dir=run_dir,
        split_name="test",
        tuned_threshold=training_summary["best_threshold"],
    )
    return {"model_config": model_config, "training": training_summary, "test": test_results}


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def run_all_baselines(dataset_splits):
    output_dir = config.train.output_dir

    # Generic baselines
    dual_tower = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    run_baseline("dual_tower", make_dual_tower_score(dual_tower), dataset_splits, output_dir)

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device=device)
    run_baseline("cross_encoder", make_cross_encoder_score(cross_encoder), dataset_splits, output_dir)

    # In-domain baselines
    modernbert = SentenceTransformer("hotchpotch/ModernBERT-embedding-CMNBRL", device=device)
    run_baseline("modernbert_dual_tower", make_modernbert_dual_tower_score(modernbert), dataset_splits, output_dir)

    hotpotqa_ce = CrossEncoder("OloriBern/hotpotqa-mixer-2000", device=device)
    run_baseline("hotpotqa_cross_encoder", make_cross_encoder_score(hotpotqa_ce), dataset_splits, output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dataset_splits = load_data()

    # Frozen backbone experiments
    print("\n=== Frozen Causal LM ===")
    run_frozen_experiment("causal", dataset_splits)

    print("\n=== Frozen Bidirectional ===")
    run_frozen_experiment("bidirectional", dataset_splits)

    # Baselines
    print("\n=== Baselines ===")
    run_all_baselines(dataset_splits)

    # LoRA fine-tuning
    lora_overrides = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, backbone_learning_rate=2e-5)
    config.models["causal_lora"].lora = lora_overrides
    config.models["bidirectional_lora"].lora = lora_overrides
    config.train.batch_size = 16
    config.models["causal_lora"].max_length = 2048
    config.models["bidirectional_lora"].max_length = 2048

    print("\n=== Causal LM + LoRA ===")
    run_lora_experiment("causal_lora", dataset_splits)

    print("\n=== Bidirectional + LoRA ===")
    run_lora_experiment("bidirectional_lora", dataset_splits)

    # Visualize
    print("\n=== Results ===")
    runs_dir = config.train.output_dir
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    print_results_table(runs_dir)
    plot_f1_comparison(runs_dir, figures_dir / "f1_comparison.png")
    plot_training_curves(runs_dir, figures_dir / "training_curves.png")
