"""Visualization utilities for experiment results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


METHODS: List[Tuple[str, str]] = [
    ("Dual-Tower", "dual_tower"),
    ("ModernBERT Dual-Tower", "modernbert_dual_tower"),
    ("Cross-Encoder", "cross_encoder"),
    ("HotpotQA Cross-Encoder", "hotpotqa_cross_encoder"),
    ("Causal (frozen)", "causal"),
    ("Bidirectional (frozen)", "bidirectional"),
    ("Causal + LoRA", "causal_lora"),
    ("Bidirectional + LoRA", "bidirectional_lora"),
]

COLORS = [
    "#8da0cb",  # Dual-Tower
    "#4c78a8",  # ModernBERT Dual-Tower
    "#fc8d62",  # Cross-Encoder
    "#f58518",  # HotpotQA Cross-Encoder
    "#a6d854",  # Causal frozen
    "#e78ac3",  # Bidirectional frozen
    "#66c2a5",  # Causal LoRA
    "#ffd92f",  # Bidirectional LoRA
]


def load_all_metrics(runs_dir: Path) -> Dict[str, Dict]:
    metrics = {}
    for display_name, run_name in METHODS:
        path = runs_dir / run_name / "test_metrics_tuned_threshold.json"
        if path.exists():
            metrics[display_name] = json.loads(path.read_text())
    return metrics


def plot_f1_comparison(runs_dir: Path, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-muted")
    metrics = load_all_metrics(runs_dir)

    categories = ["Overall", "Bridge", "Comparison"]
    fig, ax = plt.subplots(figsize=(15, 5))
    x = np.arange(len(categories))
    num_methods = len(metrics)
    width = min(0.8 / num_methods, 0.16)
    offsets = (np.arange(num_methods) - (num_methods - 1) / 2) * width

    for i, (name, m) in enumerate(metrics.items()):
        vals = [
            m["overall"]["f1"],
            m["by_question_type"]["bridge"]["f1"],
            m["by_question_type"]["comparison"]["f1"],
        ]
        ax.bar(x + offsets[i], vals, width, label=name, color=COLORS[i])

    ax.set_ylabel("F1 Score")
    ax.set_title("F1 by Method and Question Type")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=8, ncol=2)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_training_curves(runs_dir: Path, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-muted")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    curve_keys = [
        ("causal", "Causal (frozen)", COLORS[4], "-"),
        ("causal_lora", "Causal + LoRA", COLORS[6], "--"),
        ("bidirectional", "Bidir (frozen)", COLORS[5], "-"),
        ("bidirectional_lora", "Bidir + LoRA", COLORS[7], "--"),
    ]
    for model_key, label, color, ls in curve_keys:
        history = json.loads(
            (runs_dir / model_key / "training_history.json").read_text()
        )["history"]
        epochs = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        val_f1s = [h["validation"]["overall"]["f1"] for h in history]
        axes[0].plot(epochs, train_losses, marker="o", label=label, color=color, linestyle=ls)
        axes[1].plot(epochs, val_f1s, marker="o", label=label, color=color, linestyle=ls)

    for ax, title, ylabel in zip(axes, ["Training Loss", "Validation F1"], ["Loss", "F1"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=True, fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def print_results_table(runs_dir: Path) -> None:
    rows = []
    for display_name, run_name in METHODS:
        path = runs_dir / run_name / "test_metrics_tuned_threshold.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        rows.append((
            display_name,
            m["overall"]["f1"],
            m["by_question_type"]["bridge"]["f1"],
            m["by_question_type"]["comparison"]["f1"],
        ))

    best_overall = max(row[1] for row in rows)

    print("| Method | Overall F1 | Bridge F1 | Comparison F1 |")
    print("|--------|-----------|-----------|---------------|")
    for name, overall, bridge, comparison in rows:
        if overall == best_overall:
            print(f"| **{name}** | **{overall:.3f}** | **{bridge:.3f}** | **{comparison:.3f}** |")
        else:
            print(f"| {name} | {overall:.3f} | {bridge:.3f} | {comparison:.3f} |")


if __name__ == "__main__":
    runs_dir = Path("artifacts/runs")
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    print_results_table(runs_dir)
    plot_f1_comparison(runs_dir, figures_dir / "f1_comparison.png")
    plot_training_curves(runs_dir, figures_dir / "training_curves.png")
