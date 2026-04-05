from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class LoraConfig:
    """LoRA hyperparameters applied to the frozen backbone."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Modules to target; None → auto-detect (q_proj, v_proj for most transformers)
    target_modules: Optional[List[str]] = None
    # Separate LR for the backbone LoRA parameters (classifier uses TrainConfig.learning_rate)
    backbone_learning_rate: float = 2e-5
    backbone_weight_decay: float = 0.01


@dataclass
class ModelConfig:
    name: str
    model_name_or_path: str
    architecture: str
    max_length: int = 2048
    num_passages: int = 10
    trust_remote_code: bool = False
    # If set, LoRA fine-tuning is applied to the backbone instead of fully freezing it.
    lora: Optional[LoraConfig] = None


@dataclass
class DataConfig:
    dataset_name: str = "hotpot_qa"
    dataset_config_name: str = "distractor"
    cache_dir: Path = Path("artifacts/cache/hotpotqa_distractor")
    validation_ratio: float = 0.1
    max_train_examples: int = 10_000
    max_validation_examples: int = 2_000
    seed: int = 42


@dataclass
class TrainConfig:
    output_dir: Path = Path("artifacts/runs")
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 5
    patience: int = 2
    threshold_grid_size: int = 101
    num_workers: int = 0
    seed: int = 42
    gradient_accumulation_steps: int = 1


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    models: Dict[str, ModelConfig] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "ExperimentConfig":
        models = {
            "causal": ModelConfig(
                name="causal",
                model_name_or_path="Qwen/Qwen3-0.6B",
                architecture="causal",
            ),
            "bidirectional": ModelConfig(
                name="bidirectional",
                model_name_or_path="Qwen/Qwen3-Embedding-0.6B",
                architecture="bidirectional",
            ),
            "causal_lora": ModelConfig(
                name="causal_lora",
                model_name_or_path="Qwen/Qwen3-0.6B",
                architecture="causal",
                lora=LoraConfig(),
            ),
            "bidirectional_lora": ModelConfig(
                name="bidirectional_lora",
                model_name_or_path="Qwen/Qwen3-Embedding-0.6B",
                architecture="bidirectional",
                lora=LoraConfig(),
            ),
        }
        return cls(models=models)