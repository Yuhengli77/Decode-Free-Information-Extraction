from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from .config import ModelConfig

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

try:
    from transformers import AutoModel, AutoTokenizer  # type: ignore
except ImportError:
    AutoModel = None
    AutoTokenizer = None

try:
    from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType  # type: ignore
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("torch is required for modeling and training.")


def _require_transformers() -> None:
    if AutoModel is None or AutoTokenizer is None:
        raise ImportError("transformers is required for loading backbones and tokenizers.")


@dataclass
class TokenizedExample:
    input_ids: List[int]
    attention_mask: List[int]
    passage_end_positions: List[int]
    labels: List[int]
    example_id: str
    question_type: str


def _get_end_token(tokenizer: Any) -> int:
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    if tokenizer.sep_token_id is not None:
        return tokenizer.sep_token_id
    raise ValueError("Tokenizer must define either eos_token_id or sep_token_id.")


def build_query_text(question: str) -> str:
    return f"Question: {question}\n\n"


def build_passage_text(index: int, title: str, text: str) -> str:
    return f"Passage {index} | Title: {title}\n{text}\n"


def tokenize_evidence_example(
    example: Dict[str, Any],
    tokenizer: Any,
    max_length: int,
    num_passages: int = 10,
) -> TokenizedExample:
    if len(example["passages"]) != num_passages:
        raise ValueError(f"Expected {num_passages} passages, got {len(example['passages'])}.")

    end_token_id = _get_end_token(tokenizer)
    query_ids = tokenizer.encode(build_query_text(example["question"]), add_special_tokens=False)
    reserved_end_tokens = num_passages
    available_for_passages = max_length - len(query_ids) - reserved_end_tokens
    if available_for_passages < 0:
        raise ValueError(
            "Question exceeds max_length budget after reserving end tokens for passages."
        )

    passage_budget = available_for_passages // num_passages
    sequence = list(query_ids)
    passage_end_positions: List[int] = []
    for index, passage in enumerate(example["passages"], start=1):
        passage_ids = tokenizer.encode(
            build_passage_text(index, passage["title"], passage["text"]),
            add_special_tokens=False,
        )
        sequence.extend(passage_ids[:passage_budget])
        sequence.append(end_token_id)
        passage_end_positions.append(len(sequence) - 1)

    attention_mask = [1] * len(sequence)
    return TokenizedExample(
        input_ids=sequence,
        attention_mask=attention_mask,
        passage_end_positions=passage_end_positions,
        labels=list(example["labels"]),
        example_id=str(example["example_id"]),
        question_type=example["question_type"],
    )


def load_backbone_and_tokenizer(model_config: ModelConfig) -> tuple[Any, Any]:
    _require_torch()
    _require_transformers()

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.sep_token is not None:
            tokenizer.pad_token = tokenizer.sep_token
        else:
            raise ValueError("Tokenizer must define a pad, eos, or sep token.")

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model, tokenizer


def load_backbone_and_tokenizer_lora(model_config: ModelConfig) -> tuple[Any, Any]:
    """Load backbone and apply LoRA adapters (backbone params trainable via LoRA only).

    Requires ``model_config.lora`` to be set.  Falls back to fully-frozen loading
    with a warning if PEFT is not installed.
    """
    _require_torch()
    _require_transformers()

    if model_config.lora is None:
        raise ValueError("model_config.lora must be set to use load_backbone_and_tokenizer_lora.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.sep_token is not None:
            tokenizer.pad_token = tokenizer.sep_token
        else:
            raise ValueError("Tokenizer must define a pad, eos, or sep token.")

    if not _PEFT_AVAILABLE:
        import warnings
        warnings.warn(
            "peft is not installed — falling back to fully-frozen backbone. "
            "Install with: pip install peft",
            RuntimeWarning,
        )
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model, tokenizer

    lora_cfg = model_config.lora

    # Auto-detect attention projection modules if not specified.
    target_modules = lora_cfg.target_modules
    if target_modules is None:
        # Inspect the model's named modules to find q/v projection layers.
        # This covers most HuggingFace transformer architectures.
        candidate_patterns = ["q_proj", "v_proj", "query", "value", "query_key_value"]
        found = {
            name.split(".")[-1]
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and any(pat in name for pat in candidate_patterns)
        }
        target_modules = sorted(found) if found else ["q_proj", "v_proj"]

    peft_config = PeftLoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
        # FEATURE_EXTRACTION is used for encoder/embedding models and causal LMs
        # when we are NOT doing language-modelling loss (we use our own BCE head).
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    # Freeze all params first, then let PEFT unfreeze only the LoRA adapters.
    for parameter in model.parameters():
        parameter.requires_grad = False

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def get_hidden_size(backbone: Any) -> int:
    hidden_size = getattr(backbone.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(backbone.config, "d_model", None)
    if hidden_size is None:
        raise ValueError("Unable to infer hidden size from backbone config.")
    return int(hidden_size)


def tokenize_dataset(
    rows: Sequence[Dict[str, Any]],
    tokenizer: Any,
    model_config: ModelConfig,
) -> List[TokenizedExample]:
    return [
        tokenize_evidence_example(
            row,
            tokenizer=tokenizer,
            max_length=model_config.max_length,
            num_passages=model_config.num_passages,
        )
        for row in rows
    ]


def collate_batch(batch: Sequence[TokenizedExample], pad_token_id: int) -> Dict[str, Any]:
    _require_torch()

    max_seq_len = max(len(item.input_ids) for item in batch)
    input_ids = []
    attention_mask = []
    passage_end_positions = []
    labels = []
    example_ids = []
    question_types = []

    for item in batch:
        pad_length = max_seq_len - len(item.input_ids)
        input_ids.append(item.input_ids + [pad_token_id] * pad_length)
        attention_mask.append(item.attention_mask + [0] * pad_length)
        passage_end_positions.append(item.passage_end_positions)
        labels.append(item.labels)
        example_ids.append(item.example_id)
        question_types.append(item.question_type)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "passage_end_positions": torch.tensor(passage_end_positions, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "example_ids": example_ids,
        "question_types": question_types,
    }


if nn is not None:

    class HiddenStateEvidenceClassifier(nn.Module):
        def __init__(self, backbone: Any, hidden_size: int, pos_weight: float | None = None):
            super().__init__()
            self.backbone = backbone
            self.classifier = nn.Linear(hidden_size, 1)
            if pos_weight is None:
                self.register_buffer("pos_weight", torch.tensor([1.0], dtype=torch.float32))
            else:
                self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            passage_end_positions: Any,
            labels: Any | None = None,
        ) -> Dict[str, Any]:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            last_hidden_state = outputs.last_hidden_state
            hidden_size = last_hidden_state.size(-1)
            gather_index = passage_end_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
            passage_states = torch.gather(last_hidden_state, dim=1, index=gather_index)
            passage_states = passage_states.to(self.classifier.weight.dtype)
            logits = self.classifier(passage_states).squeeze(-1)

            result = {"logits": logits}
            if labels is not None:
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
                result["loss"] = loss_fn(logits, labels)
            return result

else:

    class HiddenStateEvidenceClassifier:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()
