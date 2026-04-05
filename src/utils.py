from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:
    tqdm = None


def ensure_dir(path: Path | str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(data: Any, path: Path | str) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def save_jsonl(records: Iterable[Dict[str, Any]], path: Path | str) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    return output_path


def load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device() -> str:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def format_metrics_table(metrics_by_model: Dict[str, Dict[str, Any]]) -> str:
    header = ["model", "threshold", "precision", "recall", "f1"]
    rows = [header]
    for name, metrics in metrics_by_model.items():
        overall = metrics["overall"]
        rows.append(
            [
                name,
                f"{metrics.get('threshold', 0.5):.3f}",
                f"{overall['precision']:.4f}",
                f"{overall['recall']:.4f}",
                f"{overall['f1']:.4f}",
            ]
        )

    widths = [max(len(row[idx]) for row in rows) for idx in range(len(header))]
    formatted_rows = []
    for row in rows:
        formatted_rows.append(" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))
    return "\n".join(formatted_rows)


def progress(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)
