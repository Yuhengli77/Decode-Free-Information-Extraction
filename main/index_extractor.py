import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI


DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-flash"

SYSTEM_PROMPT = (
    "You are an information extraction model.\n"
    "Task: given a user query and indexed HTML blocks, return the most relevant block id intervals.\n"
    "Rules:\n"
    "1) Output JSON only, with this schema: {\"block_intervals\": [[start_id, end_id], ...]}.\n"
    "2) Use closed intervals. If a single block is selected, use [id, id].\n"
    "3) Select only ids that appear in the input blocks.\n"
    "4) Do not output explanations.\n"
)


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        return json.loads(code_block.group(1))

    brace_block = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_block:
        return json.loads(brace_block.group(0))

    raise ValueError(f"Cannot parse JSON from model response: {text}")


def _to_lines(indexed_blocks: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for block in indexed_blocks:
        block_id = int(block["id"])
        block_text = re.sub(r"^\s*\[\d+\]\s*", "", str(block["text"]))
        lines.append(f"[{block_id}] {block_text}")
    return "\n".join(lines)


def _resolve_api_key(api_key: Optional[str]) -> Optional[str]:
    if api_key:
        return api_key
    for key_name in ("QWEN_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(key_name)
        if value:
            return value
    return None


def _expand_intervals_to_ids(
    intervals: Sequence[Sequence[Any]],
    available_ids: Sequence[int],
) -> List[int]:
    available_set = {int(block_id) for block_id in available_ids}
    selected_ids: List[int] = []
    seen = set()

    for interval in intervals:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            continue
        try:
            start = int(interval[0])
            end = int(interval[1])
        except (TypeError, ValueError):
            continue

        if start > end:
            start, end = end, start

        for block_id in range(start, end + 1):
            if block_id in available_set and block_id not in seen:
                selected_ids.append(block_id)
                seen.add(block_id)

    return selected_ids


def predict_relevant_block_ids(
    query: str,
    indexed_blocks: Sequence[Dict[str, Any]],
    *,
    top_k: int = 0,  # max number of returned ids; 0 means no limit
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.0,
    client: Optional[OpenAI] = None,
) -> List[int]:
    if not query.strip():
        raise ValueError("query must be non-empty.")
    if not indexed_blocks:
        return []

    available_ids = sorted({int(block["id"]) for block in indexed_blocks})
    if not available_ids:
        return []

    resolved_key = _resolve_api_key(api_key)
    if client is None:
        if not resolved_key:
            raise ValueError(
                "Missing API key. Set QWEN_API_KEY (or DASHSCOPE_API_KEY / OPENAI_API_KEY)."
            )
        client = OpenAI(api_key=resolved_key, base_url=base_url)

    user_prompt = (
        f"Query:\n{query}\n\n"
        # f"Top-k blocks:\n{top_k}\n\n"
        "Indexed blocks:\n"
        f"{_to_lines(indexed_blocks)}\n\n"
        "Return JSON with key `block_intervals`."
    )

    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    message = completion.choices[0].message.content or ""
    parsed = _extract_json_object(message)
    raw_intervals = parsed.get("block_intervals")
    if not isinstance(raw_intervals, list):
        raise ValueError(f"Invalid response schema: {parsed}")

    # Debug:
    print("Raw intervals:", raw_intervals)
    
    selected_ids = _expand_intervals_to_ids(raw_intervals, available_ids)

    if top_k > 0:
        selected_ids = selected_ids[:top_k]

    return selected_ids
