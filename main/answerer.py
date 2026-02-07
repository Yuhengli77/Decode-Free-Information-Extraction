import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI


DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-flash"

SYSTEM_PROMPT = (
    "You are a careful QA assistant.\n"
    "Answer the user question using only the provided extracted webpage markdown documents.\n"
    "If information is missing, say you cannot determine the answer from provided evidence.\n"
    "Output JSON only with schema: {\"answer\": string}."
)


def _resolve_api_key(api_key: Optional[str]) -> Optional[str]:
    if api_key:
        return api_key
    for key_name in ("QWEN_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(key_name)
        if value:
            return value
    return None


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


def _normalize_extract_results(
    extract_results: Sequence[Dict[str, Any]],
    *,
    max_docs: int,
    max_chars_per_doc: int,
) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for idx, item in enumerate(extract_results[:max_docs], start=1):
        markdown = str(item.get("markdown", "")).strip()
        if not markdown:
            continue

        doc_id = str(item.get("doc_id") or f"doc_{idx}")
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        normalized.append(
            {
                "doc_id": doc_id,
                "title": title,
                "url": url,
                "markdown": markdown[:max_chars_per_doc],
            }
        )

    return normalized


def _build_documents_prompt(documents: Sequence[Dict[str, str]]) -> str:
    chunks: List[str] = []
    for doc in documents:
        parts = [f"[{doc['doc_id']}]"]
        if doc["title"]:
            parts.append(f"Title: {doc['title']}")
        if doc["url"]:
            parts.append(f"URL: {doc['url']}")
        parts.append("Markdown:")
        parts.append(doc["markdown"])
        chunks.append("\n".join(parts))
    return "\n\n".join(chunks)


def answer_query_from_extract_results(
    query: str,
    extract_results: Sequence[Dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.0,
    max_docs: int = 8,
    max_chars_per_doc: int = 6000,
    client: Optional[OpenAI] = None,
) -> Dict[str, Any]:
    if not query.strip():
        raise ValueError("query must be non-empty.")
    if not isinstance(extract_results, list):
        raise ValueError("extract_results must be a list of document dicts.")
    if not extract_results:
        raise ValueError("extract_results must be non-empty.")

    docs = _normalize_extract_results(
        extract_results,
        max_docs=max_docs,
        max_chars_per_doc=max_chars_per_doc,
    )
    if not docs:
        raise ValueError("No valid markdown content found in extract_results.")

    resolved_key = _resolve_api_key(api_key)
    if client is None:
        if not resolved_key:
            raise ValueError(
                "Missing API key. Set QWEN_API_KEY (or DASHSCOPE_API_KEY / OPENAI_API_KEY)."
            )
        client = OpenAI(api_key=resolved_key, base_url=base_url)

    user_prompt = (
        f"Question:\n{query}\n\n"
        "Extracted documents:\n"
        f"{_build_documents_prompt(docs)}\n\n"
        "Return JSON with key `answer`."
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
    answer = str(parsed.get("answer", "")).strip()

    return {
        "answer": answer,
        "documents_used": len(docs),
    }
