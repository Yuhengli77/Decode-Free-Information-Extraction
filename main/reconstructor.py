import re
from typing import Any, Dict, List, Sequence


def reconstruct_html_from_block_ids(
    selected_ids: Sequence[int],
    page_payload: Dict[str, Any],
    *,
    keep_page_order: bool = True,
    wrap_container: bool = True,
) -> Dict[str, Any]:
    if not isinstance(page_payload, dict):
        raise ValueError(
            "page_payload must be a dict with keys: "
            "`url`, `title`, `indexed_blocks`."
        )

    page_url = str(page_payload.get("url") or "").strip()
    page_title = str(page_payload.get("title") or "").strip()
    indexed_blocks = page_payload.get("indexed_blocks")
    if not isinstance(indexed_blocks, list):
        raise ValueError("`indexed_blocks` must be a list.")

    id_to_text: Dict[int, str] = {}
    page_order: List[int] = []
    for block in indexed_blocks:
        if not isinstance(block, dict):
            raise ValueError("Each indexed block must be a dict with `id` and `text`.")
        if "id" not in block or "text" not in block:
            raise ValueError("Each indexed block must include keys `id` and `text`.")
        block_id = int(block["id"])
        id_to_text[block_id] = str(block["text"])
        page_order.append(block_id)

    unique_ids: List[int] = []
    seen = set()
    for value in selected_ids:
        block_id = int(value)
        if block_id in id_to_text and block_id not in seen:
            unique_ids.append(block_id)
            seen.add(block_id)

    if keep_page_order:
        selected_set = set(unique_ids)
        ordered_ids = [block_id for block_id in page_order if block_id in selected_set]
    else:
        ordered_ids = unique_ids

    chunks: List[str] = []
    for block_id in ordered_ids:
        html = re.sub(r"^\s*\[\d+\]\s*", "", id_to_text[block_id]).strip()
        if html:
            chunks.append(html)

    body_fragment = "\n".join(chunks)
    if wrap_container and body_fragment:
        body_fragment = f'<div class="extracted-evidence">\n{body_fragment}\n</div>'

    head = f"<title>{page_title}</title>" if page_title else ""
    html_doc = (
        "<!doctype html>\n"
        "<html>\n"
        "<head>\n"
        f"{head}\n"
        "</head>\n"
        "<body>\n"
        f"{body_fragment}\n"
        "</body>\n"
        "</html>"
    )

    return {
        "url": page_url,
        "title": page_title,
        "selected_ids": ordered_ids,
        "html": html_doc,
    }
