import re
from typing import Any, Dict, List, Sequence


def reconstruct_html_from_block_ids(
    selected_ids: Sequence[int],
    indexed_blocks: Sequence[Dict[str, Any]],
    *,
    keep_page_order: bool = True,
    wrap_container: bool = True,
) -> str:
    if not selected_ids:
        return ""

    id_to_text: Dict[int, str] = {}
    page_order: List[int] = []
    for block in indexed_blocks:
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

    fragment = "\n".join(chunks)
    if not wrap_container or not fragment:
        return fragment

    return f'<div class="extracted-evidence">\n{fragment}\n</div>'
