import html
import re
from typing import List
from bs4 import BeautifulSoup
from markdownify import markdownify as md


def html_to_markdown(html_text: str) -> str:
    if not html_text or not html_text.strip():
        return ""

    soup = BeautifulSoup(html_text, "html.parser")
    for tag_name in ("script", "style", "noscript"):
        for tag in soup.find_all(tag_name):
            tag.decompose()

    markdown = md(
        str(soup),
        heading_style="ATX",
        bullets="-",
        strip=["script", "style", "noscript"],
    )
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()
