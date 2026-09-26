"""Implementations of the ScrappingAgent page tools.

The tools use PageContext as their only state. They return Markdown strings
that can be passed directly back to an Ollama chat tool-call loop.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from .page_context import PageContext
from ..scraper.scraper import scrape


DEFAULT_TABLE_ROW_LIMIT: int = 10
DEFAULT_SEARCH_RESULT_LIMIT: int = 10
DEFAULT_SUMMARY_MAX_WORDS: int = 100
DEFAULT_LINK_RESULT_LIMIT: int = 20
DEFAULT_CODE_RESULT_LIMIT: int = 20
SEARCH_SNIPPET_RADIUS: int = 220
MIN_SUMMARY_SENTENCE_LENGTH: int = 30

SECTION_DIVIDER: str = "---"
MARKDOWN_TABLE_SEPARATOR_PATTERN: str = r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$"
MARKDOWN_LINK_PATTERN: str = r"\[([^\]]+)\]\(([^)]+)\)"
FENCED_CODE_PATTERN: str = r"```(?P<language>[\w+.-]*)\n(?P<code>.*?)(?:\n```|\Z)"


class PageNotLoadedError(RuntimeError):
    """Raised when a tool that needs page data is called before scraping."""


def _require_page(context: PageContext) -> None:
    """Raise an error if no current page is available.

    Args:
        context: Current agent page context.

    Raises:
        PageNotLoadedError: If the context does not contain a page.
    """
    if not context.is_loaded:
        raise PageNotLoadedError("No page is loaded. Call scrape_page first.")


def _normalise_text(value: str) -> str:
    """Normalize whitespace for matching and concise output."""
    return " ".join(value.split())


def _is_external_url(url: str, page_url: str) -> bool:
    """Return whether a link points to a different domain."""
    absolute_url = urljoin(page_url, url)
    return urlparse(absolute_url).netloc != urlparse(page_url).netloc


def _get_tables(context: PageContext) -> List[List[List[str]]]:
    """Return parsed tables from structured page content."""
    tables = context.structured.get("tables", [])
    return tables if isinstance(tables, list) else []


def _get_links(context: PageContext) -> List[Dict[str, str]]:
    """Return normalized links from structured page content."""
    raw_links = context.structured.get("links", [])
    links: List[Dict[str, str]] = []
    if not isinstance(raw_links, list):
        return links

    for link in raw_links:
        if not isinstance(link, dict):
            continue
        text = link.get("text")
        href = link.get("href")
        if isinstance(text, str) and isinstance(href, str) and text and href:
            links.append({"text": _normalise_text(text), "href": href})
    return links


def scrape_page(
    context: PageContext,
    url: str,
    force_playwright: bool = False,
) -> str:
    """Scrape a page and replace the current PageContext.

    Args:
        context: Mutable page state owned by the agent.
        url: Absolute URL to scrape.
        force_playwright: Whether to force JavaScript rendering.

    Returns:
        Compact page overview in Markdown.
    """
    result = scrape(
        url=url,
        force_playwright=force_playwright,
        format_markdown=True,
    )

    if not result.get("success", False):
        error = result.get("error", "Unknown scraping error")
        return f"# Scrape failed\n\nURL: {url}\n\nError: {error}"

    structured = result.get("structured")
    markdown = result.get("markdown")
    if not isinstance(structured, dict) or not isinstance(markdown, str):
        return "# Scrape failed\n\nThe scraper returned an invalid page payload."

    context.url = url
    context.markdown = markdown
    context.structured = structured
    context.fetch_method = str(result.get("method", ""))
    context.raw_html = None

    metadata = structured.get("metadata", {})
    title = metadata.get("title") if isinstance(metadata, dict) else None
    headings = structured.get("headings", [])
    links = _get_links(context)
    tables = _get_tables(context)
    body_text = structured.get("body_text", "")
    content_size = len(body_text) if isinstance(body_text, str) else 0

    overview_lines = [
        "# Page loaded",
        "",
        f"- URL: {context.url}",
        f"- Title: {title or 'Unknown'}",
        f"- Fetch method: {context.fetch_method}",
        f"- Text characters: {content_size}",
        f"- Headings: {len(headings) if isinstance(headings, list) else 0}",
        f"- Tables: {len(tables)}",
        f"- Links: {len(links)}",
        "",
        "Use get_page_section, search_within_page, filter_table_rows, "
        "filter_links_by_category, or extract_code_blocks to inspect only the needed data.",
    ]
    return "\n".join(overview_lines)


def filter_table_rows(
    context: PageContext,
    keyword: str,
    column: Optional[str] = None,
    limit: int = DEFAULT_TABLE_ROW_LIMIT,
) -> str:
    """Find rows in extracted tables that match a keyword.

    Args:
        context: Current page state.
        keyword: Text to find in a table row.
        column: Optional header name restricting the matching column.
        limit: Maximum number of matching rows returned.

    Returns:
        Matching rows rendered as Markdown tables.
    """
    _require_page(context)
    query = keyword.casefold().strip()
    if not query:
        return "# Table search\n\nProvide a non-empty keyword."

    max_rows = max(1, limit)
    matches: List[str] = []
    for table_index, table in enumerate(_get_tables(context), start=1):
        if not table:
            continue
        header = table[0]
        normalized_headers = [_normalise_text(cell).casefold() for cell in header]
        column_index: Optional[int] = None

        if column:
            normalized_column = column.casefold().strip()
            try:
                column_index = normalized_headers.index(normalized_column)
            except ValueError:
                continue

        matching_rows: List[List[str]] = []
        for row in table[1:]:
            searchable_cells = row
            if column_index is not None:
                searchable_cells = [row[column_index]] if column_index < len(row) else []
            if any(query in _normalise_text(cell).casefold() for cell in searchable_cells):
                matching_rows.append(row)
                if len(matching_rows) >= max_rows:
                    break

        if matching_rows:
            width = max(len(header), *(len(row) for row in matching_rows))
            padded_header = header + [""] * (width - len(header))
            matches.append(f"## Table {table_index}")
            matches.append("")
            matches.append("| " + " | ".join(padded_header) + " |")
            matches.append("| " + " | ".join(["---"] * width) + " |")
            for row in matching_rows:
                padded_row = row + [""] * (width - len(row))
                matches.append("| " + " | ".join(padded_row) + " |")
            matches.append("")

    if not matches:
        suffix = f" in column '{column}'" if column else ""
        return f"# Table search\n\nNo table rows found for '{keyword}'{suffix}."

    return "\n".join([f"# Table matches: {keyword}", "", *matches])


def search_within_page(
    context: PageContext,
    pattern: str,
    case_sensitive: bool = False,
) -> str:
    """Search the current page and return contextual snippets.

    Args:
        context: Current page state.
        pattern: Plain text or regular-expression pattern.
        case_sensitive: Whether matching should preserve case.

    Returns:
        Matching snippets and their positions.
    """
    _require_page(context)
    if not pattern.strip():
        return "# Search\n\nProvide a non-empty pattern."

    source = context.markdown
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        matches = list(re.finditer(pattern, source, flags))
    except re.error as error:
        return f"# Search error\n\nInvalid regular expression: {error}"

    if not matches:
        return f"# Search: {pattern}\n\nNo matches found."

    lines = [f"# Search: {pattern}", "", f"Matches found: {len(matches)}", ""]
    for index, match in enumerate(matches[:DEFAULT_SEARCH_RESULT_LIMIT], start=1):
        start = max(0, match.start() - SEARCH_SNIPPET_RADIUS)
        end = min(len(source), match.end() + SEARCH_SNIPPET_RADIUS)
        snippet = _normalise_text(source[start:end])
        lines.extend([f"## Result {index}", "", snippet, ""])

    if len(matches) > DEFAULT_SEARCH_RESULT_LIMIT:
        lines.append(f"Showing the first {DEFAULT_SEARCH_RESULT_LIMIT} results.")

    return "\n".join(lines)


def get_page_section(
    context: PageContext,
    heading_title: str,
    include_subsections: bool = True,
) -> str:
    """Retrieve Markdown located under a heading.

    Args:
        context: Current page state.
        heading_title: Exact heading text, excluding Markdown hashes.
        include_subsections: Whether nested headings belong to the result.

    Returns:
        The selected section or a clear not-found response.
    """
    _require_page(context)
    target = heading_title.casefold().strip()
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
    matches = list(heading_pattern.finditer(context.markdown))

    selected_index: Optional[int] = None
    selected_level = 0
    for index, match in enumerate(matches):
        text = match.group(2).strip().casefold()
        if text == target:
            selected_index = index
            selected_level = len(match.group(1))
            break

    if selected_index is None:
        return f"# Section not found\n\nNo exact heading named '{heading_title}' was found."

    start_match = matches[selected_index]
    end_position = len(context.markdown)
    for next_match in matches[selected_index + 1:]:
        next_level = len(next_match.group(1))
        if (include_subsections and next_level <= selected_level) or (
            not include_subsections and next_level <= selected_level
        ):
            end_position = next_match.start()
            break
        if not include_subsections and next_level > selected_level:
            end_position = next_match.start()
            break

    return context.markdown[start_match.start():end_position].strip()


def filter_links_by_category(
    context: PageContext,
    target_topic: str,
    external_only: bool = False,
) -> str:
    """Return links relevant to a topic.

    Args:
        context: Current page state.
        target_topic: Keyword searched in anchor text and href.
        external_only: Whether to retain only cross-domain links.

    Returns:
        Relevant Markdown links.
    """
    _require_page(context)
    query = target_topic.casefold().strip()
    if not query:
        return "# Link filter\n\nProvide a non-empty topic."

    matched_links: List[Dict[str, str]] = []
    for link in _get_links(context):
        haystack = f"{link['text']} {link['href']}".casefold()
        if query not in haystack:
            continue
        if external_only and not _is_external_url(link["href"], context.url):
            continue
        matched_links.append(link)
        if len(matched_links) >= DEFAULT_LINK_RESULT_LIMIT:
            break

    if not matched_links:
        return f"# Link filter: {target_topic}\n\nNo matching links found."

    lines = [f"# Link filter: {target_topic}", ""]
    for link in matched_links:
        absolute_url = urljoin(context.url, link["href"])
        lines.append(f"- [{link['text']}]({absolute_url})")
    return "\n".join(lines)


def extract_code_blocks(
    context: PageContext,
    language: Optional[str] = None,
    include_explanation: bool = False,
) -> str:
    """Extract fenced Markdown code and HTML pre/code blocks.

    Args:
        context: Current page state.
        language: Optional case-insensitive programming-language filter.
        include_explanation: Whether to include preceding nearby text.

    Returns:
        Markdown fenced code blocks.
    """
    _require_page(context)
    requested_language = language.casefold().strip() if language else None
    blocks: List[Dict[str, str]] = []

    for match in re.finditer(FENCED_CODE_PATTERN, context.markdown, re.DOTALL):
        block_language = match.group("language").strip() or "text"
        code = match.group("code").strip()
        if code:
            blocks.append({"language": block_language, "code": code, "start": str(match.start())})

    if context.raw_html:
        soup = BeautifulSoup(context.raw_html, "html.parser")
        for pre in soup.find_all("pre"):
            code_tag = pre.find("code")
            code = pre.get_text("\n", strip=True)
            if not code:
                continue
            if code_tag:
                classes = code_tag.get("class") or []
            else:
                classes = []
            class_text = " ".join(classes) if isinstance(classes, list) else str(classes)
            class_text = " ".join(classes) if isinstance(classes, list) else str(classes)
            language_match = re.search(r"(?:language-|lang-)([\w+.-]+)", class_text)
            block_language = language_match.group(1) if language_match else "text"
            blocks.append({"language": block_language, "code": code, "start": "0"})

    seen: set[tuple[str, str]] = set()
    unique_blocks: List[Dict[str, str]] = []
    for block in blocks:
        identity = (block["language"], block["code"])
        if identity not in seen:
            seen.add(identity)
            unique_blocks.append(block)

    if requested_language:
        unique_blocks = [
            block for block in unique_blocks
            if block["language"].casefold() == requested_language
        ]

    if not unique_blocks:
        language_suffix = f" for language '{language}'" if language else ""
        return f"# Code blocks\n\nNo code blocks found{language_suffix}."

    lines = ["# Code blocks", ""]
    for index, block in enumerate(unique_blocks[:DEFAULT_CODE_RESULT_LIMIT], start=1):
        if include_explanation:
            position = int(block["start"])
            before = context.markdown[max(0, position - SEARCH_SNIPPET_RADIUS):position]
            explanation = _normalise_text(before).split(SECTION_DIVIDER)[-1].strip()
            if explanation:
                lines.extend([f"Context: {explanation}", ""])
        lines.extend([
            f"## Block {index} ({block['language']})",
            "",
            f"```{block['language']}",
            block["code"],
            "```",
            "",
        ])

    return "\n".join(lines)


def get_summary(
    context: PageContext,
    max_length: int = DEFAULT_SUMMARY_MAX_WORDS,
) -> str:
    """Create a deterministic extractive summary of the current page.

    This tool does not call an LLM. It selects the first substantive
    extracted paragraphs, so it is fast and cheap.

    Args:
        context: Current page state.
        max_length: Maximum approximate number of words.

    Returns:
        Brief Markdown summary.
    """
    _require_page(context)
    word_limit = max(20, max_length)
    paragraphs = context.structured.get("paragraphs", [])
    headings = context.structured.get("headings", [])
    selected_parts: List[str] = []
    used_words = 0

    if isinstance(headings, list):
        heading_texts = [
            item.get("text", "") for item in headings
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        if heading_texts:
            selected_parts.append("Topics: " + "; ".join(heading_texts[:5]))
            used_words += len(selected_parts[-1].split())

    if isinstance(paragraphs, list):
        for paragraph in paragraphs:
            if not isinstance(paragraph, str):
                continue
            normalized = _normalise_text(paragraph)
            if len(normalized) < MIN_SUMMARY_SENTENCE_LENGTH:
                continue
            words = normalized.split()
            remaining = word_limit - used_words
            if remaining <= 0:
                break
            selected_parts.append(" ".join(words[:remaining]))
            used_words += min(len(words), remaining)
            if used_words >= word_limit:
                break

    if not selected_parts:
        body_text = context.structured.get("body_text", "")
        if isinstance(body_text, str) and body_text:
            selected_parts.append(" ".join(_normalise_text(body_text).split()[:word_limit]))

    if not selected_parts:
        return "# Page summary\n\nNo textual content is available."

    return "# Page summary\n\n" + "\n\n".join(selected_parts)


def compare_entities(
    context: PageContext,
    entities: Sequence[str],
    metrics: Optional[Sequence[str]] = None,
) -> str:
    """Compare entities by locating their rows in extracted data tables.

    Args:
        context: Current page state.
        entities: Names or symbols to locate in table rows.
        metrics: Optional column names to include in addition to entity columns.

    Returns:
        A compact Markdown comparison table.
    """
    _require_page(context)
    normalized_entities = [entity.strip() for entity in entities if entity.strip()]
    if len(normalized_entities) < 2:
        return "# Entity comparison\n\nProvide at least two non-empty entities."

    requested_metrics = [metric.casefold().strip() for metric in metrics or [] if metric.strip()]
    found_rows: List[Dict[str, str]] = []

    for table in _get_tables(context):
        if not table:
            continue
        header = table[0]
        normalized_header = [_normalise_text(cell).casefold() for cell in header]
        selected_indices: List[int] = list(range(len(header)))
        if requested_metrics:
            selected_indices = [
                index for index, name in enumerate(normalized_header)
                if name in requested_metrics or any(metric in name for metric in requested_metrics)
            ]
            name_indices = [
                index for index, name in enumerate(normalized_header)
                if name in {"name", "symbol", "asset", "token"}
            ]
            selected_indices = list(dict.fromkeys(name_indices + selected_indices))
            if not selected_indices:
                selected_indices = list(range(len(header)))

        for entity in normalized_entities:
            entity_key = entity.casefold()
            for row in table[1:]:
                row_text = " ".join(_normalise_text(cell) for cell in row).casefold()
                if entity_key not in row_text:
                    continue
                record: Dict[str, str] = {"Entity": entity}
                for index in selected_indices:
                    if index < len(header):
                        record[header[index] or f"Column {index + 1}"] = row[index] if index < len(row) else ""
                found_rows.append(record)
                break

    if not found_rows:
        return "# Entity comparison\n\nNone of the requested entities were found in page tables."

    columns: List[str] = ["Entity"]
    for record in found_rows:
        for key in record:
            if key not in columns:
                columns.append(key)

    lines = ["# Entity comparison", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for record in found_rows:
        lines.append("| " + " | ".join(record.get(column, "") for column in columns) + " |")
    return "\n".join(lines)


def build_tool_registry(context: PageContext) -> Dict[str, Any]:
    """Build a name-to-callable registry bound to one PageContext.

    Args:
        context: Mutable context shared by all tool calls for one agent run.

    Returns:
        Tool registry suitable for a tool-call dispatcher.
    """
    return {
        "scrape_page": lambda **kwargs: scrape_page(context, **kwargs),
        "filter_table_rows": lambda **kwargs: filter_table_rows(context, **kwargs),
        "search_within_page": lambda **kwargs: search_within_page(context, **kwargs),
        "get_page_section": lambda **kwargs: get_page_section(context, **kwargs),
        "filter_links_by_category": lambda **kwargs: filter_links_by_category(context, **kwargs),
        "extract_code_blocks": lambda **kwargs: extract_code_blocks(context, **kwargs),
        "get_summary": lambda **kwargs: get_summary(context, **kwargs),
        "compare_entities": lambda **kwargs: compare_entities(context, **kwargs),
    }
