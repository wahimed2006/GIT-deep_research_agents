"""Content formatter for LLM consumption.

This module provides functions to format extracted content into
clean, structured Markdown optimized for LLM understanding.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from markdownify import markdownify


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Output limits
MAX_HEADINGS_IN_TOC: int = 15
MAX_OUTPUT_LENGTH: int = 10000
TRUNCATION_THRESHOLD_RATIO: float = 0.8

# Section headers
SECTION_TOC: str = "## Table of Contents"
SECTION_CONTENT: str = "## Content"
SECTION_LISTS: str = "## Lists"
SECTION_TABLES: str = "## Tables"
SECTION_LINKS: str = "## Related Links"

# Section dividers
SECTION_DIVIDER: str = "---"
TRUNCATION_MESSAGE: str = "[Content truncated...]"

# Metadata labels
LABEL_AUTHOR: str = "**Author:**"
LABEL_PUBLISHED: str = "**Published:**"
LABEL_DESCRIPTION: str = "**Description:**"

# List item prefix
LIST_ITEM_PREFIX: str = "- "

# Markdown table cell separator
TABLE_CELL_SEP: str = " | "
TABLE_DIVIDER_ROW: str = "---"

# Heading styles
HEADING_STYLE_ATX: str = "ATX"
HEADING_STYLE_SETEXT: str = "SETEXT"

# Default heading style
DEFAULT_HEADING_STYLE: str = HEADING_STYLE_ATX

# Markdownify options
MARKDOWNIFY_BULLETS: str = "-"
MARKDOWNIFY_STRIP_TAGS: List[str] = ["a"]


# =============================================================================
# FUNCTIONS
# =============================================================================

def format_metadata(metadata: Dict[str, Optional[str]]) -> str:
    """Format metadata as Markdown.
    
    Args:
        metadata: Dictionary with 'title', 'description', 'author', 'published_date'.
    
    Returns:
        Formatted Markdown string.
    """
    lines: List[str] = []
    
    if metadata.get("title"):
        lines.append(f"# {metadata['title']}")
    
    lines.append("")
    
    if metadata.get("author"):
        lines.append(f"{LABEL_AUTHOR} {metadata['author']}")
    
    if metadata.get("published_date"):
        lines.append(f"{LABEL_PUBLISHED} {metadata['published_date']}")
    
    if metadata.get("description"):
        lines.append("")
        lines.append(f"{LABEL_DESCRIPTION} {metadata['description']}")
    
    return "\n".join(lines)


def format_headings(headings: List[Dict[str, str]]) -> str:
    """Format headings as Markdown table of contents.
    
    Args:
        headings: List of heading dictionaries with 'level' and 'text' keys.
    
    Returns:
        Formatted Markdown string.
    """
    if not headings:
        return ""
    
    lines: List[str] = []
    lines.append(SECTION_TOC)
    lines.append("")
    
    for h in headings[:MAX_HEADINGS_IN_TOC]:
        lines.append(f"{LIST_ITEM_PREFIX}{h['text']}")
    
    lines.append("")
    lines.append(SECTION_DIVIDER)
    lines.append("")
    
    return "\n".join(lines)


def format_paragraphs(paragraphs: List[str]) -> str:
    """Format paragraphs as Markdown.
    
    Args:
        paragraphs: List of paragraph text strings.
    
    Returns:
        Formatted Markdown string.
    """
    if not paragraphs:
        return ""
    
    lines: List[str] = []
    lines.append(SECTION_CONTENT)
    lines.append("")
    
    for p in paragraphs:
        lines.append(p)
        lines.append("")
    
    lines.append(SECTION_DIVIDER)
    lines.append("")
    
    return "\n".join(lines)


def format_lists(lists: List[List[str]]) -> str:
    """Format lists as Markdown.
    
    Args:
        lists: List of lists, where each inner list is a list item.
    
    Returns:
        Formatted Markdown string.
    """
    if not lists:
        return ""
    
    lines: List[str] = []
    lines.append(SECTION_LISTS)
    lines.append("")
    
    for i, list_items in enumerate(lists, 1):
        lines.append(f"### List {i}")
        lines.append("")
        
        for item in list_items:
            lines.append(f"{LIST_ITEM_PREFIX}{item}")
        
        lines.append("")
    
    lines.append(SECTION_DIVIDER)
    lines.append("")
    
    return "\n".join(lines)


def format_tables(tables: List[List[List[str]]]) -> str:
    """Format tables as Markdown.
    
    Args:
        tables: List of tables, where each table is a list of rows (list of cells).
    
    Returns:
        Formatted Markdown string.
    """
    if not tables:
        return ""
    
    lines: List[str] = []
    lines.append(SECTION_TABLES)
    lines.append("")
    
    for i, table in enumerate(tables, 1):
        lines.append(f"### Table {i}")
        lines.append("")
        
        # First row as header
        if table and table[0]:
            header = table[0]
            lines.append(TABLE_CELL_SEP + TABLE_CELL_SEP.join(header) + TABLE_CELL_SEP)
            lines.append(TABLE_CELL_SEP + TABLE_CELL_SEP.join([TABLE_DIVIDER_ROW] * len(header)) + TABLE_CELL_SEP)
            
            # Remaining rows as data
            for row in table[1:]:
                lines.append(TABLE_CELL_SEP + TABLE_CELL_SEP.join(row) + TABLE_CELL_SEP)
            
            lines.append("")
    
    lines.append(SECTION_DIVIDER)
    lines.append("")
    
    return "\n".join(lines)


def format_links(links: List[Dict[str, str]]) -> str:
    """Format links as Markdown.
    
    Args:
        links: List of link dictionaries with 'text' and 'href' keys.
    
    Returns:
        Formatted Markdown string.
    """
    if not links:
        return ""
    
    lines: List[str] = []
    lines.append(SECTION_LINKS)
    lines.append("")
    
    for link in links:
        lines.append(f"{LIST_ITEM_PREFIX}[{link['text']}]({link['href']})")
    
    lines.append("")
    lines.append(SECTION_DIVIDER)
    lines.append("")
    
    return "\n".join(lines)


def format_for_llm(
    structured_content: Dict[str, Any],
    url: str,
    fetch_method: str,
    max_length: int = MAX_OUTPUT_LENGTH
) -> str:
    """Format structured content as Markdown for LLM consumption.
    
    Args:
        structured_content: Dictionary from extract_content_structure().
        url: Source URL of the content.
        fetch_method: Fetch method used ('httpx' or 'playwright').
        max_length: Maximum length of the formatted output.
    
    Returns:
        Formatted Markdown string optimized for LLM understanding.
    
    Example:
        >>> content = extract_content_structure(html)
        >>> md = format_for_llm(content, "https://example.com", "httpx")
        >>> print(md[:500])
    """
    lines: List[str] = []
    
    # Header with metadata
    title = structured_content['metadata'].get('title', 'Untitled')
    lines.append(f"# Page: {title}")
    lines.append(f"## URL: {url}")
    lines.append(f"## Fetched: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"## Method: {fetch_method}")
    lines.append("")
    lines.append(SECTION_DIVIDER)
    lines.append("")
    
    # Metadata section
    meta_md = format_metadata(structured_content['metadata'])
    if meta_md:
        lines.append(meta_md)
        lines.append(SECTION_DIVIDER)
        lines.append("")
    
    # Table of contents
    toc_md = format_headings(structured_content['headings'])
    if toc_md:
        lines.append(toc_md)
    
    # Main content (paragraphs)
    content_md = format_paragraphs(structured_content['paragraphs'])
    if content_md:
        lines.append(content_md)
    
    # Lists
    lists_md = format_lists(structured_content['lists'])
    if lists_md:
        lines.append(lists_md)
    
    # Tables
    tables_md = format_tables(structured_content['tables'])
    if tables_md:
        lines.append(tables_md)
    
    # Links
    links_md = format_links(structured_content['links'])
    if links_md:
        lines.append(links_md)
    
    # Join and truncate
    full_md = "\n".join(lines)
    
    if len(full_md) > max_length:
        # Truncate at a section boundary if possible
        truncated = full_md[:max_length]
        last_section = truncated.rfind(SECTION_DIVIDER)
        if last_section > max_length * TRUNCATION_THRESHOLD_RATIO:
            truncated = truncated[:last_section]
        full_md = truncated + "\n\n" + TRUNCATION_MESSAGE
    
    return full_md


def html_to_markdown(
    html: str,
    heading_style: str = DEFAULT_HEADING_STYLE
) -> str:
    """Convert HTML to Markdown using markdownify.
    
    Args:
        html: Raw HTML string.
        heading_style: Style for headings ('ATX' for #, 'SETEXT' for underlines).
    
    Returns:
        Markdown string.
    
    Example:
        >>> html = "<h1>Title</h1><p>Text</p>"
        >>> md = html_to_markdown(html)
        >>> print(md)
        # Title
        
        Text
    """
    return markdownify(
        html,
        heading_style=heading_style,
        bullets=MARKDOWNIFY_BULLETS,
        strip=MARKDOWNIFY_STRIP_TAGS
    )