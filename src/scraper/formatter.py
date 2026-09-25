"""Content formatter for LLM consumption.

This module provides functions to format extracted content into
clean, structured Markdown optimized for LLM understanding.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from markdownify import markdownify


def format_metadata(metadata: Dict[str, Optional[str]]) -> str:
    """Format metadata as Markdown.
    
    Args:
        metadata: Dictionary with 'title', 'description', 'author', 'published_date'.
    
    Returns:
        Formatted Markdown string.
    """
    lines = []
    
    if metadata.get("title"):
        lines.append(f"# {metadata['title']}")
    
    lines.append("")
    
    if metadata.get("author"):
        lines.append(f"**Author:** {metadata['author']}")
    
    if metadata.get("published_date"):
        lines.append(f"**Published:** {metadata['published_date']}")
    
    if metadata.get("description"):
        lines.append("")
        lines.append(f"**Description:** {metadata['description']}")
    
    return "\n".join(lines)


def format_headings(headings: list[Dict[str, str]]) -> str:
    """Format headings as Markdown table of contents.
    
    Args:
        headings: List of heading dictionaries with 'level' and 'text' keys.
    
    Returns:
        Formatted Markdown string.
    """
    if not headings:
        return ""
    
    lines = []
    lines.append("## Table of Contents")
    lines.append("")
    
    for h in headings[:15]:  # Limit to 15
        lines.append(f"- {h['text']}")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    return "\n".join(lines)


def format_paragraphs(paragraphs: list[str]) -> str:
    """Format paragraphs as Markdown.
    
    Args:
        paragraphs: List of paragraph text strings.
    
    Returns:
        Formatted Markdown string.
    """
    if not paragraphs:
        return ""
    
    lines = []
    lines.append("## Content")
    lines.append("")
    
    for p in paragraphs:
        lines.append(p)
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    return "\n".join(lines)


def format_lists(lists: list[list[str]]) -> str:
    """Format lists as Markdown.
    
    Args:
        lists: List of lists, where each inner list is a list item.
    
    Returns:
        Formatted Markdown string.
    """
    if not lists:
        return ""
    
    lines = []
    lines.append("## Lists")
    lines.append("")
    
    for i, list_items in enumerate(lists, 1):
        lines.append(f"### List {i}")
        lines.append("")
        
        for item in list_items:
            lines.append(f"- {item}")
        
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    return "\n".join(lines)


def format_tables(tables: list[list[list[str]]]) -> str:
    """Format tables as Markdown.
    
    Args:
        tables: List of tables, where each table is a list of rows (list of cells).
    
    Returns:
        Formatted Markdown string.
    """
    if not tables:
        return ""
    
    lines = []
    lines.append("## Tables")
    lines.append("")
    
    for i, table in enumerate(tables, 1):
        lines.append(f"### Table {i}")
        lines.append("")
        
        # First row as header
        if table and table[0]:
            header = table[0]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("| " + " | ".join(["---"] * len(header)) + " |")
            
            # Remaining rows as data
            for row in table[1:]:
                lines.append("| " + " | ".join(row) + " |")
            
            lines.append("")
    
    lines.append("---")
    lines.append("")
    
    return "\n".join(lines)


def format_links(links: list[Dict[str, str]]) -> str:
    """Format links as Markdown.
    
    Args:
        links: List of link dictionaries with 'text' and 'href' keys.
    
    Returns:
        Formatted Markdown string.
    """
    if not links:
        return ""
    
    lines = []
    lines.append("## Related Links")
    lines.append("")
    
    for link in links:
        lines.append(f"- [{link['text']}]({link['href']})")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    return "\n".join(lines)


def format_for_llm(
    structured_content: Dict[str, Any],
    url: str,
    fetch_method: str,
    max_length: int = 10000
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
    lines = []
    
    # Header with metadata
    lines.append(f"# Page: {structured_content['metadata'].get('title', 'Untitled')}")
    lines.append(f"## URL: {url}")
    lines.append(f"## Fetched: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"## Method: {fetch_method}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Metadata section
    meta_md = format_metadata(structured_content['metadata'])
    if meta_md:
        lines.append(meta_md)
        lines.append("---")
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
        last_section = truncated.rfind("---")
        if last_section > max_length * 0.8:
            truncated = truncated[:last_section]
        full_md = truncated + "\n\n[Content truncated...]"
    
    return full_md


def html_to_markdown(html: str, heading_style: str = "ATX") -> str:
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
    return markdownify(html, heading_style=heading_style, bullets="-", strip=["a"])