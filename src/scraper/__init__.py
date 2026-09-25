"""Scraper package for web content extraction.

This package provides tools to fetch, clean, and format web content
for LLM consumption, with automatic JavaScript detection.
"""

from .fetcher import fetch_content, fetch_with_httpx, fetch_with_playwright, FetchError
from .cleaner import clean_html, extract_content_structure, extract_metadata
from .formatter import format_for_llm, html_to_markdown
from .scraper import scrape, scrape_multiple, ScraperError

__all__ = [
    # Fetcher
    "fetch_content",
    "fetch_with_httpx",
    "fetch_with_playwright",
    "FetchError",
    
    # Cleaner
    "clean_html",
    "extract_content_structure",
    "extract_metadata",
    
    # Formatter
    "format_for_llm",
    "html_to_markdown",
    
    # Main scraper
    "scrape",
    "scrape_multiple",
    "ScraperError",
]