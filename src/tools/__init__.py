"""ScrappingAgent tools package.

This package provides tool schemas, state management, and implementations
for an agent that can scrape and explore web pages.
"""
from .agent_response import AgentResponse
from .tools_definition.scrapping_agent_tools_definition import SCRAPPING_AGENT_TOOLS
from .page_context import PageContext
from .page_tools import (
    scrape_page,
    filter_table_rows,
    search_within_page,
    get_page_section,
    filter_links_by_category,
    extract_code_blocks,
    get_summary,
    compare_entities,
    build_tool_registry,
    PageNotLoadedError,
)

__all__ = [
    # Schemas
    "SCRAPPING_AGENT_TOOLS",
    # Context
    "PageContext",
    # Tools
    "scrape_page",
    "filter_table_rows",
    "search_within_page",
    "get_page_section",
    "filter_links_by_category",
    "extract_code_blocks",
    "get_summary",
    "compare_entities",
    "build_tool_registry",
    "PageNotLoadedError",
]