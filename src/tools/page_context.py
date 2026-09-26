"""State model for pages handled by the ScrappingAgent.

The tools operate only on a PageContext. This ensures that search,
section, table, link, and code extraction use the currently scraped page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PageContext:
    """Represent the current page available to the agent tools.

    Attributes:
        url: Final URL associated with the scraped page.
        markdown: Complete Markdown representation of the page.
        structured: Structured result returned by the scraper.
        raw_html: Optional rendered HTML, useful for code extraction.
        fetch_method: Fetch strategy used by the scraper.
    """

    url: str = ""
    markdown: str = ""
    structured: Dict[str, Any] = field(default_factory=dict)
    raw_html: Optional[str] = None
    fetch_method: str = ""

    @property
    def is_loaded(self) -> bool:
        """Return whether the context contains a successfully loaded page."""
        return bool(self.url and (self.markdown or self.structured))

    def clear(self) -> None:
        """Remove the current page and reset the context."""
        self.url = ""
        self.markdown = ""
        self.structured = {}
        self.raw_html = None
        self.fetch_method = ""
