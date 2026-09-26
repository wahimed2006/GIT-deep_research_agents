"""ScrappingAgent implementation.

This agent can scrape web pages and explore their content using specialized tools.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, override

# Add parent directory to path if running as script
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.agents"

from ..tools import (
    SCRAPPING_AGENT_TOOLS,
    PageContext,
    build_tool_registry,
    PageNotLoadedError,
)
from ..tools import AgentResponse
from .tools_calling_agent import ToolCallingAgent


class ScrappingAgent(ToolCallingAgent):
    """Agent capable of scraping and exploring web pages.

    This agent maintains a PageContext that is updated whenever a page
    is scraped. All exploration tools (search, filter, extract) operate
    on this context.

    Attributes:
        model_name: Name of the LLM model to use.
        system_prompt: Optional system prompt for the agent.
        context: Shared page state for all tool calls.
    """

    def __init__(
        self,
        model_name: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: str = "",
    ):
        """Initialize the ScrappingAgent.

        Args:
            model_name: Name of the LLM model (e.g., 'qwen3.8', 'llama3.1').
            tools: Optional custom tool schemas. Defaults to SCRAPPING_AGENT_TOOLS.
            system_prompt: Optional system prompt to guide agent behavior.
        """
        # Use default tools if none provided
        agent_tools = tools if tools is not None else SCRAPPING_AGENT_TOOLS

        # Build default system prompt if none provided
        default_prompt = (
            "You are a web scraping assistant that can fetch and explore web pages. "
            "You have access to tools for scraping pages, searching within content, "
            "filtering tables and links, extracting code blocks, and summarizing pages. "
            "Always start by scraping the target URL before using exploration tools."
        )
        full_prompt = system_prompt if system_prompt else default_prompt

        # Initialize parent ToolCallingAgent
        super().__init__(model_name, agent_tools, full_prompt)

        # Create shared page context and tool registry
        self.context = PageContext()
        self.tool_registry = build_tool_registry(self.context)

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> AgentResponse[str]:
        """Execute a tool by name with the given arguments.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Dictionary of arguments for the tool.

        Returns:
            AgentResponse with the tool result or error message.
        """
        if tool_name not in self.tool_registry:
            return AgentResponse[str](
                content=f"Unknown tool: {tool_name}. Available tools: {list(self.tool_registry.keys())}",
                tool_calls=[],
                tool_results=[],
                metadata={"error": True, "tool_name": tool_name},
            )

        try:
            tool_function = self.tool_registry[tool_name]
            result = tool_function(**arguments)
            return AgentResponse[str](
                content=str(result),
                tool_calls=[],
                tool_results=[{"tool": tool_name, "result": result}],
                metadata={"success": True, "tool_name": tool_name},
            )
        except PageNotLoadedError:
            return AgentResponse[str](
                content="No page is loaded. Call scrape_page first with a URL.",
                tool_calls=[],
                tool_results=[],
                metadata={"error": True, "tool_name": tool_name, "reason": "page_not_loaded"},
            )
        except Exception as error:
            return AgentResponse[str](
                content=f"Tool '{tool_name}' failed: {error}",
                tool_calls=[],
                tool_results=[],
                metadata={"error": True, "tool_name": tool_name},
                raw_response=error,
            )
    @override
    def reset(self) -> None:
        """Reset the agent's page context and conversation history."""
        self.context.clear()

    def get_current_page_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded page.

        Returns:
            Dictionary with page metadata.
        """
        if not self.context.is_loaded:
            return {"loaded": False}

        metadata = self.context.structured.get("metadata", {})
        return {
            "loaded": True,
            "url": self.context.url,
            "title": metadata.get("title", "Unknown"),
            "fetch_method": self.context.fetch_method,
            "text_length": len(self.context.structured.get("body_text", "")),
            "headings_count": len(self.context.structured.get("headings", [])),
            "tables_count": len(self.context.structured.get("tables", [])),
            "links_count": len(self.context.structured.get("links", [])),
        }

    def scrape_and_extract(
        self,
        url: str,
        extraction_query: str,
        force_playwright: bool = False,
    ) -> AgentResponse[str]:
        """Scrape a page and extract information matching a query.

        This is a convenience method that combines scraping and extraction
        in a single call.

        Args:
            url: URL of the page to scrape.
            extraction_query: What information to extract (e.g., "Bitcoin price").
            force_playwright: Whether to force JavaScript rendering.

        Returns:
            AgentResponse with the extracted information.
        """
        # Step 1: Scrape the page
        scrape_result = self.execute_tool("scrape_page", {
            "url": url,
            "force_playwright": force_playwright,
        })

        if scrape_result.metadata.get("error"):
            return scrape_result

        # Step 2: Search within page for the query
        search_result = self.execute_tool("search_within_page", {
            "pattern": extraction_query,
        })

        if search_result.metadata.get("error"):
            # Fallback: return summary
            return self.execute_tool("get_summary", {"max_length": 150})

        return search_result


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Scrape and extract information
    agent = ScrappingAgent(model_name="llama3.1:8b")

    # Scrape CoinMarketCap
    response = agent.scrape_and_extract(
        url="https://coinmarketcap.com",
        extraction_query="CoinMarketCap",
        force_playwright=True,
    )

    print("=== ScrappingAgent Result ===")
    print(response.content[:1000])
    print("\n=== Page Info ===")
    print(agent.get_current_page_info())