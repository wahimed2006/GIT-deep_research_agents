"""ScrappingAgent implementation.

This agent uses tool calling to intelligently navigate and extract
information from scraped Markdown content.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, override
from datetime import datetime

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
from ..prompt.scrapping_agent_prompt import SCRAPPING_AGENT_PROMPT
from .tools_calling_agent import ToolCallingAgent


class ScrappingAgent(ToolCallingAgent):
    """Agent that uses tools to navigate and extract from scraped MD content.

    Workflow:
    1. Orchestrator provides: query, question, and MD content
    2. Agent uses tools to search/filter/extract from MD
    3. If needed, agent scrapes linked pages recursively
    4. Returns final extracted information

    Tools enable efficient navigation without reading entire MD.
    """

    def __init__(
        self,
        model_name: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: str = "",
    ):
        """Initialize the ScrappingAgent.

        Args:
            model_name: Name of the LLM model.
            tools: Optional custom tool schemas.
            system_prompt: Optional system prompt.
        """
        agent_tools = tools if tools is not None else SCRAPPING_AGENT_TOOLS
        full_prompt = system_prompt or SCRAPPING_AGENT_PROMPT

        super().__init__(model_name, agent_tools, full_prompt)

        self.context = PageContext()
        self.tool_registry = build_tool_registry(self.context)

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> AgentResponse[str]:
        """Execute a tool by name with the given arguments."""
        if tool_name not in self.tool_registry:
            return AgentResponse[str](
                content=f"Unknown tool: {tool_name}. Available: {list(self.tool_registry.keys())}",
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
                content="No page loaded. Provide MD content first.",
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
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def load_markdown(
        self,
        markdown: str,
        url: str = "",
        structured: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Load Markdown content into the agent's context.

        Args:
            markdown: The scraped Markdown content.
            url: Optional source URL.
            structured: Optional structured data from scraper.
        """
        self.context.markdown = markdown
        self.context.url = url
        self.context.structured = structured or {}
        self.context.fetch_method = "preloaded"

    def chat(
        self,
        query: str,
        stream: bool = False,
    ) -> AgentResponse:
        """Chat with the agent to extract information from loaded MD.

        The agent will use tools (search, filter, get_section, etc.) to
        intelligently navigate the content and find relevant information.

        Args:
            query: The information to extract (e.g., "Bitcoin price").
            stream: Whether to stream the response.

        Returns:
            AgentResponse with extracted information.
        """
        if not self.context.markdown:
            return AgentResponse[str](
                content="No Markdown content loaded. Call load_markdown() first.",
                tool_calls=[],
                tool_results=[],
                metadata={"error": True, "reason": "no_content"},
            )

        enriched_query = (
            f"{query}\n\nCURRENT PAGE CONTENT:\n{self.context.markdown}"
        )
        response = super().chat(enriched_query, stream=stream)
        all_tool_calls = list(response.tool_calls)
        all_tool_results: List[Dict[str, Any]] = []

        for _ in range(5):
            if not response.tool_calls:
                break

            for tool_call in response.tool_calls:
                function = getattr(tool_call, "function", tool_call)
                if isinstance(function, dict):
                    tool_name = function.get("name", "")
                    arguments = function.get("arguments", {})
                else:
                    tool_name = getattr(function, "name", "")
                    arguments = getattr(function, "arguments", {})

                if not isinstance(arguments, dict):
                    arguments = {}
                tool_result = self.execute_tool(tool_name, arguments)
                all_tool_results.append({
                    "tool": tool_name,
                    "result": tool_result.content,
                })
                self.messages.append({
                    "role": "tool",
                    "content": tool_result.content,
                })

            response = self._chat_once()
            all_tool_calls.extend(response.tool_calls)

        response.tool_calls = all_tool_calls
        response.tool_results = all_tool_results
        return response

    def chat_with_sources(
        self,
        query: str,
        max_depth: int = 2,
    ) -> AgentResponse[Dict[str, Any]]:
        """Chat and recursively explore links if needed.

        Args:
            query: The information to extract.
            max_depth: Maximum recursion depth for link exploration.

        Returns:
            AgentResponse with extracted info and sources.
        """
        # Step 1: Extract from current page
        result = self.chat(query)

        # Step 2: Check if more info needed (agent decides via tools)
        # This is handled by the LLM through tool calls

        return AgentResponse[Dict[str, Any]](
            content=result.content,
            structured_data={
                "query": query,
                "url": self.context.url,
                "extracted": result.content,
                "tool_calls": result.tool_calls,
                "tool_results": result.tool_results,
            },
            metadata={
                "success": not result.metadata.get("error"),
                "timestamp": datetime.now().isoformat(),
            },
        )

    def get_current_page_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded page."""
        if not self.context.is_loaded:
            return {"loaded": False}

        metadata = self.context.structured.get("metadata", {})
        return {
            "loaded": True,
            "url": self.context.url,
            "title": metadata.get("title", "Unknown"),
            "text_length": len(self.context.markdown),
            "headings_count": len(self.context.structured.get("headings", [])),
            "tables_count": len(self.context.structured.get("tables", [])),
            "links_count": len(self.context.structured.get("links", [])),
        }