"""ScrappingAgent implementation.

This agent can scrape web pages and explore their content using specialized tools.
"""

from typing import Any, Dict, List, Optional

from tools_calling_agent import ToolCallingAgent
from tools import AgentResponse
from tools import SCRAPPING_AGENT_TOOLS, PageContext, build_tool_registry
from prompt import SCRAPPING_AGENT_PROMPT


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
    ):
        """Initialize the ScrappingAgent.

        Args:
            model_name: Name of the LLM model (e.g., 'qwen3.8', 'llama3.1').
            tools: Optional custom tool schemas. Defaults to SCRAPPING_AGENT_TOOLS.
            system_prompt: Optional system prompt to guide agent behavior.
        """
        super().__init__(model_name, SCRAPPING_AGENT_TOOLS, SCRAPPING_AGENT_PROMPT)

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
                metadata={"error": True},
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
        except Exception as error:
            return AgentResponse[str](
                content=f"Tool '{tool_name}' failed: {error}",
                tool_calls=[],
                tool_results=[],
                metadata={"error": True, "tool_name": tool_name},
                raw_response=error,
            )

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