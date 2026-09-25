from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Generic, TypeVar

T = TypeVar("T", bound=Any)


@dataclass
class AgentResponse(Generic[T]):
    """Generic response object for all agents.

    Attributes:
        content: Main text response from the agent.
        tool_calls: List of tool calls requested by the model (if any).
        tool_results: List of tool execution results (if tools were executed).
        metadata: Additional information (usage, model name, etc.).
        structured_data: Optional structured/parsed data (Pydantic, dict, etc.).
        raw_response: The original raw response from the LLM provider.
    """

    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    structured_data: Optional[T] = None
    raw_response: Optional[Any] = None