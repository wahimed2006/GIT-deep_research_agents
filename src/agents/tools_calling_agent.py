"""Tool calling agent module extending the base Agent class.

This module provides a ToolCallingAgent class that extends the base Agent
with tool calling capabilities, including validation of model support.
"""

from typing import Any, Dict, List, Optional
import requests
from agent import Agent
import ollama


class ToolCallingAgent(Agent):
    """An agent with tool calling capabilities.
    
    This class extends the base Agent class to support structured tool
    calling with Ollama models. It validates that the model supports
    tool calling before initialization.
    
    Attributes:
        tools: List of tool definitions in JSON schema format.
        model_name: The name of the Ollama model being used.
    
    Raises:
        ValueError: If the model does not support tool calling.
    
    Example:
        >>> tools = [
        ...     {
        ...         "type": "function",
        ...         "function": {
        ...             "name": "get_weather",
        ...             "description": "Get current weather",
        ...             "parameters": {
        ...                 "type": "object",
        ...                 "properties": {
        ...                     "location": {"type": "string"}
        ...                 },
        ...                 "required": ["location"]
        ...             }
        ...         }
        ...     }
        ... ]
        >>> agent = ToolCallingAgent("llama3.1:8b", tools=tools)
    """
    
    def __init__(
        self,
        model_name: str,
        tools: List[Dict[str, Any]],
        system_prompt: str = ""
    ):
        """Initialize the ToolCallingAgent with model, tools, and system prompt.
        
        Args:
            model_name: The name of the Ollama model to use for chat completions.
            tools: List of tool definitions in Ollama's tool format.
            system_prompt: Optional system instruction to set the agent's behavior.
        
        Raises:
            ValueError: If the specified model does not support tool calling.
            requests.RequestException: If the Ollama API is unreachable.
        
        Example:
            >>> tools = [{"type": "function", "function": {...}}]
            >>> agent = ToolCallingAgent("llama3.1:8b", tools=tools)
        """
        # Check tool support before initialization
        if not self._supports_tools(model_name):
            raise ValueError(
                f"Model '{model_name}' does not support tool calling. "
                f"Use a model like 'llama3.1:8b', 'qwen2.5:7b', or 'mistral:7b'."
            )
        
        super().__init__(model_name, system_prompt)
        self.tools = tools
    
    @staticmethod
    def _supports_tools(model_name: str) -> bool:
        """Check if an Ollama model supports tool calling.
        
        This static method queries the Ollama API to determine if a model
        has the 'tools' capability. It does not require an instance of the
        class to be fully initialized.
        
        Args:
            model_name: The name of the Ollama model to check.
        
        Returns:
            True if the model supports tool calling, False otherwise.
        
        Raises:
            requests.RequestException: If the Ollama API is unreachable.
        
        Example:
            >>> ToolCallingAgent._supports_tools("llama3.1:8b")
            True
            >>> ToolCallingAgent._supports_tools("gemma2:9b")
            False
        """
        try:
            response = requests.post(
                "http://localhost:11434/api/show",
                json={"name": model_name},
                timeout=5
            )
            response.raise_for_status()
            data: Dict[str, Any] = response.json()
            
            capabilities: List[str] = data.get("capabilities", [])
            return "tools" in capabilities
        except requests.RequestException:
            # If we can't reach Ollama, assume no tool support
            # This prevents crashes during initialization
            return False
    
    @classmethod
    def list_supported_models(cls) -> List[str]:
        """Get a list of commonly known models that support tool calling.
        
        Returns:
            A list of model names that are known to support tool calling.
            Note: This is not exhaustive; always verify with _supports_tools().
        
        Example:
            >>> supported = ToolCallingAgent.list_supported_models()
            >>> print(f"Found {len(supported)} known models")
        """
        return [
            "llama3.1:8b",
            "llama3.1:70b",
            "llama3.2:1b",
            "llama3.2:3b",
            "llama3.3:70b",
            "llama4:scout",
            "llama4:maverick",
            "qwen2.5:0.5b",
            "qwen2.5:1.5b",
            "qwen2.5:3b",
            "qwen2.5:7b",
            "qwen2.5:14b",
            "qwen2.5:32b",
            "qwen2.5:72b",
            "qwen3:8b",
            "qwen3:32b",
            "qwen3.5:7b",
            "qwen3.5:32b",
            "mistral:7b",
            "mistral-nemo:12b",
            "mistral-small3.2:24b",
            "gemma4:12b",
            "functiongemma:7b",
            "glm-5.2:9b",
        ]
    
    def chat(
        self,
        query: str,
        stream: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Send a user message and get the assistant's response with tool calling.
        
        This method extends the base chat method to support tool calling.
        If tools are provided (or if the agent has default tools), they will
        be passed to the Ollama API for structured function calling.
        
        Args:
            query: The user's message or question to send to the agent.
            stream: Whether to stream the response token by token.
                   Defaults to True for real-time output.
            tools: Optional list of tools to use for this specific call.
                  If None, uses the agent's default tools from initialization.
        
        Returns:
            The complete assistant response as a string.
        
        Raises:
            ValueError: If tools are provided but the model doesn't support them.
        
        Example:
            >>> agent = ToolCallingAgent("llama3.1:8b", tools=[...])
            >>> response = agent.chat("What's the weather in Paris?")
        """
        # Use provided tools or fall back to instance tools
        active_tools = tools if tools is not None else self.tools
        
        # Append user message to conversation history
        self.messages.append({"role": "user", "content": query})
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=self.messages,
                tools=active_tools,
                stream=stream
            )
        except Exception as e:
            # Remove the user message if the API call fails
            self.messages.pop()
            raise e
        
        full_response = ""
        
        if stream:
            print("Agent: ", end="", flush=True)
            for chunk in response:  # type: ignore
                content: str = chunk.get("message", {}).get("content", "")
                full_response += content
                print(content, end="", flush=True)
            print()
        else:
            full_response = response.get("message", {}).get("content", "")  # type: ignore
            print("Agent: ")
            print(full_response)
        
        # Append assistant response to conversation history
        self.messages.append({"role": "assistant", "content": full_response})
        
        return full_response


if __name__ == "__main__":
    # Example usage with a simple tool
    weather_tool = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    try:
        agent = ToolCallingAgent(
            "qwen3:4b ",
            tools=weather_tool,
            system_prompt="You are a helpful assistant with access to weather tools."
        )
        print("ToolCallingAgent initialized successfully!\n")
        
        # Test chat
        response = agent.chat("What's the weather in Paris?")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("\nAvailable models with tool support:")
        for model in ToolCallingAgent.list_supported_models()[:5]:
            print(f"  - {model}")
        print("  ...")