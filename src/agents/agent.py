"""Agent module for conversational AI using Ollama.

This module provides an Agent class that maintains conversation history
and interacts with local LLM models through the Ollama API.
"""

from typing import Dict, Any, List, Optional
import ollama
from ..tools import AgentResponse


class Agent:
    """A conversational agent that maintains chat history with an LLM.
    
    This class wraps the Ollama chat API and maintains a persistent message
    history including system prompts, user messages, and assistant responses.
    
    Attributes:
        model_name: The name of the Ollama model to use (e.g., 'llama3.1:8b').
        system_prompt: The system instruction that defines the agent's behavior.
        messages: List of message dictionaries maintaining conversation history.
    
    Example:
        >>> agent = Agent("llama3.1:8b", system_prompt="You are a helpful assistant.")
        >>> response = agent.chat("Hello!")
        >>> print(response)
    """
    
    def __init__(self, model_name: str, system_prompt: str = ""):
        """Initialize the Agent with a model and optional system prompt.
        
        Args:
            model_name: The name of the Ollama model to use for chat completions.
            system_prompt: Optional system instruction to set the agent's behavior.
                          Defaults to empty string.
        
        Example:
            >>> agent = Agent("llama3.2:latest", system_prompt="You are a coding assistant.")
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.messages: list[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
    
    def chat(self, query: str, stream: bool = True) -> AgentResponse[None]:
        """Send a user message and get the assistant's response.
        
        This method appends the user query to the conversation history,
        calls the Ollama API, and stores the assistant's response for
        future context.
        
        Args:
            query: The user's message or question to send to the agent.
            stream: Whether to stream the response token by token.
                   Defaults to True for real-time output.
        
        Returns:
            An AgentResponse object containing the assistant's response.
        
        Raises:
            Exception: Propagates any errors from the Ollama API call.
        
        Example:
            >>> agent = Agent("llama3.1:8b")
            >>> response = agent.chat("What is Python?")
            >>> print(f"Response content: {response.content}")
        """
        # Append user message to conversation history
        self.messages.append({"role": "user", "content": query})
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=self.messages,
                stream=stream
            )
        except Exception as e:
            # Remove the user message if the API call fails to keep history consistent
            self.messages.pop()
            raise e
        
        full_response = ""
        
        if stream:
            print("Agent: ", end="", flush=True)
            # Type-safe iteration over streaming response
            chunk: Dict[str, Any]
            for chunk in response:  # type: ignore
                content: str = chunk.get("message", {}).get("content", "")
                full_response += content
                print(content, end="", flush=True)
            print()  # Newline after streaming completes
        else:
            message: Dict[str, Any] = response.get("message", {}) if isinstance(response, dict) else {}  # type: ignore
            full_response = message.get("content", "")
            print("Agent: ")
            print(full_response)
        
        # Append assistant response to conversation history
        self.messages.append({"role": "assistant", "content": full_response})
        
        return self._build_response(
            content=full_response,
            raw_response=response
        )
    
    def reset(self) -> None:
        """Clear the conversation history and reset to initial state.
        
        This method removes all user and assistant messages while
        preserving the system prompt. Call this to start a fresh
        conversation without creating a new Agent instance.
        
        Example:
            >>> agent = Agent("llama3.1:8b")
            >>> agent.chat("First question")
            >>> agent.reset()  # Clears history but keeps system prompt
            >>> agent.chat("New conversation starts here")
        """
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
    
    def get_history(self) -> list[Dict[str, str]]:
        """Return a copy of the current conversation history.
        
        Returns:
            A list of message dictionaries containing the full conversation
            history including system prompts, user messages, and assistant responses.
        
        Example:
            >>> agent = Agent("llama3.1:8b")
            >>> agent.chat("Hello")
            >>> history = agent.get_history()
            >>> print(f"Message count: {len(history)}")
        """
        return self.messages.copy()
    
    def _build_response(
        self,
        content: str,
        raw_response: Any,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentResponse[None]:
        """Build a standardized AgentResponse from raw LLM output.
        
        Args:
            content: The main text content of the response.
            raw_response: The raw response object from the LLM provider.
            tool_calls: Optional list of tool calls requested by the model.
            tool_results: Optional list of tool execution results.
            metadata: Optional additional metadata (usage, model info, etc.).
        
        Returns:
            An AgentResponse object with the provided data.
        """
        return AgentResponse(
            content=content,
            tool_calls=tool_calls or [],
            tool_results=tool_results or [],
            metadata=metadata or {},
            raw_response=raw_response
        )


if __name__ == "__main__":
    agent = Agent(
        "llama3.1:8b",
        system_prompt="You are a helpful assistant. Respond concisely and clearly."
    )
    
    print("Chat started. Type 'quit' or 'exit' to end the conversation.\n")
    
    while True:
        try:
            query = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if query.strip().lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        
        try:
            response = agent.chat(query)
            # Access response content via the AgentResponse object
            # print(f"Response: {response.content}")
        except Exception as e:
            print(f"\nError calling Ollama API: {e}")