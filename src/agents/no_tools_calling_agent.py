"""Simple agent module for models without tool calling support.

This module provides a SimpleAgent class that extends the base Agent
for models that do not support tool calling. It provides a clean interface
for basic conversational interactions.
"""

from typing import Any, Dict, List, Optional
import ollama
from agent import Agent
from tools import AgentResponse
from typing import override


class SimpleAgent(Agent):
    """A conversational agent for models without tool calling support.
    
    This class extends the base Agent class to provide a specialized
    interface for models that do not support tool calling. It ensures
    that tool-related parameters are not used and provides clear error
    messages if tool calling is attempted.
    
    Attributes:
        model_name: The name of the Ollama model being used.
        system_prompt: The system instruction defining agent behavior.
    
    Example:
        >>> agent = SimpleAgent("gemma2:9b", system_prompt="You are helpful.")
        >>> response = agent.chat("Hello!")
        >>> print(response.content)
    """
    
    def __init__(self, model_name: str, system_prompt: str = ""):
        """Initialize the SimpleAgent with a model and optional system prompt.
        
        Args:
            model_name: The name of the Ollama model to use for chat completions.
            system_prompt: Optional system instruction to set the agent's behavior.
        
        Example:
            >>> agent = SimpleAgent("llama3.2:3b", system_prompt="Be concise.")
        """
        super().__init__(model_name, system_prompt)
    
    @override
    def chat(
        self,
        query: str,
        stream: bool = True
    ) -> AgentResponse[None]:
        """Send a user message and get the assistant's response.
        
        This method overrides the base chat method for simple conversational
        models. It does not support tool calling and will raise an error if
        tools are attempted.
        
        Args:
            query: The user's message or question to send to the agent.
            stream: Whether to stream the response token by token.
                   Defaults to True for real-time output.
        
        Returns:
            An AgentResponse containing the assistant's text response.
        
        Raises:
            Exception: Propagates any errors from the Ollama API call.
        
        Example:
            >>> agent = SimpleAgent("gemma2:9b")
            >>> response = agent.chat("What is Python?")
            >>> print(response.content)
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
            # Remove the user message if the API call fails
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
        
        # Build and return standardized response
        return self._build_response(
            content=full_response,
            raw_response=response,
            metadata={"model": self.model_name, "stream": stream}
        )
    
    def chat_with_system(self, query: str, system_prompt: str) -> AgentResponse[None]:
        """Send a query with a temporary system prompt override.
        
        This method allows you to temporarily override the system prompt
        for a single query without modifying the agent's default system prompt.
        
        Args:
            query: The user's message or question to send to the agent.
            system_prompt: Temporary system prompt for this query only.
        
        Returns:
            An AgentResponse containing the assistant's response.
        
        Example:
            >>> agent = SimpleAgent("gemma2:9b", system_prompt="Default prompt")
            >>> response = agent.chat_with_system("Help me code", system_prompt="You are a coding expert")
        """
        # Save original system prompt
        original_system = self.system_prompt
        
        # Temporarily replace system prompt
        self.system_prompt = system_prompt
        self.messages[0] = {"role": "system", "content": system_prompt}
        
        try:
            response = self.chat(query, stream=False)
        finally:
            # Restore original system prompt
            self.system_prompt = original_system
            self.messages[0] = {"role": "system", "content": original_system}
        
        return response


if __name__ == "__main__":
    agent = SimpleAgent(
        "gemma2:9b",
        system_prompt="You are a helpful assistant. Respond concisely."
    )
    
    print("SimpleAgent initialized. Type 'quit' or 'exit' to end.\n")
    
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
            # Access response via AgentResponse object
            # print(f"\nResponse: {response.content}\n")
        except Exception as e:
            print(f"\nError: {e}\n")