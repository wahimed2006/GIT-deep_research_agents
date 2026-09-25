"""Agent module for conversational AI using Ollama.

This module provides an Agent class that maintains conversation history
and interacts with local LLM models through the Ollama API.
"""

from typing import Dict, Any
import ollama


class Agent:
    """A conversational agent that maintains chat history with an LLM.
    
    This class wraps the Ollama chat API and maintains a persistent message
    history including system prompts, user messages, and assistant responses.
    
    Attributes:
        model_name: The name of the Ollama model to use (e.g., 'gemma2:latest').
        system_prompt: The system instruction that defines the agent's behavior.
        messages: List of message dictionaries maintaining conversation history.
    
    Example:
        >>> agent = Agent("gemma2:latest", system_prompt="You are a helpful assistant.")
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
    
    def chat(self, query: str, stream: bool = True) -> str:
        """Send a user message and get the assistant's response.
        
        This method appends the user query to the conversation history,
        calls the Ollama API, and stores the assistant's response for
        future context.
        
        Args:
            query: The user's message or question to send to the agent.
            stream: Whether to stream the response token by token.
                   Defaults to True for real-time output.
        
        Returns:
            The complete assistant response as a string.
        
        Raises:
            Exception: Propagates any errors from the Ollama API call.
        
        Example:
            >>> agent = Agent("gemma2:latest")
            >>> response = agent.chat("What is Python?")
            >>> print(f"Response length: {len(response)}")
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
            full_response = response.get("message", {}).get("content", "")  # type: ignore
            print("Agent: ")
            print(full_response)
        
        # Append assistant response to conversation history
        self.messages.append({"role": "assistant", "content": full_response})
        
        return full_response
    
    def reset(self) -> None:
        """Clear the conversation history and reset to initial state.
        
        This method removes all user and assistant messages while
        preserving the system prompt. Call this to start a fresh
        conversation without creating a new Agent instance.
        
        Example:
            >>> agent = Agent("gemma2:latest")
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
            history including system prompt, user messages, and assistant responses.
        
        Example:
            >>> agent = Agent("gemma2:latest")
            >>> agent.chat("Hello")
            >>> history = agent.get_history()
            >>> print(f"Message count: {len(history)}")
        """
        return self.messages.copy()


if __name__ == "__main__":
    agent = Agent(
        "gemma2:latest",
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
            agent.chat(query)
        except Exception as e:
            print(f"\nError calling Ollama API: {e}")