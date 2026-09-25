"""User Input Agent module for query decomposition.

This module provides a UserInputAgent class that decomposes user queries
into atomic, actionable sub-requests for downstream specialized agents.
"""

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.agents"

from .no_tools_calling_agent import SimpleAgent
from ..prompt.user_input_agent_prompt import INPUT_AGENT_PROMPT


class UserInputAgent(SimpleAgent):
    """An agent that decomposes user queries into sub-requests.
    
    This agent analyzes a user's raw query and breaks it down into clear,
    atomic, actionable sub-requests that specialized downstream agents
    can execute independently.
    
    Example:
        >>> agent = UserInputAgent("llama3.1:8b")
        >>> response = agent.chat("Analyse le marché des bourses du jour")
        >>> print(response.content)
    """
    
    def __init__(self, model_name: str):
        """Initialize the UserInputAgent with a decomposition-optimized model.
        
        Args:
            model_name: The name of the Ollama model to use. Should be a model
                       with good instruction-following capabilities.
        
        Example:
            >>> agent = UserInputAgent("llama3.1:8b")
        """
        super().__init__(model_name, INPUT_AGENT_PROMPT)


if __name__ == "__main__":
    agent = UserInputAgent("gemma2:latest")
    
    print("UserInputAgent initialized. Enter queries to decompose.\n")
    print("Type 'quit' or 'exit' to end.\n")
    
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
            response = agent.chat(query, stream=True)
            print(f"\nDecomposed sub-requests:\n{response.content}\n")
        except Exception as e:
            print(f"\nError: {e}\n")