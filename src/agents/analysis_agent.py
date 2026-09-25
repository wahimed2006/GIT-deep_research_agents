"""Analysis Agent module for relevance scoring.

This module provides an AnalysisWebSearchAgent class that evaluates
the relevance of search results against a user query.
"""

import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.agents"


from .no_tools_calling_agent import SimpleAgent
from ..prompt.analysis_agent_prompt import SEARCH_AGENT_PROMPT


class AnalysisWebSearchAgent(SimpleAgent):
    """An agent that scores the relevance of search results.
    
    This agent evaluates individual search results against a user query
    and assigns a relevance score from 0 to 100.
    """
    
    def __init__(self, model_name: str):
        """Initialize the agent with a scoring-optimized model.
        
        Args:
            model_name: The name of the Ollama model to use.
        """
        super().__init__(model_name, SEARCH_AGENT_PROMPT)


if __name__ == "__main__":
    agent = AnalysisWebSearchAgent("gemma2:latest")
    
    print("AnalysisWebSearchAgent initialized.\n")
    print("Type 'quit' or 'exit' to end.\n")
    
    while True:
        try:
            query = input("Query: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if query.strip().lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        
        try:
            response = agent.chat(query, stream=True)
            print(f"\nResponse: {response.content}\n")
        
        except Exception as e:
            print(f"Error: {e}\n")