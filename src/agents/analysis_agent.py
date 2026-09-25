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
import re

class ScoreParseError(Exception):
    """Raised when the score cannot be parsed from the response."""
    pass

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

    def score_relevance(self,  
        query: str,
        title: str,
        link: str,
        body: str,
    )->int:
        """Score the relevance of a search result to a query.
        
        Args:
            query: The user's search query.
            title: The title of the search result.
            link: The URL of the search result.
            body: The snippet/body text of the search result.
            model_name: The Ollama model to use for scoring.
        
        Returns:
            An integer relevance score from 0 to 100.
        
        Example:
            >>> score = score_relevance(
            ...     query="Bitcoin price today",
            ...     title="Bitcoin Price Today - Live Chart",
            ...     link="https://coinmarketcap.com/",
            ...     body="Bitcoin is $42,000 USD today"
            ... )
            >>> print(f"Relevance: {score}/100")
        """
        agent = AnalysisWebSearchAgent(self.model_name)
        
        context = f"""
            User Query: {query}

            Search Result:
            - Title: {title}
            - URL: {link}
            - Snippet: {body}
        """
        
        response = agent.chat(context, stream=False)
        return self.parse_score(response.content)
        
    def parse_score(self, text: str) -> int:
        """Parse a relevance score from the agent's response.
            
        Args:
            text: Raw response text from the agent, expected to contain
                a score between <SCORE>...</ENDSCORE> tags.
            
        Returns:
            An integer relevance score from 0 to 100.
            
        Raises:
            ScoreParseError: If no valid score can be extracted from the text.
            
        Example:
            >>> text = "<SCORE>85<ENDSCORE>"
            >>> score = parse_score(text)
            >>> print(score)
            85
        """
        # Try to extract score between <SCORE> tags
        match = re.search(r"<SCORE>\s*(\d+)\s*<ENDSCORE>", text, re.DOTALL)
        if match:
            return int(match.group(1))
            
        # Fallback: find any number 0-100
        numbers = re.findall(r"\b(\d{1,3})\b", text)
        for num_str in numbers:
            num = int(num_str)
            if 0 <= num <= 100:
                return num
            
        # If no score found, raise exception
        raise ScoreParseError(f"Could not parse score from response: {text[:200]}")

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