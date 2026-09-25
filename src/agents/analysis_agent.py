"""Analysis Agent module for relevance scoring.

This module provides an AnalysisWebSearchAgent class that evaluates
the relevance of search results against a user query.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

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
        
    def web_search_score(
        self,
        query: str,
        web_search_result: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Score multiple search results and return them with relevance scores.
        
        Args:
            query: The user's search query.
            web_search_result: List of search results, each containing 'title',
                              'link', and 'body' keys.
        
        Returns:
            List of search results with an added 'relevance_score' key,
            sorted by score in descending order.
        
        Raises:
            ScoreParseError: If any result's score cannot be parsed.
        
        Example:
            >>> agent = AnalysisWebSearchAgent("llama3.1:8b")
            >>> results = [
            ...     {"title": "Bitcoin Price", "link": "https://...", "body": "..."},
            ...     {"title": "Python Tutorial", "link": "https://...", "body": "..."}
            ... ]
            >>> scored = agent.web_search_score("Bitcoin price", results)
            >>> for r in scored:
            ...     print(f"{r['relevance_score']}/100 - {r['title']}")
        """
        scored_results = []
        
        for search in web_search_result:
            link = search.get('link', '')
            title = search.get('title', '')
            body = search.get('body', '')
            
            try:
                score = self.score_relevance(
                    query=query,
                    title=title,
                    link=link,
                    body=body
                )
                
                # Add score to result
                result_with_score = {**search, 'relevance_score': score}
                scored_results.append(result_with_score)
            
            except ScoreParseError as e:
                # Re-raise to let caller handle parsing failures
                raise e
            
            except Exception as e:
                # Add with error flag but continue processing
                result_with_score = {
                    **search,
                    'relevance_score': 50,
                    'score_error': str(e)
                }
                scored_results.append(result_with_score)
        
        # Sort by relevance score descending
        scored_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return scored_results
            

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
        
        context = f"""
            User Query: {query}

            Search Result:
            - Title: {title}
            - URL: {link}
            - Snippet: {body}
        """
        
        response = self.chat(context, stream=False)
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
    agent = AnalysisWebSearchAgent("llama3.1:8b")
    
    print("AnalysisWebSearchAgent initialized.\n")
    print("Enter query, title, link, and body to score relevance.\n")
    print("Type 'quit' or 'exit' at any prompt to end.\n")
    
    test_data = {
        "query": "Bitcoin price today",
        "title": "Bitcoin Price Today - Live BTC Price Chart",
        "link": "https://coinmarketcap.com/currencies/bitcoin/",
        "body": "Bitcoin price today is $42,350 USD with a 24-hour trading volume of $15B. Price increased by 3.5% in the last 24h."
    }
    
    while True:
        try:
            print("\n--- Test Data ---")
            print(f"Query: {test_data['query']}")
            print(f"Title: {test_data['title']}")
            print(f"Link: {test_data['link']}")
            print(f"Body: {test_data['body'][:80]}...")
            print("-----------------\n")
            
            cmd = input("Enter 'edit' to modify test data, 'score' to calculate score, or 'quit' to exit: ")
            
            if cmd.strip().lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            
            if cmd.strip().lower() == "edit":
                print("\nEnter new values (press Enter to keep current value):\n")
                
                new_query = input(f"Query [{test_data['query']}]: ").strip()
                if new_query:
                    test_data['query'] = new_query
                
                new_title = input(f"Title [{test_data['title']}]: ").strip()
                if new_title:
                    test_data['title'] = new_title
                
                new_link = input(f"Link [{test_data['link']}]: ").strip()
                if new_link:
                    test_data['link'] = new_link
                
                new_body = input(f"Body [{test_data['body']}]: ").strip()
                if new_body:
                    test_data['body'] = new_body
                
                continue
            
            if cmd.strip().lower() == "score":
                print("\nCalculating relevance score...\n")
                
                try:
                    score = agent.score_relevance(
                        query=test_data['query'],
                        title=test_data['title'],
                        link=test_data['link'],
                        body=test_data['body']
                    )
                    print(f"✓ Relevance Score: {score}/100\n")
                
                except ScoreParseError as e:
                    print(f"✗ Parse Error: {e}\n")
                
                except Exception as e:
                    print(f"✗ Error: {e}\n")
            
            else:
                print("Invalid command. Use 'edit', 'score', or 'quit'.\n")
        
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break