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
from ..prompt.analysis_agent_prompt import ANALYSIS_AGENT_PROMPT
import re


class ScoreParseError(Exception):
    """Raised when the score cannot be parsed from the response."""
    pass


class AnalysisWebSearchAgent(SimpleAgent):
    """An agent that scores the relevance of search results.
    
    This agent evaluates multiple search results in a single LLM call
    and assigns relevance scores from 0 to 100 for each.
    """
    
    def __init__(self, model_name: str):
        """Initialize the agent with a scoring-optimized model.
        
        Args:
            model_name: The name of the Ollama model to use.
        """
        super().__init__(model_name, ANALYSIS_AGENT_PROMPT)
        self.options = {
            "num_predict": 1000,  # Increased for batch scoring
            "temperature": 0.0,
            "num_ctx": 4096  # Increased context for multiple results
        }
        
    def web_search_score(
        self,
        query: str,
        web_search_result: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Score multiple search results in a SINGLE LLM call.
        
        Args:
            query: The user's search query.
            web_search_result: List of search results, each containing 'title',
                              'link', and 'body' keys.
        
        Returns:
            List of dictionaries with keys: 'query', 'title', 'link', 'body', 'relevance_score'.
            Sorted by score in descending order.
        """
        if not web_search_result:
            return []
        
        try:
            # Score all results in one batch
            scores = self.score_relevance_batch(query=query, results=web_search_result)
            
            # Build scored results
            scored_results = []
            for i, search in enumerate(web_search_result):
                result_with_score = {
                    'query': query,
                    'title': search.get('title', ''),
                    'link': search.get('href', search.get('link', '')),
                    'body': search.get('body', ''),
                    'relevance_score': scores[i] if i < len(scores) else 0,
                }
                scored_results.append(result_with_score)
            
            # Sort by relevance score descending
            scored_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return scored_results
            
        except Exception as e:
            # Fallback: score individually if batch fails
            print(f"Batch scoring failed, falling back to individual: {e}")
            return self._score_individual(query, web_search_result)
    
    def _score_individual(
        self,
        query: str,
        web_search_result: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fallback: score results one by one."""
        scored_results = []
        
        for search in web_search_result:
            link = search.get('href', search.get('link', ''))
            title = search.get('title', '')
            body = search.get('body', '')
            
            try:
                score = self.score_relevance(
                    query=query,
                    title=title,
                    link=link,
                    body=body
                )
            except Exception:
                score = 0
            
            result_with_score = {
                'query': query,
                'title': title,
                'link': link,
                'body': body,
                'relevance_score': score,
            }
            scored_results.append(result_with_score)
        
        scored_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return scored_results
    
    def score_relevance_batch(
        self,
        query: str,
        results: List[Dict[str, Any]],
        max_results: int = 10,
    ) -> List[int]:
        """Score multiple results in a single LLM call.
        
        Args:
            query: The user's search query.
            results: List of search results to score.
            max_results: Maximum number of results to include in batch.
        
        Returns:
            List of integer scores (0-100) for each result.
        """
        # Limit to avoid context overflow
        limited_results = results[:max_results]
        
        # Build batch context
        results_text = "\n\n".join([
            f"[Result {i+1}]\n"
            f"Title: {r.get('title', '')}\n"
            f"URL: {r.get('href', r.get('link', ''))}\n"
            f"Snippet: {r.get('body', '')[:500]}"  # Limit snippet length
            for i, r in enumerate(limited_results)
        ])
        
        context = f"""
User Query: {query}

Rate the relevance of each search result from 0 to 100.
Return ONLY a JSON array of scores in the same order as the results.

Example format: [85, 42, 91, 15, 67]

Search Results to Score:
{results_text}

Scores: """

        response = self.chat(context, stream=False)
        self.reset()
        
        # Parse scores from response
        scores = self.parse_scores_batch(response.content, expected_count=len(limited_results))
        return scores
    
    def score_relevance(
        self,
        query: str,
        title: str,
        link: str,
        body: str,
    ) -> int:
        """Score the relevance of a single search result (legacy method).
        
        Args:
            query: The user's search query.
            title: The title of the search result.
            link: The URL of the search result.
            body: The snippet/body text of the search result.
        
        Returns:
            An integer relevance score from 0 to 100.
        """
        context = f"""
User Query: {query}

Search Result:
- Title: {title}
- URL: {link}
- Snippet: {body}

Rate relevance from 0 to 100. Return only the score between <SCORE> tags.

Score: """
        
        response = self.chat(context, stream=False)
        self.reset()
        return self.parse_score(response.content)
    
    def parse_scores_batch(self, text: str, expected_count: int) -> List[int]:
        """Parse multiple scores from batch response.
        
        Args:
            text: Raw response text containing JSON array or comma-separated scores.
            expected_count: Expected number of scores.
        
        Returns:
            List of integer scores.
        """
        # Try to extract JSON array
        import json
        try:
            # Look for JSON array pattern
            match = re.search(r'\[\s*(\d+(\s*,\s*\d+)*)\s*\]', text)
            if match:
                scores = json.loads(match.group(0))
                if isinstance(scores, list) and all(isinstance(s, int) for s in scores):
                    return scores
        except:
            pass
        
        # Fallback: find all numbers 0-100
        numbers = re.findall(r"\b(\d{1,3})\b", text)
        scores = []
        for num_str in numbers:
            num = int(num_str)
            if 0 <= num <= 100:
                scores.append(num)
                if len(scores) >= expected_count:
                    break
        
        # If we got scores, return them
        if scores:
            return scores
        
        # Default: all zeros
        return [0] * expected_count
        
    def parse_score(self, text: str) -> int:
        """Parse a single relevance score from the agent's response.
            
        Args:
            text: Raw response text from the agent.
            
        Returns:
            An integer relevance score from 0 to 100.
            
        Raises:
            ScoreParseError: If no valid score can be extracted.
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
    
    print("AnalysisWebSearchAgent initialized (BATCH MODE).\n")
    
    test_results = [
        {
            "title": "Bitcoin Price Today - Live BTC Price Chart",
            "href": "https://coinmarketcap.com/currencies/bitcoin/",
            "body": "Bitcoin price today is $42,350 USD with a 24-hour trading volume of $15B."
        },
        {
            "title": "Ethereum Price & News",
            "href": "https://ethereum.org/",
            "body": "Ethereum is a decentralized platform for smart contracts."
        },
        {
            "title": "Python Programming Language",
            "href": "https://python.org/",
            "body": "Python is a programming language for general purposes."
        },
    ]
    
    print("Test Query: 'Bitcoin price today'\n")
    print(f"Results to score: {len(test_results)}\n")
    
    scored = agent.web_search_score(
        query="Bitcoin price today",
        web_search_result=test_results,
    )
    
    print("\n=== Scored Results ===")
    for i, result in enumerate(scored, 1):
        print(f"{i}. Score: {result['relevance_score']}/100 - {result['title']}")