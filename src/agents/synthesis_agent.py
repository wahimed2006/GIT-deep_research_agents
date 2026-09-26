"""SynthesisAgent module.

This module provides a specialized agent for synthesizing extracted source
information into clear, concise, and well-structured final answers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path if running as script
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.agents"

from .no_tools_calling_agent import SimpleAgent
from ..prompt.synthesis_agent_prompt import SYNTHESIS_AGENT_PROMPT


class SynthesisAgent(SimpleAgent):
    """Agent specialized in synthesizing extracted information into final answers.
    
    This agent takes extracted source information and produces clear, concise
    answers adapted to the complexity of the question.
    
    Attributes:
        model_name: Name of the LLM model to use.
    """
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        """Initialize the SynthesisAgent.
        
        Args:
            model_name: Name of the LLM model (default: llama3.2:3b).
        """
        super().__init__(model_name, SYNTHESIS_AGENT_PROMPT)
        self.options = {
            "num_predict": 800,
            "temperature": 0.3,
            "num_ctx": 4096
        }
    
    def synthesize(
        self,
        query: str,
        extracted_sources: List[Dict[str, Any]],
        stream: bool = False,
    ) -> str:
        """Synthesize extracted sources into a final answer.
        
        Args:
            query: Original user query.
            extracted_sources: List of dicts with 'source_url', 'source_title',
                              'relevance_score', 'extracted_info'.
            stream: Whether to stream the response.
        
        Returns:
            Final synthesized answer as a string.
        
        Example:
            >>> agent = SynthesisAgent("llama3.2:3b")
            >>> sources = [
            ...     {
            ...         "source_url": "https://coinbase.com/...",
            ...         "source_title": "Bitcoin Price",
            ...         "relevance_score": 95,
            ...         "extracted_info": "Bitcoin is $73,898.60 EUR..."
            ...     }
            ... ]
            >>> answer = agent.synthesize("What is Bitcoin's price?", sources)
            >>> print(answer)
        """
        # Build source context
        source_context = "\n\n".join([
            f"Source: {src['source_url']}\n"
            f"Title: {src['source_title']}\n"
            f"Relevance: {src['relevance_score']}/100\n"
            f"Extracted information:\n{src['extracted_info']}"
            for src in extracted_sources
        ])
        
        # Build synthesis prompt
        synthesis_prompt = (
            f"Original user question: {query}\n\n"
            f"Extracted source information:\n{source_context}\n\n"
            "Write the final answer in the language of the original question. "
            "Cite sources where relevant."
        )
        
        # Get response
        response = self.chat(synthesis_prompt, stream=stream)
        self.reset()
        
        return response.content
    
    def synthesize_with_metadata(
        self,
        query: str,
        extracted_sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Synthesize and return answer with metadata.
        
        Args:
            query: Original user query.
            extracted_sources: List of extracted source dicts.
        
        Returns:
            Dict with 'answer', 'sources_count', 'source_urls'.
        """
        answer = self.synthesize(query, extracted_sources)
        
        return {
            "answer": answer,
            "sources_count": len(extracted_sources),
            "source_urls": [src["source_url"] for src in extracted_sources],
            "query": query,
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    agent = SynthesisAgent("llama3.2:3b")
    
    # Example sources
    test_sources = [
        {
            "source_url": "https://coinbase.com/price/bitcoin",
            "source_title": "Bitcoin Price - Coinbase",
            "relevance_score": 95,
            "extracted_info": "Bitcoin's current price is $73,898.60 EUR (+0.75% in 24h). Market cap: €1.484B, 24h volume: €14.438B."
        },
        {
            "source_url": "https://coinmarketcap.com/currencies/bitcoin/",
            "source_title": "Bitcoin Price Today - CoinMarketCap",
            "relevance_score": 92,
            "extracted_info": "Bitcoin market cap is €1.484 billion. All-time high: €110,798.68. 24h trading volume: €14.438 billion."
        },
    ]
    
    # Test simple question
    print("=== Test 1: Simple Factual Question ===\n")
    answer = agent.synthesize(
        query="Quel est le prix du Bitcoin ?",
        extracted_sources=test_sources,
    )
    print(answer)
    print("\n")
    
    # Test with metadata
    print("=== Test 2: With Metadata ===\n")
    result = agent.synthesize_with_metadata(
        query="What is Bitcoin's market cap?",
        extracted_sources=test_sources,
    )
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Sources: {result['sources_count']}")
    print(f"URLs: {result['source_urls']}")