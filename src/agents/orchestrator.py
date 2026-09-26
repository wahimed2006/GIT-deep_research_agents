"""Orchestrator module.

This module provides the ResearchOrchestrator class that coordinates
all agents for end-to-end research queries.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.agents"

from ..scraper import scrape
from ..tools import AgentResponse
from ..tools.web_search import search_web
from .user_input_agent import UserInputAgent
from .analysis_agent import AnalysisWebSearchAgent
from .scrapping_agent import ScrappingAgent
from .synthesis_agent import SynthesisAgent


# =============================================================================
# GLOBAL MODEL CONFIGURATION
# =============================================================================

MODELS = {
    "user_input": "llama3.1:8b",
    "analysis": "llama3.2:3b",
    "scrapping": "llama3.2:3b",
    "synthesis": "llama3.2:3b",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ResearchResult:
    """Final result of a research query.
    
    Attributes:
        answer: Final synthesized answer.
        query: Original user query.
        sources: List of source URLs used.
        sources_count: Number of sources used.
        metadata: Additional metadata (scores, timestamps, etc.).
    """
    answer: str
    query: str
    sources: List[str] = field(default_factory=list)
    sources_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_agent_response(self) -> AgentResponse[str]:
        """Convert to AgentResponse format.
        
        Returns:
            AgentResponse with answer as content and metadata.
        """
        return AgentResponse[str](
            content=self.answer,
            tool_calls=[],  # No tool calls from orchestrator
            tool_results=[],  # No tool results from orchestrator
            metadata={
                **self.metadata,  # Merge existing metadata
                "sources": self.sources,
                "sources_count": self.sources_count,
                "query": self.query,
            },
            structured_data=None,  # Not used for orchestrator
            raw_response=None,  # Not used for orchestrator
        )


# =============================================================================
# ORCHESTRATOR CLASS
# =============================================================================

class ResearchOrchestrator:
    """Orchestrate multi-agent research workflow.
    
    This class coordinates UserInputAgent, AnalysisAgent, ScrappingAgent,
    and SynthesisAgent to answer research queries end-to-end.
    
    Attributes:
        models: Dictionary of model names for each agent.
        user_agent: UserInputAgent instance.
        analysis_agent: AnalysisWebSearchAgent instance.
        scrapping_agent: ScrappingAgent instance.
        synthesis_agent: SynthesisAgent instance.
    """
    
    def __init__(
        self,
        models: Dict[str, str] | None = None,
    ):
        """Initialize the orchestrator.
        
        Args:
            models: Optional custom model configuration.
                   Defaults to global MODELS dict.
        """
        self.models = models if models is not None else MODELS.copy()
        
        # Initialize agents
        self.user_agent = UserInputAgent(self.models["user_input"])
        self.analysis_agent = AnalysisWebSearchAgent(self.models["analysis"])
        self.scrapping_agent = ScrappingAgent(self.models["scrapping"])
        self.synthesis_agent = SynthesisAgent(self.models["synthesis"])
    
    def process_source(
        self,
        result: Dict[str, Any],
        query: str,
    ) -> Dict[str, Any] | None:
        """Scrape and extract info from one source with thread isolation.
        
        Args:
            result: Scored search result.
            query: Original user query.
        
        Returns:
            Extracted info dict or None if failed.
        """
        link = result.get('link', '')
        body = result.get('body', '')
        
        if len(body) > 250 and any(
            kw in body.lower() for kw in 
            ['album', 'sortie', 'released', '2026', '2025', '2024', 'price', 'market cap']
        ):
            print(f"  [OK] Snippet riche, skip scraping: {result.get('title', '')[:50]}")
            return {
                'query': query,
                'source_url': link,
                'source_title': result.get('title', ''),
                'relevance_score': result.get('relevance_score', 0),
                'extracted_info': body,
                'tool_calls': [],
                'tool_results': [],
            }
        
        print(f"  [Scraping] {link}...")
        scrape_result = scrape(link, force_playwright=False)
        
        if not scrape_result['success'] or len(scrape_result.get('markdown', '')) < 500:
            print(f"  [!] Contenu insuffisant/échec httpx, bascule Playwright...")
            scrape_result = scrape(link, force_playwright=True)
        
        if scrape_result['success']:
            # Instance dédiée par thread pour garantir l'isolation mémoire
            thread_scrapping_agent = ScrappingAgent(self.models["scrapping"])
            
            thread_scrapping_agent.load_markdown(
                markdown=scrape_result["markdown"],
                url=link,
                structured=scrape_result.get("structured"),
            )
            
            user_query = f"""
USER INITIAL QUERY: {query}

Context:
- PAGE TITLE: {result.get('title', '')}
- PAGE DESCRIPTION: {result.get('body', '')}
- RELEVANCE SCORE: {result.get('relevance_score', 0)}/100

Extract information that answers the query."""
            
            response = thread_scrapping_agent.chat(query=user_query, stream=False)
            
            return {
                'query': query,
                'source_url': link,
                'source_title': result.get('title', ''),
                'relevance_score': result.get('relevance_score', 0),
                'extracted_info': response.content,
                'tool_calls': response.tool_calls,
                'tool_results': response.tool_results,
            }
        else:
            print(f"  [FAIL] Scraping failed: {scrape_result.get('error', 'unknown error')}")
            return None
    
    def research(
        self,
        query: str,
        max_search_results: int = 5,
        max_sources: int = 3,
        min_score: int = 75,
        parallel_workers: int = 3,
    ) -> ResearchResult:
        """Execute full research workflow for a query."""
        import time
        start_time = time.time()
        
        # Step 1: Decompose query
        print(f"\n[Decomposition] Breaking down query...")
        questions = self.user_agent.chat(query, stream=True)
        self.user_agent.reset()
        content = self.user_agent.parse_subrequests(questions.content)
        
        # Step 2: Collect search results with error handling
        all_search_results: List[Dict[str, Any]] = []
        
        for index, question in enumerate(content, 1):
            print(f"\n=== Sub-request {index}: {question} ===\n")
            
            try:
                web_search = search_web(query=question, max_results=max_search_results)
                print(f"Found {len(web_search)} results\n")
                
                for item in web_search:
                    item_with_question = {**item, 'question': question}
                    all_search_results.append(item_with_question)
            
            except Exception as e:
                print(f"  ✗ Search failed for '{question}': {e}")
                # Continue with other sub-questions
        
        # Handle no results
        if not all_search_results:
            elapsed_time = time.time() - start_time
            return ResearchResult(
                answer=f"Aucun résultat trouvé pour : {query}",
                query=query,
                sources=[],
                sources_count=0,
                metadata={
                    "elapsed_time": elapsed_time,
                    "error": "No search results found",
                    "sub_questions": len(content),
                },
            )
        
        # Step 3: Score all results in batch
        print(f"\n[Analyse] Scoring en batch de {len(all_search_results)} résultats...")
        
        try:
            analysed = self.analysis_agent.web_search_score(
                query=query,
                web_search_result=all_search_results,
            )
            
            for result in analysed[:10]:
                print(
                    f"  ✓ [{result['relevance_score']}/100] "
                    f"{result.get('title', 'N/A')[:50]}"
                )
        except Exception as e:
            print(f"  ✗ Scoring failed: {e}")
            # Fallback: use all results with score 50
            analysed = [{**r, 'relevance_score': 50} for r in all_search_results]
        
        # Step 4: Filter top sources
        if analysed:
            analysed.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            top_results = [
                res for res in analysed 
                if res.get('relevance_score', 0) >= min_score
            ][:max_sources]
            
            if not top_results and analysed:
                top_results = [analysed[0]]
        else:
            top_results = []
        
        # Step 5: Parallel scraping
        print(f"\n[Scraping] {len(top_results)} sources to process (parallel)...")
        
        final_responses = []
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [
                executor.submit(self.process_source, result, query)
                for result in top_results
            ]
            for future in as_completed(futures):
                response = future.result()
                if response:
                    final_responses.append(response)
        
        # Step 6: Synthesize final answer
        if final_responses:
            final_answer = self.synthesis_agent.synthesize(
                query=query,
                extracted_sources=final_responses,
                stream=False,
            )
            self.synthesis_agent.reset()
        else:
            final_answer = "No usable source could be scraped for this question."
        
        # Step 7: Build result
        elapsed_time = time.time() - start_time
        
        result = ResearchResult(
            answer=final_answer,
            query=query,
            sources=[resp['source_url'] for resp in final_responses],
            sources_count=len(final_responses),
            metadata={
                "elapsed_time": elapsed_time,
                "sub_questions": len(content),
                "total_search_results": len(all_search_results),
                "scored_results": len(analysed),
                "scraped_sources": len(final_responses),
                "models_used": self.models,
            },
        )
        
        return result
    
    def research_and_print(
        self,
        query: str,
        show_sources: bool = True,
    ) -> ResearchResult:
        """Execute research and print formatted output.
        
        Args:
            query: User's research question.
            show_sources: Whether to print source details.
        
        Returns:
            ResearchResult with final answer.
        """
        result = self.research(query)
        
        print("\n" + "="*50)
        print("FINAL RESPONSE")
        print("="*50 + "\n")
        print(result.answer)
        print()
        
        if show_sources and result.sources:
            print("\n" + "="*50)
            print("SOURCES")
            print("="*50 + "\n")
            
            for i, source_url in enumerate(result.sources, 1):
                print(f"--- Source {i} ---")
                print(f"URL: {source_url}")
            
            print("="*50)
        
        return result
    
    def reset(self) -> None:
        """Reset all agents."""
        self.user_agent.reset()
        self.analysis_agent.reset()
        self.scrapping_agent.reset()
        self.synthesis_agent.reset()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    orchestrator = ResearchOrchestrator()
    
    print("Research Orchestrator initialized.\n")
    print("Models:")
    for agent, model in orchestrator.models.items():
        print(f"  - {agent}: {model}")
    print("\n")
    
    # Interactive loop
    while True:
        try:
            query = input("Vous : ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if query.strip().lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        
        result = orchestrator.research_and_print(query)
        orchestrator.reset()