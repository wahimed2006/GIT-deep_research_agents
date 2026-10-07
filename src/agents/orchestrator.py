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
from .rooter_agent import RouterAgent
from ..scraper import scrape_multiple
import asyncio
import httpx


# =============================================================================
# GLOBAL MODEL CONFIGURATION
# =============================================================================

MODELS = {
    "user_input": "llama3.2:3b",
    "analysis": "llama3.2:3b",
    "scrapping": "llama3.2:3b",
    "synthesis": "llama3.2:3b",
    "router_agent": "llama3.2:3b",
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
        self.router_agent = RouterAgent(self.models["router_agent"])
    
    async def process_source(
        self,
        result: Dict[str, Any],
        query: str,
        client: httpx.AsyncClient,
        playwright_semaphore: asyncio.Semaphore | None = None,
    ) -> Dict[str, Any] | None:
        """
        Scrape and extract information from one source asynchronously.

        The function first checks whether the search-engine snippet already
        contains enough useful information. If not, it delegates the complete
        HTTPX -> Playwright fallback strategy to the asynchronous scraper.

        Args:
            result:
                Scored search result.

            query:
                Original user query.

            client:
                Shared asynchronous HTTPX client used by the scraper.

            playwright_semaphore:
                Optional semaphore limiting concurrent Playwright executions.

        Returns:
            Extracted information dictionary, or None if the source could not
            be processed successfully.
        """
        import time

        started_at = time.perf_counter()

        link = result.get("link", "")
        body = result.get("body", "")

        # ---------------------------------------------------------
        # 1. Use search snippet directly when it is sufficiently rich
        # ---------------------------------------------------------
        if len(body) > 250 and any(
            kw in body.lower()
            for kw in [
                "album",
                "sortie",
                "released",
                "2026",
                "2025",
                "2024",
                "price",
                "market cap",
            ]
        ):
            print(
                f"  [OK] Snippet riche, skip scraping: "
                f"{result.get('title', '')[:50]}"
            )

            return {
                "query": query,
                "source_url": link,
                "source_title": result.get("title", ""),
                "relevance_score": result.get("relevance_score", 0),
                "extracted_info": body,
                "tool_calls": [],
                "tool_results": [],
                "scrape_data": {
                    "success": True,
                    "method": "search_snippet",
                    "markdown_length": len(body),
                },
                "duration_seconds": time.perf_counter() - started_at,
            }

        # ---------------------------------------------------------
        # 2. Scrape asynchronously
        # ---------------------------------------------------------
        print(f"  [Scraping] {link}...")

        try:
            scrape_result = await scrape(
                link,
                client=client,
                playwright_semaphore=playwright_semaphore,
            )

        except Exception as e:
            print(
                f"  [FAIL] Unexpected scraping error: "
                f"{type(e).__name__}: {e}"
            )

            return {
                "query": query,
                "source_url": link,
                "source_title": result.get("title", ""),
                "relevance_score": result.get("relevance_score", 0),
                "extracted_info": "",
                "tool_calls": [],
                "tool_results": [],
                "scrape_data": {
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                "duration_seconds": time.perf_counter() - started_at,
            }

        # ---------------------------------------------------------
        # 3. Check scraping result
        # ---------------------------------------------------------
        if not scrape_result.get("success"):
            print(
                f"  [FAIL] Scraping failed: "
                f"{scrape_result.get('error', 'unknown error')}"
            )

            return None

        markdown = scrape_result.get("markdown", "")

        if len(markdown) < 500:
            print(
                f"  [!] Contenu récupéré insuffisant "
                f"({len(markdown)} caractères): {link}"
            )

            return None

        # ---------------------------------------------------------
        # 4. Extract relevant information with ScrappingAgent
        # ---------------------------------------------------------
        scrapping_agent = ScrappingAgent(
            self.models["scrapping"]
        )

        scrapping_agent.load_markdown(
            markdown=markdown,
            url=link,
            structured=scrape_result.get("structured"),
        )

        user_query = f"""
    USER INITIAL QUERY: {query}

    Context:
    - PAGE TITLE: {result.get('title', '')}
    - PAGE DESCRIPTION: {result.get('body', '')}
    - RELEVANCE SCORE: {result.get('relevance_score', 0)}/100

    Extract information that answers the query.
    """

        try:
            response = scrapping_agent.chat(
                query=user_query,
                stream=False,
            )

        except Exception as e:
            print(
                f"  [FAIL] ScrappingAgent failed: "
                f"{type(e).__name__}: {e}"
            )

            return None

        # ---------------------------------------------------------
        # 5. Return normalized result
        # ---------------------------------------------------------
        return {
            "query": query,
            "source_url": link,
            "source_title": result.get("title", ""),
            "relevance_score": result.get("relevance_score", 0),
            "extracted_info": response.content,
            "tool_calls": response.tool_calls,
            "tool_results": response.tool_results,
            "scrape_data": scrape_result,
            "duration_seconds": time.perf_counter() - started_at,
        }
    
    async def research(
        self,
        query: str,
        max_search_results: int = 5,
        max_sources: int = 3,
        min_score: int = 75,
        parallel_workers: int = 3,
        telemetry: Dict[str, Any] | None = None,
    ) -> ResearchResult:
        """
        Execute the full research workflow for a user query.

        The research pipeline is composed of several stages:

            1. Query decomposition
            The user's query is decomposed into multiple sub-questions
            using the user input agent.

            2. Web search
            Each sub-question is searched independently on the web.
            Search failures are handled individually so that a failed
            search does not stop the rest of the research workflow.

            3. Search result analysis
            All collected search results are scored according to their
            relevance to the original query.

            4. Source selection
            Search results are sorted by relevance score and the most
            relevant sources are selected according to `min_score` and
            `max_sources`.

            5. Concurrent scraping
            The selected sources are scraped concurrently using the
            asynchronous `scrape_multiple` pipeline. This replaces the
            previous ThreadPoolExecutor-based scraping implementation and
            allows the scraper to reuse an asynchronous HTTP client and
            bounded concurrency.

            6. Answer synthesis
            The successfully scraped sources are passed to the synthesis
            agent, which generates the final answer to the user's query.

            7. Telemetry
            Execution times and intermediate results are recorded in the
            provided telemetry dictionary.

        Args:
            query:
                The original research query provided by the user.

            max_search_results:
                Maximum number of web search results retrieved for each
                generated sub-question.

            max_sources:
                Maximum number of sources selected for scraping and final
                synthesis.

            min_score:
                Minimum relevance score required for a search result to be
                selected as a source. If no result reaches this threshold,
                the highest-scoring result is selected as a fallback.

            parallel_workers:
                Maximum number of sources that can be scraped concurrently.
                This value is passed to `scrape_multiple` as its HTTP
                concurrency limit.

            telemetry:
                Optional dictionary used to store execution telemetry.
                When provided, it is updated in place with the query,
                generated sub-questions, search results, ranked results,
                selected sources, scraping results, and execution times.

        Returns:
            ResearchResult:
                A research result containing:
                    - the synthesized final answer,
                    - the original query,
                    - the URLs of successfully scraped sources,
                    - the number of successfully scraped sources,
                    - metadata describing the research execution.

        Raises:
            This method handles search and scraping errors internally.
            Individual source failures do not normally stop the research
            workflow. Unexpected exceptions raised outside the handled
            stages may still propagate to the caller.

        Notes:
            This method is asynchronous because the scraping stage uses
            `scrape_multiple`, which performs concurrent asynchronous
            scraping.

            Therefore, callers must await this method:

                result = await orchestrator.research(query)

            The `parallel_workers` parameter controls the maximum HTTP
            scraping concurrency. The Playwright concurrency is controlled
            separately inside `scrape_multiple`.
        """

        import time

        start_time = time.perf_counter()

        telemetry = telemetry if telemetry is not None else {}

        telemetry.update({
            "query": query,
            "stages": {},
            "questions_found": [],
            "search_results": [],
            "ranked_results": [],
            "selected_results": [],
            "scrape_results": [],
        })

        def mark_stage(
            name: str,
            started: float,
        ) -> None:
            """
            Record the elapsed time of a research stage.
            """
            telemetry["stages"][name] = (
                time.perf_counter() - started
            )

        # ============================================================
        # Step 1: Decompose query
        # ============================================================

        print(
            "\n[Decomposition] Breaking down query..."
        )

        stage_started = time.perf_counter()

        questions = self.user_agent.chat(
            query,
            stream=True,
        )

        self.user_agent.reset()

        content = self.user_agent.parse_subrequests(
            questions.content
        )

        telemetry["questions_found"] = content

        mark_stage(
            "decomposition",
            stage_started,
        )

        # ============================================================
        # Step 2: Collect search results
        # ============================================================

        all_search_results: List[Dict[str, Any]] = []

        stage_started = time.perf_counter()

        for index, question in enumerate(content, 1):

            print(
                f"\n=== Sub-request {index}: {question} ===\n"
            )

            try:

                web_search = search_web(
                    query=question,
                    max_results=max_search_results,
                )

                print(
                    f"Found {len(web_search)} results\n"
                )

                for item in web_search:

                    item_with_question = {
                        **item,
                        "question": question,
                    }

                    all_search_results.append(
                        item_with_question
                    )

            except Exception as e:

                print(
                    f"  ✗ Search failed for "
                    f"'{question}': {e}"
                )

                # Continue with the remaining sub-questions.

        telemetry["search_results"] = (
            all_search_results
        )

        mark_stage(
            "search",
            stage_started,
        )

        # ============================================================
        # Handle no search results
        # ============================================================

        if not all_search_results:

            elapsed_time = (
                time.perf_counter() - start_time
            )

            telemetry["stages"]["total"] = elapsed_time

            return ResearchResult(
                answer=(
                    f"Aucun résultat trouvé pour : "
                    f"{query}"
                ),
                query=query,
                sources=[],
                sources_count=0,
                metadata={
                    "elapsed_time": elapsed_time,
                    "error": "No search results found",
                    "sub_questions": len(content),
                },
            )

        # ============================================================
        # Step 3: Score all search results
        # ============================================================

        print(
            f"\n[Analyse] Scoring en batch de "
            f"{len(all_search_results)} résultats..."
        )

        stage_started = time.perf_counter()

        try:

            analysed = (
                self.analysis_agent.web_search_score(
                    query=query,
                    web_search_result=all_search_results,
                )
            )

            for result in analysed[:10]:

                print(
                    f"  ✓ [{result['relevance_score']}/100] "
                    f"{result.get('title', 'N/A')[:50]}"
                )

        except Exception as e:

            print(
                f"  ✗ Scoring failed: {e}"
            )

            # Fallback:
            # keep all search results with a neutral score.
            analysed = [
                {
                    **result,
                    "relevance_score": 50,
                }
                for result in all_search_results
            ]

        telemetry["ranked_results"] = analysed

        mark_stage(
            "analysis",
            stage_started,
        )

        # ============================================================
        # Step 4: Select top sources
        # ============================================================

        if analysed:

            analysed.sort(
                key=lambda x: x.get(
                    "relevance_score",
                    0,
                ),
                reverse=True,
            )

            top_results = [
                result
                for result in analysed
                if result.get(
                    "relevance_score",
                    0,
                ) >= min_score
            ][:max_sources]

            # Fallback:
            # if no result reaches min_score, use the
            # highest-scoring result.
            if not top_results:

                top_results = [
                    analysed[0]
                ]

        else:

            top_results = []

        telemetry["selected_results"] = (
            top_results
        )

        # ============================================================
        # Step 5: Concurrent scraping
        # ============================================================

        print(
            f"\n[Scraping] "
            f"{len(top_results)} sources to process "
            f"(async parallel)..."
        )

        stage_started = time.perf_counter()

        final_responses: List[Dict[str, Any]] = []

        if top_results:

            # Extract URLs from the selected search results.
            urls = [
                result["link"]
                for result in top_results
                if result.get("link")
            ]

            if urls:

                scrape_results = await scrape_multiple(
                    urls=urls,
                    max_concurrent=parallel_workers,
                    max_playwright_concurrent=2,
                )

                # Combine the original search metadata with
                # the scraped content.
                for result, scrape_result in zip(
                    top_results,
                    scrape_results,
                ):

                    if scrape_result.get("success"):

                        final_responses.append({
                            "query": query,
                            "source_url": result.get("link", scrape_result.get("url", "")),
                            "source_title": result.get("title", ""),
                            "relevance_score": result.get("relevance_score", 0),
                            "extracted_info": scrape_result.get("markdown", ""),
                            "tool_calls": [],
                            "tool_results": [],
                            "scrape_data": scrape_result,
                        })

                    else:

                        print(
                            f"  ✗ Scraping failed: "
                            f"{result.get('link', 'N/A')} - "
                            f"{scrape_result.get('error', 'Unknown error')}"
                        )

        telemetry["scrape_results"] = (
            final_responses
        )

        mark_stage(
            "scraping",
            stage_started,
        )

        # ============================================================
        # Step 6: Synthesize final answer
        # ============================================================

        stage_started = time.perf_counter()

        if final_responses:

            final_answer = (
                self.synthesis_agent.synthesize(
                    query=query,
                    extracted_sources=final_responses,
                    stream=False,
                )
            )

            self.synthesis_agent.reset()

        else:

            final_answer = (
                "No usable source could be scraped "
                "for this question."
            )

        mark_stage(
            "synthesis",
            stage_started,
        )

        # ============================================================
        # Step 7: Build final result
        # ============================================================

        elapsed_time = (
            time.perf_counter() - start_time
        )

        telemetry["stages"]["total"] = elapsed_time

        result = ResearchResult(
            answer=final_answer,
            query=query,
            sources=[
                response["source_url"]
                for response in final_responses
                if response.get("source_url")
            ],
            sources_count=len(final_responses),
            metadata={
                "elapsed_time": elapsed_time,
                "sub_questions": len(content),
                "total_search_results": len(
                    all_search_results
                ),
                "scored_results": len(analysed),
                "scraped_sources": len(
                    final_responses
                ),
                "models_used": self.models,
            },
        )

        return result
    
    async def research_and_print(
        self,
        query: str,
        show_sources: bool = True,
    ) -> ResearchResult:
        """
        Execute the research workflow and print the formatted result.

        This method is a convenience wrapper around the asynchronous
        `research` method. It executes the complete research pipeline,
        displays the final answer in the console, and optionally displays
        the sources used to generate the answer.

        Args:
            query:
                The user's research question.

            show_sources:
                Whether to display the source URLs used during the research.

        Returns:
            ResearchResult:
                The complete research result containing the final answer,
                source information, and research metadata.

        Notes:
            This method is asynchronous because the underlying `research`
            method uses asynchronous source scraping.

            Therefore, callers must await this method:

                result = await orchestrator.research_and_print(query)
        """

        result = await self.research(query)

        print("\n" + "=" * 50)
        print("FINAL RESPONSE")
        print("=" * 50 + "\n")

        print(result.answer)
        print()

        if show_sources and result.sources:

            print("\n" + "=" * 50)
            print("SOURCES")
            print("=" * 50 + "\n")

            for i, source_url in enumerate(
                result.sources,
                1,
            ):
                print(f"--- Source {i} ---")
                print(f"URL: {source_url}")

            print("=" * 50)

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
        
        result = asyncio.run(
            orchestrator.research_and_print(query)
        )
        orchestrator.reset()