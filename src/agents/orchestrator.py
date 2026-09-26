import sys
from pathlib import Path
import math
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.agents"

from ..scraper import scrape
from .user_input_agent import UserInputAgent
from .analysis_agent import AnalysisWebSearchAgent
from .no_tools_calling_agent import SimpleAgent
from ..tools.web_search import search_web
from .scrapping_agent import ScrappingAgent
from .synthesis_agent import SynthesisAgent


user = UserInputAgent("llama3.1:8b")
analysis_agent = AnalysisWebSearchAgent("llama3.2:3b")
scrapping_agent = ScrappingAgent("llama3.2:3b")
synthesis_agent = SynthesisAgent("llama3.2:3b")

def process_source(
    result: Dict[str, Any],
    query: str,
    scrapping_agent: ScrappingAgent,
) -> Dict[str, Any] | None:
    """Scrape and extract info from one source.
    
    Args:
        result: Scored search result.
        query: Original user query.
        scrapping_agent: ScrappingAgent instance.
    
    Returns:
        Extracted info dict or None if failed.
    """
    link = result.get('link', '')
    body = result.get('body', '')
    
    # Skip scraping if snippet is rich enough
    if len(body) > 250 and any(kw in body.lower() for kw in ['album', 'sortie', 'released', '2026', '2025', '2024']):
        print(f"  ✓ Snippet riche, skip scraping: {result.get('title', '')[:50]}")
        return {
            'query': query,
            'source_url': link,
            'source_title': result.get('title', ''),
            'relevance_score': result.get('relevance_score', 0),
            'extracted_info': body,
            'tool_calls': [],
            'tool_results': [],
        }
    
    # Scrape with fallback
    print(f"  [Scraping] {link}...")
    scrape_result = scrape(link, force_playwright=False)
    
    if not scrape_result['success'] or len(scrape_result.get('markdown', '')) < 500:
        print(f"  ⚠️  httpx failed/low content, retrying with Playwright...")
        scrape_result = scrape(link, force_playwright=True)
    
    if scrape_result['success']:
        scrapping_agent.load_markdown(
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
        
        response = scrapping_agent.chat(query=user_query, stream=False)
        scrapping_agent.reset()
        
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
        print(f"  ✗ Scraping failed: {scrape_result.get('error', 'unknown error')}")
        return None


while True:
    try:
        query = input("Vous : ")
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break
    
    if query.strip().lower() in ("quit", "exit"):
        print("Goodbye!")
        break
    
    # Decompose query
    questions = user.chat(query, stream=True)
    user.reset()
    content = user.parse_subrequests(questions.content)
    
    # Collect all search results
    all_search_results: List[Dict[str, Any]] = []
    
    for index, question in enumerate(content, 1):
        print(f"\n=== Sub-request {index}: {question} ===\n")
        web_search = search_web(query=question, max_results=5)
        print(f"Found {len(web_search)} results\n")
        
        for item in web_search:
            item_with_question = {**item, 'question': question}
            all_search_results.append(item_with_question)
    
    # Score all results in BATCH
    print(f"\n[Analyse] Scoring en batch de {len(all_search_results)} résultats...")
    
    if all_search_results:
        analysed = analysis_agent.web_search_score(
            query=query,
            web_search_result=all_search_results,
        )
        
        for result in analysed[:10]:
            print(
                f"  ✓ [{result['relevance_score']}/100] "
                f"{result.get('title', 'N/A')[:50]}"
            )
    else:
        analysed = []
    
    if analysed:
        # Sort and filter
        analysed.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Take top 3 with score >= 75, or fallback to top 1
        top_results = [
            res for res in analysed 
            if res.get('relevance_score', 0) >= 75
        ][:3]
        
        if not top_results and analysed:
            top_results = [analysed[0]]
        
        print(f"\n[Scraping] {len(top_results)} sources to process (parallel)...")
        
        # Parallel scraping
        final_responses = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(process_source, result, query, scrapping_agent)
                for result in top_results
            ]
            for future in as_completed(futures):
                response = future.result()
                if response:
                    final_responses.append(response)
        
        # Synthesize final answer
        if final_responses:
            source_context = "\n\n".join(
                f"Source: {response['source_url']}\n"
                f"Title: {response['source_title']}\n"
                f"Relevance: {response['relevance_score']}/100\n"
                f"Extracted information:\n{response['extracted_info']}"
                for response in final_responses
            )
            
            synthesis_prompt = (
                f"Original user question: {query}\n\n"
                f"Extracted source information:\n{source_context}\n\n"
                "Write the final answer in the language of the original question. "
                "Cite sources where relevant."
            )
            
            final_answer = synthesis_agent.synthesize(
                query=query,
                extracted_sources=final_responses,
                stream=False,
            )
            synthesis_agent.reset()
            analysis_agent.save_final_respose(final_response=final_answer)
        else:
            final_answer = "No usable source could be scraped for this question."
            analysis_agent.save_final_respose(final_response=final_answer)
        
        print("\n" + "="*50)
        print("FINAL RESPONSE")
        print("="*50 + "\n")
        print(final_answer)
        print()
        
        # Print sources
        print("\n" + "="*50)
        print("SOURCES")
        print("="*50 + "\n")
        
        for i, resp in enumerate(final_responses, 1):
            print(f"--- Source {i}: {resp['source_title']} ---")
            print(f"URL: {resp['source_url']}")
            print(f"Relevance: {resp['relevance_score']}/100")
        
        print("="*50)