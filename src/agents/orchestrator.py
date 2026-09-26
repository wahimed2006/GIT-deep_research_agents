import sys
from pathlib import Path
import math
from typing import Dict, List, Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.agents"

from ..scraper import scrape
from .user_input_agent import UserInputAgent
from .analysis_agent import AnalysisWebSearchAgent
from .no_tools_calling_agent import SimpleAgent
from ..tools.web_search import search_web
from .scrapping_agent import ScrappingAgent


user = UserInputAgent("llama3.2:3b")
analysis_agent = AnalysisWebSearchAgent("llama3.2:3b")
scrapping_agent = ScrappingAgent("llama3.2:3b")
synthesis_agent = SimpleAgent(
    "llama3.2:3b",
    system_prompt=(
        "You are a research synthesis agent. Answer the user's original question "
        "using only the extracted source information provided. Reconcile conflicts, "
        "avoid unsupported claims, and cite the relevant source URLs."
    )
)


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
        
        # Add question to each result for tracking
        for item in web_search:
            item_with_question = {**item, 'question': question}
            all_search_results.append(item_with_question)
    
    # Score all results in BATCH (single LLM call)
    print(f"\n[Analyse] Scoring en batch de {len(all_search_results)} résultats...")
    
    if all_search_results:
        # Use batch scoring
        analysed = analysis_agent.web_search_score(
            query=query,  # Score against original query
            web_search_result=all_search_results,
        )
        
        # Print scores
        for result in analysed[:10]:  # Show top 10
            print(
                f"  ✓ [{result['relevance_score']}/100] "
                f"{result.get('title', 'N/A')[:50]}"
            )
    else:
        analysed = []
    
    if analysed:
        # Sort by relevance score descending
        analysed.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Take top half (rounded up)
        top_half_count = math.ceil(len(analysed) / 2)
        top_results = [
            res for res in analysed 
            if res.get('relevance_score', 0) >= 75
        ][:3]
        if not top_results and analysed:
            top_results = [analysed[0]]
        
        final_responses = []
        
        for result in top_results:
            # Scrape the page
            link = result.get('link', '')
            print(f"\n[Scraping] {link}...")
            
            scrape_result = scrape(link)
            
            if scrape_result['success']:
                # Load Markdown into scrapping agent
                scrapping_agent.load_markdown(
                    markdown=scrape_result["markdown"],
                    url=link,
                    structured=scrape_result.get("structured"),
                )
                
                # Chat with agent to extract info
                user_query = f"""
USER INITIAL QUERY: {query}

Context:
- PAGE TITLE: {result.get('title', '')}
- PAGE DESCRIPTION: {result.get('body', '')}
- RELEVANCE SCORE: {result.get('relevance_score', 0)}/100

Extract information that answers the query."""
                
                response = scrapping_agent.chat(query=user_query, stream=False)
                
                # Store response with source
                final_responses.append({
                    'query': query,
                    'source_url': link,
                    'source_title': result.get('title', ''),
                    'relevance_score': result.get('relevance_score', 0),
                    'extracted_info': response.content,
                    'tool_calls': response.tool_calls,
                    'tool_results': response.tool_results,
                })
                
                # Reset agent for next iteration
                scrapping_agent.reset()
            else:
                print(f"  ✗ Scraping failed: {scrape_result.get('error', 'unknown error')}")
        
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
            
            synthesis = synthesis_agent.chat(synthesis_prompt, stream=False)
            synthesis_agent.reset()
            final_answer = synthesis.content
        else:
            final_answer = "No usable source could be scraped for this question."
        
        print("="*50)