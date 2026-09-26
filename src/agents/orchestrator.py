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


user = UserInputAgent("llama3.1:8b")
scrapping_agent = ScrappingAgent("llama3.1:8b")
synthesis_agent = SimpleAgent(
    "llama3.1:8b",
    system_prompt=(
        "You are a research synthesis agent. Answer the user's original question "
        "using only the extracted source information provided. Reconcile conflicts, "
        "avoid unsupported claims, and cite the relevant source URLs."
    )
)


def score_single_result(
    model_name: str,
    question: str,
    item: Dict[str, Any]
) -> Dict[str, Any]:
    """Score one search result in an isolated worker agent."""
    link = item.get("href") or item.get("link") or ""
    title = item.get("title", "")
    body = item.get("body", "")

    result: Dict[str, Any] = {
        "query": question,
        "title": title,
        "link": link,
        "body": body,
        "relevance_score": 0,
    }
    if not link:
        result["score_error"] = "Empty URL"
        return result

    try:
        agent = AnalysisWebSearchAgent(model_name)
        result["relevance_score"] = agent.score_relevance(
            query=question,
            title=title,
            link=link,
            body=body,
        )
    except Exception as error:
        result["score_error"] = str(error)

    return result


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
    
    search_tasks: List[tuple[str, Dict[str, Any]]] = []
    for index, question in enumerate(content, 1):
        print(f"\n=== Sub-request {index}: {question} ===\n")
        web_search = search_web(query=question, max_results=5)
        print(f"Found {len(web_search)} results\n")
        search_tasks.extend((question, item) for item in web_search)

    analysed: List[Dict[str, Any]] = []
    print(f"\n[Analyse] Scoring en parallèle de {len(search_tasks)} résultats...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(score_single_result, "llama3.1:8b", question, item)
            for question, item in search_tasks
        ]
        for future in as_completed(futures):
            result = future.result()
            analysed.append(result)
            print(
                f"  ✓ [{result['relevance_score']}/100] "
                f"{result.get('title', 'N/A')[:50]}"
            )
    
    if analysed:
        # Tri global par ordre décroissant de score
        analysed.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Calcul de la moitié (arrondi au supérieur pour toujours garder au moins 1 résultat)
        top_half_count = math.ceil(len(analysed) / 2)
        top_results: List[Dict[str, Any]] = analysed[:top_half_count]
        
        final_responses = []
        
        for result in top_results:
            # Scrape the page
            scrape_result = scrape(result['link'])
            
            if scrape_result['success']:
                # Load Markdown into scrapping agent
                scrapping_agent.load_markdown(
                    markdown=scrape_result["markdown"],
                    url=result['link'],
                    structured=scrape_result.get("structured"),
                )
                
                # Chat with agent to extract info
                user_query = f"""
                USER INITIAL QUERY : {query}\n\nContext: 
                PAGE TITLE : {result.get('title', '')} \n 
                PAGE DESCRIPTION: {result.get('body', '')} \n
                """
                response = scrapping_agent.chat(query=user_query, stream=False)
                
                # Store response with source
                final_responses.append({
                    'query': query,
                    'source_url': result['link'],
                    'source_title': result.get('title', ''),
                    'relevance_score': result.get('relevance_score', 0),
                    'extracted_info': response.content,
                    'tool_calls': response.tool_calls,
                    'tool_results': response.tool_results,
                })
                
                # Reset agent for next iteration
                scrapping_agent.reset()
            else:
                print(f"Scraping failed for {result['link']}: {scrape_result.get('error', 'unknown error')}")
        
        if final_responses:
            source_context = "\n\n".join(
                f"Source: {response['source_url']}\n"
                f"Title: {response['source_title']}\n"
                f"Extracted information:\n{response['extracted_info']}"
                for response in final_responses
            )
            synthesis_prompt = (
                f"Original user question: {query}\n\n"
                f"Extracted source information:\n{source_context}\n\n"
                "Write the final answer in the language of the original question."
            )
            synthesis = synthesis_agent.chat(synthesis_prompt, stream=False)
            synthesis_agent.reset()
            final_answer = synthesis.content
        else:
            final_answer = "No usable source could be scraped for this question."

        # Print final synthesized response
        print("\n" + "="*50)
        print("FINAL RESPONSE")
        print("="*50 + "\n")
        print(final_answer)
        print()
        
        for i, resp in enumerate(final_responses, 1):
            print(f"--- Source {i}: {resp['source_title']} ---")
            print(f"URL: {resp['source_url']}")
            print(f"Relevance: {resp['relevance_score']}/100")
            print(f"\n{resp['extracted_info']}\n")
        
        print("="*50)