import sys
from pathlib import Path
import math
from typing import Dict, List, Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.agents"

from ..scraper import *
from .user_input_agent import UserInputAgent
from .analysis_agent import AnalysisWebSearchAgent, ScoreParseError
from ..tools.web_search import search_web
from .scrapping_agent import ScrappingAgent


user = UserInputAgent("llama3.1:8b")
analysed_agent = AnalysisWebSearchAgent("llama3.1:8b")
scrapping_agent = ScrappingAgent("llama3.1:8b")


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
    
    i = 1
    analysed = []
    for question in content:
        print(f"\n=== Sub-request {i}: {question} ===\n")
        
        # Search web
        web_search = search_web(query=question, max_results=5)
        print(f"Found {len(web_search)} results\n")
        analysed += analysed_agent.web_search_score(query=question, web_search_result=web_search)
    
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
        
        # Print final aggregated response
        print("\n" + "="*50)
        print("FINAL RESPONSE")
        print("="*50 + "\n")
        
        for i, resp in enumerate(final_responses, 1):
            print(f"--- Source {i}: {resp['source_title']} ---")
            print(f"URL: {resp['source_url']}")
            print(f"Relevance: {resp['relevance_score']}/100")
            print(f"\n{resp['extracted_info']}\n")
        
        print("="*50)