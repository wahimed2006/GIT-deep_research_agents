import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.agents"

from ..scraper import *
from .user_input_agent import UserInputAgent
from .analysis_agent import AnalysisWebSearchAgent, ScoreParseError
from ..tools.web_search import search_web


user = UserInputAgent("llama3.1:8b")
analyse = AnalysisWebSearchAgent("llama3.1:8b")

# Test rapide
response = analyse.chat("Score this: 42", stream=True)
print(f"Test response: '{response.content}'")

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
    for question in content:
        print(f"\n=== Sub-request {i}: {question} ===\n")
        
        # Search web
        web_search = search_web(query=question, max_results=25)
        print(f"Found {len(web_search)} results\n")
        
        # Score each result
        for j, result in enumerate(web_search, 1):
            try:
                score = analyse.score_relevance(
                    query=question,
                    title=result.get('title', ''),
                    link=result.get('link', ''),
                    body=result.get('body', '')
                )
                print(f"  [{j}] Score: {score}/100 - {result.get('title', 'N/A')[:60]}")
                analyse.reset()
            
            except ScoreParseError as e:
                print(f"  [{j}] Parse Error: {e}")
            
            except Exception as e:
                print(f"  [{j}] Error: {e}")
        
        i += 1