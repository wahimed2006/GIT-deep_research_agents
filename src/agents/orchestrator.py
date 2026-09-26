import sys
from pathlib import Path
import math

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.agents"

from ..scraper import *
from .user_input_agent import UserInputAgent
from .analysis_agent import AnalysisWebSearchAgent, ScoreParseError
from ..tools.web_search import search_web


user = UserInputAgent("llama3.1:8b")
analysed_agent = AnalysisWebSearchAgent("llama3.1:8b")

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
        top_results = analysed[:top_half_count]