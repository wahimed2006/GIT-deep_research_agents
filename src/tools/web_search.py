from typing import List, Dict, Any
from ddgs import DDGS


def search_web(query: str, max_results: int = 5000) -> List[Dict[str, Any]]:
    """Search the web using DuckDuckGo.
    
    Args:
        query: The search query.
        max_results: Maximum number of results to return.
    
    Returns:
        List of search results with title, href, and body.
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    
    return results

if __name__ == "__main__":
    results = search_web("Quelles sont les nouvelles fonctionnalités du dernier iPhone d'Apple ?")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}")
        print(f"   {result['href']}")
        print(f"   {result['body']}\n")