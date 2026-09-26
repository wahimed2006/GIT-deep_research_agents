"""Main scraper module combining fetch, clean, and format.

This module provides the main scraping interface that orchestrates
fetching, cleaning, and formatting web content for LLM consumption.
"""

from typing import Dict, Any, Optional, List
from .fetcher import fetch_content, FetchError, DEFAULT_HTTPX_TIMEOUT, DEFAULT_PLAYWRIGHT_TIMEOUT
from .cleaner import clean_html, extract_content_structure
from .formatter import format_for_llm, MAX_OUTPUT_LENGTH


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Timeouts (imported from fetcher, with local overrides if needed)
DEFAULT_TIMEOUT_HTTPX: int = DEFAULT_HTTPX_TIMEOUT
DEFAULT_TIMEOUT_PLAYWRIGHT: int = DEFAULT_PLAYWRIGHT_TIMEOUT

# Output settings
DEFAULT_FORMAT_MARKDOWN: bool = True
DEFAULT_MAX_OUTPUT_LENGTH: int = MAX_OUTPUT_LENGTH

# Concurrent scraping
DEFAULT_MAX_CONCURRENT: int = 3

# Test URLs (for __main__ block)
TEST_URLS: List[str] = [
    "https://www.wikipedia.org",
    "https://coinmarketcap.com",
    "https://ollama.com/library/qwen3.8"
]


# =============================================================================
# EXCEPTIONS
# =============================================================================

class ScraperError(Exception):
    """Raised when scraping fails."""
    pass


# =============================================================================
# FUNCTIONS
# =============================================================================

def scrape(
    url: str,
    timeout_httpx: int = DEFAULT_TIMEOUT_HTTPX,
    timeout_playwright: int = DEFAULT_TIMEOUT_PLAYWRIGHT,
    force_playwright: bool = False,
    format_markdown: bool = DEFAULT_FORMAT_MARKDOWN,
    max_output_length: int = DEFAULT_MAX_OUTPUT_LENGTH
) -> Dict[str, Any]:
    """Scrape a URL and return structured content.
    
    This is the main entry point for scraping. It fetches the HTML,
    cleans it, extracts structured content, and formats it for LLM use.
    
    Args:
        url: The URL to scrape.
        timeout_httpx: Timeout for httpx requests in seconds.
        timeout_playwright: Timeout for Playwright in seconds.
        force_playwright: If True, use Playwright directly.
        format_markdown: If True, format output as Markdown.
        max_output_length: Maximum length of formatted output.
    
    Returns:
        Dictionary with scraping results:
        - 'success': bool - Whether scraping succeeded
        - 'url': str - The scraped URL
        - 'method': str - Fetch method used ('httpx' or 'playwright')
        - 'structured': Dict - Extracted structured content
        - 'markdown': Optional[str] - Formatted Markdown (if format_markdown=True)
        - 'error': Optional[str] - Error message if failed
    
    Raises:
        ScraperError: If scraping fails completely.
    
    Example:
        >>> result = scrape("https://example.com")
        >>> if result["success"]:
        ...     print(f"Title: {result['structured']['metadata']['title']}")
        ...     print(f"Method: {result['method']}")
        ...     if result["markdown"]:
        ...         print(result["markdown"][:500])
    """
    try:
        # Step 1: Fetch HTML
        fetch_result = fetch_content(
            url=url,
            timeout_httpx=timeout_httpx,
            timeout_playwright=timeout_playwright,
            force_playwright=force_playwright
        )
        
        if not fetch_result["success"]:
            return {
                "success": False,
                "url": url,
                "method": fetch_result.get("method", "unknown"),
                "structured": None,
                "markdown": None,
                "error": fetch_result.get("error", "Unknown fetch error")
            }
        
        html = fetch_result["html"]
        method = fetch_result["method"]
        
        # Step 2: Clean HTML
        cleaned_html = clean_html(html)
        
        # Step 3: Extract structured content
        structured = extract_content_structure(cleaned_html)
        
        # Step 4: Format as Markdown (optional)
        markdown: Optional[str] = None
        if format_markdown:
            markdown = format_for_llm(
                structured_content=structured,
                url=url,
                fetch_method=method,
                max_length=max_output_length
            )
        
        return {
            "success": True,
            "url": url,
            "method": method,
            "structured": structured,
            "markdown": markdown,
            "error": None
        }
    
    except Exception as e:
        return {
            "success": False,
            "url": url,
            "method": "unknown",
            "structured": None,
            "markdown": None,
            "error": f"Scraping failed: {e}"
        }


def scrape_multiple(
    urls: List[str],
    timeout_httpx: int = DEFAULT_TIMEOUT_HTTPX,
    timeout_playwright: int = DEFAULT_TIMEOUT_PLAYWRIGHT,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT
) -> List[Dict[str, Any]]:
    """Scrape multiple URLs sequentially.
    
    Args:
        urls: List of URLs to scrape.
        timeout_httpx: Timeout for httpx requests in seconds.
        timeout_playwright: Timeout for Playwright in seconds.
        max_concurrent: Maximum concurrent scrapes (for future async).
    
    Returns:
        List of scraping results (one per URL).
    
    Example:
        >>> urls = ["https://example1.com", "https://example2.com"]
        >>> results = scrape_multiple(urls)
        >>> for result in results:
        ...     if result["success"]:
        ...         print(f"✓ {result['url']}")
        ...     else:
        ...         print(f"✗ {result['url']}: {result['error']}")
    """
    results: List[Dict[str, Any]] = []
    
    for url in urls:
        result = scrape(
            url=url,
            timeout_httpx=timeout_httpx,
            timeout_playwright=timeout_playwright
        )
        results.append(result)
    
    return results


# =============================================================================
# MAIN (TESTING)
# =============================================================================

if __name__ == "__main__":
    print("Testing scraper...\n")
    
    for url in TEST_URLS:
        print(f"Scraping: {url}")
        result = scrape(url, force_playwright=True)
        
        if result["success"]:
            print(f"  ✓ Success (method: {result['method']})")
            print(f"  Title: {result['structured']['metadata'].get('title', 'N/A')}")
            print(f"  Paragraphs: {len(result['structured']['paragraphs'])}")
            print(f"  Markdown length: {len(result['markdown']) if result['markdown'] else 0}")
            print(result["structured"]["body_text"])
        else:
            print(f"  ✗ Failed: {result['error']}")
        
        print()