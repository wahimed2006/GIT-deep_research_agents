"""Main scraper module combining fetch, clean, and format.

This module provides the main scraping interface that orchestrates
fetching, cleaning, and formatting web content for LLM consumption.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, List
from .fetcher import (
    fetch_content,
    USER_AGENT,
    FetchError,
    PlaywrightSession,
    DEFAULT_HTTPX_TIMEOUT,
    DEFAULT_PLAYWRIGHT_TIMEOUT,
)
from .cleaner import clean_html, extract_content_structure
from .formatter import format_for_llm, format_for_navigation, MAX_OUTPUT_LENGTH
import asyncio
import httpx


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
    "https://ollama.com/library/qwen3.8",
    "https://www.dicofr.com/surveiller-l-utilisation-cpu-et-ram-en-ligne-de-commande-methodes-simples-pour-analyser-ses-ressources-systeme/"
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
async def scrape(
    url: str,
    client: httpx.AsyncClient,
    timeout_httpx: int = DEFAULT_TIMEOUT_HTTPX,
    timeout_playwright: int = DEFAULT_TIMEOUT_PLAYWRIGHT,
    force_playwright: bool = False,
    format_markdown: bool = DEFAULT_FORMAT_MARKDOWN,
    max_output_length: int = DEFAULT_MAX_OUTPUT_LENGTH,
    semaphore: Optional[asyncio.Semaphore] = None,
    playwright_semaphore: Optional[asyncio.Semaphore] = None,
) -> Dict[str, Any]:
    """Scrape asynchronously a URL and return structured content.

    This is the asynchronous main entry point for scraping. It fetches the
    HTML using a shared ``httpx.AsyncClient``, optionally falls back to
    Playwright for JavaScript-heavy pages, cleans the HTML, extracts
    structured content, and optionally formats it as Markdown.

    The HTTP client is provided by the caller so that several URLs can
    reuse the same connection pool. This avoids creating a new HTTP client
    for every URL and improves performance when scraping multiple sources.

    Args:
        url: The URL to scrape.
        client: Shared ``httpx.AsyncClient`` used for asynchronous HTTP
            requests.
        timeout_httpx: Timeout for HTTPX requests in seconds.
        timeout_playwright: Timeout for Playwright requests in seconds.
        force_playwright: If True, skip HTTPX and use Playwright directly.
        format_markdown: If True, format extracted content as Markdown.
        max_output_length: Maximum length of the generated Markdown output.
        semaphore: Optional semaphore limiting the number of concurrent
            scraping operations.
        playwright_semaphore: Optional semaphore limiting the number of
            concurrent Playwright operations.

    Returns:
        A dictionary containing the scraping result:

        - ``success``: bool indicating whether scraping succeeded.
        - ``url``: str containing the scraped URL.
        - ``method``: str indicating the fetch method used
          (``"httpx"`` or ``"playwright"``).
        - ``structured``: Dict containing extracted structured content.
        - ``navigation``: Navigation-oriented representation of the page.
        - ``markdown``: Optional Markdown representation of the page.
        - ``error``: Optional error message if scraping failed.

    Example:
        >>> async with httpx.AsyncClient() as client:
        ...     result = await scrape(
        ...         "https://example.com",
        ...         client=client
        ...     )
        ...
        >>> if result["success"]:
        ...     print(
        ...         result["structured"]["metadata"].get(
        ...             "title",
        ...             "N/A"
        ...         )
        ...     )
    """

    if not url or not url.startswith(("http://", "https://")):
        return {
            "success": False,
            "url": url,
            "method": "none",
            "structured": None,
            "navigation": None,
            "markdown": None,
            "error": f"URL invalide ou vide : '{url}'",
        }

    async def _scrape() -> Dict[str, Any]:
        try:
            # ---------------------------------------------------------
            # Step 1: Fetch HTML
            # ---------------------------------------------------------

            fetch_result = await fetch_content(
                client=client,
                url=url,
                timeout_httpx=timeout_httpx,
                timeout_playwright=timeout_playwright,
                force_playwright=force_playwright,
                playwright_semaphore=playwright_semaphore,
            )

            if not fetch_result["success"]:
                return {
                    "success": False,
                    "url": url,
                    "method": fetch_result.get(
                        "method",
                        "unknown"
                    ),
                    "structured": None,
                    "navigation": None,
                    "markdown": None,
                    "error": fetch_result.get(
                        "error",
                        "Unknown fetch error"
                    ),
                }

            html = fetch_result["html"]
            method = fetch_result["method"]

            # ---------------------------------------------------------
            # Step 2: Clean HTML
            # ---------------------------------------------------------

            cleaned_html = clean_html(html)

            # ---------------------------------------------------------
            # Step 3: Extract structured content
            # ---------------------------------------------------------

            structured = extract_content_structure(
                cleaned_html,
                base_url=url,
            )

            navigation = format_for_navigation(
                structured_content=structured,
                url=url,
                fetch_method=method,
            )

            # ---------------------------------------------------------
            # Step 4: Format as Markdown
            # ---------------------------------------------------------

            markdown: Optional[str] = None

            if format_markdown:
                markdown = format_for_llm(
                    structured_content=structured,
                    url=url,
                    fetch_method=method,
                    max_length=max_output_length,
                )

            return {
                "success": True,
                "url": url,
                "method": method,
                "structured": structured,
                "navigation": navigation,
                "markdown": markdown,
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "url": url,
                "method": "unknown",
                "structured": None,
                "navigation": None,
                "markdown": None,
                "error": f"Scraping failed: {e}",
            }

    # Concurrency limit for the complete scraping operation.
    if semaphore is not None:
        async with semaphore:
            return await _scrape()

    return await _scrape()


async def scrape_multiple(
    urls: List[str],
    timeout_httpx: int = DEFAULT_TIMEOUT_HTTPX,
    timeout_playwright: int = DEFAULT_TIMEOUT_PLAYWRIGHT,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    max_playwright_concurrent: int = 2,
) -> List[Dict[str, Any]]:
    """Scrape multiple URLs concurrently using HTTPX and bounded concurrency.

    This function scrapes several URLs in parallel using a shared
    ``httpx.AsyncClient``. A semaphore limits the maximum number of
    concurrent scraping operations, preventing the application from
    creating an excessive number of simultaneous requests.

    HTTP connections are reused through the shared ``AsyncClient`` and
    its connection pool. Playwright fallbacks are separately limited
    because browser-based scraping is significantly more expensive than
    regular HTTP requests.

    Duplicate and invalid URLs are removed before scraping.

    Args:
        urls: List of URLs to scrape.
        timeout_httpx: Timeout for HTTPX requests in seconds.
        timeout_playwright: Timeout for Playwright requests in seconds.
        max_concurrent: Maximum number of scraping operations that can
            execute concurrently.
        max_playwright_concurrent: Maximum number of Playwright operations
            that can execute concurrently.

    Returns:
        A list of scraping result dictionaries. The order of the returned
        results corresponds to the order of the unique valid URLs supplied
        in ``urls``.

        Each result contains:

        - ``success``: bool indicating whether scraping succeeded.
        - ``url``: str containing the URL.
        - ``method``: str indicating the scraping method used.
        - ``structured``: Dict containing extracted content when successful.
        - ``navigation``: Navigation-oriented representation.
        - ``markdown``: Optional Markdown content.
        - ``error``: Optional error message when scraping failed.

    Example:
        >>> urls = [
        ...     "https://example.com",
        ...     "https://wikipedia.org",
        ...     "https://ollama.com",
        ... ]
        >>>
        >>> results = await scrape_multiple(
        ...     urls,
        ...     max_concurrent=6,
        ...     max_playwright_concurrent=2,
        ... )
        >>>
        >>> for result in results:
        ...     if result["success"]:
        ...         print(f"✓ {result['url']}")
        ...     else:
        ...         print(f"✗ {result['url']}: {result['error']}")

    Notes:
        ``scrape_multiple`` is asynchronous and must therefore be called
        with ``await`` from an async context.
    """

    # -------------------------------------------------------------
    # Step 1: Remove invalid URLs and duplicates
    # -------------------------------------------------------------

    unique_urls = list(dict.fromkeys(
        url
        for url in urls
        if isinstance(url, str)
        and url.startswith(("http://", "https://"))
    ))

    if not unique_urls:
        return []

    # -------------------------------------------------------------
    # Step 2: HTTPX connection pool
    # -------------------------------------------------------------

    limits = httpx.Limits(
        max_connections=max_concurrent,
        max_keepalive_connections=max_concurrent,
    )

    timeout = httpx.Timeout(
        connect=5.0,
        read=timeout_httpx,
        write=timeout_httpx,
        pool=5.0,
    )

    # -------------------------------------------------------------
    # Step 3: Concurrency limits
    # -------------------------------------------------------------

    semaphore = asyncio.Semaphore(max_concurrent)
    playwright_semaphore = asyncio.Semaphore(
        max_playwright_concurrent
    )

    # -------------------------------------------------------------
    # Step 4: Shared HTTP client
    # -------------------------------------------------------------

    async with httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:

        tasks = [
            scrape(
                url=url,
                client=client,
                timeout_httpx=timeout_httpx,
                timeout_playwright=timeout_playwright,
                semaphore=semaphore,
                playwright_semaphore=playwright_semaphore,
            )
            for url in unique_urls
        ]

        # asyncio.gather preserves the order of `tasks`.
        results = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

    # -------------------------------------------------------------
    # Step 5: Normalize unexpected exceptions
    # -------------------------------------------------------------

    normalized_results: List[Dict[str, Any]] = []

    for url, result in zip(unique_urls, results):

        if isinstance(result, BaseException):
            normalized_results.append({
                "success": False,
                "url": url,
                "method": "unknown",
                "structured": None,
                "navigation": None,
                "markdown": None,
                "error": str(result),
            })
        else:
            normalized_results.append(result)

    return normalized_results

# =============================================================================
# MAIN (TESTING)
# =============================================================================

if __name__ == "__main__":
    import time

    async def _main() -> None:
        print("Testing scraper...\n")

        print("=" * 70)
        print("1. SCRAPE SIMPLE")
        print("=" * 70)

        async with httpx.AsyncClient(
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:

            simple_durations = []

            for url in TEST_URLS:
                print(f"\nScraping: {url}")

                start = time.perf_counter()

                result = await scrape(
                    url,
                    client=client,
                    force_playwright=True,
                )

                duration = time.perf_counter() - start
                simple_durations.append(duration)

                if result["success"]:
                    print(f"  ✓ Success")
                    print(f"  Method: {result['method']}")
                    print(
                        f"  Title: "
                        f"{result['structured']['metadata'].get('title', 'N/A')}"
                    )
                    print(
                        f"  Paragraphs: "
                        f"{len(result['structured']['paragraphs'])}"
                    )
                    print(
                        f"  Markdown length: "
                        f"{len(result['markdown']) if result['markdown'] else 0}"
                    )
                else:
                    print(f"  ✗ Failed: {result['error']}")

                print(f"  ⏱ Duration: {duration:.2f}s")

        print("\n")
        print("=" * 70)
        print("2. SCRAPE MULTIPLE")
        print("=" * 70)

        start = time.perf_counter()

        results = await scrape_multiple(
            TEST_URLS,
            max_concurrent=6,
            max_playwright_concurrent=2,
        )

        multi_duration = time.perf_counter() - start

        for url, result in zip(TEST_URLS, results):
            print(f"\n{url}")

            if result["success"]:
                print("  ✓ Success")
                print(f"  Method: {result['method']}")
            else:
                print(f"  ✗ Failed: {result['error']}")

        print("\n")
        print("=" * 70)
        print("3. RÉSULTATS")
        print("=" * 70)

        total_simple = sum(simple_durations)

        print(f"\nNombre d'URLs       : {len(TEST_URLS)}")
        print(f"Temps séquentiel    : {total_simple:.2f}s")
        print(f"Temps parallèle     : {multi_duration:.2f}s")

        if multi_duration > 0:
            speedup = total_simple / multi_duration
            print(f"Speedup             : {speedup:.2f}x")

        if total_simple > 0:
            gain = (1 - multi_duration / total_simple) * 100
            print(f"Gain de temps       : {gain:.1f}%")

        print(f"\nDurées individuelles:")

        for url, duration in zip(TEST_URLS, simple_durations):
            print(f"  {duration:7.2f}s  {url}")

        print()


    asyncio.run(_main())