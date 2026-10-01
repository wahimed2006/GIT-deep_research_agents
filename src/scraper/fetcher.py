"""HTTP and browser-based content fetcher.

This module provides functions to fetch HTML content from URLs,
automatically choosing between httpx (fast) and Playwright (for JS-heavy sites).
"""

import httpx
from playwright.sync_api import Browser, Page, sync_playwright
from typing import Literal, Optional, Dict, Any, List, Set
from datetime import datetime
import asyncio

# =============================================================================
# TYPE ALIASES
# =============================================================================

WaitUntilState = Literal['commit', 'domcontentloaded', 'load', 'networkidle']


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Timeouts
DEFAULT_HTTPX_TIMEOUT: int = 10
DEFAULT_PLAYWRIGHT_TIMEOUT: int = 30
DEFAULT_CONTENT_MULTIPLIER: float = 1.5

# Browser arguments (Playwright)
BROWSER_ARGS: List[str] = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--disable-accelerated-2d-canvas",
    "--disable-gpu"
]

# User agent
USER_AGENT: str = "Mozilla/5.0 (compatible; ResearchBot/1.0)"

# JavaScript detection
JS_FRAMEWORKS: Set[str] = {
    'react', 'react-dom', 'next.js', 'gatsby',
    'vue', 'vue-router', 'nuxt',
    'angular', '@angular',
    'svelte',
    'webpack', 'babel'
}

LOADING_INDICATORS: Set[str] = {
    'loading...', 'loading-spinner', 'loading-container',
    'app-root', 'app-loading'
}

JS_SIGNATURES: Set[str] = {'fetch(', 'axios'}

# Root div detection
ROOT_DIV_PATTERNS: Set[str] = {'id="root"', "id='root'"}
MAX_EMPTY_ROOT_LENGTH: int = 2000


DEFAULT_MAX_HTTP_CONNECTIONS = 8
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 8

ASYNC_LIMITS = httpx.Limits(
    max_connections=DEFAULT_MAX_HTTP_CONNECTIONS,
    max_keepalive_connections=DEFAULT_MAX_KEEPALIVE_CONNECTIONS,
)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36"
)

# =============================================================================
# EXCEPTIONS
# =============================================================================

class FetchError(Exception):
    """Raised when fetching content fails."""
    pass


class PlaywrightSession:
    """Reusable Playwright browser session for sequential fetches."""

    def __init__(self) -> None:
        self._playwright = None
        self.browser: Optional[Browser] = None

    def __enter__(self) -> "PlaywrightSession":
        self._playwright = sync_playwright().start()
        try:
            self.browser = self._playwright.chromium.launch(
                headless=True,
                args=BROWSER_ARGS
            )
        except Exception:
            self._playwright.stop()
            self._playwright = None
            raise
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        try:
            if self.browser is not None:
                self.browser.close()
        finally:
            self.browser = None
            if self._playwright is not None:
                self._playwright.stop()
                self._playwright = None

    def fetch(
        self,
        url: str,
        timeout: int = DEFAULT_PLAYWRIGHT_TIMEOUT,
        wait_until: WaitUntilState | None = "domcontentloaded",
        wait_for_selector: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch one page using the shared browser."""
        if self.browser is None:
            raise FetchError("PlaywrightSession must be used as a context manager")

        page: Page = self.browser.new_page()
        try:
            page.set_extra_http_headers({"User-Agent": USER_AGENT})
            response = page.goto(
                url,
                timeout=timeout * 1000,
                wait_until=wait_until
            )

            if wait_for_selector:
                page.wait_for_selector(
                    wait_for_selector,
                    timeout=timeout * 1000
                )

            return {
                "html": page.content(),
                "status_code": response.status if response else 0,
                "method": "playwright",
                "timestamp": datetime.now().isoformat(),
                "url": url,
                "success": True
            }
        finally:
            page.close()


# =============================================================================
# FUNCTIONS
# =============================================================================

def fetch_with_httpx(
    url: str,
    timeout: int = DEFAULT_HTTPX_TIMEOUT,
    follow_redirects: bool = True
) -> Dict[str, Any]:
    """Fetch HTML using httpx (fast, for static sites).
    
    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.
        follow_redirects: Whether to follow HTTP redirects.
    
    Returns:
        Dictionary with 'html', 'status_code', 'method', 'timestamp' keys.
    
    Raises:
        FetchError: If the request fails.
    """
    try:
        response = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=follow_redirects,
            headers={"User-Agent": USER_AGENT}
        )
        response.raise_for_status()
        
        return {
            "html": response.text,
            "status_code": response.status_code,
            "method": "httpx",
            "timestamp": datetime.now().isoformat(),
            "url": url,
            "success": True
        }
    
    except httpx.HTTPStatusError as e:
        raise FetchError(f"HTTP error {e.response.status_code}: {e}")
    except httpx.RequestError as e:
        raise FetchError(f"Request failed: {e}")
    except Exception as e:
        raise FetchError(f"Unexpected error: {e}")


def fetch_with_playwright(
    url: str,
    timeout: int = DEFAULT_PLAYWRIGHT_TIMEOUT,
    wait_until: WaitUntilState | None = "domcontentloaded",
    wait_for_selector: Optional[str] = None,
    session: Optional[PlaywrightSession] = None
) -> Dict[str, Any]:
    """Fetch HTML using Playwright (for JavaScript-heavy sites).
    
    Args:
        url: The URL to fetch.
        timeout: Page load timeout in seconds.
        wait_until: When to consider navigation complete.
                   Options: 'load', 'domcontentloaded', 'networkidle', 'commit'.
        wait_for_selector: Optional selector to wait for after navigation.
        session: Optional reusable Playwright session.
    
    Returns:
        Dictionary with 'html', 'status_code', 'method', 'timestamp' keys.
    
    Raises:
        FetchError: If the browser fails to load the page.
    """
    try:
        if session is not None:
            return session.fetch(
                url,
                timeout=timeout,
                wait_until=wait_until,
                wait_for_selector=wait_for_selector
            )

        with PlaywrightSession() as local_session:
            return local_session.fetch(
                url,
                timeout=timeout,
                wait_until=wait_until,
                wait_for_selector=wait_for_selector
            )
    
    except Exception as e:
        raise FetchError(f"Playwright failed: {e}")


def is_js_heavy(html: str) -> bool:
    """Detect if a page is likely JavaScript-heavy.
    
    Args:
        html: Raw HTML content.
    
    Returns:
        True if the page appears to use significant JavaScript.
    """
    html_lower = html.lower()
    
    # Check for framework usage
    for framework in JS_FRAMEWORKS:
        if framework in html_lower:
            return True
    
    # Check for empty body with root div
    for pattern in ROOT_DIV_PATTERNS:
        if pattern in html_lower:
            if len(html.strip()) < MAX_EMPTY_ROOT_LENGTH:
                return True
    
    # Check for loading indicators
    for indicator in LOADING_INDICATORS:
        if indicator in html_lower:
            return True
    
    # Check for inline scripts with fetch/axios
    for signature in JS_SIGNATURES:
        if signature in html_lower:
            return True
    
    return False

async def fetch_with_httpx_async(
    client: httpx.AsyncClient,
    url: str,
    timeout: int = DEFAULT_HTTPX_TIMEOUT,
) -> Dict[str, Any]:
    """
    Fetch HTML asynchronously using a shared httpx.AsyncClient.
    """

    try:
        response = await client.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )

        response.raise_for_status()

        return {
            "html": response.text,
            "status_code": response.status_code,
            "method": "httpx",
            "timestamp": datetime.now().isoformat(),
            "url": str(response.url),
            "success": True,
        }

    except httpx.HTTPStatusError as e:
        raise FetchError(
            f"HTTP error {e.response.status_code}: {e}"
        )

    except httpx.RequestError as e:
        raise FetchError(f"Request failed: {e}")

    except Exception as e:
        raise FetchError(f"Unexpected error: {e}")


async def fetch_content(
    client: httpx.AsyncClient,
    url: str,
    timeout_httpx: int = DEFAULT_HTTPX_TIMEOUT,
    timeout_playwright: int = DEFAULT_PLAYWRIGHT_TIMEOUT,
    force_playwright: bool = False,
    playwright_semaphore: Optional[asyncio.Semaphore] = None,
) -> Dict[str, Any]:
    """Fetch HTML content with automatic JS detection.
        
        Tries httpx first, then falls back to Playwright if the site
        appears to be JavaScript-heavy or if httpx returns empty content.
        
        Args:
            url: The URL to fetch.
            timeout_httpx: Timeout for httpx requests in seconds.
            timeout_playwright: Timeout for Playwright in seconds.
            force_playwright: If True, skip httpx and use Playwright directly.
            session: Optional reusable Playwright session for Playwright fallbacks.
        
        Returns:
            Dictionary with 'html', 'method', 'success', 'url', 'timestamp' keys.
            If failed, 'success' is False and 'error' contains the error message.
        
        Example:
            >>> result = fetch_content("https://example.com")
            >>> if result["success"]:
            ...     print(f"Fetched with {result['method']}")
            ...     html = result["html"]
        """

    # ---------------------------------------------------------
    # Force Playwright
    # ---------------------------------------------------------

    if force_playwright:
        try:
            if playwright_semaphore is not None:
                async with playwright_semaphore:
                    return await asyncio.to_thread(
                        fetch_with_playwright,
                        url,
                        timeout_playwright,
                    )

            return await asyncio.to_thread(
                fetch_with_playwright,
                url,
                timeout_playwright,
            )

        except FetchError as e:
            return {
                "url": url,
                "success": False,
                "error": str(e),
                "method": "playwright",
                "timestamp": datetime.now().isoformat(),
            }

    # ---------------------------------------------------------
    # HTTPX
    # ---------------------------------------------------------

    try:
        result = await fetch_with_httpx_async(
            client=client,
            url=url,
            timeout=timeout_httpx,
        )

        html = result["html"]

        # -----------------------------------------------------
        # JS-heavy detection
        # -----------------------------------------------------

        if is_js_heavy(html):

            try:
                if playwright_semaphore is not None:
                    async with playwright_semaphore:
                        pw_result = await asyncio.to_thread(
                            fetch_with_playwright,
                            url,
                            timeout_playwright,
                        )
                else:
                    pw_result = await asyncio.to_thread(
                        fetch_with_playwright,
                        url,
                        timeout_playwright,
                    )

                if (
                    pw_result["success"]
                    and len(pw_result["html"])
                    > len(html) * DEFAULT_CONTENT_MULTIPLIER
                ):
                    return pw_result

            except FetchError:
                pass

        return result

    except FetchError as httpx_error:

        # -----------------------------------------------------
        # HTTPX failed → Playwright fallback
        # -----------------------------------------------------

        try:

            if playwright_semaphore is not None:
                async with playwright_semaphore:
                    pw_result = await asyncio.to_thread(
                        fetch_with_playwright,
                        url,
                        timeout_playwright,
                    )
            else:
                pw_result = await asyncio.to_thread(
                    fetch_with_playwright,
                    url,
                    timeout_playwright,
                )

            return pw_result

        except FetchError as pw_error:

            return {
                "url": url,
                "success": False,
                "error": (
                    f"httpx failed: {httpx_error}, "
                    f"Playwright failed: {pw_error}"
                ),
                "method": "fallback",
                "timestamp": datetime.now().isoformat(),
            }