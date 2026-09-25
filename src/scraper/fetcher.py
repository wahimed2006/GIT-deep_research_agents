"""HTTP and browser-based content fetcher.

This module provides functions to fetch HTML content from URLs,
automatically choosing between httpx (fast) and Playwright (for JS-heavy sites).
"""

import httpx
from playwright.sync_api import sync_playwright
from typing import Optional, Dict, Any
from datetime import datetime


class FetchError(Exception):
    """Raised when fetching content fails."""
    pass


def fetch_with_httpx(
    url: str,
    timeout: int = 10,
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
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"
            }
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
    timeout: int = 30,
    wait_until: str = "domcontentloaded"
) -> Dict[str, Any]:
    """Fetch HTML using Playwright (for JavaScript-heavy sites).
    
    Args:
        url: The URL to fetch.
        timeout: Page load timeout in seconds.
        wait_until: When to consider navigation complete.
                   Options: 'load', 'domcontentloaded', 'networkidle', 'commit'.
    
    Returns:
        Dictionary with 'html', 'status_code', 'method', 'timestamp' keys.
    
    Raises:
        FetchError: If the browser fails to load the page.
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-accelerated-2d-canvas",
                    "--disable-gpu"
                ]
            )
            
            page = browser.new_page()
            
            # Set user agent
            page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"
            })
            
            # Navigate to page
            response = page.goto(
                url,
                timeout=timeout * 1000,
                wait_until=wait_until
            )
            
            # Wait for network to be idle (JS loaded)
            try:
                page.wait_for_load_state("networkidle", timeout=timeout * 1000)
            except Exception:
                # Not critical if networkidle times out
                pass
            
            # Get HTML and status
            html = page.content()
            status_code = response.status if response else 0
            
            browser.close()
            
            return {
                "html": html,
                "status_code": status_code,
                "method": "playwright",
                "timestamp": datetime.now().isoformat(),
                "url": url,
                "success": True
            }
    
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
    
    # Framework signatures
    frameworks = [
        'react', 'react-dom', 'next.js', 'gatsby',
        'vue', 'vue-router', 'nuxt',
        'angular', '@angular',
        'svelte',
        'webpack', 'babel'
    ]
    
    # Check for framework usage
    for framework in frameworks:
        if framework in html_lower:
            return True
    
    # Check for empty body with root div
    if 'id="root"' in html_lower or "id='root'" in html_lower:
        if len(html.strip()) < 2000:
            return True
    
    # Check for loading indicators
    loading_indicators = [
        'loading...', 'loading-spinner', 'loading-container',
        'app-root', 'app-loading'
    ]
    
    for indicator in loading_indicators:
        if indicator in html_lower:
            return True
    
    # Check for inline scripts with fetch/axios
    if 'fetch(' in html_lower or 'axios' in html_lower:
        return True
    
    return False


def fetch_content(
    url: str,
    timeout_httpx: int = 10,
    timeout_playwright: int = 30,
    force_playwright: bool = False
) -> Dict[str, Any]:
    """Fetch HTML content with automatic JS detection.
    
    Tries httpx first, then falls back to Playwright if the site
    appears to be JavaScript-heavy or if httpx returns empty content.
    
    Args:
        url: The URL to fetch.
        timeout_httpx: Timeout for httpx requests in seconds.
        timeout_playwright: Timeout for Playwright in seconds.
        force_playwright: If True, skip httpx and use Playwright directly.
    
    Returns:
        Dictionary with 'html', 'method', 'success', 'url', 'timestamp' keys.
        If failed, 'success' is False and 'error' contains the error message.
    
    Example:
        >>> result = fetch_content("https://example.com")
        >>> if result["success"]:
        ...     print(f"Fetched with {result['method']}")
        ...     html = result["html"]
    """
    # Force Playwright if requested
    if force_playwright:
        try:
            return fetch_with_playwright(url, timeout=timeout_playwright)
        except FetchError as e:
            return {
                "url": url,
                "success": False,
                "error": str(e),
                "method": "playwright",
                "timestamp": datetime.now().isoformat()
            }
    
    # Try httpx first
    try:
        result = fetch_with_httpx(url, timeout=timeout_httpx)
        html = result["html"]
        
        # Check if JS-heavy
        if is_js_heavy(html):
            # Try Playwright for better content
            try:
                pw_result = fetch_with_playwright(url, timeout=timeout_playwright)
                # Use Playwright result if it has more content
                if len(pw_result["html"]) > len(html) * 1.5:
                    return pw_result
            except FetchError:
                # Stick with httpx result
                pass
        
        return result
    
    except FetchError as e:
        # Fallback to Playwright
        try:
            return fetch_with_playwright(url, timeout=timeout_playwright)
        except FetchError as pw_error:
            return {
                "url": url,
                "success": False,
                "error": f"httpx failed: {e}, Playwright failed: {pw_error}",
                "method": "fallback",
                "timestamp": datetime.now().isoformat()
            }