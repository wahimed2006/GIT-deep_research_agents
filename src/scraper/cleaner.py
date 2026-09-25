"""HTML cleaning and content extraction.

This module provides functions to clean HTML by removing unwanted elements
and extracting semantically meaningful content.
"""

from bs4 import BeautifulSoup, Tag
from typing import Dict, Any, List, Optional, Set


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Elements to always remove (noise)
IGNORE_TAGS: Set[str] = {
    'script', 'style', 'noscript', 'iframe', 'svg', 'canvas',
    'nav', 'footer', 'header', 'aside',
    'form', 'input', 'button', 'select', 'option',
    'map', 'area', 'object', 'embed', 'param',
    'meta', 'link', 'base'  # Head elements
}

# Elements to always keep (semantic content)
KEEP_TAGS: Set[str] = {
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'p', 'li', 'dd', 'dt', 'figcaption',
    'table', 'tr', 'td', 'th', 'thead', 'tbody', 'tfoot',
    'blockquote', 'pre', 'code', 'q', 'cite',
    'article', 'main', 'section'
}

# Elements to keep conditionally
CONDITIONAL_TAGS: Set[str] = {'div', 'span', 'a', 'img', 'figure', 'ul', 'ol'}

# Minimum text length to keep an element
MIN_TEXT_LENGTH: int = 50

# Content context tags (for links)
CONTENT_CONTEXT_TAGS: Set[str] = {'p', 'li', 'article', 'section', 'main'}

# Navigation tags to skip for links
NAVIGATION_TAGS: Set[str] = {'nav', 'header', 'footer'}

# Heading tags
HEADING_TAGS: List[str] = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']

# List tags
LIST_TAGS: Set[str] = {'ul', 'ol'}

# Table cell tags
TABLE_CELL_TAGS: Set[str] = {'td', 'th'}

# Metadata tag names
META_DESCRIPTION_NAME: str = 'description'
META_AUTHOR_NAME: str = 'author'
META_PUBLISHED_PROPERTY: str = 'article:published_time'

# Output limits
MAX_HEADINGS_TEXT_LENGTH: int = 200
MAX_PARAGRAPHS: int = 60
MAX_LISTS: int = 50
MAX_TABLES: int = 30
MAX_LINKS: int = 60
MAX_BODY_TEXT_LENGTH: int = 25000
MAX_LINK_TEXT_LENGTH: int = 100


# =============================================================================
# FUNCTIONS
# =============================================================================

def should_keep_element(element: Tag) -> bool:
    """Determine if an HTML element should be kept.
    
    Args:
        element: BeautifulSoup Tag to evaluate.
    
    Returns:
        True if the element should be kept, False otherwise.
    """
    # Skip non-Tag elements
    if not isinstance(element, Tag):
        return False
    
    # Always ignore certain tags
    if element.name in IGNORE_TAGS:
        return False
    
    # Always keep certain tags
    if element.name in KEEP_TAGS:
        return True
    
    # Conditional logic for other tags
    if element.name in CONDITIONAL_TAGS:
        # Get text content
        text = element.get_text(strip=True)
        
        # Keep if has significant text
        if len(text) >= MIN_TEXT_LENGTH:
            return True
        
        # Keep <a> if in content context (not navigation)
        if element.name == 'a':
            parent = element.parent
            if parent and parent.name in CONTENT_CONTEXT_TAGS:
                return True
        
        # Keep <img> if has alt text or in figure
        if element.name == 'img':
            if element.get('alt'):
                return True
            if element.parent and element.parent.name == 'figure':
                return True
        
        # Keep <div>/<span> if contains kept children
        if element.name in ['div', 'span']:
            kept_children = [
                child for child in element.children
                if isinstance(child, Tag) and should_keep_element(child)
            ]
            if kept_children:
                return True
    
    return False


def clean_html(html: str) -> str:
    """Clean HTML by removing unwanted elements.
    
    Args:
        html: Raw HTML string.
    
    Returns:
        Cleaned HTML string with unwanted elements removed.
    
    Example:
        >>> html = "<html><body><script>...</script><p>Hello</p></body></html>"
        >>> clean = clean_html(html)
        >>> "script" not in clean
        True
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove head section entirely
    head = soup.find('head')
    if head:
        head.decompose()
    
    # Find all elements and remove unwanted ones
    all_elements = soup.find_all(True)  # All tags
    
    for element in reversed(all_elements):  # Reverse to avoid index issues
        if not should_keep_element(element):
            # Check if element has any kept descendants
            kept_descendants = [
                desc for desc in element.descendants
                if isinstance(desc, Tag) and should_keep_element(desc)
            ]
            
            # If no kept descendants, remove entirely
            if not kept_descendants:
                element.decompose()
            # If has kept descendants, unwrap (keep children, remove tag)
            else:
                element.unwrap()
    
    return str(soup)


def _get_meta_content(tag: Tag) -> Optional[str]:
    """Safely extract content attribute from meta tag.
    
    Args:
        tag: BeautifulSoup Tag (should be a meta tag).
    
    Returns:
        Content string or None.
    """
    content = tag.get('content')
    if content is None:
        return None
    if isinstance(content, str):
        return content.strip()
    # Handle case where content is a list or other type
    return str(content).strip()


def extract_metadata(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    """Extract metadata from HTML.
    
    Args:
        soup: BeautifulSoup object of the HTML.
    
    Returns:
        Dictionary with 'title', 'description', 'author', 'published_date' keys.
    """
    metadata: Dict[str, Optional[str]] = {
        "title": None,
        "description": None,
        "author": None,
        "published_date": None
    }
    
    # Priority 1: <title> tag
    title_tag = soup.find('title')
    if title_tag:
        title_text = title_tag.get_text(strip=True)
        if title_text:
            metadata["title"] = title_text
    
    # Priority 2: og:title meta
    if not metadata["title"]:
        og_title = soup.find('meta', attrs={'property': 'og:title'})
        if og_title:
            metadata["title"] = _get_meta_content(og_title)
    
    # Priority 3: First h1
    if not metadata["title"]:
        h1 = soup.find('h1')
        if h1:
            metadata["title"] = h1.get_text(strip=True)[:100]
    
    # Fallback: og:site_name
    if not metadata["title"]:
        site_meta = soup.find('meta', attrs={'property': 'og:site_name'})
        if site_meta:
            metadata["title"] = _get_meta_content(site_meta)
    
    # Extract meta description
    desc_tag = soup.find('meta', attrs={'name': META_DESCRIPTION_NAME})
    if desc_tag:
        metadata["description"] = _get_meta_content(desc_tag)
    
    # Extract author
    author_tag = soup.find('meta', attrs={'name': META_AUTHOR_NAME})
    if author_tag:
        metadata["author"] = _get_meta_content(author_tag)
    
    # Try to find article:published_time
    pub_tag = soup.find('meta', attrs={'property': META_PUBLISHED_PROPERTY})
    if pub_tag:
        metadata["published_date"] = _get_meta_content(pub_tag)
    
    return metadata
def extract_content_structure(html: str) -> Dict[str, Any]:
    """Extract structured content from HTML.
    
    Args:
        html: Raw HTML string.
    
    Returns:
        Dictionary with 'title', 'headings', 'paragraphs', 'lists', 
        'tables', 'links', 'body_text' keys.
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Extract metadata
    metadata = extract_metadata(soup)
    
    # Extract headings
    headings: List[Dict[str, str]] = []
    for h in soup.find_all(HEADING_TAGS):
        text = h.get_text(strip=True)
        if text and len(text) < MAX_HEADINGS_TEXT_LENGTH:
            level = h.name if isinstance(h.name, str) else "h1"
            headings.append({
                "level": level,
                "text": text
            })
    
    # Extract paragraphs
    paragraphs: List[str] = []
    for p in soup.find_all('p'):
        text = p.get_text(strip=True)
        if text and len(text) >= MIN_TEXT_LENGTH:
            paragraphs.append(text)
    
    # Extract lists
    lists: List[List[str]] = []
    for list_elem in soup.find_all(LIST_TAGS):
        items = [li.get_text(strip=True) for li in list_elem.find_all('li')]
        if items:
            lists.append(items)
    
    # Extract tables
    tables: List[List[List[str]]] = []
    for table in soup.find_all('table'):
        rows: List[List[str]] = []
        for tr in table.find_all('tr'):
            cells = [td.get_text(strip=True) for td in tr.find_all(TABLE_CELL_TAGS)]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)
    
    # Extract links (in content only)
    links: List[Dict[str, str]] = []
    for a in soup.find_all('a', href=True):
        # Skip nav/header/footer links
        parent = a.find_parent(NAVIGATION_TAGS)
        if parent:
            continue
        
        text = a.get_text(strip=True)
        href = a.get('href')
        if text and len(text) < MAX_LINK_TEXT_LENGTH and href and isinstance(href, str):
            links.append({
                "text": text,
                "href": href
            })
    
    # Limit outputs
    links = links[:MAX_LINKS]
    
    # Extract body text
    body_text = soup.get_text(separator=' ', strip=True)
    body_text = ' '.join(body_text.split())  # Normalize whitespace
    
    return {
        "metadata": metadata,
        "headings": headings[:MAX_PARAGRAPHS],
        "paragraphs": paragraphs[:MAX_PARAGRAPHS],
        "lists": lists[:MAX_LISTS],
        "tables": tables[:MAX_TABLES],
        "links": links,
        "body_text": body_text[:MAX_BODY_TEXT_LENGTH]
    }