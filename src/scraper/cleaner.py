"""HTML cleaning and content extraction.

This module provides functions to clean HTML by removing unwanted elements
and extracting semantically meaningful content.
"""

from bs4 import BeautifulSoup, Tag
from typing import Dict, Any, List, Optional, Set, Union


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
            if parent and parent.name in ['p', 'li', 'article', 'section', 'main']:
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
    
    # Extract title
    title_tag = soup.find('title')
    if title_tag:
        metadata["title"] = title_tag.get_text(strip=True)
    
    # Extract meta description
    desc_tag = soup.find('meta', attrs={'name': 'description'})
    if desc_tag:
        metadata["description"] = _get_meta_content(desc_tag)
    
    # Extract author
    author_tag = soup.find('meta', attrs={'name': 'author'})
    if author_tag:
        metadata["author"] = _get_meta_content(author_tag)
    
    # Try to find article:published_time
    pub_tag = soup.find('meta', attrs={'property': 'article:published_time'})
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
    for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        text = h.get_text(strip=True)
        if text and len(text) < 200:
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
    for list_elem in soup.find_all(['ul', 'ol']):
        items = [li.get_text(strip=True) for li in list_elem.find_all('li')]
        if items:
            lists.append(items)
    
    # Extract tables
    tables: List[List[List[str]]] = []
    for table in soup.find_all('table'):
        rows: List[List[str]] = []
        for tr in table.find_all('tr'):
            cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)
    
    # Extract links (in content only)
    links: List[Dict[str, str]] = []
    for a in soup.find_all('a', href=True):
        # Skip nav/header/footer links
        parent = a.find_parent(['nav', 'header', 'footer'])
        if parent:
            continue
        
        text = a.get_text(strip=True)
        href = a.get('href')
        if text and len(text) < 100 and href and isinstance(href, str):
            links.append({
                "text": text,
                "href": href
            })
    
    # Limit to first 20 links
    links = links[:20]
    
    # Extract body text
    body_text = soup.get_text(separator=' ', strip=True)
    body_text = ' '.join(body_text.split())  # Normalize whitespace
    
    return {
        "metadata": metadata,
        "headings": headings,
        "paragraphs": paragraphs[:30],  # Limit to 30
        "lists": lists[:10],  # Limit to 10
        "tables": tables[:5],  # Limit to 5
        "links": links,
        "body_text": body_text[:15000]  # Limit to 15k chars
    }