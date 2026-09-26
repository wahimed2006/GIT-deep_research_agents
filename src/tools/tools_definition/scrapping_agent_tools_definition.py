"""Tool schemas definition for ScrappingAgent.

This module contains the JSON schemas formatted for Ollama tool calling.
"""

from typing import Any, Dict, List

SCRAPPING_AGENT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "scrape_page",
            "description": "Scrape an outbound web page to fetch full details or follow an actionable URL from the ### Links section.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The absolute URL of the page to scrape."
                    },
                    "force_playwright": {
                        "type": "boolean",
                        "description": "Set to true if the target site requires JavaScript rendering (dynamic web app)."
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_table_rows",
            "description": "Search and extract specific matching rows from large Markdown data tables using an exact entity name or symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "The token, symbol, or entity to look for in table rows (e.g., 'Bitcoin', 'SOL', 'qwen3.8:27b')."
                    },
                    "column": {
                        "type": "string",
                        "description": "Filter by specific column name (e.g., 'Price', 'Market Cap'). Optional."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of rows to return (default: 10)."
                    }
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_within_page",
            "description": "Search for specific keywords or patterns within the current page to isolate text snippets without reading the full document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The word, phrase, or regular expression to look up."
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether the search should be case-sensitive (default: false)."
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_section",
            "description": "Retrieve content located strictly under a specific section heading from the Table of Contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "heading_title": {
                        "type": "string",
                        "description": "The exact heading title listed in the Table of Contents."
                    },
                    "include_subsections": {
                        "type": "boolean",
                        "description": "Whether to include content from nested subsections (default: true)."
                    }
                },
                "required": ["heading_title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_links_by_category",
            "description": "Filter available outbound links by topic keywords or intent before deciding where to navigate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_topic": {
                        "type": "string",
                        "description": "Keyword to look for in the anchor text or URL path (e.g. 'download', 'pricing', 'currencies')."
                    },
                    "external_only": {
                        "type": "boolean",
                        "description": "Filter only external links (default: false)."
                    }
                },
                "required": ["target_topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_code_blocks",
            "description": "Extract all code blocks, installation commands, or shell snippets present on the page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "Filter by programming or shell language (e.g., 'bash', 'python', 'json'). Optional."
                    },
                    "include_explanation": {
                        "type": "boolean",
                        "description": "Include surrounding explanatory text (default: false)."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_summary",
            "description": "Get a concise summary of the current page content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum length of the summary in words (default: 100)."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_entities",
            "description": "Compare two or more entities side-by-side from table data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of entity names to compare (e.g., ['Bitcoin', 'Ethereum'])."
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific metrics to compare (e.g., ['Price', 'Market Cap']). Optional."
                    }
                },
                "required": ["entities"]
            }
        }
    }
]