"""
Google Search tool using Serper API.

This module provides functions to search the web using Serper API.
"""

import os
import requests
from typing import List, Dict, Any, Optional

# Load environment variables from .env
from src.utils import load_env
load_env()


def google_search(
    query: str,
    num_results: int = 3,
    api_key: Optional[str] = None
) -> List[Dict[str, str]]:
    """Search Google using Serper API.
    
    Args:
        query: Search query string
        num_results: Number of results to return (default: 3)
        api_key: Serper API key (if None, uses SERPER_API_KEY from env)
    
    Returns:
        List of search results, each containing:
        - title: Page title
        - snippet: Short description
        - link (optional): URL to the page
    
    Raises:
        Exception: If API call fails
    """
    if api_key is None:
        api_key = os.getenv('SERPER_API_KEY')
    
    if not api_key:
        raise ValueError("Serper API key not configured")
    
    url = "https://google.serper.dev/search"
    
    payload = {
        "q": query,
        "num": num_results
    }
    
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract organic results
        organic_results = data.get("organic", [])
        
        results = []
        for result in organic_results[:num_results]:
            results.append({
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
                "link": result.get("link", "")
            })
        
        return results
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"Serper API request failed: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing search results: {str(e)}")


def format_search_results(results: List[Dict[str, str]]) -> str:
    """Format search results as a readable string.
    
    Args:
        results: List of search result dicts
    
    Returns:
        Formatted string with all results
    """
    if not results:
        return "No search results found."
    
    formatted = []
    for i, result in enumerate(results, 1):
        formatted.append(f"[{i}] {result['title']}")
        formatted.append(f"    {result['snippet']}")
        if result.get('link'):
            formatted.append(f"    URL: {result['link']}")
        formatted.append("")
    
    return "\n".join(formatted)