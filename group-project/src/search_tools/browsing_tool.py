"""
Browsing tool for fetching full web page content.

This module provides functionality to browse and extract full content from web pages,
complementing the search tool which only returns snippets.
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional
import time


def browse_webpage(url: str, timeout: int = 10) -> Dict[str, Any]:
    """Fetch and extract content from a webpage.
    
    Args:
        url: URL of the webpage to browse
        timeout: Request timeout in seconds
    
    Returns:
        Dict containing:
        - url: The URL browsed
        - title: Page title
        - content: Extracted text content (first 3000 chars)
        - success: Whether the request succeeded
        - error: Error message if failed
    """
    try:
        # Add headers to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch the page
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else "No title"
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Limit content length to avoid overwhelming the LLM
        # Increased limit to get more context for better answers
        max_chars = 5000
        if len(text) > max_chars:
            text = text[:max_chars] + "...\n[Content truncated]"
        
        return {
            "url": url,
            "title": title,
            "content": text,
            "success": True,
            "error": None
        }
    
    except requests.Timeout:
        return {
            "url": url,
            "title": None,
            "content": None,
            "success": False,
            "error": "Request timeout"
        }
    except requests.RequestException as e:
        return {
            "url": url,
            "title": None,
            "content": None,
            "success": False,
            "error": f"Request failed: {str(e)}"
        }
    except Exception as e:
        return {
            "url": url,
            "title": None,
            "content": None,
            "success": False,
            "error": f"Parsing failed: {str(e)}"
        }


def format_browse_result(result: Dict[str, Any]) -> str:
    """Format browse result for display.
    
    Args:
        result: Browse result dict
    
    Returns:
        Formatted string
    """
    if not result["success"]:
        return f"Failed to browse {result['url']}: {result['error']}"
    
    formatted = f"Title: {result['title']}\n"
    formatted += f"URL: {result['url']}\n"
    formatted += f"Content:\n{result['content']}\n"
    
    return formatted


if __name__ == "__main__":
    # Test browsing
    test_url = "https://en.wikipedia.org/wiki/Apollo_17"
    print(f"Testing browsing: {test_url}\n")
    
    result = browse_webpage(test_url)
    
    if result["success"]:
        print(f"✅ Successfully browsed page")
        print(f"Title: {result['title']}")
        print(f"Content length: {len(result['content'])} chars")
        print(f"\nFirst 500 chars of content:")
        print(result['content'][:500])
    else:
        print(f"❌ Failed: {result['error']}")
