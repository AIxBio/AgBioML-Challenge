"""Research tools for bioagents framework."""

import os
import re
import aiohttp
import requests
from typing import Dict, Any, Tuple, List
from bs4 import BeautifulSoup
from autogen_core.tools import FunctionTool
from openai import OpenAI


# Placeholder for research tools
# In the future, this can include:
# - Web search capabilities
# - Literature search
# - API integrations
# For now, we'll keep it minimal

async def perplexity_search(query: str) -> str:
    """
    Search for information using Perplexity AI's search API.
    
    This tool is particularly useful for:
    - Finding documentation and usage examples for libraries
    - Researching best practices for specific techniques
    - Getting up-to-date information about tools and methods
    - Understanding error messages and debugging approaches
    
    Args:
        query: The search query to send to Perplexity
        
    Returns:
        Search results as a formatted string with sources
    """
    api_key = os.getenv("PERPLEXITY_API_KEY")
    
    if not api_key:
        return "Perplexity API key not found. Please set PERPLEXITY_API_KEY environment variable."
    
    try:
        # Use the official OpenAI client with Perplexity's base URL
        client = OpenAI(
            api_key=api_key, 
            base_url="https://api.perplexity.ai"
        )
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an artificial intelligence assistant and you need to "
                    "engage in a helpful, detailed, polite conversation with a user."
                ),
            },
            {   
                "role": "user",
                "content": query,
            },
        ]
        
        # Chat completion without streaming
        response = client.chat.completions.create(
            model="sonar",
            messages=messages,
        )
        
        # Format the response
        response_dict = response.model_dump()
        content = response_dict["choices"][0]["message"]["content"]
        
        # Strip all ``` from the content to avoid formatting issues
        content = re.sub(r'```', '<code_delimiter>', content)
        
        # Get citations if available
        citations = response_dict.get("citations", [])
        
        # Format the citations
        if citations:
            content += "\n\nSources:\n"
            for i, citation in enumerate(citations, 1):
                content += f"{i}. {citation}\n"
        
        return content
        
    except Exception as e:
        return f"Error performing Perplexity search: {str(e)}"


async def webpage_parser(url: str) -> str:
    """
    Fetch and parse HTML content from a URL to extract readable text.
    
    This tool is useful for:
    - Reading documentation from web pages
    - Extracting content from blog posts or articles
    - Getting detailed information from sources cited by Perplexity
    
    Args:
        url: URL of the webpage to parse
        
    Returns:
        A string containing the extracted content or error message
    """
    max_length = 100_000
    
    try:
        # Extract the actual URL if it's embedded in the citation string
        # URLs in Perplexity citations are often formatted like "title - source (url)"
        url_match = re.search(r'https?://[^\s)]+', url)
        if url_match:
            url = url_match.group(0)
            
        # Fetch the webpage content
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Extract text and clean it up
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate to max_length with a note if needed
        if len(text) > max_length:
            return text[:max_length] + " [text truncated due to length]"
        return text
        
    except requests.exceptions.RequestException as e:
        return f"Error fetching webpage: {str(e)}"
    except Exception as e:
        return f"Error processing webpage: {str(e)}"


def get_research_tools() -> Dict[str, Any]:
    """Get dictionary of research tool instances."""
    return {
        "perplexity": FunctionTool(
            perplexity_search,
            name="perplexity",
            description="""Query Perplexity for AI-powered search results on technical topics.
            
            Perplexity provides up-to-date, accurate information with sources, making it ideal for:
            - Finding documentation and usage examples for libraries and methods
            - Researching best practices for specific ML/data science techniques
            - Getting current information about tools, packages, and frameworks
            - Understanding error messages and debugging approaches
            - Finding implementation examples and code patterns
            
            The search returns comprehensive answers with citations that can be further explored.
            """
        ),
        "webpage_parser": FunctionTool(
            webpage_parser,
            name="webpage_parser",
            description="""Parse and extract readable text content from web pages.
            
            This tool fetches and extracts the main text content from any web page, removing
            HTML formatting, scripts, and styling to provide clean, readable text.
            
            Particularly useful for:
            - Reading full documentation pages cited by Perplexity
            - Extracting detailed information from blog posts or tutorials
            - Getting complete context from web sources
            - Reading research papers or technical articles
            
            Note: The tool respects standard web scraping practices and includes appropriate headers.
            Content is truncated at 100,000 characters if needed.
            """
        )
    } 