import os

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# -------- Web search tool --------
@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic.
    Returns the title, URL and a short snippet for each result."""
    try:
        result = tavily.search(query=query, max_results=5)
    except Exception as e:
        # Return the error as text so the agent can recover instead of crashing
        return f"Search failed: {e}"

    out = []
    for r in result.get("results", []):
        out.append(
            f"Title: {r['title']}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r['content'][:300]}\n"
        )
    return "\n----\n".join(out) if out else "No results found."


# -------- Scrape tool --------
@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        # Raise an error for 4xx/5xx responses so the agent knows the scrape failed
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape URL: {e}"
