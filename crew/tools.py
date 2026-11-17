"""
crew/tools.py

Custom CrewAI tools using the @tool decorator:
  1. web_scraper_tool   — Calls Tavily/Serper search API and scrapes page text.
  2. calculate_financial_metrics — Loads a CSV into Pandas and returns a
     Markdown summary statistics table.
"""

from __future__ import annotations

import os
import io
import textwrap
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scrape_url(url: str, timeout: int = 10) -> str:
    """Fetch a URL and return cleaned plain-text content (max 3000 chars)."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        # Remove script / style noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Collapse blank lines
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines)[:3000]
    except Exception as exc:  # noqa: BLE001
        return f"[Could not fetch {url}: {exc}]"


def _search_tavily(query: str, num_results: int = 5) -> list[dict]:
    """Call Tavily Search API and return list of result dicts."""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise EnvironmentError("TAVILY_API_KEY is not set in environment.")

    from tavily import TavilyClient  # lazy import — only needed at runtime
    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=num_results,
        include_raw_content=False,
    )
    return response.get("results", [])


def _search_serper(query: str, num_results: int = 5) -> list[dict]:
    """Call SerperDev API and return list of result dicts."""
    api_key = os.getenv("SERPER_API_KEY", "")
    if not api_key:
        raise EnvironmentError("SERPER_API_KEY is not set in environment.")

    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": num_results}
    resp = requests.post(
        "https://google.serper.dev/search", json=payload, headers=headers, timeout=15
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"title": r.get("title", ""), "url": r.get("link", ""), "content": r.get("snippet", "")}
        for r in data.get("organic", [])
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Public Tools
# ─────────────────────────────────────────────────────────────────────────────

@tool("Web Scraper Tool")
def web_scraper_tool(query: str) -> str:
    """
    Search the web for the given query using Tavily (or Serper as fallback),
    then scrape the top result pages and return a concatenated plain-text
    summary of the findings.

    Args:
        query: The market research search query string.

    Returns:
        A string containing scraped text from top search results,
        prefixed with source URLs and titles.
    """
    provider = os.getenv("SEARCH_PROVIDER", "tavily").lower()

    try:
        if provider == "serper":
            results = _search_serper(query, num_results=5)
        else:
            results = _search_tavily(query, num_results=5)
    except Exception as exc:  # noqa: BLE001
        return f"[Search failed: {exc}]"

    if not results:
        return f"[No search results found for query: '{query}']"

    sections: list[str] = []
    for idx, result in enumerate(results[:5], start=1):
        title = result.get("title", "No Title")
        url = result.get("url", "")
        snippet = result.get("content", "")

        # Attempt to scrape full page; fall back to snippet
        page_text = _scrape_url(url) if url else snippet
        if not page_text or page_text.startswith("[Could not fetch"):
            page_text = snippet

        section = textwrap.dedent(f"""
            --- Source {idx}: {title} ---
            URL: {url}
            {page_text[:1500]}
        """).strip()
        sections.append(section)

    return "\n\n".join(sections)


@tool("Financial Metrics Calculator")
def calculate_financial_metrics(csv_path: str) -> str:
    """
    Load corporate financial metrics from a CSV file into a Pandas DataFrame,
    compute descriptive summary statistics (count, mean, std, min, 25%, 50%,
    75%, max), and return the results as a Markdown-formatted table.

    Args:
        csv_path: Absolute or relative path to a CSV file containing numeric
                  financial data columns (e.g. Revenue, Profit, MarketCap).

    Returns:
        A Markdown string containing the summary statistics table.
        Returns an error string if the file cannot be read.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return f"[Error] CSV file not found: '{csv_path}'"
    except Exception as exc:  # noqa: BLE001
        return f"[Error] Could not read CSV: {exc}"

    # Keep only numeric columns for meaningful statistics
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return "[Error] No numeric columns found in the provided CSV."

    summary = numeric_df.describe().round(2)
    try:
        md_table = summary.to_markdown()
    except Exception:  # tabulate not available — fall back to string
        md_table = summary.to_string()

    return f"## Financial Summary Statistics\n\n{md_table}\n"
