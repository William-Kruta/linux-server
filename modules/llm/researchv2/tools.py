import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import logging

# --- Configuration ---
# Configure logging to display informative messages
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Web Search and Content Retrieval ---


def search_web(query: str, max_results: int = 5):
    """
    Performs a web search using DuckDuckGo and returns the results.

    Args:
        query: The search query string.
        max_results: The maximum number of search results to return.

    Returns:
        A list of dictionaries, where each dictionary represents a search result.
        Returns an empty list if an error occurs.
    """
    logging.info(f"Performing web search for: '{query}'")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        logging.info(f"Found {len(results)} results.")
        return results
    except Exception as e:
        logging.error(f"An error occurred during web search: {e}")
        return []


def get_page_content(url: str):
    """
    Retrieves and parses the text content of a web page.

    Args:
        url: The URL of the web page.

    Returns:
        The cleaned text content of the page, or None if retrieval fails.
    """
    logging.info(f"Fetching content from URL: {url}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Use BeautifulSoup to parse the HTML and extract text
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = "\n".join(chunk for chunk in chunks if chunk)

        logging.info(f"Successfully retrieved and parsed content from {url}.")
        return cleaned_text
    except requests.RequestException as e:
        logging.error(f"Error fetching content from {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"An error occurred while parsing content from {url}: {e}")
        return None


if __name__ == "__main__":
    # This block demonstrates how to use the functions in this file.
    # It will only run when the script is executed directly.

    # 1. Perform a web search
    search_query = "latest advancements in artificial intelligence"
    search_results = search_web(search_query, max_results=3)

    if search_results:
        print("\n--- Search Results ---")
        for i, result in enumerate(search_results):
            print(f"{i+1}. {result['title']}")
            print(f"   {result['href']}")

        # 2. Get content from the first search result
        first_url = search_results[0]["href"]
        print(f"\n--- Fetching content from: {first_url} ---")
        content = get_page_content(first_url)

        if content:
            print("\n--- Page Content (first 500 characters) ---")
            print(content[:500] + "...")
    else:
        print("Could not retrieve search results.")
