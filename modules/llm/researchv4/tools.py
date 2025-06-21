import requests
from bs4 import BeautifulSoup
from googlesearch import search


def search_google(query: str, num_results: int = 5):
    return list(search(query, num_results=num_results))


def fetch_page_content(url: str, headers: str) -> str:
    resp = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    return "\n".join(p.get_text() for p in soup.find_all("p"))
