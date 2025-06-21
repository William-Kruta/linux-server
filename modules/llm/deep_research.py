import datetime as dt
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Callable, Any
from googlesearch import search
from modules.llm.ollama_model import OllamaModel


class DeepResearchFramework:
    def __init__(
        self,
        model: OllamaModel,
        website_steps: int = 5,
        user_agent: str = "Mozilla/5.0 (compatible)",
    ):
        self.model = model
        self.website_steps = website_steps
        self.headers = {"User-Agent": user_agent}

    def _search_web(self, query: str) -> List[str]:
        return list(search(query, num_results=self.website_steps))

    def _fetch_text(self, url: str) -> str:
        resp = requests.get(url, headers=self.headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        return "\n".join(p.get_text() for p in soup.find_all("p"))

    def research(
        self,
        topic: str,
        build_query: Callable[[str], str],
        extract_prompt: Callable[[str, str], str],
        refine_prompt: Callable[[str, List[Any]], str] = None,
    ) -> List[Any]:
        """
        Generic multi-step research on `topic`.

        Parameters
        ----------
        topic : str
            User’s top-level question/area.
        build_query : fn(topic) → str
            How to turn topic into a Google search query.
        extract_prompt : fn(topic, text) → str
            How to turn site text into an “extract these items” prompt.
        refine_prompt : fn(topic, found_items) → str, optional
            How to refine your next query given what you already found.

        Returns
        -------
        List of whatever your model extracted (tickers, facts, names…)
        """
        found = []
        query = build_query(topic)

        for step in range(self.website_steps):
            urls = self._search_web(query)
            print(f"Urls: {urls}")
            exit()
            if not urls:
                break

            doc = self._fetch_text(urls[0])[:2000]  # truncate for context
            prompt = extract_prompt(topic, doc)
            reply = self.model.get_response(prompt)

            # simple CSV/line split
            items = [i.strip() for i in reply.split(",") if i.strip()]
            found.extend(items)

            if refine_prompt:
                query = refine_prompt(topic, found)
            else:
                # default: re-search the same topic
                query = build_query(topic)

        # unique, preserve order
        seen = set()
        return [x for x in found if not (x in seen or seen.add(x))]

    def deep_dive(
        self,
        items: List[Any],
        build_query: Callable[[Any], str],
        extract_prompt: Callable[[Any, str], str],
    ) -> Dict[Any, List[Any]]:
        """
        Apply the same research loop to each extracted item.
        """
        results = {}
        for item in items:
            q = build_query(item)
            # reuse research() but without further refinement
            candidates = self.research(
                topic=item,
                build_query=lambda t: q,
                extract_prompt=extract_prompt,
                refine_prompt=None,
            )
            results[item] = candidates
        return results
