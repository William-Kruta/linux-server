import re
import logging
from modules.llm.enchanced_ollama import EnhancedOllamaModel
import requests
from bs4 import BeautifulSoup
from googlesearch import search


class Agent:
    def __init__(self, model_name: str, system_prompt: str, task_name: str):
        self.model = EnhancedOllamaModel(
            model_name=model_name, system_prompt=system_prompt
        )
        self.task_name = task_name
        logging.info(
            f"Initialized {self.__class__.__name__} with model '{model_name}'."
        )

    def _create_prompt(self, task: str, context: str = None) -> str:
        """
        Creates the final prompt to be sent to the LLM, combining the task
        and any relevant context from research.

        Args:
            task: The main task or question for the agent.
            context: The text content gathered from web searches.

        Returns:
            The formatted prompt string.
        """
        if context:
            return f"""
Based on the following context, please perform the requested task.
--- CONTEXT ---
{context}
--- END CONTEXT ---

TASK: {task}
"""
        else:
            return task

    def run(
        self,
        task: str,
        search_query: str = None,
        context_window: int = 4096,
        max_tokens: int = 4096,
        num_web_results: int = 5,
        web_page_limit: int = 10000,
    ) -> str:
        """
        Executes a given task. If a search query is provided, it will first
        search the web, retrieve content, and then use that context to perform the task.

        Args:
            task: The description of the task for the agent to perform.
            search_query: An optional search query to gather context from the web.

        Returns:
            The agent's response as a string.
        """
        logging.info(f"'{self.__class__.__name__}' is running task: '{task}'")
        context = None
        if search_query:
            logging.info(f"Performing web search for query: '{search_query}'")
            search_results = self.query_google(
                search_query, num_results=num_web_results
            )
            print(f"Search: {search_results}")
            if search_results:
                # Get content from the first result for simplicity
                # url = search_results[0]["href"]
                page_content = [
                    self.fetch_page_content(url)[:web_page_limit]
                    for url in search_results
                ]
                print(f"Page: {page_content}")
                if page_content:
                    # Limit context size to avoid overwhelming the model
                    context = self._clean_multiple_texts(page_content)
            else:
                logging.warning("No search results found.")

        final_prompt = self._create_prompt(task, context)
        response = self.model.get_response(
            final_prompt, context_window=context_window, max_tokens=max_tokens
        )
        logging.info(f"'{self.__class__.__name__}' finished task.")
        return response

    def query_google(self, query: str, num_results: int = 5):
        return list(search(query, num_results=num_results))

    def fetch_page_content(self, url: str, headers: dict = {}) -> str:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        try:
            resp = requests.get(url, headers=headers, timeout=10)
        except requests.exceptions.ReadTimeout:
            return ""
        except requests.exceptions.MissingSchema:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        return "\n".join(p.get_text() for p in soup.find_all("p"))

    def _clean_text(self, text: str):
        text = text.replace("\n", " ").replace("\r", "")
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def _clean_multiple_texts(self, texts: list) -> list:
        return [self._clean_text(text) for text in texts]


class AnalystAgent(Agent):
    """An agent focused on data analysis and evidence-based conclusions."""

    def __init__(self, model_name: str = "llama3"):
        system_prompt = """
You are a meticulous Analyst Agent. Your purpose is to analyze data and text to draw
evidence-based conclusions. You must focus on facts, identify patterns, and remain
objective. Do not speculate; base all your findings on the provided context.
"""
        super().__init__(model_name, system_prompt, "Analyst")


class SynthesizerAgent(Agent):
    """An agent focused on combining information from multiple sources."""

    def __init__(self, model_name: str = "llama3"):
        system_prompt = """
You are a Synthesizer Agent. Your role is to read and combine information
from various sources into a single, coherent, and comprehensive summary.
Identify the main themes, connections, and discrepancies across the provided context.
"""
        super().__init__(model_name, system_prompt, "Synthesizer")


class CriticAgent(Agent):
    """An agent focused on questioning assumptions and identifying biases."""

    def __init__(self, model_name: str = "llama3"):
        system_prompt = """
You are a Critic Agent. Your function is to critically evaluate information,
question assumptions, and identify potential biases or logical fallacies.
Challenge the arguments presented in the context and highlight any weaknesses
or areas that lack sufficient evidence.
"""
        super().__init__(model_name, system_prompt, "Critic")


class ExplorerAgent(Agent):
    """An agent focused on identifying new research directions."""

    def __init__(self, model_name: str = "llama3"):
        system_prompt = """
You are an Explorer Agent. Your goal is to look beyond the provided information
and identify new research directions, unanswered questions, and potential areas for
further investigation. Brainstorm creative ideas and suggest novel paths based on
the gaps in the current knowledge.
"""
        super().__init__(model_name, system_prompt, "Explorer")
