from modules.llm.researchv4.tools import search_google, fetch_page_content
from modules.llm.enchanced_ollama import EnhancedOllamaModel


class DeepResearch:
    def __init__(self, model: str):
        self.model = EnhancedOllamaModel(model)
