import datetime as dt
from typing import List
from modules.llm.ollama_model import OllamaModel


def stock_query(topic: str) -> str:
    today = dt.datetime.now().date().isoformat()
    return f"{topic} as of {today}"


def extract_tickers_prompt(topic: str, text: str) -> str:
    return (
        f"From the following text about '{topic}', list all stock tickers mentioned "
        f"(comma-separated):\n\n{text}"
    )


def refine_by_top(found: List[str]) -> str:
    if not found:
        return "latest AI market trends"
    return f"Latest news on {found[0]} in AI sector"


def make_extract_prompt_via_llm(model: OllamaModel, topic: str, text: str) -> str:
    # ask the model to produce the instruction
    meta = (
        f"You are a prompt engineer.  Given a user question:\n\n"
        f"    {topic!r}\n\n"
        "Produce a single-sentence instruction that tells another LLM exactly "
        "what to extract from an article on that topic."
    )
    instruction = model.get_response(meta)
    # now plug that into your actual extraction
    return f"{instruction}\n\nArticle excerpt:\n{text}"
