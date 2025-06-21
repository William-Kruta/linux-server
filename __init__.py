import asyncio
from config.config import get_candles_path, get_response_model

from modules.data.candles import get_candles
from modules.data.options import get_options_chain
from modules.data.news import get_news
from modules.data.database_connection import DatabaseConnection
from modules.data.financial_statements import get_income_statement, get_cash_flow
from modules.llm.ollama_model import OllamaModel

# from modules.llm.deep_research import DeepResearchFramework
from modules.llm.prompts import stock_query, extract_tickers_prompt, refine_by_top


from modules.llm.researchv1.deep_research import DeepResearchFramework
from modules.llm.researchv2.orchestrator import run_research_task
from modules.llm.researchv3.orchestrator import create_research_framework_example

from modules.utils.utils import get_tickers_from_etf

import pandas as pd


if __name__ == "__main__":
    path = "files\\lists\\{}.csv"

    df = get_tickers_from_etf(path.format("watchlist"))
    print(f"DF: {df}")
    # topic = "The impact of generative AI on software development productivity"
    # create_research_framework_example("Best IPOs in 2025?")
    # run_research_task(topic)
    # Run the example
    # asyncio.run(main())
    # framework, results, report = create_research_framework_example()
    # print("Research framework created successfully!")
    # print(f"Report preview:\n{report[:500]}...")
