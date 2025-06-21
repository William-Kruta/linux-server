import time
import argparse

from log import Log

# Config
from config.config import get_candles_path

# Modules
from modules.data.candles import get_candles
from modules.data.options import get_options_chain
from modules.data.financial_statements import (
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
)
from modules.utils.utils import get_tickers_from_etf

ETF_PATH = "files\\lists\\{}.csv"

app_log = Log("log.txt")


def _download_statement(ticker: str, period: str, statement: str):
    quarter_params = ["q", "quarter", "qtr"]
    annual_params = ["a", "annual", "ann"]
    path = get_candles_path()
    if period.lower() in quarter_params:
        quarter = True
    else:
        quarter = False

    if statement == "income_statement":
        data = get_income_statement(ticker, quarter, path)
    elif statement == "balance_sheet":
        data = get_balance_sheet(ticker, quarter, path)
    elif statement == "cash_flow":
        data = get_cash_flow(ticker, quarter, path)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI for data retrieval and research.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    """
    ####################################################
    Candles
    """
    # Subparser for fetching candles data
    candles_parser = subparsers.add_parser(
        "download_candles", help="Fetch and store candle data"
    )
    candles_parser.add_argument("ticker", type=str, help="Ticker symbol")
    candles_parser.add_argument(
        "--interval", type=str, default="1m", help="Candle interval (default: 1m)"
    )
    candles_parser.add_argument(
        "--period", type=str, default="max", help="Data period (default: max)"
    )
    candles_parser.add_argument(
        "--db_path",
        type=str,
        default="default.db",
        help="Path to the database (default: default.db)",
    )  # Added a default DB path since it's used a lot

    etf_parser = subparsers.add_parser(
        "download_candles_using_list", help="Fetch and store candle data"
    )
    etf_parser.add_argument("ticker", type=str, help="Ticker symbol")
    etf_parser.add_argument(
        "--interval", type=str, default="1m", help="Candle interval (default: 1m)"
    )
    etf_parser.add_argument(
        "--period", type=str, default="max", help="Data period (default: max)"
    )
    etf_parser.add_argument(
        "--db_path",
        type=str,
        default="default.db",
        help="Path to the database (default: default.db)",
    )  # Added a default DB path since it's used a lot
    etf_parser.add_argument(
        "--force_update", action="store_true", help="Force update from web"
    )
    etf_parser.add_argument(
        "--sleep",
        type=int,
        default=5,
        help="Time to sleep between requests",
    )
    """
    ####################################################
    Financial Statements
    """
    # Subparser for fetching financial statements
    statement_parser = subparsers.add_parser(
        "download_statements", help="Fetch and store financial statements"
    )
    statement_parser.add_argument("ticker", type=str, help="Ticker symbol")
    statement_parser.add_argument("statement", type=str, help="Ticker symbol")
    statement_parser.add_argument(
        "--period",
        type=str,
        default="Q",
        help="Period of Statement (default: Q, meaning quarterly)",
    )

    multi_statement_parser = subparsers.add_parser(
        "download_statements_using_list",
        help="Fetch and store financial statements using a list as a source.",
    )
    multi_statement_parser.add_argument("list_name", type=str, help="Name of the list")
    multi_statement_parser.add_argument("statement", type=str, help="Type of statement")
    multi_statement_parser.add_argument(
        "--period",
        type=str,
        default="Q",
        help="Period of Statement (default: Q, meaning quarterly)",
    )
    multi_statement_parser.add_argument(
        "--sleep",
        type=int,
        default=5,
        help="Time to sleep between requests",
    )
    """
    ####################################################
    Options
    """
    # Subparser for fetching candles data
    options_parser = subparsers.add_parser(
        "download_options", help="Fetch and store options chain"
    )
    options_parser.add_argument(
        "ticker", type=str, help="Ticker symbol"
    )  # Added a default DB path since it's used a lot

    multi_option_parser = subparsers.add_parser(
        "download_options_using_list",
        help="Fetch and store options chain using a list as a source.",
    )
    multi_option_parser.add_argument("list_name", type=str, help="Name of the list")
    multi_option_parser.add_argument(
        "--sleep",
        type=int,
        default=5,
        help="Time to sleep between requests",
    )
    """
    ####################################################
    Command Execution
    """
    args = parser.parse_args()
    if args.command == "download_candles":
        # try: #Added try catch
        # from modules.data.candles import get_candles # Local import
        candles_data = get_candles(
            ticker=args.ticker,
            interval=args.interval,
            period=args.period,
            db_path=get_candles_path(),
            force_update=args.force_update,
        )

        app_log.add(
            f"[Candles] Downloaded candles for ticker: {args.ticker} and interval {args.interval}"
        )

        print(candles_data)  # Or save to file etc

    elif args.command == "download_candles_using_list":
        tickers = get_tickers_from_etf(ETF_PATH.format(args.ticker))
        for t in tickers:
            candles_data = get_candles(
                ticker=t,
                interval=args.interval,
                period=args.period,
                db_path=get_candles_path(),
                force_update=args.force_update,
            )
            app_log.add(
                f"[Candles] Downloaded candles for ticker: {t} and interval {args.interval}"
            )
            time.sleep(args.sleep)

    elif args.command == "download_statements":
        _download_statement(args.ticker, args.period, args.statement)
        app_log.add(
            f"[Statements] Downloaded '{args.statement}' for ticker {args.ticker}"
        )

    elif args.command == "download_statements_using_list":
        tickers = get_tickers_from_etf(ETF_PATH.format(args.list_name))
        for t in tickers:
            _download_statement(t, args.period, args.statement)
            app_log.add(f"[Statements] Downloaded '{args.statement}' for ticker {t}")
            time.sleep(args.sleep)
    elif args.command == "download_options":
        chain = get_options_chain(args.ticker, get_candles_path())
        app_log.add(f"[Options] Downloaded options chain for ticker {args.ticker}")

    elif args.command == "download_options_using_list":
        tickers = get_tickers_from_etf(ETF_PATH.format(args.list_name))
        path = get_candles_path()
        for t in tickers:
            get_options_chain(t, path)
            app_log.add(f"[Options] Downloaded options chain for ticker {t}")
            time.sleep(args.sleep)
