import numpy as np
import polars as pl
import yfinance as yf
import datetime as dt
from modules.data.database_connection import DatabaseConnection
from modules.data.candles import get_candles, calculate_directional_percentage_change


def get_news(ticker: str, db_path: str):
    table_name = "news"
    _create_table(db_path, table_name)

    local_news = _read_news(ticker, db_path, table_name)

    if local_news.is_empty():
        candles = get_candles(ticker, "1d", db_path=db_path)
        news = _fetch_yahoo_news(ticker, candles)
        _insert_news(news, db_path, table_name)
        local_news = _read_news(ticker, db_path, table_name)

    else:
        last_date = local_news["date_collected"].max()
        stale = is_stale(last_date, 3)
        if stale: 
            candles = get_candles(ticker, "1d", db_path=db_path)
            news = _fetch_yahoo_news(ticker, candles)
            _insert_news(news, db_path, table_name)
            local_news = _read_news(ticker, db_path, table_name)
    return local_news

def _fetch_yahoo_news(
    ticker: str, candles: pl.DataFrame, backtest_windows: list = [1, 5, 22]
):
    ticker = ticker.upper()

    for bw in backtest_windows:

        candles = calculate_directional_percentage_change(
            candles, "close", bw, forward=True
        )
        candles = calculate_directional_percentage_change(
            candles, "close", bw, forward=False
        )

    now = dt.datetime.now().date()
    obj = yf.Ticker(ticker)
    news = obj.news
    if not news:
        return f"No news found for ticker '{ticker}'."
    data = []
    for n in news:
        content = n["content"]
        try:
            forward_1 = candles["1_backward_pct_change"][-1] * 100
        except TypeError:
            forward_1 = np.nan
        try:
            forward_5 = candles["5_backward_pct_change"][-1] * 100
        except TypeError:
            forward_5 = np.nan
        try:
            forward_22 = candles["22_backward_pct_change"][-1] * 100
        except TypeError:
            forward_22 = np.nan
        try:
            backward_1 = candles["1_forward_pct_change"][-1] * 100
        except TypeError:
            backward_1 = np.nan
        try:
            backward_5 = candles["5_forward_pct_change"][-1] * 100
        except TypeError:
            backward_5 = np.nan
        try:
            backward_22 = candles["22_forward_pct_change"][-1] * 100
        except TypeError:
            backward_22 = np.nan

        pub_date = dt.datetime.strptime(content["pubDate"], "%Y-%m-%dT%H:%M:%SZ")

        data.append(
            (
                ticker,
                content["title"],
                content["summary"],
                pub_date,
                content["canonicalUrl"]["url"],
                candles["close"][-1],
                backward_1,
                backward_5,
                backward_22,
                forward_1,
                forward_5,
                forward_22,
                now,
                "yahoo",
            )
        )
    return data


def _read_news(ticker: str, db_path: str, table_name: str):
    with DatabaseConnection(db_path) as db:
        query = f"""SELECT * FROM {table_name} WHERE ticker = '{ticker}'"""
        records = pl.read_database(query, db)
        return records


def _create_table(db_path: str, table_name: str):
    with DatabaseConnection(db_path) as db:
        create_table = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                ticker TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                published DATETIME NOT NULL,
                url TEXT,
                stockPrice REAL, 
                previous_day_change REAL, 
                previous_week_change REAL, 
                previous_month_change REAL, 
                next_day_change REAL, 
                next_week_change REAL, 
                next_month_change REAL, 
                date_collected DATETIME NOT NULL,
                source TEXT NOT NULL,
                PRIMARY KEY (ticker, published)
            )
        """
        db.execute(create_table)


def _insert_news(news: list, db_path: str, table_name: str):
    with DatabaseConnection(db_path) as db:
        query = f"""
        INSERT INTO {table_name} (ticker, title, summary, published, url, stockPrice, previous_day_change, previous_week_change, previous_month_change, next_day_change, next_week_change, next_month_change, date_collected, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        db.executemany(query, news)


def is_stale(date: str, stale_threshold: int):

    if isinstance(date, str):
        date = dt.datetime.strptime(date, "%Y-%m-%d").date()

    now = dt.datetime.now().date()

    delta = now - date
    if delta.days >= stale_threshold:
        return True
    else:
        return False
