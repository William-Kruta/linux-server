import pandas as pd
import polars as pl
import yfinance as yf
import datetime as dt

from modules.data.database_connection import DatabaseConnection


def get_candles(
    ticker: str,
    interval: str = "1m",
    period: str = "max",
    db_path: str = "",
    force_update: bool = False,
):
    ticker = ticker.upper()
    table_name = f"candles_{interval}"
    print(f"TableName: {table_name}")
    _create_table(db_path, table_name)
    if force_update:
        web_candles = _fetch_candles(ticker, interval, period)
        _insert_candles(ticker, web_candles, db_path, table_name)
        local_candles = _read_candles(ticker, db_path, table_name)
    else:
        local_candles = _read_candles(ticker, db_path, table_name)
        if local_candles.is_empty():
            web_candles = _fetch_candles(ticker, interval, period)
            _insert_candles(ticker, web_candles, db_path, table_name)
            local_candles = _read_candles(ticker, db_path, table_name)
        else:
            last_date = local_candles["timestamp"].max()
            stale = _is_stale(last_date, 3)
            if stale:
                web_candles = _fetch_candles(ticker, interval, period)
                _insert_candles(ticker, web_candles, db_path, table_name)
                local_candles = _read_candles(ticker, db_path, table_name)

    return local_candles


def _fetch_candles(ticker: str, interval: str, period: str) -> pl.DataFrame:
    mapping = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    if interval == "1m":
        mapping["Datetime"] = "timestamp"
    elif interval == "1d":
        mapping["Date"] = "timestamp"

    candles = yf.download(
        ticker, interval=interval, period=period, multi_level_index=False
    )

    candles.reset_index(inplace=True)
    candles.rename(mapping, axis=1, inplace=True)
    candles = candles[list(mapping.values())]
    candles = pl.from_pandas(candles)
    return candles


def _read_candles(ticker: str, db_path: str, table_name: str):
    ticker = ticker.upper()
    with DatabaseConnection(db_path) as db:
        query = f"""SELECT * FROM {table_name} WHERE ticker = '{ticker}'"""
        records = pl.read_database(query, db)
    if not records.is_empty():
        records = records.drop(["ticker"])
    return records


def _insert_candles(ticker: str, df: pl.DataFrame, db_path: str, table_name: str):
    rows_to_append = []
    for row in df.rows(named=True):
        rows_to_append.append(
            (
                ticker,
                row["timestamp"],
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
            )
        )
    with DatabaseConnection(db_path) as db:
        query = """
                INSERT OR IGNORE INTO {} (ticker, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """.format(
            table_name
        )
        db.executemany(query, rows_to_append)


def _create_table(db_path: str, table_name: str):
    with DatabaseConnection(db_path) as db:
        create_table = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                ticker TEXT NOT NULL,       
                timestamp DATETIME NOT NULL,         
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                PRIMARY KEY (ticker, timestamp)
            )
        """
        db.execute(create_table)


def _is_stale(date: str, threshold_days: int):
    if isinstance(date, str):
        if "." in date:
            date = date.split(".")[0]
        try:
            date = dt.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").date()
        except ValueError:
            try:
                date = dt.datetime.strptime(date, "%Y-%m-%d %H:%M:%S%z").date()
            except ValueError:
                date = dt.datetime.strptime(date, "%Y-%m-%d").date()
    now = dt.datetime.now().date()
    try:
        delta = now - date
    except TypeError:
        delta = now - date.date()
    if delta.days > threshold_days:
        return True
    else:
        return False


def calculate_directional_percentage_change(
    df: pl.DataFrame, column_name: str, period: int, forward: bool
) -> pl.DataFrame:
    """
    Calculates the forward percentage change over a specified period for a given column in a Polars DataFrame.
    Forward change is calculated relative to the *current* row's value.  Newer dates should be at the tail of the DataFrame.

    Args:
        df: The Polars DataFrame.  It's assumed that newer dates are at the tail of the DataFrame
        column_name: The name of the column to calculate the percentage change for.
        period: The period over which to calculate the forward percentage change (e.g., 30 for 30 days).

    Returns:
        A new Polars DataFrame with an additional column named "{column_name}_forward_pct_change"
        containing the forward percentage change.
        The forward percentage change is calculated as ((future_value - current_value) / current_value) * 100.
    """

    if forward:
        result_label = f"{period}_forward_pct_change"
        period *= -1
    else:
        result_label = f"{period}_backward_pct_change"

    return df.with_columns(
        ((pl.col(column_name).shift(period) / pl.col(column_name)) - 1).alias(
            result_label
        )
    )
