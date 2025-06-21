import polars as pl
import pandas as pd
import yfinance as yf
import datetime as dt
from modules.data.database_connection import DatabaseConnection

valid_statements = ["income_statement", "balance_sheet", "cash_flow"]


income_statement_mapping = {
    "Research And Development": "R&D",
    "Selling General And Administration": "SG&A",
    "Selling And Marketing Expense": "S&M",
    "General And Administrative Expense": "G&A",
    "Other Gand A": "Other G&A",
}


def create_table_name(statement: str, quarter: bool = False):
    if statement not in valid_statements:
        raise Exception
    if quarter:
        table_name = "quarterly_{}".format(statement)
    else:
        table_name = "annual_{}".format(statement)
    return table_name


def _get_stale_limit(quarter: bool):
    if quarter:
        return 120
    else:
        return 420


def get_income_statement(ticker: str, quarter: bool, db_path: str) -> pl.DataFrame:
    df = _get_statement(ticker, quarter, "income_statement", db_path)
    return df


def get_balance_sheet(ticker: str, quarter: bool, db_path: str) -> pl.DataFrame:
    df = _get_statement(ticker, quarter, "balance_sheet", db_path)
    return df


def get_cash_flow(ticker: str, quarter: bool, db_path: str) -> pl.DataFrame:
    df = _get_statement(ticker, quarter, "cash_flow", db_path)
    return df


def _get_statement(
    ticker: str, quarter: bool, statement: str, db_path: str
) -> pl.DataFrame:
    ticker = ticker.upper()
    table_name = create_table_name(statement, quarter)
    _create_table(db_path, table_name)
    local_data = _read_statement_data(ticker, db_path, table_name)
    if local_data.is_empty():
        data = _fetch_statement(ticker, quarter, statement)
        _insert_statement_data(data, db_path, table_name)
        local_data = _read_statement_data(ticker, db_path, table_name)
    else:
        dates = local_data.columns
        if "metric" in dates:
            dates = dates[1:]
        datetime_objects = [
            dt.datetime.strptime(date_string, "%Y-%m-%d").date()
            for date_string in dates
        ]
        # Find the maximum datetime object
        latest_datetime = max(datetime_objects)
        stale = is_stale(latest_datetime, _get_stale_limit(quarter))
        if stale:
            data = _fetch_statement(ticker, quarter, statement)
            _insert_statement_data(data, db_path, table_name)
            local_data = _read_statement_data(ticker, db_path, table_name)

    return local_data


def _fetch_statement(
    ticker: str, quarter: bool, statement_type: str, obj: yf.Ticker = None
):
    if obj is None:
        obj = yf.Ticker(ticker)

    if statement_type not in valid_statements:
        raise Exception

    if statement_type == "income_statement":
        if quarter:
            data = obj.quarterly_income_stmt
        else:
            data = obj.income_stmt
        data = data.rename(income_statement_mapping, axis=0)

    elif statement_type == "balance_sheet":
        if quarter:
            data = obj.quarterly_balance_sheet
        else:
            data = obj.balance_sheet

    elif statement_type == "cash_flow":
        if quarter:
            data = obj.quarterly_cash_flow
        else:
            data = obj.cash_flow

    data = data.iloc[:, ::-1]
    data = pl.from_pandas(data.reset_index())
    data = data.with_columns(pl.lit(ticker).alias("ticker"))
    melted = data.unpivot(
        index=["index", "ticker"], variable_name="date", value_name="value"
    )
    return melted


def _insert_statement_data(df: pl.DataFrame, db_path: str, table_name: str):
    rows_to_append = []
    for row in df.rows(named=True):
        rows_to_append.append((row["index"], row["ticker"], row["date"], row["value"]))

    with DatabaseConnection(db_path) as db:
        query = f"""INSERT OR IGNORE INTO {table_name} (metric, ticker, date, value) VALUES (?, ?, ?, ?)"""
        db.executemany(query, rows_to_append)


def _read_statement_data(ticker: str, db_path: str, table_name: str) -> pl.DataFrame:
    with DatabaseConnection(db_path) as db:
        query = (
            f"""SELECT * FROM {table_name} WHERE ticker='{ticker}' ORDER BY date ASC;"""
        )
        records = pl.read_database(query, db)
        records = records.pivot(
            values="value", index=["ticker", "date"], columns="metric"
        )
        # Convert to pandas to make edits easier
        records = records.to_pandas().T
        dates = records.loc["date"]
        dates = [dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S").date() for d in dates]
        records.columns = dates
        records.drop(["date", "ticker"], axis=0, inplace=True)
        records.index.name = "metric"
        records.reset_index(inplace=True)
        records["metric"] = records["metric"].astype(str)
        # Reconvert to polars.
        records = pl.from_pandas(records)
        return records


def _create_table(db_path: str, table_name: str):
    with DatabaseConnection(db_path) as db:
        query = f"""CREATE TABLE IF NOT EXISTS {table_name} (
                metric TEXT NOT NULL,
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                value REAL,
                PRIMARY KEY (metric, ticker, date)
                );"""
        db.execute(query)


def is_stale(date: str, stale_days_limit: int) -> bool:
    if isinstance(date, str):
        date = dt.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").date()
    now = dt.datetime.now().date()
    delta = now - date
    if delta.days >= stale_days_limit:
        return True
    else:
        return False
