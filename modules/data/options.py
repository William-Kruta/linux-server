import re
import pandas as pd
import polars as pl
import yfinance as yf
import sqlite3
import datetime as dt

from modules.data.candles import get_candles
from modules.data.database_connection import DatabaseConnection


chain_columns_mapping = {
    "openInterest": "OI",
    "impliedVolatility": "IV",
    "inTheMoney": "ITM",
}

columns = [
    "ticker",
    "contractSymbol",
    "lastTradeDate",
    "strike",
    "lastPrice",
    "bid",
    "ask",
    "change",
    "percentChange",
    "volume",
    "OI",
    "IV",
    "ITM",
    "contractSize",
    "currency",
    "type",
    "expirationDate",
    "dte",
    "date_collected",
    "time_collected",
]


def get_options_chain(ticker, db_path: str, all_expirations: bool = True, fetch_new: bool =True):
    ticker = ticker.upper()
    table_name = "options"
    _create_table(db_path, table_name)
    local_chain = _read_chain(ticker, db_path, table_name)
    
    if fetch_new:
        chain = _fetch_chain(ticker, db_path, all_expirations)
        try:
            chain = chain.rename(chain_columns_mapping)
        except pl.exceptions.ColumnNotFoundError:
            return pl.DataFrame()
        _insert_chain(ticker, chain, db_path, table_name)
        local_chain = _read_chain(ticker, db_path, table_name)
        
    return local_chain


def _fetch_chain(ticker: str, db_path: str, all_expirations: bool):
    obj = yf.Ticker(ticker)
    data_frames = []
    candles = get_candles(ticker, "1d", db_path=db_path)
    risk_free_rate = get_candles("^IRX", "1d", db_path=db_path)["close"][-1] / 100
    obj = yf.Ticker(ticker)
    exp_dates = obj.options
    stock_price = candles["close"][-1]

    if not all_expirations:
        exp_dates = [exp_dates[0]]
    for exp in exp_dates:
        if not exp:
            continue
        chain = obj.option_chain(exp)
        call_df = pl.from_pandas(chain.calls)
        put_df = pl.from_pandas(chain.puts)

        cols_to_cast = ["openInterest", "volume"]
        for col_name in cols_to_cast:
            if col_name in call_df.columns:
                call_df = call_df.with_columns(
                    pl.col(col_name).cast(pl.Float64, strict=False)
                )
            if col_name in put_df.columns:
                put_df = put_df.with_columns(
                    pl.col(col_name).cast(pl.Float64, strict=False)
                )

        call_df = call_df.with_columns(
            [
                pl.lit("call").alias("type"),
                pl.lit(ticker).alias("ticker"),
                pl.lit(stock_price).alias("stock_price"),
                pl.lit(0).alias("dividend_yield"),
                pl.lit(risk_free_rate).alias("risk_free_rate"),
            ]
        )
        put_df = put_df.with_columns(
            [
                pl.lit("put").alias("type"),
                pl.lit(ticker).alias("ticker"),
                pl.lit(stock_price).alias("stock_price"),
                pl.lit(0).alias("dividend_yield"),
                pl.lit(risk_free_rate).alias("risk_free_rate"),
            ]
        )
        data_frames.extend([call_df, put_df])

    if not data_frames:
        return pl.DataFrame()

    data = pl.concat(data_frames, how="diagonal")
    data = data.with_columns(
        [
            pl.col("contractSymbol")
            .map_elements(parse_expiration_date, return_dtype=pl.Date, skip_nulls=True)
            .alias("expirationDate"),
        ]
    )
    data = data.with_columns(
        [
            pl.col("expirationDate")
            .map_elements(
                get_days_to_expiration, return_dtype=pl.Int64, skip_nulls=True
            )
            .alias("dte")
        ]
    )
    """
    ========================================
    Add any greek code here. 
    
    """

    return data


def parse_expiration_date(contract_symbol: str):
    """
    Parses the expiration date from a yfinance contract symbol using regex.
    This is robust against variable-length ticker symbols.
    """
    if not isinstance(contract_symbol, str):
        return None
    # Find the first digit in the contract symbol, which marks the start of the date
    match = re.search(r"\d", contract_symbol)
    if not match:
        return None

    start_index = match.start()
    # The date is the 6 characters from the first digit
    date_str = contract_symbol[start_index : start_index + 6]

    try:
        # Return a date object, which is what pl.Date expects
        return dt.datetime.strptime(date_str, "%y%m%d").date()
    except (ValueError, TypeError):
        # Return None if parsing fails
        return None


def get_days_to_expiration(expiration_date, reference_date: str = ""):
    if isinstance(expiration_date, str):
        expiration_date = dt.datetime.strptime(expiration_date, "%Y-%m-%d").date()

    if reference_date == "":
        reference_date = dt.datetime.now().date()
    else:
        if isinstance(reference_date, str):
            reference_date = dt.datetime.strptime(reference_date, "%Y-%m-%d").date()
    delta = expiration_date - reference_date
    return delta.days


def _create_table(db_path: str, table_name: str):
    with DatabaseConnection(db_path) as db:
        query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            ticker TEXT NOT NULL,
            contractSymbol TEXT,
            lastTradeDate TIMESTAMP,
            strike REAL NOT NULL,
            lastPrice REAL,
            bid REAL,
            ask REAL,
            change REAL,
            percentChange REAL,
            volume INTEGER,
            OI INTEGER,
            IV REAL,
            ITM BOOLEAN,
            contractSize TEXT,
            currency TEXT,
            type TEXT CHECK(type IN ('call', 'put')) NOT NULL,
            expirationDate DATE NOT NULL,
            dte INTEGER,
            date_collected TIMESTAMP,
            time_collected TIMESTAMP,
            PRIMARY KEY (ticker, type, strike, expirationDate)
        )
        """
        db.execute(query)


def _read_chain(ticker: str, db_path: str, table_name: str) -> pl.DataFrame:

    with DatabaseConnection(db_path) as db:
        query = f"""SELECT * FROM {table_name} WHERE ticker = '{ticker}'"""
        records = pl.read_database(query, db)
        return records


def _insert_chain(ticker: str, df: pl.DataFrame, db_path: str, table_name: str):

    rows_to_append = []
    now = dt.datetime.now()
    for row in df.rows(named=True):
        row = (
            ticker,
            row["contractSymbol"],
            row["lastTradeDate"],
            row["strike"],
            row["lastPrice"],
            row["bid"],
            row["ask"],
            row["change"],
            row["percentChange"],
            row["volume"],
            row["OI"],
            row["IV"],
            row["ITM"],
            row["contractSize"],
            row["currency"],
            row["type"],
            row["expirationDate"],
            row["dte"],
            now.date(),
            now.time(),
        )
        rows_to_append.append(row)

    with DatabaseConnection(db_path) as db:
        query = f"""
            INSERT INTO {table_name} (
                ticker,
                contractSymbol,
                lastTradeDate,
                strike,
                lastPrice,
                bid,
                ask,
                change,
                percentChange,
                volume,
                OI,
                IV,
                ITM,
                contractSize,
                currency,
                type,
                expirationDate,
                dte,
                date_collected,
                time_collected
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        db.executemany(query, rows_to_append)
