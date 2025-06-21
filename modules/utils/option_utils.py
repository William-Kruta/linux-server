import re
import datetime as dt
import pandas as pd
import numpy as np
from scipy.stats import norm


def parse_expiration_date(contract_symbol: str):
    match = re.search(r"[A-Z]{1,6}(\d{6})[CP]", contract_symbol)
    if match:
        exp_date = match.group(1)  # '250530'
        # Optionally convert to YYYY-MM-DD
        formatted_date = f"20{exp_date[:2]}-{exp_date[2:4]}-{exp_date[4:]}"
        return formatted_date


def parse_expiration_date_2(contract_symbol: str):
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


def get_strike_spread(row: pd.Series, return_percent: bool = True) -> float:
    if row["type"] == "call":
        # (S - K) / K
        spread = (row["stock_price"] - row["strike"]) / row["strike"]

    elif row["type"] == "put":
        # (K - S) / K
        spread = (row["strike"] - row["stock_price"]) / row["strike"]
    if return_percent:
        spread *= 100
    return spread
