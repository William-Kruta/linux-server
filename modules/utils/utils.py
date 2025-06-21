import datetime as dt
import pandas as pd


def is_stale(date: str, threshold_days: int):
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


def is_stale_intraday(date, threshold_minutes):
    # Convert to seconds
    threshold_seconds = threshold_minutes * 60
    if isinstance(date, str):
        if "." in date:
            date = date.split(".")[0]
        try:
            date = dt.datetime.strptime(date, "%Y-%m-%d %H:%M:%S%z")
        except ValueError:
            # If that fails, try parsing without timezone, assuming UTC
            date = dt.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        now = dt.datetime.now(dt.timezone.utc)  # Make now timezone aware for comparison
        delta = now - date
        seconds_stale = int(delta.total_seconds())
        if seconds_stale > threshold_seconds:
            return True
        else:
            return False


def get_tickers_from_etf(
    path, column_name: str = "Symbol", clean_for_yahoo: bool = True
):
    df = pd.read_csv(path)
    tickers = df[column_name].to_list()
    if clean_for_yahoo:
        tickers = [t.replace(".", "-") for t in tickers]
    return tickers
