import requests
import json
import polars as pl


class API:
    def __init__(self, server_url: str, time_zone=None):
        self.server_url = server_url
        self.time_zone = time_zone
        if isinstance(self.time_zone, str):
            self.time_zone = self.time_zone.upper()

    def get_candles(self, ticker: str, interval: str):
        """
        Get candles from the server

        Parameters
        ----------
        ticker : str
            Ticker symbol of the stock.
        interval : str
            Interval of the candles ("1d" = Daily, "1m" = Minute intra-day candles)

        Returns
        -------
        pl.DataFrame
           Polars Dataframe
        """
        url = f"{self.server_url}/get_candles/{ticker.upper()}"
        resp = requests.get(url, params={"interval": interval})
        resp.raise_for_status()
        candles = resp.json()
        candles = json.loads(candles)
        df = pl.from_dicts(candles)
        df = df.with_columns(pl.col("timestamp").str.to_datetime())
        if self.time_zone == "PST":
            df = df.with_columns(
                pl.col("timestamp").dt.convert_time_zone("America/Los_Angeles")
            )
        return df

    def get_options(self, ticker: str, all_expirations: bool = True):
        url = f"{self.server_url}/get_options/{ticker.upper()}"
        resp = requests.get(url, params={"all_expirations": all_expirations})
        resp.raise_for_status()
        candles = resp.json()
        candles = json.loads(candles)
        df = pl.from_dicts(candles)
        print(f"DF: {df}")

        return df

    def get_income_statement(self, ticker: str, quarter: bool = False):
        url = f"{self.server_url}/get_income_statement/{ticker.upper()}"
        resp = requests.get(url, params={"quarter": quarter})
        resp.raise_for_status()
        candles = resp.json()
        candles = json.loads(candles)
        df = pl.from_dicts(candles)
        return df

    def get_balance_sheet(self, ticker: str, quarter: bool = False):
        url = f"{self.server_url}/get_balance_sheet/{ticker.upper()}"
        resp = requests.get(url, params={"quarter": quarter})
        resp.raise_for_status()
        candles = resp.json()
        candles = json.loads(candles)
        df = pl.from_dicts(candles)
        return df

    def get_cash_flow(self, ticker: str, quarter: bool = False):
        url = f"{self.server_url}/get_cash_flow/{ticker.upper()}"
        resp = requests.get(url, params={"quarter": quarter})
        resp.raise_for_status()
        candles = resp.json()
        candles = json.loads(candles)
        df = pl.from_dicts(candles)
        return df

    def get_news(self, ticker: str):
        url = f"{self.server_url}/get_news/{ticker.upper()}"
        resp = requests.get(url)
        resp.raise_for_status()
        candles = resp.json()
        candles = json.loads(candles)
        df = pl.from_dicts(candles)
        return df

    def test(self):
        url = f"{self.server_url}/test"
        resp = requests.get(url)
        resp.raise_for_status()
        print(f"Response: {resp.json()}")
