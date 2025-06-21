import os
import pandas as pd
import numpy as np
import polars as pl
import yfinance as yf
import sqlite3


import pandas as pd
import yfinance as yf
import sqlite3
from modules.data.candles import Candles
from modules.utils.mappings import chain_columns_mapping
from modules.utils.option_utils import (
    parse_expiration_date,
    parse_expiration_date_2,
    get_days_to_expiration,
    get_strike_spread,
)
from modules.utils.greeks import OptionGreeksCalculator, ManualGreeks

from scipy.stats import norm
import time
import datetime as dt


class OptionsChain:
    def __init__(self, option_db_path: str, candles_db_path: str):
        self.options_db_path = option_db_path
        self.candles_db_path = candles_db_path
        self.greeks = ManualGreeks()
        self.options_greeks = OptionGreeksCalculator()
        self.risk_free_rate = np.nan
        self.candles = {}
        self.stock_prices = {}

    def get_chain(self, tickers: list) -> pd.DataFrame:
        if isinstance(tickers, str):
            tickers = [tickers]  # Convert string to list to allow similar logic.

        self._fetch_chain_polars(tickers)

    def set_candles(self, tickers):
        obj = Candles(self.candles_db_path, "1d")
        for t in tickers:
            self.candles[t] = obj.get_candles(t)
            self.stock_prices[t] = self.candles[t]["close"].iloc[-1]

        self.candles_set = True

    def _fetch_chain(self, tickers: list) -> pd.DataFrame:
        if self.candles == {}:
            self.set_candles(tickers)
        data = []
        # Fetch chain from yahoo finance.
        for t in tickers:
            obj = yf.Ticker(t)
            exp_dates = obj.options
            stock_price = self.candles[t]["close"].iloc[-1]
            for exp in exp_dates:
                chain = obj.option_chain(exp)
                call_df = chain.calls
                call_df["type"] = "call"
                call_df["ticker"] = t
                call_df["stock_price"] = stock_price
                put_df = chain.puts
                put_df["type"] = "put"
                put_df["ticker"] = t
                put_df["stock_price"] = stock_price
                data.append(call_df)
                data.append(put_df)

        # Format chain
        data = pd.concat(data).reset_index(drop=True)
        data["expiration_date"] = data["contractSymbol"].apply(parse_expiration_date)
        data["dte"] = data["expiration_date"].apply(get_days_to_expiration)
        # data["strike_spread"] = (
        #     (data["stock_price"] - data["strike"]) / data["stock_price"]
        # ) * 100
        data["strike_spread"] = data.apply(lambda x: get_strike_spread(x), axis=1)
        data = data.apply(lambda x: self.backtest(x), axis=1)
        data.rename(chain_columns_mapping, axis=1, inplace=True)

        # Calculate greeks
        data = data.apply(lambda x: self._apply_greeks(x), axis=1)

        print(f"Data: {data}")

    def _fetch_chain_polars(self, tickers: list) -> pl.DataFrame:
        if self.candles == {}:
            self.set_candles(tickers)

        risk_free_rate = self._get_risk_free_rate()

        data_frames = []
        for t in tickers:
            obj = yf.Ticker(t)
            exp_dates = obj.options
            stock_price = self.stock_prices[t]
            print(f"Ticker: {t}, Stock Price: {stock_price}")

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
                        pl.lit(t).alias("ticker"),
                        pl.lit(stock_price).alias("stock_price"),
                        pl.lit(0).alias("dividend_yield"),
                        pl.lit(risk_free_rate).alias("risk_free_rate"),
                    ]
                )
                put_df = put_df.with_columns(
                    [
                        pl.lit("put").alias("type"),
                        pl.lit(t).alias("ticker"),
                        pl.lit(stock_price).alias("stock_price"),
                        pl.lit(0).alias("dividend_yield"),
                        pl.lit(risk_free_rate).alias("risk_free_rate"),
                    ]
                )
                data_frames.extend([call_df, put_df])

        if not data_frames:
            return pl.DataFrame()

        data = pl.concat(data_frames, how="diagonal")

        data = self._calculate_profit_probability(data)
        # data = self._calculate_greeks(data)
        column_mapping = {
            "type": "option_type",
            "stock_price": "underlying_price",
            "strike": "strike_price",
            "impliedVolatility": "volatility",
            "dte": "days_to_expiration",
            # These two already match, but including them is fine for clarity
            "dividend_yield": "dividend_yield",
            "risk_free_rate": "risk_free_rate",
        }
        data_renamed = data.rename(column_mapping)

        data_renamed = data_renamed.select(list(column_mapping.values()))
        # print(f"Data: {_data}")
        result_df = data_renamed.with_columns(
            pl.struct(data_renamed.columns)
            .map_elements(
                lambda row_dict: self.options_greeks.calculate_greeks(**row_dict),
                return_dtype=pl.Struct(
                    [
                        pl.Field("bs_price", pl.Float64),
                        pl.Field("delta", pl.Float64),
                        pl.Field("gamma", pl.Float64),
                        pl.Field("theta", pl.Float64),
                        pl.Field("vega", pl.Float64),
                        pl.Field("rho", pl.Float64),
                    ]
                ),
            )
            .alias("greeks")
        ).unnest("greeks")
        # cols = [
        #     "type",
        #     "stock_price",
        #     "strike",
        #     "impliedVolatility",
        #     "dte",
        #     "dividend_yield",
        #     "risk_free_rate",
        # ]

        # greeks_df = data.map_rows(
        #     lambda row: self.options_greeks.calculate_greeks(
        #         **dict(zip(cols, row))  # Create a dictionary and unpack it
        #     )
        # )
        print(f"Result: {result_df}")
        data = pl.concat(
            [
                data,
                result_df.select(
                    ["bs_price", "delta", "gamma", "theta", "vega", "rho"]
                ),
            ],
            how="horizontal",
        )
        data = data.rename(chain_columns_mapping)

        print("Data processed with Polars:")
        print(data.head())
        return data

    def _set_risk_free_rate(self, ticker: str = "^IRX"):
        obj = Candles(self.candles_db_path, interval="1d")
        candles = obj.get_candles(ticker)
        self.risk_free_rate = candles["close"].iloc[-1]

    def _get_risk_free_rate(
        self, ticker: str = "^IRX", return_percent: bool = False
    ) -> float:
        if self.risk_free_rate is np.nan:
            self._set_risk_free_rate(ticker)
        if not return_percent:
            rfr = self.risk_free_rate / 100
        return rfr

    def _apply_greeks(self, row: pd.Series):

        return row

    def backtest(self, row: pd.Series):
        ticker = row["ticker"]
        S = self.stock_prices[ticker]
        K = row["strike"]
        T = row["dte"] / 365
        r = self._get_risk_free_rate()
        try:
            sigma = row["IV"]
        except KeyError:
            sigma = row["impliedVolatility"]
        _type = row["type"]
        short_prob = self.greeks.calculate_probability_of_profit(
            S, K, T, r, sigma, row["bid"], _type, "short"
        )
        long_prob = self.greeks.calculate_probability_of_profit(
            S, K, T, r, sigma, row["ask"], _type, "long"
        )
        row["short_probability_of_profit"] = short_prob
        row["long_probability_of_profit"] = long_prob
        return row

    def _backtest_polars(self, row: dict) -> dict:
        """
        A version of the backtest function that accepts a dictionary (a polars row)
        and returns a dictionary of new values.
        """
        ticker = row["ticker"]
        S = self.stock_prices[ticker]
        K = row["strike"]
        T = row["dte"] / 365
        r = self._get_risk_free_rate()

        # Use .get() for safe access in case a column is missing
        sigma = row.get("impliedVolatility") or row.get("IV", 0.0)
        _type = row["type"]

        short_prob = self.greeks.calculate_probability_of_profit(
            S, K, T, r, sigma, row["bid"], _type, "short"
        )
        long_prob = self.greeks.calculate_probability_of_profit(
            S, K, T, r, sigma, row["ask"], _type, "long"
        )

        return {
            "short_probability_of_profit": short_prob,
            "long_probability_of_profit": long_prob,
        }

    def _calculate_profit_probability(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates greeks and probability of profit using Polars expressions.
        This replaces the row-wise backtest function.
        """
        r = self._get_risk_free_rate()

        # Helper function to calculate d1 and d2 using Polars expressions
        def calculate_d2_expr(S_expr, K_expr, T_expr, r_lit, sigma_expr):
            # Corrected usage of .log() method on the expression
            d1 = (
                (S_expr / K_expr).log() + (r_lit + 0.5 * sigma_expr.pow(2)) * T_expr
            ) / (sigma_expr * T_expr.sqrt())
            return d1 - sigma_expr * T_expr.sqrt()

        df = df.with_columns(
            [
                pl.col("contractSymbol")
                .map_elements(
                    parse_expiration_date_2, return_dtype=pl.Date, skip_nulls=True
                )
                .alias("expiration_date"),
            ]
        )

        df = df.with_columns(
            [
                pl.col("expiration_date")
                .map_elements(
                    get_days_to_expiration, return_dtype=pl.Int64, skip_nulls=True
                )
                .alias("dte")
            ]
        )

        # Define common expressions
        S = pl.col("stock_price")
        K = pl.col("strike")
        T = (pl.col("dte") / 365.0).clip(lower_bound=1e-9)  # Time in years, avoid T=0
        sigma = (
            pl.col("impliedVolatility").fill_null(0.0).clip(lower_bound=1e-9)
        )  # Avoid sigma=0

        # Calculate breakeven prices
        long_breakeven = (
            pl.when(pl.col("type") == "call")
            .then(K + pl.col("ask"))
            .otherwise(K - pl.col("ask"))
        )
        short_breakeven = (
            pl.when(pl.col("type") == "call")
            .then(K + pl.col("bid"))
            .otherwise(K - pl.col("bid"))
        )

        # Calculate d2 for both long and short breakeven points
        d2_long = calculate_d2_expr(S, long_breakeven, T, pl.lit(r), sigma)
        d2_short = calculate_d2_expr(S, short_breakeven, T, pl.lit(r), sigma)

        # Calculate Probability of Profit using nested `when/then/otherwise`
        prob_long = (
            pl.when(pl.col("ask").is_not_null() & (long_breakeven > 0))
            .then(pl.when(pl.col("type") == "call").then(d2_long).otherwise(-d2_long))
            .otherwise(None)
            .map_elements(
                lambda d: norm.cdf(d) * 100 if d is not None else None,
                return_dtype=pl.Float64,
            )
        )

        prob_short = (
            pl.when(pl.col("bid").is_not_null() & (short_breakeven > 0))
            .then(pl.when(pl.col("type") == "call").then(-d2_short).otherwise(d2_short))
            .otherwise(None)
            .map_elements(
                lambda d: norm.cdf(d) * 100 if d is not None else None,
                return_dtype=pl.Float64,
            )
        )

        return df.with_columns(
            [
                prob_long.alias("long_probability_of_profit"),
                prob_short.alias("short_probability_of_profit"),
                (
                    pl.when(pl.col("type") == "call")
                    .then((S - K) / K)
                    .otherwise((K - S) / K)
                    * 100
                ).alias("strike_spread"),
            ]
        )

    def _calculate_greeks(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculates option greeks using Polars expressions."""
        r = self._get_risk_free_rate()

        # Define common expressions
        S = pl.col("stock_price")
        K = pl.col("strike")
        # Ensure DTE exists before this function is called
        T = (pl.col("dte") / 365.0).clip(lower_bound=1e-9)
        sigma = pl.col("impliedVolatility").fill_null(0.0).clip(lower_bound=1e-9)

        # Helper function to calculate d1 and d2
        def calculate_d1_d2_expr(S_expr, K_expr, T_expr, r_lit, sigma_expr):
            d1 = (
                (S_expr / K_expr).log() + (r_lit + 0.5 * sigma_expr.pow(2)) * T_expr
            ) / (sigma_expr * T_expr.sqrt())
            d2 = d1 - sigma_expr * T_expr.sqrt()
            return d1, d2

        d1, d2 = calculate_d1_d2_expr(S, K, T, pl.lit(r), sigma)

        # Helpers to apply scipy functions vectorized
        norm_cdf_expr = lambda x: x.map_elements(
            lambda s: norm.cdf(s) if s is not None and np.isfinite(s) else None,
            return_dtype=pl.Float64,
        )
        norm_pdf_expr = lambda x: x.map_elements(
            lambda s: norm.pdf(s) if s is not None and np.isfinite(s) else None,
            return_dtype=pl.Float64,
        )

        # --- GREEKS ---
        delta = (
            pl.when(pl.col("type") == "call")
            .then(norm_cdf_expr(d1))
            .otherwise(norm_cdf_expr(d1) - 1)
        )
        gamma = norm_pdf_expr(d1) / (S * sigma * T.sqrt())
        vega = (S * norm_pdf_expr(d1) * T.sqrt()) / 100

        theta_part1 = -(S * norm_pdf_expr(d1) * sigma) / (2 * T.sqrt())
        theta_call_part2 = -pl.lit(r) * K * (-(pl.lit(r)) * T).exp() * norm_cdf_expr(d2)
        theta_put_part2 = pl.lit(r) * K * (-(pl.lit(r)) * T).exp() * norm_cdf_expr(-d2)
        theta = (
            pl.when(pl.col("type") == "call")
            .then(theta_part1 + theta_call_part2)
            .otherwise(theta_part1 + theta_put_part2)
        ) / 365

        rho_call = (K * T * (-(pl.lit(r)) * T).exp() * norm_cdf_expr(d2)) / 100
        rho_put = (-K * T * (-(pl.lit(r)) * T).exp() * norm_cdf_expr(-d2)) / 100
        rho = pl.when(pl.col("type") == "call").then(rho_call).otherwise(rho_put)

        return df.with_columns(
            delta.alias("delta"),
            gamma.alias("gamma"),
            vega.alias("vega"),
            theta.alias("theta"),
            rho.alias("rho"),
        )


# class OptionsChain:
#     def __init__(
#         self, tickers: list[str], db_path: str, candles_db_path: str, log: bool = True
#     ):
#         self.db_path = db_path
#         self.candles_db_path = candles_db_path
#         self.conn = sqlite3.connect(self.db_path)
#         self.cursor = self.conn.cursor()

#         self._create_options_table()

#         if isinstance(tickers, str):
#             self.tickers = [tickers]
#         else:
#             self.tickers = tickers
#         self.log = log
#         # Class variables
#         self.chain = pd.DataFrame()
#         self.options_calc = OptionGreeksCalculator()
#         self.candles = {}

#     def set_candles(self):
#         candle = Candles(self.candles_db_path)
#         for t in self.tickers:
#             data = candle.get_candles(t)
#             self.candles[t] = {"candles": data, "spot_price": data["close"].iloc[-1]}

#     def get_candles(self):
#         if self.candles == {}:
#             self.set_candles()
#         return self.candles

#     def get_chain(self):
#         chain = self._fetch_chain()
#         self._create_options_table()

#         self.add_to_table(chain)

#     def _fetch_chain(self) -> pd.DataFrame:
#         all_data = []
#         start = time.time()
#         now = dt.datetime.now()
#         date_collected = now.date()
#         time_collected = now.strftime("%H:%M:%S")
#         for t in self.tickers:
#             obj = yf.Ticker(t)
#             expiration_dates = obj.options
#             for exp in expiration_dates:
#                 chain = obj.option_chain(exp)
#                 calls = chain.calls
#                 puts = chain.puts
#                 _symbol = calls["contractSymbol"].iloc[0]
#                 exp_date = parse_expiration_date(_symbol)
#                 calls["ticker"] = t
#                 puts["ticker"] = t
#                 calls["type"] = "call"
#                 puts["type"] = "put"
#                 calls["expirationDate"] = exp_date
#                 puts["expirationDate"] = exp_date
#                 calls["dte"] = get_days_to_expiration(exp_date)
#                 puts["dte"] = get_days_to_expiration(exp_date)
#                 all_data.append(calls)
#                 all_data.append(puts)
#         chain = pd.concat(all_data, ignore_index=True)
#         chain.rename(columns=chain_columns_mapping, inplace=True)
#         if self.candles == {}:
#             self.set_candles()
#         chain = chain.apply(lambda row: self._apply_greeks(row), axis=1)
#         chain["date_collected"] = date_collected
#         chain["time_collected"] = time_collected

#         end = time.time()
#         print(f"Chain: {chain}")
#         elapse = end - start
#         if self.log:
#             print(f"Fetched option chain data for: {self.tickers}")
#             print(f"Elapse: {elapse}")
#         return chain

#     def _apply_greeks(self, row: pd.Series):

#         greek_data = self.options_calc.calculate_greeks(
#             option_type=row["type"],
#             underlying_price=self.candles[row["ticker"]]["spot_price"],
#             strike_price=row["strike"],
#             risk_free_rate=0.04,
#             volatility=row["IV"],
#             dividend_yield=0,
#             days_to_expiration=row["dte"],
#         )
#         row["bs_price"] = greek_data["bs_price"]
#         row["delta"] = greek_data["delta"]
#         row["gamma"] = greek_data["gamma"]
#         row["theta"] = greek_data["theta"]
#         row["vega"] = greek_data["vega"]
#         row["rho"] = greek_data["rho"]
#         return row

#     def _create_options_table(self):
#         query = """
#         CREATE TABLE IF NOT EXISTS options_chain (
#             contractSymbol TEXT,
#             lastTradeDate TIMESTAMP,
#             strike REAL NOT NULL,
#             lastPrice REAL,
#             bid REAL,
#             ask REAL,
#             change REAL,
#             percentChange REAL,
#             volume INTEGER,
#             OI INTEGER,
#             IV REAL,
#             ITM BOOLEAN,
#             contractSize TEXT,
#             currency TEXT,
#             ticker TEXT NOT NULL,
#             type TEXT CHECK(type IN ('call', 'put')) NOT NULL,
#             expirationDate DATE NOT NULL,
#             dte INTEGER
#             bs_price REAL,
#             delta REAL,
#             gamma REAL,
#             theta REAL,
#             vega REAL,
#             rho REAL,
#             date_collected TIMESTAMP,
#             time_collected TIMESTAMP,
#             PRIMARY KEY (contractSymbol, date_collected, time_collected)
#         )
#         """
#         self.cursor.execute(query)
#         self.conn.commit()

#     def add_to_table(self, data: pd.DataFrame):

#         rows_to_append = []
#         for index, row in data.iterrows():
#             rows_to_append.append(row)
#         query = """
#             INSERT INTO options_chain (
#                 contractSymbol,
#                 lastTradeDate,
#                 strike,
#                 lastPrice,
#                 bid,
#                 ask,
#                 change,
#                 percentChange,
#                 volume,
#                 OI,
#                 IV,
#                 ITM,
#                 contractSize,
#                 currency,
#                 ticker,
#                 type,
#                 expirationDate,
#                 dte,
#                 bs_price,
#                 delta,
#                 gamma,
#                 theta,
#                 vega,
#                 rho
#             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#         """
#         self.cursor.execute(query, rows_to_append)
#         self.conn.commit()

#     def read_table(self) -> pd.DataFrame:
#         query = "SELECT * FROM options_chain"
#         return pd.read_sql_query(query, self.conn)

#     def query_by_ticker(self, ticker: str, option_type=None):
#         if not option_type:
#             query = f"SELECT * FROM options_chain WHERE contractSymbol = '{ticker}'"
#         else:
#             query = f"SELECT * FROM options_chain WHERE contractSymbol = '{ticker}' AND type = '{option_type}'"
#         return pd.read_sql_query(query, self.conn)
