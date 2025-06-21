import math
import pandas as pd
import numpy as np
from scipy.stats import norm

import QuantLib as ql


class ManualGreeks:
    def __init__(self):
        pass

    def calculate_d1_d2(self, S, K, T, r, sigma):
        """
        Helper function to calculate d1 and d2, which are used in the
        Black-Scholes formula and the greeks.

        Args:
            S (float): Current price of the underlying asset (e.g., stock price)
            K (float): Strike price of the option
            T (float): Time to expiration (in years)
            r (float): Risk-free interest rate (annual)
            sigma (float): Volatility of the underlying asset's returns (annual)

        Returns:
            tuple: A tuple containing d1 and d2.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    def calculate_delta(self, S, K, T, r, sigma, option_type="call"):
        """
        Calculates the Delta of an option. Delta measures the rate of change of the
        option's price with respect to a $1 change in the underlying asset's price.

        Args:
            S (float): Current price of the underlying asset
            K (float): Strike price of the option
            T (float): Time to expiration (in years)
            r (float): Risk-free interest rate
            sigma (float): Volatility of the underlying asset
            option_type (str): Type of the option, 'call' or 'put'.

        Returns:
            float: The Delta of the option.
        """
        d1, _ = self.calculate_d1_d2(S, K, T, r, sigma)

        if option_type.lower() == "call":
            delta = norm.cdf(d1)
        elif option_type.lower() == "put":
            delta = norm.cdf(d1) - 1
        else:
            raise ValueError("Invalid option type. Choose 'call' or 'put'.")

        return delta

    def calculate_gamma(self, S, K, T, r, sigma):
        """
        Calculates the Gamma of an option. Gamma measures the rate of change in an
        option's delta per $1 change in the price of the underlying asset.

        Args:
            S (float): Current price of the underlying asset
            K (float): Strike price of the option
            T (float): Time to expiration (in years)
            r (float): Risk-free interest rate
            sigma (float): Volatility of the underlying asset

        Returns:
            float: The Gamma of the option.
        """
        d1, _ = self.calculate_d1_d2(S, K, T, r, sigma)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma

    def calculate_probability_of_profit(
        self, S, K, T, r, sigma, premium, option_type="call", position_type="long"
    ):
        """
        Calculates the probability of an option position being profitable at expiration.
        This is the probability that the underlying price will be beyond the breakeven point.

        Args:
            S (float): Current price of the underlying asset
            K (float): Strike price of the option
            T (float): Time to expiration (in years)
            r (float): Risk-free interest rate
            sigma (float): Volatility of the underlying asset
            premium (float): The price at which the option was bought or sold
            option_type (str): Type of the option, 'call' or 'put'.
            position_type (str): Type of the position, 'long' or 'short'.

        Returns:
            float: The probability of profit as a percentage.
        """
        if option_type.lower() == "call":
            breakeven_price = K + premium
        elif option_type.lower() == "put":
            breakeven_price = K - premium
        else:
            raise ValueError("Invalid option type. Choose 'call' or 'put'.")

        # We can't have a negative breakeven price for this calculation
        if breakeven_price <= 0:
            return np.nan  # Not a meaningful calculation

        # Calculate d2 using the breakeven price as the effective strike price
        _, d2_breakeven = self.calculate_d1_d2(S, breakeven_price, T, r, sigma)

        if d2_breakeven == np.inf:  # Handle expired options
            if (
                (
                    position_type.lower() == "long"
                    and option_type.lower() == "call"
                    and S > breakeven_price
                )
                or (
                    position_type.lower() == "long"
                    and option_type.lower() == "put"
                    and S < breakeven_price
                )
                or (
                    position_type.lower() == "short"
                    and option_type.lower() == "call"
                    and S < breakeven_price
                )
                or (
                    position_type.lower() == "short"
                    and option_type.lower() == "put"
                    and S > breakeven_price
                )
            ):
                return 100.0
            else:
                return 0.0

        if position_type.lower() == "long":
            if option_type.lower() == "call":
                # Probability that S > breakeven_price
                prob_of_profit = norm.cdf(d2_breakeven)
            else:  # put
                # Probability that S < breakeven_price
                prob_of_profit = norm.cdf(-d2_breakeven)
        elif position_type.lower() == "short":
            if option_type.lower() == "call":
                # Probability that S < breakeven_price
                prob_of_profit = norm.cdf(-d2_breakeven)
            else:  # put
                # Probability that S > breakeven_price
                prob_of_profit = norm.cdf(d2_breakeven)
        else:
            raise ValueError("Invalid position type. Choose 'long' or 'short'.")

        return prob_of_profit * 100  # Return as a percentage


class Greeks:
    def __init__(self):
        pass

    def delta(self, S, K, T, r, sigma, option_type):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        if option_type == "C" or option_type == "call":
            delta = norm.cdf(d1)
        elif option_type == "P" or option_type == "put":
            delta = norm.cdf(d1) - 1
        else:
            raise ValueError("Invalid option type. Must be 'C' or 'P'.")

        return delta

    def theta(
        self, S, K, T, r, sigma, option_type: str, days_in_year: int = 365
    ) -> float:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        d2 = d1 - sigma * np.sqrt(T)

        # Standard normal probability density function
        N_prime_d1 = norm.pdf(d1)

        if option_type == "C" or option_type == "call":
            theta = -(S * N_prime_d1 * sigma / (2 * np.sqrt(T))) - (
                r * K * np.exp(-r * T) * norm.cdf(d2)
            )
        elif option_type == "P" or option_type == "put":
            theta = -(S * N_prime_d1 * sigma / (2 * np.sqrt(T))) + (
                r * K * np.exp(-r * T) * norm.cdf(-d2)
            )
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        # Convert from annual theta to daily theta
        theta_per_day = theta / days_in_year
        return theta_per_day

    def gamma(self, S, K, T, r, sigma) -> float:
        if T <= 0:
            return 0  # Gamma is zero at expiration
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma

    def vega(self, S, K, T, r, sigma) -> float:
        if T <= 0:
            return 0  # Vega is zero at expiration

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        return vega / 100

    def american_options(self, S, K, r, q, sigma, T, option_type, steps: int = 200):
        today = ql.Date().todaysDate()
        ql.Settings.instance().evaluationDate = today
        maturity = today + int(T * 365)
        payoff = ql.PlainVanillaPayoff(
            ql.Option.Put if option_type.lower() == "put" else ql.Option.Call, K
        )
        exercise = ql.AmericanExercise(today, maturity)
        option = ql.VanillaOption(payoff, exercise)

        # 2. build term structures
        day_count = ql.Actual365Fixed()
        rf_curve = ql.FlatForward(today, r, day_count)
        div_curve = ql.FlatForward(today, q, day_count)
        vol_curve = ql.BlackConstantVol(today, ql.NullCalendar(), sigma, day_count)

        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(S)),
            ql.YieldTermStructureHandle(div_curve),
            ql.YieldTermStructureHandle(rf_curve),
            ql.BlackVolTermStructureHandle(vol_curve),
        )

        # 3. choose a binomial engine (CRR here) and attach
        engine = ql.BinomialVanillaEngine(process, "crr", steps)
        option.setPricingEngine(engine)

        # 4. pull price and greeks
        return {
            "npv": option.NPV(),
            "delta": option.delta(),
            "theta": option.theta(),  # per-day theta
        }

    def binomial_american_greeks(self, S0, K, r, q, sigma, T, option_type="put", N=200):
        dt = T / N
        print(f"DF: {dt}")
        print(f"Type: {type(dt)}")
        u = math.exp(sigma * math.sqrt(dt))
        d = 1 / u
        disc = math.exp(-r * dt)
        p = (math.exp((r - q) * dt) - d) / (u - d)

        # terminal payoffs
        ST = np.array([S0 * u**j * d ** (N - j) for j in range(N + 1)])
        if option_type == "call":
            vals = np.maximum(ST - K, 0.0)
        else:
            vals = np.maximum(K - ST, 0.0)

        # back-induction
        for i in range(N - 1, -1, -1):
            ST = ST[: i + 1] / u  # move stock prices one step backwards
            cont = disc * (p * vals[1:] + (1 - p) * vals[:-1])
            if option_type == "call":
                exercise = np.maximum(ST - K, 0.0)
            else:
                exercise = np.maximum(K - ST, 0.0)
            vals = np.maximum(cont, exercise)

            # capture Δ at first step
            if i == 1:
                V_up, V_down = vals[1], vals[0]
                delta = (V_up - V_down) / (S0 * (u - d))

        price = vals[0]

        # Theta: re-price with T-dt and N-1 steps, holding S0 constant
        price_short, _, _ = (
            self.binomial_american_greeks(
                S0, K, r, q, sigma, T - dt, option_type, N - 1
            )
            if N > 1
            else (price, None, None)
        )
        theta = (price_short - price) / dt
        return price, delta, theta


class OptionGreeksCalculator:
    def __init__(self):
        self.day_count = ql.Actual365Fixed()
        self.calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        self.business_convention = ql.Following

    def calculate_greeks(
        self,
        option_type="call",
        underlying_price=100.0,
        strike_price=100.0,
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.01,
        days_to_expiration=30,
    ):
        """
        Calculate option Greeks for American options

        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        underlying_price : float
            Current price of the underlying asset
        strike_price : float
            Strike price of the option
        risk_free_rate : float
            Risk-free interest rate (annualized)
        volatility : float
            Implied volatility (annualized)
        dividend_yield : float
            Dividend yield (annualized)
        days_to_expiration : int
            Number of days until option expiration

        Returns:
        --------
        dict : Dictionary containing option Greeks and price
        """
        # Set up dates
        calculation_date = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = calculation_date
        expiration_date = calculation_date + days_to_expiration

        # Option payoff
        option_type_ql = (
            ql.Option.Call if option_type.lower() == "call" else ql.Option.Put
        )
        payoff = ql.PlainVanillaPayoff(option_type_ql, strike_price)
        exercise = ql.AmericanExercise(calculation_date, expiration_date)
        option = ql.VanillaOption(payoff, exercise)

        # Market data
        spot = ql.SimpleQuote(underlying_price)
        spot_handle = ql.QuoteHandle(spot)

        rf_rate = ql.SimpleQuote(risk_free_rate)
        rf_handle = ql.QuoteHandle(rf_rate)

        div_rate = ql.SimpleQuote(dividend_yield)
        div_handle = ql.QuoteHandle(div_rate)

        vol = ql.SimpleQuote(volatility)
        vol_handle = ql.QuoteHandle(vol)

        risk_free_curve = ql.FlatForward(calculation_date, rf_handle, self.day_count)

        dividend_curve = ql.FlatForward(calculation_date, div_handle, self.day_count)

        volatility_curve = ql.BlackConstantVol(
            calculation_date, self.calendar, vol_handle, self.day_count
        )

        # Create handles
        rf_ts = ql.YieldTermStructureHandle(risk_free_curve)
        div_ts = ql.YieldTermStructureHandle(dividend_curve)
        vol_ts = ql.BlackVolTermStructureHandle(volatility_curve)

        # Create process
        bsm_process = ql.BlackScholesMertonProcess(spot_handle, div_ts, rf_ts, vol_ts)

        # American options pricing engine with finite differences
        time_steps = 200
        grid_points = 200
        engine = ql.FdBlackScholesVanillaEngine(bsm_process, time_steps, grid_points)
        option.setPricingEngine(engine)

        # Calculate price and available Greeks
        price = option.NPV()
        delta = option.delta()
        gamma = option.gamma()

        try:
            theta = option.theta() / 365.0  # Daily theta
        except:
            # Calculate theta using finite difference if method not available
            h = 1.0 / 365.0  # One day
            ql.Settings.instance().evaluationDate = calculation_date + 1
            theta = (option.NPV() - price) / h
            ql.Settings.instance().evaluationDate = calculation_date

        # Calculate vega with bump and reprice (since FdEngine might not provide vega)
        vol_bump = 0.01  # 1% volatility bump
        vol.setValue(volatility + vol_bump)
        option_bumped = ql.VanillaOption(payoff, exercise)
        option_bumped.setPricingEngine(engine)
        price_up = option_bumped.NPV()
        vega = (price_up - price) / vol_bump
        vol.setValue(volatility)  # Reset volatility

        # Calculate rho with bump and reprice
        rate_bump = 0.01  # 1% rate bump
        rf_rate.setValue(risk_free_rate + rate_bump)
        option_bumped = ql.VanillaOption(payoff, exercise)
        option_bumped.setPricingEngine(engine)
        price_up = option_bumped.NPV()
        rho = (price_up - price) / rate_bump

        # Return the results
        return {
            "bs_price": round(price, 2),
            "delta": round(delta, 4),
            "gamma": round(gamma, 4),
            "theta": round(theta, 4),
            "vega": round(vega, 4),
            "rho": round(rho, 4),
        }


###################################################################
# DO NOT USE V
class Greeks:
    def __init__(self):
        pass

    def delta(self, S, K, T, r, sigma, option_type):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        # print(
        #     f"S: {S}, K: {K}, T: {T}, r: {r}, sigma: {sigma}, option_type: {option_type}"
        # )
        # print(f"D1: {d1}")

        if option_type == "C" or option_type == "call":
            delta = norm.cdf(d1)
        elif option_type == "P" or option_type == "put":
            delta = norm.cdf(d1) - 1
        else:
            raise ValueError("Invalid option type. Must be 'C' or 'P'.")

        return delta

    def theta(
        self, S, K, T, r, sigma, option_type: str, days_in_year: int = 365
    ) -> float:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        d2 = d1 - sigma * np.sqrt(T)

        # Standard normal probability density function
        N_prime_d1 = norm.pdf(d1)

        if option_type == "C" or option_type == "call":
            theta = -(S * N_prime_d1 * sigma / (2 * np.sqrt(T))) - (
                r * K * np.exp(-r * T) * norm.cdf(d2)
            )
        elif option_type == "P" or option_type == "put":
            theta = -(S * N_prime_d1 * sigma / (2 * np.sqrt(T))) + (
                r * K * np.exp(-r * T) * norm.cdf(-d2)
            )
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        # Convert from annual theta to daily theta
        theta_per_day = theta / days_in_year
        return theta_per_day

    def gamma(self, S, K, T, r, sigma) -> float:
        if T <= 0:
            return 0  # Gamma is zero at expiration
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma

    def vega(self, S, K, T, r, sigma) -> float:
        if T <= 0:
            return 0  # Vega is zero at expiration

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        return vega / 100
