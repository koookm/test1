"""
Technical Indicators Calculator
Computes all indicators used by the Technical and Quantitative agents.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from loguru import logger


class TechnicalIndicators:
    """
    Computes technical analysis indicators from OHLCV data.
    Used by the Technical Agent for fine-grained analysis tasks.
    """

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators for a price DataFrame.

        Args:
            df: DataFrame with columns [Open, High, Low, Close, Volume]

        Returns:
            DataFrame with all indicators appended
        """
        result = df.copy()

        # Moving Averages
        result = TechnicalIndicators.moving_averages(result)

        # Momentum Indicators
        result = TechnicalIndicators.rsi(result)
        result = TechnicalIndicators.macd(result)
        result = TechnicalIndicators.momentum(result)

        # Volatility Indicators
        result = TechnicalIndicators.bollinger_bands(result)
        result = TechnicalIndicators.atr(result)

        # Trend Indicators
        result = TechnicalIndicators.adx(result)

        # Volume Indicators
        result = TechnicalIndicators.volume_analysis(result)

        # Price Patterns
        result = TechnicalIndicators.price_patterns(result)

        return result

    @staticmethod
    def moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Compute SMA and EMA at multiple timeframes."""
        close = df["Close"]

        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f"SMA_{period}"] = close.rolling(window=period, min_periods=1).mean()

        # Exponential Moving Averages
        for period in [12, 26, 50]:
            df[f"EMA_{period}"] = close.ewm(span=period, adjust=False).mean()

        # Golden/Death Cross signals
        df["golden_cross"] = (
            (df["SMA_50"] > df["SMA_200"]) &
            (df["SMA_50"].shift(1) <= df["SMA_200"].shift(1))
        ).astype(int)

        df["death_cross"] = (
            (df["SMA_50"] < df["SMA_200"]) &
            (df["SMA_50"].shift(1) >= df["SMA_200"].shift(1))
        ).astype(int)

        # Price vs Moving Average (trend direction)
        df["price_vs_sma20"] = (close / df["SMA_20"] - 1) * 100
        df["price_vs_sma50"] = (close / df["SMA_50"] - 1) * 100
        df["price_vs_sma200"] = (close / df["SMA_200"] - 1) * 100

        return df

    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Relative Strength Index."""
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI_14"] = 100 - (100 / (1 + rs))

        # RSI signals
        df["rsi_oversold"] = (df["RSI_14"] < 30).astype(int)
        df["rsi_overbought"] = (df["RSI_14"] > 70).astype(int)
        df["rsi_bullish"] = (
            (df["RSI_14"] > 50) &
            (df["RSI_14"].shift(1) <= 50)
        ).astype(int)

        return df

    @staticmethod
    def macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """MACD (Moving Average Convergence/Divergence)."""
        close = df["Close"]
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        df["MACD"] = ema_fast - ema_slow
        df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
        df["MACD_histogram"] = df["MACD"] - df["MACD_signal"]

        # MACD signals
        df["macd_bullish_crossover"] = (
            (df["MACD"] > df["MACD_signal"]) &
            (df["MACD"].shift(1) <= df["MACD_signal"].shift(1))
        ).astype(int)

        df["macd_bearish_crossover"] = (
            (df["MACD"] < df["MACD_signal"]) &
            (df["MACD"].shift(1) >= df["MACD_signal"].shift(1))
        ).astype(int)

        return df

    @staticmethod
    def bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        """Bollinger Bands."""
        close = df["Close"]
        sma = close.rolling(window=period, min_periods=1).mean()
        std_dev = close.rolling(window=period, min_periods=1).std()

        df["BB_upper"] = sma + std * std_dev
        df["BB_lower"] = sma - std * std_dev
        df["BB_middle"] = sma
        df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / sma * 100

        # %B indicator: position within bands
        df["BB_pctB"] = (close - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

        # Bollinger Band signals
        df["bb_squeeze"] = (df["BB_width"] < df["BB_width"].rolling(20).quantile(0.20)).astype(int)
        df["bb_breakout_up"] = ((close > df["BB_upper"]) & (close.shift(1) <= df["BB_upper"].shift(1))).astype(int)
        df["bb_breakout_down"] = ((close < df["BB_lower"]) & (close.shift(1) >= df["BB_lower"].shift(1))).astype(int)

        return df

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average True Range (volatility measure)."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR_14"] = true_range.ewm(span=period, adjust=False).mean()

        # Normalized ATR (as % of price)
        df["ATR_pct"] = df["ATR_14"] / close * 100

        return df

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average Directional Index (trend strength)."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        smoothed_tr = tr.ewm(span=period, adjust=False).mean()
        smoothed_plus_dm = plus_dm.ewm(span=period, adjust=False).mean()
        smoothed_minus_dm = minus_dm.ewm(span=period, adjust=False).mean()

        plus_di = 100 * smoothed_plus_dm / smoothed_tr.replace(0, np.nan)
        minus_di = 100 * smoothed_minus_dm / smoothed_tr.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df["ADX_14"] = dx.ewm(span=period, adjust=False).mean()
        df["DI_plus"] = plus_di
        df["DI_minus"] = minus_di

        # Trend strength signals
        df["strong_trend"] = (df["ADX_14"] > 25).astype(int)
        df["trend_bullish"] = ((df["DI_plus"] > df["DI_minus"]) & (df["ADX_14"] > 25)).astype(int)

        return df

    @staticmethod
    def momentum(df: pd.DataFrame) -> pd.DataFrame:
        """Price momentum at multiple lookback periods."""
        close = df["Close"]

        # Returns over various horizons
        for days in [1, 5, 10, 20, 60, 120, 240]:
            df[f"return_{days}d"] = close.pct_change(days) * 100

        # Momentum signal: 12-month minus 1-month (standard Jegadeesh-Titman)
        df["momentum_jt"] = df.get("return_240d", 0) - df.get("return_20d", 0)

        return df

    @staticmethod
    def volume_analysis(df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based indicators."""
        volume = df["Volume"]
        close = df["Close"]

        # Volume moving averages
        df["vol_SMA_20"] = volume.rolling(window=20, min_periods=1).mean()
        df["vol_ratio"] = volume / df["vol_SMA_20"]

        # On-Balance Volume (OBV)
        price_change = close.diff()
        obv = (
            np.where(price_change > 0, volume, np.where(price_change < 0, -volume, 0))
        )
        df["OBV"] = obv.cumsum() if hasattr(obv, "cumsum") else pd.Series(obv).cumsum()

        # High volume flag
        df["high_volume"] = (df["vol_ratio"] > 2.0).astype(int)

        return df

    @staticmethod
    def price_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Simple price pattern detection."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # 52-week high/low
        df["52w_high"] = high.rolling(window=252, min_periods=1).max()
        df["52w_low"] = low.rolling(window=252, min_periods=1).min()
        df["pct_from_52w_high"] = (close / df["52w_high"] - 1) * 100
        df["pct_from_52w_low"] = (close / df["52w_low"] - 1) * 100

        # Near 52-week high (potential breakout)
        df["near_52w_high"] = (df["pct_from_52w_high"] > -5).astype(int)

        return df

    @staticmethod
    def get_latest_snapshot(df: pd.DataFrame) -> Dict:
        """
        Extract the latest values of all technical indicators as a flat dict.
        Used as input context for the Technical Agent prompt.
        """
        if df.empty:
            return {}

        latest = df.iloc[-1]
        result = {}

        for col in df.columns:
            val = latest.get(col)
            if pd.notna(val):
                result[col] = float(val) if isinstance(val, (int, float, np.number)) else val

        return result


class QuantitativeFactors:
    """
    Computes quantitative/systematic factors for the Quantitative Agent.
    These factors include momentum, volatility, beta, correlation, liquidity.
    """

    @staticmethod
    def compute_factors(
        ticker: str,
        price_data: Dict[str, pd.DataFrame],
        index_data: pd.DataFrame,
        as_of_date: str,
    ) -> Dict:
        """
        Compute all quantitative factors for a given ticker.

        Args:
            ticker: Stock ticker
            price_data: Dict of all price DataFrames
            index_data: Benchmark index price data
            as_of_date: Analysis date (prevents look-ahead bias)

        Returns:
            Dict of factor values
        """
        factors = {"ticker": ticker, "as_of_date": as_of_date}

        if ticker not in price_data:
            return factors

        df = price_data[ticker]
        cutoff = pd.to_datetime(as_of_date)
        df = df[df.index <= cutoff]

        if df.empty or len(df) < 20:
            return factors

        close = df["Close"]
        returns = close.pct_change().dropna()

        # === Momentum Factors ===
        for months, days in [(1, 21), (3, 63), (6, 126), (12, 252)]:
            if len(close) >= days:
                mom = close.iloc[-1] / close.iloc[-days] - 1
                factors[f"momentum_{months}m"] = float(mom)

        # 12-1 Momentum (Jegadeesh-Titman: 12-month minus 1-month)
        if "momentum_12m" in factors and "momentum_1m" in factors:
            factors["momentum_jt"] = factors["momentum_12m"] - factors["momentum_1m"]

        # === Volatility Factors ===
        if len(returns) >= 21:
            factors["vol_1m"] = float(returns.tail(21).std() * np.sqrt(252))
        if len(returns) >= 63:
            factors["vol_3m"] = float(returns.tail(63).std() * np.sqrt(252))
        if len(returns) >= 252:
            factors["vol_12m"] = float(returns.tail(252).std() * np.sqrt(252))

        # === Beta & Correlation with Index ===
        if not index_data.empty:
            index_close = index_data
            if isinstance(index_data, pd.DataFrame):
                if isinstance(index_data.columns, pd.MultiIndex):
                    index_close = index_data["Close"] if "Close" in index_data.columns.get_level_values(0) else index_data.iloc[:, 0]
                elif "Close" in index_data.columns:
                    index_close = index_data["Close"]
                else:
                    index_close = index_data.iloc[:, 0]

            index_close = index_close[index_close.index <= cutoff]
            index_returns = index_close.pct_change().dropna()

            # Align dates
            common_idx = returns.index.intersection(index_returns.index)
            if len(common_idx) >= 30:
                r = returns.loc[common_idx].tail(252)
                ir = index_returns.loc[common_idx].tail(252)

                if len(r) >= 30:
                    cov_matrix = np.cov(r, ir)
                    if cov_matrix[1, 1] != 0:
                        factors["beta"] = float(cov_matrix[0, 1] / cov_matrix[1, 1])
                    factors["correlation_1y"] = float(np.corrcoef(r, ir)[0, 1])

        # === Liquidity Factor ===
        if "Volume" in df.columns:
            vol = df["Volume"]
            avg_vol_20d = vol.tail(20).mean()
            avg_vol_60d = vol.tail(60).mean()
            factors["avg_daily_volume_20d"] = float(avg_vol_20d) if pd.notna(avg_vol_20d) else None
            factors["liquidity_ratio"] = float(avg_vol_20d / avg_vol_60d) if (
                pd.notna(avg_vol_20d) and pd.notna(avg_vol_60d) and avg_vol_60d != 0
            ) else None

        # === Mean Reversion Factor ===
        if len(returns) >= 5:
            # 5-day mean reversion signal
            r5 = returns.tail(5).sum()
            factors["short_term_reversal"] = float(-r5)  # negative of short-term return

        # === Drawdown ===
        if len(close) >= 252:
            peak = close.tail(252).expanding().max()
            drawdown = (close.tail(252) / peak - 1)
            factors["max_drawdown_1y"] = float(drawdown.min())
            factors["current_drawdown"] = float(drawdown.iloc[-1])

        return factors
