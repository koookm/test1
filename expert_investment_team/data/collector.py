"""
Data Collection Module
Fetches: stock prices, financial statements, news, macro indicators
for Japanese stocks (TOPIX 100 universe)
"""

import warnings
warnings.filterwarnings("ignore")

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from expert_investment_team.config.settings import SystemConfig, DEFAULT_CONFIG


class StockDataCollector:
    """
    Collects OHLCV price data and technical indicators for TOPIX stocks
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.config = config

    def get_price_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV price data for a list of tickers.
        Returns dict: ticker -> DataFrame(Date, Open, High, Low, Close, Volume)
        """
        price_data = {}
        logger.info(f"Fetching price data for {len(tickers)} tickers...")

        for ticker in tickers:
            try:
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                )
                if df.empty:
                    logger.warning(f"No price data for {ticker}")
                    continue

                df.index = pd.to_datetime(df.index)
                df.index.name = "Date"

                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                price_data[ticker] = df[["Open", "High", "Low", "Close", "Volume"]]
                logger.debug(f"Got {len(df)} rows for {ticker}")

            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")

        logger.info(f"Successfully fetched price data for {len(price_data)} tickers")
        return price_data

    def get_returns_matrix(
        self,
        price_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Build a returns matrix (Date x Ticker) from price data dict.
        """
        closes = {
            ticker: df["Close"]
            for ticker, df in price_data.items()
        }
        prices = pd.DataFrame(closes)
        returns = prices.pct_change().dropna()
        return returns

    def get_topix_index(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch TOPIX index data for market-neutral benchmark."""
        try:
            df = yf.download(
                "^TOPX",
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                # Fallback: use EWJ (iShares MSCI Japan ETF) as proxy
                df = yf.download(
                    "EWJ",
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception as e:
            logger.error(f"Error fetching TOPIX index: {e}")
            return pd.DataFrame()


class FundamentalDataCollector:
    """
    Collects fundamental/financial statement data via yfinance.
    Covers: income statement, balance sheet, cash flow, key ratios.
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.config = config

    def get_financials(self, ticker: str) -> Dict:
        """
        Fetch comprehensive financial data for a given ticker.
        Returns dict with income_stmt, balance_sheet, cash_flow, info.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}

            # Financial statements (quarterly preferred for timeliness)
            income_stmt = stock.quarterly_income_stmt
            balance_sheet = stock.quarterly_balance_sheet
            cash_flow = stock.quarterly_cashflow

            return {
                "ticker": ticker,
                "info": {
                    "company_name": info.get("longName", ticker),
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                    "market_cap": info.get("marketCap"),
                    "enterprise_value": info.get("enterpriseValue"),
                    "pe_ratio": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "pb_ratio": info.get("priceToBook"),
                    "ps_ratio": info.get("priceToSalesTrailing12Months"),
                    "ev_ebitda": info.get("enterpriseToEbitda"),
                    "roe": info.get("returnOnEquity"),
                    "roa": info.get("returnOnAssets"),
                    "gross_margin": info.get("grossMargins"),
                    "operating_margin": info.get("operatingMargins"),
                    "net_margin": info.get("profitMargins"),
                    "revenue_growth": info.get("revenueGrowth"),
                    "earnings_growth": info.get("earningsGrowth"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "current_ratio": info.get("currentRatio"),
                    "quick_ratio": info.get("quickRatio"),
                    "dividend_yield": info.get("dividendYield"),
                    "beta": info.get("beta"),
                    "52w_high": info.get("fiftyTwoWeekHigh"),
                    "52w_low": info.get("fiftyTwoWeekLow"),
                    "avg_volume": info.get("averageVolume"),
                    "float_shares": info.get("floatShares"),
                },
                "income_stmt": income_stmt,
                "balance_sheet": balance_sheet,
                "cash_flow": cash_flow,
            }
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {e}")
            return {"ticker": ticker, "info": {}, "error": str(e)}

    def compute_derived_metrics(self, financials: Dict) -> Dict:
        """
        Compute additional derived financial metrics not available from yfinance.
        These are used by the Fundamental Agent's fine-grained tasks.
        """
        info = financials.get("info", {})
        income = financials.get("income_stmt")
        balance = financials.get("balance_sheet")
        cf = financials.get("cash_flow")

        metrics = {}

        # Quality indicators
        if isinstance(income, pd.DataFrame) and not income.empty:
            # Revenue trend (last 4 quarters)
            if "Total Revenue" in income.index:
                rev_vals = income.loc["Total Revenue"].dropna().sort_index()
                if len(rev_vals) >= 2:
                    metrics["revenue_trend_qoq"] = (
                        rev_vals.iloc[-1] / rev_vals.iloc[-2] - 1
                        if rev_vals.iloc[-2] != 0 else None
                    )
            # Earnings trend
            if "Net Income" in income.index:
                ni_vals = income.loc["Net Income"].dropna().sort_index()
                if len(ni_vals) >= 2:
                    metrics["earnings_trend_qoq"] = (
                        ni_vals.iloc[-1] / ni_vals.iloc[-2] - 1
                        if ni_vals.iloc[-2] != 0 else None
                    )

        # Piotroski F-Score components (fundamental quality scoring)
        metrics["piotroski_roa_positive"] = (
            1 if info.get("returnOnAssets", 0) and info["returnOnAssets"] > 0 else 0
        )
        metrics["piotroski_low_leverage"] = (
            1 if info.get("debtToEquity") and info["debtToEquity"] < 1.0 else 0
        )
        metrics["piotroski_high_current_ratio"] = (
            1 if info.get("currentRatio") and info["currentRatio"] > 1.0 else 0
        )
        metrics["piotroski_positive_gross_margin"] = (
            1 if info.get("grossMargins") and info["grossMargins"] > 0 else 0
        )

        # Valuation score (lower = cheaper)
        pe = info.get("trailingPE")
        pb = info.get("priceToBook")
        if pe and pb and pe > 0 and pb > 0:
            metrics["valuation_composite"] = (pe * pb) ** 0.5  # geometric mean
        else:
            metrics["valuation_composite"] = None

        return metrics


class MacroDataCollector:
    """
    Collects macroeconomic indicators for Japan and US.
    Uses FRED API for US data, and manual/scraping for JP data.
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.config = config
        self.fred_key = config.fred_api_key

    def get_macro_data(self, as_of_date: str) -> Dict:
        """
        Fetch macro indicators available as of the given date.
        Returns 5-dimensional macro assessment aligned with paper.
        """
        macro_data = {
            "as_of_date": as_of_date,
            "jp_indicators": self._get_jp_macro(as_of_date),
            "us_indicators": self._get_us_macro(as_of_date),
        }
        return macro_data

    def _get_us_macro(self, as_of_date: str) -> Dict:
        """Fetch US macro indicators from FRED."""
        us_data = {}

        # Use yfinance proxies for key US indicators
        fred_series = {
            "DGS10": "us_10y_yield",          # 10-Year Treasury
            "FEDFUNDS": "us_fed_funds_rate",   # Federal Funds Rate
            "CPIAUCSL": "us_cpi_yoy",          # CPI
            "UNRATE": "us_unemployment",       # Unemployment Rate
        }

        end_dt = pd.to_datetime(as_of_date)
        start_dt = end_dt - timedelta(days=90)

        for series_id, key in fred_series.items():
            try:
                if self.fred_key:
                    url = (
                        f"https://api.stlouisfed.org/fred/series/observations"
                        f"?series_id={series_id}"
                        f"&api_key={self.fred_key}"
                        f"&file_type=json"
                        f"&observation_start={start_dt.strftime('%Y-%m-%d')}"
                        f"&observation_end={end_dt.strftime('%Y-%m-%d')}"
                    )
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        obs = resp.json().get("observations", [])
                        if obs:
                            latest = obs[-1]
                            val = float(latest["value"]) if latest["value"] != "." else None
                            us_data[key] = {"value": val, "date": latest["date"]}
            except Exception as e:
                logger.debug(f"FRED fetch failed for {series_id}: {e}")
                us_data[key] = {"value": None, "date": None}

        # Fallback: use Yahoo Finance ETF proxies
        if not us_data:
            us_data = self._get_us_macro_proxy(as_of_date)

        return us_data

    def _get_us_macro_proxy(self, as_of_date: str) -> Dict:
        """Use ETF/index proxies when FRED API is unavailable."""
        proxies = {
            "TLT": "us_long_bond",
            "SHY": "us_short_bond",
            "GLD": "gold",
            "VIX": "volatility",
        }
        result = {}
        end_dt = pd.to_datetime(as_of_date)
        start_dt = end_dt - timedelta(days=30)

        for symbol, key in proxies.items():
            try:
                df = yf.download(
                    symbol,
                    start=start_dt.strftime("%Y-%m-%d"),
                    end=end_dt.strftime("%Y-%m-%d"),
                    progress=False,
                    auto_adjust=True,
                )
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    last_price = float(df["Close"].iloc[-1])
                    prev_price = float(df["Close"].iloc[0]) if len(df) > 1 else last_price
                    result[key] = {
                        "value": last_price,
                        "change_pct": (last_price / prev_price - 1) * 100,
                        "date": df.index[-1].strftime("%Y-%m-%d"),
                    }
            except Exception:
                result[key] = {"value": None, "change_pct": None}

        return result

    def _get_jp_macro(self, as_of_date: str) -> Dict:
        """
        Fetch Japanese macro indicators.
        Uses yfinance + ETF proxies for Japan-specific data.
        """
        jp_data = {}
        end_dt = pd.to_datetime(as_of_date)
        start_dt = end_dt - timedelta(days=60)

        # Japan ETF/Index proxies
        jp_proxies = {
            "^N225": "nikkei_225",       # Nikkei 225
            "^NKX": "topix",             # TOPIX (proxy)
            "EWJ": "japan_etf",          # iShares MSCI Japan
            "USDJPY=X": "usd_jpy",       # USD/JPY exchange rate
            "^TNX": "us_10y_yield",      # US 10Y (global reference)
        }

        for symbol, key in jp_proxies.items():
            try:
                df = yf.download(
                    symbol,
                    start=start_dt.strftime("%Y-%m-%d"),
                    end=end_dt.strftime("%Y-%m-%d"),
                    progress=False,
                    auto_adjust=True,
                )
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    last = float(df["Close"].iloc[-1])
                    prev = float(df["Close"].iloc[0]) if len(df) > 1 else last
                    mom_1m = (last / prev - 1) * 100
                    jp_data[key] = {
                        "value": last,
                        "momentum_1m": mom_1m,
                        "date": df.index[-1].strftime("%Y-%m-%d"),
                    }
            except Exception as e:
                logger.debug(f"JP macro proxy {symbol} failed: {e}")
                jp_data[key] = {"value": None, "momentum_1m": None}

        return jp_data


class NewsDataCollector:
    """
    Collects news and sentiment data for stocks and market.
    Uses NewsAPI and yfinance news.
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.config = config
        self.news_api_key = config.news_api_key

    def get_stock_news(
        self,
        ticker: str,
        company_name: str,
        as_of_date: str,
        lookback_days: int = 30,
    ) -> List[Dict]:
        """
        Fetch recent news for a specific stock.
        Returns list of news articles with title, summary, sentiment indicators.
        """
        news_articles = []

        # Method 1: yfinance news
        try:
            stock = yf.Ticker(ticker)
            yf_news = stock.news
            if yf_news:
                for article in yf_news[:10]:
                    news_articles.append({
                        "source": "yfinance",
                        "title": article.get("title", ""),
                        "summary": article.get("summary", ""),
                        "url": article.get("link", ""),
                        "published": article.get("providerPublishTime", ""),
                        "publisher": article.get("publisher", ""),
                    })
        except Exception as e:
            logger.debug(f"yfinance news failed for {ticker}: {e}")

        # Method 2: NewsAPI (if key available)
        if self.news_api_key and not news_articles:
            try:
                end_dt = pd.to_datetime(as_of_date)
                start_dt = end_dt - timedelta(days=lookback_days)
                url = (
                    f"https://newsapi.org/v2/everything"
                    f"?q={company_name}"
                    f"&from={start_dt.strftime('%Y-%m-%d')}"
                    f"&to={end_dt.strftime('%Y-%m-%d')}"
                    f"&language=en"
                    f"&sortBy=relevancy"
                    f"&pageSize=10"
                    f"&apiKey={self.news_api_key}"
                )
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    articles = resp.json().get("articles", [])
                    for article in articles:
                        news_articles.append({
                            "source": "newsapi",
                            "title": article.get("title", ""),
                            "summary": article.get("description", ""),
                            "url": article.get("url", ""),
                            "published": article.get("publishedAt", ""),
                            "publisher": article.get("source", {}).get("name", ""),
                        })
            except Exception as e:
                logger.debug(f"NewsAPI failed for {company_name}: {e}")

        return news_articles

    def get_market_news(self, as_of_date: str) -> List[Dict]:
        """Fetch general Japanese market news."""
        market_news = []

        # Fetch Japan market ETF news as proxy
        jp_tickers = ["EWJ", "DXJ"]
        for ticker in jp_tickers:
            try:
                stock = yf.Ticker(ticker)
                news = stock.news
                if news:
                    for article in news[:5]:
                        market_news.append({
                            "source": "yfinance",
                            "title": article.get("title", ""),
                            "summary": article.get("summary", ""),
                            "published": article.get("providerPublishTime", ""),
                        })
            except Exception:
                pass

        return market_news
