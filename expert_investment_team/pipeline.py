"""
Main Investment Analysis Pipeline
Orchestrates all agents to produce investment scores and portfolio decisions.

Based on arXiv:2602.23330: "Toward Expert Investment Teams:
A Multi-Agent LLM System with Fine-Grained Trading Tasks"

Pipeline flow:
1. Data Collection (prices, financials, news, macro)
2. Technical Indicators Computation
3. Quantitative Factor Computation
4. Parallel Agent Analysis (Fundamental, Technical, Quantitative, Sentiment)
5. Sector Agent Synthesis (per stock)
6. Macro Agent Analysis (market-level)
7. Portfolio Manager Score Generation
8. Portfolio Construction
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from loguru import logger

from expert_investment_team.config.settings import (
    SystemConfig, DEFAULT_CONFIG, TOPIX100_TICKERS
)
from expert_investment_team.data.collector import (
    StockDataCollector, FundamentalDataCollector,
    MacroDataCollector, NewsDataCollector
)
from expert_investment_team.data.technical_indicators import (
    TechnicalIndicators, QuantitativeFactors
)
from expert_investment_team.agents.fundamental_agent import FundamentalAgent
from expert_investment_team.agents.technical_agent import TechnicalAgent
from expert_investment_team.agents.quantitative_agent import QuantitativeAgent
from expert_investment_team.agents.sentiment_agent import SentimentAgent
from expert_investment_team.agents.sector_agent import SectorAgent
from expert_investment_team.agents.macro_agent import MacroAgent
from expert_investment_team.agents.portfolio_manager_agent import PortfolioManagerAgent
from expert_investment_team.utils.sector_stats import compute_sector_stats, get_peer_stats
from expert_investment_team.metrics.performance import PerformanceMetrics


class InvestmentPipeline:
    """
    Orchestrates the complete multi-agent investment analysis pipeline.
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.config = config

        # Initialize data collectors
        self.price_collector = StockDataCollector(config)
        self.fundamental_collector = FundamentalDataCollector(config)
        self.macro_collector = MacroDataCollector(config)
        self.news_collector = NewsDataCollector(config)

        # Initialize agents
        self.fundamental_agent = FundamentalAgent(config)
        self.technical_agent = TechnicalAgent(config)
        self.quantitative_agent = QuantitativeAgent(config)
        self.sentiment_agent = SentimentAgent(config)
        self.sector_agent = SectorAgent(config)
        self.macro_agent = MacroAgent(config)
        self.pm_agent = PortfolioManagerAgent(config)

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_universe(
        self,
        tickers: List[str],
        as_of_date: str,
        price_data: Optional[Dict] = None,
        save_outputs: bool = True,
    ) -> Dict[str, float]:
        """
        Run complete analysis pipeline for a universe of stocks.

        Args:
            tickers: List of stock tickers to analyze
            as_of_date: Analysis date (point-in-time, prevents look-ahead bias)
            price_data: Pre-loaded price data (optional, for backtesting efficiency)
            save_outputs: Whether to save intermediate outputs

        Returns:
            Dict of ticker -> final attractiveness score (0-100)
        """
        logger.info(f"Starting analysis for {len(tickers)} stocks as of {as_of_date}")

        # === Step 1: Data Collection ===
        logger.info("Step 1: Collecting data...")

        if price_data is None:
            # Fetch price data with lookback
            start_dt = pd.Timestamp(as_of_date) - pd.Timedelta(
                days=self.config.data.price_lookback_days * 1.5
            )
            price_data = self.price_collector.get_price_data(
                tickers,
                start_date=start_dt.strftime("%Y-%m-%d"),
                end_date=as_of_date,
            )

        # Fetch fundamentals and news (can be parallelized)
        fundamentals_data = self._collect_fundamentals(tickers)
        macro_data = self.macro_collector.get_macro_data(as_of_date)
        news_data = self._collect_news(tickers, fundamentals_data, as_of_date)

        # === Step 2: Compute Technical Indicators ===
        logger.info("Step 2: Computing technical indicators...")
        indicator_data = {}
        for ticker in tickers:
            if ticker in price_data and not price_data[ticker].empty:
                df = price_data[ticker][price_data[ticker].index <= pd.Timestamp(as_of_date)]
                if not df.empty:
                    with_indicators = TechnicalIndicators.compute_all(df)
                    indicator_data[ticker] = TechnicalIndicators.get_latest_snapshot(with_indicators)

        # === Step 3: Compute Quantitative Factors ===
        logger.info("Step 3: Computing quantitative factors...")
        index_data = self.price_collector.get_topix_index(
            start_date=(pd.Timestamp(as_of_date) - pd.Timedelta(days=400)).strftime("%Y-%m-%d"),
            end_date=as_of_date,
        )

        quant_factors = {}
        for ticker in tickers:
            quant_factors[ticker] = QuantitativeFactors.compute_factors(
                ticker=ticker,
                price_data=price_data,
                index_data=index_data,
                as_of_date=as_of_date,
            )

        # === Step 4: Compute Sector Stats ===
        sector_stats_all = compute_sector_stats(fundamentals_data)

        # === Step 5: Macro Agent Analysis ===
        logger.info("Step 5: Running Macro Agent...")
        macro_output = self.macro_agent.analyze({
            "macro_data": macro_data,
            "as_of_date": as_of_date,
        })

        # === Step 6: Per-Stock Analysis (Parallel) ===
        logger.info("Step 6: Running per-stock analysis (4 agents per stock)...")
        sector_outputs = {}

        with ThreadPoolExecutor(max_workers=min(4, len(tickers))) as executor:
            futures = {}
            for ticker in tickers:
                future = executor.submit(
                    self._analyze_single_stock,
                    ticker=ticker,
                    as_of_date=as_of_date,
                    fundamentals=fundamentals_data.get(ticker, {}),
                    indicators=indicator_data.get(ticker, {}),
                    quant_factors=quant_factors.get(ticker, {}),
                    universe_factors=quant_factors,
                    news=news_data.get(ticker, []),
                    sector_stats=sector_stats_all,
                )
                futures[future] = ticker

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    sector_outputs[ticker] = result
                    logger.info(f"Completed {ticker}: sector score = {result.get('score', 50):.1f}")
                except Exception as e:
                    logger.error(f"Analysis failed for {ticker}: {e}")
                    sector_outputs[ticker] = {"score": 50.0, "ticker": ticker}

        # === Step 7: Portfolio Manager Scoring ===
        logger.info("Step 7: Portfolio Manager generating final scores...")
        pm_decisions = self.pm_agent.generate_scores(sector_outputs, macro_output)

        # Extract final scores
        final_scores = {
            ticker: data.get("final_score", 50.0)
            for ticker, data in pm_decisions.items()
        }

        # === Step 8: Save Intermediate Outputs ===
        if save_outputs and self.config.save_intermediate_outputs:
            self._save_pipeline_outputs(
                as_of_date=as_of_date,
                macro_output=macro_output,
                sector_outputs=sector_outputs,
                pm_decisions=pm_decisions,
                final_scores=final_scores,
            )

        logger.info(f"Analysis complete. Generated scores for {len(final_scores)} stocks.")
        return final_scores

    def _analyze_single_stock(
        self,
        ticker: str,
        as_of_date: str,
        fundamentals: Dict,
        indicators: Dict,
        quant_factors: Dict,
        universe_factors: Dict,
        news: List[Dict],
        sector_stats: Dict,
    ) -> Dict:
        """Run all 4 analyst agents + sector agent for a single stock."""
        # Get company info
        info = fundamentals.get("info", {})
        company_name = info.get("company_name", ticker)
        sector = info.get("sector", "Unknown")

        # === Fundamental Analysis ===
        fundamental_output = self.fundamental_agent.analyze({
            "ticker": ticker,
            "financials": {
                **fundamentals,
                "derived_metrics": self.fundamental_collector.compute_derived_metrics(fundamentals),
            },
            "sector_peers": get_peer_stats(ticker, sector, sector_stats),
        })

        # === Technical Analysis ===
        technical_output = self.technical_agent.analyze({
            "ticker": ticker,
            "indicators": indicators,
        })

        # === Quantitative Analysis ===
        quantitative_output = self.quantitative_agent.analyze({
            "ticker": ticker,
            "factors": quant_factors,
            "universe_factors": universe_factors,
        })

        # === Sentiment Analysis ===
        sentiment_output = self.sentiment_agent.analyze({
            "ticker": ticker,
            "company_name": company_name,
            "news_articles": news,
        })

        # === Sector Synthesis ===
        sector_output = self.sector_agent.analyze({
            "ticker": ticker,
            "company_name": company_name,
            "sector": sector,
            "analyst_outputs": {
                "fundamental": fundamental_output,
                "technical": technical_output,
                "quantitative": quantitative_output,
                "sentiment": sentiment_output,
            },
            "sector_stats": get_peer_stats(ticker, sector, sector_stats),
        })

        return sector_output

    def _collect_fundamentals(self, tickers: List[str]) -> Dict[str, Dict]:
        """Collect fundamental data for all tickers."""
        fundamentals = {}
        for ticker in tickers:
            try:
                fundamentals[ticker] = self.fundamental_collector.get_financials(ticker)
            except Exception as e:
                logger.warning(f"Fundamentals collection failed for {ticker}: {e}")
                fundamentals[ticker] = {"ticker": ticker, "info": {}}
        return fundamentals

    def _collect_news(
        self, tickers: List[str], fundamentals: Dict, as_of_date: str
    ) -> Dict[str, List[Dict]]:
        """Collect news for all tickers."""
        news_data = {}
        for ticker in tickers:
            try:
                company_name = fundamentals.get(ticker, {}).get("info", {}).get("company_name", ticker)
                news_data[ticker] = self.news_collector.get_stock_news(
                    ticker=ticker,
                    company_name=company_name,
                    as_of_date=as_of_date,
                    lookback_days=self.config.data.news_lookback_days,
                )
            except Exception as e:
                logger.debug(f"News collection failed for {ticker}: {e}")
                news_data[ticker] = []
        return news_data

    def _save_pipeline_outputs(
        self,
        as_of_date: str,
        macro_output: Dict,
        sector_outputs: Dict,
        pm_decisions: Dict,
        final_scores: Dict,
    ) -> None:
        """Save all pipeline outputs for explainability analysis."""
        output_file = self.output_dir / f"pipeline_output_{as_of_date}.json"

        output = {
            "as_of_date": as_of_date,
            "macro": {
                "overall_score": macro_output.get("overall_score"),
                "environment": macro_output.get("macro_environment"),
                "dimensions": {
                    k: {"score": v.get("score"), "reasoning": v.get("reasoning")}
                    for k, v in macro_output.get("dimension_scores", {}).items()
                },
            },
            "stock_outputs": {},
            "final_scores": final_scores,
        }

        for ticker in final_scores:
            sector_out = sector_outputs.get(ticker, {})
            pm_out = pm_decisions.get(ticker, {})
            output["stock_outputs"][ticker] = {
                "sector_score": sector_out.get("score"),
                "final_score": pm_out.get("final_score"),
                "signal": pm_out.get("long_short_signal"),
                "thesis": pm_out.get("investment_thesis"),
                "conviction": pm_out.get("conviction_level"),
            }

        def convert(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            raise TypeError

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False, default=convert)
            logger.debug(f"Pipeline output saved: {output_file}")
        except Exception as e:
            logger.warning(f"Failed to save pipeline output: {e}")

    def run_live_analysis(
        self,
        tickers: Optional[List[str]] = None,
        as_of_date: Optional[str] = None,
    ) -> Dict:
        """
        Run live analysis for the current date.

        Args:
            tickers: Stock universe (default: TOPIX100 representative set)
            as_of_date: Analysis date (default: today)

        Returns:
            Dict with portfolio recommendations
        """
        tickers = tickers or TOPIX100_TICKERS
        as_of_date = as_of_date or datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Running live analysis: {len(tickers)} stocks, date={as_of_date}")

        # Run full analysis
        scores = self.analyze_universe(tickers, as_of_date)

        # Construct portfolio
        portfolio = self.pm_agent.construct_portfolio(
            pm_decisions={t: {"final_score": s} for t, s in scores.items()},
        )

        # Format recommendation report
        report = self._format_report(scores, portfolio, as_of_date)

        return {
            "scores": scores,
            "portfolio": portfolio,
            "report": report,
            "as_of_date": as_of_date,
        }

    def _format_report(
        self, scores: Dict, portfolio: Dict, as_of_date: str
    ) -> str:
        """Format investment report."""
        lines = [
            "=" * 70,
            f"EXPERT INVESTMENT TEAM - PORTFOLIO REPORT",
            f"Analysis Date: {as_of_date}",
            f"Strategy: Long-Short Market Neutral (TOPIX 100)",
            "=" * 70,
            "",
            "LONG POSITIONS (Top Picks):",
            "-" * 40,
        ]

        for pos in portfolio.get("long_positions", []):
            lines.append(
                f"  {pos['ticker']:15} Score: {pos['score']:.1f}/100  "
                f"Conviction: {pos.get('conviction', 'N/A')}"
            )
            if pos.get("thesis"):
                lines.append(f"  Thesis: {pos['thesis'][:80]}...")

        lines.extend([
            "",
            "SHORT POSITIONS (Avoid/Short):",
            "-" * 40,
        ])

        for pos in portfolio.get("short_positions", []):
            lines.append(
                f"  {pos['ticker']:15} Score: {pos['score']:.1f}/100  "
                f"Conviction: {pos.get('conviction', 'N/A')}"
            )

        meta = portfolio.get("metadata", {})
        lines.extend([
            "",
            "=" * 70,
            "PORTFOLIO SUMMARY:",
            f"  Long Positions:    {meta.get('n_long', 0)}",
            f"  Short Positions:   {meta.get('n_short', 0)}",
            f"  Net Exposure:      {meta.get('net_exposure', 0):.0%} (Market Neutral)",
            f"  Gross Exposure:    {meta.get('total_gross_exposure', 0):.0%}",
            f"  Score Spread:      {meta.get('score_spread', 0):.1f} points",
            "=" * 70,
        ])

        return "\n".join(lines)
