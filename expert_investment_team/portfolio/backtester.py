"""
Backtesting Engine
Implements leakage-controlled backtesting as described in arXiv:2602.23330.

Key features:
- Point-in-time data (no look-ahead bias)
- Monthly rebalancing
- Transaction cost modeling
- TOPIX 100 universe
- Backtesting period: Sep 2023 - Nov 2025
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

from expert_investment_team.config.settings import SystemConfig, DEFAULT_CONFIG
from expert_investment_team.portfolio.optimizer import PortfolioOptimizer


class Backtester:
    """
    Leakage-controlled backtesting engine for the multi-agent investment system.

    Implements monthly rebalancing with:
    - Strict point-in-time data access (no future data)
    - Transaction cost modeling
    - Performance tracking
    """

    def __init__(
        self,
        config: SystemConfig = DEFAULT_CONFIG,
        output_dir: str = "results/backtest",
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = PortfolioOptimizer(
            risk_free_rate=0.001,  # Japan near-zero rate
        )

    def run_backtest(
        self,
        price_data: Dict[str, pd.DataFrame],
        index_data: pd.DataFrame,
        analysis_function: Callable,
        start_date: str = None,
        end_date: str = None,
        transaction_cost_bps: float = None,
    ) -> Dict:
        """
        Run the full backtesting pipeline.

        Args:
            price_data: Dict of ticker -> OHLCV DataFrame
            index_data: Benchmark index OHLCV data
            analysis_function: Callable(tickers, as_of_date) -> Dict[ticker, score]
            start_date: Backtest start (default from config)
            end_date: Backtest end (default from config)
            transaction_cost_bps: Transaction cost in basis points

        Returns:
            Dict with full backtest results and performance metrics
        """
        start_date = start_date or self.config.data.backtest_start
        end_date = end_date or self.config.data.backtest_end
        tc_bps = transaction_cost_bps or self.config.portfolio.transaction_cost_bps
        tc_rate = tc_bps / 10000

        logger.info(f"Starting backtest: {start_date} to {end_date}")

        # Build monthly rebalancing dates
        rebal_dates = self._get_monthly_rebalance_dates(start_date, end_date)
        logger.info(f"Rebalancing dates: {len(rebal_dates)}")

        # Build returns matrix
        returns_matrix = self._build_returns_matrix(price_data)

        # Get index returns
        index_returns = self._get_index_returns(index_data)

        # Initialize tracking
        portfolio_returns = []
        portfolio_history = []
        intermediate_outputs = []
        current_weights = {}

        for i, rebal_date in enumerate(rebal_dates):
            logger.info(f"Rebalancing {i+1}/{len(rebal_dates)}: {rebal_date}")

            # Get analysis scores (point-in-time, no look-ahead)
            tickers = list(price_data.keys())
            scores = analysis_function(tickers, rebal_date)

            if not scores:
                logger.warning(f"No scores generated for {rebal_date}, skipping")
                continue

            # Select long/short positions based on scores
            new_weights = self._scores_to_weights(scores)

            # Compute period returns (from this rebal to next rebal)
            next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else pd.Timestamp(end_date)
            period_returns = self._compute_period_return(
                new_weights, returns_matrix, rebal_date, next_date, tc_rate, current_weights
            )

            portfolio_returns.append({
                "date": rebal_date,
                "return": period_returns["net_return"],
                "gross_return": period_returns["gross_return"],
                "tc_cost": period_returns["tc_cost"],
                "n_long": period_returns["n_long"],
                "n_short": period_returns["n_short"],
            })

            portfolio_history.append({
                "date": rebal_date,
                "long_positions": [t for t, w in new_weights.items() if w > 0],
                "short_positions": [t for t, w in new_weights.items() if w < 0],
                "scores": scores,
            })

            # Save intermediate outputs for explainability analysis
            intermediate_outputs.append({
                "date": str(rebal_date),
                "scores": {k: float(v) for k, v in scores.items()},
                "portfolio": {
                    "long": [t for t, w in new_weights.items() if w > 0],
                    "short": [t for t, w in new_weights.items() if w < 0],
                },
            })

            current_weights = new_weights

        # Compute aggregate performance metrics
        returns_series = pd.Series(
            [r["return"] for r in portfolio_returns],
            index=[r["date"] for r in portfolio_returns],
        )

        metrics = self.optimizer.compute_portfolio_metrics(
            weights={},  # Already applied in period returns
            returns=pd.DataFrame(),
            benchmark_returns=None,
        )

        # Compute metrics directly from monthly returns
        metrics = self._compute_monthly_metrics(returns_series, index_returns)

        # Save results
        results = {
            "config": {
                "start_date": start_date,
                "end_date": end_date,
                "n_long": self.config.portfolio.n_long,
                "n_short": self.config.portfolio.n_short,
                "transaction_cost_bps": tc_bps,
                "use_fine_grained": self.config.agent.use_fine_grained_tasks,
            },
            "performance": metrics,
            "monthly_returns": returns_series.to_dict(),
            "portfolio_history": portfolio_history,
            "intermediate_outputs": intermediate_outputs,
        }

        # Save to disk
        self._save_results(results)

        logger.info(
            f"Backtest complete. Sharpe: {metrics.get('sharpe_ratio', 0):.3f}, "
            f"Annual Return: {metrics.get('annual_return', 0):.1%}"
        )

        return results

    def _get_monthly_rebalance_dates(
        self, start_date: str, end_date: str
    ) -> List[pd.Timestamp]:
        """Generate end-of-month rebalancing dates."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        dates = pd.date_range(start=start, end=end, freq="ME")  # Month-End
        return [d for d in dates if d >= start and d <= end]

    def _build_returns_matrix(
        self, price_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Build daily returns matrix from price data."""
        closes = {}
        for ticker, df in price_data.items():
            if "Close" in df.columns and not df.empty:
                closes[ticker] = df["Close"]

        if not closes:
            return pd.DataFrame()

        prices = pd.DataFrame(closes)
        returns = prices.pct_change()
        return returns

    def _get_index_returns(self, index_data: pd.DataFrame) -> pd.Series:
        """Extract index return series."""
        if index_data is None or index_data.empty:
            return pd.Series(dtype=float)

        if isinstance(index_data.columns, pd.MultiIndex):
            close = index_data["Close"] if "Close" in index_data.columns.get_level_values(0) else index_data.iloc[:, 0]
        elif "Close" in index_data.columns:
            close = index_data["Close"]
        else:
            close = index_data.iloc[:, 0]

        # Handle Series vs DataFrame
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        return close.pct_change().dropna()

    def _scores_to_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Convert attractiveness scores to portfolio weights."""
        n_long = self.config.portfolio.n_long
        n_short = self.config.portfolio.n_short

        # Sort by score
        sorted_tickers = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        weights = {}

        # Top n_long = long positions
        long_tickers = [t for t, s in sorted_tickers[:n_long]]
        for t in long_tickers:
            weights[t] = 1.0 / n_long

        # Bottom n_short = short positions
        short_tickers = [t for t, s in sorted_tickers[-n_short:]]
        for t in short_tickers:
            if t not in weights:  # Avoid overlap
                weights[t] = -1.0 / n_short

        return weights

    def _compute_period_return(
        self,
        new_weights: Dict[str, float],
        returns_matrix: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        tc_rate: float,
        prev_weights: Dict[str, float],
    ) -> Dict:
        """Compute portfolio return for a holding period."""
        # Get daily returns for the period
        period_mask = (returns_matrix.index > start_date) & (returns_matrix.index <= end_date)
        period_returns = returns_matrix[period_mask]

        if period_returns.empty:
            return {"net_return": 0.0, "gross_return": 0.0, "tc_cost": 0.0, "n_long": 0, "n_short": 0}

        # Compute portfolio daily returns
        portfolio_daily = pd.Series(0.0, index=period_returns.index)

        for ticker, weight in new_weights.items():
            if ticker in period_returns.columns:
                ticker_returns = period_returns[ticker].fillna(0)
                portfolio_daily += weight * ticker_returns

        # Period gross return (compound daily returns)
        gross_return = float((1 + portfolio_daily).prod() - 1)

        # Transaction costs (proportional to turnover)
        turnover = sum(
            abs(new_weights.get(t, 0) - prev_weights.get(t, 0))
            for t in set(list(new_weights.keys()) + list(prev_weights.keys()))
        )
        tc_cost = turnover * tc_rate
        net_return = gross_return - tc_cost

        n_long = sum(1 for w in new_weights.values() if w > 0)
        n_short = sum(1 for w in new_weights.values() if w < 0)

        return {
            "gross_return": gross_return,
            "net_return": net_return,
            "tc_cost": tc_cost,
            "n_long": n_long,
            "n_short": n_short,
        }

    def _compute_monthly_metrics(
        self,
        monthly_returns: pd.Series,
        index_returns: pd.Series,
    ) -> Dict:
        """Compute performance metrics from monthly return series."""
        if monthly_returns.empty:
            return {}

        n_months = len(monthly_returns)
        n_years = n_months / 12

        # Annualized return
        total_return = float((1 + monthly_returns).prod() - 1)
        annual_return = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0

        # Sharpe ratio (monthly, annualized)
        monthly_rf = self.config.portfolio.transaction_cost_bps / 12 / 10000
        excess_returns = monthly_returns - monthly_rf
        sharpe = float(
            excess_returns.mean() / excess_returns.std() * np.sqrt(12)
        ) if excess_returns.std() > 0 else 0

        # Drawdown
        cum = (1 + monthly_returns).cumprod()
        rolling_max = cum.expanding().max()
        drawdowns = cum / rolling_max - 1
        max_drawdown = float(drawdowns.min())

        metrics = {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": float(monthly_returns.std() * np.sqrt(12)),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "calmar_ratio": float(-annual_return / max_drawdown) if max_drawdown != 0 else 0,
            "win_rate_monthly": float((monthly_returns > 0).mean()),
            "n_months": n_months,
            "avg_monthly_return": float(monthly_returns.mean()),
            "best_month": float(monthly_returns.max()),
            "worst_month": float(monthly_returns.min()),
        }

        # Market-neutral alpha (vs index monthly returns)
        if not index_returns.empty:
            # Aggregate index to monthly returns
            try:
                monthly_index = index_returns.resample("ME").apply(
                    lambda r: (1 + r).prod() - 1
                )
                common = monthly_returns.index.intersection(monthly_index.index)
                if len(common) >= 3:
                    alpha_series = monthly_returns.loc[common] - monthly_index.loc[common]
                    metrics["information_ratio"] = float(
                        alpha_series.mean() / alpha_series.std() * np.sqrt(12)
                        if alpha_series.std() > 0 else 0
                    )
                    metrics["annual_alpha"] = float(alpha_series.mean() * 12)
                    metrics["correlation_with_market"] = float(
                        monthly_returns.loc[common].corr(monthly_index.loc[common])
                    )
            except Exception as e:
                logger.debug(f"Alpha computation failed: {e}")

        return metrics

    def _save_results(self, results: Dict) -> None:
        """Save backtest results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "fine_grained" if results["config"]["use_fine_grained"] else "coarse_grained"
        filename = self.output_dir / f"backtest_{mode}_{timestamp}.json"

        # Convert non-serializable objects
        def convert(obj):
            if isinstance(obj, (pd.Timestamp, datetime)):
                return str(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Not serializable: {type(obj)}")

        try:
            with open(filename, "w") as f:
                json.dump(results, f, indent=2, default=convert)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
