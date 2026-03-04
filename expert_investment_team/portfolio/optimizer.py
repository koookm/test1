"""
Portfolio Optimization Module
Implements mean-variance optimization and risk-adjusted portfolio construction.

Per paper arXiv:2602.23330:
"The authors conduct standard portfolio optimization, exploiting low correlation
with the stock index and the variance of each system's output."
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger


class PortfolioOptimizer:
    """
    Portfolio optimization for long-short market-neutral strategy.
    Supports equal-weight and mean-variance optimization.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.001,  # Japan near-zero rate
        target_volatility: float = 0.10,  # 10% annual target vol
    ):
        self.risk_free_rate = risk_free_rate
        self.target_volatility = target_volatility

    def equal_weight_longshort(
        self,
        long_tickers: List[str],
        short_tickers: List[str],
    ) -> Dict[str, float]:
        """
        Construct equal-weighted long-short portfolio.
        This is the baseline approach used in the paper.

        Returns:
            Dict of ticker -> weight (positive=long, negative=short)
        """
        weights = {}

        if long_tickers:
            long_w = 1.0 / len(long_tickers)
            for t in long_tickers:
                weights[t] = long_w

        if short_tickers:
            short_w = -1.0 / len(short_tickers)
            for t in short_tickers:
                # Handle overlap (if same ticker in both long and short, it's flat)
                if t in weights:
                    weights[t] = 0.0  # Cancel out
                    logger.warning(f"Ticker {t} appears in both long and short - set to flat")
                else:
                    weights[t] = short_w

        return weights

    def optimize_sharpe(
        self,
        returns: pd.DataFrame,
        long_tickers: List[str],
        short_tickers: List[str],
        constraints: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Maximize Sharpe ratio subject to long-short constraints.
        Uses mean-variance optimization via scipy.
        """
        try:
            from scipy.optimize import minimize

            all_tickers = list(set(long_tickers + short_tickers))
            common_tickers = [t for t in all_tickers if t in returns.columns]

            if len(common_tickers) < 2:
                logger.warning("Insufficient data for optimization, using equal weight")
                return self.equal_weight_longshort(long_tickers, short_tickers)

            ret_subset = returns[common_tickers].dropna()
            if len(ret_subset) < 30:
                return self.equal_weight_longshort(long_tickers, short_tickers)

            mu = ret_subset.mean().values * 252  # Annualized
            cov = ret_subset.cov().values * 252   # Annualized

            n = len(common_tickers)

            def neg_sharpe(w):
                port_return = np.dot(w, mu)
                port_vol = np.sqrt(w @ cov @ w)
                if port_vol < 1e-10:
                    return 0
                return -(port_return - self.risk_free_rate) / port_vol

            # Constraints
            cons = [{"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 2.0}]  # Gross = 200%
            if not constraints or constraints.get("market_neutral", True):
                cons.append({"type": "eq", "fun": lambda w: np.sum(w)})  # Net = 0

            # Bounds: long tickers positive, short tickers negative
            bounds = []
            for t in common_tickers:
                if t in long_tickers:
                    bounds.append((0.0, 0.20))   # 0-20% long
                elif t in short_tickers:
                    bounds.append((-0.20, 0.0))  # 0-20% short
                else:
                    bounds.append((-0.20, 0.20))

            # Initial weights (equal weight)
            w0 = self.equal_weight_longshort(long_tickers, short_tickers)
            x0 = np.array([w0.get(t, 0.0) for t in common_tickers])

            result = minimize(
                neg_sharpe,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 1000, "ftol": 1e-9},
            )

            if result.success:
                optimized = {t: float(result.x[i]) for i, t in enumerate(common_tickers)}
                logger.info(f"Optimization succeeded: Sharpe ≈ {-result.fun:.3f}")
                return optimized
            else:
                logger.warning(f"Optimization failed: {result.message}, using equal weight")
                return self.equal_weight_longshort(long_tickers, short_tickers)

        except ImportError:
            logger.warning("scipy not available, using equal weight")
            return self.equal_weight_longshort(long_tickers, short_tickers)
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return self.equal_weight_longshort(long_tickers, short_tickers)

    def compute_portfolio_metrics(
        self,
        weights: Dict[str, float],
        returns: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Compute portfolio performance metrics.

        Args:
            weights: Dict of ticker -> weight
            returns: DataFrame of stock returns
            benchmark_returns: Benchmark (TOPIX) return series

        Returns:
            Dict of performance metrics
        """
        # Align weights with available return columns
        valid_tickers = [t for t in weights if t in returns.columns]
        if not valid_tickers:
            return {}

        w = np.array([weights[t] for t in valid_tickers])
        ret_subset = returns[valid_tickers].dropna()

        if ret_subset.empty:
            return {}

        # Portfolio returns (daily)
        port_returns = ret_subset @ w

        # Annualized metrics
        annual_return = port_returns.mean() * 252
        annual_vol = port_returns.std() * np.sqrt(252)
        sharpe = (annual_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0

        # Drawdown
        cum_returns = (1 + port_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns / rolling_max) - 1
        max_drawdown = drawdowns.min()

        metrics = {
            "annual_return": float(annual_return),
            "annual_volatility": float(annual_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "win_rate": float((port_returns > 0).mean()),
            "skewness": float(port_returns.skew()),
            "kurtosis": float(port_returns.kurtosis()),
            "calmar_ratio": float(-annual_return / max_drawdown) if max_drawdown != 0 else 0,
        }

        # Information ratio vs benchmark
        if benchmark_returns is not None:
            common_idx = port_returns.index.intersection(benchmark_returns.index)
            if len(common_idx) > 0:
                excess = port_returns.loc[common_idx] - benchmark_returns.loc[common_idx]
                ir = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
                metrics["information_ratio"] = float(ir)
                metrics["alpha_vs_benchmark"] = float(excess.mean() * 252)

        return metrics
