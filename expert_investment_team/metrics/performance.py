"""
Performance Metrics Module
Comprehensive performance measurement for investment strategies.
Implements all metrics from the paper arXiv:2602.23330.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from loguru import logger


class PerformanceMetrics:
    """
    Compute comprehensive performance metrics for investment strategies.

    Key metric: Sharpe ratio (primary metric in the paper)
    "Performance is assessed using the standard measure of risk-adjusted return,
    namely the Sharpe ratio, calculated as the mean of monthly portfolio returns
    divided by their standard deviation."
    """

    def __init__(self, risk_free_rate_annual: float = 0.001):
        """
        Args:
            risk_free_rate_annual: Annual risk-free rate (Japan near-zero)
        """
        self.rf_annual = risk_free_rate_annual
        self.rf_monthly = (1 + risk_free_rate_annual) ** (1/12) - 1

    def compute_all(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        frequency: str = "monthly",
    ) -> Dict:
        """
        Compute all performance metrics.

        Args:
            returns: Return series (monthly or daily)
            benchmark_returns: Benchmark return series for relative metrics
            frequency: "daily" or "monthly"

        Returns:
            Dict of all performance metrics
        """
        annualization_factor = 12 if frequency == "monthly" else 252
        rf = self.rf_monthly if frequency == "monthly" else self.rf_annual / 252

        metrics = {}

        # === Core Return Metrics ===
        metrics.update(self._core_returns(returns, annualization_factor, rf))

        # === Risk Metrics ===
        metrics.update(self._risk_metrics(returns, annualization_factor))

        # === Drawdown Metrics ===
        metrics.update(self._drawdown_metrics(returns))

        # === Distribution Metrics ===
        metrics.update(self._distribution_metrics(returns))

        # === Relative Metrics (vs benchmark) ===
        if benchmark_returns is not None:
            metrics.update(
                self._relative_metrics(returns, benchmark_returns, annualization_factor, rf)
            )

        return metrics

    def _core_returns(
        self, returns: pd.Series, ann_factor: int, rf: float
    ) -> Dict:
        """Core return and Sharpe ratio metrics."""
        n = len(returns)
        if n == 0:
            return {}

        # Annualized return (geometric)
        total_return = float((1 + returns).prod() - 1)
        n_years = n / ann_factor
        annual_return = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0

        # Volatility
        annual_vol = float(returns.std() * np.sqrt(ann_factor))

        # Sharpe Ratio (as defined in paper: mean / std of monthly returns)
        # Paper: "calculated as the mean of monthly portfolio returns divided by std"
        excess_returns = returns - rf
        sharpe_paper = float(returns.mean() / returns.std()) if returns.std() > 0 else 0
        sharpe_standard = float(
            excess_returns.mean() / excess_returns.std() * np.sqrt(ann_factor)
        ) if excess_returns.std() > 0 else 0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe_standard,
            "sharpe_paper_definition": sharpe_paper,  # Paper's exact definition
            "avg_monthly_return": float(returns.mean()),
            "monthly_volatility": float(returns.std()),
        }

    def _risk_metrics(self, returns: pd.Series, ann_factor: int) -> Dict:
        """Risk-specific metrics."""
        # Downside deviation (Sortino denominator)
        negative_returns = returns[returns < 0]
        downside_dev = float(negative_returns.std() * np.sqrt(ann_factor)) if len(negative_returns) > 0 else 0

        # Value at Risk (95%, 99%)
        var_95 = float(np.percentile(returns, 5))
        var_99 = float(np.percentile(returns, 1))

        # Expected Shortfall (CVaR)
        cvar_95 = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else var_95

        # Sortino Ratio
        rf = 0
        excess_mean = returns.mean() - rf
        sortino = float(excess_mean / downside_dev * np.sqrt(ann_factor)) if downside_dev > 0 else 0

        return {
            "downside_deviation": downside_dev,
            "sortino_ratio": sortino,
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
        }

    def _drawdown_metrics(self, returns: pd.Series) -> Dict:
        """Drawdown analysis."""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns / rolling_max) - 1

        max_dd = float(drawdowns.min())
        current_dd = float(drawdowns.iloc[-1])

        # Calmar ratio
        annual_return = float((1 + (cum_returns.iloc[-1] - 1)) ** (12 / len(returns)) - 1)
        calmar = float(-annual_return / max_dd) if max_dd != 0 else 0

        # Recovery analysis
        in_drawdown = drawdowns < 0
        dd_periods = []
        start = None
        for i, (idx, val) in enumerate(drawdowns.items()):
            if val < 0 and start is None:
                start = idx
            elif val >= 0 and start is not None:
                dd_periods.append(i - drawdowns.index.get_loc(start))
                start = None

        avg_recovery = np.mean(dd_periods) if dd_periods else 0

        return {
            "max_drawdown": max_dd,
            "current_drawdown": current_dd,
            "calmar_ratio": calmar,
            "avg_recovery_periods": float(avg_recovery),
            "time_in_drawdown_pct": float(in_drawdown.mean()),
        }

    def _distribution_metrics(self, returns: pd.Series) -> Dict:
        """Return distribution statistics."""
        return {
            "skewness": float(returns.skew()),
            "kurtosis": float(returns.kurtosis()),
            "positive_months": int((returns > 0).sum()),
            "negative_months": int((returns < 0).sum()),
            "win_rate": float((returns > 0).mean()),
            "avg_win": float(returns[returns > 0].mean()) if (returns > 0).any() else 0,
            "avg_loss": float(returns[returns < 0].mean()) if (returns < 0).any() else 0,
            "best_period": float(returns.max()),
            "worst_period": float(returns.min()),
            "profit_factor": float(
                abs(returns[returns > 0].sum() / returns[returns < 0].sum())
                if (returns < 0).any() and returns[returns < 0].sum() != 0 else float("inf")
            ),
        }

    def _relative_metrics(
        self,
        returns: pd.Series,
        benchmark: pd.Series,
        ann_factor: int,
        rf: float,
    ) -> Dict:
        """Metrics relative to benchmark (TOPIX)."""
        common_idx = returns.index.intersection(benchmark.index)
        if len(common_idx) < 3:
            return {}

        r = returns.loc[common_idx]
        b = benchmark.loc[common_idx]
        active = r - b

        # Information Ratio
        ir = float(
            active.mean() / active.std() * np.sqrt(ann_factor)
        ) if active.std() > 0 else 0

        # Beta
        cov = np.cov(r, b)
        beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 1.0

        # Alpha (Jensen's)
        alpha = float((r.mean() - rf - beta * (b.mean() - rf)) * ann_factor)

        # Correlation
        corr = float(r.corr(b))

        # Tracking Error
        tracking_error = float(active.std() * np.sqrt(ann_factor))

        return {
            "information_ratio": ir,
            "alpha_annualized": alpha,
            "beta": beta,
            "correlation_with_benchmark": corr,
            "tracking_error": tracking_error,
            "active_return_annual": float(active.mean() * ann_factor),
        }

    def print_summary(self, metrics: Dict) -> str:
        """Format metrics as a human-readable summary."""
        lines = [
            "=" * 60,
            "PERFORMANCE SUMMARY",
            "=" * 60,
            f"Total Return:          {metrics.get('total_return', 0):.2%}",
            f"Annual Return:         {metrics.get('annual_return', 0):.2%}",
            f"Annual Volatility:     {metrics.get('annual_volatility', 0):.2%}",
            f"Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):.3f}",
            f"  (Paper Definition):  {metrics.get('sharpe_paper_definition', 0):.3f}",
            f"Sortino Ratio:         {metrics.get('sortino_ratio', 0):.3f}",
            f"Calmar Ratio:          {metrics.get('calmar_ratio', 0):.3f}",
            "-" * 60,
            f"Max Drawdown:          {metrics.get('max_drawdown', 0):.2%}",
            f"VaR (95%):             {metrics.get('var_95', 0):.2%}",
            f"CVaR (95%):            {metrics.get('cvar_95', 0):.2%}",
            "-" * 60,
            f"Win Rate (Monthly):    {metrics.get('win_rate', 0):.1%}",
            f"Avg Win:               {metrics.get('avg_win', 0):.2%}",
            f"Avg Loss:              {metrics.get('avg_loss', 0):.2%}",
            f"Profit Factor:         {metrics.get('profit_factor', 0):.2f}",
            f"Skewness:              {metrics.get('skewness', 0):.3f}",
        ]

        if "information_ratio" in metrics:
            lines.extend([
                "-" * 60,
                "MARKET-NEUTRAL METRICS",
                f"Information Ratio:     {metrics.get('information_ratio', 0):.3f}",
                f"Annual Alpha:          {metrics.get('alpha_annualized', 0):.2%}",
                f"Beta:                  {metrics.get('beta', 0):.3f}",
                f"Correlation w/Market:  {metrics.get('correlation_with_benchmark', 0):.3f}",
                f"Tracking Error:        {metrics.get('tracking_error', 0):.2%}",
            ])

        lines.append("=" * 60)
        summary = "\n".join(lines)
        print(summary)
        return summary


class AblationAnalyzer:
    """
    Analyzes the contribution of each agent through ablation studies.
    As described in the paper: systematic removal and replacement of individual agents.
    """

    def compare_configurations(
        self, results: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Compare performance across different agent configurations.

        Args:
            results: Dict of config_name -> backtest_results

        Returns:
            DataFrame with comparative metrics
        """
        rows = []
        for config_name, result in results.items():
            perf = result.get("performance", {})
            row = {
                "Configuration": config_name,
                "Sharpe Ratio": perf.get("sharpe_ratio", 0),
                "Annual Return": perf.get("annual_return", 0),
                "Annual Vol": perf.get("annual_volatility", 0),
                "Max Drawdown": perf.get("max_drawdown", 0),
                "Calmar": perf.get("calmar_ratio", 0),
                "Win Rate": perf.get("win_rate", 0),
                "Info Ratio": perf.get("information_ratio", 0),
            }
            rows.append(row)

        df = pd.DataFrame(rows).set_index("Configuration")
        return df

    def agent_contribution(
        self,
        full_results: Dict,
        ablation_results: Dict[str, Dict],
    ) -> pd.DataFrame:
        """
        Measure individual agent contribution by comparing full system
        against ablated versions.
        """
        full_sharpe = full_results.get("performance", {}).get("sharpe_ratio", 0)

        rows = []
        for agent_removed, result in ablation_results.items():
            ablated_sharpe = result.get("performance", {}).get("sharpe_ratio", 0)
            rows.append({
                "Agent Removed": agent_removed,
                "Full System Sharpe": full_sharpe,
                "Ablated Sharpe": ablated_sharpe,
                "Contribution (Sharpe Δ)": full_sharpe - ablated_sharpe,
                "Contribution %": (full_sharpe - ablated_sharpe) / abs(full_sharpe) * 100
                if full_sharpe != 0 else 0,
            })

        df = pd.DataFrame(rows).sort_values("Contribution (Sharpe Δ)", ascending=False)
        return df
