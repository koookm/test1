"""
Unit tests for the investment agents.
Tests without LLM calls using mock responses.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from typing import Dict

from expert_investment_team.config.settings import SystemConfig, DEFAULT_CONFIG
from expert_investment_team.agents.base_agent import BaseAgent
from expert_investment_team.agents.fundamental_agent import FundamentalAgent
from expert_investment_team.agents.technical_agent import TechnicalAgent
from expert_investment_team.agents.quantitative_agent import QuantitativeAgent
from expert_investment_team.agents.macro_agent import MacroAgent
from expert_investment_team.agents.portfolio_manager_agent import PortfolioManagerAgent


def make_mock_config() -> SystemConfig:
    config = SystemConfig()
    config.anthropic_api_key = "test_key"
    config.agent.use_fine_grained_tasks = True
    return config


def mock_llm_response(response_dict: Dict) -> str:
    return json.dumps(response_dict)


class TestBaseAgent:
    def test_validate_score_valid(self):
        config = make_mock_config()
        agent = FundamentalAgent(config)
        assert agent._validate_score(75.0) == 75.0
        assert agent._validate_score(0) == 0.0
        assert agent._validate_score(100) == 100.0

    def test_validate_score_clamp(self):
        config = make_mock_config()
        agent = FundamentalAgent(config)
        assert agent._validate_score(-10) == 0.0
        assert agent._validate_score(110) == 100.0

    def test_validate_score_invalid(self):
        config = make_mock_config()
        agent = FundamentalAgent(config)
        assert agent._validate_score("invalid") == 50.0
        assert agent._validate_score(None) == 50.0

    def test_parse_json_response_direct(self):
        config = make_mock_config()
        agent = FundamentalAgent(config)
        result = agent._parse_json_response('{"score": 75, "reasoning": "test"}')
        assert result == {"score": 75, "reasoning": "test"}

    def test_parse_json_response_markdown(self):
        config = make_mock_config()
        agent = FundamentalAgent(config)
        markdown = '```json\n{"score": 80}\n```'
        result = agent._parse_json_response(markdown)
        assert result.get("score") == 80

    def test_parse_json_response_invalid(self):
        config = make_mock_config()
        agent = FundamentalAgent(config)
        result = agent._parse_json_response("not json at all")
        assert result == {}


class TestFundamentalAgent:
    def setup_method(self):
        self.config = make_mock_config()
        self.agent = FundamentalAgent(self.config)

    def test_coarse_grained_analysis(self):
        """Test coarse-grained mode as baseline."""
        self.config.agent.use_fine_grained_tasks = False
        agent = FundamentalAgent(self.config)

        mock_response = {
            "score": 65.0,
            "investment_thesis": "Reasonably valued with stable earnings.",
            "reasoning": "P/E is below sector average."
        }

        with patch.object(agent, "_call_llm", return_value=json.dumps(mock_response)):
            result = agent.analyze({
                "ticker": "7203.T",
                "financials": {"info": {"pe_ratio": 12.5, "roe": 0.15}},
                "sector_peers": {},
            })

        assert result["ticker"] == "7203.T"
        assert result["agent"] == "fundamental"
        assert 0 <= result["score"] <= 100

    def test_fine_grained_analysis_fallback(self):
        """Test fine-grained mode returns valid structure even with LLM errors."""
        with patch.object(self.agent, "_call_llm", side_effect=Exception("API error")):
            result = self.agent._task_valuation(
                "7203.T",
                {"pe_ratio": 12, "pb_ratio": 1.2},
                {}
            )

        assert "score" in result
        assert 0 <= result["score"] <= 100

    def test_format_peers(self):
        peers = {"avg_pe_ratio": 15.2, "avg_roe": 0.12}
        formatted = self.agent._format_peers(peers)
        assert "avg_pe_ratio" in formatted
        assert "15.2" in formatted

    def test_format_peers_empty(self):
        formatted = self.agent._format_peers({})
        assert "not available" in formatted.lower() or formatted != ""


class TestTechnicalAgent:
    def setup_method(self):
        self.config = make_mock_config()
        self.agent = TechnicalAgent(self.config)

    def test_coarse_mode(self):
        self.config.agent.use_fine_grained_tasks = False
        agent = TechnicalAgent(self.config)

        mock_response = {"score": 70, "trading_bias": "buy", "reasoning": "RSI oversold"}
        with patch.object(agent, "_call_llm", return_value=json.dumps(mock_response)):
            result = agent.analyze({
                "ticker": "7203.T",
                "indicators": {"RSI_14": 28.5, "MACD": -50.0}
            })

        assert result["score"] == 70.0
        assert result["trading_bias"] == "buy"

    def test_fine_grained_synthesis_fallback(self):
        """Test synthesis uses weighted average when LLM fails."""
        with patch.object(self.agent, "_call_llm", side_effect=Exception("error")):
            result = self.agent._task_synthesize(
                "TEST",
                trend={"score": 60.0},
                momentum={"score": 70.0},
                volatility={"score": 50.0},
                volume={"score": 65.0},
            )

        # Weighted: 60*0.35 + 70*0.35 + 65*0.20 + 50*0.10 = 21+24.5+13+5 = 63.5
        expected = 60 * 0.35 + 70 * 0.35 + 65 * 0.20 + 50 * 0.10
        assert abs(result["final_score"] - expected) < 0.01


class TestQuantitativeAgent:
    def setup_method(self):
        self.config = make_mock_config()
        self.agent = QuantitativeAgent(self.config)

    def test_compute_rankings_empty_universe(self):
        """Test rankings with empty universe."""
        rankings = self.agent._compute_rankings(
            ticker="7203.T",
            factors={"momentum_1m": 0.05},
            universe_factors={},
        )
        assert rankings == {}

    def test_compute_rankings_single_stock(self):
        """Test rankings with a small universe."""
        factors = {"momentum_1m": 0.05, "momentum_3m": 0.10}
        universe_factors = {
            "7203.T": {"momentum_1m": 0.05, "momentum_3m": 0.10},
            "6758.T": {"momentum_1m": -0.02, "momentum_3m": 0.03},
            "8306.T": {"momentum_1m": 0.08, "momentum_3m": 0.15},
        }
        rankings = self.agent._compute_rankings("7203.T", factors, universe_factors)
        # 7203.T has momentum_1m=0.05, between -0.02 and 0.08
        assert "momentum_1m_rank" in rankings
        # Should be around 67th percentile (2nd out of 3)
        assert 50 <= rankings["momentum_1m_rank"] <= 80


class TestMacroAgent:
    def setup_method(self):
        self.config = make_mock_config()
        self.agent = MacroAgent(self.config)

    def test_analyze_returns_valid_structure(self):
        mock_dim_response = {
            "score": 60,
            "primary_trend": "bull",
            "reasoning": "Markets trending higher."
        }
        mock_overall_response = {
            "environment": "favorable",
            "equity_implications": "Positive for equities.",
            "japan_specific": "BOJ remains accommodative.",
            "reasoning": "Supportive macro backdrop."
        }

        call_count = [0]
        def mock_llm(prompt, use_fast_model=False):
            call_count[0] += 1
            if call_count[0] <= 5:
                return json.dumps(mock_dim_response)
            return json.dumps(mock_overall_response)

        with patch.object(self.agent, "_call_llm", side_effect=mock_llm):
            result = self.agent.analyze({
                "macro_data": {
                    "jp_indicators": {"nikkei_225": {"value": 38000, "momentum_1m": 2.5}},
                    "us_indicators": {},
                },
                "as_of_date": "2024-06-30",
            })

        assert "overall_score" in result
        assert "dimension_scores" in result
        assert 0 <= result["overall_score"] <= 100
        assert len(result["dimension_scores"]) == 5

    def test_analyze_handles_no_macro_data(self):
        """Test graceful handling of empty macro data."""
        with patch.object(self.agent, "_call_llm", side_effect=Exception("API error")):
            result = self.agent.analyze({
                "macro_data": {"jp_indicators": {}, "us_indicators": {}},
                "as_of_date": "2024-06-30",
            })

        assert result["agent"] == "macro"
        assert 0 <= result["overall_score"] <= 100


class TestPortfolioManager:
    def setup_method(self):
        self.config = make_mock_config()
        self.agent = PortfolioManagerAgent(self.config)

    def test_construct_portfolio_basic(self):
        """Test basic portfolio construction."""
        pm_decisions = {
            f"STOCK_{i}.T": {"final_score": float(i * 5), "long_short_signal": "neutral"}
            for i in range(20)
        }

        portfolio = self.agent.construct_portfolio(
            pm_decisions, n_long=5, n_short=5
        )

        assert len(portfolio["long_positions"]) == 5
        assert len(portfolio["short_positions"]) == 5

        # Verify long positions have highest scores
        long_scores = [p["score"] for p in portfolio["long_positions"]]
        short_scores = [p["score"] for p in portfolio["short_positions"]]
        assert min(long_scores) > max(short_scores)

    def test_portfolio_weights_sum(self):
        """Verify long weights sum to 1 and short weights sum to -1."""
        pm_decisions = {
            f"STOCK_{i}.T": {"final_score": float(100 - i)}
            for i in range(20)
        }
        portfolio = self.agent.construct_portfolio(pm_decisions, n_long=5, n_short=5)

        long_weight_sum = sum(p["weight"] for p in portfolio["long_positions"])
        short_weight_sum = sum(p["weight"] for p in portfolio["short_positions"])

        assert abs(long_weight_sum - 1.0) < 1e-10
        assert abs(short_weight_sum - (-1.0)) < 1e-10

    def test_format_macro_context(self):
        macro_output = {
            "dimension_scores": {
                "market_direction": {"score": 70, "reasoning": "Bullish trend"},
                "risk_sentiment": {"score": 60, "reasoning": "Risk-on"},
            },
            "equity_implications": "Positive for equities",
        }
        formatted = self.agent._format_macro_context(macro_output)
        assert "Market Direction" in formatted
        assert "70" in formatted


class TestDataPipeline:
    """Test data collection and processing."""

    def test_technical_indicators_compute(self):
        """Test that all technical indicators are computed without errors."""
        import pandas as pd
        import numpy as np
        from expert_investment_team.data.technical_indicators import TechnicalIndicators

        # Generate synthetic OHLCV data
        n = 300
        np.random.seed(42)
        prices = 100 * (1 + np.random.randn(n) * 0.01).cumprod()
        df = pd.DataFrame({
            "Open": prices * (1 + np.random.randn(n) * 0.002),
            "High": prices * (1 + np.abs(np.random.randn(n)) * 0.005),
            "Low": prices * (1 - np.abs(np.random.randn(n)) * 0.005),
            "Close": prices,
            "Volume": np.random.randint(1000000, 5000000, n).astype(float),
        }, index=pd.date_range("2023-01-01", periods=n))

        result = TechnicalIndicators.compute_all(df)

        # Check all expected columns exist
        required_cols = [
            "RSI_14", "MACD", "MACD_signal", "MACD_histogram",
            "BB_upper", "BB_lower", "ATR_14", "ADX_14",
            "SMA_20", "SMA_50", "SMA_200",
        ]
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"

        # Check no completely-NaN columns
        for col in required_cols:
            non_null = result[col].dropna()
            assert len(non_null) > 0, f"Column {col} is all NaN"

    def test_quant_factors_compute(self):
        """Test quantitative factor computation."""
        import pandas as pd
        import numpy as np
        from expert_investment_team.data.technical_indicators import QuantitativeFactors

        n = 300
        np.random.seed(42)
        prices = 100 * (1 + np.random.randn(n) * 0.01).cumprod()
        df = pd.DataFrame({
            "Open": prices,
            "High": prices * 1.005,
            "Low": prices * 0.995,
            "Close": prices,
            "Volume": np.ones(n) * 1000000,
        }, index=pd.date_range("2023-01-01", periods=n))

        price_data = {"7203.T": df}
        index_df = pd.DataFrame({
            "Close": 2 * prices
        }, index=pd.date_range("2023-01-01", periods=n))

        factors = QuantitativeFactors.compute_factors(
            ticker="7203.T",
            price_data=price_data,
            index_data=index_df,
            as_of_date="2023-12-31",
        )

        assert "momentum_1m" in factors
        assert "momentum_12m" in factors
        assert "vol_1m" in factors
        assert factors["vol_1m"] > 0

    def test_portfolio_optimizer_equal_weight(self):
        """Test equal-weight portfolio construction."""
        from expert_investment_team.portfolio.optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer()
        long_tickers = ["STOCK_A", "STOCK_B", "STOCK_C"]
        short_tickers = ["STOCK_X", "STOCK_Y"]

        weights = optimizer.equal_weight_longshort(long_tickers, short_tickers)

        # Long weights sum to 1
        long_sum = sum(w for w in weights.values() if w > 0)
        short_sum = sum(w for w in weights.values() if w < 0)

        assert abs(long_sum - 1.0) < 1e-10
        assert abs(short_sum - (-1.0)) < 1e-10

    def test_performance_metrics_sharpe(self):
        """Test Sharpe ratio computation matches paper definition."""
        import pandas as pd
        import numpy as np
        from expert_investment_team.metrics.performance import PerformanceMetrics

        np.random.seed(42)
        monthly_returns = pd.Series(
            np.random.randn(27) * 0.03 + 0.01  # ~12% annual, ~10% vol
        )

        pm = PerformanceMetrics()
        metrics = pm.compute_all(monthly_returns, frequency="monthly")

        # Paper definition: mean / std (not annualized)
        paper_sharpe = monthly_returns.mean() / monthly_returns.std()
        assert abs(metrics["sharpe_paper_definition"] - paper_sharpe) < 1e-10

        # Verify all key metrics exist
        required_metrics = [
            "total_return", "annual_return", "annual_volatility",
            "sharpe_ratio", "max_drawdown", "win_rate",
        ]
        for m in required_metrics:
            assert m in metrics, f"Missing metric: {m}"
