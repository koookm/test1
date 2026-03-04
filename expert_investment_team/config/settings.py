"""
Configuration settings for Expert Investment Team
Based on: "Toward Expert Investment Teams: A Multi-Agent LLM System with Fine-Grained Trading Tasks"
arXiv: 2602.23330
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """LLM model configuration

    Provider options:
      "anthropic_oauth" - Anthropic OAuth access token (Claude Code 계정, 1안 기본값)
      "anthropic"       - Anthropic API key (ANTHROPIC_API_KEY)
      "gemini"          - Google Gemini API key (GOOGLE_API_KEY, 2안)
      "openai"          - OpenAI API key (OPENAI_API_KEY)
    """
    provider: str = "anthropic_oauth"   # 1안: OAuth (기본값)
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.1
    max_tokens: int = 4096
    # Fallback (저비용 고속 태스크용)
    fast_model: str = "claude-haiku-4-5-20251001"

    # Gemini 모델명 (2안)
    gemini_model: str = "gemini-2.0-flash"
    gemini_fast_model: str = "gemini-2.0-flash-lite"


@dataclass
class DataConfig:
    """Data source configuration"""
    # Japanese Market - TOPIX 100 universe
    market: str = "JP"
    index: str = "TOPIX100"

    # Backtesting period (paper: Sep 2023 ~ Nov 2025)
    backtest_start: str = "2023-09-01"
    backtest_end: str = "2025-11-30"

    # Data types
    price_lookback_days: int = 252       # ~1 year of trading days
    financial_quarters: int = 8          # 2 years of quarterly data
    news_lookback_days: int = 30         # 1 month of news

    # Japan macro indicators
    jp_macro_indicators: List[str] = field(default_factory=lambda: [
        "JP_GDP",
        "JP_CPI",
        "JP_UNEMPLOYMENT",
        "JP_INTEREST_RATE",
        "JP_PMI",
        "JP_TRADE_BALANCE",
        "JP_INDUSTRIAL_PRODUCTION",
    ])

    # US macro indicators (for global context)
    us_macro_indicators: List[str] = field(default_factory=lambda: [
        "US_GDP",
        "US_CPI",
        "US_UNEMPLOYMENT",
        "US_FEDERAL_FUNDS_RATE",
        "US_PMI",
        "US_ISM_MANUFACTURING",
        "US_10Y_YIELD",
    ])


@dataclass
class PortfolioConfig:
    """Portfolio construction configuration"""
    # Long-short market-neutral strategy
    strategy: str = "long_short_market_neutral"

    # Number of stocks for long and short positions
    n_long: int = 10
    n_short: int = 10

    # Equal weighting
    weighting: str = "equal"

    # Rebalancing frequency
    rebalance_frequency: str = "monthly"

    # Score thresholds (0-100 scale)
    score_long_threshold: float = 70.0
    score_short_threshold: float = 30.0

    # Risk management
    max_single_stock_weight: float = 0.15
    max_sector_weight: float = 0.40
    stop_loss_pct: float = 0.10

    # Transaction costs (realistic for Japanese market)
    transaction_cost_bps: float = 5.0  # 5 basis points


@dataclass
class AgentConfig:
    """Agent system configuration"""
    # Fine-grained task decomposition (key contribution of the paper)
    use_fine_grained_tasks: bool = True

    # Agent weights for final score aggregation
    agent_weights: dict = field(default_factory=lambda: {
        "fundamental": 0.25,
        "technical": 0.20,
        "quantitative": 0.20,
        "sentiment": 0.15,
        "sector": 0.10,
        "macro": 0.10,
    })

    # Scoring dimensions for Macro Agent (5 dimensions from paper)
    macro_dimensions: List[str] = field(default_factory=lambda: [
        "market_direction",
        "risk_sentiment",
        "economic_growth",
        "interest_rates",
        "inflation",
    ])

    # Technical analysis indicators
    technical_indicators: List[str] = field(default_factory=lambda: [
        "RSI",
        "MACD",
        "BB",          # Bollinger Bands
        "MA",          # Moving Averages (20, 50, 200)
        "VOLUME",      # Volume analysis
        "MOMENTUM",    # Price momentum
        "ATR",         # Average True Range (volatility)
        "ADX",         # Average Directional Index (trend strength)
    ])

    # Quantitative analysis factors
    quant_factors: List[str] = field(default_factory=lambda: [
        "momentum_1m",
        "momentum_3m",
        "momentum_6m",
        "momentum_12m",
        "volatility_realized",
        "beta",
        "correlation_with_index",
        "liquidity_score",
        "size_factor",
        "value_factor",
        "quality_factor",
    ])

    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class SystemConfig:
    """Main system configuration"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    # API Keys (loaded from environment)
    # 1안: Anthropic OAuth (Claude Code 로그인 토큰, ANTHROPIC_ACCESS_TOKEN 자동 감지)
    anthropic_access_token: str = field(default_factory=lambda: os.getenv("ANTHROPIC_ACCESS_TOKEN", ""))
    # 일반 API Key (OAuth 없을 때 fallback)
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    # 2안: Google Gemini
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    alpha_vantage_key: str = field(default_factory=lambda: os.getenv("ALPHA_VANTAGE_API_KEY", ""))
    fred_api_key: str = field(default_factory=lambda: os.getenv("FRED_API_KEY", ""))
    news_api_key: str = field(default_factory=lambda: os.getenv("NEWS_API_KEY", ""))

    # Output settings
    output_dir: str = "results"
    log_level: str = "INFO"
    save_intermediate_outputs: bool = True  # For explainability analysis


# Default TOPIX 100 constituents (simplified representative set)
# In production: fetch dynamically from TSE or data provider
TOPIX100_TICKERS = [
    "7203.T",  # Toyota Motor
    "6758.T",  # Sony Group
    "8306.T",  # Mitsubishi UFJ Financial
    "9984.T",  # SoftBank Group
    "6861.T",  # Keyence
    "7974.T",  # Nintendo
    "4502.T",  # Takeda Pharmaceutical
    "8035.T",  # Tokyo Electron
    "6098.T",  # Recruit Holdings
    "9432.T",  # NTT
    "4063.T",  # Shin-Etsu Chemical
    "7267.T",  # Honda Motor
    "8316.T",  # Sumitomo Mitsui Financial
    "6954.T",  # Fanuc
    "9983.T",  # Fast Retailing
    "4519.T",  # Chugai Pharmaceutical
    "6367.T",  # Daikin Industries
    "8411.T",  # Mizuho Financial
    "7751.T",  # Canon
    "6723.T",  # Renesas Electronics
    "4568.T",  # Daiichi Sankyo
    "9022.T",  # Central Japan Railway
    "5108.T",  # Bridgestone
    "8031.T",  # Mitsui & Co.
    "8766.T",  # Tokio Marine Holdings
    "6702.T",  # Fujitsu
    "6503.T",  # Mitsubishi Electric
    "7741.T",  # Hoya Corporation
    "8058.T",  # Mitsubishi Corporation
    "4523.T",  # Eisai
]

# TOPIX index ticker for market-neutral benchmark
TOPIX_INDEX_TICKER = "^TOPX"


DEFAULT_CONFIG = SystemConfig()
