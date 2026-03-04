"""
Expert Investment Team - Main Entry Point
arXiv:2602.23330: "Toward Expert Investment Teams: A Multi-Agent LLM System with Fine-Grained Trading Tasks"

Usage:
  # Live analysis (current date)
  python main.py --mode live

  # Backtest (paper's period: Sep 2023 - Nov 2025)
  python main.py --mode backtest

  # Single stock analysis
  python main.py --mode single --ticker 7203.T

  # Ablation study
  python main.py --mode ablation
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from expert_investment_team.config.settings import (
    SystemConfig, DEFAULT_CONFIG, TOPIX100_TICKERS
)
from expert_investment_team.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Expert Investment Team - Multi-Agent LLM System (arXiv:2602.23330)"
    )
    parser.add_argument(
        "--mode",
        choices=["live", "backtest", "single", "ablation", "demo"],
        default="demo",
        help="Execution mode",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="7203.T",
        help="Single stock ticker (for --mode single)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Analysis date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--fine-grained",
        action="store_true",
        default=True,
        help="Use fine-grained task decomposition (paper's key contribution)",
    )
    parser.add_argument(
        "--coarse",
        action="store_true",
        default=False,
        help="Use coarse-grained tasks (baseline for ablation)",
    )
    parser.add_argument(
        "--n-stocks",
        type=int,
        default=10,
        help="Number of stocks to analyze (for speed)",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="LLM provider",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory",
    )
    return parser.parse_args()


def check_api_keys(config: SystemConfig) -> bool:
    """Verify required API keys are configured."""
    if config.llm.provider == "anthropic" and not config.anthropic_api_key:
        logger.error(
            "ANTHROPIC_API_KEY not set. Please set it in your environment or .env file:\n"
            "  export ANTHROPIC_API_KEY=your_key_here"
        )
        return False
    if config.llm.provider == "openai" and not config.openai_api_key:
        logger.error(
            "OPENAI_API_KEY not set. Please set it in your environment or .env file:\n"
            "  export OPENAI_API_KEY=your_key_here"
        )
        return False
    return True


def run_demo_mode(config: SystemConfig, n_stocks: int = 5):
    """
    Demo mode: shows the system architecture and sample prompts
    without making real LLM calls. Good for testing without API keys.
    """
    logger.info("Running in DEMO mode (no LLM calls)")
    logger.info("This demonstrates the system architecture based on arXiv:2602.23330")

    tickers = TOPIX100_TICKERS[:n_stocks]
    print("\n" + "=" * 70)
    print("EXPERT INVESTMENT TEAM - DEMO MODE")
    print("arXiv:2602.23330: Multi-Agent LLM System with Fine-Grained Tasks")
    print("=" * 70)

    print(f"\n📊 Universe: TOPIX 100 ({len(TOPIX100_TICKERS)} stocks)")
    print(f"🗓️  Backtest Period: {config.data.backtest_start} to {config.data.backtest_end}")
    print(f"🏗️  Strategy: Long-Short Market Neutral")
    print(f"⚖️  Rebalancing: Monthly")
    print(f"📈 Long Positions: {config.portfolio.n_long}")
    print(f"📉 Short Positions: {config.portfolio.n_short}")

    print("\n🤖 AGENT ARCHITECTURE:")
    print("-" * 40)
    agents = [
        ("1. Fundamental Agent", "5 fine-grained tasks: Valuation, Quality, Growth, Balance Sheet, Synthesis"),
        ("2. Technical Agent", "5 fine-grained tasks: Trend, Momentum, Volatility, Volume, Synthesis"),
        ("3. Quantitative Agent", "4 fine-grained tasks: Momentum Factors, Risk Factors, Liquidity, Synthesis"),
        ("4. Sentiment Agent", "3 tasks: News Sentiment, Event Detection, Synthesis"),
        ("5. Sector Agent", "Synthesizes analyst outputs + sector peer comparison → Score for PM"),
        ("6. Macro Agent", "5 dimensions: Market Direction, Risk Sentiment, Growth, Rates, Inflation"),
        ("7. PM Agent", "Integrates bottom-up (Sector) + top-down (Macro) → Final Score 0-100"),
    ]
    for name, desc in agents:
        print(f"  {name}")
        print(f"    → {desc}")

    print(f"\n🔬 KEY INNOVATION (Paper's Contribution):")
    print("  Fine-Grained vs Coarse-Grained Task Decomposition")
    print("  - COARSE: 'Analyze the company's fundamentals' (vague)")
    print("  - FINE:   5 explicit sub-tasks with specific metrics and thresholds")
    print("  → Result: Significantly improved Sharpe ratio")

    print(f"\n📦 Demo: Data collection for {tickers[0]}...")
    try:
        from expert_investment_team.data.collector import StockDataCollector, FundamentalDataCollector
        from expert_investment_team.data.technical_indicators import TechnicalIndicators

        price_collector = StockDataCollector(config)
        fund_collector = FundamentalDataCollector(config)

        print(f"  Fetching price data for {tickers[0]}...")
        try:
            price_data = price_collector.get_price_data(
                [tickers[0]],
                start_date="2024-01-01",
                end_date="2024-12-31",
            )
            if price_data and tickers[0] in price_data:
                df = price_data[tickers[0]]
                with_indicators = TechnicalIndicators.compute_all(df)
                snapshot = TechnicalIndicators.get_latest_snapshot(with_indicators)
                print(f"  ✅ Got {len(df)} days of price data")
                print(f"  RSI(14): {snapshot.get('RSI_14', 'N/A'):.1f}")
                print(f"  MACD: {snapshot.get('MACD', 'N/A'):.4f}")
                print(f"  SMA20: {snapshot.get('SMA_20', 'N/A'):.1f}")
                print(f"  Price vs SMA200: {snapshot.get('price_vs_sma200', 'N/A'):.1f}%")
        except Exception as e:
            print(f"  ⚠️  Price data fetch: {e}")

        print("\n  Fetching fundamental data...")
        try:
            fundamentals = fund_collector.get_financials(tickers[0])
            info = fundamentals.get("info", {})
            print(f"  ✅ Company: {info.get('company_name', 'N/A')}")
            print(f"  Sector: {info.get('sector', 'N/A')}")
            print(f"  P/E: {info.get('pe_ratio', 'N/A')}")
            print(f"  ROE: {info.get('roe', 'N/A')}")
        except Exception as e:
            print(f"  ⚠️  Fundamentals fetch: {e}")
    except ModuleNotFoundError as e:
        print(f"  ℹ️  Install yfinance to enable data collection: pip install yfinance requests")

    print("\n✅ Demo complete. System is ready for live analysis.")
    print("\nTo run with LLM analysis:")
    print("  export ANTHROPIC_API_KEY=your_key")
    print("  python main.py --mode live --n-stocks 10")
    print("\nTo run backtest:")
    print("  python main.py --mode backtest --n-stocks 20")
    print("=" * 70)


def run_live_analysis(config: SystemConfig, tickers: list, as_of_date: str):
    """Run live investment analysis."""
    if not check_api_keys(config):
        sys.exit(1)

    from expert_investment_team.pipeline import InvestmentPipeline

    pipeline = InvestmentPipeline(config)
    result = pipeline.run_live_analysis(tickers=tickers, as_of_date=as_of_date)

    print(result["report"])

    # Save scores
    output_file = Path(config.output_dir) / f"scores_{as_of_date}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(
            {k: float(v) for k, v in result["scores"].items()},
            f, indent=2
        )
    logger.info(f"Scores saved to {output_file}")

    return result


def run_backtest(config: SystemConfig, tickers: list):
    """Run full backtesting pipeline."""
    if not check_api_keys(config):
        sys.exit(1)

    from expert_investment_team.pipeline import InvestmentPipeline
    from expert_investment_team.portfolio.backtester import Backtester
    from expert_investment_team.data.collector import StockDataCollector
    from expert_investment_team.metrics.performance import PerformanceMetrics
    import pandas as pd

    # Initialize components
    pipeline = InvestmentPipeline(config)
    backtester = Backtester(config)

    logger.info("Loading price data for full backtest period...")
    price_collector = StockDataCollector(config)
    price_data = price_collector.get_price_data(
        tickers,
        start_date="2022-01-01",  # Extra history for indicators
        end_date=config.data.backtest_end,
    )
    index_data = price_collector.get_topix_index(
        start_date="2022-01-01",
        end_date=config.data.backtest_end,
    )

    def analysis_function(ticker_list, as_of_date):
        """Wrapper to pass price_data for efficiency."""
        scores = pipeline.analyze_universe(
            tickers=ticker_list,
            as_of_date=as_of_date,
            price_data={t: price_data[t] for t in ticker_list if t in price_data},
        )
        return scores

    results = backtester.run_backtest(
        price_data=price_data,
        index_data=index_data,
        analysis_function=analysis_function,
        start_date=config.data.backtest_start,
        end_date=config.data.backtest_end,
    )

    # Print performance summary
    perf = PerformanceMetrics()
    monthly_returns = pd.Series(results.get("monthly_returns", {}))
    if not monthly_returns.empty:
        metrics = perf.compute_all(monthly_returns, frequency="monthly")
        perf.print_summary(metrics)

    return results


def run_ablation_study(config: SystemConfig, tickers: list):
    """
    Run ablation study comparing fine-grained vs coarse-grained configurations.
    Per paper: systematic removal and replacement of individual agents.
    """
    if not check_api_keys(config):
        sys.exit(1)

    from expert_investment_team.portfolio.backtester import Backtester
    from expert_investment_team.pipeline import InvestmentPipeline
    from expert_investment_team.data.collector import StockDataCollector
    from expert_investment_team.metrics.performance import AblationAnalyzer
    import copy

    price_collector = StockDataCollector(config)
    price_data = price_collector.get_price_data(
        tickers,
        start_date="2022-01-01",
        end_date=config.data.backtest_end,
    )
    index_data = price_collector.get_topix_index(
        start_date="2022-01-01",
        end_date=config.data.backtest_end,
    )

    all_results = {}

    # Configuration 1: Fine-grained (paper's proposed method)
    logger.info("Running: Fine-Grained Configuration...")
    config_fine = copy.deepcopy(config)
    config_fine.agent.use_fine_grained_tasks = True
    pipeline_fine = InvestmentPipeline(config_fine)
    backtester_fine = Backtester(config_fine)
    results_fine = backtester_fine.run_backtest(
        price_data=price_data,
        index_data=index_data,
        analysis_function=lambda tickers, date: pipeline_fine.analyze_universe(
            tickers, date, price_data={t: price_data[t] for t in tickers if t in price_data}
        ),
    )
    all_results["fine_grained"] = results_fine

    # Configuration 2: Coarse-grained (baseline)
    logger.info("Running: Coarse-Grained Configuration...")
    config_coarse = copy.deepcopy(config)
    config_coarse.agent.use_fine_grained_tasks = False
    pipeline_coarse = InvestmentPipeline(config_coarse)
    backtester_coarse = Backtester(config_coarse)
    results_coarse = backtester_coarse.run_backtest(
        price_data=price_data,
        index_data=index_data,
        analysis_function=lambda tickers, date: pipeline_coarse.analyze_universe(
            tickers, date, price_data={t: price_data[t] for t in tickers if t in price_data}
        ),
    )
    all_results["coarse_grained"] = results_coarse

    # Compare
    analyzer = AblationAnalyzer()
    comparison_df = analyzer.compare_configurations(all_results)
    print("\nABLATION STUDY RESULTS:")
    print("=" * 60)
    print(comparison_df.to_string())

    # Visualize
    try:
        from expert_investment_team.visualization import plot_performance_comparison
        plot_performance_comparison(all_results)
        logger.info("Comparison chart saved to results/performance_comparison.png")
    except Exception as e:
        logger.debug(f"Visualization failed: {e}")

    return all_results


def main():
    args = parse_args()

    # Setup logging
    setup_logger(
        log_level="DEBUG" if os.getenv("DEBUG") else "INFO",
        log_file=f"{args.output_dir}/logs/system.log",
    )

    # Build configuration
    config = SystemConfig()
    config.llm.provider = args.provider
    config.output_dir = args.output_dir
    config.agent.use_fine_grained_tasks = not args.coarse

    if args.provider == "openai":
        config.llm.model = "gpt-4o"
        config.llm.fast_model = "gpt-4o-mini"

    # Select stock universe
    n_stocks = min(args.n_stocks, len(TOPIX100_TICKERS))
    tickers = TOPIX100_TICKERS[:n_stocks]

    logger.info(f"Expert Investment Team - Mode: {args.mode}")
    logger.info(f"Fine-grained tasks: {config.agent.use_fine_grained_tasks}")
    logger.info(f"Universe: {n_stocks} stocks")

    if args.mode == "demo":
        run_demo_mode(config, n_stocks=min(5, n_stocks))

    elif args.mode == "live":
        result = run_live_analysis(config, tickers, args.date)
        print(f"\nTop long: {result['portfolio']['long_positions'][0]['ticker'] if result['portfolio']['long_positions'] else 'N/A'}")

    elif args.mode == "backtest":
        run_backtest(config, tickers)

    elif args.mode == "single":
        if not check_api_keys(config):
            sys.exit(1)
        from expert_investment_team.pipeline import InvestmentPipeline
        pipeline = InvestmentPipeline(config)
        scores = pipeline.analyze_universe([args.ticker], args.date)
        print(f"\nSingle Stock Analysis: {args.ticker}")
        print(f"Final Attractiveness Score: {scores.get(args.ticker, 50):.1f}/100")

    elif args.mode == "ablation":
        run_ablation_study(config, tickers)


if __name__ == "__main__":
    main()
