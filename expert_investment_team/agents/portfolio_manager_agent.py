"""
Portfolio Manager (PM) Agent
Integrates bottom-up stock analysis (from Sector Agent) with
top-down macro view (from Macro Agent) to generate final investment scores.

Per paper arXiv:2602.23330:
"This agent integrates the bottom-up view (from the Sector Agent) with
the top-down view (from the Macro Agent), then generates a final attractive score
(0-100 scale) for long-short portfolio construction."

This is the final decision-making agent in the multi-agent hierarchy.
"""

from typing import Dict, List
from loguru import logger
from expert_investment_team.agents.base_agent import BaseAgent
from expert_investment_team.config.settings import SystemConfig, DEFAULT_CONFIG


class PortfolioManagerAgent(BaseAgent):
    """
    Portfolio Manager Agent - the apex of the agent hierarchy.

    Combines:
    - Bottom-up view: Sector Agent output (fundamental + technical + quant + sentiment)
    - Top-down view: Macro Agent output (5 macro dimensions)

    Produces:
    - Final attractiveness score (0-100) for long-short portfolio construction
    - Investment thesis
    - Position sizing recommendation
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        super().__init__(config)
        self.portfolio_config = config.portfolio

    def analyze(self, data: Dict) -> Dict:
        """
        BaseAgent abstract method implementation.
        For the PM agent, primary interface is generate_scores().
        This method handles single-stock PM decisions when called directly.
        """
        ticker = data.get("ticker", "UNKNOWN")
        sector_data = data.get("sector_data", {"score": 50.0})
        macro_output = data.get("macro_output", {})
        macro_context = self._format_macro_context(macro_output)
        return self._generate_single_score(ticker, sector_data, macro_context, macro_output)

    def generate_scores(
        self,
        sector_outputs: Dict[str, Dict],
        macro_output: Dict,
    ) -> Dict[str, Dict]:
        """
        Generate final investment scores for all stocks in the universe.

        Args:
            sector_outputs: Dict of ticker -> sector_agent output
            macro_output: Output from MacroAgent

        Returns:
            Dict of ticker -> final PM decision with score
        """
        logger.info(f"[PM Agent] Generating scores for {len(sector_outputs)} stocks...")

        macro_context = self._format_macro_context(macro_output)
        pm_decisions = {}

        for ticker, sector_data in sector_outputs.items():
            decision = self._generate_single_score(
                ticker=ticker,
                sector_data=sector_data,
                macro_context=macro_context,
                macro_output=macro_output,
            )
            pm_decisions[ticker] = decision

        return pm_decisions

    def _generate_single_score(
        self,
        ticker: str,
        sector_data: Dict,
        macro_context: str,
        macro_output: Dict,
    ) -> Dict:
        """Generate final score for a single stock."""
        # Extract key info from sector agent output
        sector_score = sector_data.get("score", 50)
        sector = sector_data.get("sector", "Unknown")
        investment_thesis = sector_data.get("investment_thesis", "")
        sector_position = sector_data.get("sector_position", "avg")

        macro_score = macro_output.get("overall_score", 50)
        macro_env = macro_output.get("macro_environment", "neutral")

        prompt = f"""You are the Portfolio Manager making the final investment decision for {ticker}.

## Bottom-Up Analysis (from Sector Team):
- Stock: {ticker}
- Sector: {sector}
- Bottom-Up Score: {sector_score}/100
- Sector Position: {sector_position}
- Investment Thesis: {investment_thesis}

## Top-Down Macro Analysis:
{macro_context}
- Macro Environment: {macro_env}
- Macro Score: {macro_score}/100

## Portfolio Construction Parameters:
- Strategy: Long-Short Market Neutral
- Universe: TOPIX 100 (Japanese Large-Caps)
- Rebalancing: Monthly
- Target: {self.portfolio_config.n_long} long + {self.portfolio_config.n_short} short positions

## PM Decision Task:
As Portfolio Manager, your role is to:
1. **Integrate Views**: Does the macro environment support the bottom-up thesis?
   - Macro tailwinds amplify positive stock ideas
   - Macro headwinds may reduce conviction or increase short appeal
2. **Macro Adjustment**: Apply a sector-specific macro multiplier
   - E.g., If macro is bullish for exporters and this is an exporter → boost score
   - If macro is hawkish and this is a highly leveraged growth stock → reduce score
3. **Final Score**: Generate the final attractiveness score (0-100)
   - Score > 70: Long candidate
   - Score 40-70: Neutral / Hold
   - Score < 30: Short candidate
4. **Position Size Signal**: Based on conviction level

Respond with valid JSON only:
{{
  "ticker": "{ticker}",
  "final_score": <0-100>,
  "long_short_signal": "<strong_long|long|neutral|short|strong_short>",
  "macro_adjustment": <-20 to +20, positive=boosted by macro>,
  "conviction_level": "<low|medium|high|very_high>",
  "weight_signal": "<0.5x|1.0x|1.5x|2.0x of equal weight>,
  "investment_thesis": "<concise 2-sentence thesis combining bottom-up + macro>",
  "key_risks": ["<risk1>", "<risk2>"],
  "monitoring_triggers": ["<what would change the thesis>"],
  "reasoning": "<PM's synthesis reasoning>"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            result["final_score"] = self._validate_score(result.get("final_score", 50))
            result["ticker"] = ticker
            result["sector"] = sector
            result["bottom_up_score"] = sector_score
            result["macro_score"] = macro_score
            return result
        except Exception as e:
            logger.error(f"PM decision failed for {ticker}: {e}")
            # Fallback: bottom-up score with small macro adjustment
            macro_adj = (macro_score - 50) * 0.1
            final = max(0, min(100, sector_score + macro_adj))
            return {
                "ticker": ticker,
                "final_score": final,
                "long_short_signal": "neutral",
                "conviction_level": "low",
                "investment_thesis": "Analysis unavailable",
                "key_risks": [],
                "reasoning": "Fallback calculation",
            }

    def _format_macro_context(self, macro_output: Dict) -> str:
        """Format macro output as context for PM decision."""
        dim_scores = macro_output.get("dimension_scores", {})
        lines = []

        dimension_labels = {
            "market_direction": "Market Direction",
            "risk_sentiment": "Risk Sentiment",
            "economic_growth": "Economic Growth",
            "interest_rates": "Interest Rates",
            "inflation": "Inflation",
        }

        for dim, label in dimension_labels.items():
            if dim in dim_scores:
                score = dim_scores[dim].get("score", 50)
                reasoning = dim_scores[dim].get("reasoning", "")[:80]
                lines.append(f"- {label}: {score:.0f}/100 — {reasoning}")

        equity_impl = macro_output.get("equity_implications", "")
        japan_spec = macro_output.get("japan_specific", "")

        if equity_impl:
            lines.append(f"\nEquity Implications: {equity_impl}")
        if japan_spec:
            lines.append(f"Japan Specifics: {japan_spec}")

        return "\n".join(lines) if lines else "Macro data unavailable"

    def construct_portfolio(
        self,
        pm_decisions: Dict[str, Dict],
        n_long: int = None,
        n_short: int = None,
    ) -> Dict:
        """
        Construct the long-short portfolio based on PM scores.

        Args:
            pm_decisions: Dict of ticker -> PM decision
            n_long: Number of long positions (default from config)
            n_short: Number of short positions (default from config)

        Returns:
            Portfolio dict with long/short positions and weights
        """
        n_long = n_long or self.portfolio_config.n_long
        n_short = n_short or self.portfolio_config.n_short

        # Sort by final score
        scored = [
            (ticker, data.get("final_score", 50), data)
            for ticker, data in pm_decisions.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Select long positions (top n by score)
        long_candidates = scored[:n_long]

        # Select short positions (bottom n by score)
        short_candidates = scored[-n_short:]

        # Equal weighting
        long_weight = 1.0 / n_long
        short_weight = -1.0 / n_short  # Negative for short positions

        portfolio = {
            "long_positions": [
                {
                    "ticker": t,
                    "score": s,
                    "weight": long_weight,
                    "signal": d.get("long_short_signal", "long"),
                    "conviction": d.get("conviction_level", "medium"),
                    "thesis": d.get("investment_thesis", ""),
                    "key_risks": d.get("key_risks", []),
                }
                for t, s, d in long_candidates
            ],
            "short_positions": [
                {
                    "ticker": t,
                    "score": s,
                    "weight": short_weight,
                    "signal": d.get("long_short_signal", "short"),
                    "conviction": d.get("conviction_level", "medium"),
                    "thesis": d.get("investment_thesis", ""),
                }
                for t, s, d in short_candidates
            ],
            "metadata": {
                "n_long": n_long,
                "n_short": n_short,
                "total_gross_exposure": 2.0,  # 100% long + 100% short
                "net_exposure": 0.0,  # Market neutral
                "score_spread": (
                    long_candidates[0][1] - short_candidates[-1][1]
                    if long_candidates and short_candidates else 0
                ),
            },
        }

        logger.info(
            f"Portfolio constructed: {n_long} longs, {n_short} shorts, "
            f"score spread: {portfolio['metadata']['score_spread']:.1f}"
        )

        return portfolio
