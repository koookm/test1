"""
Sector Analysis Agent
Synthesizes outputs from Fundamental, Technical, Quantitative, and Sentiment agents.
Compares stock metrics against sector averages and re-evaluates attractiveness.

Per paper arXiv:2602.23330:
"This agent synthesizes the outputs from the four analyst agents and compares
the quantitative figures of each stock with the sector averages, providing
a re-evaluated attractive score (0-100 scale) and an investment thesis to the PM agent."
"""

from typing import Dict, List
from loguru import logger
from expert_investment_team.agents.base_agent import BaseAgent
from expert_investment_team.config.settings import SystemConfig, DEFAULT_CONFIG


class SectorAgent(BaseAgent):
    """
    Sector Analysis Agent.
    Acts as a mid-level aggregator between analyst agents and PM agent.
    Re-evaluates individual analyst scores in the context of sector dynamics.
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        super().__init__(config)

    def analyze(self, data: Dict) -> Dict:
        """
        Args:
            data: Dict with:
                - 'ticker': stock ticker
                - 'company_name': company name
                - 'sector': sector name
                - 'analyst_outputs': Dict with results from fundamental, technical, quant, sentiment agents
                - 'sector_stats': sector-level statistical data

        Returns:
            Dict with re-evaluated score (0-100) and sector-relative assessment
        """
        ticker = data.get("ticker", "UNKNOWN")
        company_name = data.get("company_name", ticker)
        sector = data.get("sector", "Unknown")
        analyst_outputs = data.get("analyst_outputs", {})
        sector_stats = data.get("sector_stats", {})

        logger.info(f"[SectorAgent] Sector analysis for {ticker} ({sector})...")

        # Step 1: Sector Context Assessment
        sector_context = self._assess_sector_context(sector, sector_stats)

        # Step 2: Relative Scoring (stock vs sector peers)
        relative_score = self._assess_relative_position(
            ticker, company_name, sector, analyst_outputs, sector_stats
        )

        # Step 3: Synthesize with sector adjustment
        final_result = self._synthesize(
            ticker, company_name, sector, analyst_outputs, sector_context, relative_score
        )

        return {
            "agent": "sector",
            "ticker": ticker,
            "sector": sector,
            "score": final_result.get("final_score", 50.0),
            "sector_context_score": sector_context.get("score", 50.0),
            "relative_score": relative_score.get("score", 50.0),
            "investment_thesis": final_result.get("investment_thesis", ""),
            "sector_position": final_result.get("sector_position", "inline"),
            "reasoning": final_result.get("reasoning", ""),
        }

    def _assess_sector_context(self, sector: str, sector_stats: Dict) -> Dict:
        """Assess the overall sector attractiveness in current market environment."""
        sector_data = self._format_sector_stats(sector_stats)

        prompt = f"""You are a sector analyst assessing the {sector} sector for Japanese equities.

## Sector Statistics:
{sector_data}

## Sector Assessment Task:
Evaluate the sector's current investment attractiveness:
1. **Sector Valuation**: Is the sector cheap or expensive vs historical?
2. **Sector Momentum**: Is the sector outperforming or underperforming TOPIX?
3. **Sector Fundamental Trend**: Revenue/earnings trends for the sector
4. **Macro Sensitivity**: How sensitive is this sector to BOJ policy, USD/JPY, commodity prices?

Japanese sector context:
- Technology/Electronics: Export-sensitive, USD/JPY impact
- Financials: BOJ rate policy sensitive
- Consumer: Domestic demand, demographics
- Industrials: Global capex cycle, China exposure
- Healthcare/Pharma: Aging population tailwind

Respond with valid JSON only:
{{
  "score": <0-100, where 100=most attractive sector environment>,
  "sector_trend": "<outperforming|inline|underperforming>",
  "macro_tailwinds": ["<tailwind1>"],
  "macro_headwinds": ["<headwind1>"],
  "sector_view": "<positive|neutral|negative>",
  "reasoning": "<explanation>"
}}"""

        try:
            response = self._call_llm(prompt, use_fast_model=True)
            result = self._parse_json_response(response)
            result["score"] = self._validate_score(result.get("score", 50))
            return result
        except Exception as e:
            logger.error(f"Sector context assessment failed: {e}")
            return {"score": 50.0, "sector_view": "neutral"}

    def _assess_relative_position(
        self,
        ticker: str,
        company_name: str,
        sector: str,
        analyst_outputs: Dict,
        sector_stats: Dict,
    ) -> Dict:
        """Assess stock's position relative to sector peers."""
        fundamental = analyst_outputs.get("fundamental", {})
        technical = analyst_outputs.get("technical", {})
        quant = analyst_outputs.get("quantitative", {})
        sentiment = analyst_outputs.get("sentiment", {})

        # Extract sub-scores
        fund_score = fundamental.get("score", 50)
        tech_score = technical.get("score", 50)
        quant_score = quant.get("score", 50)
        sent_score = sentiment.get("score", 50)

        prompt = f"""You are a sector analyst evaluating {company_name} ({ticker}) relative to {sector} sector peers.

## Individual Analyst Scores:
- Fundamental Score: {fund_score}/100
  - Valuation: {fundamental.get('sub_scores', {}).get('valuation', 'N/A')}/100
  - Quality: {fundamental.get('sub_scores', {}).get('quality', 'N/A')}/100
  - Growth: {fundamental.get('sub_scores', {}).get('growth', 'N/A')}/100
  - Balance Sheet: {fundamental.get('sub_scores', {}).get('balance_sheet', 'N/A')}/100
  - Thesis: {fundamental.get('investment_thesis', 'N/A')[:150]}

- Technical Score: {tech_score}/100
  - Trading Bias: {technical.get('trading_bias', 'N/A')}
  - Top Signals: {technical.get('signals', [])[:2]}

- Quantitative Score: {quant_score}/100
  - Quant View: {quant.get('quant_view', 'N/A')}
  - Factor Signals: {quant.get('factor_signals', [])[:2]}

- Sentiment Score: {sent_score}/100
  - Overall Sentiment: {sentiment.get('overall_sentiment', 'N/A')}

## Sector Comparison Data:
{self._format_sector_stats(sector_stats)}

## Relative Position Task:
Evaluate {company_name} vs sector peers:
1. Is the stock's fundamental profile better or worse than sector average?
2. Does its technical momentum diverge from sector trend?
3. Is there a valuation discount/premium vs sector peers justified by fundamentals?
4. Final relative attractiveness score within sector

Respond with valid JSON only:
{{
  "score": <0-100, where 100=most attractive within sector>,
  "vs_sector_fundamental": "<significant_premium|premium|inline|discount|significant_discount>",
  "vs_sector_momentum": "<leading|inline|lagging>",
  "valuation_vs_peers": "<expensive|fair|cheap>",
  "relative_quality": "<top_tier|above_avg|avg|below_avg|bottom_tier>",
  "key_differentiators": ["<diff1>", "<diff2>"],
  "reasoning": "<explanation>"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            result["score"] = self._validate_score(result.get("score", 50))
            return result
        except Exception as e:
            logger.error(f"Relative position assessment failed for {ticker}: {e}")
            # Fallback: simple average of analyst scores
            avg_score = (fund_score + tech_score + quant_score + sent_score) / 4
            return {"score": avg_score, "relative_quality": "avg"}

    def _synthesize(
        self,
        ticker: str,
        company_name: str,
        sector: str,
        analyst_outputs: Dict,
        sector_context: Dict,
        relative_score: Dict,
    ) -> Dict:
        """Synthesize sector analysis into final sector-adjusted score and thesis."""
        prompt = f"""You are the sector team lead generating the final sector assessment for {company_name} ({ticker}).

## Sector Context (Score: {sector_context.get('score', 50)}/100):
Sector View: {sector_context.get('sector_view', 'N/A')}
Trend: {sector_context.get('sector_trend', 'N/A')}
Tailwinds: {sector_context.get('macro_tailwinds', [])}
Headwinds: {sector_context.get('macro_headwinds', [])}

## Relative Position (Score: {relative_score.get('score', 50)}/100):
Quality: {relative_score.get('relative_quality', 'N/A')}
Valuation vs Peers: {relative_score.get('valuation_vs_peers', 'N/A')}
Momentum vs Peers: {relative_score.get('vs_sector_momentum', 'N/A')}
Key Differentiators: {relative_score.get('key_differentiators', [])}

## Synthesis Task:
Generate the final sector-adjusted score to pass to the Portfolio Manager:
- Sector Context: 30% weight
- Relative Position: 70% weight (stock-specific > sector)

Also write a concise investment thesis (2-3 sentences) for the PM.

Respond with valid JSON only:
{{
  "final_score": <0-100>,
  "investment_thesis": "<2-3 sentence thesis for PM>",
  "sector_position": "<sector_leader|above_avg|avg|below_avg|sector_laggard>",
  "conviction": "<low|medium|high>",
  "reasoning": "<explanation>"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            result["final_score"] = self._validate_score(result.get("final_score", 50))
            return result
        except Exception as e:
            logger.error(f"Sector synthesis failed for {ticker}: {e}")
            weighted = (
                sector_context.get("score", 50) * 0.30 +
                relative_score.get("score", 50) * 0.70
            )
            return {
                "final_score": weighted,
                "investment_thesis": "Synthesis unavailable",
                "sector_position": "avg",
                "reasoning": "Fallback to weighted average",
            }

    def _format_sector_stats(self, stats: Dict) -> str:
        if not stats:
            return "Sector comparison data not available"
        lines = []
        for key, value in stats.items():
            lines.append(f"- Sector {key}: {value}")
        return "\n".join(lines) if lines else "No sector data"
