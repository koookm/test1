"""
Macro Analysis Agent
Analyzes the macroeconomic environment across 5 dimensions.

Per paper arXiv:2602.23330:
"This agent analyzes the economic environment with five dimensions—
Market Direction, Risk Sentiment, Economic Growth, Interest Rates, and Inflation—
based on the absolute levels and month-to-month changes of JP/US economic indicators,
then provides scores (0-100 scale) for each dimension and reasoning texts to the PM agent."
"""

from typing import Dict
from loguru import logger
from expert_investment_team.agents.base_agent import BaseAgent
from expert_investment_team.config.settings import SystemConfig, DEFAULT_CONFIG


class MacroAgent(BaseAgent):
    """
    Macro Analysis Agent providing top-down view to the Portfolio Manager.

    Analyzes 5 macro dimensions as specified in the paper:
    1. Market Direction
    2. Risk Sentiment
    3. Economic Growth
    4. Interest Rates
    5. Inflation
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        super().__init__(config)
        self.dimensions = config.agent.macro_dimensions

    def analyze(self, data: Dict) -> Dict:
        """
        Args:
            data: Dict with 'macro_data' containing JP and US indicators,
                  'as_of_date'

        Returns:
            Dict with score (0-100) for each macro dimension + overall macro score
        """
        macro_data = data.get("macro_data", {})
        as_of_date = data.get("as_of_date", "")

        logger.info(f"[MacroAgent] Macro analysis as of {as_of_date}...")

        jp_data = macro_data.get("jp_indicators", {})
        us_data = macro_data.get("us_indicators", {})

        # Analyze each of the 5 macro dimensions
        dimension_scores = {}

        dim1 = self._analyze_market_direction(jp_data, us_data, as_of_date)
        dim2 = self._analyze_risk_sentiment(jp_data, us_data, as_of_date)
        dim3 = self._analyze_economic_growth(jp_data, us_data, as_of_date)
        dim4 = self._analyze_interest_rates(jp_data, us_data, as_of_date)
        dim5 = self._analyze_inflation(jp_data, us_data, as_of_date)

        dimension_scores = {
            "market_direction": dim1,
            "risk_sentiment": dim2,
            "economic_growth": dim3,
            "interest_rates": dim4,
            "inflation": dim5,
        }

        # Overall macro score (equal-weighted as per paper)
        scores = [d.get("score", 50) for d in dimension_scores.values()]
        overall_score = sum(scores) / len(scores) if scores else 50.0

        # Generate overall macro assessment
        overall = self._generate_overall_assessment(dimension_scores, as_of_date)

        return {
            "agent": "macro",
            "as_of_date": as_of_date,
            "overall_score": self._validate_score(overall_score),
            "score": self._validate_score(overall_score),
            "dimension_scores": {
                dim: {"score": d.get("score", 50), "reasoning": d.get("reasoning", "")}
                for dim, d in dimension_scores.items()
            },
            "macro_environment": overall.get("environment", "neutral"),
            "equity_implications": overall.get("equity_implications", ""),
            "japan_specific": overall.get("japan_specific", ""),
            "reasoning": overall.get("reasoning", ""),
        }

    def _analyze_market_direction(
        self, jp_data: Dict, us_data: Dict, as_of_date: str
    ) -> Dict:
        """Dimension 1: Overall market direction (bull/bear/sideways)."""
        nikkei = jp_data.get("nikkei_225", {})
        topix = jp_data.get("japan_etf", {})
        us_bond = us_data.get("us_long_bond", {})

        prompt = f"""Analyze the Market Direction dimension for Japanese equities as of {as_of_date}.

## Market Indicators:
### Japanese Market:
- Nikkei 225 Level: {nikkei.get('value', 'N/A')}
- Nikkei 1M Momentum: {nikkei.get('momentum_1m', 'N/A')}%
- Japan ETF (EWJ) Level: {topix.get('value', 'N/A')}
- Japan ETF 1M Momentum: {topix.get('momentum_1m', 'N/A')}%

### USD/JPY:
- USD/JPY Rate: {jp_data.get('usd_jpy', {}).get('value', 'N/A')}
- USD/JPY 1M Change: {jp_data.get('usd_jpy', {}).get('momentum_1m', 'N/A')}%

### Global Context:
- US Long Bond (TLT) Momentum: {us_data.get('us_long_bond', {}).get('change_pct', 'N/A')}%

## Market Direction Assessment:
Evaluate the current market trend for Japanese equities:
1. Is the primary trend bullish or bearish?
2. USD/JPY impact: JPY weakening (USD/JPY rising) is typically positive for Japanese exporters
3. Is momentum accelerating or decelerating?

Respond with valid JSON only:
{{
  "score": <0-100, where 100=strongest bullish market direction>,
  "primary_trend": "<strong_bull|bull|neutral|bear|strong_bear>",
  "momentum_status": "<accelerating|stable|decelerating>",
  "usd_jpy_impact": "<very_positive|positive|neutral|negative|very_negative>",
  "reasoning": "<2-3 sentences>"
}}"""

        try:
            response = self._call_llm(prompt, use_fast_model=True)
            result = self._parse_json_response(response)
            result["score"] = self._validate_score(result.get("score", 50))
            return result
        except Exception as e:
            logger.error(f"Market direction analysis failed: {e}")
            return {"score": 50.0, "primary_trend": "neutral"}

    def _analyze_risk_sentiment(
        self, jp_data: Dict, us_data: Dict, as_of_date: str
    ) -> Dict:
        """Dimension 2: Risk-on vs risk-off market sentiment."""
        gold = us_data.get("gold", {})
        vix = us_data.get("volatility", {})

        prompt = f"""Analyze the Risk Sentiment dimension for Japanese equities as of {as_of_date}.

## Risk Sentiment Indicators:
- Gold Price Change (1M): {gold.get('change_pct', 'N/A')}% (rising=risk-off)
- VIX Level: {vix.get('value', 'N/A')} (>20=elevated fear)
- VIX Change: {vix.get('change_pct', 'N/A')}%
- US Long Bond Demand: {us_data.get('us_long_bond', {}).get('change_pct', 'N/A')}% (rising=risk-off)
- Japan ETF Volatility Proxy: {jp_data.get('japan_etf', {}).get('momentum_1m', 'N/A')}%

## Risk Sentiment Assessment:
1. Risk-On: Equities bid, gold falls, VIX low, high-beta outperforms
2. Risk-Off: Flight to safety, gold rises, VIX spikes, defensives outperform
3. Japan-specific: Carry trade dynamics (low JPY rate = risk-on carry)

Respond with valid JSON only:
{{
  "score": <0-100, where 100=maximum risk-on (good for equities)>,
  "risk_regime": "<strong_risk_on|risk_on|neutral|risk_off|strong_risk_off>",
  "fear_level": "<low|moderate|elevated|high|extreme>",
  "equity_implication": "<very_favorable|favorable|neutral|unfavorable|very_unfavorable>",
  "reasoning": "<2-3 sentences>"
}}"""

        try:
            response = self._call_llm(prompt, use_fast_model=True)
            result = self._parse_json_response(response)
            result["score"] = self._validate_score(result.get("score", 50))
            return result
        except Exception as e:
            logger.error(f"Risk sentiment analysis failed: {e}")
            return {"score": 50.0, "risk_regime": "neutral"}

    def _analyze_economic_growth(
        self, jp_data: Dict, us_data: Dict, as_of_date: str
    ) -> Dict:
        """Dimension 3: Economic growth trajectory."""
        us_gdp = us_data.get("us_gdp", {})
        us_unemp = us_data.get("us_unemployment", {})

        prompt = f"""Analyze the Economic Growth dimension as of {as_of_date}.

## Economic Growth Indicators:
### US Economy (Global Leader):
- US Unemployment Rate: {us_unemp.get('value', 'N/A')}% (lower=better growth)
- US GDP Growth: {us_gdp.get('value', 'N/A')}

### Japan Economy:
- Japan PMI/Growth Proxy (Nikkei 1M return): {jp_data.get('nikkei_225', {}).get('momentum_1m', 'N/A')}%
- Japan ETF Momentum: {jp_data.get('japan_etf', {}).get('momentum_1m', 'N/A')}%
- USD/JPY (export competitiveness): {jp_data.get('usd_jpy', {}).get('value', 'N/A')}

## Economic Growth Assessment:
1. Is global growth (led by US) supportive of Japanese corporate earnings?
2. Japan domestic demand vs export demand balance
3. Cyclical vs defensive positioning implication

Respond with valid JSON only:
{{
  "score": <0-100, where 100=strongest growth environment for Japanese equities>,
  "growth_phase": "<expansion|late_cycle|contraction|recovery>",
  "japan_growth_outlook": "<strong|moderate|weak>",
  "cyclical_bias": "<favor_cyclicals|neutral|favor_defensives>",
  "reasoning": "<2-3 sentences>"
}}"""

        try:
            response = self._call_llm(prompt, use_fast_model=True)
            result = self._parse_json_response(response)
            result["score"] = self._validate_score(result.get("score", 50))
            return result
        except Exception as e:
            logger.error(f"Economic growth analysis failed: {e}")
            return {"score": 50.0, "growth_phase": "expansion"}

    def _analyze_interest_rates(
        self, jp_data: Dict, us_data: Dict, as_of_date: str
    ) -> Dict:
        """Dimension 4: Interest rate environment."""
        us_fed = us_data.get("us_fed_funds_rate", {})
        us_10y = us_data.get("us_10y_yield", {})
        jp_10y = jp_data.get("us_10y_yield", {})  # Using as proxy

        prompt = f"""Analyze the Interest Rates dimension as of {as_of_date}.

## Interest Rate Indicators:
### US Rates:
- Federal Funds Rate: {us_fed.get('value', 'N/A')}%
- US 10Y Treasury Yield: {us_10y.get('value', 'N/A')}%

### Japan Rates:
- USD/JPY Rate (BOJ policy proxy): {jp_data.get('usd_jpy', {}).get('value', 'N/A')}
- USD/JPY Change (1M): {jp_data.get('usd_jpy', {}).get('momentum_1m', 'N/A')}%
- US Long Bond Performance: {us_data.get('us_long_bond', {}).get('change_pct', 'N/A')}%

## Interest Rate Assessment for Japanese Equities:
1. BOJ Policy: Negative/near-zero rates historically supported Japanese equities
   - Rising JP rates: Pressure on growth stocks, benefit to financials
   - Maintained low rates: Supportive of equity valuations
2. US-Japan rate differential: Drives USD/JPY carry trade
3. Rate impact on equity valuation (discount rate)

Respond with valid JSON only:
{{
  "score": <0-100, where 100=most equity-favorable rate environment>,
  "rate_environment": "<very_accommodative|accommodative|neutral|restrictive|very_restrictive>",
  "boj_stance_inference": "<dovish|neutral|hawkish>",
  "rate_equity_impact": "<strong_positive|positive|neutral|negative|strong_negative>",
  "financial_sector_implication": "<positive|neutral|negative>",
  "reasoning": "<2-3 sentences>"
}}"""

        try:
            response = self._call_llm(prompt, use_fast_model=True)
            result = self._parse_json_response(response)
            result["score"] = self._validate_score(result.get("score", 50))
            return result
        except Exception as e:
            logger.error(f"Interest rate analysis failed: {e}")
            return {"score": 50.0, "rate_environment": "accommodative"}

    def _analyze_inflation(
        self, jp_data: Dict, us_data: Dict, as_of_date: str
    ) -> Dict:
        """Dimension 5: Inflation environment."""
        us_cpi = us_data.get("us_cpi_yoy", {})
        gold = us_data.get("gold", {})

        prompt = f"""Analyze the Inflation dimension as of {as_of_date}.

## Inflation Indicators:
### US Inflation:
- US CPI (YoY): {us_cpi.get('value', 'N/A')}%
- Gold Price (inflation hedge): {gold.get('value', 'N/A')}
- Gold 1M Change: {gold.get('change_pct', 'N/A')}%

### Japan Inflation Proxy:
- USD/JPY Level (import inflation proxy): {jp_data.get('usd_jpy', {}).get('value', 'N/A')}
- JPY depreciation drives import cost inflation in Japan

## Inflation Assessment for Japanese Equities:
1. Japan reflation: Moderate inflation (1-3%) has been positive after decades of deflation
2. Import inflation risk: Weak JPY raises input costs for domestic-oriented firms
3. Pricing power: Can companies pass through cost inflation?
4. Real rate impact on equity valuations

Respond with valid JSON only:
{{
  "score": <0-100, where 100=optimal inflation for Japanese equities>,
  "inflation_regime": "<deflation|low|moderate|elevated|high>",
  "japan_import_inflation": "<low|moderate|high>",
  "pricing_power_environment": "<good|neutral|poor>",
  "real_rate_impact": "<positive|neutral|negative>",
  "reasoning": "<2-3 sentences>"
}}"""

        try:
            response = self._call_llm(prompt, use_fast_model=True)
            result = self._parse_json_response(response)
            result["score"] = self._validate_score(result.get("score", 50))
            return result
        except Exception as e:
            logger.error(f"Inflation analysis failed: {e}")
            return {"score": 50.0, "inflation_regime": "moderate"}

    def _generate_overall_assessment(
        self, dimension_scores: Dict, as_of_date: str
    ) -> Dict:
        """Generate overall macro environment summary for the PM agent."""
        scores_summary = "\n".join([
            f"- {dim.replace('_', ' ').title()}: {d.get('score', 50):.0f}/100 - {d.get('reasoning', 'N/A')[:80]}"
            for dim, d in dimension_scores.items()
        ])

        prompt = f"""You are the chief macro economist summarizing the macro environment as of {as_of_date}.

## Five Macro Dimensions for Japanese Equities:
{scores_summary}

## Overall Macro Assessment:
Synthesize into:
1. Overall macro environment (favorable/neutral/unfavorable for Japanese equities)
2. Key equity implications for portfolio construction
3. Japan-specific considerations (BOJ, USD/JPY, global trade)

Respond with valid JSON only:
{{
  "environment": "<strongly_favorable|favorable|neutral|unfavorable|strongly_unfavorable>",
  "equity_implications": "<key implications for equity positioning>",
  "japan_specific": "<BOJ policy, yen, trade considerations>",
  "sector_preference": "<defensive|balanced|cyclical>",
  "reasoning": "<2-3 sentence synthesis>"
}}"""

        try:
            response = self._call_llm(prompt, use_fast_model=True)
            result = self._parse_json_response(response)
            return result
        except Exception as e:
            logger.error(f"Overall macro assessment failed: {e}")
            return {
                "environment": "neutral",
                "equity_implications": "Mixed macro signals",
                "reasoning": "Analysis unavailable",
            }
