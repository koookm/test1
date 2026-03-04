"""
Quantitative Analysis Agent
Systematic factor-based analysis.
Second primary experimental target in arXiv:2602.23330.

Fine-grained tasks:
  1. Momentum Factor Analysis (1M, 3M, 6M, 12M, JT momentum)
  2. Risk Factor Analysis (volatility, beta, max drawdown)
  3. Liquidity Factor Analysis (volume, float, market impact)
  4. Cross-Sectional Factor Ranking (rank vs universe)
  5. Quantitative Score Synthesis
"""

from typing import Dict, List
from loguru import logger
from expert_investment_team.agents.base_agent import BaseAgent
from expert_investment_team.config.settings import SystemConfig, DEFAULT_CONFIG


class QuantitativeAgent(BaseAgent):
    """
    Quantitative Analysis Agent using systematic factor analysis.

    Processes numerical data through fine-grained tasks to produce
    factor-based investment signals that directly feed into portfolio construction.
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        super().__init__(config)
        self.use_fine_grained = config.agent.use_fine_grained_tasks

    def analyze(self, data: Dict) -> Dict:
        """
        Args:
            data: Dict with 'ticker', 'factors' (from QuantitativeFactors.compute_factors),
                  'universe_factors' (cross-sectional rankings)

        Returns:
            Dict with score (0-100) and quantitative factor analysis
        """
        ticker = data.get("ticker", "UNKNOWN")
        factors = data.get("factors", {})
        universe_factors = data.get("universe_factors", {})

        logger.info(f"[QuantitativeAgent] Analyzing {ticker}...")

        if self.use_fine_grained:
            return self._fine_grained_analysis(ticker, factors, universe_factors)
        else:
            return self._coarse_grained_analysis(ticker, factors)

    def _fine_grained_analysis(
        self, ticker: str, factors: Dict, universe_factors: Dict
    ) -> Dict:
        """Fine-grained: 4 factor sub-tasks + synthesis."""

        # Compute cross-sectional rankings
        rankings = self._compute_rankings(ticker, factors, universe_factors)

        # Task 1: Momentum Factors
        momentum_result = self._task_momentum_factors(ticker, factors, rankings)

        # Task 2: Risk Factors
        risk_result = self._task_risk_factors(ticker, factors, rankings)

        # Task 3: Liquidity Factors
        liquidity_result = self._task_liquidity_factors(ticker, factors, rankings)

        # Task 4: Final Synthesis
        final_result = self._task_synthesize(
            ticker,
            momentum=momentum_result,
            risk=risk_result,
            liquidity=liquidity_result,
            rankings=rankings,
        )

        return {
            "agent": "quantitative",
            "ticker": ticker,
            "score": final_result.get("final_score", 50.0),
            "sub_scores": {
                "momentum": momentum_result.get("score", 50.0),
                "risk": risk_result.get("score", 50.0),
                "liquidity": liquidity_result.get("score", 50.0),
            },
            "rankings": rankings,
            "factor_signals": final_result.get("factor_signals", []),
            "reasoning": final_result.get("reasoning", ""),
            "raw_outputs": {
                "momentum": momentum_result,
                "risk": risk_result,
                "liquidity": liquidity_result,
            },
        }

    def _compute_rankings(
        self, ticker: str, factors: Dict, universe_factors: Dict
    ) -> Dict:
        """
        Compute cross-sectional percentile rankings of this stock vs the universe.
        Returns percentile ranks (0=worst, 100=best).
        """
        rankings = {}
        if not universe_factors:
            return rankings

        factor_keys = [
            "momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m",
            "momentum_jt", "vol_1m", "vol_3m", "beta", "correlation_1y",
        ]

        for key in factor_keys:
            stock_val = factors.get(key)
            if stock_val is None:
                continue

            universe_vals = [
                v.get(key) for v in universe_factors.values()
                if v.get(key) is not None
            ]
            if not universe_vals:
                continue

            universe_vals_sorted = sorted(universe_vals)
            n = len(universe_vals_sorted)
            rank = sum(1 for v in universe_vals_sorted if v <= stock_val)
            percentile = rank / n * 100

            # For volatility and beta, lower = better (flip the rank)
            if key in ("vol_1m", "vol_3m", "beta"):
                percentile = 100 - percentile

            rankings[f"{key}_rank"] = round(percentile, 1)

        return rankings

    def _task_momentum_factors(
        self, ticker: str, factors: Dict, rankings: Dict
    ) -> Dict:
        """Task 1: Evaluate multi-horizon momentum factors."""
        prompt = f"""You are a quantitative analyst evaluating momentum factors for {ticker}.

## Momentum Factor Data:
- 1-Month Return: {factors.get('momentum_1m', 'N/A')}
- 3-Month Return: {factors.get('momentum_3m', 'N/A')}
- 6-Month Return: {factors.get('momentum_6m', 'N/A')}
- 12-Month Return: {factors.get('momentum_12m', 'N/A')}
- Jegadeesh-Titman Momentum (12M-1M): {factors.get('momentum_jt', 'N/A')}
- Short-Term Reversal (5D): {factors.get('short_term_reversal', 'N/A')}

## Cross-Sectional Rankings vs TOPIX 100 Universe:
- 1M Momentum Percentile: {rankings.get('momentum_1m_rank', 'N/A')}th
- 3M Momentum Percentile: {rankings.get('momentum_3m_rank', 'N/A')}th
- 6M Momentum Percentile: {rankings.get('momentum_6m_rank', 'N/A')}th
- 12M Momentum Percentile: {rankings.get('momentum_12m_rank', 'N/A')}th
- JT Momentum Percentile: {rankings.get('momentum_jt_rank', 'N/A')}th

## Fine-Grained Momentum Factor Task:
Evaluate momentum using systematic factor analysis:

1. **Cross-Sectional Momentum**: Is the stock a top or bottom momentum name in the universe?
   - Top quintile (>80th percentile) = Strong buy signal
   - Bottom quintile (<20th percentile) = Strong sell signal
2. **Multi-Horizon Consistency**: Is momentum consistent across 1M, 3M, 6M, 12M?
3. **Reversal Risk**: Is the 1M momentum too high (reversal risk)?
4. **JT Factor Signal**: The academic momentum signal (12M-1M) for monthly rebalancing

Respond with valid JSON only:
{{
  "score": <0-100, where 100=strongest momentum buy signal>,
  "jt_factor_signal": "<strong_buy|buy|neutral|sell|strong_sell>",
  "momentum_consistency": "<consistent|mostly_consistent|mixed|inconsistent>",
  "reversal_risk": "<low|moderate|high>",
  "cross_sectional_rank": "<top_quintile|second_quintile|middle|fourth_quintile|bottom_quintile>",
  "key_observations": ["<obs1>", "<obs2>"],
  "reasoning": "<explanation>"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            result["score"] = self._validate_score(result.get("score", 50))
            return result
        except Exception as e:
            logger.error(f"Momentum factor task failed for {ticker}: {e}")
            return {"score": 50.0}

    def _task_risk_factors(
        self, ticker: str, factors: Dict, rankings: Dict
    ) -> Dict:
        """Task 2: Evaluate risk factor profile."""
        prompt = f"""You are a quantitative risk analyst evaluating risk factors for {ticker}.

## Risk Factor Data:
- Realized Volatility (1M annualized): {factors.get('vol_1m', 'N/A')}
- Realized Volatility (3M annualized): {factors.get('vol_3m', 'N/A')}
- Realized Volatility (12M annualized): {factors.get('vol_12m', 'N/A')}
- Beta vs TOPIX: {factors.get('beta', 'N/A')}
- Correlation with TOPIX (1Y): {factors.get('correlation_1y', 'N/A')}
- Max Drawdown (1Y): {factors.get('max_drawdown_1y', 'N/A')}
- Current Drawdown: {factors.get('current_drawdown', 'N/A')}

## Cross-Sectional Risk Rankings:
- Volatility Rank (lower vol = better): {rankings.get('vol_1m_rank', 'N/A')}th percentile
- Beta Rank (lower beta = better): {rankings.get('beta_rank', 'N/A')}th percentile

## Fine-Grained Risk Factor Task:
Conduct systematic risk assessment:

1. **Absolute Volatility**: Is the stock's volatility appropriate for a large-cap Japanese equity?
   - TOPIX 100 typical annualized vol: 15-25%
   - High vol for large-cap: >30%
2. **Beta Analysis**: Market sensitivity assessment
   - Beta < 0.8: Defensive (low market risk)
   - Beta 0.8-1.2: Market-like
   - Beta > 1.2: Aggressive (high market risk)
3. **Drawdown Risk**: How severe has the drawdown been?
4. **Risk Score** (for risk-adjusted return optimization)

Respond with valid JSON only:
{{
  "score": <0-100, where 100=best risk profile for long position>,
  "vol_level": "<very_low|low|normal|high|very_high>",
  "beta_class": "<defensive|market_like|aggressive>",
  "drawdown_severity": "<minimal|mild|moderate|severe>",
  "risk_adjusted_rank": "<low_risk|medium_risk|high_risk>",
  "idiosyncratic_risk": "<low|medium|high>",
  "key_observations": ["<obs1>", "<obs2>"],
  "reasoning": "<explanation>"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            result["score"] = self._validate_score(result.get("score", 50))
            return result
        except Exception as e:
            logger.error(f"Risk factor task failed for {ticker}: {e}")
            return {"score": 50.0}

    def _task_liquidity_factors(
        self, ticker: str, factors: Dict, rankings: Dict
    ) -> Dict:
        """Task 3: Evaluate liquidity and market impact factors."""
        prompt = f"""You are a quantitative analyst evaluating liquidity factors for {ticker}.

## Liquidity Factor Data:
- Average Daily Volume (20D): {factors.get('avg_daily_volume_20d', 'N/A')}
- Liquidity Ratio (20D/60D volume): {factors.get('liquidity_ratio', 'N/A')}

## Fine-Grained Liquidity Task:
Evaluate trading liquidity for institutional portfolio construction:

1. **Absolute Liquidity**: Can we trade meaningful size without market impact?
   - TOPIX 100 stocks generally have high liquidity
2. **Liquidity Trend**: Is volume increasing or decreasing?
3. **Market Impact Risk**: For a 10-stock concentrated portfolio, is liquidity sufficient?
4. **Liquidity Score**

Respond with valid JSON only:
{{
  "score": <0-100, where 100=best liquidity>,
  "liquidity_level": "<very_low|low|adequate|good|excellent>",
  "market_impact_risk": "<low|moderate|high>",
  "volume_trend": "<declining|stable|increasing>",
  "tradability": "<restricted|limited|adequate|easy|very_easy>",
  "key_observations": ["<obs1>"],
  "reasoning": "<explanation>"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            result["score"] = self._validate_score(result.get("score", 50))
            return result
        except Exception as e:
            logger.error(f"Liquidity task failed for {ticker}: {e}")
            return {"score": 70.0, "liquidity_level": "adequate"}  # TOPIX100 default

    def _task_synthesize(
        self,
        ticker: str,
        momentum: Dict,
        risk: Dict,
        liquidity: Dict,
        rankings: Dict,
    ) -> Dict:
        """Task 4: Synthesize quantitative factors into final quant score."""
        prompt = f"""You are the head quantitative analyst finalizing the factor analysis for {ticker}.

## Quantitative Factor Results:

### Momentum Factors (Score: {momentum.get('score', 50)}/100):
JT Signal: {momentum.get('jt_factor_signal', 'N/A')}
Consistency: {momentum.get('momentum_consistency', 'N/A')}
Reversal Risk: {momentum.get('reversal_risk', 'N/A')}
Cross-Sectional Rank: {momentum.get('cross_sectional_rank', 'N/A')}

### Risk Factors (Score: {risk.get('score', 50)}/100):
Volatility: {risk.get('vol_level', 'N/A')}
Beta: {risk.get('beta_class', 'N/A')}
Drawdown: {risk.get('drawdown_severity', 'N/A')}

### Liquidity Factors (Score: {liquidity.get('score', 50)}/100):
Liquidity: {liquidity.get('liquidity_level', 'N/A')}
Tradability: {liquidity.get('tradability', 'N/A')}

## Rankings vs Universe:
{self._format_rankings(rankings)}

## Synthesis Task:
Synthesize into final quantitative attractiveness score:
- Momentum: 50% weight (key alpha signal)
- Risk: 35% weight (critical for Sharpe optimization)
- Liquidity: 15% weight (implementation constraint)

Respond with valid JSON only:
{{
  "final_score": <0-100>,
  "factor_signals": ["<signal1>", "<signal2>", "<signal3>"],
  "sharpe_outlook": "<positive|neutral|negative>",
  "quant_view": "<strong_long|long|neutral|short|strong_short>",
  "reasoning": "<explanation>"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            result["final_score"] = self._validate_score(result.get("final_score", 50))
            return result
        except Exception as e:
            logger.error(f"Quant synthesis failed for {ticker}: {e}")
            weighted = (
                momentum.get("score", 50) * 0.50 +
                risk.get("score", 50) * 0.35 +
                liquidity.get("score", 50) * 0.15
            )
            return {
                "final_score": weighted,
                "factor_signals": [],
                "quant_view": "neutral",
                "reasoning": "Fallback to weighted average",
            }

    def _coarse_grained_analysis(self, ticker: str, factors: Dict) -> Dict:
        """Coarse-grained baseline for ablation study."""
        prompt = f"""Analyze quantitative factors for {ticker}.
Momentum 12M: {factors.get('momentum_12m', 'N/A')}, Volatility: {factors.get('vol_1m', 'N/A')},
Beta: {factors.get('beta', 'N/A')}.

Provide a quant score 0-100.
JSON: {{"score": <number>, "quant_view": "<view>", "reasoning": "<text>"}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            return {
                "agent": "quantitative",
                "ticker": ticker,
                "score": self._validate_score(result.get("score", 50)),
                "quant_view": result.get("quant_view", "neutral"),
                "reasoning": result.get("reasoning", ""),
                "mode": "coarse_grained",
            }
        except Exception as e:
            logger.error(f"Coarse quant analysis failed for {ticker}: {e}")
            return {"agent": "quantitative", "ticker": ticker, "score": 50.0}

    def _format_rankings(self, rankings: Dict) -> str:
        lines = []
        for key, val in rankings.items():
            lines.append(f"- {key}: {val}th percentile")
        return "\n".join(lines) if lines else "No ranking data available"
