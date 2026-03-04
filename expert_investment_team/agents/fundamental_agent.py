"""
Fundamental Analysis Agent
Analyzes financial statements, valuation, quality, and growth.
Implements fine-grained task decomposition as per paper arXiv:2602.23330.

Fine-grained tasks:
  1. Valuation Analysis (P/E, P/B, EV/EBITDA, DCF relative assessment)
  2. Profitability & Quality Analysis (ROE, ROA, margins, Piotroski F-Score)
  3. Growth Analysis (revenue growth, earnings growth trends)
  4. Balance Sheet Strength (debt ratios, liquidity, working capital)
  5. Dividend & Shareholder Returns (yield, payout ratio, buybacks)
  6. Final Score Synthesis
"""

from typing import Dict
from loguru import logger
from expert_investment_team.agents.base_agent import BaseAgent
from expert_investment_team.config.settings import SystemConfig, DEFAULT_CONFIG


FUNDAMENTAL_SYSTEM_PROMPT = """You are an expert fundamental analyst at a top-tier institutional investment firm specializing in Japanese equities.

Your role is to conduct rigorous, professional-grade fundamental analysis following the workflows of real-world buy-side analysts. You provide evidence-based, quantitative assessments with clear investment theses.

Always respond in valid JSON format as specified."""


class FundamentalAgent(BaseAgent):
    """
    Fundamental Analysis Agent with fine-grained task decomposition.

    Performs 5 distinct sub-tasks before generating a final investment score,
    mirroring the analytical workflow of professional buy-side analysts.
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        super().__init__(config)
        self.use_fine_grained = config.agent.use_fine_grained_tasks

    def analyze(self, data: Dict) -> Dict:
        """
        Orchestrate all fundamental analysis sub-tasks.

        Args:
            data: Dict containing 'ticker', 'financials', 'sector_peers'

        Returns:
            Dict with score (0-100), sub-scores, reasoning, investment_thesis
        """
        ticker = data.get("ticker", "UNKNOWN")
        financials = data.get("financials", {})
        sector_peers = data.get("sector_peers", {})

        logger.info(f"[FundamentalAgent] Analyzing {ticker}...")

        if self.use_fine_grained:
            return self._fine_grained_analysis(ticker, financials, sector_peers)
        else:
            return self._coarse_grained_analysis(ticker, financials)

    def _fine_grained_analysis(
        self, ticker: str, financials: Dict, sector_peers: Dict
    ) -> Dict:
        """
        Fine-grained approach: 5 explicit sub-tasks with detailed prompts.
        This is the paper's key contribution for fundamental analysis.
        """
        info = financials.get("info", {})

        # === Task 1: Valuation Analysis ===
        valuation_result = self._task_valuation(ticker, info, sector_peers)

        # === Task 2: Profitability & Quality Analysis ===
        quality_result = self._task_quality(ticker, info, financials)

        # === Task 3: Growth Analysis ===
        growth_result = self._task_growth(ticker, info, financials)

        # === Task 4: Balance Sheet Analysis ===
        balance_result = self._task_balance_sheet(ticker, info, financials)

        # === Task 5: Final Score Synthesis ===
        final_result = self._task_synthesize(
            ticker,
            valuation=valuation_result,
            quality=quality_result,
            growth=growth_result,
            balance=balance_result,
        )

        return {
            "agent": "fundamental",
            "ticker": ticker,
            "score": final_result.get("final_score", 50.0),
            "sub_scores": {
                "valuation": valuation_result.get("score", 50.0),
                "quality": quality_result.get("score", 50.0),
                "growth": growth_result.get("score", 50.0),
                "balance_sheet": balance_result.get("score", 50.0),
            },
            "investment_thesis": final_result.get("investment_thesis", ""),
            "key_risks": final_result.get("key_risks", []),
            "reasoning": final_result.get("reasoning", ""),
            "raw_outputs": {
                "valuation": valuation_result,
                "quality": quality_result,
                "growth": growth_result,
                "balance": balance_result,
            },
        }

    def _task_valuation(self, ticker: str, info: Dict, peers: Dict) -> Dict:
        """Task 1: Assess whether the stock is overvalued, fairly valued, or undervalued."""
        prompt = f"""You are conducting valuation analysis for {ticker}, a Japanese equity.

## Stock Valuation Data:
- P/E Ratio (Trailing): {info.get('pe_ratio', 'N/A')}
- Forward P/E: {info.get('forward_pe', 'N/A')}
- Price/Book Ratio: {info.get('pb_ratio', 'N/A')}
- Price/Sales Ratio: {info.get('ps_ratio', 'N/A')}
- EV/EBITDA: {info.get('ev_ebitda', 'N/A')}
- Market Cap: {info.get('market_cap', 'N/A')}
- Enterprise Value: {info.get('enterprise_value', 'N/A')}
- Sector: {info.get('sector', 'N/A')}
- Industry: {info.get('industry', 'N/A')}
- 52-Week High: {info.get('52w_high', 'N/A')}
- 52-Week Low: {info.get('52w_low', 'N/A')}

## Peer Comparison (Sector Averages):
{self._format_peers(peers)}

## Fine-Grained Valuation Task:
Conduct a comprehensive valuation analysis following these specific steps:

1. **Absolute Valuation**: Assess each multiple (P/E, P/B, EV/EBITDA) against historical norms for Japanese equities
2. **Relative Valuation**: Compare all multiples to sector peers
3. **Composite Valuation Score**: Weight the metrics appropriately
4. **Valuation Signal**: Is the stock cheap, fair, or expensive?

Respond with valid JSON only:
{{
  "score": <0-100, where 100=extremely undervalued/attractive>,
  "pe_assessment": "<cheap|fair|expensive|N/A>",
  "pb_assessment": "<cheap|fair|expensive|N/A>",
  "ev_ebitda_assessment": "<cheap|fair|expensive|N/A>",
  "vs_peers": "<discount|inline|premium>",
  "discount_premium_pct": <estimated % vs peers, positive=discount>,
  "key_observations": ["<obs1>", "<obs2>"],
  "reasoning": "<2-3 sentence explanation>"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            result["score"] = self._validate_score(result.get("score", 50))
            return result
        except Exception as e:
            logger.error(f"Valuation task failed for {ticker}: {e}")
            return {"score": 50.0, "reasoning": "Analysis unavailable"}

    def _task_quality(self, ticker: str, info: Dict, financials: Dict) -> Dict:
        """Task 2: Assess profitability quality using systematic scoring."""
        derived = financials.get("derived_metrics", {})
        piotroski = sum([
            derived.get("piotroski_roa_positive", 0),
            derived.get("piotroski_low_leverage", 0),
            derived.get("piotroski_high_current_ratio", 0),
            derived.get("piotroski_positive_gross_margin", 0),
        ])

        prompt = f"""You are conducting profitability and quality analysis for {ticker}.

## Financial Quality Metrics:
- Return on Equity (ROE): {info.get('roe', 'N/A')}
- Return on Assets (ROA): {info.get('roa', 'N/A')}
- Gross Margin: {info.get('gross_margin', 'N/A')}
- Operating Margin: {info.get('operating_margin', 'N/A')}
- Net Profit Margin: {info.get('net_margin', 'N/A')}
- Piotroski F-Score (partial, 4 components): {piotroski}/4
- Sector: {info.get('sector', 'N/A')}

## Fine-Grained Quality Task:
Evaluate profitability following these explicit steps:

1. **Return Quality**: Are ROE and ROA above sector averages for Japanese firms?
   - Japan large-cap typical ROE: 8-12%, excellent: >15%
   - Japan large-cap typical ROA: 3-6%, excellent: >8%
2. **Margin Assessment**: Evaluate gross/operating/net margins for sustainability
3. **Quality Composite**: Synthesize into a quality score

Respond with valid JSON only:
{{
  "score": <0-100, where 100=highest quality>,
  "roe_assessment": "<poor|average|good|excellent>",
  "roa_assessment": "<poor|average|good|excellent>",
  "margin_quality": "<poor|average|good|excellent>",
  "piotroski_score": {piotroski},
  "quality_level": "<low|medium|high|very_high>",
  "key_observations": ["<obs1>", "<obs2>"],
  "reasoning": "<2-3 sentence explanation>"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            result["score"] = self._validate_score(result.get("score", 50))
            return result
        except Exception as e:
            logger.error(f"Quality task failed for {ticker}: {e}")
            return {"score": 50.0, "reasoning": "Analysis unavailable"}

    def _task_growth(self, ticker: str, info: Dict, financials: Dict) -> Dict:
        """Task 3: Analyze revenue and earnings growth trajectory."""
        derived = financials.get("derived_metrics", {})

        prompt = f"""You are conducting growth analysis for {ticker}.

## Growth Metrics:
- Revenue Growth (YoY): {info.get('revenue_growth', 'N/A')}
- Earnings Growth (YoY): {info.get('earnings_growth', 'N/A')}
- Revenue Trend QoQ: {derived.get('revenue_trend_qoq', 'N/A')}
- Earnings Trend QoQ: {derived.get('earnings_trend_qoq', 'N/A')}
- Forward P/E (implies growth expectations): {info.get('forward_pe', 'N/A')}
- Industry: {info.get('industry', 'N/A')}

## Fine-Grained Growth Task:
Analyze growth following these steps:

1. **Revenue Growth Quality**: Is growth organic? Sustainable?
2. **Earnings Growth**: Are margins expanding or compressing?
3. **Growth Trend**: Accelerating, stable, or decelerating?
4. **Growth-Adjusted Valuation**: Is growth priced fairly (implicit PEG assessment)?

Respond with valid JSON only:
{{
  "score": <0-100, where 100=exceptional growth outlook>,
  "revenue_growth_assessment": "<declining|slow|moderate|strong|exceptional>",
  "earnings_growth_assessment": "<declining|slow|moderate|strong|exceptional>",
  "growth_trend": "<decelerating|stable|accelerating>",
  "growth_sustainability": "<low|medium|high>",
  "key_observations": ["<obs1>", "<obs2>"],
  "reasoning": "<2-3 sentence explanation>"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            result["score"] = self._validate_score(result.get("score", 50))
            return result
        except Exception as e:
            logger.error(f"Growth task failed for {ticker}: {e}")
            return {"score": 50.0, "reasoning": "Analysis unavailable"}

    def _task_balance_sheet(self, ticker: str, info: Dict, financials: Dict) -> Dict:
        """Task 4: Assess financial health and balance sheet strength."""
        prompt = f"""You are conducting balance sheet analysis for {ticker}.

## Balance Sheet Metrics:
- Debt/Equity Ratio: {info.get('debt_to_equity', 'N/A')}
- Current Ratio: {info.get('current_ratio', 'N/A')}
- Quick Ratio: {info.get('quick_ratio', 'N/A')}
- Dividend Yield: {info.get('dividend_yield', 'N/A')}
- Float Shares: {info.get('float_shares', 'N/A')}
- Average Daily Volume: {info.get('avg_volume', 'N/A')}

## Fine-Grained Balance Sheet Task:
Evaluate financial health following these steps:

1. **Leverage Assessment**: Is debt level manageable? Compare to Japanese corporate norms
   - Japan corporates often carry conservative D/E < 1.0
2. **Liquidity Assessment**: Can the company meet short-term obligations?
3. **Capital Allocation**: Dividend policy signals management confidence
4. **Overall Financial Health**: Synthesize into financial health score

Respond with valid JSON only:
{{
  "score": <0-100, where 100=strongest financial health>,
  "leverage_level": "<very_low|low|moderate|high|very_high>",
  "liquidity_level": "<poor|adequate|good|excellent>",
  "capital_allocation_quality": "<poor|average|good|excellent>",
  "financial_risk": "<low|medium|high>",
  "key_observations": ["<obs1>", "<obs2>"],
  "reasoning": "<2-3 sentence explanation>"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            result["score"] = self._validate_score(result.get("score", 50))
            return result
        except Exception as e:
            logger.error(f"Balance sheet task failed for {ticker}: {e}")
            return {"score": 50.0, "reasoning": "Analysis unavailable"}

    def _task_synthesize(
        self,
        ticker: str,
        valuation: Dict,
        quality: Dict,
        growth: Dict,
        balance: Dict,
    ) -> Dict:
        """Task 5: Synthesize all sub-analyses into a final fundamental score."""
        prompt = f"""You are the lead fundamental analyst synthesizing all research for {ticker}.

## Sub-Analysis Results:

### 1. Valuation Analysis (Score: {valuation.get('score', 50)}/100):
{valuation.get('reasoning', 'No data')}
Key: P/E={valuation.get('pe_assessment','N/A')}, P/B={valuation.get('pb_assessment','N/A')}, vs Peers={valuation.get('vs_peers','N/A')}

### 2. Quality Analysis (Score: {quality.get('score', 50)}/100):
{quality.get('reasoning', 'No data')}
Quality Level: {quality.get('quality_level', 'N/A')}

### 3. Growth Analysis (Score: {growth.get('score', 50)}/100):
{growth.get('reasoning', 'No data')}
Growth Trend: {growth.get('growth_trend', 'N/A')}

### 4. Balance Sheet Analysis (Score: {balance.get('score', 50)}/100):
{balance.get('reasoning', 'No data')}
Financial Risk: {balance.get('financial_risk', 'N/A')}

## Synthesis Task:
As the senior analyst, synthesize these findings into:
1. A weighted final fundamental attractiveness score (0-100)
   - Valuation: 30% weight, Quality: 30%, Growth: 25%, Balance Sheet: 15%
2. A concise investment thesis (2-3 sentences)
3. Top 3 key risks

Respond with valid JSON only:
{{
  "final_score": <0-100>,
  "investment_thesis": "<2-3 sentence thesis>",
  "key_risks": ["<risk1>", "<risk2>", "<risk3>"],
  "overall_view": "<strong_buy|buy|neutral|sell|strong_sell>",
  "reasoning": "<synthesis explanation>"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            result["final_score"] = self._validate_score(result.get("final_score", 50))
            return result
        except Exception as e:
            logger.error(f"Synthesis task failed for {ticker}: {e}")
            weighted_score = (
                valuation.get("score", 50) * 0.30 +
                quality.get("score", 50) * 0.30 +
                growth.get("score", 50) * 0.25 +
                balance.get("score", 50) * 0.15
            )
            return {
                "final_score": weighted_score,
                "investment_thesis": "Synthesis unavailable due to LLM error",
                "key_risks": [],
                "reasoning": "Fallback to weighted average",
            }

    def _coarse_grained_analysis(self, ticker: str, financials: Dict) -> Dict:
        """
        Coarse-grained baseline (used for ablation comparison).
        Single prompt with high-level instruction.
        """
        info = financials.get("info", {})
        prompt = f"""Analyze the fundamental investment attractiveness of {ticker} based on:
P/E: {info.get('pe_ratio')}, P/B: {info.get('pb_ratio')}, ROE: {info.get('roe')},
Revenue Growth: {info.get('revenue_growth')}, Debt/Equity: {info.get('debt_to_equity')}

Provide a score 0-100 and brief thesis.

JSON: {{"score": <number>, "investment_thesis": "<text>", "reasoning": "<text>"}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            return {
                "agent": "fundamental",
                "ticker": ticker,
                "score": self._validate_score(result.get("score", 50)),
                "investment_thesis": result.get("investment_thesis", ""),
                "reasoning": result.get("reasoning", ""),
                "mode": "coarse_grained",
            }
        except Exception as e:
            logger.error(f"Coarse fundamental analysis failed for {ticker}: {e}")
            return {"agent": "fundamental", "ticker": ticker, "score": 50.0}

    def _format_peers(self, peers: Dict) -> str:
        """Format peer comparison data for prompt."""
        if not peers:
            return "Sector peer data not available"
        lines = []
        for metric, value in peers.items():
            lines.append(f"- Sector Avg {metric}: {value}")
        return "\n".join(lines) if lines else "No peer data available"
