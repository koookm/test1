"""
Technical Analysis Agent
Fine-grained tasks for price action, trend, momentum, and pattern analysis.
Primary experimental target in arXiv:2602.23330.

Fine-grained tasks:
  1. Trend Analysis (MA alignment, ADX trend strength)
  2. Momentum Analysis (RSI, MACD, price momentum at multiple timeframes)
  3. Volatility Analysis (Bollinger Bands, ATR, vol regime)
  4. Volume Analysis (OBV, volume trend confirmation)
  5. Signal Synthesis (combine signals into actionable technical score)
"""

from typing import Dict
from loguru import logger
from expert_investment_team.agents.base_agent import BaseAgent
from expert_investment_team.config.settings import SystemConfig, DEFAULT_CONFIG


class TechnicalAgent(BaseAgent):
    """
    Technical Analysis Agent with fine-grained task decomposition.

    The paper identifies this as a primary experimental target because
    numerical data processing can be clearly defined with fine-grained tasks,
    making it easy to compare coarse vs fine-grained performance.
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        super().__init__(config)
        self.use_fine_grained = config.agent.use_fine_grained_tasks

    def analyze(self, data: Dict) -> Dict:
        """
        Args:
            data: Dict with 'ticker', 'indicators' (from TechnicalIndicators.get_latest_snapshot)

        Returns:
            Dict with score (0-100) and detailed technical analysis
        """
        ticker = data.get("ticker", "UNKNOWN")
        indicators = data.get("indicators", {})

        logger.info(f"[TechnicalAgent] Analyzing {ticker}...")

        if self.use_fine_grained:
            return self._fine_grained_analysis(ticker, indicators)
        else:
            return self._coarse_grained_analysis(ticker, indicators)

    def _fine_grained_analysis(self, ticker: str, indicators: Dict) -> Dict:
        """Fine-grained: 4 explicit technical sub-tasks + synthesis."""

        # Task 1: Trend Analysis
        trend_result = self._task_trend(ticker, indicators)

        # Task 2: Momentum Analysis
        momentum_result = self._task_momentum(ticker, indicators)

        # Task 3: Volatility Analysis
        volatility_result = self._task_volatility(ticker, indicators)

        # Task 4: Volume Analysis
        volume_result = self._task_volume(ticker, indicators)

        # Task 5: Signal Synthesis
        final_result = self._task_synthesize(
            ticker,
            trend=trend_result,
            momentum=momentum_result,
            volatility=volatility_result,
            volume=volume_result,
        )

        return {
            "agent": "technical",
            "ticker": ticker,
            "score": final_result.get("final_score", 50.0),
            "sub_scores": {
                "trend": trend_result.get("score", 50.0),
                "momentum": momentum_result.get("score", 50.0),
                "volatility": volatility_result.get("score", 50.0),
                "volume": volume_result.get("score", 50.0),
            },
            "signals": final_result.get("signals", []),
            "trading_bias": final_result.get("trading_bias", "neutral"),
            "reasoning": final_result.get("reasoning", ""),
            "raw_outputs": {
                "trend": trend_result,
                "momentum": momentum_result,
                "volatility": volatility_result,
                "volume": volume_result,
            },
        }

    def _task_trend(self, ticker: str, ind: Dict) -> Dict:
        """Task 1: Determine the prevailing trend direction and strength."""
        prompt = f"""You are a technical analyst conducting trend analysis for {ticker}.

## Moving Average Data:
- SMA 20: {ind.get('SMA_20', 'N/A'):.2f} | SMA 50: {ind.get('SMA_50', 'N/A'):.2f} | SMA 200: {ind.get('SMA_200', 'N/A'):.2f}
- EMA 12: {ind.get('EMA_12', 'N/A'):.2f} | EMA 26: {ind.get('EMA_26', 'N/A'):.2f} | EMA 50: {ind.get('EMA_50', 'N/A'):.2f}
- Current Price vs SMA20: {ind.get('price_vs_sma20', 'N/A'):.2f}%
- Current Price vs SMA50: {ind.get('price_vs_sma50', 'N/A'):.2f}%
- Current Price vs SMA200: {ind.get('price_vs_sma200', 'N/A'):.2f}%
- Golden Cross (recent): {ind.get('golden_cross', 0)}
- Death Cross (recent): {ind.get('death_cross', 0)}

## Trend Strength (ADX):
- ADX 14: {ind.get('ADX_14', 'N/A')}
- DI+: {ind.get('DI_plus', 'N/A')} | DI-: {ind.get('DI_minus', 'N/A')}
- Strong Trend (ADX>25): {ind.get('strong_trend', 0)}
- Trend Bullish: {ind.get('trend_bullish', 0)}

## Fine-Grained Trend Analysis Task:
Systematically evaluate the trend:

1. **MA Alignment**: Are short/medium/long-term MAs in bullish (ascending) or bearish (descending) alignment?
   - Bullish: Price > SMA20 > SMA50 > SMA200
   - Bearish: Price < SMA20 < SMA50 < SMA200
2. **Trend Strength**: Does ADX confirm a meaningful trend?
   - ADX < 20: No trend (ranging)
   - ADX 20-25: Emerging trend
   - ADX > 25: Strong trend
3. **Trend Score**: Convert to 0-100 score

Respond with valid JSON only:
{{
  "score": <0-100, where 100=strongest bullish trend>,
  "trend_direction": "<strong_bullish|bullish|neutral|bearish|strong_bearish>",
  "ma_alignment": "<fully_bullish|mostly_bullish|mixed|mostly_bearish|fully_bearish>",
  "trend_strength": "<no_trend|weak|moderate|strong>",
  "key_observations": ["<obs1>", "<obs2>"],
  "reasoning": "<explanation>"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            result["score"] = self._validate_score(result.get("score", 50))
            return result
        except Exception as e:
            logger.error(f"Trend task failed for {ticker}: {e}")
            return {"score": 50.0, "trend_direction": "neutral"}

    def _task_momentum(self, ticker: str, ind: Dict) -> Dict:
        """Task 2: Analyze price momentum across multiple timeframes."""
        prompt = f"""You are a technical analyst conducting momentum analysis for {ticker}.

## RSI Data:
- RSI 14: {ind.get('RSI_14', 'N/A')}
- RSI Oversold (<30): {ind.get('rsi_oversold', 0)}
- RSI Overbought (>70): {ind.get('rsi_overbought', 0)}
- RSI Bullish Crossover (crossed 50 upward): {ind.get('rsi_bullish', 0)}

## MACD Data:
- MACD Line: {ind.get('MACD', 'N/A')}
- MACD Signal Line: {ind.get('MACD_signal', 'N/A')}
- MACD Histogram: {ind.get('MACD_histogram', 'N/A')}
- MACD Bullish Crossover: {ind.get('macd_bullish_crossover', 0)}
- MACD Bearish Crossover: {ind.get('macd_bearish_crossover', 0)}

## Price Returns (Momentum):
- 1-Day Return: {ind.get('return_1d', 'N/A')}%
- 5-Day Return: {ind.get('return_5d', 'N/A')}%
- 20-Day Return (1M): {ind.get('return_20d', 'N/A')}%
- 60-Day Return (3M): {ind.get('return_60d', 'N/A')}%
- 120-Day Return (6M): {ind.get('return_120d', 'N/A')}%
- 240-Day Return (12M): {ind.get('return_240d', 'N/A')}%
- JT Momentum (12M-1M): {ind.get('momentum_jt', 'N/A')}

## Fine-Grained Momentum Analysis:
1. **RSI Assessment**: Is the stock oversold (buy signal) or overbought (sell signal)?
2. **MACD Assessment**: Bullish or bearish momentum based on MACD crossover and histogram
3. **Multi-Timeframe Momentum**: Is momentum consistent across 1M, 3M, 6M, 12M?
4. **Momentum Score**: Synthesize RSI + MACD + price momentum

Respond with valid JSON only:
{{
  "score": <0-100, where 100=strongest bullish momentum>,
  "rsi_signal": "<oversold_buy|neutral|overbought_sell>",
  "macd_signal": "<bullish_crossover|bullish|neutral|bearish|bearish_crossover>",
  "momentum_direction": "<strongly_positive|positive|neutral|negative|strongly_negative>",
  "momentum_consistency": "<consistent_up|mixed_positive|mixed|mixed_negative|consistent_down>",
  "key_observations": ["<obs1>", "<obs2>"],
  "reasoning": "<explanation>"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            result["score"] = self._validate_score(result.get("score", 50))
            return result
        except Exception as e:
            logger.error(f"Momentum task failed for {ticker}: {e}")
            return {"score": 50.0, "momentum_direction": "neutral"}

    def _task_volatility(self, ticker: str, ind: Dict) -> Dict:
        """Task 3: Analyze volatility regime and Bollinger Band signals."""
        prompt = f"""You are a technical analyst conducting volatility analysis for {ticker}.

## Bollinger Bands:
- BB Upper: {ind.get('BB_upper', 'N/A')}
- BB Middle (SMA20): {ind.get('BB_middle', 'N/A')}
- BB Lower: {ind.get('BB_lower', 'N/A')}
- BB Width (%): {ind.get('BB_width', 'N/A')}
- BB %B (position within bands): {ind.get('BB_pctB', 'N/A')}
- BB Squeeze (low volatility): {ind.get('bb_squeeze', 0)}
- BB Upward Breakout: {ind.get('bb_breakout_up', 0)}
- BB Downward Breakout: {ind.get('bb_breakout_down', 0)}

## ATR (Average True Range):
- ATR 14: {ind.get('ATR_14', 'N/A')}
- ATR as % of Price: {ind.get('ATR_pct', 'N/A')}%

## Fine-Grained Volatility Analysis:
1. **Volatility Regime**: Is the stock in a low-vol squeeze (potential breakout) or high-vol expansion?
2. **Bollinger Band Position**: Where is price relative to bands?
   - %B > 1.0: Price above upper band (overbought/breakout)
   - %B 0.8-1.0: Near upper band (bullish)
   - %B 0.2-0.8: Within bands (neutral)
   - %B < 0.2: Near lower band (oversold/breakdown)
3. **Risk Assessment**: Is ATR indicating elevated risk?
4. **Volatility Score for Risk-Adjusted Timing**

Respond with valid JSON only:
{{
  "score": <0-100, where 100=best volatility setup for buying>,
  "vol_regime": "<squeeze|low|normal|elevated|high>",
  "bb_position": "<above_upper|near_upper|middle|near_lower|below_lower>",
  "breakout_potential": "<high|medium|low>",
  "risk_level": "<low|moderate|high|very_high>",
  "key_observations": ["<obs1>", "<obs2>"],
  "reasoning": "<explanation>"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            result["score"] = self._validate_score(result.get("score", 50))
            return result
        except Exception as e:
            logger.error(f"Volatility task failed for {ticker}: {e}")
            return {"score": 50.0, "vol_regime": "normal"}

    def _task_volume(self, ticker: str, ind: Dict) -> Dict:
        """Task 4: Volume confirmation of price action."""
        prompt = f"""You are a technical analyst conducting volume analysis for {ticker}.

## Volume Data:
- Current Volume / 20-Day Avg: {ind.get('vol_ratio', 'N/A')}x
- High Volume Flag (>2x avg): {ind.get('high_volume', 0)}
- OBV (On-Balance Volume trend): available

## Fine-Grained Volume Analysis:
1. **Volume Confirmation**: Does volume confirm the price trend?
   - Rising price + Rising volume = Strong bullish confirmation
   - Rising price + Falling volume = Weak/suspicious rally
   - Falling price + Rising volume = Strong bearish distribution
2. **Unusual Volume**: Is there unusual volume activity suggesting institutional interest?
3. **Volume Score**

Respond with valid JSON only:
{{
  "score": <0-100, where 100=strongest bullish volume confirmation>,
  "volume_confirmation": "<strong_bullish|bullish|neutral|bearish|strong_bearish>",
  "unusual_activity": "<yes|no>",
  "institutional_signal": "<accumulation|neutral|distribution>",
  "key_observations": ["<obs1>"],
  "reasoning": "<explanation>"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            result["score"] = self._validate_score(result.get("score", 50))
            return result
        except Exception as e:
            logger.error(f"Volume task failed for {ticker}: {e}")
            return {"score": 50.0, "volume_confirmation": "neutral"}

    def _task_synthesize(
        self,
        ticker: str,
        trend: Dict,
        momentum: Dict,
        volatility: Dict,
        volume: Dict,
    ) -> Dict:
        """Task 5: Synthesize all technical signals into final score."""
        prompt = f"""You are the head technical analyst finalizing the technical view for {ticker}.

## Technical Sub-Analysis Results:

### Trend Analysis (Score: {trend.get('score', 50)}/100):
Direction: {trend.get('trend_direction', 'N/A')}
MA Alignment: {trend.get('ma_alignment', 'N/A')}
Trend Strength: {trend.get('trend_strength', 'N/A')}
{trend.get('reasoning', '')}

### Momentum Analysis (Score: {momentum.get('score', 50)}/100):
RSI Signal: {momentum.get('rsi_signal', 'N/A')}
MACD Signal: {momentum.get('macd_signal', 'N/A')}
Momentum: {momentum.get('momentum_direction', 'N/A')}
{momentum.get('reasoning', '')}

### Volatility Analysis (Score: {volatility.get('score', 50)}/100):
Vol Regime: {volatility.get('vol_regime', 'N/A')}
BB Position: {volatility.get('bb_position', 'N/A')}
Risk: {volatility.get('risk_level', 'N/A')}
{volatility.get('reasoning', '')}

### Volume Analysis (Score: {volume.get('score', 50)}/100):
Volume Confirmation: {volume.get('volume_confirmation', 'N/A')}
Institutional: {volume.get('institutional_signal', 'N/A')}
{volume.get('reasoning', '')}

## Synthesis Task:
As the head technical analyst, synthesize these signals:
1. Final technical score (0-100)
   - Trend: 35% weight, Momentum: 35%, Volume: 20%, Volatility: 10%
2. List the top 3 actionable signals
3. Overall trading bias

Respond with valid JSON only:
{{
  "final_score": <0-100>,
  "trading_bias": "<strong_buy|buy|neutral|sell|strong_sell>",
  "signals": ["<signal1>", "<signal2>", "<signal3>"],
  "time_horizon": "<short_term|medium_term|both>",
  "reasoning": "<synthesis explanation>"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            result["final_score"] = self._validate_score(result.get("final_score", 50))
            return result
        except Exception as e:
            logger.error(f"Technical synthesis failed for {ticker}: {e}")
            weighted = (
                trend.get("score", 50) * 0.35 +
                momentum.get("score", 50) * 0.35 +
                volume.get("score", 50) * 0.20 +
                volatility.get("score", 50) * 0.10
            )
            return {
                "final_score": weighted,
                "trading_bias": "neutral",
                "signals": [],
                "reasoning": "Fallback to weighted average",
            }

    def _coarse_grained_analysis(self, ticker: str, indicators: Dict) -> Dict:
        """Coarse-grained baseline for ablation study."""
        prompt = f"""Analyze the technical outlook for {ticker}.
RSI: {indicators.get('RSI_14', 'N/A')}, MACD: {indicators.get('MACD', 'N/A')},
SMA20: {indicators.get('SMA_20', 'N/A')}, Price vs SMA200: {indicators.get('price_vs_sma200', 'N/A')}%.

Provide a technical score 0-100 and brief summary.

JSON: {{"score": <number>, "trading_bias": "<bias>", "reasoning": "<text>"}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            return {
                "agent": "technical",
                "ticker": ticker,
                "score": self._validate_score(result.get("score", 50)),
                "trading_bias": result.get("trading_bias", "neutral"),
                "reasoning": result.get("reasoning", ""),
                "mode": "coarse_grained",
            }
        except Exception as e:
            logger.error(f"Coarse technical analysis failed for {ticker}: {e}")
            return {"agent": "technical", "ticker": ticker, "score": 50.0}
