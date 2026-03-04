"""
Sentiment & News Analysis Agent
Analyzes news sentiment and market perception for Japanese equities.

Tasks:
  1. News Sentiment Scoring (positive/negative/neutral)
  2. Event Detection (earnings, M&A, regulatory, macro events)
  3. Market Perception Assessment
  4. Sentiment Score Synthesis
"""

from typing import Dict, List
from loguru import logger
from expert_investment_team.agents.base_agent import BaseAgent
from expert_investment_team.config.settings import SystemConfig, DEFAULT_CONFIG


class SentimentAgent(BaseAgent):
    """
    Sentiment and News Analysis Agent.
    Processes news articles to extract investment-relevant sentiment signals.
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        super().__init__(config)

    def analyze(self, data: Dict) -> Dict:
        """
        Args:
            data: Dict with 'ticker', 'company_name', 'news_articles'

        Returns:
            Dict with score (0-100) and sentiment analysis
        """
        ticker = data.get("ticker", "UNKNOWN")
        company_name = data.get("company_name", ticker)
        news_articles = data.get("news_articles", [])

        logger.info(f"[SentimentAgent] Analyzing {ticker} with {len(news_articles)} articles...")

        if not news_articles:
            return self._no_news_result(ticker)

        # Task 1: News Sentiment
        sentiment_result = self._task_news_sentiment(ticker, company_name, news_articles)

        # Task 2: Event Detection
        event_result = self._task_event_detection(ticker, company_name, news_articles)

        # Task 3: Synthesis
        final_result = self._task_synthesize(ticker, sentiment_result, event_result)

        return {
            "agent": "sentiment",
            "ticker": ticker,
            "score": final_result.get("final_score", 50.0),
            "sub_scores": {
                "news_sentiment": sentiment_result.get("score", 50.0),
                "event_impact": event_result.get("score", 50.0),
            },
            "overall_sentiment": final_result.get("overall_sentiment", "neutral"),
            "key_events": event_result.get("detected_events", []),
            "reasoning": final_result.get("reasoning", ""),
        }

    def _task_news_sentiment(
        self, ticker: str, company_name: str, articles: List[Dict]
    ) -> Dict:
        """Task 1: Score sentiment of each news article and aggregate."""
        # Format articles for prompt (limit to top 10)
        articles_text = "\n\n".join([
            f"[Article {i+1}]\nTitle: {a.get('title', 'N/A')}\nSummary: {a.get('summary', 'N/A')[:200]}\nSource: {a.get('publisher', 'N/A')}"
            for i, a in enumerate(articles[:10])
        ])

        prompt = f"""You are a sentiment analyst processing news for {company_name} ({ticker}), a Japanese equity.

## Recent News Articles:
{articles_text}

## Fine-Grained Sentiment Task:
For each article, assess:
1. **Sentiment Direction**: Positive, Negative, or Neutral for the stock
2. **Materiality**: How material is this news for stock price?
3. **Aggregate Sentiment**: Overall news sentiment weighted by materiality

Consider:
- Earnings beats/misses
- Management changes
- Product launches/failures
- Regulatory news
- M&A activity
- Macro/industry headwinds

Respond with valid JSON only:
{{
  "score": <0-100, where 100=most bullish news sentiment>,
  "sentiment_breakdown": {{
    "positive_count": <number>,
    "negative_count": <number>,
    "neutral_count": <number>
  }},
  "weighted_sentiment": "<strongly_positive|positive|neutral|negative|strongly_negative>",
  "most_impactful_headline": "<headline>",
  "sentiment_trend": "<improving|stable|deteriorating>",
  "reasoning": "<explanation>"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            result["score"] = self._validate_score(result.get("score", 50))
            return result
        except Exception as e:
            logger.error(f"News sentiment task failed for {ticker}: {e}")
            return {"score": 50.0, "weighted_sentiment": "neutral"}

    def _task_event_detection(
        self, ticker: str, company_name: str, articles: List[Dict]
    ) -> Dict:
        """Task 2: Detect specific events that could catalyze price movement."""
        articles_text = "\n".join([
            f"- {a.get('title', 'N/A')}"
            for a in articles[:15]
        ])

        prompt = f"""You are an event analyst for {company_name} ({ticker}).

## News Headlines:
{articles_text}

## Event Detection Task:
Identify any of these event types:
1. **Earnings Events**: Upcoming earnings, guidance changes, estimate revisions
2. **Corporate Events**: M&A, management changes, restructuring, buybacks
3. **Product/Business**: New products, contracts, market share shifts
4. **Regulatory**: Government policy, regulatory approvals/rejections
5. **Macro Sensitivity**: USD/JPY movements, BOJ policy sensitivity

For each detected event, assess its investment impact (positive/negative/neutral).

Respond with valid JSON only:
{{
  "score": <0-100, where 100=strongest positive catalyst detected>,
  "detected_events": [
    {{"type": "<event_type>", "description": "<brief>", "impact": "<positive|negative|neutral>", "magnitude": "<low|medium|high>"}}
  ],
  "catalyst_present": <true|false>,
  "upcoming_catalyst": "<description or none>",
  "event_risk": "<low|moderate|high>",
  "reasoning": "<explanation>"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            result["score"] = self._validate_score(result.get("score", 50))
            return result
        except Exception as e:
            logger.error(f"Event detection failed for {ticker}: {e}")
            return {"score": 50.0, "detected_events": [], "catalyst_present": False}

    def _task_synthesize(
        self, ticker: str, sentiment: Dict, events: Dict
    ) -> Dict:
        """Task 3: Synthesize sentiment and events into final score."""
        prompt = f"""Synthesize sentiment analysis for {ticker}.

Sentiment Score: {sentiment.get('score', 50)}/100
News Sentiment: {sentiment.get('weighted_sentiment', 'neutral')}
Trend: {sentiment.get('sentiment_trend', 'stable')}

Event Score: {events.get('score', 50)}/100
Catalyst Present: {events.get('catalyst_present', False)}
Event Risk: {events.get('event_risk', 'moderate')}

Compute final sentiment score weighted:
- News Sentiment: 60%, Events: 40%

JSON: {{
  "final_score": <0-100>,
  "overall_sentiment": "<bullish|neutral|bearish>",
  "conviction": "<low|medium|high>",
  "reasoning": "<1-2 sentences>"
}}"""

        try:
            response = self._call_llm(prompt, use_fast_model=True)
            result = self._parse_json_response(response)
            result["final_score"] = self._validate_score(result.get("final_score", 50))
            return result
        except Exception as e:
            logger.error(f"Sentiment synthesis failed for {ticker}: {e}")
            weighted = sentiment.get("score", 50) * 0.6 + events.get("score", 50) * 0.4
            return {
                "final_score": weighted,
                "overall_sentiment": "neutral",
                "reasoning": "Fallback to weighted average",
            }

    def _no_news_result(self, ticker: str) -> Dict:
        """Return neutral result when no news is available."""
        return {
            "agent": "sentiment",
            "ticker": ticker,
            "score": 50.0,
            "overall_sentiment": "neutral",
            "key_events": [],
            "reasoning": "No news articles available for analysis",
        }
