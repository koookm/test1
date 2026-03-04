"""
Base Agent class for all investment analysis agents.
Implements LLM interaction, retry logic, and output validation.
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from expert_investment_team.config.settings import SystemConfig, DEFAULT_CONFIG


class BaseAgent(ABC):
    """
    Abstract base class for all investment agents.
    Provides common LLM interaction infrastructure.
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.config = config
        self.llm_config = config.llm
        self._client = None

    def _get_client(self):
        """Lazy initialization of LLM client."""
        if self._client is not None:
            return self._client

        if self.llm_config.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=self.config.anthropic_api_key
                )
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")
        elif self.llm_config.provider == "openai":
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=self.config.openai_api_key
                )
            except ImportError:
                raise ImportError("openai package required: pip install openai")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_config.provider}")

        return self._client

    def _call_llm(self, prompt: str, use_fast_model: bool = False) -> str:
        """
        Call the LLM with retry logic.
        Returns the text response.
        """
        client = self._get_client()
        model = (
            self.llm_config.fast_model
            if use_fast_model
            else self.llm_config.model
        )

        for attempt in range(self.config.agent.max_retries):
            try:
                if self.llm_config.provider == "anthropic":
                    response = client.messages.create(
                        model=model,
                        max_tokens=self.llm_config.max_tokens,
                        temperature=self.llm_config.temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return response.content[0].text

                elif self.llm_config.provider == "openai":
                    response = client.chat.completions.create(
                        model=model,
                        max_tokens=self.llm_config.max_tokens,
                        temperature=self.llm_config.temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return response.choices[0].message.content

            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < self.config.agent.max_retries - 1:
                    time.sleep(self.config.agent.retry_delay * (2 ** attempt))
                else:
                    raise

        return ""

    def _parse_json_response(self, response: str) -> Dict:
        """
        Parse JSON from LLM response.
        Handles cases where JSON is embedded in markdown code blocks.
        """
        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code blocks
        import re
        patterns = [
            r"```json\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
            r"\{.*\}",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

        logger.error(f"Failed to parse JSON from response: {response[:200]}...")
        return {}

    @abstractmethod
    def analyze(self, data: Dict) -> Dict:
        """
        Main analysis method. Each agent implements its own fine-grained tasks.

        Args:
            data: Input data dict

        Returns:
            Analysis results with score (0-100) and reasoning
        """
        pass

    def _validate_score(self, score: Any) -> float:
        """Ensure score is in valid 0-100 range."""
        try:
            s = float(score)
            return max(0.0, min(100.0, s))
        except (TypeError, ValueError):
            logger.warning(f"Invalid score value: {score}, defaulting to 50.0")
            return 50.0
