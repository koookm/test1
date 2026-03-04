"""
Base Agent class for all investment analysis agents.
Implements LLM interaction, retry logic, and output validation.

지원 provider:
  "anthropic_oauth" - Anthropic OAuth access token (1안, Claude Code 계정)
  "anthropic"       - Anthropic API key
  "gemini"          - Google Gemini API (2안)
  "openai"          - OpenAI API
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from loguru import logger

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

        provider = self.llm_config.provider

        # === 1안: Anthropic OAuth (access token) ===
        if provider == "anthropic_oauth":
            self._client = self._init_anthropic_oauth()

        # === Anthropic API Key ===
        elif provider == "anthropic":
            self._client = self._init_anthropic_apikey()

        # === 2안: Google Gemini ===
        elif provider == "gemini":
            self._client = self._init_gemini()

        # === OpenAI ===
        elif provider == "openai":
            self._client = self._init_openai()

        else:
            raise ValueError(
                f"Unsupported provider: '{provider}'. "
                "Choose from: anthropic_oauth, anthropic, gemini, openai"
            )

        return self._client

    # ------------------------------------------------------------------
    # Client initializers
    # ------------------------------------------------------------------

    def _init_anthropic_oauth(self):
        """
        1안: Anthropic OAuth access token.

        토큰 탐색 순서:
          1) config.anthropic_access_token (ANTHROPIC_ACCESS_TOKEN env)
          2) Claude Code 세션 토큰 (~/.claude/credentials.json 또는 keyring)
          3) 없으면 API Key로 자동 fallback
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        access_token = self.config.anthropic_access_token

        # Claude Code 로컬 크리덴셜에서 자동 로드 시도
        if not access_token:
            access_token = self._load_claude_code_token()

        if access_token:
            logger.info("Anthropic: OAuth access token 사용")
            # Anthropic SDK는 api_key 파라미터로 access token도 수용
            return anthropic.Anthropic(api_key=access_token)

        # fallback → API Key
        if self.config.anthropic_api_key:
            logger.info("Anthropic OAuth 토큰 없음 → API Key로 fallback")
            return anthropic.Anthropic(api_key=self.config.anthropic_api_key)

        raise ValueError(
            "Anthropic 인증 정보 없음.\n"
            "방법 1 (OAuth): export ANTHROPIC_ACCESS_TOKEN=<Claude Code 토큰>\n"
            "방법 2 (API Key): export ANTHROPIC_API_KEY=<API 키>"
        )

    def _init_anthropic_apikey(self):
        """Anthropic standard API key."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        if not self.config.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY 환경변수를 설정해주세요.")

        return anthropic.Anthropic(api_key=self.config.anthropic_api_key)

    def _init_gemini(self):
        """
        2안: Google Gemini via google-genai SDK.
        pip install google-genai
        """
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "Google Gemini 패키지가 필요합니다: pip install google-genai"
            )

        if not self.config.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY 환경변수를 설정해주세요.\n"
                "발급: https://aistudio.google.com/app/apikey"
            )

        client = genai.Client(api_key=self.config.google_api_key)
        logger.info(f"Gemini 클라이언트 초기화 완료 (model: {self.llm_config.gemini_model})")
        return client

    def _init_openai(self):
        """OpenAI API key."""
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai")

        if not self.config.openai_api_key:
            raise ValueError("OPENAI_API_KEY 환경변수를 설정해주세요.")

        return openai.OpenAI(api_key=self.config.openai_api_key)

    def _load_claude_code_token(self) -> str:
        """
        Claude Code 로컬 세션에서 OAuth 토큰을 자동 로드합니다.
        ~/.claude/credentials.json 또는 keyring에서 읽기 시도.
        """
        import json
        from pathlib import Path

        # 방법 1: ~/.claude/credentials.json
        cred_path = Path.home() / ".claude" / "credentials.json"
        if cred_path.exists():
            try:
                creds = json.loads(cred_path.read_text())
                token = (
                    creds.get("claudeAiOauth", {}).get("accessToken")
                    or creds.get("access_token")
                    or creds.get("oauthToken")
                )
                if token:
                    logger.debug("Claude Code OAuth 토큰 로드: ~/.claude/credentials.json")
                    return token
            except Exception as e:
                logger.debug(f"credentials.json 파싱 실패: {e}")

        # 방법 2: keyring (선택적)
        try:
            import keyring
            token = keyring.get_password("claude-code", "access_token")
            if token:
                logger.debug("Claude Code OAuth 토큰 로드: keyring")
                return token
        except Exception:
            pass

        return ""

    # ------------------------------------------------------------------
    # LLM call dispatcher
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, use_fast_model: bool = False) -> str:
        """
        LLM 호출 (retry 포함).
        provider에 따라 자동으로 적절한 호출 방식 선택.
        """
        client = self._get_client()
        provider = self.llm_config.provider

        for attempt in range(self.config.agent.max_retries):
            try:
                if provider in ("anthropic_oauth", "anthropic"):
                    return self._call_anthropic(client, prompt, use_fast_model)
                elif provider == "gemini":
                    return self._call_gemini(client, prompt, use_fast_model)
                elif provider == "openai":
                    return self._call_openai(client, prompt, use_fast_model)

            except Exception as e:
                logger.warning(f"LLM 호출 {attempt + 1}회 실패: {e}")
                if attempt < self.config.agent.max_retries - 1:
                    time.sleep(self.config.agent.retry_delay * (2 ** attempt))
                else:
                    raise

        return ""

    def _call_anthropic(self, client, prompt: str, use_fast: bool) -> str:
        """Anthropic Messages API 호출 (OAuth 및 API Key 공통)."""
        model = self.llm_config.fast_model if use_fast else self.llm_config.model
        response = client.messages.create(
            model=model,
            max_tokens=self.llm_config.max_tokens,
            temperature=self.llm_config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _call_gemini(self, client, prompt: str, use_fast: bool) -> str:
        """Google Gemini API 호출 (google-genai SDK)."""
        from google.genai import types

        model = (
            self.llm_config.gemini_fast_model if use_fast
            else self.llm_config.gemini_model
        )
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.llm_config.temperature,
                max_output_tokens=self.llm_config.max_tokens,
            ),
        )
        return response.text

    def _call_openai(self, client, prompt: str, use_fast: bool) -> str:
        """OpenAI Chat Completions API 호출."""
        model = self.llm_config.fast_model if use_fast else self.llm_config.model
        response = client.chat.completions.create(
            model=model,
            max_tokens=self.llm_config.max_tokens,
            temperature=self.llm_config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _parse_json_response(self, response: str) -> Dict:
        """
        LLM 응답에서 JSON 파싱.
        마크다운 코드블록 내 JSON도 처리.
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

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

        logger.error(f"JSON 파싱 실패: {response[:200]}...")
        return {}

    @abstractmethod
    def analyze(self, data: Dict) -> Dict:
        """각 에이전트의 메인 분석 메서드 (0-100 점수 반환)."""
        pass

    def _validate_score(self, score: Any) -> float:
        """점수를 0-100 범위로 제한."""
        try:
            s = float(score)
            return max(0.0, min(100.0, s))
        except (TypeError, ValueError):
            logger.warning(f"유효하지 않은 점수: {score}, 기본값 50.0 사용")
            return 50.0
