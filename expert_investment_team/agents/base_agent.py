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
        1안: Anthropic OAuth / API Key 인증.

        ※ Claude Code 웹/원격 환경 주의사항:
           Claude Code의 내부 세션 토큰(sk-ant-si-...)은 Anthropic Messages API에
           직접 사용할 수 없습니다. 사용자가 별도의 인증 정보를 제공해야 합니다.

        탐색 순서:
          1) ANTHROPIC_ACCESS_TOKEN 환경변수 (진짜 OAuth access token)
          2) ~/.claude/credentials.json (로컬 Claude Code 설치, claudeAiOauth.accessToken)
          3) ANTHROPIC_API_KEY 환경변수 (일반 API key, fallback)
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        import os

        # 1) 명시적 OAuth access token (claude.ai에서 발급한 실제 OAuth token)
        access_token = self.config.anthropic_access_token
        if not access_token:
            access_token = self._load_claude_code_credentials_token()

        base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip() or None

        if access_token:
            logger.info(f"Anthropic OAuth access token 사용")
            kwargs = {"api_key": access_token}
            if base_url:
                kwargs["base_url"] = base_url
            return anthropic.Anthropic(**kwargs)

        # 2) 일반 API Key fallback
        if self.config.anthropic_api_key:
            logger.info("Anthropic API Key 사용")
            kwargs = {"api_key": self.config.anthropic_api_key}
            if base_url:
                kwargs["base_url"] = base_url
            return anthropic.Anthropic(**kwargs)

        raise ValueError(
            "\n"
            "═══════════════════════════════════════════════════\n"
            " Anthropic 인증 정보가 필요합니다\n"
            "═══════════════════════════════════════════════════\n"
            " [1안] Anthropic API Key (권장):\n"
            "   export ANTHROPIC_API_KEY=sk-ant-api...\n"
            "   발급: https://console.anthropic.com/\n"
            "\n"
            " [2안] Google Gemini (무료 티어):\n"
            "   export GOOGLE_API_KEY=...\n"
            "   python main.py --provider gemini\n"
            "   발급: https://aistudio.google.com/app/apikey\n"
            "═══════════════════════════════════════════════════\n"
            " ※ Claude Code 세션 토큰(sk-ant-si-)은 내부 전용으로\n"
            "    Anthropic Messages API에 사용 불가합니다.\n"
            "═══════════════════════════════════════════════════"
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

    def _load_claude_code_credentials_token(self) -> str:
        """
        로컬 Claude Code 설치 환경에서 OAuth access token 로드.
        ~/.claude/credentials.json의 claudeAiOauth.accessToken을 읽습니다.

        ※ Claude Code 웹/원격 환경의 세션 토큰(sk-ant-si-)은 여기서 로드하지 않습니다.
           해당 토큰은 Claude Code 내부 통신 전용이며 Messages API에 사용 불가합니다.
        """
        import json
        from pathlib import Path

        cred_path = Path.home() / ".claude" / "credentials.json"
        if not cred_path.exists():
            return ""

        try:
            creds = json.loads(cred_path.read_text())
            # 로컬 설치의 실제 OAuth access token
            token = creds.get("claudeAiOauth", {}).get("accessToken", "")
            if token and not token.startswith("sk-ant-si-"):
                logger.debug("OAuth token 로드: ~/.claude/credentials.json")
                return token
        except Exception as e:
            logger.debug(f"credentials.json 파싱 실패: {e}")

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
