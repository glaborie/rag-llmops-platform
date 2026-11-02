"""OpenAI-compatible chat-completions provider.

Works against the OpenAI API, Azure OpenAI, Ollama (``/v1``), vLLM, LM Studio
and anything else that speaks the ``/chat/completions`` protocol. The import
surface is ``requests`` only — no SDK required.
"""

from __future__ import annotations

import time

import requests

from ragllmops.llm.base import LLMResult
from ragllmops.logging_utils import get_logger

logger = get_logger("llm.openai_compat")


class OpenAICompatLLM:
    """Chat-completions client for any OpenAI-compatible endpoint."""

    provider = "openai"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        provider_name: str = "openai",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.provider = provider_name

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        context: list[str] | None = None,
    ) -> LLMResult:
        started = time.perf_counter()
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        url = f"{self.base_url}/chat/completions"
        response = requests.post(
            url, json=payload, headers=self._headers(), timeout=self.timeout
        )
        response.raise_for_status()
        body = response.json()

        choices = body.get("choices") or []
        text = ""
        if choices:
            text = (choices[0].get("message") or {}).get("content") or ""
        usage = body.get("usage") or {}

        return LLMResult(
            text=text.strip(),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=(time.perf_counter() - started) * 1000,
            model=self.model,
            provider=self.provider,
            raw={"finish_reason": choices[0].get("finish_reason") if choices else None},
        )

    def health(self) -> bool:
        """Best-effort reachability check used by ``cli doctor``."""
        try:
            response = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=5)
            return response.status_code < 500
        except requests.RequestException:
            return False
