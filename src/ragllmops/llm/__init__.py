"""LLM providers and the factory that selects one."""

from __future__ import annotations

from ragllmops.config import Settings, get_settings
from ragllmops.llm.base import LLMProvider, LLMResult
from ragllmops.llm.extractive import REFUSAL_TEXT, ExtractiveLLM

__all__ = [
    "ExtractiveLLM",
    "LLMProvider",
    "LLMResult",
    "REFUSAL_TEXT",
    "build_llm",
]


def build_llm(settings: Settings | None = None, **overrides: object) -> LLMProvider:
    """Instantiate the configured LLM provider.

    ``overrides`` lets the caller pin a provider/model for a single run
    (the evaluation runner does this to compare candidates).
    """
    s = settings or get_settings()
    provider = str(overrides.get("provider") or s.llm_provider).lower()
    model = str(overrides.get("model") or s.llm_model)
    base_url = overrides.get("base_url") or s.llm_base_url
    api_key = overrides.get("api_key") or s.llm_api_key
    timeout = float(overrides.get("timeout") or s.llm_timeout_seconds)

    if provider == "extractive":
        return ExtractiveLLM(model=model)
    if provider in {"openai", "ollama", "openai-compatible"}:
        if not base_url:
            raise ValueError(
                f"provider {provider!r} requires a base URL "
                "(set RAG_LLM_BASE_URL, e.g. http://localhost:11434/v1)"
            )
        from ragllmops.llm.openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(
            model=model,
            base_url=str(base_url),
            api_key=str(api_key) if api_key else None,
            timeout=timeout,
            provider_name="ollama" if provider == "ollama" else "openai",
        )
    raise ValueError(f"unknown LLM provider: {provider!r}")
