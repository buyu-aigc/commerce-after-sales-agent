from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LlmSettings:
    api_key: str
    model: str
    base_url: str | None
    provider: str


def resolve_llm_settings(default_model: str = "gpt-4o-mini") -> LlmSettings:
    """Resolve OpenAI-compatible LLM settings from local environment variables."""

    openai_key = os.environ.get("OPENAI_API_KEY")
    aihubmix_key = os.environ.get("AIHUBMIX_API_KEY")
    api_key = openai_key or aihubmix_key or "offline"
    provider = "openai_compatible" if openai_key else "aihubmix" if aihubmix_key else "offline"

    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url and aihubmix_key:
        base_url = "https://aihubmix.com/v1"

    return LlmSettings(
        api_key=api_key,
        model=os.environ.get("OPENAI_MODEL", default_model),
        base_url=base_url,
        provider=provider,
    )
