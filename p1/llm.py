"""LLM client factory (ASU Voyager, OpenAI-compatible endpoint)."""

import os

from langchain_openai import ChatOpenAI

from p1 import config


class LLMUnavailable(RuntimeError):
    """Raised when no API key is configured; callers fall back to rules."""


def api_key() -> str | None:
    key = (os.environ.get(config.LLM_API_KEY_ENV) or "").strip()
    return key or None


def llm_available() -> bool:
    return api_key() is not None


def get_llm(model: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    key = api_key()
    if key is None:
        raise LLMUnavailable(
            f"{config.LLM_API_KEY_ENV} is not set. Add it to .env (see .env.example)."
        )
    return ChatOpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=key,
        model=model or config.AGENT_MODEL,
        temperature=temperature,
        timeout=config.LLM_TIMEOUT_SEC,
        max_tokens=config.LLM_MAX_TOKENS,
        max_retries=2,
    )
