"""
Unified LLM client — OpenAI-compatible API for DeepInfra, OpenRouter, and Ollama.

All three expose POST /chat/completions with Bearer auth:
  - Ollama:      http://localhost:11434/v1
  - DeepInfra:   https://api.deepinfra.com/v1/openai
  - OpenRouter:  https://openrouter.ai/api/v1

Usage:
    from mestrado.utils.llm_client import get_client

    client = get_client()                    # uses LLM_PROVIDER env var
    client = get_client("deepinfra")         # explicit provider
    text = client.complete("llama3.1:8b", messages=[...])
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Thin HTTP wrapper around any OpenAI-compatible /chat/completions endpoint.
    No external dependencies — uses stdlib urllib.request only.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        provider: str = "unknown",
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.provider = provider
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        """
        POST /chat/completions — returns the raw parsed JSON response.

        Args:
            model:       Provider-specific model ID.
            messages:    OpenAI-style messages list.
            temperature: Sampling temperature (0 = deterministic).
            max_tokens:  Max output tokens.
            json_mode:   If True, request JSON output (response_format).

        Returns:
            Parsed response dict. Empty {"choices": [], "usage": {}} on error.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://anonymous.4open.science"
            headers["X-Title"] = "Ulysses-MTRAG"

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")
            logger.warning(f"[{self.provider}] HTTP {e.code}: {body[:300]}")
            return {"choices": [], "usage": {}}
        except Exception as e:
            logger.warning(f"[{self.provider}] Request failed: {e}")
            return {"choices": [], "usage": {}}

    def extract_text(self, response: dict[str, Any]) -> str:
        """Extract content string from first choice."""
        try:
            text = response["choices"][0]["message"]["content"] or ""
            # Strip markdown code fences that some models add
            return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        except (KeyError, IndexError):
            return ""

    def extract_usage(self, response: dict[str, Any]) -> tuple[int, int]:
        """Returns (input_tokens, output_tokens)."""
        usage = response.get("usage", {})
        return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> str:
        """Convenience: chat() → extract_text()."""
        resp = self.chat(model, messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)
        return self.extract_text(resp)

    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider!r}, base_url={self.base_url!r})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_client(provider: str | None = None) -> LLMClient:
    """
    Build an LLMClient from config.

    Priority: explicit `provider` arg → LLM_PROVIDER env var → "ollama"

    Examples:
        client = get_client()              # from LLM_PROVIDER env var
        client = get_client("deepinfra")   # DeepInfra
        client = get_client("openrouter")  # OpenRouter
        client = get_client("ollama")      # Local Ollama
    """
    from mestrado.config import llm_provider as provider_cfg

    p = provider or provider_cfg.provider

    if p == "deepinfra":
        from mestrado.config import deepinfra as cfg
        if not cfg.api_key:
            raise ValueError("DEEPINFRA_API_KEY não configurada no .env")
        return LLMClient(cfg.base_url, cfg.api_key, provider="deepinfra", timeout=cfg.timeout)

    if p == "openrouter":
        from mestrado.config import openrouter as cfg
        if not cfg.api_key:
            raise ValueError("OPENROUTER_API_KEY não configurada no .env")
        return LLMClient(cfg.base_url, cfg.api_key, provider="openrouter")

    # Default: Ollama via its OpenAI-compatible endpoint
    from mestrado.config import ollama as cfg
    base_url = cfg.base_url.rstrip("/") + "/v1"
    return LLMClient(base_url, "ollama", provider="ollama", timeout=cfg.timeout)
