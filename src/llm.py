"""LLM client with Ollama (local) and OpenAI (remote) fallback."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class LLMError(Exception):
    """Raised when all LLM backends fail."""


def _post_json(url: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    """POST JSON and return parsed response."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def call_ollama(
    prompt: str,
    model: str = "llama3",
    base_url: str = "http://localhost:11434",
    system: str = "",
) -> str:
    """Call Ollama /api/generate and return the full response text."""
    url = f"{base_url}/api/generate"
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    resp = _post_json(url, payload)
    return resp.get("response", "")


def call_openai(
    prompt: str,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    base_url: str = "https://api.openai.com/v1",
    system: str = "",
) -> str:
    """Call OpenAI-compatible chat completions endpoint."""
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY not set")
    url = f"{base_url}/chat/completions"
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    resp = _post_json(url, payload, timeout=120)
    return resp["choices"][0]["message"]["content"]


def call_llm(
    prompt: str,
    model: str = "llama3",
    system: str = "",
    ollama_url: str = "http://localhost:11434",
    openai_key: str | None = None,
) -> str:
    """Try Ollama first, fall back to OpenAI-compatible API."""
    errors: list[str] = []
    try:
        return call_ollama(prompt, model=model, base_url=ollama_url, system=system)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        errors.append(f"Ollama: {exc}")

    try:
        return call_openai(prompt, system=system, api_key=openai_key)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        errors.append(f"OpenAI: {exc}")

    raise LLMError("All LLM backends failed:\n" + "\n".join(errors))
