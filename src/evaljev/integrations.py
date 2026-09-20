from __future__ import annotations

import os
from typing import Any

VECTOR_BASE_URL = "https://proxy.vectorinstitute.ai/v1"


def call_claude(
    prompt: str,
    *,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 1200,
) -> str:
    """Call Claude through the Vector Institute proxy.

    The proxy speaks the OpenAI chat-completions protocol, so this uses the
    ``openai`` client pointed at ``VECTOR_BASE_URL`` rather than the Anthropic
    SDK. Requires ``VECTOR_API_KEY`` (a ``vp_``-prefixed proxy key).
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Install evaljev[llm] to use the Claude integration") from e
    api_key = os.getenv("VECTOR_API_KEY")
    if not api_key:
        raise RuntimeError("Set VECTOR_API_KEY to your Vector proxy key (starts with 'vp_')")
    client = OpenAI(api_key=api_key, base_url=os.getenv("VECTOR_BASE_URL", VECTOR_BASE_URL))
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def call_gemini(prompt: str, *, model: str = "gemini-2.5-pro") -> str:
    """Call Gemini directly against the Google API. Requires ``GEMINI_API_KEY``."""
    try:
        from google import genai
    except ImportError as e:
        raise RuntimeError("Install evaljev[llm] to use the Gemini integration") from e
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY to your Google AI Studio key")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text or ""


def execute_model_route(action: str, prompt: str) -> dict[str, Any]:
    if action == "claude":
        return {"provider": "claude", "text": call_claude(prompt)}
    if action == "gemini":
        return {"provider": "gemini", "text": call_gemini(prompt)}
    if action == "both":
        return {"provider": "both", "claude": call_claude(prompt), "gemini": call_gemini(prompt)}
    raise ValueError(f"Unknown route: {action}")
