from __future__ import annotations

import os
from typing import Any


def call_claude(prompt: str, *, model: str = "claude-sonnet-4-5") -> str:
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("Install evaljev[llm] to use Claude integration") from e
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    msg = client.messages.create(model=model, max_tokens=1200, messages=[{"role": "user", "content": prompt}])
    return "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")


def call_gemini(prompt: str, *, model: str = "gemini-2.5-pro") -> str:
    try:
        from google import genai
    except ImportError as e:
        raise RuntimeError("Install evaljev[llm] to use Gemini integration") from e
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
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
