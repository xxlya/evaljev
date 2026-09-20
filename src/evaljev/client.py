from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

import httpx

JEV_BASE_URL = "https://jevtypesafeai.com/api/v1/decide"


class JevHTTPClient:
    """Minimal direct HTTP client for the Jev decision API.

    This avoids coupling EvalJev to a particular SDK version while preserving the
    documented `state + questions` API contract.

    Reads ``JEV_API_KEY`` (a ``jv_live_``-prefixed key) and, for continuity with
    earlier revisions, falls back to ``TYPESAFE_API_KEY``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("JEV_API_KEY") or os.getenv("TYPESAFE_API_KEY")
        if not self.api_key:
            raise ValueError("Set JEV_API_KEY or pass api_key")
        self.base_url = base_url or os.getenv("JEV_BASE_URL", JEV_BASE_URL)
        # Not sent in the request body — the API infers the model. Kept so
        # Monitor can fall back to it when labelling a trace.
        self.model = model
        self.timeout = timeout

    def decide(self, *, state: Any, questions: Mapping[str, Any]) -> tuple[dict, float]:
        t0 = time.perf_counter()
        response = httpx.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"state": state, "questions": questions},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json(), (time.perf_counter() - t0) * 1000
