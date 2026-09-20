from __future__ import annotations

import os
import time
from typing import Any, Mapping

import httpx


class JevHTTPClient:
    """Minimal direct HTTP client for TypeSafe System One/Jev.

    This avoids coupling EvalJev to a particular SDK version while preserving the
    documented `model + state + questions` API contract.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = "https://api.typesafe.ai/v1/systemone",
        model: str = "jev-latest",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY")
        if not self.api_key:
            raise ValueError("Set TYPESAFE_API_KEY or pass api_key")
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def decide(self, *, state: Any, questions: Mapping[str, Any]) -> tuple[dict, float]:
        t0 = time.perf_counter()
        response = httpx.post(
            self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "state": state, "questions": questions},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json(), (time.perf_counter() - t0) * 1000
