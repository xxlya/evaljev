"""Hard spend ceilings for live evaluation runs.

Evaluation loops call paid APIs in bulk — a replay over 10k traces is 10k billed
requests. This module enforces a per-API ceiling so a runaway loop stops instead
of draining an account.

The library enforces; the caller prices. :class:`SpendBudget` only ever receives
USD amounts, so provider rate cards live in application code and can change
without touching EvalJev.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class BudgetExceeded(RuntimeError):
    """Raised before a call that would exceed the configured ceiling."""


class SpendBudget:
    """Track spend per API against a hard per-API ceiling.

    >>> budget = SpendBudget({"jev": 5.00})
    >>> budget.check("jev", estimate=0.001)
    >>> budget.record("jev", 0.000153)
    >>> round(budget.remaining("jev"), 6)
    4.999847

    An API with no configured limit is unlimited but still tracked, so a summary
    always reports true spend.
    """

    def __init__(self, limits: Mapping[str, float] | None = None) -> None:
        self.limits: dict[str, float] = dict(limits or {})
        self.spent: dict[str, float] = dict.fromkeys(self.limits, 0.0)
        self.calls: dict[str, int] = dict.fromkeys(self.limits, 0)

    def remaining(self, api: str) -> float:
        limit = self.limits.get(api)
        if limit is None:
            return float("inf")
        return limit - self.spent.get(api, 0.0)

    def check(self, api: str, estimate: float = 0.0) -> None:
        """Raise if a call costing ``estimate`` would break the ceiling.

        Called *before* spending. Costs are only known exactly after a response
        arrives, so the estimate should be an upper bound.
        """
        remaining = self.remaining(api)
        if estimate > remaining:
            raise BudgetExceeded(
                f"{api}: budget exhausted — spent ${self.spent.get(api, 0.0):.4f} of "
                f"${self.limits[api]:.2f}, next call needs ~${estimate:.4f}"
            )

    def record(self, api: str, usd: float) -> None:
        """Record actual spend after a call. Never raises — the money is gone."""
        self.spent[api] = self.spent.get(api, 0.0) + max(usd, 0.0)
        self.calls[api] = self.calls.get(api, 0) + 1

    def spend(self, api: str, usd: float, *, estimate: float | None = None) -> None:
        """check() then record(), for callers that know the cost up front."""
        self.check(api, estimate if estimate is not None else usd)
        self.record(api, usd)

    def summary(self) -> dict[str, dict[str, Any]]:
        apis = set(self.limits) | set(self.spent)
        return {
            api: {
                "calls": self.calls.get(api, 0),
                "spent_usd": round(self.spent.get(api, 0.0), 6),
                "limit_usd": self.limits.get(api),
                "remaining_usd": (
                    None if self.limits.get(api) is None else round(self.remaining(api), 6)
                ),
            }
            for api in sorted(apis)
        }


class BudgetedJevClient:
    """Wrap a Jev client so every ``decide`` is metered against a budget.

    Duck-types :class:`~evaljev.client.JevHTTPClient`, so it drops into
    ``Monitor.run``, ``stability_check`` and ``replay`` unchanged.

    Jev returns ``usage.cost_usd`` per response, so recorded spend is the real
    billed amount, not an approximation. ``estimate_usd`` is only the upper bound
    used to refuse a call before it is made.
    """

    def __init__(
        self,
        client: Any,
        budget: SpendBudget,
        *,
        api: str = "jev",
        estimate_usd: float = 0.01,
    ) -> None:
        self.client = client
        self.budget = budget
        self.api = api
        self.estimate_usd = estimate_usd
        self.credits_remaining_usd: float | None = None

    @property
    def model(self) -> str | None:
        """Monitor reads this when a response omits its own model label."""
        return getattr(self.client, "model", None)

    def decide(self, *, state: Any, questions: Mapping[str, Any]) -> tuple[dict, float]:
        self.budget.check(self.api, self.estimate_usd)
        response, latency_ms = self.client.decide(state=state, questions=questions)
        usage = response.get("usage") or {}
        self.budget.record(self.api, float(usage.get("cost_usd") or 0.0))
        if usage.get("credits_remaining_usd") is not None:
            self.credits_remaining_usd = float(usage["credits_remaining_usd"])
        return response, latency_ms
