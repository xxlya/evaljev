"""Offline tests for the spend guard. No network, no keys."""

import pytest

from evaljev.budget import BudgetedJevClient, BudgetExceeded, SpendBudget


def test_records_spend_and_remaining():
    budget = SpendBudget({"jev": 5.00})
    budget.record("jev", 0.000153)
    assert budget.spent["jev"] == pytest.approx(0.000153)
    assert budget.remaining("jev") == pytest.approx(4.999847)
    assert budget.calls["jev"] == 1


def test_check_raises_only_when_estimate_exceeds_remaining():
    budget = SpendBudget({"jev": 1.00})
    budget.record("jev", 0.99)
    budget.check("jev", estimate=0.01)  # exactly at the line is allowed
    with pytest.raises(BudgetExceeded, match=r"jev: budget exhausted"):
        budget.check("jev", estimate=0.02)


def test_unlimited_api_is_tracked_but_never_blocks():
    budget = SpendBudget()
    budget.check("gemini", estimate=1_000.0)
    budget.record("gemini", 12.5)
    assert budget.remaining("gemini") == float("inf")
    assert budget.summary()["gemini"] == {
        "calls": 1,
        "spent_usd": 12.5,
        "limit_usd": None,
        "remaining_usd": None,
    }


def test_spend_checks_before_recording():
    budget = SpendBudget({"claude": 0.10})
    budget.spend("claude", 0.04)
    with pytest.raises(BudgetExceeded):
        budget.spend("claude", 0.07)
    # The refused call must not be billed.
    assert budget.spent["claude"] == pytest.approx(0.04)
    assert budget.calls["claude"] == 1


def test_summary_reports_every_api():
    budget = SpendBudget({"jev": 5.0, "claude": 5.0})
    budget.record("jev", 0.001)
    assert budget.summary() == {
        "claude": {"calls": 0, "spent_usd": 0.0, "limit_usd": 5.0, "remaining_usd": 5.0},
        "jev": {"calls": 1, "spent_usd": 0.001, "limit_usd": 5.0, "remaining_usd": 4.999},
    }


class FakeJev:
    """Stands in for JevHTTPClient: returns the live response shape."""

    model = "jev-fake"

    def __init__(self, cost_usd=0.000153, credits=9.99):
        self.cost_usd = cost_usd
        self.credits = credits
        self.calls = 0

    def decide(self, *, state, questions):
        self.calls += 1
        return {
            "model": "jev-1.13.0",
            "answers": {"route": {"choice": "claude", "confidence": 0.8}},
            "usage": {"cost_usd": self.cost_usd, "credits_remaining_usd": self.credits},
        }, 12.0


def test_wrapper_records_billed_cost_from_usage():
    budget = SpendBudget({"jev": 5.0})
    client = BudgetedJevClient(FakeJev(), budget)
    response, latency_ms = client.decide(state={}, questions={})
    assert response["model"] == "jev-1.13.0"
    assert latency_ms == 12.0
    assert budget.spent["jev"] == pytest.approx(0.000153)
    assert client.credits_remaining_usd == pytest.approx(9.99)


def test_wrapper_blocks_the_call_that_would_breach_the_cap():
    budget = SpendBudget({"jev": 0.001})
    inner = FakeJev(cost_usd=0.0009)
    client = BudgetedJevClient(inner, budget, estimate_usd=0.0009)
    client.decide(state={}, questions={})
    with pytest.raises(BudgetExceeded):
        client.decide(state={}, questions={})
    # The second call never reached the network.
    assert inner.calls == 1


def test_wrapper_tolerates_a_response_without_usage():
    budget = SpendBudget({"jev": 5.0})

    class NoUsage(FakeJev):
        def decide(self, *, state, questions):
            return {"answers": {}}, 5.0

    client = BudgetedJevClient(NoUsage(), budget)
    client.decide(state={}, questions={})
    assert budget.spent["jev"] == 0.0
    assert budget.calls["jev"] == 1


def test_wrapper_exposes_inner_model_for_monitor():
    assert BudgetedJevClient(FakeJev(), SpendBudget()).model == "jev-fake"
