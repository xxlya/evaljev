"""Wrapping a client must be invisible to code that already works."""

import asyncio
import json

import pytest

from evaljev import InMemoryTraceStore, JsonlTraceStore, build_report, watch

QUESTION = {
    "intent": {
        "type": "choice",
        "instructions": "Which action is requested?",
        "criteria": {"refund": "Money back.", "status": "Where is it.", "other": "Neither."},
    }
}
URGENCY = {
    "urgency": {
        "type": "choice",
        "instructions": "How urgent?",
        "criteria": {"now": "Now.", "later": "Later."},
    }
}


class FakeClient:
    """Stands in for JevHTTPClient: same `decide` contract, no network."""

    model = "jev-1.13.0"

    def __init__(self, choice="refund", p=0.92):
        self.choice, self.p, self.calls = choice, p, 0

    def decide(self, *, state, questions):
        self.calls += 1
        name = min(questions)
        labels = list(questions[name]["criteria"])
        rest = (1 - self.p) / max(1, len(labels) - 1)
        choice = self.choice if self.choice in labels else labels[0]
        probs = {lab: (self.p if lab == choice else rest) for lab in labels}
        return {"answers": {name: {"choice": choice, "probabilities": probs}},
                "usage": {"cost_usd": 0.00015}}, 180.0


def test_the_call_site_does_not_change():
    inner = FakeClient()
    client = watch(inner, workflow_id="app", store=InMemoryTraceStore())
    response, latency = client.decide(state={"text": "hi"}, questions=QUESTION)
    assert response["answers"]["intent"]["choice"] == "refund"
    assert latency == 180.0          # the wrapped client's own return, unchanged
    assert inner.calls == 1


def test_a_decision_is_recorded_without_being_told_anything_else():
    store = InMemoryTraceStore()
    client = watch(FakeClient(), workflow_id="app", store=store)
    client.decide(state={"text": "hi"}, questions=QUESTION)
    trace = store.list()[0]
    assert trace.workflow_id == "app"
    assert trace.node_id == "intent"      # named after the question, absent anything better
    assert trace.action == "refund"       # the branch the model chose, absent a policy
    assert trace.model == "jev-1.13.0"
    assert trace.metadata["cost_usd"] == 0.00015
    assert trace.latency_ms == 180.0


def test_a_request_block_ties_its_decisions_together():
    store = InMemoryTraceStore()
    client = watch(FakeClient(), workflow_id="app", store=store)
    with client.request("req-1"):
        with client.step("classify"):
            client.decide(state={"text": "hi"}, questions=QUESTION)
        with client.step("triage"):
            client.decide(state={"text": "hi"}, questions=URGENCY)
    client.decide(state={"text": "other"}, questions=QUESTION)   # outside the block

    traces = store.list()
    assert [t.node_id for t in traces] == ["classify", "triage", "intent"]
    assert [t.metadata.get("request_id") for t in traces] == ["req-1", "req-1", None]


def test_labels_do_not_leak_between_concurrent_steps():
    """Two coroutines sharing one client must not overwrite each other's labels."""
    store = InMemoryTraceStore()
    client = watch(FakeClient(), workflow_id="app", store=store)

    async def handle(request_id, node):
        with client.step(node, request_id=request_id):
            await asyncio.sleep(0)       # hand control to the other coroutine mid-step
            client.decide(state={"text": request_id}, questions=QUESTION)

    async def both():
        await asyncio.gather(handle("a", "first"), handle("b", "second"))

    asyncio.run(both())
    seen = {(t.node_id, t.metadata["request_id"]) for t in store.list()}
    assert seen == {("first", "a"), ("second", "b")}


def test_an_outcome_attaches_to_the_decision_it_belongs_to():
    store = InMemoryTraceStore()
    client = watch(FakeClient(), workflow_id="app", store=store)
    client.decide(state={"text": "hi"}, questions=QUESTION)
    first = client.last_trace
    client.decide(state={"text": "again"}, questions=QUESTION)

    client.record_outcome({"reviewer": "agreed"}, correct=True)          # the last one
    client.record_outcome({"reviewer": "disagreed"}, correct=False, trace=first)

    by_id = {t.trace_id: t for t in store.list()}
    assert by_id[first.trace_id].outcome_correct is False
    assert by_id[client.last_trace.trace_id].outcome_correct is True


def test_a_policy_still_wins_over_the_models_own_branch():
    store = InMemoryTraceStore()
    client = watch(
        FakeClient(p=0.4),
        workflow_id="app",
        store=store,
        policy=lambda answers: "needs_review",
    )
    client.decide(state={"text": "hi"}, questions=QUESTION)
    assert store.list()[0].action == "needs_review"


def test_anything_else_on_the_client_still_reaches_it():
    inner = FakeClient()
    client = watch(inner, workflow_id="app", store=InMemoryTraceStore())
    assert client.model == "jev-1.13.0"
    with pytest.raises(AttributeError):
        _ = client.no_such_method


def test_one_wrapped_workflow_produces_a_report(tmp_path):
    """The whole path: wrap, run, audit — with nothing else written by hand."""
    path = tmp_path / "traces.jsonl"
    client = watch(FakeClient(), workflow_id="app", path=path)
    for i in range(12):
        with client.request(f"r{i}"):
            client.decide(state={"text": f"message {i}"}, questions=QUESTION)

    report = build_report(JsonlTraceStore(path).list())
    assert report["meta"]["n"] == 12
    assert report["queue"]["total"] == 12
    assert [w["node_id"] for w in report["workflow"]] == ["intent"]
    assert json.dumps(report)      # serialisable end to end
