from evaljev import InMemoryTraceStore, Monitor, replay, summarize_replay


class FakeClient:
    model = "jev-test"

    def __init__(self, selected="a"):
        self.selected = selected

    def decide(self, *, state, questions):
        p = {"a": 0.9, "b": 0.1} if self.selected == "a" else {"a": 0.1, "b": 0.9}
        return {"model": self.model, "answers": {"route": {"choice": self.selected, "confidence": 0.9, "probabilities": p}}}, 5.0


Q = {"route": {"type": "choice", "instructions": "Choose the correct route for this request", "criteria": {"a": "A", "b": "B", "other": "Other"}}}


def policy(a):
    return a["route"].selected


def test_monitor_and_replay():
    store = InMemoryTraceStore()
    m = Monitor(store)
    _, trace = m.run(FakeClient("a"), workflow_id="w", node_id="n", state={"x": 1}, questions=Q, policy=policy)
    trace.outcome_correct = False
    assert trace.action == "a"
    results = replay(store.list(), FakeClient("b"), questions=Q, policy=policy, judge=lambda t, a: a == "b")
    summary = summarize_replay(results)
    assert summary["changed"] == 1
    assert summary["improvements"] == 1
