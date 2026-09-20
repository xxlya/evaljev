"""Outcomes recorded after a durable write must reach the store."""

from evaljev import InMemoryTraceStore, JsonlTraceStore, Monitor, calibration_report
from evaljev.store import latest_by_trace_id

Q = {
    "route": {
        "type": "choice",
        "instructions": "Choose the correct route for this request",
        "criteria": {"a": "A", "b": "B", "other": "Other"},
    }
}


class FakeClient:
    model = "jev-test"

    def decide(self, *, state, questions):
        return {
            "model": self.model,
            "answers": {
                "route": {"choice": "a", "confidence": 0.9, "probabilities": {"a": 0.9, "b": 0.1}}
            },
        }, 5.0


def run_one(monitor, state):
    _, trace = monitor.run(
        FakeClient(),
        workflow_id="w",
        node_id="n",
        state=state,
        questions=Q,
        policy=lambda a: a["route"].selected,
    )
    return trace


def test_latest_by_trace_id_keeps_newest_at_first_position():
    class T:
        def __init__(self, trace_id, tag):
            self.trace_id, self.tag = trace_id, tag

    rows = [T("a", 1), T("b", 1), T("a", 2), T("c", 1), T("a", 3)]
    out = latest_by_trace_id(rows)
    assert [(t.trace_id, t.tag) for t in out] == [("a", 3), ("b", 1), ("c", 1)]


def test_outcome_persists_to_jsonl_store(tmp_path):
    path = tmp_path / "traces.jsonl"
    monitor = Monitor(JsonlTraceStore(path))
    trace = run_one(monitor, {"x": 1})
    monitor.record_outcome(trace, outcome={"accepted": True}, correct=True)

    # Re-read from disk: the outcome must survive the round trip.
    reloaded = JsonlTraceStore(path).list("w")
    assert len(reloaded) == 1
    assert reloaded[0].trace_id == trace.trace_id
    assert reloaded[0].outcome == {"accepted": True}
    assert reloaded[0].outcome_correct is True


def test_jsonl_store_keeps_the_full_event_history(tmp_path):
    path = tmp_path / "traces.jsonl"
    monitor = Monitor(JsonlTraceStore(path))
    trace = run_one(monitor, {"x": 1})
    monitor.record_outcome(trace, outcome={"accepted": False}, correct=False)

    # Two events on disk, one trace on read.
    assert len(path.read_text().strip().splitlines()) == 2
    assert len(JsonlTraceStore(path).list()) == 1


def test_in_memory_store_does_not_duplicate_after_outcome():
    monitor = Monitor(InMemoryTraceStore())
    trace = run_one(monitor, {"x": 1})
    monitor.record_outcome(trace, outcome={"accepted": True}, correct=True)
    traces = monitor.store.list("w")
    assert len(traces) == 1
    assert traces[0].outcome_correct is True


def test_calibration_sees_outcomes_written_to_disk(tmp_path):
    """The bug this guards: calibration over a JSONL store used to report nothing."""
    path = tmp_path / "traces.jsonl"
    monitor = Monitor(JsonlTraceStore(path))
    for i in range(4):
        trace = run_one(monitor, {"x": i})
        monitor.record_outcome(trace, outcome={"i": i}, correct=i % 2 == 0)

    report = calibration_report(JsonlTraceStore(path).list("w"))
    assert report["route"]["n"] == 4
    assert report["route"]["accuracy"] == 0.5
    assert report["route"]["mean_confidence"] == 0.9


def test_workflow_filter_applies_before_collapsing():
    monitor = Monitor(InMemoryTraceStore())
    trace = run_one(monitor, {"x": 1})
    monitor.record_outcome(trace, outcome={}, correct=True)
    assert len(monitor.store.list("w")) == 1
    assert monitor.store.list("other") == []
