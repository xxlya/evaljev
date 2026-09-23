"""CLI tests. No network, no browser — the commands must work from a file alone."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from evaljev import DecisionAnswer, DecisionTrace, JsonlTraceStore, QuestionSpec
from evaljev.cli import _load, _sample_traces, main

T0 = datetime(2026, 9, 21, 9, 0, tzinfo=timezone.utc)


def trace(i, correct=None):
    return DecisionTrace(
        workflow_id="support",
        node_id="triage",
        timestamp=T0 + timedelta(minutes=i),
        state={"text": f"request {i}"},
        questions=[QuestionSpec(name="intent", type="choice", instructions="Which action?",
                                criteria={"refund": "Money back.", "other": "Neither."})],
        answers=[DecisionAnswer(question_name="intent", type="choice", selected="refund",
                                probabilities={"refund": 0.9, "other": 0.1})],
        action="refund",
        latency_ms=800,
        outcome_correct=correct,
    )


@pytest.fixture
def traces_file(tmp_path):
    store = JsonlTraceStore(tmp_path / "traces.jsonl")
    store.extend(trace(i) for i in range(12))
    return tmp_path / "traces.jsonl"


def test_console_writes_a_self_contained_page(tmp_path, traces_file, capsys):
    out = tmp_path / "report.html"
    assert main(["console", str(traces_file), "-o", str(out)]) == 0
    html = out.read_text()
    assert html.startswith("<!doctype html>")
    assert "__EVALJEV_REPORT__" not in html
    assert "src=" not in html  # nothing to fetch — the page works offline
    assert "All 12 decisions came through clear" in capsys.readouterr().out


def test_the_json_carries_everything_the_page_leaves_out(tmp_path, traces_file):
    out, blob = tmp_path / "r.html", tmp_path / "r.json"
    main(["console", str(traces_file), "-o", str(out), "--json", str(blob)])
    report = json.loads(blob.read_text())
    assert report["meta"]["n"] == 12
    # The page renders four keys; the JSON keeps the whole analysis behind them.
    assert report["checks"] and report["decisions"] and report["nodes"]
    assert "__EVALJEV_REPORT__" not in out.read_text()


def test_missing_file_fails_with_a_readable_message(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["console", str(tmp_path / "nope.jsonl"), "-o", str(tmp_path / "x.html")])
    assert "no such trace file" in str(exc.value)


def test_empty_file_says_how_to_record_traces(tmp_path):
    empty = tmp_path / "traces.jsonl"
    empty.write_text("")
    with pytest.raises(SystemExit) as exc:
        main(["console", str(empty), "-o", str(tmp_path / "x.html")])
    assert "Monitor" in str(exc.value)


def test_outcome_rewrites_are_collapsed_across_files(tmp_path):
    """An outcome is appended as a second copy of the trace — never counted twice."""
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    first = trace(0)
    JsonlTraceStore(a).append(first)
    first.outcome_correct = True
    JsonlTraceStore(b).append(first)
    loaded = _load([str(a), str(b)])
    assert len(loaded) == 1
    assert loaded[0].outcome_correct is True


def test_demo_renders_from_the_bundled_run(tmp_path):
    out = tmp_path / "demo.html"
    assert main(["demo", "-o", str(out)]) == 0
    assert out.stat().st_size > 20_000
    assert len(_sample_traces()) == 351


def test_unsure_threshold_reaches_the_report(tmp_path, traces_file):
    blob = tmp_path / "r.json"
    main(["console", str(traces_file), "-o", str(tmp_path / "r.html"),
          "--unsure-below", "0.95", "--json", str(blob)])
    report = json.loads(blob.read_text())
    assert report["meta"]["unsure_below"] == 0.95
    # every decision sits at 0.90, so a 0.95 line puts all of them in the pile
    assert report["checks"][0]["id"] == "certainty"
    assert report["checks"][0]["status"] == "problem"
    # ...and every request becomes one the workflow answered alone under doubt.
    assert report["queue"]["counts"]["acted_alone"] == 12
