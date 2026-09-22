"""Report tests: the translation layer must not invent verdicts.

The rules being defended here are the ones that make the page trustworthy — a
check with too little evidence says ``unknown``, a rate never appears without its
interval, and a wording change is never reported as model jitter.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from evaljev import (
    DecisionAnswer,
    DecisionTrace,
    QuestionSpec,
    build_report,
    declared_label_sets,
    render_html,
)

T0 = datetime(2026, 9, 21, 9, 0, tzinfo=timezone.utc)
CRITERIA = {"refund": "Money back.", "status": "Where is it.", "other": "Neither."}


def trace(
    *,
    minutes=0,
    action="refund",
    probs=None,
    correct=None,
    state=None,
    node="triage",
    question_version="q1",
    qtype="choice",
    criteria=CRITERIA,
    value=None,
):
    answer = (
        DecisionAnswer(question_name="intent", type="choice", selected=action,
                       probabilities=probs or {"refund": 0.9, "status": 0.07, "other": 0.03})
        if qtype == "choice"
        else DecisionAnswer(question_name="intent", type="noul", value=value)
    )
    return DecisionTrace(
        workflow_id="support",
        node_id=node,
        timestamp=T0 + timedelta(minutes=minutes),
        model="jev-1.13.0",
        question_version=question_version,
        state=state if state is not None else {"text": f"request {minutes}"},
        questions=[QuestionSpec(name="intent", type=qtype, instructions="Which action is requested?",
                                criteria=criteria)],
        answers=[answer],
        latency_ms=800 + minutes,
        action=action,
        outcome_correct=correct,
    )


def check(report, check_id):
    return next(c for c in report["checks"] if c["id"] == check_id)


def test_empty_input_is_an_error_not_an_empty_page():
    with pytest.raises(ValueError):
        build_report([])


def test_declared_label_sets_skips_noul():
    traces = [
        trace(),
        trace(qtype="noul", value=0.8, criteria={"true": "yes", "false": "no"}),
    ]
    # A noul answer is a bare probability with no distribution, so scoring it for
    # schema adherence would report a correct answer as a schema failure.
    assert declared_label_sets(traces) == {"intent": ["other", "refund", "status"]}


def test_score_labels_come_from_criteria_positions():
    t = DecisionTrace(
        workflow_id="w", node_id="n", state={}, action="1",
        questions=[QuestionSpec(name="sev", type="score", instructions="How bad?",
                                criteria=["fine", "bad", "worse"])],
        answers=[DecisionAnswer(question_name="sev", type="score", value=1.0,
                                probabilities={"0": 0.1, "1": 0.8, "2": 0.1})],
    )
    assert declared_label_sets([t]) == {"sev": ["0", "1", "2"]}


def test_no_outcomes_means_unknown_not_healthy():
    report = build_report([trace(minutes=i) for i in range(12)])
    assert check(report, "outcomes")["status"] == "unknown"
    assert check(report, "calibration")["status"] == "unknown"
    assert "Results are coming back" in report["headline"]["unknown"]
    # An unknown must not be scored as a pass or as a failure.
    assert report["headline"]["score"] == 100


def test_answer_outside_the_declared_options_is_a_problem():
    rows = [trace(minutes=i) for i in range(10)]
    rows.append(trace(minutes=11, probs={"refund": 0.6, "escalate": 0.4}))
    report = build_report(rows)
    valid = check(report, "valid_answers")
    assert valid["status"] == "problem"
    assert valid["value"] == "10 / 11"
    assert report["headline"]["status"] == "problem"


def test_rates_always_carry_their_interval():
    rows = [trace(minutes=i, correct=i != 0) for i in range(12)]
    report = build_report(rows)
    accuracy = check(report, "outcomes")["evidence"]["accuracy"]
    assert accuracy["ci"][0] < accuracy["rate"] < accuracy["ci"][1]
    assert "range" in check(report, "certainty")["detail"]


def test_repeat_of_the_same_input_that_flips_is_caught():
    state = {"text": "where is my parcel"}
    rows = [trace(minutes=i, state=state) for i in range(6)]
    rows.append(trace(minutes=7, state=state, action="status",
                      probs={"refund": 0.3, "status": 0.6, "other": 0.1}))
    consistency = check(build_report(rows), "consistency")
    assert consistency["status"] in ("watch", "problem")
    assert consistency["evidence"]["flipped"]


def test_a_rewording_is_not_reported_as_jitter():
    """Same input under a new question version is a different question, not a flip."""
    state = {"text": "where is my parcel"}
    rows = [trace(minutes=i, state=state, question_version="q1") for i in range(4)]
    rows += [
        trace(minutes=10 + i, state=state, question_version="q2", action="status",
              probs={"refund": 0.2, "status": 0.7, "other": 0.1})
        for i in range(4)
    ]
    report = build_report(rows)
    assert not check(report, "consistency")["evidence"].get("flipped")
    # ...but the version change itself is still surfaced.
    assert check(report, "config")["status"] == "info"
    assert "question version" in check(report, "config")["detail"]


def test_drift_says_unknown_rather_than_ok_on_thin_traffic():
    report = build_report([trace(minutes=i) for i in range(6)], window_size=5)
    drift = check(report, "drift")
    assert drift["status"] == "unknown"
    assert drift["evidence"]["untestable"]


def test_drift_is_tested_per_decision_point():
    rows = [trace(minutes=i, node="triage") for i in range(20)]
    rows += [trace(minutes=i, node="routing") for i in range(20)]
    report = build_report(rows, window_size=5)
    tested = {n["node_id"] for n in check(report, "drift")["evidence"]["nodes"]}
    assert tested == {"routing", "triage"}


def test_unsure_threshold_counts_the_least_certain_answer():
    rows = [trace(minutes=i) for i in range(9)]
    rows.append(trace(minutes=10, probs={"refund": 0.4, "status": 0.35, "other": 0.25}))
    report = build_report(rows, unsure_below=0.6)
    assert check(report, "certainty")["evidence"]["rate"]["successes"] == 1
    assert report["decisions"][0]["unsure"] is True


def test_report_is_json_serializable_and_renders_without_the_placeholder():
    report = build_report([trace(minutes=i, correct=True) for i in range(12)])
    json.dumps(report)  # must not raise
    html = render_html(report)
    assert "__EVALJEV_REPORT__" not in html
    assert "window.EVALJEV_REPORT" in html
    assert "</script>" not in json.dumps(report)


def test_html_escapes_a_closing_script_tag_in_the_data():
    rows = [trace(minutes=i, state={"text": "</script><b>x"}) for i in range(4)]
    html = render_html(build_report(rows))
    assert "</script><b>x" not in html
    assert "<\\/script>" in html


def test_every_check_carries_the_next_step():
    report = build_report([trace(minutes=i, correct=True) for i in range(12)])
    for c in report["checks"]:
        assert c["question"].endswith("?")
        assert c["advice"]
        assert c["status"] in ("ok", "watch", "problem", "unknown", "info")
