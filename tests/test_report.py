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
from evaljev.report import PAGE_KEYS

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
    # An unknown must not be read as a pass or as a failure.
    assert report["headline"]["status"] == "ok"


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
    assert "triage question wording: q1 → q2" in check(report, "config")["detail"]


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
    rows = [
        trace(minutes=i, state={"text": "</script><b>x"}, probs={"refund": 0.4, "status": 0.35, "other": 0.25})
        for i in range(4)
    ]
    html = render_html(build_report(rows))
    assert "</script><b>x" not in html
    assert "<\\/script>" in html


def test_every_check_carries_the_next_step():
    report = build_report([trace(minutes=i, correct=True) for i in range(12)])
    for c in report["checks"]:
        assert c["question"].endswith("?")
        assert c["advice"]
        assert c["status"] in ("ok", "watch", "problem", "unknown", "info")


def workflow_traces(n=24, *, broken_from=None, request_prefix="r"):
    """A two-node workflow, optionally rewording its first question part-way."""
    rows = []
    for i in range(n):
        broken = broken_from is not None and i >= broken_from
        rid = f"{request_prefix}{i}"
        message = f"message {i % 8}"
        rows.append(
            DecisionTrace(
                workflow_id="support",
                node_id="classify",
                timestamp=T0 + timedelta(minutes=i),
                state={"message": message},
                question_version="v2" if broken else "v1",
                questions=[QuestionSpec(name="intent", type="choice",
                                        instructions="Short." if broken else "Which action?",
                                        criteria=CRITERIA)],
                answers=[DecisionAnswer(
                    question_name="intent", type="choice",
                    selected="status" if broken else "refund",
                    probabilities={"refund": 0.2, "status": 0.5, "other": 0.3} if broken
                    else {"refund": 0.98, "status": 0.01, "other": 0.01})],
                action="status" if broken else "refund",
                latency_ms=700,
                metadata={"request_id": rid},
            )
        )
        rows.append(
            DecisionTrace(
                workflow_id="support",
                node_id="handoff",
                timestamp=T0 + timedelta(minutes=i, seconds=30),
                state={"message": message, "category": rows[-1].action},
                question_version="h1",
                questions=[QuestionSpec(name="needs_human", type="noul",
                                        instructions="Does this need a person?")],
                answers=[DecisionAnswer(question_name="needs_human", type="noul", value=0.2)],
                action="auto",
                latency_ms=600,
                metadata={"request_id": rid},
            )
        )
    return rows


def test_requests_are_followed_through_the_workflow():
    report = build_report(workflow_traces())
    assert [n["node_id"] for n in report["workflow"]] == ["classify", "handoff"]
    assert report["trajectories"]
    steps = report["trajectories"][0]["steps"]
    assert [s["node_id"] for s in steps] == ["classify", "handoff"]
    assert steps[0]["top"][0][0] == "refund"


def test_without_a_request_id_each_decision_stands_on_its_own():
    """No request id is a missing path, not a missing console."""
    rows = workflow_traces(n=12)
    for row in rows:
        row.metadata = {}
    report = build_report(rows)
    assert report["trajectories"] == []                 # a path was never recorded
    assert report["queue"]["unit"] == "decision"        # ...so the unit is the decision
    assert report["queue"]["total"] == 24
    assert "24 decisions" in report["queue"]["headline"]
    # The nodes are still listed — only the path through them is unknown.
    assert {n["node_id"] for n in report["workflow"]} == {"classify", "handoff"}


def test_the_same_input_before_and_after_a_flagged_change_is_surfaced():
    report = build_report(workflow_traces(n=48, broken_from=36), window_size=12)
    pairs = report["comparisons"]
    assert pairs["total"] > 0
    assert pairs["changed"] > 0
    example = next(p for p in pairs["pairs"] if p["changed"])
    assert example["before"]["branch"] == "refund"
    assert example["after"]["branch"] == "status"
    assert example["node_id"] == "classify"


def test_the_story_names_what_moved_what_changed_and_what_it_did():
    story = build_report(workflow_traces(n=48, broken_from=36), window_size=12)["story"]
    assert story["kind"] == "incident"
    assert story["node_id"] == "classify"
    labels = [b["label"] for b in story["bullets"]]
    assert labels == ["What moved", "What changed underneath", "What it did to requests"]
    assert "certainty" in story["bullets"][0]["text"]
    assert "rewritten" in story["bullets"][1]["text"]


def test_a_clean_stream_gets_a_story_that_claims_nothing():
    story = build_report(workflow_traces(n=48), window_size=12)["story"]
    assert story["kind"] == "clean"
    assert "not proof" in story["bullets"][1]["text"]


def test_state_renders_as_text_a_person_can_read():
    rows = workflow_traces(n=16)
    report = build_report(rows)
    shown = report["decisions"][0]["state"]
    assert shown.startswith("message: ")
    assert "{" not in shown and '"' not in shown


def test_the_queue_triages_requests_rather_than_summarising_them():
    report = build_report(workflow_traces(n=48, broken_from=36), window_size=12)
    queue = report["queue"]
    assert queue["total"] == 48
    assert sum(queue["counts"].values()) == 48
    # Every listed row carries the reason and the action, not just a score.
    for row in queue["rows"]:
        assert row["reason"] and row["advice"]
        assert row["flagged_at"] in {"classify", "handoff"}
        assert any(step["flags"] for step in row["steps"])


def test_a_shaky_decision_the_workflow_made_alone_is_the_one_that_asks_for_you():
    """The distinction the whole console turns on.

    Counting every flagged request as "needs a person" restates what the application
    already decided: a workflow that escalates most of its traffic would show a huge
    number that asks nothing of anyone. What asks something is a flag on a decision
    the workflow answered *by itself*.
    """
    rows = workflow_traces(n=12)
    rows[0].state = {"message": "a one-off message"}      # so no repeat flag confuses it
    rows[0].answers[0].probabilities = {"refund": 0.45, "status": 0.4, "other": 0.15}
    rows[0].action = "refund"          # answered alone, and unsure
    rows[2].state = {"message": "another one-off message"}
    rows[2].answers[0].probabilities = {"refund": 0.45, "status": 0.4, "other": 0.15}
    rows[2].action = "needs_review"    # unsure, but the workflow escalated it

    queue = build_report(rows, unsure_below=0.6)["queue"]
    alone = next(r for r in queue["rows"] if r["status"] == "acted_alone")
    caught = next(r for r in queue["rows"] if r["status"] == "with_person")
    assert alone["reason"] == "model was unsure"
    assert alone["went_to_person"] is False
    assert caught["went_to_person"] is True
    assert queue["counts"]["acted_alone"] == 1
    assert queue["counts"]["with_person"] == 1


def test_the_headline_is_a_rate_and_not_a_bare_count():
    rows = workflow_traces(n=12)
    rows[0].state = {"message": "a one-off message"}
    rows[0].answers[0].probabilities = {"refund": 0.45, "status": 0.4, "other": 0.15}
    rows[0].action = "refund"
    headline = build_report(rows, unsure_below=0.6)["queue"]["headline"]
    assert "of 12 requests" in headline
    assert "1 of those it should not have" in headline


def test_naming_your_own_human_branches_changes_the_split():
    rows = workflow_traces(n=12)
    rows[0].state = {"message": "a one-off message"}
    rows[0].answers[0].probabilities = {"refund": 0.45, "status": 0.4, "other": 0.15}
    rows[0].action = "refund"
    queue = build_report(rows, unsure_below=0.6, human_actions=["refund"])["queue"]
    assert queue["counts"]["acted_alone"] == 0     # "refund" now means a person has it
    assert queue["counts"]["with_person"] == 1
    assert queue["human_actions_inferred"] is False


def test_an_answer_outside_the_schema_outranks_a_softer_flag():
    """The row's reason is its worst flag, not the first one found."""
    rows = workflow_traces(n=48, broken_from=36)
    broken = next(r for r in rows if r.node_id == "classify" and r.question_version == "v2")
    broken.answers[0].probabilities = {"refund": 0.5, "escalate": 0.5}
    queue = build_report(rows, window_size=12)["queue"]
    row = next(r for r in queue["rows"] if r["request_id"] == broken.metadata["request_id"])
    assert row["reason"] == "answer broke the schema"


def test_clean_requests_are_counted_but_not_queued():
    queue = build_report(workflow_traces(n=12))["queue"]
    assert queue["counts"]["clear"] == 12
    assert queue["rows"] == []


def test_the_page_carries_only_what_it_renders():
    """The analysis stays in the report; the page ships the keys it reads."""
    report = build_report(workflow_traces(n=24))
    page = render_html(report)
    assert "__EVALJEV_REPORT__" not in page
    assert 'id="runs"' in page and 'id="audit"' in page

    embedded = json.loads(
        page.split("window.EVALJEV_REPORT = ", 1)[1].split(";\n", 1)[0].replace("<\\/", "</")
    )
    assert set(embedded) == set(PAGE_KEYS)
    # The analysis the page does not draw stays in the report, and in --json.
    assert "reliability" in report and "reliability" not in embedded
    assert "decisions" in report and "decisions" not in embedded


def test_a_run_is_a_stretch_of_traffic_with_one_configuration():
    """Runs are cut where something shipped, not where the arithmetic landed."""
    rows = workflow_traces(n=48, broken_from=24)
    runs = build_report(rows)["runs"]
    assert [r["index"] for r in runs] == [2, 1]          # newest first
    assert runs[1]["verdict"] == "baseline"
    assert runs[1]["requests"] == 24 and runs[0]["requests"] == 24
    assert runs[0]["label"] == "v2"


def test_a_run_that_broke_something_says_what_and_what_to_do():
    run = build_report(workflow_traces(n=48, broken_from=24))["runs"][0]
    assert run["verdict"] == "harmful"
    assert run["effects"] and all(e["direction"] == "worse" for e in run["effects"])
    assert any("instructions" in c["plain"] for c in run["changes"])
    assert "Roll this one back" in run["advice"]
    # ...and the evidence a reader can check by eye.
    assert run["evidence"]["changed"] > 0


def test_a_run_that_changed_nothing_measurable_is_not_called_safe():
    """The wording matters: nothing shown is not the same as nothing wrong."""
    rows = workflow_traces(n=40)
    for row in rows[20:]:
        row.question_version = "v2"      # a version bump that changes no behaviour
    run = build_report(rows)["runs"][0]
    assert run["verdict"] == "no effect shown"
    assert "Not the same as safe" in run["advice"]


def test_the_baseline_pools_the_runs_that_held_still():
    """Two dozen against two dozen cannot show what ninety against two dozen can."""
    rows = workflow_traces(n=72, broken_from=48)
    for row in rows[24:48]:
        row.question_version = "v1b"     # a quiet run in between
    runs = build_report(rows)["runs"]
    newest = runs[0]
    assert newest["compared_against"] == 48      # both earlier runs, pooled
    assert newest["verdict"] == "harmful"


def test_a_caller_can_declare_its_own_runs():
    """A batch job knows what a run is; it should not have to be inferred."""
    rows = workflow_traces(n=24)
    for i, row in enumerate(rows):
        row.metadata = {
            **row.metadata,
            "run_id": "batch-a" if i < 24 else "batch-b",
            "run_outcome": "passed" if i < 24 else "failed",
        }
    runs = build_report(rows)["runs"]
    assert [r["index"] for r in runs] == [2, 1]
    assert {r["outcome"] for r in runs} == {"passed", "failed"}
    # ...and the configuration never changed, so nothing is attributed to the split.
    assert runs[0]["changes"] == []


def test_declared_runs_win_over_inferred_ones():
    rows = workflow_traces(n=24, broken_from=12)      # a config change mid-stream
    for row in rows:
        row.metadata = {**row.metadata, "run_id": "one-long-batch"}
    runs = build_report(rows)["runs"]
    assert len(runs) == 1                              # the caller said it was one run
    assert runs[0]["verdict"] == "baseline"
