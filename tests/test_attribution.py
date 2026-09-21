"""Attribution is a diff, not an inference. These tests pin that it stays one."""

import pytest

from evaljev import (
    DecisionAnswer,
    DecisionTrace,
    QuestionSpec,
    action_mix,
    attribute,
    behaviour_summary,
    schema_diff,
    state_diff,
    vendor_drift_check,
    version_diff,
)

CRITERIA = {"claude": "Long reasoning.", "gemini": "Fast lookup.", "other": "Neither."}


def trace(
    *,
    action="claude",
    state=None,
    criteria=None,
    instructions="Which route handles this?",
    probs=None,
    model_version="jev-1.13.0",
    question_version="q1",
    policy_version="p1",
    correct=None,
    latency=100.0,
):
    return DecisionTrace(
        workflow_id="w",
        node_id="route",
        model_version=model_version,
        question_version=question_version,
        policy_version=policy_version,
        state=state if state is not None else {"request": "x"},
        questions=[
            QuestionSpec(
                name="route",
                type="choice",
                instructions=instructions,
                criteria=CRITERIA if criteria is None else criteria,
            )
        ],
        answers=[
            DecisionAnswer(
                question_name="route",
                type="choice",
                selected=action,
                probabilities=probs or {"claude": 0.9, "gemini": 0.05, "other": 0.05},
            )
        ],
        action=action,
        latency_ms=latency,
        outcome_correct=correct,
    )


# --- version axes -----------------------------------------------------------


def test_model_version_change_is_detected_exactly():
    findings = version_diff([trace()], [trace(model_version="jev-1.14.0")])
    assert len(findings) == 1
    assert findings[0].component == "model"
    assert "jev-1.13.0" in findings[0].summary and "jev-1.14.0" in findings[0].summary


def test_identical_versions_produce_no_findings():
    assert version_diff([trace(), trace()], [trace()]) == []


def test_each_axis_is_reported_separately():
    findings = version_diff(
        [trace()], [trace(model_version="jev-2", question_version="q2", policy_version="p2")]
    )
    assert {f.component for f in findings} == {"model", "question", "policy"}


# --- schema ------------------------------------------------------------------


def test_option_added_is_flagged_with_the_threshold_it_moves():
    """The failure this whole component exists for: k changes, thresholds move."""
    bigger = {**CRITERIA, "both": "Run both."}
    findings = schema_diff([trace()], [trace(criteria=bigger)], confidence_threshold=0.55)
    option = next(f for f in findings if f.component == "option_set")
    assert option.evidence["added"] == ["both"]
    assert option.evidence["k_reference"] == 3
    assert option.evidence["k_current"] == 4
    # confidence<0.55 fires at p_max 0.700 with 3 options, 0.663 with 4.
    assert option.evidence["implied_pmax_before"] == pytest.approx(0.700, abs=0.001)
    assert option.evidence["implied_pmax_after"] == pytest.approx(0.663, abs=0.001)
    assert "p_max" in option.summary


def test_rewritten_instructions_are_flagged():
    findings = schema_diff([trace()], [trace(instructions="Pick the best option.")])
    assert any("instructions rewritten" in f.summary for f in findings)


def test_reworded_criteria_are_flagged_without_a_k_change():
    reworded = {**CRITERIA, "claude": "Totally different description."}
    findings = schema_diff([trace()], [trace(criteria=reworded)])
    assert [f.component for f in findings] == ["schema"]
    assert findings[0].evidence["labels"] == ["claude"]


def test_identical_schema_produces_no_findings():
    assert schema_diff([trace()], [trace()]) == []


# --- state -------------------------------------------------------------------


def test_state_key_changes_are_detected():
    findings = state_diff([trace(state={"request": "x"})], [trace(state={"request": "x", "locale": "de"})])
    assert any(f.evidence.get("added") == ["locale"] for f in findings)


def test_large_state_size_change_is_detected():
    small = [trace(state={"request": "x"}) for _ in range(5)]
    big = [trace(state={"request": "y" * 500}) for _ in range(5)]
    findings = state_diff(small, big)
    assert any("median state size" in f.summary for f in findings)


def test_comparable_states_produce_no_findings():
    assert state_diff([trace()] * 3, [trace()] * 3) == []


# --- behaviour ---------------------------------------------------------------


def test_action_mix_and_summary():
    traces = [trace(action="claude"), trace(action="claude"), trace(action="gemini")]
    assert action_mix(traces) == pytest.approx({"claude": 2 / 3, "gemini": 1 / 3})
    summary = behaviour_summary(traces)
    assert summary["n"] == 3
    assert summary["mean_pmax"] == pytest.approx(0.9)
    assert summary["accuracy"] is None  # unlabelled


def test_behaviour_summary_reports_accuracy_when_labelled():
    summary = behaviour_summary([trace(correct=True), trace(correct=False)])
    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["labelled"] == 2


# --- the whole report --------------------------------------------------------


def test_a_change_with_a_recorded_cause_is_attributed_to_it():
    ref = [trace(action="claude") for _ in range(10)]
    cur = [trace(action="gemini", question_version="q2") for _ in range(10)]
    report = attribute(ref, cur)
    assert report["behaviour_moved"] is True
    assert "question" in report["changed_components"]
    assert report["residual"] is False
    assert "check those diffs first" in report["next_step"]


def test_a_change_with_no_recorded_cause_is_a_residual():
    """Same versions, same schema, same inputs — behaviour moved anyway."""
    ref = [trace(action="claude") for _ in range(10)]
    cur = [trace(action="gemini") for _ in range(10)]
    report = attribute(ref, cur)
    assert report["behaviour_moved"] is True
    assert report["changed_components"] == []
    assert report["residual"] is True
    assert "vendor_drift_check" in report["next_step"]


def test_stable_behaviour_needs_no_attribution():
    report = attribute([trace()] * 10, [trace()] * 10)
    assert report["behaviour_moved"] is False
    assert report["residual"] is False


def test_a_recorded_change_without_a_behaviour_change_is_still_reported():
    """A schema edit that did not move behaviour is still worth surfacing."""
    report = attribute([trace()] * 5, [trace(instructions="Reworded.")] * 5)
    assert report["behaviour_moved"] is False
    assert "schema" in report["changed_components"]


def test_empty_windows_are_rejected():
    with pytest.raises(ValueError, match="at least one trace"):
        attribute([], [trace()])


# --- vendor drift ------------------------------------------------------------


class ReplayClient:
    """Answers with a fixed distribution, standing in for today's model."""

    def __init__(self, probs):
        self.probs = probs
        self.calls = 0

    def decide(self, *, state, questions):
        self.calls += 1
        top = max(sorted(self.probs), key=lambda k: self.probs[k])
        return {"answers": {"route": {"choice": top, "probabilities": self.probs}}}, 5.0


def test_vendor_drift_detected_when_the_model_no_longer_reproduces_itself():
    traces = [trace(probs={"claude": 0.9, "gemini": 0.05, "other": 0.05}) for _ in range(4)]
    client = ReplayClient({"claude": 0.1, "gemini": 0.85, "other": 0.05})
    result = vendor_drift_check(traces, client)
    assert client.calls == 4
    assert result["vendor_drift"] is True
    assert result["branch_flip_rate"] == 1.0
    assert "model changed underneath" in result["verdict"]


def test_no_vendor_drift_points_at_the_input_distribution():
    probs = {"claude": 0.9, "gemini": 0.05, "other": 0.05}
    result = vendor_drift_check([trace(probs=probs) for _ in range(4)], ReplayClient(probs))
    assert result["vendor_drift"] is False
    assert result["mean_distribution_shift"] == pytest.approx(0.0)
    assert "input distribution" in result["verdict"]
