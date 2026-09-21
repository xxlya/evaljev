"""Repair tests. The safety properties matter more than the happy path.

No network: the generator is a stub and the client is scripted.
"""

import json

import pytest

from evaljev import (
    Candidate,
    DecisionAnswer,
    DecisionTrace,
    QuestionSpec,
    baseline_questions,
    propose_candidates,
    repair_cycle,
    split_traces,
    validate_candidate,
)

CRITERIA = {
    "claude": "Multi-step reasoning or code.",
    "gemini": "Short factual lookup.",
    "other": "Neither applies.",
}
BASELINE = {
    "type": "choice",
    "instructions": "Which execution strategy should handle this request?",
    "criteria": CRITERIA,
}


def trace(tid, action="claude", expected="claude", correct=True):
    return DecisionTrace(
        trace_id=tid,
        workflow_id="w",
        node_id="route",
        state={"request": f"request {tid}"},
        questions=[
            QuestionSpec(
                name="route",
                type="choice",
                instructions=BASELINE["instructions"],
                criteria=CRITERIA,
            )
        ],
        answers=[DecisionAnswer(question_name="route", type="choice", selected=action)],
        action=action,
        outcome_correct=correct,
        metadata={"expected": expected},
    )


def candidate(instructions="Route to the strategy that minimizes expected failure.", criteria=None):
    return Candidate(
        question_name="route",
        question={
            "type": "choice",
            "instructions": instructions,
            "criteria": CRITERIA if criteria is None else criteria,
        },
    )


# --- the split --------------------------------------------------------------


def test_split_is_deterministic():
    traces = [trace(str(i)) for i in range(40)]
    a1, b1 = split_traces(traces)
    a2, b2 = split_traces(traces)
    assert [t.trace_id for t in a1] == [t.trace_id for t in a2]
    assert [t.trace_id for t in b1] == [t.trace_id for t in b2]


def test_split_partitions_without_overlap():
    traces = [trace(str(i)) for i in range(60)]
    dev, holdout = split_traces(traces)
    assert len(dev) + len(holdout) == 60
    assert {t.trace_id for t in dev}.isdisjoint({t.trace_id for t in holdout})
    assert 15 < len(holdout) < 45  # roughly half, not degenerate


def test_split_fraction_is_respected():
    traces = [trace(str(i)) for i in range(400)]
    _, holdout = split_traces(traces, holdout_fraction=0.25)
    assert 70 < len(holdout) < 130


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.5, 2.0])
def test_degenerate_splits_are_rejected(fraction):
    with pytest.raises(ValueError, match="between 0 and 1"):
        split_traces([trace("a")], holdout_fraction=fraction)


# --- validation: the constraints that protect production --------------------


def test_a_candidate_may_not_add_an_option():
    """Changing k silently moves every confidence threshold downstream."""
    bigger = {**CRITERIA, "both": "Run both models."}
    problems = validate_candidate(candidate(criteria=bigger), BASELINE)
    assert any("label set changed" in p for p in problems)
    assert any("confidence threshold" in p for p in problems)


def test_a_candidate_may_not_remove_an_option():
    smaller = {k: v for k, v in CRITERIA.items() if k != "other"}
    assert any("label set changed" in p for p in validate_candidate(candidate(criteria=smaller), BASELINE))


def test_a_candidate_may_not_change_the_question_type():
    c = candidate()
    c.question["type"] = "noul"
    assert any("question type changed" in p for p in validate_candidate(c, BASELINE))


def test_a_candidate_may_not_be_empty_or_unchanged():
    assert any("instructions are empty" in p for p in validate_candidate(candidate(instructions="  "), BASELINE))
    unchanged = Candidate(question_name="route", question=dict(BASELINE))
    assert any("identical" in p for p in validate_candidate(unchanged, BASELINE))


def test_a_candidate_may_not_gut_a_criterion():
    hollow = {**CRITERIA, "other": ""}
    assert any("no description" in p for p in validate_candidate(candidate(criteria=hollow), BASELINE))


def test_lint_warnings_reject_a_candidate():
    """Subjective wording is caught before it costs a replay."""
    vague = candidate(instructions="Pick whichever option seems appropriate.")
    assert any("lint" in p for p in validate_candidate(vague, BASELINE))


def test_a_clean_candidate_passes():
    assert validate_candidate(candidate(), BASELINE) == []


def test_baseline_is_reconstructed_from_traces():
    assert baseline_questions([trace("a")], "route") == BASELINE
    with pytest.raises(ValueError, match="no question named"):
        baseline_questions([trace("a")], "missing")


# --- proposal parsing -------------------------------------------------------


def test_proposals_are_parsed_from_a_fenced_response():
    raw = (
        "Here you go:\n```json\n"
        f'[{{"instructions": "Route by expected failure.", "criteria": {json.dumps(CRITERIA)},'
        ' "rationale": "sharper"}]\n```'
    )
    got = propose_candidates(BASELINE, [], question_name="route", generate=lambda _: raw)
    assert len(got) == 1
    assert got[0].question["instructions"] == "Route by expected failure."
    assert got[0].rationale == "sharper"


def test_a_proposal_without_criteria_inherits_the_baseline():
    raw = '[{"instructions": "Only the wording changed."}]'
    got = propose_candidates(BASELINE, [], question_name="route", generate=lambda _: raw)
    assert got[0].question["criteria"] == CRITERIA


def test_an_unparseable_response_raises():
    with pytest.raises(ValueError, match="no JSON array"):
        propose_candidates(BASELINE, [], question_name="route", generate=lambda _: "sorry, no")


def test_the_prompt_carries_the_failures_and_the_constraints():
    seen = {}

    def generate(prompt):
        seen["prompt"] = prompt
        return "[]"

    failures = [trace("f1", action="gemini", expected="claude", correct=False)]
    propose_candidates(BASELINE, failures, question_name="route", generate=generate, total=10)
    assert "request f1" in seen["prompt"]
    assert "should have been: claude" in seen["prompt"]
    assert "Keep exactly the same option keys" in seen["prompt"]


# --- the cycle --------------------------------------------------------------


class ScriptedClient:
    """Answers with `verdicts[state_request]` so replay outcomes are exact."""

    def __init__(self, answer_for):
        self.answer_for = answer_for
        self.calls = 0

    def decide(self, *, state, questions):
        self.calls += 1
        choice = self.answer_for(state)
        return {"answers": {"route": {"choice": choice, "probabilities": {choice: 1.0}}}}, 1.0


def judge(trace, action):
    return action == trace.metadata["expected"]


def test_a_candidate_that_fixes_nothing_is_not_approved():
    traces = [trace(str(i), action="gemini", expected="claude", correct=False) for i in range(40)]
    client = ScriptedClient(lambda s: "gemini")  # still wrong
    report = repair_cycle(
        traces,
        client,
        question_name="route",
        generate=lambda _: '[{"instructions": "A different but equally wrong rule."}]',
        judge=judge,
    )
    assert report["approved"] == []
    assert "Nothing to apply" in report["next_step"]


def test_a_candidate_that_fixes_everything_is_approved_on_holdout():
    traces = [trace(str(i), action="gemini", expected="claude", correct=False) for i in range(40)]
    client = ScriptedClient(lambda s: "claude")  # now right
    report = repair_cycle(
        traces,
        client,
        question_name="route",
        generate=lambda _: '[{"instructions": "Prefer the reasoning model for code questions."}]',
        judge=judge,
    )
    assert len(report["approved"]) == 1
    assert report["approved"][0]["verify"]["verdict"] == "improvement"
    assert report["approved"][0]["verify"]["p_value"] < 0.05
    assert "apply manually" in report["next_step"]


def test_an_invalid_candidate_never_reaches_the_api():
    traces = [trace(str(i), correct=False) for i in range(40)]
    client = ScriptedClient(lambda s: "claude")
    bigger = json.dumps({**CRITERIA, "both": "Run both."})
    report = repair_cycle(
        traces,
        client,
        question_name="route",
        generate=lambda _: f'[{{"instructions": "Add an option.", "criteria": {bigger}}}]',
        judge=judge,
    )
    assert client.calls == 0  # rejected at validate, before spending anything
    assert report["candidates"][0]["rejected_at"] == "validate"
    assert report["approved"] == []


def test_the_generator_only_sees_dev_failures():
    """The holdout must stay invisible to whatever writes the candidates."""
    traces = [trace(str(i), action="gemini", expected="claude", correct=False) for i in range(40)]
    dev, holdout = split_traces(traces)
    seen = {}

    def generate(prompt):
        seen["prompt"] = prompt
        return "[]"

    repair_cycle(
        traces, ScriptedClient(lambda s: "claude"), question_name="route", generate=generate, judge=judge
    )
    for t in holdout:
        assert t.state["request"] not in seen["prompt"]
    assert any(t.state["request"] in seen["prompt"] for t in dev)


def test_too_few_traces_to_split_is_an_error():
    with pytest.raises(ValueError, match="dev and a holdout"):
        repair_cycle(
            [trace("a")], ScriptedClient(lambda s: "claude"), question_name="route", generate=lambda _: "[]"
        )
