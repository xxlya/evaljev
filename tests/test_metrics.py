from evaljev import DecisionAnswer, DecisionTrace, QuestionSpec
from evaljev.metrics import (
    brier_score,
    calibration_bins,
    distribution_shift,
    expected_calibration_error,
    schema_adherence,
)


def test_metrics_smoke():
    ps = [0.9, 0.8, 0.2, 0.1]
    ys = [1, 1, 0, 0]
    assert brier_score(ps, ys) < 0.1
    assert expected_calibration_error(ps, ys, bins=5) >= 0
    assert distribution_shift({"a": 1, "b": 0}, {"a": 1, "b": 0}) == 0


def _trace(p, correct):
    return DecisionTrace(
        workflow_id="w",
        node_id="n",
        state={},
        questions=[QuestionSpec(name="q", type="choice", instructions="Which?",
                                criteria={"a": "A", "b": "B"})],
        answers=[DecisionAnswer(question_name="q", type="choice", selected="a",
                                probabilities={"a": p, "b": 1 - p})],
        outcome_correct=correct,
    )


def test_calibration_bins_reports_the_curve_not_just_the_gap():
    traces = [_trace(0.95, True) for _ in range(8)] + [_trace(0.55, False) for _ in range(4)]
    rows = calibration_bins(traces)
    # Empty buckets are omitted: a bucket with no decisions is not a data point,
    # and drawing it as zero accuracy would invent a failure.
    assert [(r["low"], r["n"], r["accuracy"]) for r in rows] == [
        (0.5, 4, 0.0),
        (0.9, 8, 1.0),
    ]


def test_calibration_bins_ignores_traces_with_no_recorded_outcome():
    assert calibration_bins([_trace(0.9, None)]) == []


def test_adherence_scores_each_answer_against_its_own_option_set():
    """A stream whose schema changed is the normal case for monitoring.

    Holding every answer to the newest label set reports a schema failure for every
    decision made before the option existed — a false alarm precisely when someone
    is reading the number.
    """
    def trace(labels, chosen_probs):
        return DecisionTrace(
            workflow_id="w",
            node_id="n",
            state={},
            questions=[QuestionSpec(name="q", type="choice", instructions="Which?",
                                    criteria={k: k for k in labels})],
            answers=[DecisionAnswer(question_name="q", type="choice", selected="a",
                                    probabilities=chosen_probs)],
        )

    before = [trace(["a", "b"], {"a": 0.8, "b": 0.2}) for _ in range(3)]
    after = [trace(["a", "b", "c"], {"a": 0.7, "b": 0.2, "c": 0.1}) for _ in range(2)]
    assert schema_adherence(before + after)["q"]["adherence"] == 1.0

    # Pass a fixed contract and the older answers are held to it, as documented.
    forced = schema_adherence(before + after, {"q": ["a", "b", "c"]})
    assert forced["q"]["valid"] == 2
