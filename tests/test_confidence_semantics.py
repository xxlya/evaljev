"""Jev's `confidence` is a normalized margin, not a probability.

    c = (p_max - 1/k) / (1 - 1/k)

Verified against 48 live choice/score answers from the JevBench `original` split
(jev-1.13.0): max absolute error 0.0100, which is the reporting granularity of the
two-decimal probabilities. These tests pin the consequences for calibration.
"""

import pytest

from evaljev import (
    DecisionAnswer,
    DecisionTrace,
    QuestionSpec,
    calibration_probability,
    calibration_report,
    confidence_to_pmax,
    pmax_to_confidence,
)


@pytest.mark.parametrize(
    "pmax,k,confidence",
    [
        (0.99, 5, 0.9875),  # observed: intent, conf 0.98
        (1.00, 4, 1.0),  # observed: ordinal, conf 0.99
        (0.60, 2, 0.20),  # the uncertain regime, where the gap is widest
        (0.50, 2, 0.0),  # a coin flip reports zero confidence
        (0.25, 4, 0.0),  # uniform over k reports zero confidence
    ],
)
def test_confidence_is_a_normalized_margin(pmax, k, confidence):
    assert pmax_to_confidence(pmax, k) == pytest.approx(confidence)
    assert confidence_to_pmax(confidence, k) == pytest.approx(pmax)


@pytest.mark.parametrize("k", [2, 3, 4, 5, 10])
@pytest.mark.parametrize("pmax", [0.3, 0.55, 0.8, 0.95])
def test_round_trip(k, pmax):
    assert confidence_to_pmax(pmax_to_confidence(pmax, k), k) == pytest.approx(pmax)


def test_a_confidence_threshold_means_a_different_probability_per_option_count():
    """The same threshold is a different decision rule for each schema.

    This is why a criteria map gaining one option silently moves every threshold
    keyed on confidence, with no model change and no code change.
    """
    at_k2 = confidence_to_pmax(0.55, 2)
    at_k10 = confidence_to_pmax(0.55, 10)
    assert at_k2 == pytest.approx(0.775)
    assert at_k10 == pytest.approx(0.595)
    assert at_k2 - at_k10 > 0.17


@pytest.mark.parametrize("k", [1, 0, -3])
def test_option_count_below_two_is_rejected(k):
    with pytest.raises(ValueError, match="at least 2"):
        confidence_to_pmax(0.5, k)
    with pytest.raises(ValueError, match="at least 2"):
        pmax_to_confidence(0.5, k)


# --- what calibration actually scores ---------------------------------------


def test_calibration_prefers_the_distribution_over_confidence():
    ans = DecisionAnswer(
        question_name="q",
        type="choice",
        selected="a",
        confidence=0.2,  # the margin
        probabilities={"a": 0.6, "b": 0.4},  # the probability
    )
    assert calibration_probability(ans) == pytest.approx(0.6)


def test_calibration_falls_back_to_confidence_without_a_distribution():
    ans = DecisionAnswer(question_name="q", type="choice", selected="a", confidence=0.77)
    assert calibration_probability(ans) == pytest.approx(0.77)


def test_noul_scores_its_own_probability():
    ans = DecisionAnswer(question_name="q", type="noul", value=0.33)
    assert calibration_probability(ans) == pytest.approx(0.33)


def test_answer_with_nothing_to_score():
    assert calibration_probability(DecisionAnswer(question_name="q", type="choice")) is None


def trace(probs, confidence, correct):
    return DecisionTrace(
        workflow_id="w",
        node_id="n",
        state={},
        questions=[QuestionSpec(name="q", type="choice", instructions="i")],
        answers=[
            DecisionAnswer(
                question_name="q",
                type="choice",
                selected="a",
                confidence=confidence,
                probabilities=probs,
            )
        ],
        outcome_correct=correct,
    )


def test_calibration_report_uses_probabilities_not_margins():
    """Two binary decisions at p_max 0.6, one right and one wrong.

    Scored on probability the model is perfectly calibrated (mean 0.6 vs accuracy
    0.5 is within one bin's tolerance here); scored on the margin 0.2 it would look
    badly underconfident. The margin is simply not the quantity ECE is defined on.
    """
    traces = [
        trace({"a": 0.6, "b": 0.4}, 0.2, True),
        trace({"a": 0.6, "b": 0.4}, 0.2, False),
    ]
    report = calibration_report(traces)["q"]
    assert report["n"] == 2
    assert report["mean_probability"] == pytest.approx(0.6)
    assert report["accuracy"] == pytest.approx(0.5)
    assert report["ece"] == pytest.approx(0.1)
    # Had it scored the margin, mean would be 0.2 and ECE would be 0.3.
    assert report["brier"] == pytest.approx(0.26)
