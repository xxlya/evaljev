"""Offline tests for the JevBench adapter and the schema-adherence metric.

No network: every answer here is constructed by hand.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

from jevbench_adapter import (
    BenchTask,
    expected_level,
    load_split,
    paraphrase_groups,
    to_label_space,
)

from evaljev import DecisionAnswer, DecisionTrace, QuestionSpec, schema_adherence


def task(qtype, labels, expected="yes", family="policy", group="g1", tid="g1-0"):
    return BenchTask(
        id=tid,
        family=family,
        group=group,
        state="s",
        question={"type": qtype, "instructions": "i"},
        labels=labels,
        expected=expected,
    )


# --- noul: a scalar P(true) becomes a two-label distribution -----------------


@pytest.mark.parametrize(
    "p,label",
    [(0.9, "yes"), (0.5, "yes"), (0.49, "no"), (0.04, "no"), (0.0, "no"), (1.0, "yes")],
)
def test_noul_scalar_maps_to_label_and_distribution(p, label):
    ans = DecisionAnswer(question_name="policy", type="noul", value=p)
    predicted, probs = to_label_space(ans, task("noul", ["no", "yes"]))
    assert predicted == label
    assert probs == pytest.approx({"yes": p, "no": 1 - p})
    assert sum(probs.values()) == pytest.approx(1.0)


def test_noul_without_a_value_yields_nothing():
    ans = DecisionAnswer(question_name="policy", type="noul", value=None)
    assert to_label_space(ans, task("noul", ["no", "yes"])) == (None, None)


# --- choice and score: argmax over the model's own distribution -------------


def test_choice_predicts_argmax_not_the_selected_field():
    """The distribution is authoritative; `selected` is only a fallback."""
    ans = DecisionAnswer(
        question_name="intent",
        type="choice",
        selected="cancel",
        probabilities={"cancel": 0.3, "refund": 0.7},
    )
    predicted, probs = to_label_space(ans, task("choice", ["cancel", "refund"]))
    assert predicted == "refund"
    assert probs == {"cancel": 0.3, "refund": 0.7}


def test_choice_falls_back_to_selected_without_a_distribution():
    ans = DecisionAnswer(question_name="intent", type="choice", selected="cancel")
    assert to_label_space(ans, task("choice", ["cancel", "refund"]))[0] == "cancel"


def test_argmax_breaks_ties_on_the_smallest_label():
    ans = DecisionAnswer(
        question_name="intent", type="choice", probabilities={"b": 0.5, "a": 0.5}
    )
    assert to_label_space(ans, task("choice", ["a", "b"]))[0] == "a"


def test_score_uses_argmax_not_the_expected_level():
    """[0.4, 0.2, 0.4] is exactly where argmax and expected value disagree."""
    ans = DecisionAnswer(
        question_name="ordinal",
        type="score",
        value=1.0,
        probabilities={"0": 0.4, "1": 0.2, "2": 0.4},
    )
    predicted, probs = to_label_space(ans, task("score", ["0", "1", "2"]))
    assert predicted == "0"  # tie between "0" and "2", smallest label wins
    assert expected_level(probs) == pytest.approx(1.0)


def test_expected_level_ignores_non_numeric_labels():
    assert expected_level({"yes": 0.5, "no": 0.5}) is None
    assert expected_level(None) is None


# --- dataset loading --------------------------------------------------------


def test_load_split_and_paraphrase_groups(tmp_path):
    rows = [
        {"id": "a-0", "family": "policy", "group": "a", "state": "s1",
         "question": {"type": "noul"}, "labels": ["no", "yes"], "expected": "yes"},
        {"id": "a-1", "family": "policy", "group": "a", "state": "s2",
         "question": {"type": "noul"}, "labels": ["no", "yes"], "expected": "yes"},
        {"id": "b-0", "family": "intent", "group": "b", "state": "s3",
         "question": {"type": "choice"}, "labels": ["x"], "expected": "x"},
    ]
    path = tmp_path / "split.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    tasks = load_split(path)
    assert [t.id for t in tasks] == ["a-0", "a-1", "b-0"]
    # Only groups with more than one phrasing count as paraphrase pairs.
    assert list(paraphrase_groups(tasks)) == ["a"]
    assert [t.id for t in load_split(path, families=["intent"])] == ["b-0"]


# --- schema adherence -------------------------------------------------------


def trace_with(probs, labels_name="policy"):
    return DecisionTrace(
        workflow_id="w",
        node_id="n",
        state={},
        questions=[QuestionSpec(name=labels_name, type="choice", instructions="i")],
        answers=[
            DecisionAnswer(question_name=labels_name, type="choice", probabilities=probs)
        ],
    )


def test_schema_adherence_counts_valid_distributions():
    traces = [
        trace_with({"a": 0.5, "b": 0.5}),
        trace_with({"a": 1.0, "b": 0.0}),
        trace_with({"a": 0.5, "c": 0.5}),  # label outside the declared set
        trace_with({"a": 0.4, "b": 0.4}),  # does not sum to 1
        trace_with(None),  # no distribution at all
    ]
    report = schema_adherence(traces, {"policy": ["a", "b"]})
    assert report["policy"]["n"] == 5
    assert report["policy"]["valid"] == 2
    assert report["policy"]["adherence"] == pytest.approx(0.4)
    assert sum(report["policy"]["reasons"].values()) == 3


def test_schema_adherence_skips_questions_with_no_declared_labels():
    assert schema_adherence([trace_with({"a": 1.0})], {"other": ["a"]}) == {}


def test_schema_adherence_tolerance_accepts_rounded_distributions():
    """Models rounding to three decimals should not be scored as malformed."""
    rounded = trace_with({"a": 0.333, "b": 0.333, "c": 0.334})
    assert schema_adherence([rounded], {"policy": ["a", "b", "c"]})["policy"]["valid"] == 1
    nine_way = trace_with({"a": 0.999, "b": 0.0})
    assert schema_adherence([nine_way], {"policy": ["a", "b"]}, tol=2e-2)["policy"]["valid"] == 1
