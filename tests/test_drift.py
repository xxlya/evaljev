"""Drift tests: windowing, unpaired comparison, and noticing without being told."""

from datetime import datetime, timedelta, timezone

import pytest

from evaljev import (
    DecisionAnswer,
    DecisionTrace,
    QuestionSpec,
    compare_rates,
    count_windows,
    drift_report,
    fisher_exact_2x2,
    time_windows,
    trend,
    window_metrics,
)

T0 = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
CRITERIA = {"claude": "Reasoning.", "gemini": "Lookup.", "other": "Neither."}


def trace(minutes=0, action="claude", correct=True, question_version="q1", probs=None):
    return DecisionTrace(
        workflow_id="w",
        node_id="route",
        timestamp=T0 + timedelta(minutes=minutes),
        question_version=question_version,
        state={"request": "x"},
        questions=[
            QuestionSpec(name="route", type="choice", instructions="Which?", criteria=CRITERIA)
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
        latency_ms=100.0,
        outcome_correct=correct,
    )


# --- the unpaired test ------------------------------------------------------


def test_fisher_matches_reference_values():
    assert fisher_exact_2x2(3, 1, 1, 3) == pytest.approx(0.4857, abs=1e-4)
    assert fisher_exact_2x2(10, 0, 0, 10) == pytest.approx(1.083e-5, rel=1e-3)
    assert fisher_exact_2x2(5, 5, 5, 5) == pytest.approx(1.0)


def test_fisher_handles_degenerate_tables():
    assert fisher_exact_2x2(0, 0, 0, 0) == 1.0
    assert fisher_exact_2x2(5, 0, 5, 0) == 1.0  # no variation in the outcome


def test_negative_counts_rejected():
    with pytest.raises(ValueError, match="negative"):
        fisher_exact_2x2(-1, 1, 1, 1)


def test_compare_rates_reports_intervals_and_a_test():
    result = compare_rates(90, 100, 70, 100)
    assert result["delta"] == pytest.approx(-0.20)
    assert result["changed"] is True
    assert result["intervals_overlap"] is False


def test_compare_rates_will_not_call_a_small_difference():
    result = compare_rates(9, 10, 8, 10)
    assert result["changed"] is False
    assert result["intervals_overlap"] is True


# --- windowing --------------------------------------------------------------


def test_time_windows_bucket_by_duration():
    traces = [trace(minutes=m) for m in (0, 5, 10, 70, 75, 140)]
    windows = time_windows(traces, window=timedelta(hours=1))
    assert [len(w) for w in windows] == [3, 2, 1]
    assert [w.index for w in windows] == [0, 1, 2]


def test_time_windows_skip_empty_spans():
    """A quiet period produces no window rather than an empty one."""
    traces = [trace(minutes=0), trace(minutes=500)]
    windows = time_windows(traces, window=timedelta(hours=1))
    assert [len(w) for w in windows] == [1, 1]


def test_time_windows_handle_no_traces():
    assert time_windows([], window=timedelta(hours=1)) == []


def test_count_windows_bucket_by_size():
    windows = count_windows([trace(minutes=m) for m in range(10)], size=4)
    assert [len(w) for w in windows] == [4, 4, 2]


def test_windows_are_ordered_oldest_first_regardless_of_input_order():
    traces = [trace(minutes=m) for m in (50, 0, 100)]
    assert [w.traces[0].timestamp for w in count_windows(traces, size=1)] == [
        T0,
        T0 + timedelta(minutes=50),
        T0 + timedelta(minutes=100),
    ]


@pytest.mark.parametrize("bad", [0, -3])
def test_degenerate_window_sizes_rejected(bad):
    with pytest.raises(ValueError, match="at least 1"):
        count_windows([trace()], size=bad)
    with pytest.raises(ValueError, match="positive duration"):
        time_windows([trace()], window=timedelta(minutes=bad))


def test_window_metrics_describe_a_window():
    window = count_windows([trace(action="claude"), trace(action="gemini")], size=2)[0]
    row = window_metrics(window)
    assert row["n"] == 2
    assert row["action_mix"] == pytest.approx({"claude": 0.5, "gemini": 0.5})
    assert row["accuracy"] == 1.0


def test_window_metrics_include_adherence_when_labels_are_known():
    window = count_windows([trace()], size=1)[0]
    row = window_metrics(window, labels={"route": ["claude", "gemini", "other"]})
    assert row["adherence"]["route"] == 1.0


# --- the report -------------------------------------------------------------


def test_one_window_cannot_be_compared():
    report = drift_report([trace(minutes=m) for m in range(3)], size=10)
    assert report["drifted"] is False
    assert "at least two windows" in report["next_step"]


def test_exactly_one_of_window_or_size():
    with pytest.raises(ValueError, match="exactly one"):
        drift_report([trace()], size=2, window=timedelta(hours=1))
    with pytest.raises(ValueError, match="exactly one"):
        drift_report([trace()])


def test_stable_traffic_does_not_drift():
    report = drift_report([trace(minutes=m) for m in range(40)], size=10)
    assert report["drifted"] is False
    assert report["signals"] == []
    assert len(report["series"]) == 4


def test_a_routing_shift_is_noticed_without_being_told_where_to_look():
    """The point of the module: notice, then attribute."""
    stable = [trace(minutes=m, action="claude") for m in range(30)]
    shifted = [trace(minutes=30 + m, action="gemini") for m in range(10)]
    report = drift_report(stable + shifted, size=10)
    assert report["drifted"] is True
    assert any("action share changed" in s for s in report["signals"])
    assert set(report["moved_actions"]) == {"claude", "gemini"}
    assert report["current"]["n"] == 10
    assert report["reference"]["n"] == 30


def test_an_accuracy_drop_is_tested_not_eyeballed():
    good = [trace(minutes=m, correct=True) for m in range(60)]
    bad = [trace(minutes=60 + m, correct=m % 5 != 0) for m in range(20)]
    report = drift_report(good + bad, size=20)
    assert report["accuracy"]["changed"] is True
    assert report["accuracy"]["delta"] < 0
    assert any("accuracy" in s for s in report["signals"])


def test_drift_with_a_recorded_cause_points_at_the_diff():
    stable = [trace(minutes=m, action="claude") for m in range(30)]
    changed = [
        trace(minutes=30 + m, action="gemini", question_version="q2") for m in range(10)
    ]
    report = drift_report(stable + changed, size=10)
    assert report["drifted"] is True
    assert "question" in report["attribution"]["changed_components"]
    assert "Start from those diffs" in report["next_step"]


def test_drift_with_no_recorded_cause_sends_you_to_the_drift_check():
    stable = [trace(minutes=m, action="claude") for m in range(30)]
    changed = [trace(minutes=30 + m, action="gemini") for m in range(10)]
    report = drift_report(stable + changed, size=10)
    assert report["attribution"]["changed_components"] == []
    assert "vendor_drift_check" in report["next_step"]


def test_series_exposes_a_trend_even_when_nothing_trips():
    traces = []
    for w in range(4):
        for i in range(10):
            # Gemini's share creeps up each window without ever jumping.
            action = "gemini" if i < w else "claude"
            traces.append(trace(minutes=w * 10 + i, action=action))
    report = drift_report(traces, size=10)
    shares = [row["action_mix"].get("gemini", 0.0) for row in report["series"]]
    assert shares == sorted(shares)
    assert trend(report["series"], "n") == [(0, 10), (1, 10), (2, 10), (3, 10)]


# --- comparability ----------------------------------------------------------


def node_trace(minutes, node, action="claude"):
    t = trace(minutes=minutes, action=action)
    t.node_id = node
    return t


def test_a_changed_traffic_mix_is_flagged_as_a_confound_not_a_finding():
    """Traffic ordered by type — a benchmark sweep, a nightly batch — breaks
    comparability silently. Found on real JevBench traces, which sweep family by
    family, so windowing by count compared different task mixes."""
    early = [node_trace(m, "intent") for m in range(20)]
    late = [node_trace(20 + m, "routing", action="gemini") for m in range(10)]
    report = drift_report(early + late, size=10)
    assert report["comparable"] is False
    assert report["composition_shift"] > 0.9
    assert "not comparable" in report["signals"][0]
    assert "Compare like with like" in report["next_step"]


def test_same_traffic_mix_is_comparable():
    traces = [node_trace(m, "intent" if m % 2 else "routing") for m in range(40)]
    report = drift_report(traces, size=10)
    assert report["comparable"] is True
    assert report["composition_shift"] < 0.1


def test_comparability_warning_precedes_the_other_signals():
    early = [node_trace(m, "intent") for m in range(20)]
    late = [node_trace(20 + m, "routing", action="gemini") for m in range(10)]
    signals = drift_report(early + late, size=10)["signals"]
    assert "composition" in signals[0]
    assert len(signals) > 1  # the action-share signal is still reported, just after


def test_action_share_is_tested_not_thresholded():
    """A raw JS distance is large on a small window whatever the deployment does.

    Found on real traces: filtering the JevBench stream to one node and windowing
    by 6 tripped a 0.553 JS "shift" that was only small-sample noise.
    """
    # Six traces whose label mix differs from the reference purely by sampling.
    ref = [trace(minutes=m, action="claude" if m % 2 else "gemini") for m in range(18)]
    cur = [trace(minutes=18 + m, action="claude" if m < 4 else "gemini") for m in range(6)]
    report = drift_report(ref + cur, size=6)
    assert report["action_mix_shift"] > 0.1  # the description says "big"
    assert report["moved_actions"] == []  # the test says "not established"
    assert report["drifted"] is False


def test_a_certainty_collapse_fires_without_any_labels():
    """The signal a broken question actually produces.

    Rewording a question rarely makes the model answer outside its schema, and on a
    small window no single action's share reaches significance. What moves is the
    probability on the winning label — so this fires while the action mix, the label
    set and the accuracy column are all unchanged.
    """
    sure = [
        trace(minutes=m, probs={"claude": 0.99, "gemini": 0.005, "other": 0.005})
        for m in range(24)
    ]
    unsure = [
        trace(minutes=24 + m, probs={"claude": 0.55, "gemini": 0.30, "other": 0.15})
        for m in range(12)
    ]
    report = drift_report(sure + unsure, size=12)
    assert report["drifted"] is True
    assert report["moved_actions"] == []          # nothing changed branch
    assert any("certainty moved" in s for s in report["signals"])
    assert report["certainty"]["p_value"] < 0.01
    assert report["certainty"]["median_a"] == 0.99
    assert report["certainty"]["median_b"] == 0.55


def test_a_certainty_test_on_a_thin_window_is_not_quoted():
    """Under eight values a side the approximation is not worth reading."""
    sure = [trace(minutes=m, probs={"claude": 0.99, "gemini": 0.01}) for m in range(6)]
    unsure = [trace(minutes=6 + m, probs={"claude": 0.55, "gemini": 0.45}) for m in range(3)]
    report = drift_report(sure + unsure, size=3)
    assert report["certainty"]["underpowered"] is True
    assert not any("certainty moved" in s for s in report["signals"])


def test_a_significant_but_meaningless_certainty_move_is_suppressed():
    """Significance is not the same as worth reading.

    A distribution piled at 1.00 makes a 0.01 move highly significant in the ranks.
    The signal needs the move to be large enough to act on as well.
    """
    sure = [trace(minutes=m, probs={"claude": 1.0, "gemini": 0.0}) for m in range(12)]
    barely = [trace(minutes=12 + m, probs={"claude": 0.99, "gemini": 0.01}) for m in range(12)]
    report = drift_report(sure + barely, size=12)
    assert report["certainty"]["p_value"] < 0.05     # the ranks did move
    assert report["certainty"]["median_shift"] < 0.05
    assert report["certainty"]["moved"] is False     # ...by nothing worth reporting
    assert not any("certainty moved" in s for s in report["signals"])
