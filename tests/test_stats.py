"""Tests for the inference layer: intervals, paired tests, and noise floors."""

import pytest

from evaljev import (
    answer_branch,
    exact_mcnemar,
    min_discordant_for_significance,
    paired_comparison,
    rank_sum_test,
    rate_with_ci,
    stability_check,
    summarize_replay,
    wilson_interval,
)
from evaljev.models import DecisionAnswer, ReplayResult

# --- intervals --------------------------------------------------------------


def test_the_interval_that_motivated_this_module():
    """1 flip in 36 paraphrase pairs is not "0.028"."""
    lo, hi = wilson_interval(1, 36)
    assert lo == pytest.approx(0.005, abs=0.001)
    assert hi == pytest.approx(0.142, abs=0.001)
    assert hi - lo > 0.13


def test_interval_shrinks_with_n():
    widths = [
        wilson_interval(round(0.028 * n), n)[1] - wilson_interval(round(0.028 * n), n)[0]
        for n in (36, 100, 400, 1000)
    ]
    assert widths == sorted(widths, reverse=True)


def test_interval_stays_inside_the_unit_range_at_the_edges():
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and 0.0 < hi < 0.25
    lo, hi = wilson_interval(20, 20)
    assert hi == 1.0 and 0.75 < lo < 1.0


def test_no_samples_means_no_information():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_impossible_counts_are_rejected():
    with pytest.raises(ValueError, match="between 0 and n"):
        wilson_interval(5, 3)


def test_rate_with_ci_reports_its_inputs():
    assert rate_with_ci(1, 36)["rate"] == pytest.approx(1 / 36)
    assert rate_with_ci(1, 36)["successes"] == 1
    assert rate_with_ci(0, 0)["rate"] == 0.0


# --- paired test ------------------------------------------------------------


def test_no_change_is_no_evidence_not_agreement():
    assert exact_mcnemar(0, 0) == 1.0


def test_a_single_change_can_never_be_significant():
    """The vague-instructions replay: 1 improvement, 0 regressions, p = 1.0."""
    assert exact_mcnemar(1, 0) == 1.0


@pytest.mark.parametrize("d,p", [(2, 0.5), (3, 0.25), (4, 0.125), (5, 0.0625), (6, 0.03125)])
def test_one_sided_discordance_reaches_significance_at_six(d, p):
    assert exact_mcnemar(d, 0) == pytest.approx(p)


def test_six_is_the_minimum_that_can_conclude_anything():
    assert min_discordant_for_significance(0.05) == 6
    assert min_discordant_for_significance(0.01) == 8


def test_test_is_symmetric_in_direction():
    assert exact_mcnemar(7, 1) == exact_mcnemar(1, 7)


def test_negative_counts_are_rejected():
    with pytest.raises(ValueError, match="negative"):
        exact_mcnemar(-1, 2)


def test_paired_comparison_calls_a_real_improvement():
    result = paired_comparison([False] * 8 + [True] * 20, [True] * 8 + [True] * 20)
    assert result["improvements"] == 8
    assert result["regressions"] == 0
    assert result["verdict"] == "improvement"
    assert result["p_value"] < 0.05


def test_paired_comparison_calls_a_real_regression():
    result = paired_comparison([True] * 8, [False] * 8)
    assert result["verdict"] == "regression"


def test_paired_comparison_refuses_to_conclude_from_one_change():
    result = paired_comparison([True, False, True], [True, True, True])
    assert result["discordant"] == 1
    assert result["verdict"] == "insufficient evidence"
    assert result["min_discordant_needed"] == 6


def test_unlabelled_pairs_are_dropped():
    result = paired_comparison([None, True, False], [True, None, True])
    assert result["n"] == 1
    assert result["improvements"] == 1


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="equal length"):
        paired_comparison([True], [True, False])


# --- replay summary ---------------------------------------------------------


def replay_rows(pairs):
    return [
        ReplayResult(
            trace_id=str(i),
            old_action="a" if old else "b",
            new_action="a" if new else "b",
            changed=old != new,
            old_correct=old,
            new_correct=new,
        )
        for i, (old, new) in enumerate(pairs)
    ]


def test_summarize_replay_gates_on_evidence():
    """A candidate that fixed one item is not a candidate you ship."""
    summary = summarize_replay(replay_rows([(False, True)] + [(True, True)] * 71))
    assert summary["n"] == 72
    assert summary["improvements"] == 1
    assert summary["discordant"] == 1
    assert summary["p_value"] == 1.0
    assert summary["verdict"] == "insufficient evidence"
    assert summary["min_discordant_needed"] == 6


def test_summarize_replay_passes_a_candidate_with_enough_evidence():
    summary = summarize_replay(replay_rows([(False, True)] * 7 + [(True, True)] * 65))
    assert summary["verdict"] == "improvement"
    assert summary["p_value"] < 0.05
    lo, hi = summary["change_rate_ci"]
    assert lo < summary["change_rate"] < hi


def test_summarize_replay_without_labels_cannot_conclude():
    rows = [ReplayResult(trace_id="1", old_action="a", new_action="b", changed=True)]
    summary = summarize_replay(rows)
    assert summary["labelled"] == 0
    assert summary["verdict"] == "insufficient evidence"
    assert summary["new_accuracy"] is None


# --- branch extraction ------------------------------------------------------


@pytest.mark.parametrize(
    "ans,branch",
    [
        (DecisionAnswer(question_name="q", type="choice", selected="a"), "a"),
        (
            DecisionAnswer(question_name="q", type="choice", probabilities={"a": 0.2, "b": 0.8}),
            "b",
        ),
        (DecisionAnswer(question_name="q", type="noul", value=0.86), "yes"),
        (DecisionAnswer(question_name="q", type="noul", value=0.33), "no"),
        (DecisionAnswer(question_name="q", type="noul", value=0.5), "yes"),
        (
            DecisionAnswer(question_name="q", type="score", probabilities={"0": 0.1, "2": 0.9}),
            "2",
        ),
        (DecisionAnswer(question_name="q", type="noul"), None),
    ],
)
def test_branch_is_comparable_across_answer_types(ans, branch):
    assert answer_branch(ans) == branch


# --- stability --------------------------------------------------------------


class ScriptedClient:
    """Returns a queued answer per call, so flips are exactly controlled."""

    model = "fake"

    def __init__(self, values, qtype="noul"):
        self.values = list(values)
        self.qtype = qtype
        self.calls = 0

    def decide(self, *, state, questions):
        v = self.values[self.calls]
        self.calls += 1
        if self.qtype == "noul":
            body = {"noul": v}
        else:
            body = {"choice": v, "probabilities": {v: 1.0}}
        return {"answers": {"q": body}}, 1.0


Q_NOUL = {"q": {"type": "noul", "instructions": "i"}}
Q_CHOICE = {"q": {"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": "B"}}}


def test_noul_instability_is_no_longer_invisible():
    """Before answer_branch, every noul check reported a 0.0 flip rate."""
    client = ScriptedClient([0.86, 0.33, 0.90])
    out = stability_check(client, states=[1, 2, 3], questions=Q_NOUL, question_name="q")
    assert out["base_branch"] == "yes"
    assert out["flips"] == 1
    assert out["comparisons"] == 2  # the base is not compared with itself
    assert out["branch_flip_rate"] == pytest.approx(0.5)


def test_base_state_is_excluded_from_its_own_comparison():
    client = ScriptedClient(["a", "b"], qtype="choice")
    out = stability_check(client, states=[1, 2], questions=Q_CHOICE, question_name="q")
    assert out["comparisons"] == 1
    assert out["branch_flip_rate"] == 1.0  # not 0.5


def test_noise_floor_separates_jitter_from_phrasing_sensitivity():
    # Each state queried twice. State 2 disagrees with the base, and state 3
    # disagrees with *itself* across repeats — that second one is jitter.
    client = ScriptedClient([0.86, 0.85, 0.33, 0.32, 0.90, 0.20])
    out = stability_check(
        client, states=[1, 2, 3], questions=Q_NOUL, question_name="q", repeats=2
    )
    assert client.calls == 6
    assert out["branch_flip_rate"] == pytest.approx(0.5)  # state 2 flipped vs base
    assert out["noise_floor"]["flips"] == 1  # state 3 flipped against itself
    assert out["noise_floor"]["branch_flip_rate"] == pytest.approx(1 / 3)
    assert out["excess_flip_rate"] == pytest.approx(0.5 - 1 / 3)


def test_repeats_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        stability_check(ScriptedClient([]), states=[], questions=Q_NOUL, question_name="q", repeats=0)


def test_no_states_is_handled():
    out = stability_check(ScriptedClient([]), states=[], questions=Q_NOUL, question_name="q")
    assert out["n"] == 0 and out["comparisons"] == 0


def test_noul_distribution_is_derived_for_shift_metrics():
    from evaljev import answer_distribution

    ans = DecisionAnswer(question_name="q", type="noul", value=0.86)
    assert answer_distribution(ans) == pytest.approx({"yes": 0.86, "no": 0.14})
    assert answer_distribution(DecisionAnswer(question_name="q", type="noul")) is None


def test_shift_ratio_is_more_sensitive_than_the_binary_flip_test():
    """The live policy-06 pair: JS shift 0.482 vs a 0.011 noise floor.

    The binary test cannot certify a single flip, but the continuous distance
    separates signal from jitter by a wide margin. Both are reported so a caller
    can see when the evidence is strong but the sample is too small to prove it.
    """
    client = ScriptedClient([0.86, 0.85, 0.86, 0.33, 0.32, 0.33])
    out = stability_check(
        client, states=[1, 2], questions=Q_NOUL, question_name="q", repeats=3
    )
    assert out["branch_flip_rate"] == 1.0
    assert out["noise_floor"]["branch_flip_rate"] == 0.0
    assert out["exceeds_noise"] is False  # one comparison cannot clear the interval
    assert out["shift_ratio"] > 20  # but the distributions are far apart


def test_shift_ratio_is_none_without_a_noise_measurement():
    client = ScriptedClient([0.86, 0.33])
    out = stability_check(client, states=[1, 2], questions=Q_NOUL, question_name="q")
    assert "shift_ratio" not in out
    assert out["mean_distribution_shift"] is not None  # noul now yields a distribution


# --- signed-rank: the sensitive test ----------------------------------------


@pytest.mark.parametrize(
    "diffs,expected",
    [
        ([0.1] * 6, 0.03125),
        ([0.1] * 5, 0.0625),
        ([0.1, -0.2, 0.3, 0.4, 0.5, 0.6], 0.09375),
    ],
)
def test_signed_rank_matches_the_exact_reference_values(diffs, expected):
    from evaljev import wilcoxon_signed_rank

    assert wilcoxon_signed_rank(diffs)["p_value"] == pytest.approx(expected)


def test_signed_rank_drops_zeros_as_the_test_defines():
    from evaljev import wilcoxon_signed_rank

    result = wilcoxon_signed_rank([0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert result["n"] == 6
    assert result["dropped_zeros"] == 2
    assert result["p_value"] == pytest.approx(0.03125)


def test_all_zero_differences_carry_no_evidence():
    from evaljev import wilcoxon_signed_rank

    assert wilcoxon_signed_rank([0.0, 0.0, 0.0])["p_value"] == 1.0


def test_signed_rank_handles_tied_magnitudes():
    from evaljev import wilcoxon_signed_rank

    assert 0 < wilcoxon_signed_rank([0.2, 0.2, 0.2, -0.2, 0.5, 0.5])["p_value"] <= 1.0


def test_it_sees_improvement_that_never_flips_a_decision():
    """The whole reason this test exists.

    Eight items where the correct label gained 0.19 of probability mass. Not one
    crossed a threshold, so McNemar sees zero discordant pairs and reports p=1.0.
    The signed-rank test sees a consistent move and reports it.
    """
    from evaljev import paired_comparison, paired_shift

    deltas = [0.19] * 8
    assert paired_comparison([True] * 8, [True] * 8)["p_value"] == 1.0  # blind to it
    shift = paired_shift(deltas)
    assert shift["verdict"] == "improvement"
    assert shift["p_value"] < 0.01
    assert shift["median_delta"] == pytest.approx(0.19)


def test_paired_shift_names_a_regression():
    from evaljev import paired_shift

    assert paired_shift([-0.2] * 7)["verdict"] == "regression"


def test_paired_shift_will_not_conclude_from_too_little():
    from evaljev import paired_shift

    assert paired_shift([0.4, 0.3])["verdict"] == "insufficient evidence"


def test_both_tests_need_six_but_count_different_things():
    """The power gain stated precisely: six flips versus six movements."""
    from evaljev import min_discordant_for_significance, min_samples_for_signed_rank

    assert min_discordant_for_significance(0.05) == 6  # six *flipped* decisions
    assert min_samples_for_signed_rank(0.05) == 6  # six *moved* probabilities
    assert min_samples_for_signed_rank(0.01) == 8


def test_four_moved_items_cannot_reach_significance():
    """Observed live: holdout of 4 with a +0.90 median shift still cannot conclude."""
    from evaljev import paired_shift

    result = paired_shift([0.9, 0.88, 0.91, 0.86])
    assert result["p_value"] == pytest.approx(0.125)
    assert result["verdict"] == "insufficient evidence"
    assert result["min_samples_needed"] == 6


def test_rank_sum_u_matches_a_brute_force_count():
    """U is the number of (a, b) pairs where a wins, with ties counting a half."""
    a, b = [1.0, 3.0, 5.0, 7.0], [2.0, 3.0, 4.0]
    expected = sum((x > y) + 0.5 * (x == y) for x in a for y in b)
    assert rank_sum_test(a, b)["u"] == expected


def test_rank_sum_p_value_tracks_the_exact_permutation_test():
    """The normal approximation is approximate — but not by much, even at n=6."""
    from itertools import combinations

    a, b = [0.91, 0.94, 0.96, 0.97, 0.98, 0.99], [0.42, 0.55, 0.61, 0.70, 0.77, 0.88]
    pooled = a + b
    n_a = len(a)

    def u_of(group):
        rest = list(pooled)
        for v in group:
            rest.remove(v)
        return sum((x > y) + 0.5 * (x == y) for x in group for y in rest)

    observed = u_of(a)
    mu = n_a * len(b) / 2
    splits = list(combinations(pooled, n_a))
    exact = sum(abs(u_of(s) - mu) >= abs(observed - mu) for s in splits) / len(splits)

    approx = rank_sum_test(a, b)["p_value"]
    assert abs(approx - exact) < 0.01


def test_rank_sum_says_nothing_when_nothing_moved():
    same = [0.2, 0.4, 0.6, 0.8, 1.0, 0.3, 0.5, 0.7]
    result = rank_sum_test(same, list(same))
    assert result["p_value"] == 1.0
    assert result["changed"] is False


def test_rank_sum_survives_a_fully_tied_sample():
    # Every answer at 1.00 is the normal case for a confident decision model, and
    # it drives the tie-corrected variance to zero.
    result = rank_sum_test([1.0] * 8, [1.0] * 8)
    assert result["p_value"] == 1.0
    assert result["changed"] is False


def test_rank_sum_flags_a_sample_too_small_to_read():
    result = rank_sum_test([1.0, 0.9, 0.8], [0.2, 0.1, 0.05])
    assert result["underpowered"] is True
    assert result["min_samples_needed"] == 8
