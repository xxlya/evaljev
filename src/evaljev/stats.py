"""Uncertainty for the rates this library reports.

A rate computed from 36 samples is not a number, it is an interval. Reporting
``0.028`` for 1 flip out of 36 hides a 95% interval of [0.005, 0.142] — a
twenty-eight-fold range. Every rate EvalJev reports now carries its interval, and
every before/after comparison carries a test, because the alternative is making
deployment decisions on noise.

Comparisons here are **paired**: replay re-runs the same items through a candidate,
so each item is its own control. That is far more powerful than comparing two
independent accuracies, and it is why the test only looks at items that changed.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from math import comb, sqrt


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves sensibly at 0 and 1 successes, which is exactly where decision rates
    live (a healthy flip rate is near zero).
    """
    if n <= 0:
        return (0.0, 1.0)
    if successes < 0 or successes > n:
        raise ValueError("successes must be between 0 and n")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def rate_with_ci(successes: int, n: int, z: float = 1.96) -> dict:
    """A proportion reported honestly: point estimate plus its interval."""
    lo, hi = wilson_interval(successes, n, z)
    return {
        "rate": successes / n if n else 0.0,
        "ci_low": lo,
        "ci_high": hi,
        "n": n,
        "successes": successes,
    }


def exact_mcnemar(n01: int, n10: int) -> float:
    """Two-sided exact McNemar test on discordant pairs.

    ``n01`` is the count that went wrong -> right, ``n10`` right -> wrong. Items
    that did not change carry no information about direction and are excluded by
    construction: under the null the discordant pairs split 50/50, so this is a
    two-sided binomial test on ``min(n01, n10)`` out of ``n01 + n10``.

    Returns 1.0 when nothing changed — no evidence either way, not agreement.
    """
    if n01 < 0 or n10 < 0:
        raise ValueError("counts must not be negative")
    d = n01 + n10
    if d == 0:
        return 1.0
    tail = sum(comb(d, k) for k in range(min(n01, n10) + 1))
    return min(1.0, 2 * tail / 2**d)


def min_discordant_for_significance(alpha: float = 0.05) -> int:
    """Smallest number of same-direction changes that could ever reach ``alpha``.

    Below this, a comparison cannot produce a significant result no matter how
    lopsided it is — worth knowing before spending on a replay that cannot conclude.
    With the default alpha this is 6.
    """
    d = 1
    while d < 1000:
        if exact_mcnemar(d, 0) < alpha:
            return d
        d += 1
    return d


def paired_comparison(
    before: Sequence[bool | None],
    after: Sequence[bool | None],
    *,
    alpha: float = 0.05,
) -> dict:
    """Compare paired correct/incorrect outcomes and say whether the change is real.

    Pairs where either side is unlabelled are dropped. The verdict is deliberately
    conservative: anything that does not clear ``alpha`` reads as insufficient
    evidence, never as "no difference".
    """
    if len(before) != len(after):
        raise ValueError("before and after must have equal length")
    pairs = [(b, a) for b, a in zip(before, after) if b is not None and a is not None]
    n01 = sum(1 for b, a in pairs if not b and a)
    n10 = sum(1 for b, a in pairs if b and not a)
    p = exact_mcnemar(n01, n10)
    if p < alpha:
        verdict = "improvement" if n01 > n10 else "regression"
    else:
        verdict = "insufficient evidence"
    return {
        "n": len(pairs),
        "improvements": n01,
        "regressions": n10,
        "discordant": n01 + n10,
        "p_value": p,
        "alpha": alpha,
        "verdict": verdict,
        "min_discordant_needed": min_discordant_for_significance(alpha),
    }


def _rank_magnitudes(values: Sequence[float]) -> list[float]:
    """Ranks of |value|, ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: abs(values[i]))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(values[order[j + 1]]) == abs(values[order[i]]):
            j += 1
        shared = (i + j) / 2 + 1  # average of 1-based ranks i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def wilcoxon_signed_rank(differences: Sequence[float], *, zero_tol: float = 1e-12) -> dict:
    """Two-sided exact Wilcoxon signed-rank test on paired differences.

    Where McNemar asks only "did the decision flip", this asks "did the quantity
    move, and by how much". That difference in what is measured is the whole point:
    a candidate that pushes the correct label's probability from 0.30 to 0.49 has
    improved something real, and a test that only counts flips is blind to it.

    Zero differences are dropped, as the test is defined. The null distribution is
    computed exactly by subset-sum over the ranks for small samples, so the p-value
    is trustworthy at the sample sizes decision schemas actually produce.
    """
    nonzero = [d for d in differences if abs(d) > zero_tol]
    n = len(nonzero)
    if n == 0:
        return {"n": 0, "dropped_zeros": len(differences), "statistic": 0.0, "p_value": 1.0}

    ranks = _rank_magnitudes(nonzero)
    w_plus = sum(r for d, r in zip(nonzero, ranks) if d > 0)
    w_minus = sum(r for d, r in zip(nonzero, ranks) if d < 0)
    statistic = min(w_plus, w_minus)

    # Ranks are integers or half-integers; double them so subset sums stay exact.
    doubled = [round(r * 2) for r in ranks]
    total = sum(doubled)
    counts = [0] * (total + 1)
    counts[0] = 1
    for r in doubled:
        for s in range(total, r - 1, -1):
            if counts[s - r]:
                counts[s] += counts[s - r]
    target = round(statistic * 2)
    at_or_below = sum(counts[: target + 1])
    p = min(1.0, 2 * at_or_below / 2**n)

    return {
        "n": n,
        "dropped_zeros": len(differences) - n,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "statistic": statistic,
        "p_value": p,
    }


def paired_shift(
    differences: Sequence[float], *, alpha: float = 0.05, zero_tol: float = 1e-12
) -> dict:
    """Signed-rank test plus the direction and size of the move.

    The verdict names the direction only when the test clears ``alpha``; otherwise
    it is "insufficient evidence", for the same reason as ``paired_comparison``.
    """
    result = wilcoxon_signed_rank(differences, zero_tol=zero_tol)
    moved = [d for d in differences if abs(d) > zero_tol]
    improved = sum(1 for d in moved if d > 0)
    worsened = sum(1 for d in moved if d < 0)
    if result["p_value"] < alpha:
        verdict = "improvement" if improved > worsened else "regression"
    else:
        verdict = "insufficient evidence"
    return {
        **result,
        "alpha": alpha,
        "improved": improved,
        "worsened": worsened,
        "median_delta": statistics.median(moved) if moved else 0.0,
        "mean_delta": statistics.fmean(differences) if differences else 0.0,
        "verdict": verdict,
    }
