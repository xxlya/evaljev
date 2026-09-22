"""Watch a deployment over time instead of comparing two piles by hand.

``attribute`` answers "what changed between these two windows" but expects you to
have already chosen the windows and already noticed something was wrong. That is
the wrong way round for monitoring, where the unit is a **delta against a
baseline** and the first job is noticing at all.

This module cuts a trace stream into windows, reports the series so a trend is
visible, and compares the most recent window against the ones before it. Windows
are *not* paired — they are different traffic — so comparisons here use exact
tests for independent samples, not the paired tests replay uses.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from .attribution import action_mix, attribute, behaviour_summary
from .metrics import distribution_shift, schema_adherence
from .models import DecisionTrace, answer_distribution
from .stats import compare_rates, rank_sum_test


@dataclass
class Window:
    """A slice of the trace stream."""

    index: int
    start: datetime
    end: datetime
    traces: list[DecisionTrace]

    def __len__(self) -> int:
        return len(self.traces)


def _sorted(traces: Iterable[DecisionTrace]) -> list[DecisionTrace]:
    return sorted(traces, key=lambda t: t.timestamp)


def time_windows(traces: Iterable[DecisionTrace], *, window: timedelta) -> list[Window]:
    """Cut the stream into fixed-duration windows. Empty spans are skipped."""
    if window <= timedelta(0):
        raise ValueError("window must be a positive duration")
    rows = _sorted(traces)
    if not rows:
        return []
    out: list[Window] = []
    start = rows[0].timestamp
    bucket: list[DecisionTrace] = []
    for trace in rows:
        while trace.timestamp >= start + window:
            if bucket:
                out.append(Window(len(out), start, start + window, bucket))
                bucket = []
            start = start + window
        bucket.append(trace)
    if bucket:
        out.append(Window(len(out), start, rows[-1].timestamp, bucket))
    return out


def count_windows(traces: Iterable[DecisionTrace], *, size: int) -> list[Window]:
    """Cut the stream into fixed-size windows, oldest first.

    Useful when traffic is bursty and clock time says more about business hours
    than about the deployment.
    """
    if size < 1:
        raise ValueError("size must be at least 1")
    rows = _sorted(traces)
    out = []
    for i, start in enumerate(range(0, len(rows), size)):
        chunk = rows[start : start + size]
        out.append(Window(i, chunk[0].timestamp, chunk[-1].timestamp, chunk))
    return out


def _certainties(traces: Sequence[DecisionTrace]) -> list[float]:
    """Probability on each answer's leading label — the label-free health signal."""
    out = []
    for trace in traces:
        for ans in trace.answers:
            dist = answer_distribution(ans)
            if dist:
                out.append(max(dist.values()))
    return out


def _node_mix(traces: Sequence[DecisionTrace]) -> dict[str, float]:
    """Share of traffic per decision node — what was asked, not what was answered."""
    counts: dict[str, int] = {}
    for trace in traces:
        counts[trace.node_id] = counts.get(trace.node_id, 0) + 1
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def window_metrics(window: Window, *, labels: Mapping[str, Sequence[str]] | None = None) -> dict:
    """The label-free description of one window, plus accuracy where known."""
    summary = behaviour_summary(window.traces)
    row = {
        "index": window.index,
        "start": window.start.isoformat(),
        "end": window.end.isoformat(),
        **summary,
        "cost_usd": round(
            sum(t.metadata.get("cost_usd") or 0.0 for t in window.traces), 6
        ),
    }
    if labels:
        row["adherence"] = {
            q: stats["adherence"] for q, stats in schema_adherence(window.traces, labels).items()
        }
    return row


def drift_report(
    traces: Iterable[DecisionTrace],
    *,
    window: timedelta | None = None,
    size: int | None = None,
    labels: Mapping[str, Sequence[str]] | None = None,
    alpha: float = 0.05,
    action_shift_threshold: float = 0.1,
    composition_threshold: float = 0.1,
    certainty_shift_threshold: float = 0.05,
    confidence_threshold: float | None = None,
) -> dict:
    """Compare the newest window against everything before it, and say why.

    Pass exactly one of ``window`` (a duration) or ``size`` (a trace count). The
    reference is every earlier window pooled, which gives the baseline the
    tightest interval the history supports.

    The result carries the per-window ``series`` so a trend is visible even when
    no single step trips a threshold, and the ``attribution`` of whatever did.

    ``action_mix_shift`` is reported as a description only. The signal that fires
    is ``moved_actions``, where each action's share is tested exactly — a JS
    distance on a ten-trace window is large whatever the deployment is doing.
    ``comparable`` is checked first: if the windows carry different mixes of
    decision nodes, nothing below it can be read as deployment drift.

    ``certainty`` tests how sure the model was as a distribution, which is the
    signal a reworded question actually produces: valid answers, a plausible
    label, and much less probability on it. It fires only when the rank test
    clears ``alpha`` *and* the median moved at least
    ``certainty_shift_threshold``, since significance alone will report a 0.01
    move on a distribution that sits at 1.00.
    """
    if (window is None) == (size is None):
        raise ValueError("pass exactly one of window= or size=")
    windows = time_windows(traces, window=window) if window else count_windows(traces, size=size)
    if len(windows) < 2:
        return {
            "windows": len(windows),
            "series": [window_metrics(w, labels=labels) for w in windows],
            "drifted": False,
            "next_step": "Need at least two windows to compare. Collect more traffic.",
        }

    current = windows[-1]
    reference_traces = [t for w in windows[:-1] for t in w.traces]

    ref_mix, cur_mix = action_mix(reference_traces), action_mix(current.traces)
    mix_shift = distribution_shift(ref_mix, cur_mix)

    # Every other signal assumes the windows are samples of comparable traffic.
    # If the mix of decision nodes changed, they are not, and a shift in behaviour
    # cannot be told apart from a shift in what was asked. Traffic ordered by type
    # — a benchmark sweep, a nightly batch job — breaks this silently, so it is
    # checked first and reported as a confound rather than as a finding.
    composition_shift = distribution_shift(
        _node_mix(reference_traces), _node_mix(current.traces)
    )
    comparable = composition_shift < composition_threshold

    ref_labelled = [t for t in reference_traces if t.outcome_correct is not None]
    cur_labelled = [t for t in current.traces if t.outcome_correct is not None]
    accuracy = None
    if ref_labelled and cur_labelled:
        accuracy = compare_rates(
            sum(bool(t.outcome_correct) for t in ref_labelled),
            len(ref_labelled),
            sum(bool(t.outcome_correct) for t in cur_labelled),
            len(cur_labelled),
            alpha=alpha,
        )

    # How sure the model was, tested as a distribution rather than as a rate. A
    # broken question usually keeps answering inside its schema and keeps picking a
    # plausible label; what collapses is the probability on the winner. Counting
    # how many answers crossed a threshold discards almost all of that evidence,
    # which is why this is the signal that fires without labels when wording breaks.
    certainty = rank_sum_test(
        _certainties(reference_traces), _certainties(current.traces), alpha=alpha
    )

    adherence = None
    if labels:
        ref_stats = schema_adherence(reference_traces, labels)
        cur_stats = schema_adherence(current.traces, labels)
        adherence = {
            q: compare_rates(
                ref_stats[q]["valid"],
                ref_stats[q]["n"],
                cur_stats[q]["valid"],
                cur_stats[q]["n"],
                alpha=alpha,
            )
            for q in sorted(set(ref_stats) & set(cur_stats))
        }

    attribution = attribute(
        reference_traces,
        current.traces,
        confidence_threshold=confidence_threshold,
        action_shift_threshold=action_shift_threshold,
    )

    # A JS distance is a description, not a test: on a small window it is large
    # by construction. Each action's share gets an exact test of its own, so the
    # signal fires on evidence rather than on window size.
    actions = sorted(set(ref_mix) | set(cur_mix))
    action_tests = {
        a: compare_rates(
            sum(1 for t in reference_traces if (t.action or "<none>") == a),
            len(reference_traces),
            sum(1 for t in current.traces if (t.action or "<none>") == a),
            len(current.traces),
            alpha=alpha,
        )
        for a in actions
    }
    moved_actions = [a for a, row in action_tests.items() if row["changed"]]

    signals = []
    if not comparable:
        signals.append(
            f"traffic composition changed (JS {composition_shift:.3f}) — the windows are "
            "not comparable, and every signal below may reflect that rather than the deployment"
        )
    if moved_actions:
        detail = ", ".join(
            f"{a} {action_tests[a]['reference']['rate']:.2f}->{action_tests[a]['current']['rate']:.2f} "
            f"(p={action_tests[a]['p_value']:.4f})"
            for a in moved_actions
        )
        signals.append(f"action share changed: {detail}")
    # Two gates, because either alone is wrong. Significance alone flags a median
    # move of 0.01 on a distribution piled at 1.00 — real in the ranks, worth
    # nothing to anyone. Size alone flags small-window noise.
    certainty_moved = (
        certainty["changed"]
        and not certainty["underpowered"]
        and (certainty["median_shift"] or 0.0) >= certainty_shift_threshold
    )
    certainty["moved"] = certainty_moved
    certainty["shift_threshold"] = certainty_shift_threshold
    if certainty_moved:
        signals.append(
            f"certainty moved: median {certainty['median_a']:.2f} -> "
            f"{certainty['median_b']:.2f} (p={certainty['p_value']:.4g})"
        )
    if accuracy and accuracy["changed"]:
        signals.append(
            f"accuracy {accuracy['reference']['rate']:.3f} -> {accuracy['current']['rate']:.3f} "
            f"(p={accuracy['p_value']:.4f})"
        )
    if adherence:
        signals += [
            f"schema adherence on {q} {row['reference']['rate']:.3f} -> {row['current']['rate']:.3f}"
            for q, row in adherence.items()
            if row["changed"]
        ]

    drifted = bool(signals)
    if not drifted:
        next_step = "No signal cleared its threshold against the pooled baseline."
    elif not comparable:
        next_step = (
            "The windows carry different mixes of decision nodes. Compare like with like "
            "first — filter to one node_id, or window so each covers the same traffic."
        )
    elif attribution["changed_components"]:
        next_step = (
            f"Drift detected, and {attribution['changed_components']} changed on the traces. "
            "Start from those diffs."
        )
    else:
        next_step = (
            "Drift detected with no recorded component change. Run vendor_drift_check() "
            "on a sample of the reference window."
        )

    return {
        "windows": len(windows),
        "series": [window_metrics(w, labels=labels) for w in windows],
        "reference": {"windows": len(windows) - 1, "n": len(reference_traces)},
        "current": {"index": current.index, "n": len(current.traces)},
        "action_mix_shift": mix_shift,
        "action_tests": action_tests,
        "moved_actions": moved_actions,
        "composition_shift": composition_shift,
        "comparable": comparable,
        "accuracy": accuracy,
        "certainty": certainty,
        "adherence": adherence,
        "signals": signals,
        "drifted": drifted,
        "attribution": attribution,
        "next_step": next_step,
    }


def trend(series: Sequence[Mapping[str, Any]], key: str) -> list[tuple[int, Any]]:
    """Pull one metric out of a window series, for plotting or eyeballing."""
    return [(row["index"], row.get(key)) for row in series]
