from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from math import sqrt

from .models import DecisionTrace


def _labeled_pairs(traces: Iterable[DecisionTrace]):
    for trace in traces:
        if trace.outcome_correct is None:
            continue
        for ans in trace.answers:
            if ans.confidence is not None:
                yield float(ans.confidence), int(trace.outcome_correct), ans.question_name
            elif ans.type == "noul" and ans.value is not None:
                # For Noul, the probability itself is meaningful only if the observed label
                # corresponds to the proposition. Consumers can store proposition truth in metadata.
                truth = trace.metadata.get("noul_truth", {}).get(ans.question_name)
                if truth is not None:
                    yield float(ans.value), int(bool(truth)), ans.question_name


def expected_calibration_error(
    confidences: Sequence[float], labels: Sequence[int], bins: int = 10
) -> float:
    if len(confidences) != len(labels):
        raise ValueError("confidences and labels must have the same length")
    if not confidences:
        return 0.0
    n = len(confidences)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if lo <= c <= hi if b == bins - 1 or c < hi]
        if not idx:
            continue
        avg_conf = sum(confidences[i] for i in idx) / len(idx)
        avg_acc = sum(labels[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(avg_conf - avg_acc)
    return ece


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")
    if not probabilities:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(probabilities)


def calibration_report(traces: Iterable[DecisionTrace], bins: int = 10) -> dict:
    grouped: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for p, y, q in _labeled_pairs(traces):
        grouped[q].append((p, y))

    report: dict[str, dict] = {}
    for q, rows in grouped.items():
        ps = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        report[q] = {
            "n": len(rows),
            "accuracy": sum(ys) / len(ys),
            "mean_confidence": sum(ps) / len(ps),
            "ece": expected_calibration_error(ps, ys, bins=bins),
            "brier": brier_score(ps, ys),
        }
    return report


def selective_risk_curve(
    confidences: Sequence[float], labels: Sequence[int], thresholds: Sequence[float] | None = None
) -> list[dict]:
    thresholds = thresholds or [i / 20 for i in range(21)]
    rows = []
    for threshold in thresholds:
        kept = [(c, y) for c, y in zip(confidences, labels) if c >= threshold]
        coverage = len(kept) / len(confidences) if confidences else 0.0
        accuracy = sum(y for _, y in kept) / len(kept) if kept else None
        risk = 1 - accuracy if accuracy is not None else None
        rows.append({"threshold": threshold, "coverage": coverage, "accuracy": accuracy, "risk": risk})
    return rows


def branch_flip_rate(before: Sequence[str], after: Sequence[str]) -> float:
    if len(before) != len(after):
        raise ValueError("before and after must have equal length")
    if not before:
        return 0.0
    return sum(a != b for a, b in zip(before, after)) / len(before)


def distribution_shift(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon distance-like score in [0, 1] using base-2 logs."""
    import math

    keys = set(p) | set(q)
    pp = {k: max(float(p.get(k, 0.0)), 0.0) for k in keys}
    qq = {k: max(float(q.get(k, 0.0)), 0.0) for k in keys}
    ps, qs = sum(pp.values()), sum(qq.values())
    if ps == 0 or qs == 0:
        return 1.0
    pp = {k: v / ps for k, v in pp.items()}
    qq = {k: v / qs for k, v in qq.items()}
    mm = {k: (pp[k] + qq[k]) / 2 for k in keys}

    def kl(a, b):
        return sum(v * math.log2(v / b[k]) for k, v in a.items() if v > 0)

    jsd = 0.5 * kl(pp, mm) + 0.5 * kl(qq, mm)
    return sqrt(jsd)
