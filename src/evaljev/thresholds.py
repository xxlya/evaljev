from __future__ import annotations

from collections.abc import Sequence


def optimize_binary_threshold(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
    step: float = 0.01,
) -> dict:
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have equal length")
    best = None
    t = 0.0
    while t <= 1.0000001:
        fp = fn = tp = tn = 0
        for p, y in zip(probabilities, labels):
            pred = int(p >= t)
            tp += pred == 1 and y == 1
            tn += pred == 0 and y == 0
            fp += pred == 1 and y == 0
            fn += pred == 0 and y == 1
        cost = fp * false_positive_cost + fn * false_negative_cost
        row = {"threshold": round(t, 6), "cost": cost, "tp": tp, "tn": tn, "fp": fp, "fn": fn}
        if best is None or cost < best["cost"]:
            best = row
        t += step
    return best or {"threshold": 0.5, "cost": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
