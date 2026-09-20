from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from .metrics import distribution_shift
from .monitor import _parse_answers


def stability_check(
    client: Any,
    *,
    states: Iterable[Any],
    questions: Mapping[str, Any],
    question_name: str,
    base_index: int = 0,
) -> dict:
    rows = []
    for state in states:
        response, _ = client.decide(state=state, questions=questions)
        ans = {a.question_name: a for a in _parse_answers(response, questions)}[question_name]
        rows.append(ans)
    if not rows:
        return {"n": 0, "branch_flip_rate": 0.0, "mean_distribution_shift": 0.0}
    base = rows[base_index]
    flips = 0
    shifts = []
    for ans in rows:
        if ans.selected != base.selected:
            flips += 1
        if base.probabilities and ans.probabilities:
            shifts.append(distribution_shift(base.probabilities, ans.probabilities))
    return {
        "n": len(rows),
        "base_choice": base.selected,
        "branch_flip_rate": flips / len(rows),
        "mean_distribution_shift": sum(shifts) / len(shifts) if shifts else None,
    }
