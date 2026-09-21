from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .metrics import distribution_shift
from .models import DecisionAnswer
from .monitor import _parse_answers
from .stats import wilson_interval


def answer_branch(ans: DecisionAnswer) -> str | None:
    """The branch a decision actually took, comparable across phrasings.

    ``selected`` is only populated for choice answers. A score answer carries a
    level, a noul answer a bare probability — so comparing ``selected`` alone
    reports every noul and score decision as perfectly stable no matter what the
    model did. Each type gets the label its branch is keyed on:

    - choice: the selected label, or the distribution's argmax
    - score:  the argmax level, since that is what an ordinal branch keys on
    - noul:   the proposition's truth at the natural 0.5 cut
    """
    if ans.type == "noul":
        return None if ans.value is None else ("yes" if ans.value >= 0.5 else "no")
    if ans.selected is not None:
        return ans.selected
    if ans.probabilities:
        return max(sorted(ans.probabilities), key=lambda k: ans.probabilities[k])
    return None if ans.value is None else str(ans.value)


def answer_distribution(ans: DecisionAnswer) -> dict[str, float] | None:
    """The answer's distribution in label space, or None if it has none.

    A noul answer reports a bare P(true), so every distribution-based metric sees
    ``None`` and silently skips it. Expanding it to ``{"yes": p, "no": 1 - p}``
    makes noul comparable with the other two types.
    """
    if ans.probabilities:
        return ans.probabilities
    if ans.type == "noul" and ans.value is not None:
        p = float(ans.value)
        return {"yes": p, "no": 1.0 - p}
    return None


def stability_check(
    client: Any,
    *,
    states: Iterable[Any],
    questions: Mapping[str, Any],
    question_name: str,
    base_index: int = 0,
    repeats: int = 1,
) -> dict:
    """Does the decision survive a rephrasing of the same request?

    Every state is compared against ``states[base_index]``, which is excluded from
    its own comparison — a base state never flips relative to itself, and counting
    it only dilutes the rate.

    ``repeats > 1`` queries each state several times to measure the **noise floor**:
    how often the branch moves when the input does not move at all. Paraphrase
    instability is only meaningful above that floor. It costs
    ``len(states) * repeats`` calls, so it defaults to off.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    rows: list[list[DecisionAnswer]] = []
    for state in states:
        answers = []
        for _ in range(repeats):
            response, _ = client.decide(state=state, questions=questions)
            answers.append({a.question_name: a for a in _parse_answers(response, questions)}[question_name])
        rows.append(answers)

    if not rows:
        return {"n": 0, "repeats": repeats, "branch_flip_rate": 0.0, "comparisons": 0}

    base = rows[base_index][0]
    base_branch = answer_branch(base)
    base_probs = answer_distribution(base)

    flips, comparisons, shifts = 0, 0, []
    for i, answers in enumerate(rows):
        if i == base_index:
            continue
        comparisons += 1
        if answer_branch(answers[0]) != base_branch:
            flips += 1
        other_probs = answer_distribution(answers[0])
        if base_probs and other_probs:
            shifts.append(distribution_shift(base_probs, other_probs))

    lo, hi = wilson_interval(flips, comparisons)
    out = {
        "n": len(rows),
        "repeats": repeats,
        "base_branch": base_branch,
        "comparisons": comparisons,
        "flips": flips,
        "branch_flip_rate": flips / comparisons if comparisons else 0.0,
        "branch_flip_ci": (lo, hi),
        "mean_distribution_shift": sum(shifts) / len(shifts) if shifts else None,
    }

    if repeats > 1:
        # Same input, repeated: anything that moves here is the API's own variance.
        noise_flips, noise_comparisons, noise_shifts = 0, 0, []
        for answers in rows:
            first = answers[0]
            first_branch = answer_branch(first)
            first_probs = answer_distribution(first)
            for other in answers[1:]:
                noise_comparisons += 1
                if answer_branch(other) != first_branch:
                    noise_flips += 1
                other_probs = answer_distribution(other)
                if first_probs and other_probs:
                    noise_shifts.append(distribution_shift(first_probs, other_probs))
        noise_rate = noise_flips / noise_comparisons if noise_comparisons else 0.0
        nlo, nhi = wilson_interval(noise_flips, noise_comparisons)
        out["noise_floor"] = {
            "branch_flip_rate": noise_rate,
            "branch_flip_ci": (nlo, nhi),
            "comparisons": noise_comparisons,
            "flips": noise_flips,
            "mean_distribution_shift": (
                sum(noise_shifts) / len(noise_shifts) if noise_shifts else None
            ),
        }
        # Instability worth acting on is what exceeds the model's own jitter.
        out["excess_flip_rate"] = max(0.0, out["branch_flip_rate"] - noise_rate)
        # Conservative and low-powered: a binary flip rate needs many pairs before
        # its interval clears the noise interval.
        out["exceeds_noise"] = lo > nhi
        # Far more sensitive, because it uses the whole distribution rather than
        # which side of the threshold it landed on. A large ratio alongside
        # exceeds_noise=False means a probably-real instability that the binary
        # test has too few pairs to certify — collect more paraphrases.
        paraphrase_shift = out["mean_distribution_shift"]
        noise_shift = out["noise_floor"]["mean_distribution_shift"]
        out["shift_ratio"] = (
            paraphrase_shift / noise_shift
            if paraphrase_shift is not None and noise_shift
            else None
        )

    return out
