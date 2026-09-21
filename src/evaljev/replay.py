from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .models import DecisionAnswer, DecisionTrace, ReplayResult, answer_distribution
from .monitor import _parse_answers
from .stats import paired_comparison, paired_shift, wilson_interval

ActionPolicy = Callable[[dict[str, DecisionAnswer]], str | None]
QuestionBuilder = Callable[[DecisionTrace], Mapping[str, Any]]
StateBuilder = Callable[[DecisionTrace], Any]
OutcomeJudge = Callable[[DecisionTrace, str | None], bool | None]
LabelOf = Callable[[DecisionTrace], str | None]


def replay(
    traces: Iterable[DecisionTrace],
    client: Any,
    *,
    questions: Mapping[str, Any] | QuestionBuilder | None = None,
    state_builder: StateBuilder | None = None,
    policy: ActionPolicy | None = None,
    judge: OutcomeJudge | None = None,
    label_of: LabelOf | None = None,
) -> list[ReplayResult]:
    """Rerun traces through a candidate schema, model or policy.

    Pass ``label_of`` to record which label each trace *should* have landed on.
    That is what lets ``summarize_replay`` measure the probability movement as
    well as the decision flips — a far more sensitive signal on small samples.
    """
    results = []
    for trace in traces:
        q = questions(trace) if callable(questions) else questions
        if q is None:
            q = {
                spec.name: {
                    "type": spec.type,
                    "instructions": spec.instructions,
                    **({"criteria": spec.criteria} if spec.criteria is not None else {}),
                }
                for spec in trace.questions
            }
        state = state_builder(trace) if state_builder else trace.state
        response, _ = client.decide(state=state, questions=q)
        answers = _parse_answers(response, q)
        amap = {a.question_name: a for a in answers}
        new_action = policy(amap) if policy else _default_action(amap)
        new_correct = judge(trace, new_action) if judge else None

        # Keep both distributions so a comparison can measure how far the
        # probability moved, not only whether the decision flipped.
        name = next(iter(amap)) if len(amap) == 1 else None
        old_answer = next((a for a in trace.answers if a.question_name == name), None)
        results.append(
            ReplayResult(
                trace_id=trace.trace_id,
                old_action=trace.action,
                new_action=new_action,
                changed=trace.action != new_action,
                old_correct=trace.outcome_correct,
                new_correct=new_correct,
                expected_label=label_of(trace) if label_of else None,
                old_probabilities=answer_distribution(old_answer) if old_answer else None,
                new_probabilities=answer_distribution(amap[name]) if name else None,
            )
        )
    return results


def summarize_replay(results: Iterable[ReplayResult], *, alpha: float = 0.05) -> dict:
    """Summarize a replay, with an explicit verdict on whether the change is real.

    ``verdict`` is the gate a candidate has to pass before it goes anywhere near
    production. It is paired (McNemar) because replay re-runs the same items, and
    it is conservative: a replay that produced too few changed decisions reports
    "insufficient evidence", never "no difference". Check
    ``min_discordant_needed`` against ``discordant`` to see whether the run could
    have concluded anything at all.

    When the replay recorded an expected label, ``probability_shift`` adds a second,
    far more sensitive comparison: a signed-rank test on how much probability mass
    moved onto the correct label per item. The two answer different questions.
    McNemar asks whether behaviour changed; the signed-rank test asks whether the
    underlying probability improved. A candidate can pass the second and fail the
    first — that is real progress that has not yet crossed a threshold, which is a
    leading indicator rather than a behaviour change.
    """
    rows = list(results)
    changed = sum(r.changed for r in rows)
    known_new = [r for r in rows if r.new_correct is not None]
    comparison = paired_comparison(
        [r.old_correct for r in rows], [r.new_correct for r in rows], alpha=alpha
    )
    deltas = [d for d in (r.probability_delta() for r in rows) if d is not None]
    return {
        "probability_shift": paired_shift(deltas, alpha=alpha) if deltas else None,
        "n": len(rows),
        "changed": changed,
        "change_rate": changed / len(rows) if rows else 0.0,
        "change_rate_ci": wilson_interval(changed, len(rows)),
        "improvements": comparison["improvements"],
        "regressions": comparison["regressions"],
        "discordant": comparison["discordant"],
        "min_discordant_needed": comparison["min_discordant_needed"],
        "p_value": comparison["p_value"],
        "verdict": comparison["verdict"],
        "labelled": comparison["n"],
        "new_accuracy": (
            sum(bool(r.new_correct) for r in known_new) / len(known_new) if known_new else None
        ),
    }


def _default_action(answers: dict[str, DecisionAnswer]) -> str | None:
    if len(answers) != 1:
        return None
    ans = next(iter(answers.values()))
    return ans.selected if ans.type == "choice" else str(ans.value)
