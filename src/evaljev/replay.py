from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .models import DecisionAnswer, DecisionTrace, ReplayResult
from .monitor import _parse_answers

ActionPolicy = Callable[[dict[str, DecisionAnswer]], str | None]
QuestionBuilder = Callable[[DecisionTrace], Mapping[str, Any]]
StateBuilder = Callable[[DecisionTrace], Any]
OutcomeJudge = Callable[[DecisionTrace, str | None], bool | None]


def replay(
    traces: Iterable[DecisionTrace],
    client: Any,
    *,
    questions: Mapping[str, Any] | QuestionBuilder | None = None,
    state_builder: StateBuilder | None = None,
    policy: ActionPolicy | None = None,
    judge: OutcomeJudge | None = None,
) -> list[ReplayResult]:
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
        results.append(
            ReplayResult(
                trace_id=trace.trace_id,
                old_action=trace.action,
                new_action=new_action,
                changed=trace.action != new_action,
                old_correct=trace.outcome_correct,
                new_correct=new_correct,
            )
        )
    return results


def summarize_replay(results: Iterable[ReplayResult]) -> dict:
    rows = list(results)
    changed = sum(r.changed for r in rows)
    improvements = sum(r.old_correct is False and r.new_correct is True for r in rows)
    regressions = sum(r.old_correct is True and r.new_correct is False for r in rows)
    known_new = [r for r in rows if r.new_correct is not None]
    return {
        "n": len(rows),
        "changed": changed,
        "change_rate": changed / len(rows) if rows else 0.0,
        "improvements": improvements,
        "regressions": regressions,
        "new_accuracy": (
            sum(bool(r.new_correct) for r in known_new) / len(known_new) if known_new else None
        ),
    }


def _default_action(answers: dict[str, DecisionAnswer]) -> str | None:
    if len(answers) != 1:
        return None
    ans = next(iter(answers.values()))
    return ans.selected if ans.type == "choice" else str(ans.value)
