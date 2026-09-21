from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

QuestionType = Literal["choice", "noul", "score"]


class QuestionSpec(BaseModel):
    name: str
    type: QuestionType
    instructions: Any
    criteria: Any | None = None


def question_specs(questions: Mapping[str, Any] | Iterable[QuestionSpec]) -> list[QuestionSpec]:
    """Normalize a question schema into ``QuestionSpec`` objects.

    Accepts the raw ``{name: {"type": ..., "instructions": ...}}`` mapping sent to
    the API, so the same object can be linted, traced and requested.
    """
    if isinstance(questions, Mapping):
        return [
            QuestionSpec(
                name=name,
                type=q["type"],
                instructions=q.get("instructions"),
                criteria=q.get("criteria"),
            )
            for name, q in questions.items()
        ]
    return [q if isinstance(q, QuestionSpec) else QuestionSpec(**q) for q in questions]


class DecisionAnswer(BaseModel):
    question_name: str
    type: QuestionType
    selected: str | None = None
    value: float | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None


class DecisionTrace(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    workflow_id: str
    node_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: str | None = None
    model_version: str | None = None
    question_version: str | None = None
    policy_version: str | None = None
    workflow_version: str | None = None
    state: Any
    questions: list[QuestionSpec]
    answers: list[DecisionAnswer]
    latency_ms: float | None = None
    action: str | None = None
    outcome: Any | None = None
    outcome_correct: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReplayResult(BaseModel):
    trace_id: str
    old_action: str | None
    new_action: str | None
    changed: bool
    old_correct: bool | None = None
    new_correct: bool | None = None
    # Kept so a comparison can measure how far the probability moved, not only
    # whether the decision flipped. `expected_label` is the ground-truth label the
    # mass should be on, when the caller can supply one.
    expected_label: str | None = None
    old_probabilities: dict[str, float] | None = None
    new_probabilities: dict[str, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def probability_delta(self) -> float | None:
        """Change in probability mass on the correct label, or None if unknown."""
        if self.expected_label is None or not self.old_probabilities or not self.new_probabilities:
            return None
        return self.new_probabilities.get(self.expected_label, 0.0) - self.old_probabilities.get(
            self.expected_label, 0.0
        )


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
