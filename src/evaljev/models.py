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
    metadata: dict[str, Any] = Field(default_factory=dict)
