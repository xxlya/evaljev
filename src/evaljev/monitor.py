from __future__ import annotations

from typing import Any, Callable, Mapping

from .models import DecisionAnswer, DecisionTrace, QuestionSpec
from .store import InMemoryTraceStore, TraceStore


ActionPolicy = Callable[[dict[str, DecisionAnswer]], str | None]


def _question_specs(questions: Mapping[str, Any]) -> list[QuestionSpec]:
    specs = []
    for name, q in questions.items():
        specs.append(
            QuestionSpec(
                name=name,
                type=q["type"],
                instructions=q.get("instructions"),
                criteria=q.get("criteria"),
            )
        )
    return specs


def _parse_answers(response: dict, questions: Mapping[str, Any]) -> list[DecisionAnswer]:
    root = response.get("answers", response)
    answers = []
    for name, q in questions.items():
        raw = root.get(name, {})
        qtype = q["type"]
        if qtype == "choice":
            answers.append(
                DecisionAnswer(
                    question_name=name,
                    type="choice",
                    selected=raw.get("choice"),
                    confidence=raw.get("confidence"),
                    probabilities=raw.get("probabilities"),
                )
            )
        elif qtype == "score":
            probs = raw.get("probabilities")
            if isinstance(probs, list):
                probs = {str(i): float(p) for i, p in enumerate(probs)}
            answers.append(
                DecisionAnswer(
                    question_name=name,
                    type="score",
                    value=raw.get("score"),
                    confidence=raw.get("confidence"),
                    probabilities=probs,
                )
            )
        else:
            noul = raw.get("noul") if isinstance(raw, dict) else raw
            answers.append(DecisionAnswer(question_name=name, type="noul", value=noul))
    return answers


class Monitor:
    def __init__(self, store: TraceStore | None = None) -> None:
        self.store = store or InMemoryTraceStore()

    def run(
        self,
        client: Any,
        *,
        workflow_id: str,
        node_id: str,
        state: Any,
        questions: Mapping[str, Any],
        policy: ActionPolicy | None = None,
        question_version: str | None = None,
        policy_version: str | None = None,
        workflow_version: str | None = None,
        metadata: dict | None = None,
    ) -> tuple[dict, DecisionTrace]:
        response, latency_ms = client.decide(state=state, questions=questions)
        answers = _parse_answers(response, questions)
        answer_map = {a.question_name: a for a in answers}
        action = policy(answer_map) if policy else None
        model = response.get("model") or getattr(client, "model", None)
        trace = DecisionTrace(
            workflow_id=workflow_id,
            node_id=node_id,
            model=model,
            model_version=model,
            question_version=question_version,
            policy_version=policy_version,
            workflow_version=workflow_version,
            state=state,
            questions=_question_specs(questions),
            answers=answers,
            latency_ms=latency_ms,
            action=action,
            metadata=metadata or {},
        )
        self.store.append(trace)
        return response, trace

    def record_outcome(
        self,
        trace: DecisionTrace,
        *,
        outcome: Any,
        correct: bool | None = None,
    ) -> DecisionTrace:
        trace.outcome = outcome
        trace.outcome_correct = correct
        # stores that persist immutable lines should append the enriched trace; users can
        # de-duplicate by trace_id during analytics. In-memory objects update in place.
        return trace
