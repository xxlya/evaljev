from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .models import DecisionAnswer, DecisionTrace, question_specs
from .store import InMemoryTraceStore, TraceStore

ActionPolicy = Callable[[dict[str, DecisionAnswer]], str | None]


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
        # The API returns the exact billed amount per call. Recording it here means
        # every trace can answer "what is this workflow costing" without the caller
        # remembering to plumb it through. A caller's own value always wins.
        meta = dict(metadata or {})
        cost = (response.get("usage") or {}).get("cost_usd")
        if cost is not None and "cost_usd" not in meta:
            meta["cost_usd"] = cost
        trace = DecisionTrace(
            workflow_id=workflow_id,
            node_id=node_id,
            model=model,
            model_version=model,
            question_version=question_version,
            policy_version=policy_version,
            workflow_version=workflow_version,
            state=state,
            questions=question_specs(questions),
            answers=answers,
            latency_ms=latency_ms,
            action=action,
            metadata=meta,
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
        """Attach a downstream outcome and persist the enriched trace.

        The enriched trace is appended rather than edited in place: append-only
        stores keep the full history, and ``TraceStore.list`` returns the newest
        version of each ``trace_id``. Without this second append, an outcome
        recorded after a durable write would never reach the store.
        """
        trace.outcome = outcome
        trace.outcome_correct = correct
        self.store.append(trace)
        return trace
