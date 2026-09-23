"""One line of instrumentation, at the call site you already have.

``Monitor.run`` asks you to restructure a call: pass the client in, name the node,
supply a policy, thread a request id through. That is the right shape for a harness
being written from scratch and the wrong shape for a workflow that already exists —
and a tool nobody can adopt in an afternoon does not get adopted.

``watch`` wraps the client instead. Your call sites do not change:

    client = watch(JevHTTPClient(), workflow_id="support-assistant")
    response, latency = client.decide(state=state, questions=questions)   # recorded

Everything else is optional and additive: name the step and group a request when
you want the console to show a path, record an outcome when you learn one.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .models import DecisionTrace, answer_branch
from .monitor import ActionPolicy, Monitor
from .store import JsonlTraceStore, TraceStore

__all__ = ["WatchedClient", "watch"]


@dataclass(frozen=True)
class _Step:
    """What the next decision should be labelled with."""

    node_id: str | None = None
    request_id: str | None = None
    policy: ActionPolicy | None = None
    question_version: str | None = None
    policy_version: str | None = None
    workflow_version: str | None = None
    metadata: dict | None = None


# A context variable rather than an attribute: agents run their steps concurrently,
# and two coroutines sharing one client must not overwrite each other's labels.
_CURRENT: ContextVar[_Step | None] = ContextVar("evaljev_step", default=None)
_UNLABELLED = _Step()


class WatchedClient:
    """A Jev client that records every decision it makes.

    ``decide`` keeps the wrapped client's contract exactly — same arguments, same
    ``(response, latency_ms)`` return — so wrapping is invisible to code that
    already works.
    """

    def __init__(
        self,
        client: Any,
        *,
        workflow_id: str,
        store: TraceStore,
        policy: ActionPolicy | None = None,
        defaults: _Step | None = None,
    ) -> None:
        self.client = client
        self.workflow_id = workflow_id
        self.monitor = Monitor(store)
        self.policy = policy
        self.defaults = defaults or _UNLABELLED
        self.last_trace: DecisionTrace | None = None

    # -- the wrapped call ---------------------------------------------------
    def decide(self, *, state: Any, questions: Any) -> tuple[dict, float | None]:
        step = self._step()
        metadata = dict(step.metadata or {})
        if step.request_id is not None:
            metadata.setdefault("request_id", step.request_id)
        response, trace = self.monitor.run(
            self.client,
            workflow_id=self.workflow_id,
            node_id=step.node_id or _name_from(questions),
            state=state,
            questions=questions,
            policy=step.policy or self.policy,
            question_version=step.question_version,
            policy_version=step.policy_version,
            workflow_version=step.workflow_version,
            metadata=metadata,
        )
        if trace.action is None and trace.answers:
            # No policy handed in: record the branch the model itself chose, so the
            # audit has something to compare. A policy's own action always wins.
            trace.action = answer_branch(trace.answers[0])
        self.last_trace = trace
        return response, trace.latency_ms

    # -- optional labelling -------------------------------------------------
    @contextmanager
    def step(
        self,
        node_id: str | None = None,
        *,
        request_id: str | None = None,
        policy: ActionPolicy | None = None,
        question_version: str | None = None,
        policy_version: str | None = None,
        workflow_version: str | None = None,
        **metadata: Any,
    ):
        """Label the decisions made inside this block.

        Nesting inherits: a ``request`` block around three ``step`` blocks ties
        those three decisions into one request without repeating the id.
        """
        current = self._step()
        token = _CURRENT.set(
            replace(
                current,
                node_id=node_id if node_id is not None else current.node_id,
                request_id=request_id if request_id is not None else current.request_id,
                policy=policy or current.policy,
                question_version=question_version or current.question_version,
                policy_version=policy_version or current.policy_version,
                workflow_version=workflow_version or current.workflow_version,
                metadata={**(current.metadata or {}), **metadata},
            )
        )
        try:
            yield self
        finally:
            _CURRENT.reset(token)

    def request(self, request_id: str, **metadata: Any):
        """Tie the decisions made inside this block into one request."""
        return self.step(request_id=request_id, **metadata)

    def record_outcome(
        self, outcome: Any, *, correct: bool | None = None, trace: DecisionTrace | None = None
    ) -> DecisionTrace | None:
        """Attach what actually happened to a decision — the last one by default."""
        target = trace or self.last_trace
        if target is None:
            return None
        return self.monitor.record_outcome(target, outcome=outcome, correct=correct)

    def _step(self) -> _Step:
        current = _CURRENT.get() or _UNLABELLED
        return replace(
            current,
            node_id=current.node_id or self.defaults.node_id,
            policy=current.policy or self.defaults.policy,
            question_version=current.question_version or self.defaults.question_version,
            policy_version=current.policy_version or self.defaults.policy_version,
            workflow_version=current.workflow_version or self.defaults.workflow_version,
            metadata={**(self.defaults.metadata or {}), **(current.metadata or {})},
        )

    def __getattr__(self, name: str) -> Any:
        # Anything else the caller used on their client still reaches it.
        return getattr(self.client, name)


def _name_from(questions: Any) -> str:
    """A node name from the questions, when the caller did not give one.

    The question is the decision point in every workflow that has not been
    restructured yet, so it is a better default than "unnamed".
    """
    try:
        names = sorted(questions)
    except TypeError:
        return "decision"
    return "+".join(names) if names else "decision"


def watch(
    client: Any,
    *,
    workflow_id: str,
    path: str | Path | None = "traces.jsonl",
    store: TraceStore | None = None,
    policy: ActionPolicy | None = None,
    node_id: str | None = None,
    question_version: str | None = None,
    policy_version: str | None = None,
    workflow_version: str | None = None,
    **metadata: Any,
) -> WatchedClient:
    """Wrap a Jev client so every decision it makes is recorded.

    ``path`` is where the traces go; pass ``store`` instead for somewhere else, or
    ``store=InMemoryTraceStore()`` in a test. Everything after it is a default for
    the labels, overridable per call through :meth:`WatchedClient.step`.
    """
    if store is None:
        store = JsonlTraceStore(path or "traces.jsonl")
    return WatchedClient(
        client,
        workflow_id=workflow_id,
        store=store,
        policy=policy,
        defaults=_Step(
            node_id=node_id,
            question_version=question_version,
            policy_version=policy_version,
            workflow_version=workflow_version,
            metadata=metadata or None,
        ),
    )
