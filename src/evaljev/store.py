from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from .models import DecisionTrace


def latest_by_trace_id(traces: Iterable[DecisionTrace]) -> list[DecisionTrace]:
    """Collapse an append-only event log to the newest version of each trace.

    A trace is written once when the decision is made and again when its outcome
    arrives, so the same ``trace_id`` can appear several times. Later versions win;
    each trace keeps the position of its first appearance.
    """
    latest: dict[str, DecisionTrace] = {}
    for trace in traces:
        latest[trace.trace_id] = trace
    return list(latest.values())


class TraceStore(Protocol):
    def append(self, trace: DecisionTrace) -> None:
        """Record a version of a trace. Called again when an outcome is attached."""
        ...

    def list(self, workflow_id: str | None = None) -> list[DecisionTrace]:
        """Return the newest version of each trace, most-recently-written wins."""
        ...


class InMemoryTraceStore:
    def __init__(self) -> None:
        self._traces: list[DecisionTrace] = []

    def append(self, trace: DecisionTrace) -> None:
        self._traces.append(trace)

    def list(self, workflow_id: str | None = None) -> list[DecisionTrace]:
        if workflow_id is None:
            return latest_by_trace_id(self._traces)
        return latest_by_trace_id(t for t in self._traces if t.workflow_id == workflow_id)


class JsonlTraceStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, trace: DecisionTrace) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(trace.model_dump_json() + "\n")

    def list(self, workflow_id: str | None = None) -> list[DecisionTrace]:
        if not self.path.exists():
            return []
        out: list[DecisionTrace] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                trace = DecisionTrace.model_validate(json.loads(line))
                if workflow_id is None or trace.workflow_id == workflow_id:
                    out.append(trace)
        return latest_by_trace_id(out)

    def extend(self, traces: Iterable[DecisionTrace]) -> None:
        for trace in traces:
            self.append(trace)
