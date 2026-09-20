from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from .models import DecisionTrace


class TraceStore(Protocol):
    def append(self, trace: DecisionTrace) -> None: ...
    def list(self, workflow_id: str | None = None) -> list[DecisionTrace]: ...


class InMemoryTraceStore:
    def __init__(self) -> None:
        self._traces: list[DecisionTrace] = []

    def append(self, trace: DecisionTrace) -> None:
        self._traces.append(trace)

    def list(self, workflow_id: str | None = None) -> list[DecisionTrace]:
        if workflow_id is None:
            return list(self._traces)
        return [t for t in self._traces if t.workflow_id == workflow_id]


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
        return out

    def extend(self, traces: Iterable[DecisionTrace]) -> None:
        for trace in traces:
            self.append(trace)
