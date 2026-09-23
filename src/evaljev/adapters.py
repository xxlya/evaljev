"""Read another harness's logs as :class:`DecisionTrace` objects.

A harness that already calls Jev has already recorded everything this library
needs — the state, the questions, the full distributions, the latency, the cost.
It just wrote them in its own shape. An adapter is a translation, never a second
instrumentation pass: nothing here re-runs a decision or invents a field.

The one adapter here reads the `jev-plays-starcraft-2
<https://github.com/rapidstartup/jev-plays-starcraft-2>`_ harness, whose
``runs/<stamp>/`` directories hold an ``events.jsonl`` and a ``result.json``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import DecisionTrace, QuestionSpec
from .monitor import _parse_answers

__all__ = ["load_sc2_run", "load_sc2_runs", "schema_fingerprint"]

WORKFLOW = "jev-plays-sc2"


def schema_fingerprint(question: Mapping[str, Any]) -> str:
    """A stable short name for a question's exact wording and option set.

    Harnesses that build their questions in code rarely version them, which leaves
    the one field attribution most needs — "did the wording change?" — empty. The
    fingerprint fills it without asking anyone to remember: reword the instructions
    or touch the options and the version changes, because it is those bytes.
    """
    payload = json.dumps(
        {
            "instructions": question.get("instructions"),
            "criteria": question.get("criteria"),
            "type": question.get("type"),
        },
        sort_keys=True,
        default=str,
    )
    return "q-" + hashlib.sha1(payload.encode()).hexdigest()[:8]


def _rows(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # a run killed mid-write ends in half a line


def load_sc2_run(directory: str | Path, *, workflow_id: str = WORKFLOW) -> list[DecisionTrace]:
    """One game directory as traces — one per question asked, not one per API call.

    A single call carries several questions (the strategic priority, an order for
    each unit type, the economy). They are separate decisions with separate
    branches and separate failure modes, and the API answers them independently,
    so they are separate traces over a shared state.
    """
    directory = Path(directory)
    events = directory / "events.jsonl"
    if not events.exists():
        return []

    outcome, calls = None, None
    result_file = directory / "result.json"
    if result_file.exists():
        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
            outcome, calls = result.get("status"), result.get("calls")
        except json.JSONDecodeError:
            pass

    traces: list[DecisionTrace] = []
    revision, loop = None, None
    for row in _rows(events):
        event = row.get("event")
        if event == "tick":
            # player.py is reloaded from git at decision boundaries, so the policy
            # can change *inside* a run. That is exactly the thing worth recording.
            revision = row.get("revision", revision)
            loop = row.get("loop", loop)
            continue
        if event != "jev":
            continue

        state = row.get("state")
        questions = row.get("questions") or {}
        response = row.get("response") or {}
        if not questions or not isinstance(response, Mapping):
            continue
        usage = response.get("usage") or {}
        stamp = datetime.fromtimestamp(row.get("time", 0), tz=timezone.utc)
        answers = {a.question_name: a for a in _parse_answers(response, questions)}

        for name, question in questions.items():
            answer = answers.get(name)
            if answer is None:
                continue
            traces.append(
                DecisionTrace(
                    workflow_id=workflow_id,
                    node_id=name,
                    timestamp=stamp,
                    model=response.get("model") or row.get("via"),
                    question_version=schema_fingerprint(question),
                    policy_version=revision,
                    state=state,
                    questions=[
                        QuestionSpec(
                            name=name,
                            type=question.get("type", "choice"),
                            instructions=question.get("instructions"),
                            criteria=question.get("criteria"),
                        )
                    ],
                    answers=[answer],
                    latency_ms=row.get("latency_ms"),
                    action=answer.selected,
                    metadata={
                        "run_id": directory.name,
                        # One game loop is one request: every question asked at that
                        # boundary is part of the same decision the harness took.
                        "request_id": f"{directory.name}:{loop}",
                        "run_outcome": outcome,
                        "loop": loop,
                        "via": row.get("via"),
                        "cost_usd": usage.get("cost") or usage.get("cost_usd"),
                        "run_calls": calls,
                    },
                )
            )
    return traces


def load_sc2_runs(root: str | Path, *, workflow_id: str = WORKFLOW) -> list[DecisionTrace]:
    """Every game under a ``runs/`` directory, oldest first."""
    traces: list[DecisionTrace] = []
    for directory in sorted(Path(root).iterdir()):
        if directory.is_dir():
            traces.extend(load_sc2_run(directory, workflow_id=workflow_id))
    return sorted(traces, key=lambda t: t.timestamp)
