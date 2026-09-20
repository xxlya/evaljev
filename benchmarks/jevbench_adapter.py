"""Adapt JevBench public tasks to EvalJev traces.

JevBench (https://github.com/fstandhartinger/jevbench) ships labelled typed
decisions. EvalJev does not re-implement its scoring: this module only maps a
task's schema onto a Jev request and maps a Jev answer back onto the task's
exact label set, following the protocol in JevBench's IMPLEMENTATION.md:

    noul   - the model returns P(true) as a scalar; store {"yes": p, "no": 1-p}
    choice - the model's distribution over option keys, used as it comes
    score  - the model's distribution over level indices; argmax is the prediction

The labelled data is a fixture for EvalJev's detectors, not a leaderboard: it is
how we check that a calibration or instability signal fires when it should.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# JevBench's noul label convention, in the order it declares them.
NOUL_TRUE, NOUL_FALSE = "yes", "no"


@dataclass(frozen=True)
class BenchTask:
    id: str
    family: str
    group: str
    state: Any
    question: dict
    labels: list[str]
    expected: str | None

    @property
    def qtype(self) -> str:
        return self.question["type"]

    def as_questions(self) -> dict[str, dict]:
        """The request schema. Naming the question after its family gives
        per-family calibration straight out of ``calibration_report``."""
        return {self.family: self.question}


def load_split(path: str | Path, *, families: list[str] | None = None) -> list[BenchTask]:
    tasks = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if families and row["family"] not in families:
            continue
        tasks.append(
            BenchTask(
                id=row["id"],
                family=row["family"],
                group=row.get("group") or row["id"],
                state=row["state"],
                question=row["question"],
                labels=[str(x) for x in row["labels"]],
                expected=None if row.get("expected") is None else str(row["expected"]),
            )
        )
    return tasks


def paraphrase_groups(tasks: list[BenchTask]) -> dict[str, list[BenchTask]]:
    """Groups holding more than one phrasing of the same decision."""
    groups: dict[str, list[BenchTask]] = {}
    for task in tasks:
        groups.setdefault(task.group, []).append(task)
    return {g: rows for g, rows in groups.items() if len(rows) > 1}


def to_label_space(answer, task: BenchTask) -> tuple[str | None, dict[str, float] | None]:
    """Map an EvalJev ``DecisionAnswer`` onto the task's exact label set.

    Returns ``(predicted, probabilities)``. ``predicted`` is None when the model
    produced nothing usable; a distribution whose keys do not match the label set
    is returned unchanged so ``schema_adherence`` can see the violation.
    """
    if task.qtype == "noul":
        if answer.value is None:
            return None, None
        p_true = float(answer.value)
        probs = {NOUL_TRUE: p_true, NOUL_FALSE: 1.0 - p_true}
        return (NOUL_TRUE if p_true >= 0.5 else NOUL_FALSE), probs

    probs = answer.probabilities
    if task.qtype == "score":
        # Accuracy uses argmax over level indices, not the expected value.
        if not probs:
            return (None if answer.value is None else str(answer.value)), probs
        return _argmax(probs), probs

    if probs:
        return _argmax(probs), probs
    return answer.selected, probs


def _argmax(probs: dict[str, float]) -> str:
    """Deterministic argmax, ties broken by the lexicographically smallest label."""
    return max(sorted(probs), key=lambda k: probs[k])


def expected_level(probs: dict[str, float] | None) -> float | None:
    """Probability-weighted level for ordinal questions (JevBench reports MAE on this)."""
    if not probs:
        return None
    try:
        return sum(float(k) * float(v) for k, v in probs.items())
    except ValueError:
        return None
