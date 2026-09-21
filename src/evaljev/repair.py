"""Propose schema repairs, and refuse to believe them without evidence.

The cycle is ``propose -> validate -> screen -> verify -> approve``. Only the
first step benefits from a language model: generating candidate wordings is an
open-ended search with no closed form. Everything after it is deterministic,
because the output is a change to production configuration.

Three properties are structural, not configurable:

**Nothing is ever applied.** The cycle returns ranked proposals with their
evidence. A human applies them.

**The proposer never sees the holdout.** Candidates are generated from failures in
the dev split and judged on traces the generator could not have fitted to.
Optimising a schema against the traces you measure it on is the exact mechanism
that makes auto-repair dangerous.

**A candidate may not change the label set.** Adding or removing an option changes
the option count, which silently moves every threshold keyed on ``confidence`` —
a repair that quietly re-tunes every downstream cutoff is not a repair. Candidates
that do it are rejected before they cost anything.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .lint import lint_questions
from .models import DecisionTrace
from .replay import replay, summarize_replay

Generate = Callable[[str], str]
Judge = Callable[[DecisionTrace, str | None], bool | None]


@dataclass
class Candidate:
    """A proposed replacement schema for one question."""

    question_name: str
    question: dict
    rationale: str = ""
    source: str = "llm"
    problems: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.problems

    def as_questions(self) -> dict[str, dict]:
        return {self.question_name: self.question}

    def as_dict(self) -> dict:
        return {
            "question_name": self.question_name,
            "question": self.question,
            "rationale": self.rationale,
            "source": self.source,
            "problems": self.problems,
        }


def split_traces(
    traces: Iterable[DecisionTrace], *, holdout_fraction: float = 0.5, salt: str = "evaljev"
) -> tuple[list[DecisionTrace], list[DecisionTrace]]:
    """Split traces into (dev, holdout), deterministically by trace id.

    Hashing rather than shuffling means the same trace lands in the same split on
    every run, so a proposal cannot be laundered by re-rolling the split until the
    holdout happens to agree.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    dev, holdout = [], []
    cutoff = int(holdout_fraction * 2**32)
    for trace in traces:
        digest = hashlib.sha256(f"{salt}:{trace.trace_id}".encode()).digest()
        bucket = int.from_bytes(digest[:4], "big")
        (holdout if bucket < cutoff else dev).append(trace)
    return dev, holdout


def baseline_questions(traces: Sequence[DecisionTrace], question_name: str) -> dict:
    """Reconstruct the schema these traces were produced with."""
    for trace in traces:
        for spec in trace.questions:
            if spec.name == question_name:
                return {
                    "type": spec.type,
                    "instructions": spec.instructions,
                    **({"criteria": spec.criteria} if spec.criteria is not None else {}),
                }
    raise ValueError(f"no question named {question_name!r} in these traces")


def validate_candidate(candidate: Candidate, baseline: Mapping[str, Any]) -> list[str]:
    """Reject a candidate before it costs anything. Returns the problems found."""
    problems = []
    question = candidate.question
    if not isinstance(question, Mapping):
        return ["candidate is not an object"]
    if question.get("type") != baseline.get("type"):
        problems.append(f"question type changed: {baseline.get('type')} -> {question.get('type')}")
    if not str(question.get("instructions") or "").strip():
        problems.append("instructions are empty")

    base_criteria, new_criteria = baseline.get("criteria"), question.get("criteria")
    if isinstance(base_criteria, Mapping):
        if not isinstance(new_criteria, Mapping):
            problems.append("criteria map is missing")
        else:
            added = sorted(set(new_criteria) - set(base_criteria))
            removed = sorted(set(base_criteria) - set(new_criteria))
            if added or removed:
                # The constraint that matters: k must not move.
                problems.append(
                    f"label set changed (added={added}, removed={removed}); this moves every "
                    "confidence threshold downstream"
                )
            empty = sorted(k for k, v in new_criteria.items() if not str(v or "").strip())
            if empty:
                problems.append(f"criteria have no description for {empty}")

    if question == dict(baseline):
        problems.append("identical to the current schema")

    for issue in lint_questions({candidate.question_name: question}):
        if issue.severity == "warning":
            problems.append(f"lint {issue.code}: {issue.message}")
    return problems


def evaluate_candidate(
    candidate: Candidate,
    traces: Sequence[DecisionTrace],
    client: Any,
    *,
    policy: Callable | None = None,
    judge: Judge | None = None,
    label_of: Callable | None = None,
    alpha: float = 0.05,
) -> dict:
    """Replay traces through a candidate and summarize with a paired test."""
    results = replay(
        traces,
        client,
        questions=candidate.as_questions(),
        policy=policy,
        judge=judge,
        label_of=label_of,
    )
    summary = summarize_replay(results, alpha=alpha)
    summary["question_name"] = candidate.question_name
    return summary


def passes_gate(summary: Mapping[str, Any], gate: str) -> bool:
    """Did this candidate clear the configured bar?

    ``decision`` is the default and the strict one: the decisions themselves must
    have improved significantly. ``probability`` accepts a significant movement of
    probability mass onto the correct label, which is much more sensitive on small
    samples but is a *leading indicator* — mass can move a long way without any
    decision crossing a threshold. Use it when decision-level power is genuinely
    unavailable, and confirm on more traffic before applying.
    """
    if gate == "decision":
        return summary["verdict"] == "improvement"
    if gate == "probability":
        shift = summary.get("probability_shift")
        return bool(shift and shift["verdict"] == "improvement")
    if gate == "both":
        shift = summary.get("probability_shift")
        return summary["verdict"] == "improvement" and bool(
            shift and shift["verdict"] == "improvement"
        )
    raise ValueError(f"unknown gate: {gate!r}")


PROPOSE_PROMPT = """You are improving one typed decision schema for a production system.

The schema below is used to make a choice. Some decisions came out wrong. Rewrite \
the schema so those cases are handled correctly, without breaking the others.

HARD CONSTRAINTS — a proposal violating any of these is discarded:
- Keep exactly the same option keys. Do not add, remove or rename any.
- Keep the same "type".
- Every option keeps a concrete, discriminating description.
- Do not mention specific examples from the failures verbatim; fix the rule, not the case.

CURRENT SCHEMA:
{schema}

DECISIONS THAT CAME OUT WRONG ({n_failures} of {n_total}):
{failures}

Return {n} alternative schemas as a JSON array. Each element must be an object:
{{"instructions": "...", "criteria": {{...}}, "rationale": "one sentence on what you changed and why"}}

Return only the JSON array."""


def _format_failures(traces: Sequence[DecisionTrace], question_name: str, limit: int = 12) -> str:
    lines = []
    for trace in traces[:limit]:
        expected = trace.metadata.get("expected") or trace.metadata.get("expected_route")
        lines.append(
            f"- state: {json.dumps(trace.state, default=str)[:300]}\n"
            f"  chose: {trace.action}   should have been: {expected}"
        )
    return "\n".join(lines) if lines else "(none)"


def _extract_json_array(text: str) -> list:
    """Pull a JSON array out of a model response, tolerating code fences."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON array found in the response")
    parsed = json.loads(text[start : end + 1])
    if isinstance(parsed, list):
        return parsed
    # A model returning the wrong shape is bad data, not a programming error.
    raise ValueError("response was not a JSON array")


def propose_candidates(
    baseline: Mapping[str, Any],
    failures: Sequence[DecisionTrace],
    *,
    question_name: str,
    generate: Generate,
    n: int = 3,
    total: int | None = None,
) -> list[Candidate]:
    """Ask a language model for alternative schemas. The only generative step."""
    prompt = PROPOSE_PROMPT.format(
        schema=json.dumps(dict(baseline), indent=2),
        failures=_format_failures(failures, question_name),
        n_failures=len(failures),
        n_total=total if total is not None else len(failures),
        n=n,
    )
    raw = generate(prompt)
    candidates = []
    for item in _extract_json_array(raw):
        if not isinstance(item, Mapping):
            continue
        question = {
            "type": baseline.get("type"),
            "instructions": item.get("instructions"),
        }
        if "criteria" in item:
            question["criteria"] = item["criteria"]
        elif baseline.get("criteria") is not None:
            question["criteria"] = baseline["criteria"]
        candidates.append(
            Candidate(
                question_name=question_name,
                question=question,
                rationale=str(item.get("rationale") or "")[:300],
            )
        )
    return candidates


def repair_cycle(
    traces: Sequence[DecisionTrace],
    client: Any,
    *,
    question_name: str,
    generate: Generate,
    policy: Callable | None = None,
    judge: Judge | None = None,
    label_of: Callable | None = None,
    n_candidates: int = 3,
    holdout_fraction: float = 0.5,
    alpha: float = 0.05,
    gate: str = "decision",
) -> dict:
    """Run the full cycle and return ranked proposals. Applies nothing.

    Candidates are screened on the dev split first, so only the ones that survive a
    cheap look cost a full holdout replay. ``approved`` contains those that reached
    a significant improvement on traces the generator never saw — which is the only
    claim worth acting on.
    """
    dev, holdout = split_traces(traces, holdout_fraction=holdout_fraction)
    if not dev or not holdout:
        raise ValueError("not enough traces to form both a dev and a holdout split")

    baseline = baseline_questions(traces, question_name)
    failures = [t for t in dev if t.outcome_correct is False]

    candidates = propose_candidates(
        baseline,
        failures,
        question_name=question_name,
        generate=generate,
        n=n_candidates,
        total=len(dev),
    )
    for candidate in candidates:
        candidate.problems = validate_candidate(candidate, baseline)

    evaluated = []
    for candidate in candidates:
        row: dict[str, Any] = {"candidate": candidate.as_dict()}
        if not candidate.valid:
            row["rejected_at"] = "validate"
            evaluated.append(row)
            continue
        screen = evaluate_candidate(
            candidate, dev, client, policy=policy, judge=judge, label_of=label_of, alpha=alpha
        )
        row["screen"] = screen
        if screen["regressions"] > screen["improvements"]:
            row["rejected_at"] = "screen"
            evaluated.append(row)
            continue
        row["verify"] = evaluate_candidate(
            candidate, holdout, client, policy=policy, judge=judge, label_of=label_of, alpha=alpha
        )
        row["rejected_at"] = None if passes_gate(row["verify"], gate) else "verify"
        evaluated.append(row)

    approved = [r for r in evaluated if r.get("rejected_at") is None]
    approved.sort(key=lambda r: (r["verify"]["p_value"], -r["verify"]["improvements"]))
    return {
        "question_name": question_name,
        "gate": gate,
        "baseline": baseline,
        "dev_traces": len(dev),
        "holdout_traces": len(holdout),
        "dev_failures": len(failures),
        "candidates": evaluated,
        "approved": approved,
        "next_step": (
            f"{len(approved)} candidate(s) cleared the {gate} gate on the holdout at "
            f"alpha={alpha}. "
            "Review the diff and apply manually."
            if approved
            else "No candidate showed a significant improvement on held-out traces. "
            "Nothing to apply."
        ),
    }
