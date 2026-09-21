"""Which component changed between two windows of traffic?

A deployed typed decision has five moving parts: state construction, the question
schema, the model, the policy/threshold, and the downstream action. When behaviour
moves, exactly one question matters — which of them moved?

Four of the five are already recorded on every trace as explicit fields, so this is
a join and a diff, not an inference problem. That is deliberate: an attribution
report is evidence for changing production configuration, so it has to be
deterministic, reproducible and readable. "The criteria map gained an option, here
is the diff" is auditable in a way that a model's opinion is not.

Only the fifth possibility — every version identical, inputs comparable, behaviour
still moved — cannot be settled from stored traces alone. That one is the residual,
and :func:`vendor_drift_check` resolves it by replaying old inputs against today's
model.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .metrics import confidence_to_pmax, distribution_shift
from .models import DecisionTrace
from .perturb import answer_branch, answer_distribution

VERSION_FIELDS = ("model_version", "question_version", "policy_version", "workflow_version")


@dataclass
class Finding:
    """One candidate explanation, with the evidence that supports it."""

    component: str
    changed: bool
    summary: str
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "component": self.component,
            "changed": self.changed,
            "summary": self.summary,
            "evidence": self.evidence,
        }


def _values(traces: Sequence[DecisionTrace], attr: str) -> set:
    return {getattr(t, attr) for t in traces}


def version_diff(reference: Sequence[DecisionTrace], current: Sequence[DecisionTrace]) -> list[Finding]:
    """Compare the four version axes recorded on every trace. Free, exact."""
    findings = []
    for attr in VERSION_FIELDS:
        ref, cur = _values(reference, attr), _values(current, attr)
        if ref == cur:
            continue
        component = attr.replace("_version", "")
        findings.append(
            Finding(
                component=component,
                changed=True,
                summary=f"{attr} changed: {sorted(map(str, ref))} -> {sorted(map(str, cur))}",
                evidence={"field": attr, "reference": sorted(map(str, ref)), "current": sorted(map(str, cur))},
            )
        )
    return findings


def _schemas(traces: Sequence[DecisionTrace]) -> dict[str, dict]:
    """Latest schema seen per question name, as plain comparable dicts."""
    out: dict[str, dict] = {}
    for trace in traces:
        for spec in trace.questions:
            out[spec.name] = {
                "type": spec.type,
                "instructions": spec.instructions,
                "criteria": spec.criteria,
            }
    return out


def schema_diff(
    reference: Sequence[DecisionTrace],
    current: Sequence[DecisionTrace],
    *,
    confidence_threshold: float | None = None,
) -> list[Finding]:
    """Diff the question schemas the traces were actually produced with.

    A criteria map changing size gets its own finding, because option count is the
    one schema edit that silently moves every threshold keyed on ``confidence``
    without touching the model, the policy code or the accuracy.
    """
    findings = []
    ref_schemas, cur_schemas = _schemas(reference), _schemas(current)

    for name in sorted(set(ref_schemas) | set(cur_schemas)):
        ref, cur = ref_schemas.get(name), cur_schemas.get(name)
        if ref is None or cur is None:
            findings.append(
                Finding(
                    "schema",
                    True,
                    f"question {name!r} {'added' if ref is None else 'removed'}",
                    {"question": name},
                )
            )
            continue
        if ref == cur:
            continue

        if ref["instructions"] != cur["instructions"]:
            findings.append(
                Finding(
                    "schema",
                    True,
                    f"question {name!r}: instructions rewritten",
                    {
                        "question": name,
                        "reference": str(ref["instructions"])[:200],
                        "current": str(cur["instructions"])[:200],
                    },
                )
            )

        ref_c, cur_c = ref["criteria"], cur["criteria"]
        if isinstance(ref_c, Mapping) and isinstance(cur_c, Mapping):
            added, removed = sorted(set(cur_c) - set(ref_c)), sorted(set(ref_c) - set(cur_c))
            if added or removed:
                evidence: dict[str, Any] = {
                    "question": name,
                    "added": added,
                    "removed": removed,
                    "k_reference": len(ref_c),
                    "k_current": len(cur_c),
                }
                summary = f"question {name!r}: option set changed, k {len(ref_c)} -> {len(cur_c)}"
                if len(ref_c) >= 2 and len(cur_c) >= 2:
                    c = 0.5 if confidence_threshold is None else confidence_threshold
                    before = confidence_to_pmax(c, len(ref_c))
                    after = confidence_to_pmax(c, len(cur_c))
                    evidence["confidence_threshold"] = c
                    evidence["implied_pmax_before"] = before
                    evidence["implied_pmax_after"] = after
                    evidence["implied_pmax_shift"] = after - before
                    if abs(after - before) > 1e-9:
                        summary += (
                            f"; any confidence<{c} threshold now fires at p_max "
                            f"{before:.3f} -> {after:.3f}"
                        )
                findings.append(Finding("option_set", True, summary, evidence))
            reworded = sorted(k for k in set(ref_c) & set(cur_c) if ref_c[k] != cur_c[k])
            if reworded:
                findings.append(
                    Finding(
                        "schema",
                        True,
                        f"question {name!r}: criteria reworded for {reworded}",
                        {"question": name, "labels": reworded},
                    )
                )
        elif ref_c != cur_c:
            findings.append(
                Finding("schema", True, f"question {name!r}: criteria changed", {"question": name})
            )
    return findings


def _state_shape(state: Any) -> tuple[frozenset, int]:
    if isinstance(state, Mapping):
        return frozenset(map(str, state)), len(json.dumps(state, default=str))
    return frozenset(), len(json.dumps(state, default=str))


def state_diff(
    reference: Sequence[DecisionTrace],
    current: Sequence[DecisionTrace],
    *,
    size_ratio_threshold: float = 1.5,
) -> list[Finding]:
    """Structural drift in the inputs: which keys are present, and how big they are.

    This is deliberately shallow. It catches a state builder that started or stopped
    emitting a field, and gross changes in input size. It does **not** detect semantic
    drift in the traffic — same keys, same size, different subject matter — which needs
    an embedding of the state and is out of scope here. A clean report from this
    function is not evidence that the inputs are unchanged.
    """
    findings = []
    ref_shapes = [_state_shape(t.state) for t in reference]
    cur_shapes = [_state_shape(t.state) for t in current]
    if not ref_shapes or not cur_shapes:
        return findings

    ref_keys = frozenset().union(*(s[0] for s in ref_shapes))
    cur_keys = frozenset().union(*(s[0] for s in cur_shapes))
    if ref_keys != cur_keys:
        findings.append(
            Finding(
                "state",
                True,
                f"state keys changed: added {sorted(cur_keys - ref_keys)}, "
                f"removed {sorted(ref_keys - cur_keys)}",
                {"added": sorted(cur_keys - ref_keys), "removed": sorted(ref_keys - cur_keys)},
            )
        )

    ref_size = statistics.median(s[1] for s in ref_shapes)
    cur_size = statistics.median(s[1] for s in cur_shapes)
    ratio = cur_size / ref_size if ref_size else float("inf")
    if ratio >= size_ratio_threshold or (ratio and ratio <= 1 / size_ratio_threshold):
        findings.append(
            Finding(
                "state",
                True,
                f"median state size moved {ref_size:.0f} -> {cur_size:.0f} chars ({ratio:.2f}x)",
                {"reference_median": ref_size, "current_median": cur_size, "ratio": ratio},
            )
        )
    return findings


def action_mix(traces: Iterable[DecisionTrace]) -> dict[str, float]:
    """How often each branch was taken. Needs no labels."""
    counts: dict[str, int] = {}
    total = 0
    for trace in traces:
        key = trace.action if trace.action is not None else "<none>"
        counts[key] = counts.get(key, 0) + 1
        total += 1
    return {k: v / total for k, v in counts.items()} if total else {}


def behaviour_summary(traces: Sequence[DecisionTrace]) -> dict:
    """The label-free description of what a window of traffic did."""
    probs = []
    for trace in traces:
        for ans in trace.answers:
            dist = answer_distribution(ans)
            if dist:
                probs.append(max(dist.values()))
    labelled = [t for t in traces if t.outcome_correct is not None]
    return {
        "n": len(traces),
        "action_mix": action_mix(traces),
        "mean_pmax": statistics.mean(probs) if probs else None,
        "median_latency_ms": (
            statistics.median([t.latency_ms for t in traces if t.latency_ms is not None])
            if any(t.latency_ms is not None for t in traces)
            else None
        ),
        "accuracy": (
            sum(bool(t.outcome_correct) for t in labelled) / len(labelled) if labelled else None
        ),
        "labelled": len(labelled),
    }


def attribute(
    reference: Iterable[DecisionTrace],
    current: Iterable[DecisionTrace],
    *,
    confidence_threshold: float | None = None,
    action_shift_threshold: float = 0.1,
) -> dict:
    """Explain a behaviour change between two windows of traffic.

    Returns the findings that have evidence behind them, plus ``residual``: true
    when behaviour moved and nothing recorded on the traces accounts for it. A
    residual is not a conclusion, it is the instruction to run
    :func:`vendor_drift_check`, which is the only way to separate "the model changed
    underneath us" from "our traffic changed".
    """
    ref, cur = list(reference), list(current)
    if not ref or not cur:
        raise ValueError("both windows need at least one trace")

    findings = version_diff(ref, cur)
    findings += schema_diff(ref, cur, confidence_threshold=confidence_threshold)
    findings += state_diff(ref, cur)

    ref_behaviour, cur_behaviour = behaviour_summary(ref), behaviour_summary(cur)
    mix_shift = distribution_shift(ref_behaviour["action_mix"], cur_behaviour["action_mix"])
    behaviour_moved = mix_shift >= action_shift_threshold
    if behaviour_moved:
        findings.append(
            Finding(
                "behaviour",
                True,
                f"action mix shifted (JS distance {mix_shift:.3f})",
                {
                    "distribution_shift": mix_shift,
                    "reference": ref_behaviour["action_mix"],
                    "current": cur_behaviour["action_mix"],
                },
            )
        )

    causes = [f for f in findings if f.component != "behaviour"]
    residual = behaviour_moved and not causes
    if residual:
        next_step = (
            "No recorded component changed. Run vendor_drift_check() to separate a "
            "model change from an input change."
        )
    elif behaviour_moved:
        next_step = f"Behaviour moved and {len(causes)} component(s) changed; check those diffs first."
    else:
        next_step = "Behaviour is within the shift threshold; nothing to attribute."

    return {
        "reference": ref_behaviour,
        "current": cur_behaviour,
        "action_mix_shift": mix_shift,
        "behaviour_moved": behaviour_moved,
        "findings": [f.as_dict() for f in findings],
        "changed_components": sorted({f.component for f in causes}),
        "residual": residual,
        "next_step": next_step,
    }


def vendor_drift_check(
    traces: Sequence[DecisionTrace],
    client: Any,
    *,
    question_name: str | None = None,
    shift_threshold: float = 0.1,
) -> dict:
    """Replay recorded inputs against today's model and compare to what was stored.

    This resolves the residual. Both arms use the *recorded* state and the *recorded*
    schema, so the only thing that differs is when the call was made:

    - distributions moved  -> the model changed underneath you (vendor drift)
    - distributions held   -> the model is stable; your traffic moved instead

    Costs one call per trace, so pass a sample.
    """
    shifts, flips, compared = [], 0, 0
    for trace in traces:
        questions = {
            spec.name: {
                "type": spec.type,
                "instructions": spec.instructions,
                **({"criteria": spec.criteria} if spec.criteria is not None else {}),
            }
            for spec in trace.questions
        }
        if not questions:
            continue
        name = question_name or trace.questions[0].name
        old = next((a for a in trace.answers if a.question_name == name), None)
        if old is None:
            continue

        response, _ = client.decide(state=trace.state, questions=questions)
        from .monitor import _parse_answers  # local import keeps the module graph acyclic

        fresh = next(
            (a for a in _parse_answers(response, questions) if a.question_name == name), None
        )
        if fresh is None:
            continue

        compared += 1
        old_dist, new_dist = answer_distribution(old), answer_distribution(fresh)
        if old_dist and new_dist:
            shifts.append(distribution_shift(old_dist, new_dist))
        if answer_branch(old) != answer_branch(fresh):
            flips += 1

    mean_shift = statistics.mean(shifts) if shifts else None
    drifted = mean_shift is not None and mean_shift >= shift_threshold
    return {
        "compared": compared,
        "mean_distribution_shift": mean_shift,
        "branch_flips": flips,
        "branch_flip_rate": flips / compared if compared else 0.0,
        "vendor_drift": drifted,
        "verdict": (
            "model changed underneath the deployment"
            if drifted
            else "model reproduces its recorded answers; look at the input distribution"
        ),
    }
