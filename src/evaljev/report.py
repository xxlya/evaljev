"""Turn a stream of traces into a monitoring report a non-specialist can read.

Everything in :func:`build_report` is computed by functions the package already
exposes — adherence, drift, attribution, calibration, intervals. This module adds
no statistics of its own. What it adds is **translation**: each measurement is
paired with the plain question it answers, the verdict it supports, and the next
thing to do about it, so the output is readable by someone who has never heard of
expected calibration error.

Two rules from the rest of the library survive the translation, because dropping
them would make the page lie:

- Every rate is reported with its interval, so "1 flip in 36" never renders as a
  clean 2.8%.
- A check that cannot conclude says ``unknown``, never ``ok``. Too little evidence
  means nothing was shown — not that everything is fine.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

from .attribution import action_mix
from .drift import count_windows, drift_report
from .lint import lint_questions
from .metrics import (
    calibration_bins,
    calibration_report,
    schema_adherence,
)
from .models import DecisionTrace, QuestionSpec, answer_branch, answer_distribution
from .stats import rate_with_ci

__all__ = [
    "GLOSSARY",
    "build_report",
    "declared_label_sets",
    "render_html",
    "write_report",
]

# A check is one row on the page: a question in plain words, a verdict, and a
# next step. Severity orders them and drives the headline.
SEVERITY = {"ok": 0, "info": 0, "unknown": 1, "watch": 2, "problem": 3}

#: Every term the page uses that a reader might not know, in one sentence each.
#: The page renders these as hover definitions, so no jargon appears undefined.
GLOSSARY = {
    "decision": "One call to the decision model: your inputs go in, a probability "
    "over your options comes out, your code picks a branch.",
    "certainty": "How much probability the model put on its top answer. 1.00 means "
    "all of it; 0.34 out of three options means it was almost guessing.",
    "unsure": "A decision where the top answer got less probability than your "
    "review threshold — the ones worth sending to a human.",
    "options": "The labels you defined for a question. The model must answer with "
    "one of them, and the probabilities across them must add up to 1.",
    "valid answer": "An answer that used exactly your labels and whose probabilities "
    "add up to 1. Checking this needs no ground truth, so it works on live traffic.",
    "drift": "Recent traffic behaving differently from earlier traffic at the same "
    "decision point — different answers, or different certainty.",
    "confidence interval": "The range the true rate is probably in. A rate measured "
    "on 30 decisions is a wide range, not a precise number.",
    "calibration": "Whether stated certainty matches reality: of the decisions made "
    "at 80% certainty, about 80% should turn out right.",
    "calibration gap": "Average distance between stated certainty and how often those "
    "decisions were actually right. 0 is perfect; above 0.1 is worth a look.",
    "outcome": "What actually happened after the decision — the human's correction, "
    "the click, the refund that went through. You attach it with record_outcome().",
    "decision point": "One place in your workflow that asks the model something. "
    "EvalJev tracks each separately, because they fail separately.",
    "attribution": "Naming which part changed — the question wording, the options, "
    "the model version, the inputs — by diffing what was recorded on the traces.",
    "statistically real": "Unlikely to be a coincidence of small numbers. EvalJev "
    "uses exact tests, so a change only counts once the evidence supports it.",
}


def declared_label_sets(traces: Sequence[DecisionTrace]) -> dict[str, list[str]]:
    """The exact label set each question declared, keyed by question name.

    Choice questions declare theirs as the keys of ``criteria``; score questions as
    the positions of a criteria list. Noul questions are left out on purpose: the
    API answers them with a bare probability and no distribution, so scoring them
    for schema adherence would report a correct answer as a schema failure.
    """
    labels: dict[str, list[str]] = {}
    for trace in traces:
        for q in trace.questions:
            if q.type == "choice" and isinstance(q.criteria, Mapping):
                labels[q.name] = sorted(str(k) for k in q.criteria)
            elif q.type == "score" and isinstance(q.criteria, (list, tuple)):
                labels[q.name] = [str(i) for i in range(len(q.criteria))]
    return labels


def _certainty(ans) -> float | None:
    """How much probability landed on the answer's top label."""
    dist = answer_distribution(ans)
    return max(dist.values()) if dist else None


def _decision_certainty(trace: DecisionTrace) -> float | None:
    """The least certain answer in a decision — the one that would escalate it."""
    values = [c for c in (_certainty(a) for a in trace.answers) if c is not None]
    return min(values) if values else None


def _cost(trace: DecisionTrace) -> float:
    value = trace.metadata.get("cost_usd") if isinstance(trace.metadata, dict) else None
    return float(value) if isinstance(value, (int, float)) else 0.0


def _rate(successes: int, n: int) -> dict:
    """A rate with its interval, rounded for display but not for comparison."""
    row = rate_with_ci(successes, n)
    return {
        "rate": row["rate"],
        "ci": [row["ci_low"], row["ci_high"]],
        "n": n,
        "successes": successes,
    }


def _plural(n: int, noun: str) -> str:
    return f"across {n} {noun}" + ("" if n == 1 else "s")


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%".replace(".0%", "%")


def _ci_text(row: Mapping[str, Any]) -> str:
    return f"{_pct(row['rate'])} ({_pct(row['ci'][0])}–{_pct(row['ci'][1])} range, n={row['n']})"


def _short(value: Any, limit: int = 180) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _latest_questions(traces: Sequence[DecisionTrace]) -> list[QuestionSpec]:
    """The most recent schema seen for each question name."""
    latest: dict[str, QuestionSpec] = {}
    for trace in traces:
        for q in trace.questions:
            latest[q.name] = q
    return [latest[name] for name in sorted(latest)]


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #


def _check_valid_answers(traces, labels) -> dict:
    stats = schema_adherence(traces, labels)
    total = sum(row["n"] for row in stats.values())
    valid = sum(row["valid"] for row in stats.values())
    reasons: dict[str, int] = {}
    for row in stats.values():
        for reason, count in row["reasons"].items():
            reasons[reason] = reasons.get(reason, 0) + count

    if not total:
        return {
            "id": "valid_answers",
            "title": "Answers fit the options you defined",
            "question": "Did every answer use your labels, with probabilities adding up to 1?",
            "status": "unknown",
            "value": "not checked",
            "detail": "No question in this workflow declares a fixed option set — "
            "yes/no questions answer with a single probability, so there is nothing "
            "to validate against.",
            "advice": "Nothing to do. This check applies to multiple-choice and rating questions.",
            "evidence": {},
        }

    rate = _rate(valid, total)
    if valid == total:
        status, detail = "ok", f"All {total} answers were valid."
    else:
        bad = total - valid
        status = "problem" if rate["rate"] < 0.99 else "watch"
        worst = max(reasons.items(), key=lambda kv: kv[1])[0] if reasons else "invalid"
        detail = f"{bad} of {total} answers were not valid. Most common: {worst}."
    return {
        "id": "valid_answers",
        "title": "Answers fit the options you defined",
        "question": "Did every answer use your labels, with probabilities adding up to 1?",
        "status": status,
        "value": f"{valid} / {total}",
        "detail": detail,
        "advice": (
            "Nothing to do — this is the one health signal that needs no ground truth, "
            "so it keeps working on live traffic."
            if status == "ok"
            else "An answer outside your own schema breaks the code downstream of it. "
            "Check the question's option list against what your policy expects. "
            "(Any invalid answer is flagged; below 99% valid it is called a problem.)"
        ),
        "evidence": {"per_question": stats, "reasons": reasons, "rate": rate},
    }


def _check_certainty(traces, unsure_below: float) -> dict:
    values = [c for c in (_decision_certainty(t) for t in traces) if c is not None]
    if not values:
        return {
            "id": "certainty",
            "title": "How often the model was unsure",
            "question": f"How many decisions landed below your {unsure_below:.2f} review line?",
            "status": "unknown",
            "value": "no distributions",
            "detail": "None of these decisions recorded a probability distribution.",
            "advice": "Check that the traces were written by Monitor.run().",
            "evidence": {},
        }
    unsure = sum(1 for c in values if c < unsure_below)
    rate = _rate(unsure, len(values))
    status = "ok" if rate["rate"] < 0.15 else "watch" if rate["rate"] < 0.30 else "problem"
    return {
        "id": "certainty",
        "title": "How often the model was unsure",
        "question": f"How many decisions landed below your {unsure_below:.2f} review line?",
        "status": status,
        "value": _pct(rate["rate"]),
        "detail": f"{unsure} of {len(values)} decisions put less than {unsure_below:.2f} "
        f"probability on their top answer — {_ci_text(rate)}. Median certainty was "
        f"{statistics.median(values):.2f}.",
        "advice": (
            "Healthy. These are the decisions to route to a human if you add review."
            if status == "ok"
            else "A high unsure rate usually means the options overlap or the question "
            "is underspecified, not that the model is weak. Look at the unsure "
            "decisions in the table below — they normally cluster on one option pair. "
            "(This check flags above 15% and calls it a problem above 30%; those are "
            "defaults, and the review line itself is --unsure-below.)"
        ),
        "evidence": {
            "threshold": unsure_below,
            "rate": rate,
            "median_certainty": statistics.median(values),
            "verdict_thresholds": {"ok_below": 0.15, "problem_at_or_above": 0.30},
        },
    }


def _repeat_key(trace: DecisionTrace) -> tuple:
    """What has to match before two decisions count as the same question, twice.

    The schema is part of the key, not just the input. Asking the same input under
    a reworded question is a different question — grouping the two would report a
    wording change as model jitter, which is the opposite of the truth.
    """
    schema = json.dumps(
        [q.model_dump() for q in sorted(trace.questions, key=lambda q: q.name)],
        sort_keys=True,
        default=str,
    )
    return (
        trace.node_id,
        json.dumps(trace.state, sort_keys=True, default=str),
        trace.question_version,
        schema,
    )


def _check_repeat_consistency(traces) -> dict:
    """Did the same input, asked twice, take the same branch?

    This is the only stability signal available without deliberately perturbing
    anything, and it needs no labels: identical inputs that answer differently are
    the model's own jitter. Most production streams have no exact repeats, in which
    case the honest answer is "not measured" rather than "stable".
    """
    groups: dict[tuple, list[DecisionTrace]] = {}
    for trace in traces:
        groups.setdefault(_repeat_key(trace), []).append(trace)
    repeats = {k: v for k, v in groups.items() if len(v) > 1}
    if not repeats:
        return {
            "id": "consistency",
            "title": "Same input, same answer",
            "question": "When the identical input came in twice, did it decide the same way?",
            "status": "unknown",
            "value": "no repeats",
            "detail": "No input appeared twice in this traffic, so repeat consistency "
            "could not be measured.",
            "advice": "To measure stability deliberately, run stability_check() with a "
            "handful of rephrasings of one real request. Rephrasing finds flips that "
            "repeating the same words never will.",
            "evidence": {},
        }
    flipped = []
    for (node, state, _version, _schema), rows in repeats.items():
        branches = {
            r.action if r.action is not None else tuple(answer_branch(a) for a in r.answers)
            for r in rows
        }
        if len(branches) > 1:
            flipped.append({"node_id": node, "state": _short(json.loads(state)), "n": len(rows)})
    rate = _rate(len(flipped), len(repeats))
    status = "ok" if not flipped else "watch" if rate["rate"] < 0.05 else "problem"
    return {
        "id": "consistency",
        "title": "Same input, same answer",
        "question": "When the identical input came in twice, did it decide the same way?",
        "status": status,
        "value": f"{len(flipped)} / {len(repeats)} flipped",
        "detail": f"{len(repeats)} inputs arrived more than once. "
        + (
            f"All of them decided the same way every time ({_ci_text(rate)} flipped)."
            if not flipped
            else f"{len(flipped)} decided differently on different occasions — {_ci_text(rate)}."
        ),
        "advice": (
            "Good, but note this only covers inputs that happened to repeat."
            if not flipped
            else "Identical inputs taking different branches is the model's own jitter. "
            "Compare it against a rephrasing test before blaming a change you made."
        ),
        "evidence": {"rate": rate, "flipped": flipped[:20]},
    }


def _translate_signal(signal: str) -> str:
    """Rewrite a drift signal into something a reader can act on."""
    if signal.startswith("action share changed"):
        return "Answers shifted: " + signal.split(":", 1)[1].strip()
    if signal.startswith("certainty moved"):
        return "The model got less sure of itself — " + signal[len("certainty moved: ") :]
    if signal.startswith("accuracy"):
        return "Accuracy moved — " + signal
    if signal.startswith("schema adherence"):
        return "Answers stopped fitting the declared options — " + signal
    return signal


def _check_drift(traces, labels, *, window_size: int | None) -> dict:
    """Is the newest traffic at each decision point behaving like the older traffic?

    Drift is tested **per decision point**, never across the whole stream. Two
    windows that carry different mixes of decision points are not comparable, and
    a shift in what was asked is indistinguishable from a shift in how it was
    answered — so the guard would fire on almost every real workflow and say
    nothing useful. Filtering to one node makes the windows comparable by
    construction.
    """
    by_node: dict[str, list[DecisionTrace]] = {}
    for trace in traces:
        by_node.setdefault(trace.node_id, []).append(trace)

    nodes, drifted_nodes, untestable = [], [], []
    for node in sorted(by_node):
        rows = by_node[node]
        size = window_size or max(5, len(rows) // 4)
        if len(rows) < 2 * size:
            untestable.append({"node_id": node, "n": len(rows), "need": 2 * size})
            continue
        report = drift_report(rows, size=size, labels=labels)
        entry = {
            "node_id": node,
            "n": len(rows),
            "window_size": size,
            "windows": report["windows"],
            "drifted": report["drifted"],
            "signals": [_translate_signal(s) for s in report["signals"]],
            "moved_actions": report["moved_actions"],
            "next_step": report["next_step"],
            "changed_components": report["attribution"]["changed_components"],
            "findings": report["attribution"]["findings"],
            "series": [
                {
                    "index": row["index"],
                    "n": row["n"],
                    "start": row["start"],
                    "mean_certainty": row["mean_pmax"],
                    "median_latency_ms": row["median_latency_ms"],
                    "accuracy": row["accuracy"],
                    "action_mix": row["action_mix"],
                }
                for row in report["series"]
            ],
        }
        nodes.append(entry)
        if report["drifted"]:
            drifted_nodes.append(entry)

    if not nodes:
        return {
            "id": "drift",
            "title": "Recent traffic still behaves like earlier traffic",
            "question": "Has anything started answering differently than it used to?",
            "status": "unknown",
            "value": "not enough history",
            "detail": "Every decision point has too little traffic to split into a "
            "current window and a baseline to compare it against.",
            "advice": "Keep collecting. A comparison needs roughly 10 decisions per "
            "decision point before it can show anything.",
            "evidence": {"untestable": untestable},
        }

    if drifted_nodes:
        headline = " | ".join(
            f"{e['node_id']}: " + "; ".join(e["signals"][:2] or ["changed"])
            for e in drifted_nodes[:3]
        )
        causes = sorted({c for e in drifted_nodes for c in e["changed_components"]})
        advice = (
            f"Start from what changed on the traces: {', '.join(causes)}. "
            "The diff is in the 'What changed' panel below."
            if causes
            else "Nothing recorded on the traces accounts for it — the wording, the "
            "options, the versions and the input shape are all unchanged. That points "
            "outside your config; run vendor_drift_check() on a sample of the older "
            "traffic to test whether the model itself moved."
        )
    else:
        headline, advice = "", (
            "No decision point moved enough to clear its statistical test. Note this "
            "is 'nothing shown', not 'proved identical' — small windows cannot show much."
        )

    return {
        "id": "drift",
        "title": "Recent traffic still behaves like earlier traffic",
        "question": "Has anything started answering differently than it used to?",
        "status": "problem" if drifted_nodes else "ok",
        "value": f"{len(drifted_nodes)} of {len(nodes)} decision points changed",
        "detail": (
            headline
            or f"Compared the newest window against all earlier traffic at "
            f"{len(nodes)} decision point(s); none of the shifts were statistically real."
        ),
        "advice": advice,
        "evidence": {"nodes": nodes, "untestable": untestable},
    }


def _check_outcomes(traces) -> dict:
    labelled = [t for t in traces if t.outcome_correct is not None]
    coverage = _rate(len(labelled), len(traces))
    if not labelled:
        return {
            "id": "outcomes",
            "title": "Results are coming back",
            "question": "Do you know how these decisions actually turned out?",
            "status": "unknown",
            "value": "0 recorded",
            "detail": "No decision has a recorded result, so accuracy and calibration "
            "cannot be measured. Everything else on this page still works — it is all "
            "computed without ground truth.",
            "advice": "Call monitor.record_outcome(trace, outcome=..., correct=...) when "
            "the answer arrives — a human correction, a click, a refund that went "
            "through. Even a few hundred unlocks accuracy and calibration.",
            "evidence": {"coverage": coverage},
        }
    correct = _rate(sum(bool(t.outcome_correct) for t in labelled), len(labelled))
    return {
        "id": "outcomes",
        "title": "Results are coming back",
        "question": "Do you know how these decisions actually turned out?",
        "status": "ok" if coverage["rate"] >= 0.05 else "watch",
        "value": f"{len(labelled)} of {len(traces)}",
        "detail": f"{_pct(coverage['rate'])} of decisions have a recorded result. "
        f"Of those, {_ci_text(correct)} matched it.",
        "advice": "Accuracy on a subset is only as representative as the subset. If "
        "results only come back for the decisions someone reviewed, read it as review "
        "accuracy, not workflow accuracy.",
        "evidence": {"coverage": coverage, "accuracy": correct},
    }


def _check_calibration(traces, *, min_labelled: int = 30) -> dict:
    per_question = calibration_report(traces)
    total = sum(row["n"] for row in per_question.values())
    if total < min_labelled:
        return {
            "id": "calibration",
            "title": "Stated certainty matches reality",
            "question": "When it says it is 80% sure, is it right about 80% of the time?",
            "status": "unknown",
            "value": f"{total} of ~{min_labelled} needed",
            "detail": "Calibration needs recorded results to compare against. There are "
            f"{total}; roughly {min_labelled} is the point where the number stops being noise.",
            "advice": "Record more outcomes, then come back to this check.",
            "evidence": {"per_question": per_question},
        }
    gap = sum(row["ece"] * row["n"] for row in per_question.values()) / total
    mean_p = sum(row["mean_probability"] * row["n"] for row in per_question.values()) / total
    acc = sum(row["accuracy"] * row["n"] for row in per_question.values()) / total
    status = "ok" if gap < 0.1 else "watch" if gap < 0.2 else "problem"
    direction = "over-confident" if mean_p > acc else "under-confident"
    return {
        "id": "calibration",
        "title": "Stated certainty matches reality",
        "question": "When it says it is 80% sure, is it right about 80% of the time?",
        "status": status,
        "value": f"gap {gap:.3f}",
        # Two different numbers, and conflating them is how a calibration report
        # gets read as "fine": the headline gap is the average per-bucket error,
        # while the overall lean nets out an over-confident band against an
        # under-confident one and can sit near zero while the gap is large.
        "detail": f"Across {total} decisions with a recorded result, the typical stated "
        f"certainty was {mean_p:.2f} and {_pct(acc)} turned out right. Bucket by bucket — "
        f"which is what the {gap:.3f} gap measures — the stated number was off by that "
        f"much on average; netted out, it leans {direction} by {abs(mean_p - acc):.2f}.",
        "advice": (
            "Certainty can be trusted as a routing signal at this gap. (A gap under 0.1 "
            "passes here, over 0.2 is a problem.)"
            if status == "ok"
            else f"The model is {direction}. Pick your review threshold from the observed "
            "curve below rather than from the stated number, and use the probability "
            "itself — never the API's `confidence` field, which rescales with the option count."
        ),
        "evidence": {
            "per_question": per_question,
            "gap": gap,
            "bins": calibration_bins(traces),
            "verdict_thresholds": {"ok_below": 0.1, "problem_at_or_above": 0.2},
        },
    }


def _check_wording(traces) -> dict:
    issues = [i.__dict__ for i in lint_questions(_latest_questions(traces))]
    warnings = [i for i in issues if i["severity"] == "warning"]
    if not issues:
        return {
            "id": "wording",
            "title": "Question wording holds up",
            "question": "Is anything in the schema vague, subjective or double-barrelled?",
            "status": "ok",
            "value": "no issues",
            "detail": "No question tripped a wording heuristic.",
            "advice": "These are five simple text heuristics, not a proof. A clean "
            "result is not evidence the wording is good.",
            "evidence": {"issues": []},
        }
    return {
        "id": "wording",
        "title": "Question wording holds up",
        "question": "Is anything in the schema vague, subjective or double-barrelled?",
        "status": "watch" if warnings else "info",
        "value": f"{len(issues)} flagged",
        "detail": "; ".join(f"{i['question']}: {i['message']}" for i in issues[:3])
        + ("…" if len(issues) > 3 else ""),
        "advice": "Wording is the cheapest thing to fix and the most common cause of an "
        "unstable decision. Rephrasings of one real request will tell you whether it matters.",
        "evidence": {"issues": issues},
    }


def _check_config(traces) -> dict:
    axes = {
        "model": "model version",
        "question_version": "question version",
        "policy_version": "policy version",
        "workflow_version": "workflow version",
    }
    changed = {}
    for attr, label in axes.items():
        # In first-seen order, not sorted: the arrow in the detail line reads as a
        # sequence, and alphabetical order would invent a history that did not happen.
        seen: list[str] = []
        for trace in traces:
            value = getattr(trace, attr)
            if value is not None and value not in seen:
                seen.append(value)
        if len(seen) > 1:
            changed[label] = seen
    if not changed:
        return {
            "id": "config",
            "title": "Configuration held still",
            "question": "Did the model, wording, options or policy change mid-stream?",
            "status": "ok",
            "value": "unchanged",
            "detail": "One model version, one question version, one policy version "
            "across this whole window.",
            "advice": "This is what makes the comparisons above meaningful: anything "
            "that moved, moved on its own.",
            "evidence": {},
        }
    return {
        "id": "config",
        "title": "Configuration changed mid-stream",
        "question": "Did the model, wording, options or policy change mid-stream?",
        "status": "info",
        "value": f"{len(changed)} axis changed",
        "detail": "; ".join(f"{k}: {' → '.join(v)}" for k, v in changed.items()),
        "advice": "Not a fault — but any behaviour change in this window has a candidate "
        "cause, and before/after comparisons that straddle the switch are comparing two "
        "different systems.",
        "evidence": {"changed": changed},
    }


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #


def _tiles(traces, checks, unsure_below) -> list[dict]:
    certainties = [c for c in (_decision_certainty(t) for t in traces) if c is not None]
    latencies = sorted(t.latency_ms for t in traces if t.latency_ms is not None)
    cost = sum(_cost(t) for t in traces)
    labelled = [t for t in traces if t.outcome_correct is not None]
    unsure = sum(1 for c in certainties if c < unsure_below)

    tiles = [
        {
            "label": "Decisions",
            "value": f"{len(traces):,}",
            "note": _plural(len({t.node_id for t in traces}), "decision point"),
        },
        {
            "label": "Typical certainty",
            "value": f"{statistics.median(certainties):.2f}" if certainties else "—",
            "note": "median probability on the chosen answer",
            "term": "certainty",
        },
        {
            "label": "Unsure",
            "value": _pct(unsure / len(certainties)) if certainties else "—",
            "note": f"below {unsure_below:.2f} — the human-review pile",
            "term": "unsure",
            "tone": next(c["status"] for c in checks if c["id"] == "certainty"),
        },
        {
            "label": "Speed",
            "value": f"{round(statistics.median(latencies)):,} ms" if latencies else "—",
            "note": f"p95 {round(latencies[int(0.95 * len(latencies)) - 1]):,} ms"
            if latencies
            else "no latency recorded",
        },
    ]
    if cost:
        tiles.append(
            {
                "label": "Cost per 1,000",
                "value": f"${1000 * cost / len(traces):,.2f}",
                "note": f"${cost:.4f} spent on this window",
            }
        )
    if labelled:
        correct = sum(bool(t.outcome_correct) for t in labelled)
        tiles.append(
            {
                "label": "Accuracy",
                "value": _pct(correct / len(labelled)),
                "note": f"on the {len(labelled)} with a recorded result",
                "term": "outcome",
            }
        )
    return tiles


def _certainty_histogram(traces, bins: int = 20) -> list[dict]:
    rows = [{"low": b / bins, "high": (b + 1) / bins, "n": 0} for b in range(bins)]
    for trace in traces:
        c = _decision_certainty(trace)
        if c is None:
            continue
        rows[min(int(c * bins), bins - 1)]["n"] += 1
    return rows


def _node_rows(traces, labels) -> list[dict]:
    by_node: dict[str, list[DecisionTrace]] = {}
    for trace in traces:
        by_node.setdefault(trace.node_id, []).append(trace)
    adherence = schema_adherence(traces, labels)
    rows = []
    for node in sorted(by_node):
        group = by_node[node]
        certainties = [c for c in (_decision_certainty(t) for t in group) if c is not None]
        latencies = [t.latency_ms for t in group if t.latency_ms is not None]
        labelled = [t for t in group if t.outcome_correct is not None]
        questions = sorted({a.question_name for t in group for a in t.answers})
        node_adherence = [adherence[q]["adherence"] for q in questions if q in adherence]
        rows.append(
            {
                "node_id": node,
                "n": len(group),
                "questions": questions,
                "types": sorted({a.type for t in group for a in t.answers}),
                "median_certainty": statistics.median(certainties) if certainties else None,
                "median_latency_ms": round(statistics.median(latencies)) if latencies else None,
                "accuracy": (
                    sum(bool(t.outcome_correct) for t in labelled) / len(labelled)
                    if labelled
                    else None
                ),
                "labelled": len(labelled),
                "adherence": min(node_adherence) if node_adherence else None,
                "top_action": (
                    max(action_mix(group).items(), key=lambda kv: kv[1])[0] if group else None
                ),
            }
        )
    return rows


def _volume_series(traces, *, buckets: int = 24, unsure_below: float) -> list[dict]:
    """Activity over time, in equal-count slices so every point carries equal weight."""
    rows = sorted(traces, key=lambda t: t.timestamp)
    size = max(1, len(rows) // buckets)
    out = []
    for window in count_windows(rows, size=size):
        group = window.traces
        certainties = [c for c in (_decision_certainty(t) for t in group) if c is not None]
        latencies = [t.latency_ms for t in group if t.latency_ms is not None]
        unsure = sum(1 for c in certainties if c < unsure_below)
        out.append(
            {
                "index": window.index,
                "start": window.start.isoformat(),
                "end": window.end.isoformat(),
                "n": len(group),
                "unsure_rate": unsure / len(certainties) if certainties else None,
                "mean_certainty": statistics.mean(certainties) if certainties else None,
                "median_latency_ms": round(statistics.median(latencies)) if latencies else None,
                "cost_usd": round(sum(_cost(t) for t in group), 6),
            }
        )
    return out


def _decision_rows(traces, *, limit: int, unsure_below: float) -> list[dict]:
    rows = []
    for trace in sorted(traces, key=lambda t: t.timestamp, reverse=True)[:limit]:
        certainty = _decision_certainty(trace)
        answers = []
        for ans in trace.answers:
            dist = answer_distribution(ans) or {}
            answers.append(
                {
                    "question": ans.question_name,
                    "type": ans.type,
                    "branch": answer_branch(ans),
                    "top": sorted(dist.items(), key=lambda kv: -kv[1])[:4],
                }
            )
        rows.append(
            {
                "trace_id": trace.trace_id,
                "time": trace.timestamp.isoformat(),
                "node_id": trace.node_id,
                "action": trace.action,
                "certainty": certainty,
                "unsure": certainty is not None and certainty < unsure_below,
                "latency_ms": round(trace.latency_ms) if trace.latency_ms else None,
                "outcome_correct": trace.outcome_correct,
                "state": _short(trace.state),
                "answers": answers,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def build_report(
    traces: Iterable[DecisionTrace],
    *,
    workflow_id: str | None = None,
    title: str | None = None,
    unsure_below: float = 0.6,
    window_size: int | None = None,
    recent: int = 60,
    live_interval_ms: int | None = None,
    links: Sequence[Mapping[str, str]] | None = None,
    note: str | None = None,
) -> dict:
    """Compute the whole monitoring report as one JSON-serializable dict.

    ``unsure_below`` is the probability under which a decision counts as one a
    human should see. It is deliberately a probability and not the API's
    ``confidence``, which rescales with the number of options, so the same
    threshold would mean different things for two questions.
    """
    rows = [t for t in traces if workflow_id is None or t.workflow_id == workflow_id]
    rows.sort(key=lambda t: t.timestamp)
    if not rows:
        raise ValueError(
            f"no traces to report on{f' for workflow_id={workflow_id!r}' if workflow_id else ''}"
        )

    labels = declared_label_sets(rows)
    checks = [
        _check_valid_answers(rows, labels),
        _check_certainty(rows, unsure_below),
        _check_drift(rows, labels, window_size=window_size),
        _check_repeat_consistency(rows),
        _check_calibration(rows),
        _check_outcomes(rows),
        _check_wording(rows),
        _check_config(rows),
    ]
    checks.sort(key=lambda c: (-SEVERITY[c["status"]], c["id"]))

    problems = [c for c in checks if c["status"] == "problem"]
    watches = [c for c in checks if c["status"] == "watch"]
    status = "problem" if problems else "watch" if watches else "ok"
    score = max(0, 100 - 25 * len(problems) - 8 * len(watches))

    if problems:
        noun = "thing needs" if len(problems) == 1 else "things need"
        summary = f"{len(problems)} {noun} attention: " + "; ".join(
            c["title"].lower() for c in problems
        )
    elif watches:
        noun = "thing" if len(watches) == 1 else "things"
        summary = f"Mostly healthy, {len(watches)} {noun} worth a look: " + "; ".join(
            c["title"].lower() for c in watches
        )
    else:
        summary = "Every check that could run came back clean."

    unknowns = [c for c in checks if c["status"] == "unknown"]
    drift = next(c for c in checks if c["id"] == "drift")
    # Every diff the traces support, not only the ones that moved behaviour. A
    # component that changed while behaviour held still is not a fault, but it is
    # the first thing anyone asks about when a number looks odd next week.
    # Every recorded diff the traces support, not only the ones that moved
    # behaviour: a component that changed while behaviour held still is not a
    # fault, but it is the first thing anyone asks about when a number looks odd
    # next week. The ``behaviour`` finding itself is left out — it is a JS
    # distance, which is a description and not a test, and on a small window it
    # is large whatever the deployment is doing. What fires is in the drift check.
    findings = [
        {**f, "node_id": node["node_id"], "behaviour_moved": node["drifted"]}
        for node in drift["evidence"].get("nodes", [])
        for f in node["findings"]
        if f["component"] != "behaviour"
    ]

    return {
        "meta": {
            "title": title or f"{rows[0].workflow_id} — decision health",
            "workflow_id": workflow_id or ", ".join(sorted({t.workflow_id for t in rows})),
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "first_seen": rows[0].timestamp.isoformat(),
            "last_seen": rows[-1].timestamp.isoformat(),
            "n": len(rows),
            "nodes": sorted({t.node_id for t in rows}),
            "models": sorted({t.model for t in rows if t.model}),
            "unsure_below": unsure_below,
            "live_interval_ms": live_interval_ms,
            "links": list(links or []),
            "note": note,
        },
        "headline": {
            "status": status,
            "score": score,
            "summary": summary,
            "unknown": [c["title"] for c in unknowns],
        },
        "checks": checks,
        "tiles": _tiles(rows, checks, unsure_below),
        "volume": _volume_series(rows, unsure_below=unsure_below),
        "actions": [
            {"action": action, "share": share, "n": round(share * len(rows))}
            for action, share in sorted(action_mix(rows).items(), key=lambda kv: -kv[1])
        ],
        "certainty_histogram": _certainty_histogram(rows),
        "reliability": calibration_bins(rows),
        "nodes": _node_rows(rows, labels),
        "questions": [
            {
                "name": q.name,
                "type": q.type,
                "instructions": q.instructions,
                "options": labels.get(q.name),
                "k": len(labels[q.name]) if q.name in labels else None,
            }
            for q in _latest_questions(rows)
        ],
        "attribution": findings,
        "decisions": _decision_rows(rows, limit=recent, unsure_below=unsure_below),
        "glossary": GLOSSARY,
    }


def render_html(report: Mapping[str, Any]) -> str:
    """Render a report as one self-contained HTML page — no network, no build step."""
    from importlib import resources

    template = (
        resources.files("evaljev").joinpath("assets/dashboard.html").read_text(encoding="utf-8")
    )
    payload = json.dumps(report, default=str).replace("</", "<\\/")
    return template.replace('"__EVALJEV_REPORT__"', payload)


def write_report(report: Mapping[str, Any], path) -> Any:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(report), encoding="utf-8")
    return out
