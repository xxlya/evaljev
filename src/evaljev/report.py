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
    declared_label_sets,
    declared_labels,
    distribution_is_valid,
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

# Within one severity, the order a reader should take them in: what moved, then
# whether the answers are usable, then the slower-burning ones.
CHECK_ORDER = [
    "drift",
    "valid_answers",
    "certainty",
    "consistency",
    "calibration",
    "outcomes",
    "wording",
    "config",
]

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
    """A state as a person would read it, not as a serializer would write it.

    A dict of inputs renders as ``message: where is my parcel · channel: email``,
    longest text first, because that is what a reader scans for. JSON braces,
    quotes and ``\u2014`` escapes carry no information here and cost the reader
    the only thing this line has to do.
    """
    if isinstance(value, str):
        text = value
    elif isinstance(value, Mapping):
        parts = sorted(
            value.items(),
            key=lambda kv: -len(str(kv[1])) if isinstance(kv[1], str) else 0,
        )
        text = " · ".join(
            f"{k}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)}"
            for k, v in parts
        )
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
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


def _check_valid_answers(traces) -> dict:
    # No label map: a stream being monitored is exactly the stream whose schema
    # changed, and one global map would score every earlier answer against a label
    # that did not exist yet.
    stats = schema_adherence(traces)
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
        seen: list[str] = []
        for row in rows:
            branch = row.action if row.action is not None else "/".join(
                str(answer_branch(a)) for a in row.answers
            )
            if branch not in seen:
                seen.append(branch)
        if len(seen) > 1:
            flipped.append(
                {
                    "node_id": node,
                    "state": _short(json.loads(state)),
                    "n": len(rows),
                    "branches": seen,
                    "certainties": [
                        round(c, 2)
                        for c in (_decision_certainty(r) for r in rows)
                        if c is not None
                    ],
                    "trace_ids": [r.trace_id for r in rows],
                }
            )
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
            else f"{len(flipped)} decided differently on different occasions — "
            f"{_ci_text(rate)}. "
            + "; ".join(
                f"{row['node_id']} chose {' then '.join(row['branches'])}" for row in flipped[:2]
            )
            + "."
        ),
        "advice": (
            "Good, but note this only covers inputs that happened to repeat."
            if not flipped
            else "Identical inputs taking different branches is the model's own jitter. "
            "Compare it against a rephrasing test before blaming a change you made."
        ),
        "evidence": {"rate": rate, "flipped": flipped[:20]},
    }


def _p(value: float) -> str:
    return "p<0.001" if value < 0.001 else f"p={value:.3f}"


def _plain_signals(report: Mapping[str, Any]) -> list[str]:
    """What moved, in the words a reader would use.

    Built from the tested numbers rather than by rewriting the strings
    ``drift_report`` prints for a console, so the page and the library cannot
    drift apart in wording or in rounding.
    """
    out: list[str] = []
    for action in report["moved_actions"]:
        test = report["action_tests"][action]
        before, after = test["reference"]["rate"], test["current"]["rate"]
        direction = "up from" if after > before else "down from"
        out.append(
            f"\u201c{action}\u201d is now {_pct(after)} of decisions, {direction} "
            f"{_pct(before)} ({_p(test['p_value'])})"
        )
    certainty = report.get("certainty") or {}
    if certainty.get("moved"):
        out.append(
            f"typical certainty fell from {certainty['median_a']:.2f} to "
            f"{certainty['median_b']:.2f} ({_p(certainty['p_value'])})"
        )
    accuracy = report.get("accuracy")
    if accuracy and accuracy["changed"]:
        before, after = accuracy["reference"]["rate"], accuracy["current"]["rate"]
        verb = "rose" if after > before else "fell"
        out.append(
            f"accuracy on the decisions someone reviewed {verb} from {_pct(before)} to "
            f"{_pct(after)} ({_p(accuracy['p_value'])})"
        )
    for question, row in (report.get("adherence") or {}).items():
        if row["changed"]:
            out.append(
                f"answers on \u201c{question}\u201d stopped fitting the declared options: "
                f"{_pct(row['reference']['rate'])} \u2192 {_pct(row['current']['rate'])}"
            )
    return out or list(report["signals"])


def _plain_finding(finding: Mapping[str, Any]) -> str:
    """An attribution finding as prose, built from its evidence.

    Reading the structured evidence rather than the printed summary keeps the
    sentence short: a window that straddles two versions lists three of them in
    the summary, and what a reader needs is which one is in force now.
    """
    evidence = finding.get("evidence") or {}
    component, summary = finding.get("component"), finding.get("summary", "")

    if component in ("model", "question", "policy", "workflow") and "field" in evidence:
        field = str(evidence["field"]).replace("_", " ")
        current, reference = evidence.get("current") or [], evidence.get("reference") or []
        if current:
            was = [v for v in reference if v not in current]
            return f"{field} is now {current[-1]}" + (f" (was {was[-1]})" if was else "")
    if summary.endswith("instructions rewritten"):
        question = evidence.get("question", "this question")
        return f"the instructions for \u201c{question}\u201d were rewritten"
    if "criteria reworded" in summary:
        labels = evidence.get("labels") or []
        question = evidence.get("question", "this question")
        return (
            f"{len(labels)} option descriptions on \u201c{question}\u201d were reworded "
            "\u2014 the labels themselves are unchanged"
        )
    if "option set changed" in summary:
        return summary.replace("option set changed, k", "option count changed from").replace(
            " -> ", " to "
        )
    return summary


def _check_drift(traces, *, window_size: int | None) -> dict:
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
        # Eight a side is the floor the certainty test needs before its p-value is
        # worth reading, so windows never go below it however thin the traffic is.
        size = window_size or max(8, len(rows) // 4)
        if len(rows) < 2 * size:
            untestable.append({"node_id": node, "n": len(rows), "need": 2 * size})
            continue
        report = drift_report(rows, size=size)
        current = count_windows(rows, size=size, align="end")[-1]
        entry = {
            "node_id": node,
            "n": len(rows),
            "window_size": size,
            "current_from": current.start.isoformat(),
            "current_trace_ids": [t.trace_id for t in current.traces],
            "windows": report["windows"],
            "drifted": report["drifted"],
            "signals": _plain_signals(report),
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
    """Did anything about the setup move while this traffic was being served?

    Tracked per decision point, not per stream. Each node asks its own question
    with its own version, so pooling them produces a sequence of changes that
    never happened — and hides which node the change belongs to.
    """
    axes = {
        "model": "model",
        "question_version": "question wording",
        "policy_version": "policy",
        "workflow_version": "workflow",
    }
    by_node: dict[str, list[DecisionTrace]] = {}
    for trace in traces:
        by_node.setdefault(trace.node_id, []).append(trace)

    changed: list[dict] = []
    for node in sorted(by_node):
        for attr, label in axes.items():
            # First-seen order, not sorted: the arrow reads as a sequence, and
            # alphabetical order would invent a history.
            seen: list[str] = []
            for trace in by_node[node]:
                value = getattr(trace, attr)
                if value is not None and value not in seen:
                    seen.append(value)
            if len(seen) > 1:
                changed.append({"node_id": node, "axis": label, "versions": seen})

    if not changed:
        return {
            "id": "config",
            "title": "Configuration held still",
            "question": "Did the model, wording, options or policy change mid-stream?",
            "status": "ok",
            "value": "unchanged",
            "detail": "One model version, one question version and one policy version "
            "at every decision point across this whole window.",
            "advice": "This is what makes the comparisons above meaningful: anything "
            "that moved, moved on its own.",
            "evidence": {},
        }
    return {
        "id": "config",
        "title": "Configuration changed mid-stream",
        "question": "Did the model, wording, options or policy change mid-stream?",
        "status": "info",
        "value": f"{len(changed)} change(s)",
        "detail": "; ".join(
            f"{row['node_id']} {row['axis']}: {' → '.join(row['versions'])}" for row in changed[:3]
        )
        + ("…" if len(changed) > 3 else ""),
        "advice": "Not a fault — but any behaviour change in this window has a candidate "
        "cause, and a before/after comparison that straddles the switch is comparing two "
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


def _node_rows(traces) -> list[dict]:
    by_node: dict[str, list[DecisionTrace]] = {}
    for trace in traces:
        by_node.setdefault(trace.node_id, []).append(trace)
    adherence = schema_adherence(traces)
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


def _request_groups(traces, request_key: str) -> dict[str, list[DecisionTrace]]:
    """Decisions grouped into the request that produced them, in the order taken.

    A workflow makes several decisions per request, and reading them as unrelated
    rows loses the thing an engineer actually wants: the path one request took.
    Grouping needs an id shared across the nodes — by convention
    ``metadata["request_id"]`` — and without one this returns nothing rather than
    guessing, because guessing would invent a workflow that does not exist.
    """
    groups: dict[str, list[DecisionTrace]] = {}
    for trace in traces:
        value = trace.metadata.get(request_key) if isinstance(trace.metadata, dict) else None
        if value is None:
            continue
        groups.setdefault(str(value), []).append(trace)
    for rows in groups.values():
        rows.sort(key=lambda t: t.timestamp)
    return groups


def _step(trace: DecisionTrace, unsure_below: float) -> dict:
    certainty = _decision_certainty(trace)
    answer = trace.answers[0] if trace.answers else None
    dist = answer_distribution(answer) if answer else None
    return {
        "node_id": trace.node_id,
        "question": answer.question_name if answer else None,
        "type": answer.type if answer else None,
        "branch": trace.action if trace.action is not None else (answer_branch(answer) if answer else None),
        "certainty": certainty,
        "unsure": certainty is not None and certainty < unsure_below,
        "latency_ms": round(trace.latency_ms) if trace.latency_ms else None,
        "outcome_correct": trace.outcome_correct,
        "version": trace.question_version,
        "top": sorted((dist or {}).items(), key=lambda kv: -kv[1])[:4],
    }


def _trajectory(request_id: str, rows: Sequence[DecisionTrace], unsure_below: float) -> dict:
    steps = [_step(t, unsure_below) for t in rows]
    outcomes = [t.outcome_correct for t in rows if t.outcome_correct is not None]
    return {
        "request_id": request_id,
        "time": rows[0].timestamp.isoformat(),
        "input": _short(rows[0].state, 220),
        "steps": steps,
        "final_action": steps[-1]["branch"] if steps else None,
        "unsure_steps": sum(1 for s in steps if s["unsure"]),
        "outcome_correct": (all(outcomes) if outcomes else None),
    }


def _workflow_path(traces, groups) -> list[dict]:
    """The decision points in the order a request meets them."""
    position: dict[str, list[int]] = {}
    for rows in groups.values():
        for i, trace in enumerate(rows):
            position.setdefault(trace.node_id, []).append(i)
    by_node: dict[str, list[DecisionTrace]] = {}
    for trace in traces:
        by_node.setdefault(trace.node_id, []).append(trace)

    def order(node: str) -> tuple:
        seen = position.get(node)
        if seen:
            return (0, statistics.median(seen), node)
        first = min(t.timestamp for t in by_node[node])
        return (1, first.timestamp(), node)

    path = []
    for node in sorted(by_node, key=order):
        rows = by_node[node]
        certainties = [c for c in (_decision_certainty(t) for t in rows) if c is not None]
        answer = rows[-1].answers[0] if rows[-1].answers else None
        path.append(
            {
                "node_id": node,
                "question": answer.question_name if answer else None,
                "type": answer.type if answer else None,
                "n": len(rows),
                "median_certainty": statistics.median(certainties) if certainties else None,
                "top_action": (
                    max(action_mix(rows).items(), key=lambda kv: kv[1])[0] if rows else None
                ),
            }
        )
    return path


def _same_input_comparisons(traces, drift_check, unsure_below: float, limit: int = 4) -> dict:
    """The same input at a flagged decision point, before and after it was flagged.

    This is the closest thing to a controlled experiment that live traffic offers:
    identical state, identical node, one side of the window boundary each. It is
    evidence, not a test — the two sides are not a random split — so it is presented
    as examples and never as a verdict.
    """
    flagged = [n for n in drift_check["evidence"].get("nodes", []) if n["drifted"]]
    if not flagged:
        return {"pairs": [], "total": 0, "changed": 0}
    out: list[dict] = []
    for node in flagged:
        current_ids = set(node["current_trace_ids"])
        rows = [t for t in traces if t.node_id == node["node_id"]]
        by_state: dict[str, dict[str, list[DecisionTrace]]] = {}
        for trace in rows:
            key = json.dumps(trace.state, sort_keys=True, default=str)
            side = "after" if trace.trace_id in current_ids else "before"
            by_state.setdefault(key, {"before": [], "after": []})[side].append(trace)
        pairs = []
        for key, sides in by_state.items():
            if not sides["before"] or not sides["after"]:
                continue
            before, after = sides["before"][-1], sides["after"][-1]
            pairs.append(
                {
                    "node_id": node["node_id"],
                    "input": _short(json.loads(key), 200),
                    "before": _step(before, unsure_below),
                    "after": _step(after, unsure_below),
                    "changed": (before.action != after.action),
                }
            )
        # Changed decisions first: they are the ones worth a reader's attention.
        pairs.sort(
            key=lambda p: (
                not p["changed"],
                -abs((p["before"]["certainty"] or 0) - (p["after"]["certainty"] or 0)),
            )
        )
        out += pairs
    # The counts describe every pair found; only the examples are trimmed, because
    # "6 of 6" from a truncated list would be a different and much stronger claim.
    return {
        "pairs": out[:limit],
        "total": len(out),
        "changed": sum(1 for p in out if p["changed"]),
    }


FLAG_LABELS = {
    "unsure": "model was unsure",
    "invalid": "answer broke the schema",
    "inconsistent": "same input decided differently",
    "disagreed": "a reviewer disagreed",
    "changed": "decided after this step changed",
}

#: Which flag a request is filed under when it trips several. An answer that is not
#: a valid distribution outranks everything: whatever the code downstream did with
#: it, it did by accident. A reviewer's disagreement is next, because it is the one
#: flag that is already confirmed rather than suspected.
FLAG_PRIORITY = ["invalid", "disagreed", "inconsistent", "unsure", "changed"]

#: One line per flag, because a queue that says what is wrong and not what to do
#: about it just moves the thinking somewhere else.
FLAG_ADVICE = {
    "unsure": "Have a person decide this one. If a lot of requests land here, the "
    "options probably overlap — that is a schema problem, not a model problem.",
    "invalid": "Do not act on this answer: it is not a distribution over the options "
    "you declared, so whatever your code did with it, it did by accident.",
    "inconsistent": "Check both answers. An identical input answering two ways is the "
    "model's own jitter, and this decision is on the wrong side of a boundary.",
    "disagreed": "Already reviewed and wrong. These are the examples to reach for when "
    "rewording the question.",
    "changed": "Recheck what this step decided after the change, and read the full "
    "report for the diff of what moved.",
}


def _flag_sets(traces, checks, drift) -> dict[str, dict[str, Any]]:
    """Per-trace flags, resolved once so the queue is a lookup rather than a scan."""
    labels_ok: dict[str, tuple[bool, str | None]] = {}
    for trace in traces:
        by_name = {q.name: q for q in trace.questions}
        verdict: tuple[bool, str | None] = (True, None)
        for ans in trace.answers:
            question = by_name.get(ans.question_name)
            expected = declared_labels(question) if question else None
            if expected is None:
                continue
            ok, reason = distribution_is_valid(ans.probabilities, expected)
            if not ok:
                verdict = (False, reason)
                break
        labels_ok[trace.trace_id] = verdict

    inconsistent: set[str] = set()
    consistency = next((c for c in checks if c["id"] == "consistency"), None)
    for row in (consistency or {}).get("evidence", {}).get("flipped", []):
        inconsistent.update(row.get("trace_ids", []))

    changed_nodes = {
        node["node_id"]: set(node["current_trace_ids"])
        for node in drift["evidence"].get("nodes", [])
        if node["drifted"]
    }
    return {
        "valid": labels_ok,
        "inconsistent": inconsistent,
        "changed": changed_nodes,
    }


def _flags_for(trace, certainty, unsure_below, resolved) -> list[dict]:
    """Every reason this one decision is worth a person's time."""
    flags = []
    if certainty is not None and certainty < unsure_below:
        flags.append(
            {
                "code": "unsure",
                "severity": "high",
                "node_id": trace.node_id,
                "detail": f"{certainty:.2f} on its top answer, under your {unsure_below:.2f} "
                "review line",
            }
        )
    ok, reason = resolved["valid"].get(trace.trace_id, (True, None))
    if not ok:
        flags.append(
            {
                "code": "invalid",
                "severity": "high",
                "node_id": trace.node_id,
                "detail": reason or "the distribution did not match the declared options",
            }
        )
    if trace.trace_id in resolved["inconsistent"]:
        flags.append(
            {
                "code": "inconsistent",
                "severity": "high",
                "node_id": trace.node_id,
                "detail": "an identical input took a different branch on another occasion",
            }
        )
    if trace.outcome_correct is False:
        flags.append(
            {
                "code": "disagreed",
                "severity": "high",
                "node_id": trace.node_id,
                "detail": "the recorded outcome does not match what was decided",
            }
        )
    if trace.trace_id in resolved["changed"].get(trace.node_id, set()):
        flags.append(
            {
                "code": "changed",
                "severity": "medium",
                "node_id": trace.node_id,
                "detail": "this decision point started answering differently in this window, "
                "so its answers are suspect even when they look confident",
            }
        )
    return flags


def _queue(groups, traces, checks, drift, unsure_below: float, limit: int = 80) -> dict:
    """The operator's view: which requests need a person, and where they went wrong.

    A request is triaged on its worst step. ``needs a person`` is not a guess about
    the application's own escalation rules — it is the review line the caller passed
    in, plus the three failures that make an answer unusable whatever the threshold
    was: a distribution outside the schema, an identical input answered two ways, and
    an outcome someone already disagreed with.
    """
    resolved = _flag_sets(traces, checks, drift)
    rows, by_node, by_flag = [], {}, {}
    counts = {"needs_human": 0, "watch": 0, "auto": 0}

    for request_id, steps in groups.items():
        flags, step_rows = [], []
        for trace in steps:
            certainty = _decision_certainty(trace)
            step_flags = _flags_for(trace, certainty, unsure_below, resolved)
            step = _step(trace, unsure_below)
            step["flags"] = [
                {**f, "label": FLAG_LABELS[f["code"]], "advice": FLAG_ADVICE[f["code"]]}
                for f in step_flags
            ]
            step_rows.append(step)
            flags += step["flags"]

        severity = {f["severity"] for f in flags}
        status = "needs_human" if "high" in severity else "watch" if flags else "auto"
        counts[status] += 1
        for flag in flags:
            by_node[flag["node_id"]] = by_node.get(flag["node_id"], 0) + 1
            by_flag[flag["code"]] = by_flag.get(flag["code"], 0) + 1
        if status == "auto":
            continue

        # The row's reason is its worst flag, not its first: "decided after this
        # step changed" must not stand in front of "the answer broke the schema".
        worst = min(
            (f for s in step_rows for f in s["flags"]),
            key=lambda f: FLAG_PRIORITY.index(f["code"]),
            default=None,
        )
        first = next(
            (s for s in step_rows if worst and worst in s["flags"]), step_rows[0]
        )
        rows.append(
            {
                "request_id": request_id,
                "time": steps[0].timestamp.isoformat(),
                "input": _short(steps[0].state, 160),
                "status": status,
                "flagged_at": first["node_id"],
                "reason": worst["label"] if worst else "",
                "detail": worst["detail"] if worst else "",
                "advice": worst["advice"] if worst else "",
                "final_action": step_rows[-1]["branch"],
                "steps": step_rows,
            }
        )

    rows.sort(key=lambda r: (r["status"] != "needs_human", r["time"]), reverse=False)
    rows.sort(key=lambda r: (r["status"] == "needs_human", r["time"]), reverse=True)
    return {
        "rows": rows[:limit],
        "counts": counts,
        "total": sum(counts.values()),
        "shown": min(len(rows), limit),
        "by_node": by_node,
        "by_flag": [
            {"code": code, "label": FLAG_LABELS[code], "n": n}
            for code, n in sorted(by_flag.items(), key=lambda kv: -kv[1])
        ],
        "review_line": unsure_below,
    }


def _story(checks, drift, comparisons, findings, n_requests: int) -> dict:
    """The two or three sentences a reader needs before any chart.

    Built here rather than in the page, because it is a claim about the data and
    claims belong where they can be tested. Every line is assembled from a check
    that already ran; nothing here re-derives a number of its own.
    """
    flagged = [n for n in drift["evidence"].get("nodes", []) if n["drifted"]]
    problems = [c for c in checks if c["status"] == "problem"]

    if not flagged:
        other = [c for c in problems if c["id"] != "drift"]
        if other:
            return {
                "kind": "problem",
                "headline": other[0]["title"],
                "when": None,
                "node_id": None,
                "bullets": [{"label": c["title"], "text": c["detail"]} for c in other[:3]],
            }
        return {
            "kind": "clean",
            "headline": "Nothing has changed in how this workflow decides",
            "when": None,
            "node_id": None,
            "bullets": [
                {
                    "label": "Compared",
                    "text": drift["detail"],
                },
                {
                    "label": "Worth remembering",
                    "text": "A clean comparison is not proof the workflow is correct — it "
                    "says nothing moved. Checks that could not run are listed as not measured.",
                },
            ],
        }

    node = flagged[0]
    bullets = [{"label": "What moved", "text": "; ".join(node["signals"])}]

    causes = [f for f in findings if f["node_id"] == node["node_id"]]
    if causes:
        bullets.append(
            {
                "label": "What changed underneath",
                "text": "; ".join(dict.fromkeys(_plain_finding(f) for f in causes)),
            }
        )
    else:
        bullets.append(
            {
                "label": "What changed underneath",
                "text": "Nothing recorded on the traces — the wording, the options, the "
                "versions and the input shape are unchanged. That points outside your "
                "config; vendor_drift_check() is the next step.",
            }
        )

    if comparisons["changed"]:
        landing: dict[str, int] = {}
        for row in comparisons["pairs"]:
            if row["changed"]:
                landing[row["after"]["branch"]] = landing.get(row["after"]["branch"], 0) + 1
        where = ", ".join(f"{k}" for k, _ in sorted(landing.items(), key=lambda kv: -kv[1]))
        bullets.append(
            {
                "label": "What it did to requests",
                "text": f"{comparisons['changed']} of the {comparisons['total']} inputs that "
                f"arrived both before and after took a different branch the second time"
                + (f", now going to {where}" if where else "")
                + ". Same input, same workflow, different answer.",
            }
        )

    return {
        "kind": "incident",
        "headline": f"{node['node_id']} started answering differently",
        "when": node["current_from"],
        "node_id": node["node_id"],
        "bullets": bullets,
    }


def _select_trajectories(trajectories: Sequence[dict], limit: int = 8) -> list[dict]:
    """A few requests worth opening: the ones that stalled, then ordinary ones.

    A reader wants two things from this section — what a healthy path looks like,
    and what the bad ones have in common. So the awkward requests come first and an
    ordinary one is always kept, even when everything is on fire.
    """
    if not trajectories:
        return []
    ranked = sorted(
        trajectories,
        key=lambda t: (
            -t["unsure_steps"],
            t["outcome_correct"] is False and -1 or 0,
            -(len(t["steps"])),
        ),
    )
    # One entry per distinct input: the same message stalling three times is one
    # fact, and three chips that read alike are harder to scan than three facts.
    picked, seen_inputs = [], set()
    for row in ranked:
        if row["input"] in seen_inputs:
            continue
        seen_inputs.add(row["input"])
        picked.append(row)
        if len(picked) >= limit - 1:
            break
    ordinary = [
        t for t in reversed(trajectories)
        if t["unsure_steps"] == 0 and t["input"] not in seen_inputs
    ]
    if ordinary:
        picked.append(ordinary[0])
    return sorted(picked, key=lambda t: t["time"])


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
    request_key: str = "request_id",
) -> dict:
    """Compute the whole monitoring report as one JSON-serializable dict.

    ``unsure_below`` is the probability under which a decision counts as one a
    human should see. It is deliberately a probability and not the API's
    ``confidence``, which rescales with the number of options, so the same
    threshold would mean different things for two questions.

    ``request_key`` names the metadata field that ties the decisions of one
    request together (``metadata={"request_id": ...}`` on every ``Monitor.run``).
    With it the report can show the path a request took through the workflow
    rather than a list of unrelated decisions; without it that section is simply
    absent, since inferring the grouping would be inventing one.
    """
    rows = [t for t in traces if workflow_id is None or t.workflow_id == workflow_id]
    rows.sort(key=lambda t: t.timestamp)
    if not rows:
        raise ValueError(
            f"no traces to report on{f' for workflow_id={workflow_id!r}' if workflow_id else ''}"
        )

    labels = declared_label_sets(rows)
    checks = [
        _check_valid_answers(rows),
        _check_certainty(rows, unsure_below),
        _check_drift(rows, window_size=window_size),
        _check_repeat_consistency(rows),
        _check_calibration(rows),
        _check_outcomes(rows),
        _check_wording(rows),
        _check_config(rows),
    ]
    checks.sort(key=lambda c: (-SEVERITY[c["status"]], CHECK_ORDER.index(c["id"])))

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
    groups = _request_groups(rows, request_key)
    trajectories = [
        _trajectory(rid, group_rows, unsure_below)
        for rid, group_rows in sorted(groups.items(), key=lambda kv: kv[1][0].timestamp)
    ]
    comparisons = _same_input_comparisons(rows, drift, unsure_below)
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
        "nodes": _node_rows(rows),
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
        "story": _story(checks, drift, comparisons, findings, len(groups)),
        "queue": _queue(groups, rows, checks, drift, unsure_below),
        "workflow": _workflow_path(rows, groups),
        "trajectories": _select_trajectories(trajectories),
        "comparisons": comparisons,
        "decisions": _decision_rows(rows, limit=recent, unsure_below=unsure_below),
        "glossary": GLOSSARY,
    }


VIEWS = {"console": "assets/console.html", "report": "assets/dashboard.html"}


def render_html(report: Mapping[str, Any], view: str = "console") -> str:
    """Render a report as one self-contained HTML page — no network, no build step.

    ``console`` is the operational view: which requests need a person, and where in
    the workflow they were flagged. ``report`` is the full analysis behind it. Both
    read the same report dict, so the two can never disagree about a number.
    """
    from importlib import resources

    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; expected one of {sorted(VIEWS)}")
    template = resources.files("evaljev").joinpath(VIEWS[view]).read_text(encoding="utf-8")
    payload = json.dumps(report, default=str).replace("</", "<\\/")
    return template.replace('"__EVALJEV_REPORT__"', payload)


def write_report(report: Mapping[str, Any], path, view: str = "console") -> Any:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(report, view), encoding="utf-8")
    return out
