"""Export the published site from real traces, so it cannot drift from the run.

Two files, both generated, both computed by the library itself:

- ``docs/index.html`` — the monitoring dashboard, exactly what
  ``evaljev report`` writes for anyone else's traces, rendered here from the
  recorded JevBench run.
- ``docs/data.js`` — the measurements the "how it works" page walks through.

Regenerate after a new benchmark run:

    python benchmarks/export_demo_data.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaljev import (
    JsonlTraceStore,
    attribute,
    build_report,
    calibration_report,
    confidence_to_pmax,
    drift_report,
    render_html,
    schema_adherence,
)

REPO = Path(__file__).resolve().parent.parent
WORKFLOW = "jevbench"
OUT = REPO / "docs" / "data.js"
DASHBOARD = REPO / "docs" / "index.html"

# The run the published dashboard is rendered from: the baseline sweep plus the two
# deliberately degraded schema variants, in the order they were recorded.
RUNS = ("jevbench-baseline", "jevbench-vague", "jevbench-stripped")

# Measurements from live runs that are not recoverable from the stored traces.
# Each is reproduced by a script under benchmarks/ or documented in the README.
RESAMPLE = {
    "source": "5 repeats per phrasing against jev-1.13.0",
    "phrasings": [
        {
            "id": "original-policy-06-0",
            "text": "Policy: bookings can be cancelled free until 24 hours before "
            "departure, including exactly 24 hours. Departure is 24 hours away.",
            "p_true": [0.85, 0.86, 0.85, 0.85, 0.86],
            "expected": "yes",
        },
        {
            "id": "original-policy-06-1",
            "text": "There are exactly 24 hours until departure. Free cancellation is "
            "allowed at or before that 24-hour cutoff.",
            "p_true": [0.32, 0.32, 0.32, 0.33, 0.32],
            "expected": "yes",
        },
    ],
}

BATCHING = {
    "states": 10,
    "contamination_js": 0.0058,
    "jitter_js": 0.0035,
    "max_contamination_js": 0.0353,
    "max_jitter_js": 0.0353,
    "cost_saving_pct": 34.7,
    "token_saving_pct": 34.6,
    "latency_saving_pct": 49.4,
}

REPAIR = {
    "fault": "criteria descriptions rotated one position (same keys, same k)",
    "baseline_accuracy": "12/12",
    "broken_accuracy": "3/12",
    "rows": [
        {"stage": "validate", "detail": "label set / type / lint", "cost": "free", "rejected": 1},
        {"stage": "screen", "detail": "replay on the dev split", "cost": "cheap", "rejected": 0},
        {"stage": "verify", "detail": "replay on the holdout", "cost": "full", "rejected": 0},
    ],
    "verify": {
        "holdout_n": 7,
        "improvements": 7,
        "regressions": 0,
        "discordant": 7,
        "p_value": 0.0156,
        "verdict": "improvement",
    },
    "underpowered_run": {
        "holdout_n": 6,
        "decision_p": 0.25,
        "decision_verdict": "insufficient evidence",
        "probability_p": 0.0312,
        "probability_median_delta": 0.600,
        "probability_verdict": "improvement",
    },
    "rationale": (
        "The original descriptions were incorrectly mapped to their keys; this version "
        "realigns each description with its semantically correct action key."
    ),
}


def confidence_table(c: float = 0.55) -> list[dict]:
    return [{"k": k, "pmax": round(confidence_to_pmax(c, k), 4)} for k in (2, 3, 4, 5, 10)]


def write_dashboard() -> None:
    """Render the published dashboard with the same code path anyone else gets."""
    traces = []
    for name in RUNS:
        traces += JsonlTraceStore(REPO / "benchmarks" / f"{name}.jsonl").list(WORKFLOW)
    report = build_report(
        traces,
        title="JevBench run — decision health",
        note=(
            "<b>This is a live example.</b> Every number below was computed by EvalJev from "
            "156 real decisions recorded against the Jev API — nothing here is mocked up. "
            "The same page is one command away for your own workflow: "
            "<code>evaljev report traces.jsonl -o report.html</code> \u00b7 "
            "<a href=\"how-it-works.html\">How it works &rarr;</a>"
        ),
        links=[
            {"label": "How it works", "href": "how-it-works.html"},
            {"label": "GitHub", "href": "https://github.com/xxlya/evaljev"},
        ],
    )
    DASHBOARD.write_text(render_html(report), encoding="utf-8")
    head = report["headline"]
    print(f"wrote {DASHBOARD} ({DASHBOARD.stat().st_size // 1024} KB)")
    print(f"  {report['meta']['n']} decisions · {head['status']} · score {head['score']}/100")


def main() -> int:
    baseline = JsonlTraceStore(REPO / "benchmarks" / "jevbench-baseline.jsonl").list(WORKFLOW)
    vague = JsonlTraceStore(REPO / "benchmarks" / "jevbench-vague.jsonl").list(WORKFLOW)
    if not baseline:
        print("No baseline traces. Run benchmarks/run_jevbench.py first.")
        return 1

    labels = {t.metadata["family"]: t.metadata["labels"] for t in baseline}
    calib = calibration_report(baseline)
    adherence = schema_adherence(baseline, labels)

    families = []
    for name in sorted({t.metadata["family"] for t in baseline}):
        rows = [t for t in baseline if t.metadata["family"] == name]
        families.append(
            {
                "family": name,
                "qtype": rows[0].metadata["qtype"],
                "n": len(rows),
                "correct": sum(bool(t.outcome_correct) for t in rows),
                "accuracy": sum(bool(t.outcome_correct) for t in rows) / len(rows),
                "ece": calib[name]["ece"],
                "mean_probability": calib[name]["mean_probability"],
                "adherence": adherence[name]["adherence"],
                "p50_ms": round(statistics.median(t.latency_ms for t in rows)),
                "cost_usd": round(sum(t.metadata.get("cost_usd") or 0.0 for t in rows), 6),
                "k": len(rows[0].metadata["labels"]),
            }
        )

    latencies = sorted(t.latency_ms for t in baseline)

    # Paraphrase flips, recomputed from the stored traces rather than remembered.
    by_group: dict[str, list] = {}
    for t in baseline:
        by_group.setdefault(t.metadata["group"], []).append(t)
    pairs = {g: rows for g, rows in by_group.items() if len(rows) > 1}
    flipped = [
        g
        for g, rows in pairs.items()
        if len({r.action for r in rows}) > 1
    ]

    intent_only = [t for t in baseline + vague if t.metadata["family"] == "intent"]
    drift_all = drift_report(baseline + vague, size=36, labels=labels)
    drift_one = drift_report(intent_only, size=6, labels=labels)

    attribution = attribute(
        [t for t in baseline if t.metadata["family"] == "intent"],
        [t for t in vague if t.metadata["family"] == "intent"],
        confidence_threshold=0.55,
    )

    sample = next(t for t in baseline if t.metadata["family"] == "intent")
    sample_dict = json.loads(sample.model_dump_json())
    sample_dict["metadata"].pop("predicted_probs", None)

    data = {
        "generated_from": "benchmarks/jevbench-{baseline,vague}.jsonl",
        "model": baseline[0].model_version,
        "overall": {
            "n": len(baseline),
            "correct": sum(bool(t.outcome_correct) for t in baseline),
            "accuracy": sum(bool(t.outcome_correct) for t in baseline) / len(baseline),
            "cost_usd": round(sum(t.metadata.get("cost_usd") or 0.0 for t in baseline), 6),
            "cost_per_decision": round(
                sum(t.metadata.get("cost_usd") or 0.0 for t in baseline) / len(baseline), 6
            ),
            "p50_ms": round(statistics.median(latencies)),
            "p95_ms": round(latencies[int(0.95 * len(latencies)) - 1]),
            "adherence": min(row["adherence"] for row in adherence.values()),
            "paraphrase_pairs": len(pairs),
            "paraphrase_flips": len(flipped),
            "flipped_groups": flipped,
        },
        "families": families,
        "resample": RESAMPLE,
        "confidence_table": confidence_table(),
        "confidence_threshold": 0.55,
        "batching": BATCHING,
        "repair": REPAIR,
        "drift": {
            "mixed": {
                "comparable": drift_all["comparable"],
                "composition_shift": round(drift_all["composition_shift"], 4),
                "action_mix_shift": round(drift_all["action_mix_shift"], 4),
                "signals": drift_all["signals"],
                "next_step": drift_all["next_step"],
            },
            "filtered": {
                "comparable": drift_one["comparable"],
                "composition_shift": round(drift_one["composition_shift"], 4),
                "action_mix_shift": round(drift_one["action_mix_shift"], 4),
                "moved_actions": drift_one["moved_actions"],
                "drifted": drift_one["drifted"],
                "next_step": drift_one["next_step"],
            },
        },
        "attribution": {
            "changed_components": attribution["changed_components"],
            "findings": attribution["findings"],
        },
        "sample_trace": sample_dict,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "// Generated by benchmarks/export_demo_data.py — do not edit by hand.\n"
        "window.EVALJEV_DATA = " + json.dumps(data, indent=2, default=str) + ";\n",
        encoding="utf-8",
    )
    write_dashboard()
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    print(f"  {data['overall']['correct']}/{data['overall']['n']} correct, "
          f"{len(families)} families, {len(pairs)} paraphrase pairs, "
          f"{len(flipped)} flip(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
