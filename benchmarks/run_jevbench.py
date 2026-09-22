"""Run a JevBench split through EvalJev's monitoring layer, under a spend cap.

    python benchmarks/run_jevbench.py --data <jevbench>/datasets/public/original.jsonl
    python benchmarks/run_jevbench.py --data ... --degrade rotate-criteria --out degraded.jsonl
    python benchmarks/run_jevbench.py --compare baseline.jsonl degraded.jsonl

This is not a JevBench re-implementation. JevBench asks "is this decision model
good?" and needs labels to answer. EvalJev asks "is this deployed decision
workflow still healthy?", which has to be answerable without labels — so the
report separates the two: signals that need ground truth, and signals that do
not. The labelled split is how we check the label-free signals actually fire.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import httpx
from jevbench_adapter import (
    BenchTask,
    expected_level,
    load_split,
    paraphrase_groups,
    to_label_space,
)

from evaljev import (
    BudgetedJevClient,
    BudgetExceeded,
    JevHTTPClient,
    JsonlTraceStore,
    Monitor,
    SpendBudget,
    calibration_report,
    distribution_shift,
    paired_comparison,
    schema_adherence,
    wilson_interval,
)

WORKFLOW = "jevbench"


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        load_dotenv(env, override=False)


def degrade(question: dict, mode: str) -> dict:
    """Inject a workflow fault of the kind that happens in production.

    `strip-criteria` drops the per-label rubric while keeping the labels — the
    schema edit someone makes to "clean up" a prompt, with no test that catches it.

    `rotate-criteria` shifts every description one position along, so each label
    keeps its name but inherits its neighbour's meaning. Same keys, same `k`, same
    instructions: nothing a schema validator, a diff of the label set, or a
    confidence threshold would notice — and the model answers confidently and
    wrongly. It is the mistake a copy-paste or a reordered dict actually makes.
    """
    if mode == "none":
        return question
    if mode == "strip-criteria":
        return {k: v for k, v in question.items() if k != "criteria"}
    if mode == "vague-instructions":
        return {**question, "instructions": "Pick the best option."}
    if mode == "rotate-criteria":
        criteria = question.get("criteria")
        if isinstance(criteria, dict):
            keys = list(criteria)
            values = [criteria[k] for k in keys]
            rotated = dict(zip(keys, values[1:] + values[:1]))
            return {**question, "criteria": rotated}
        if isinstance(criteria, list):
            return {**question, "criteria": criteria[1:] + criteria[:1]}
        return question
    raise ValueError(f"unknown degradation: {mode}")


def run_split(tasks: list[BenchTask], *, cap: float, mode: str, out: Path) -> list:
    out.unlink(missing_ok=True)
    budget = SpendBudget({"jev": cap})
    client = BudgetedJevClient(JevHTTPClient(), budget)
    monitor = Monitor(JsonlTraceStore(out))
    rejected: list[tuple[str, int, str]] = []

    for i, task in enumerate(tasks, 1):
        questions = {task.family: degrade(task.question, mode)}
        predicted: dict = {}

        def policy(answers, _task=task, _slot=predicted):
            label, probs = to_label_space(answers[_task.family], _task)
            _slot["label"], _slot["probs"] = label, probs
            return label

        try:
            response, trace = monitor.run(
                client,
                workflow_id=WORKFLOW,
                node_id=task.family,
                state=task.state,
                questions=questions,
                policy=policy,
                question_version=mode,
                workflow_version="jevbench-public",
                metadata={
                    "task_id": task.id,
                    "family": task.family,
                    "group": task.group,
                    "qtype": task.qtype,
                    "labels": task.labels,
                    "expected": task.expected,
                },
            )
        except BudgetExceeded as e:
            print(f"  stopped after {i - 1} tasks: {e}")
            break
        except httpx.HTTPStatusError as e:
            # A rejected schema is itself a finding: record it and keep going,
            # rather than losing the rest of the run to one bad item.
            body = e.response.text[:120]
            rejected.append((task.id, e.response.status_code, body))
            if len(rejected) <= 3:
                print(f"  {task.id}: HTTP {e.response.status_code} {body}")
            continue

        # Both are only known after the call; record_outcome persists them below.
        probs = predicted.get("probs")
        trace.metadata["predicted_probs"] = probs
        trace.metadata["cost_usd"] = (response.get("usage") or {}).get("cost_usd")
        # Put the answer in label space on the trace itself. A noul answer is a bare
        # scalar, so without this schema_adherence sees "no distribution" and reports
        # a violation that did not happen. The raw scalar stays in answer.value.
        if probs:
            trace.answers[0].probabilities = probs
        if task.qtype == "noul":
            # Calibration for noul reads the truth of the proposition, not a confidence.
            trace.metadata["noul_truth"] = {task.family: task.expected == "yes"}
        monitor.record_outcome(
            trace,
            outcome={"predicted": trace.action},
            correct=(trace.action == task.expected),
        )
        if i % 12 == 0:
            print(f"  {i}/{len(tasks)} … ${budget.spent['jev']:.4f}")

    traces = JsonlTraceStore(out).list(WORKFLOW)
    print(
        f"  {len(traces)} traces | ${budget.spent['jev']:.4f} of ${cap:.2f} | "
        f"credits left ${client.credits_remaining_usd:.4f}"
    )
    if rejected:
        codes = {code for _, code, _ in rejected}
        print(f"  {len(rejected)}/{len(tasks)} rejected by the API (status {sorted(codes)})")
    return traces


def by_family(traces) -> dict:
    out: dict[str, dict] = {}
    for t in traces:
        row = out.setdefault(t.metadata["family"], {"n": 0, "correct": 0, "latency": [], "cost": 0.0})
        row["n"] += 1
        row["correct"] += int(bool(t.outcome_correct))
        row["latency"].append(t.latency_ms)
        row["cost"] += t.metadata.get("cost_usd") or 0.0
    for row in out.values():
        row["accuracy"] = row["correct"] / row["n"]
        row["p50_ms"] = round(statistics.median(row["latency"]))
        row["cost_usd"] = round(row["cost"], 6)
        del row["latency"], row["cost"]
    return out


def paraphrase_stability(traces, groups: dict) -> dict:
    """Branch flips between phrasings of the same decision. Needs no labels."""
    by_id = {t.metadata["task_id"]: t for t in traces}
    flips, shifts, pairs, flipped_ids = 0, [], 0, []
    for group, rows in groups.items():
        present = [by_id[r.id] for r in rows if r.id in by_id]
        if len(present) < 2:
            continue
        pairs += 1
        base, *rest = present
        for other in rest:
            if base.action != other.action:
                flips += 1
                flipped_ids.append(group)
            p, q = base.metadata.get("predicted_probs"), other.metadata.get("predicted_probs")
            if p and q:
                shifts.append(distribution_shift(p, q))
    lo, hi = wilson_interval(flips, pairs)
    return {
        "pairs": pairs,
        "flips": flips,
        "branch_flip_rate": flips / pairs if pairs else 0.0,
        "ci": (lo, hi),
        "mean_distribution_shift": round(statistics.mean(shifts), 4) if shifts else None,
        "flipped_groups": flipped_ids,
    }


def report(traces, tasks, title: str) -> dict:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")
    labels = {t.metadata["family"]: t.metadata["labels"] for t in traces}
    correct = sum(bool(t.outcome_correct) for t in traces)
    lat = [t.latency_ms for t in traces]
    cost = sum(t.metadata.get("cost_usd") or 0.0 for t in traces)

    print("\n-- needs ground truth (what JevBench measures) --")
    print(f"accuracy: {correct}/{len(traces)} = {correct / len(traces):.3f}")
    fam = by_family(traces)
    for name in sorted(fam):
        r = fam[name]
        print(f"  {name:<12} acc {r['accuracy']:.3f}  n={r['n']:<3} p50 {r['p50_ms']}ms  ${r['cost_usd']:.5f}")
    print("\ncalibration (per family):")
    qtypes = {t.metadata["family"]: t.metadata["qtype"] for t in traces}
    calib = calibration_report(traces)
    for name in sorted(calib):
        c = calib[name]
        # For noul the paired probability is P(proposition true), so the rate column
        # is the base rate of the proposition, not the decision accuracy above.
        rate = "base" if qtypes.get(name) == "noul" else "acc "
        print(
            f"  {name:<12} n={c['n']:<3} {rate} {c['accuracy']:.3f}  p̄ {c['mean_probability']:.3f}  "
            f"ECE {c['ece']:.3f}  Brier {c['brier']:.3f}"
        )
    ordinal = [t for t in traces if t.metadata["qtype"] == "score" and t.metadata["expected"]]
    if ordinal:
        errs = [
            abs(expected_level(t.metadata["predicted_probs"]) - float(t.metadata["expected"]))
            for t in ordinal
            if expected_level(t.metadata["predicted_probs"]) is not None
        ]
        if errs:
            print(f"  ordinal MAE (expected-level): {statistics.mean(errs):.3f} over n={len(errs)}")

    print("\n-- no ground truth needed (what EvalJev can watch in production) --")
    adherence = schema_adherence(traces, labels)
    for name in sorted(adherence):
        a = adherence[name]
        note = f"  reasons={a['reasons']}" if a["reasons"] else ""
        print(f"  {name:<12} schema adherence {a['adherence']:.3f} ({a['valid']}/{a['n']}){note}")
    stab = paraphrase_stability(traces, paraphrase_groups(tasks))
    print(
        f"  paraphrase pairs: {stab['pairs']}  branch-flip rate {stab['branch_flip_rate']:.3f} "
        f"95% CI [{stab['ci'][0]:.3f}, {stab['ci'][1]:.3f}]  mean JS shift {stab['mean_distribution_shift']}"
    )
    if stab["flipped_groups"]:
        print(f"  flipped groups: {', '.join(stab['flipped_groups'])}")
    print(f"\nlatency p50 {statistics.median(lat):.0f}ms  p95 {sorted(lat)[int(0.95 * len(lat)) - 1]:.0f}ms")
    print(f"total cost ${cost:.5f} over {len(traces)} decisions (${cost / len(traces):.6f}/decision)")
    return {
        "accuracy": correct / len(traces),
        "adherence": adherence,
        "stability": stab,
        "calibration": calib,
    }


def compare(baseline: Path, candidate: Path, tasks) -> None:
    """Diff two runs the way a deployment would see them: label-free first."""
    a = JsonlTraceStore(baseline).list(WORKFLOW)
    b = JsonlTraceStore(candidate).list(WORKFLOW)
    by_id_a = {t.metadata["task_id"]: t for t in a}
    shared = [(by_id_a[t.metadata["task_id"]], t) for t in b if t.metadata["task_id"] in by_id_a]

    flips = [(x, y) for x, y in shared if x.action != y.action]
    shifts = [
        distribution_shift(x.metadata["predicted_probs"], y.metadata["predicted_probs"])
        for x, y in shared
        if x.metadata.get("predicted_probs") and y.metadata.get("predicted_probs")
    ]
    labels = {t.metadata["family"]: t.metadata["labels"] for t in a}

    print(f"\n{'=' * 74}\nWorkflow diff: {baseline.name} -> {candidate.name}\n{'=' * 74}")
    print("\n-- label-free signals (available in production, on live traffic) --")
    print(f"  decisions compared:    {len(shared)}")
    print(f"  action change rate:    {len(flips) / len(shared):.3f}  ({len(flips)} changed)")
    print(f"  mean distribution shift: {statistics.mean(shifts):.4f}" if shifts else "")
    ad_a = schema_adherence(a, labels)
    ad_b = schema_adherence(b, labels)
    for name in sorted(set(ad_a) | set(ad_b)):
        before = ad_a.get(name, {}).get("adherence", 0.0)
        after = ad_b.get(name, {}).get("adherence", 0.0)
        mark = "  <-- REGRESSION" if after < before else ""
        print(f"  adherence {name:<12} {before:.3f} -> {after:.3f}{mark}")
    stab_a = paraphrase_stability(a, paraphrase_groups(tasks))
    stab_b = paraphrase_stability(b, paraphrase_groups(tasks))
    # Only call it a regression when the intervals do not overlap.
    overlap = stab_a["ci"][1] >= stab_b["ci"][0] and stab_b["ci"][1] >= stab_a["ci"][0]
    mark = "" if overlap else "  <-- REGRESSION"
    print(
        f"  paraphrase flip rate   {stab_a['branch_flip_rate']:.3f} "
        f"[{stab_a['ci'][0]:.3f}, {stab_a['ci'][1]:.3f}] -> "
        f"{stab_b['branch_flip_rate']:.3f} [{stab_b['ci'][0]:.3f}, {stab_b['ci'][1]:.3f}]{mark}"
    )
    if overlap:
        print("    (intervals overlap — not distinguishable at this sample size)")

    print("\n-- with ground truth (only available offline; confirms the above) --")
    acc_a = sum(bool(t.outcome_correct) for t in a) / len(a)
    acc_b = sum(bool(t.outcome_correct) for t in b) / len(b)
    print(f"  accuracy {acc_a:.3f} -> {acc_b:.3f}  (delta {acc_b - acc_a:+.3f})")

    # Paired, because both runs saw the same items. Unpaired accuracy deltas on
    # n=72 cannot separate a real change from one item moving.
    verdict = paired_comparison([x.outcome_correct for x, _ in shared], [y.outcome_correct for _, y in shared])
    print(
        f"  paired test: {verdict['improvements']} improvements, {verdict['regressions']} regressions, "
        f"{verdict['discordant']} discordant -> p={verdict['p_value']:.3f}"
    )
    print(f"  VERDICT: {verdict['verdict'].upper()}")
    if verdict["discordant"] < verdict["min_discordant_needed"]:
        print(
            f"  (needed {verdict['min_discordant_needed']} same-direction changes to reach "
            f"p<{verdict['alpha']}; this run had {verdict['discordant']} — it could not have concluded)"
        )
    regressions = [(x, y) for x, y in flips if x.outcome_correct and not y.outcome_correct]
    for x, y in regressions[:6]:
        print(f"    {x.metadata['task_id']:<26} {x.action} -> {y.action}  (expected {x.metadata['expected']})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", help="path to a JevBench public .jsonl split")
    p.add_argument("--cap", type=float, default=5.00, help="Jev USD ceiling")
    p.add_argument("--limit", type=int, help="only run the first N tasks")
    p.add_argument("--families", nargs="*", help="restrict to these families")
    p.add_argument(
        "--degrade",
        default="none",
        choices=["none", "strip-criteria", "vague-instructions", "rotate-criteria"],
    )
    p.add_argument("--out", default="benchmarks/jevbench-traces.jsonl")
    p.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"))
    args = p.parse_args()

    load_env()
    if not args.data:
        p.error("--data is required")
    tasks = load_split(args.data, families=args.families)
    if args.limit:
        tasks = tasks[: args.limit]

    if args.compare:
        compare(Path(args.compare[0]), Path(args.compare[1]), tasks)
        return 0

    groups = paraphrase_groups(tasks)
    print(
        f"{len(tasks)} tasks | {len(groups)} paraphrase groups | "
        f"families: {sorted({t.family for t in tasks})} | degradation: {args.degrade}"
    )
    traces = run_split(tasks, cap=args.cap, mode=args.degrade, out=Path(args.out))
    if traces:
        report(traces, tasks, f"JevBench {Path(args.data).stem} through EvalJev [{args.degrade}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
