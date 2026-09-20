"""End-to-end EvalJev MVP against the live Jev API, under a hard spend cap.

Exercises every MVP feature in one run:

    lint -> monitor -> execute route -> record outcome -> calibration
         -> stability -> threshold -> replay -> spend report

Every paid call goes through :class:`~evaljev.budget.SpendBudget`, which refuses
a call once an API's ceiling is reached. Jev reports ``usage.cost_usd`` per
response, so its spend is the exact billed amount; the LLM providers are charged
against a conservative upper-bound estimate from the rate table below.

    python examples/mvp_demo.py                 # caps: $5 per API
    python examples/mvp_demo.py --cap 0.50      # tighter cap
    CLAUDE_MODEL=gpt-oss-120b python examples/mvp_demo.py

Needs JEV_API_KEY; VECTOR_API_KEY and GEMINI_API_KEY only for the execution leg,
which degrades to a recorded failure outcome if a provider is unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from evaljev import (
    BudgetedJevClient,
    BudgetExceeded,
    JevHTTPClient,
    JsonlTraceStore,
    Monitor,
    SpendBudget,
    calibration_report,
    lint_questions,
    optimize_binary_threshold,
    replay,
    stability_check,
    summarize_replay,
)
from evaljev.integrations import call_claude, call_gemini

REPO = Path(__file__).resolve().parent.parent
ENV_PATH = REPO / ".env"

# USD per 1M tokens. Upper bounds only — used to refuse a call before it is made,
# never to report spend. Update when a provider changes its rate card.
PRICES = {
    "claude": {"input": 3.00, "output": 15.00},
    "gemini": {"input": 0.30, "output": 2.50},
}
MAX_TOKENS = 400
# p_max below this escalates to both models. Equivalent to the old confidence<0.55
# cutoff at k=4, the option count of the route schema.
ESCALATE_BELOW = 0.663
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def load_env() -> None:
    """Load the repo's .env, pinned to this repo (see examples/router.py)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=False)


# --- The decision under evaluation ------------------------------------------

QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which execution strategy should handle this request?",
        "criteria": {
            "claude": "Long-form reasoning, code, or nuanced writing.",
            "gemini": "Fast factual lookup, summarization, or short extraction.",
            "both": "High-stakes request where independent verification is valuable.",
            "other": "No strategy is suitable.",
        },
    }
}

# A sharpened schema to replay against: same choices, more decisive criteria.
CANDIDATE_QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": (
            "Route this request to the execution strategy that minimizes expected failure. "
            "Prefer a single model unless an error would be costly and hard to detect."
        ),
        "criteria": {
            "claude": "Multi-step reasoning, code generation, or careful writing.",
            "gemini": "Short factual or summarization work where latency matters most.",
            "both": "Safety-critical or irreversible actions needing a second opinion.",
            "other": "The request is not a model-answerable task.",
        },
    }
}

# `expected` is a human routing label, used as the downstream outcome signal.
DATASET = [
    ("Explain why a Python async task can silently swallow an exception.", "claude"),
    ("Summarize this three-sentence changelog into one line.", "gemini"),
    ("Is this medication dosage safe for a six-year-old? Double-check it.", "both"),
    ("Write a recursive descent parser for a small arithmetic grammar.", "claude"),
    ("What UTC offset does New York use in January?", "gemini"),
    ("Decide whether to deploy this database migration to production tonight.", "both"),
]

PARAPHRASES = [
    "I want my money back.",
    "Please refund the payment.",
    "Can you return what I paid?",
    "I'd like a refund for this order.",
]


def policy(answers):
    """An uncertain route escalates to running both models.

    Thresholds on p_max, not on `confidence`. The API's confidence is a margin over
    a uniform prior, so the same confidence cutoff is a different probability for
    every option count — and moves the moment a criteria map gains an option.
    """
    ans = answers["route"]
    p_max = max(ans.probabilities.values()) if ans.probabilities else 1.0
    return "both" if p_max < ESCALATE_BELOW else ans.selected


# --- Budgeted execution leg --------------------------------------------------


def estimate_llm_cost(provider: str, prompt: str) -> float:
    """Upper bound: assume the full max_tokens budget is emitted."""
    rate = PRICES[provider]
    input_tokens = len(prompt) / 3  # deliberately pessimistic chars-per-token
    return (input_tokens * rate["input"] + MAX_TOKENS * rate["output"]) / 1_000_000


def run_provider(provider: str, prompt: str, budget: SpendBudget) -> dict:
    """Call one provider under the budget, converting failure into a recorded outcome."""
    estimate = estimate_llm_cost(provider, prompt)
    budget.check(provider, estimate)
    try:
        if provider == "claude":
            text = call_claude(prompt, model=CLAUDE_MODEL, max_tokens=MAX_TOKENS)
        else:
            text = call_gemini(prompt, model=GEMINI_MODEL)
    except Exception as e:  # noqa: BLE001 - provider outage is a recordable outcome
        budget.record(provider, estimate)  # assume worst case when usage is unknown
        return {"provider": provider, "ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
    budget.record(provider, estimate)
    return {"provider": provider, "ok": True, "chars": len(text), "preview": text[:80]}


def execute_route(action: str | None, prompt: str, budget: SpendBudget) -> dict:
    """Like integrations.execute_model_route, but metered and failure-tolerant."""
    if action in ("claude", "gemini"):
        return run_provider(action, prompt, budget)
    if action == "both":
        results = [run_provider(p, prompt, budget) for p in ("claude", "gemini")]
        failed = [f"{r['provider']}: {r.get('error', '')}" for r in results if not r["ok"]]
        return {
            "provider": "both",
            "ok": not failed,
            "results": results,
            **({"error": "; ".join(failed)} if failed else {}),
        }
    return {"provider": action, "ok": False, "error": "no executable route"}


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=float, default=5.00, help="per-API USD ceiling")
    parser.add_argument("--execute", type=int, default=3, help="traces to actually execute")
    parser.add_argument("--out", default=str(REPO / "traces.jsonl"))
    args = parser.parse_args()

    load_env()
    if not (os.getenv("JEV_API_KEY") or os.getenv("TYPESAFE_API_KEY")):
        print("JEV_API_KEY is not set — see .env.example")
        return 1

    budget = SpendBudget({"jev": args.cap, "claude": args.cap, "gemini": args.cap})
    client = BudgetedJevClient(JevHTTPClient(), budget)
    # Start from an empty log so the reported numbers describe this run only.
    out_path = Path(args.out)
    out_path.unlink(missing_ok=True)
    store = JsonlTraceStore(out_path)
    monitor = Monitor(store)
    print(f"per-API cap: ${args.cap:.2f}   jev endpoint: {client.client.base_url}")

    # 1. Lint the schema before spending anything on it.
    section("1. Decision schema lint (offline, $0)")
    issues = lint_questions(QUESTIONS)
    print(f"{len(issues)} issue(s)")
    for issue in issues:
        print(f"  [{issue.severity}] {issue.question}: {issue.code} — {issue.message}")

    # 2. Monitor live decisions.
    section(f"2. Monitor {len(DATASET)} live Jev decisions")
    traces = []
    for prompt, expected in DATASET:
        try:
            _, trace = monitor.run(
                client,
                workflow_id="mvp-demo",
                node_id="route-model",
                state={"user_request": prompt, "available_models": ["claude", "gemini"]},
                questions=QUESTIONS,
                policy=policy,
                question_version="route-q1",
                policy_version=f"escalate-below-pmax-{ESCALATE_BELOW}",
                workflow_version="mvp-1",
                metadata={"expected_route": expected},
            )
        except BudgetExceeded as e:
            print(f"  stopped: {e}")
            break
        ans = trace.answers[0]
        traces.append(trace)
        print(
            f"  {prompt[:46]:<46} choice={ans.selected:<7} p_max={max(ans.probabilities.values()) if ans.probabilities else 1.0:<5} "
            f"action={trace.action:<7} expected={expected:<7} {trace.latency_ms:.0f}ms"
        )

    # 3. Execute routes and record downstream outcomes.
    section(f"3. Execute {min(args.execute, len(traces))} routes and record outcomes")
    print(f"claude->{CLAUDE_MODEL}  gemini->{GEMINI_MODEL}  max_tokens={MAX_TOKENS}")
    for trace in traces[: args.execute]:
        prompt = trace.state["user_request"]
        try:
            result = execute_route(trace.action, prompt, budget)
        except BudgetExceeded as e:
            print(f"  stopped: {e}")
            break
        trace.metadata["execution"] = result
        monitor.record_outcome(
            trace,
            outcome=result,
            correct=(trace.action == trace.metadata["expected_route"]),
        )
        status = "ok" if result["ok"] else f"FAILED ({result.get('error', '')[:60]})"
        print(f"  {trace.action:<7} {status}")

    # Unexecuted traces still get the routing label — the router is what we score.
    for trace in traces[args.execute :]:
        monitor.record_outcome(
            trace,
            outcome={"executed": False},
            correct=(trace.action == trace.metadata["expected_route"]),
        )

    # 4. Calibration, read back from the durable store rather than from memory.
    section("4. Calibration report (p_max vs. correct routing)")
    persisted = JsonlTraceStore(out_path).list("mvp-demo")
    print(f"read {len(persisted)} traces back from {out_path.name}")
    print(json.dumps(calibration_report(persisted), indent=2))

    # 5. Stability across paraphrases.
    section(f"5. Stability check over {len(PARAPHRASES)} paraphrases")
    try:
        stability = stability_check(
            client,
            states=[{"user_request": p, "available_models": ["claude", "gemini"]} for p in PARAPHRASES],
            questions=QUESTIONS,
            question_name="route",
        )
        print(json.dumps(stability, indent=2))
    except BudgetExceeded as e:
        print(f"  stopped: {e}")

    # 6. Threshold optimization on the escalation decision.
    section("6. Threshold optimization (when to escalate to 'both')")
    confidences = [
        max(t.answers[0].probabilities.values()) if t.answers[0].probabilities else 0.0
        for t in traces
    ]
    labels = [int(bool(t.outcome_correct)) for t in traces]
    if confidences:
        best = optimize_binary_threshold(
            confidences, labels, false_positive_cost=1.0, false_negative_cost=4.0
        )
        print(f"  p_max      ={[round(c, 3) for c in confidences]}")
        print(f"  correct    ={labels}")
        print(f"  best (FN 4x costlier than FP): {best}")
        if len(set(labels)) < 2:
            print(
                "  NOTE: every routing decision was correct, so there is no operating point "
                "to learn — the optimizer degenerates to accept-everything. Needs failures."
            )

    # 7. Replay the same traces against the candidate schema.
    section("7. Replay stored traces against a sharpened schema")
    try:
        results = replay(
            traces,
            client,
            questions=CANDIDATE_QUESTIONS,
            policy=policy,
            judge=lambda trace, action: action == trace.metadata["expected_route"],
        )
        for r in results:
            flag = "CHANGED" if r.changed else "same   "
            print(f"  {flag} {r.old_action:<7} -> {r.new_action:<7} correct: {r.old_correct} -> {r.new_correct}")
        print(json.dumps(summarize_replay(results), indent=2))
    except BudgetExceeded as e:
        print(f"  stopped: {e}")

    # 8. Report spend. Traces were persisted as they were produced.
    section("8. Spend report")
    print("jev = exact billed cost from usage.cost_usd; llm = upper-bound estimate")
    print(json.dumps(budget.summary(), indent=2))
    if client.credits_remaining_usd is not None:
        print(f"jev credits remaining: ${client.credits_remaining_usd:.4f}")
    events = len(out_path.read_text().splitlines()) if out_path.exists() else 0
    print(f"{events} events -> {len(traces)} enriched traces in {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
