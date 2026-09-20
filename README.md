# EvalJev

**Runtime assurance for Jev / System-One decision workflows.**

EvalJev instruments typed probabilistic decisions, links them to downstream outcomes, detects calibration and stability failures, replays production traces against candidate changes, and provides the primitives needed for safe auto-repair.

> Jev makes decisions fast. EvalJev helps you know when those decisions — and the workflow around them — are reliable.

## Why this exists

A Jev-style workflow is not just `prompt -> output`. It is:

```text
state -> typed decision -> probability distribution -> policy/threshold
      -> action -> next state -> downstream outcome
```

Failures can come from the model, question schema, state construction, thresholds, policy code, version drift, or interactions between multiple decision nodes. Traditional LLM observability mostly traces requests; EvalJev evaluates the **decision semantics and workflow outcome**.

## MVP features

- **Decision tracing**: state, question schema, full distributions, confidence, selected action, latency, model/question/policy/workflow versions.
- **Calibration**: ECE, Brier score, accuracy/confidence summaries, selective-risk curves.
- **Stability checks**: branch-flip rate and distribution shift across semantically equivalent states.
- **Threshold optimization**: choose a binary operating point from labeled outcomes and asymmetric costs.
- **Decision schema linting**: flags underspecified, subjective, composite, duplicate, and missing-fallback schemas.
- **Decision replay**: rerun historical traces through a new model/question/policy and count changes, improvements, and regressions.
- **Spend ceilings**: a per-API USD cap that refuses the call which would breach it, so an evaluation loop cannot drain an account.
- **Claude/Gemini example**: Jev routes work to Claude, Gemini, or both; EvalJev records the decision path.

## Install

```bash
pip install -e .
# Optional Claude + Gemini integrations
pip install -e '.[llm,dev]'
# Optional: .env loading for examples/router.py
pip install -e '.[examples]'
```

### Credentials

Three keys, all read from the environment. Copy the template and fill it in:

```bash
cp .env.example .env
```

The library never reads `.env` — it only reads `os.environ`. So either source it
yourself, or let the example scripts load it (`pip install -e '.[examples]'`
installs `python-dotenv`; `examples/router.py` loads the repo's `.env` on start):

```bash
set -a; source .env; set +a
```

| Variable | Used by | Required |
| --- | --- | --- |
| `JEV_API_KEY` | `JevHTTPClient` → `jevtypesafeai.com/api/v1/decide` | Yes |
| `VECTOR_API_KEY` | `call_claude` → Vector Institute proxy | Only for Claude |
| `GEMINI_API_KEY` | `call_gemini` → Google AI Studio | Only for Gemini |

Claude is reached through the Vector Institute's OpenAI-compatible proxy
(`VECTOR_BASE_URL`, default `https://proxy.vectorinstitute.ai/v1`), not the
Anthropic API, so the key is a `vp_`-prefixed proxy key rather than an
`ANTHROPIC_API_KEY`. Or export directly instead of using `.env`:

```bash
export JEV_API_KEY=jv_live_...
export VECTOR_API_KEY=vp_...
export GEMINI_API_KEY=...
```

`.env` is gitignored. The test suite needs none of these — it mocks every
client and makes no network calls.

## 1. Monitor a Jev decision

```python
from evaljev import JevHTTPClient, Monitor, JsonlTraceStore

client = JevHTTPClient()
monitor = Monitor(JsonlTraceStore("traces.jsonl"))

questions = {
    "route": {
        "type": "choice",
        "instructions": "Which execution strategy should handle this request?",
        "criteria": {
            "claude": "Use Claude.",
            "gemini": "Use Gemini.",
            "both": "Use both when independent verification is valuable.",
            "other": "No suitable route.",
        },
    }
}

def policy(answers):
    # Threshold on the probability, not on `confidence` — see the note below.
    r = answers["route"]
    p_max = max(r.probabilities.values()) if r.probabilities else 1.0
    return "both" if p_max < 0.64 else r.selected

response, trace = monitor.run(
    client,
    workflow_id="research-agent",
    node_id="model-router",
    state={"request": "Analyze this technical claim"},
    questions=questions,
    policy=policy,
    question_version="route-q3",
    policy_version="route-p2",
    workflow_version="agent-v7",
)

# Later, when downstream ground truth or a reviewer signal arrives:
monitor.record_outcome(trace, outcome={"accepted": True}, correct=True)
```

Trace stores are append-only: `record_outcome` appends the enriched trace rather
than editing the original, so a durable store keeps the full history of each
decision. `TraceStore.list()` returns the newest version of each `trace_id`, so
analytics see one row per decision with its outcome attached.

### `confidence` is a margin, not a probability

Jev reports Choice `confidence` as a normalized margin over a uniform prior,
`c = (p_max - 1/k) / (1 - 1/k)` — verified against live traces to within the
two-decimal reporting granularity. It carries no information beyond `p_max` and the
option count `k`, and it has two sharp consequences:

- **Calibration is scored on `p_max`.** `calibration_report` uses the answer's leading
  probability, falling back to `confidence` only when no distribution is present.
  Scoring a rescaled margin with ECE or Brier measures the wrong quantity.
- **A confidence threshold is not portable.** `confidence < 0.55` means `p_max < 0.775`
  when k=2 but `p_max < 0.595` when k=10. Adding one option to a `criteria` map moves
  every threshold keyed on confidence, with no model or code change. Prefer thresholds
  on `p_max`; `confidence_to_pmax(c, k)` converts an existing one.

## 2. Calibration report

```python
from evaljev import calibration_report

report = calibration_report(monitor.store.list("research-agent"))
# {"route": {"n": ..., "accuracy": ..., "mean_probability": ..., "ece": ..., "brier": ...}}
print(report)
```

## 3. Stability test

```python
from evaljev import stability_check

states = [
    {"request": "I want my money back"},
    {"request": "Please refund the payment"},
    {"request": "Can you return what I paid?"},
]

print(stability_check(
    client,
    states=states,
    questions=questions,
    question_name="route",
))
```

## 4. Replay production traces

```python
from evaljev import replay, summarize_replay

candidate_questions = {
    "route": {
        "type": "choice",
        "instructions": "Which model route minimizes expected failure for this request?",
        "criteria": {
            "claude": "Prefer Claude only.",
            "gemini": "Prefer Gemini only.",
            "both": "Run independent responses from both models.",
            "other": "No route is appropriate.",
        },
    }
}

results = replay(
    monitor.store.list("research-agent"),
    client,
    questions=candidate_questions,
    policy=policy,
)
print(summarize_replay(results))
```

## 5. Cap what an evaluation run can spend

Replaying 10k traces is 10k billed requests. `SpendBudget` sets a hard per-API
ceiling and `BudgetedJevClient` meters every `decide` against it — it duck-types
`JevHTTPClient`, so it drops into `Monitor.run`, `stability_check`, and `replay`
unchanged. Jev returns `usage.cost_usd` per response, so recorded Jev spend is
the exact billed amount rather than an estimate.

```python
from evaljev import BudgetedJevClient, BudgetExceeded, JevHTTPClient, SpendBudget

budget = SpendBudget({"jev": 5.00, "claude": 5.00, "gemini": 5.00})
client = BudgetedJevClient(JevHTTPClient(), budget)

try:
    response, trace = monitor.run(client, ...)
except BudgetExceeded as e:
    print(e)  # jev: budget exhausted — spent $5.0000 of $5.00, ...

print(budget.summary())
print(client.credits_remaining_usd)  # account balance, straight from the API
```

The ceiling is checked *before* each call, so the request that would breach it is
never sent. The library only ever receives USD amounts — provider rate cards stay
in your code. For providers that do not report cost, charge a conservative upper
bound (see `estimate_llm_cost` in `examples/mvp_demo.py`).

## End-to-end demo

`examples/mvp_demo.py` runs every feature above against the live API under a
per-API cap, and writes enriched traces to `traces.jsonl`:

```bash
python examples/mvp_demo.py              # $5 per API
python examples/mvp_demo.py --cap 0.50   # tighter cap
```

It lints the schema, monitors six live routing decisions, executes the chosen
route, records outcomes, then reports calibration, stability, an optimized
threshold, a replay against a sharpened schema, and total spend. Provider
outages are captured as recorded failure outcomes rather than crashing the run.

## Suggested product architecture

```text
Application
   |
   +-- Jev/System-One node -----> EvalJev Monitor
   |                                |
   |                                +-- Trace store
   |                                +-- Calibration
   |                                +-- Stability
   |                                +-- Failure attribution (next)
   |                                +-- Replay
   |                                +-- Repair proposer (next)
   |
   +-- Claude/Gemini/tool action
   |
   +-- outcome/reviewer signal ----> EvalJev
```

## What I would build next

1. OpenTelemetry exporter + Langfuse/LangSmith-compatible trace bridge.
2. SQLite/Postgres trace store with immutable events and outcome joins.
3. Workflow graph representation and node-to-workflow failure attribution.
4. Dataset runner with semantic perturbation generators.
5. Auto-repair proposer for questions/criteria/thresholds, followed by mandatory replay gates.
6. Web dashboard: Traces / Calibration / Instability / Failures / AutoFix.

## Safety of auto-fix

EvalJev should use `propose -> replay -> verify -> approve`, not silently mutate production policy. A proposed repair is only useful if it improves held-out or replayed outcomes without introducing unacceptable regressions.

## Status

This is an MVP library skeleton intended for experimentation. The Jev adapter posts the documented direct HTTP contract (`state`, `questions`) to `JEV_BASE_URL` (default `https://jevtypesafeai.com/api/v1/decide`), so the monitoring layer is not tightly coupled to one SDK release.
