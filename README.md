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
- **Claude/Gemini example**: Jev routes work to Claude, Gemini, or both; EvalJev records the decision path.

## Install

```bash
pip install -e .
# Optional Claude + Gemini integrations
pip install -e '.[llm,dev]'
```

### Credentials

Three keys, all read from the environment. Copy the template and fill it in:

```bash
cp .env.example .env
set -a; source .env; set +a   # nothing loads .env automatically
```

| Variable | Used by | Required |
| --- | --- | --- |
| `TYPESAFE_API_KEY` | `JevHTTPClient` → `api.typesafe.ai` | Yes |
| `VECTOR_API_KEY` | `call_claude` → Vector Institute proxy | Only for Claude |
| `GEMINI_API_KEY` | `call_gemini` → Google AI Studio | Only for Gemini |

Claude is reached through the Vector Institute's OpenAI-compatible proxy
(`VECTOR_BASE_URL`, default `https://proxy.vectorinstitute.ai/v1`), not the
Anthropic API, so the key is a `vp_`-prefixed proxy key rather than an
`ANTHROPIC_API_KEY`. Or export directly instead of using `.env`:

```bash
export TYPESAFE_API_KEY=...
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
    r = answers["route"]
    return "both" if r.confidence is not None and r.confidence < 0.55 else r.selected

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

## 2. Calibration report

```python
from evaljev import calibration_report

report = calibration_report(monitor.store.list("research-agent"))
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

This is an MVP library skeleton intended for experimentation. The Jev adapter uses the documented direct HTTP contract (`model`, `state`, `questions`) so the monitoring layer is not tightly coupled to one SDK release.
