# EvalJev

**Runtime assurance for Jev / System-One decision workflows.**

EvalJev instruments typed probabilistic decisions, links them to downstream outcomes, detects calibration and stability failures, replays production traces against candidate changes, and provides the primitives needed for safe auto-repair.

> Jev makes decisions fast. EvalJev helps you know when those decisions — and the workflow around them — are reliable.

**Live dashboard: <https://xxlya.github.io/evaljev/>** — a real 156-decision run, rendered
by the command below. Point it at your own traces and you get the same page:

```bash
pip install "git+https://github.com/xxlya/evaljev"
evaljev report traces.jsonl -o report.html
```

## Why this exists

A Jev-style workflow is not just `prompt -> output`. It is:

```text
state -> typed decision -> probability distribution -> policy/threshold
      -> action -> next state -> downstream outcome
```

Failures can come from the model, question schema, state construction, thresholds, policy code, version drift, or interactions between multiple decision nodes. Traditional LLM observability mostly traces requests; EvalJev evaluates the **decision semantics and workflow outcome**.

## MVP features

- **Decision tracing**: state, question schema, full distributions, confidence, selected action, latency, model/question/policy/workflow versions.
- **Calibration**: ECE, Brier score, accuracy/probability summaries, selective-risk curves.
- **Stability checks**: branch-flip rate and distribution shift across semantically equivalent states.
- **Threshold optimization**: choose a binary operating point from labeled outcomes and asymmetric costs.
- **Decision schema linting**: flags underspecified, subjective, composite, duplicate, and missing-fallback schemas.
- **Decision replay**: rerun historical traces through a new model/question/policy, with a paired test that says whether the difference is real.
- **Drift detection**: window a trace stream, test the newest window against the pooled baseline, and refuse to compare windows that are not composition-comparable.
- **Failure attribution**: diff two windows of traffic and name the component that changed — schema, option set, policy, model, or inputs.
- **Repair proposal**: generate candidate schemas from observed failures, then gate them on a held-out replay against a contemporaneous control. Proposes; never applies.
- **Spend ceilings**: a per-API USD cap that refuses the call which would breach it, so an evaluation loop cannot drain an account.
- **Monitoring page**: one self-contained HTML file — health checks in plain language, certainty, drift, what changed, recent decisions — built from a trace file by `evaljev report`, with no service, no key and no network call.
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
    repeats=3,   # optional: measure the API's own jitter as a noise floor
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
# {"changed": 7, "improvements": 7, "regressions": 0, "discordant": 7,
#  "p_value": 0.016, "verdict": "improvement", "min_discordant_needed": 6, ...}
```

### Rates carry intervals, comparisons carry verdicts

A rate computed from 36 samples is an interval, not a number: 1 flip in 36 pairs is
`0.028` with a 95% interval of `[0.005, 0.142]`. Reporting the point estimate alone
invites deployment decisions on noise, so `stability_check` returns
`branch_flip_ci` and `summarize_replay` returns `change_rate_ci`.

Replay is **paired** — the candidate reruns the same items — so `summarize_replay`
compares with an exact McNemar test on the decisions that changed, not with an
accuracy delta. Its `verdict` is `improvement`, `regression`, or
`insufficient evidence`, and it is never "no difference": a replay with too few
changed decisions has not shown equivalence, it has shown nothing.
`min_discordant_needed` is the floor below which the comparison could not have
concluded at all (6 at the default alpha) — check it before paying for a replay.

### Two tests, because they ask different questions

McNemar can only use items whose decision flipped. That throws away everything
else, and on a small schema there may not be six flips to have. Pass `label_of=`
to `replay` and `summarize_replay` adds `probability_shift`: an exact Wilcoxon
signed-rank test on how much probability mass moved onto the correct label, per
item. It uses every item, including the ones that were already right.

The difference is not academic. On a live repair of a deliberately broken schema,
with the same six held-out decisions and the same candidate:

```
DECISION    +3 -0   discordant=3   p=0.2500   insufficient evidence
PROBABILITY 6/6 improved, median delta +0.600, p=0.0312   improvement
```

Only three of the six were wrong to begin with, so only three *could* flip — the
decision test was structurally unable to conclude, while the probability test saw
a consistent 0.6 shift. Mass can also move a long way without any decision
crossing a threshold, which is real progress a flip-counting test cannot see.

They are not interchangeable. McNemar asks whether **behaviour** changed;
signed-rank asks whether the **probability** improved, which is a leading
indicator. `repair_cycle(gate=...)` takes `"decision"` (default, strict),
`"probability"` (sensitive; confirm on more traffic before applying), or
`"both"`.

Both need six items at the default alpha, and the difference between *which* six
is the entire power gain: `min_discordant_for_significance()` counts decisions
that **flipped**, `min_samples_for_signed_rank()` counts probabilities that
**moved**. Check them before paying for a replay that cannot conclude.

## 5. Measure a candidate against a contemporaneous control

`evaluate_candidate_paired` puts the baseline and the candidate in **one request**
against the same state, instead of comparing against a recording that may be days
old. Drift and per-call jitter cancel, because both arms are measured in the same
call.

This is safe because questions attend to the shared state, not to each other —
verified on the live API before relying on it. A question's distribution moves no
more when a second question shares its request (mean JS **0.0058**) than between
two identical requests (**0.0035**), and the largest single shift, 0.0353, was
identical in both arms.

It is also cheaper, since the state is encoded once: measured over 10 states,
one two-question request versus two one-question requests cost **34.7% less**
(34.6% fewer input tokens) and returned **49.4% faster**.

```python
repair_cycle(traces, client, question_name="route", generate=..., control="paired")
```

`stability_check(repeats=n)` additionally queries each state `n` times to measure a
**noise floor**: how often the branch moves when the input does not. The API is not
deterministic, so paraphrase instability only means something above that floor, which
is what `excess_flip_rate` and `exceeds_noise` report.

## 6. Cap what an evaluation run can spend

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

## 7. Watch for drift over time

`attribute` compares two windows you already chose, having already noticed
something was wrong. Monitoring is the other way round: the unit is a delta
against a baseline, and the first job is **noticing**.

```python
from datetime import timedelta
from evaljev import drift_report

report = drift_report(store.list("research-agent"), window=timedelta(hours=1))
print(report["signals"])
print(report["next_step"])
for row in report["series"]:
    print(row["index"], row["n"], row["action_mix"], row["accuracy"])
```

Windows are cut by duration (`window=`) or by trace count (`size=`), and the
newest is compared against every earlier one pooled. Windows are **not paired** —
they are different traffic — so comparisons use exact tests for independent
samples (Fisher), not the paired tests replay uses. The `series` is returned so a
trend is visible even when no single step trips a threshold.

Two guards exist because live traces showed the report is misleading without them:

- **`comparable`** is checked first. If the windows carry different mixes of
  decision nodes, they are not samples of the same thing, and nothing else can be
  read as deployment drift. Traffic ordered by type — a benchmark sweep, a nightly
  batch — breaks this silently. On the real JevBench stream this fires
  (`composition_shift 0.678`) and the report says *compare like with like* instead
  of pretending to have found something.
- **Action shares are tested, not thresholded.** A JS distance on a ten-trace
  window is large whatever the deployment is doing. `action_mix_shift` is reported
  as a description; the signal that fires is `moved_actions`, where each action's
  share gets its own exact test. On one real window that distinction was the
  difference between a reported `0.553 shift` and the correct answer of *no
  established change*.

## 8. Attribute a change to a component

A deployed decision has five moving parts: state construction, the question schema,
the model, the policy/threshold, and the downstream action. When behaviour moves,
the only question that matters is which one moved.

Four of the five are already recorded on every trace, so this is a diff, not an
inference — and it costs nothing:

```python
from evaljev import attribute, vendor_drift_check

report = attribute(last_week, this_week, confidence_threshold=0.55)
for f in report["findings"]:
    print(f["component"], f["summary"])
# option_set  question 'route': option set changed, k 3 -> 4;
#             any confidence<0.55 threshold now fires at p_max 0.700 -> 0.663
```

Attribution is deliberately deterministic. Its output is evidence for changing
production configuration, so it has to be reproducible and readable — "the criteria
map gained an option, here is the diff" is auditable in a way that a model's opinion
is not. `option_set` gets its own finding because option count is the one edit that
silently moves every `confidence` threshold while leaving the model, the policy code
and the accuracy untouched.

When behaviour moved and **nothing recorded accounts for it**, the report sets
`residual: true`. That is not a conclusion, it is an instruction — one replay
separates the two remaining explanations:

```python
vendor_drift_check(sample_of_old_traces, client)
# {"mean_distribution_shift": 0.007, "branch_flip_rate": 0.0, "vendor_drift": False,
#  "verdict": "model reproduces its recorded answers; look at the input distribution"}
```

Both arms use the recorded state and the recorded schema, so the only variable is
*when* the call was made. Distributions moved → the model changed underneath you.
Distributions held → your traffic changed instead.

`state_diff` is deliberately shallow: it catches a state builder that started or
stopped emitting a field, and gross size changes. It does **not** detect semantic
drift in the traffic — same keys, same size, different subject matter — which needs
an embedding of the state. A clean state report is not evidence that inputs are
unchanged.

## 9. Propose a repair, and refuse to believe it without evidence

`propose -> validate -> screen -> verify -> approve`. Only the first step uses a
language model; everything after it is deterministic, because the output is a change
to production configuration.

```python
from evaljev import repair_cycle
from evaljev.integrations import call_gemini

report = repair_cycle(
    traces, client,
    question_name="route",
    generate=lambda prompt: call_gemini(prompt, model="gemini-2.5-pro"),
    judge=lambda trace, action: action == trace.metadata["expected"],
)
print(report["next_step"])
for row in report["approved"]:
    print(row["candidate"]["rationale"], row["verify"]["p_value"])
```

Three properties are structural rather than configurable:

- **Nothing is ever applied.** The cycle returns ranked proposals with their evidence.
- **The proposer never sees the holdout.** Traces split deterministically by hashed
  `trace_id`, so a proposal cannot be laundered by re-rolling the split until the
  holdout agrees. Candidates are written from dev failures and judged on traces the
  generator could not have fitted to.
- **A candidate may not change the label set.** Adding or removing an option changes
  `k`, which silently moves every threshold keyed on `confidence`. A repair that
  quietly re-tunes every downstream cutoff is not a repair, so those candidates are
  rejected before they cost anything.

Candidates are filtered cheapest-first: free validation (label set, type, lint),
then a dev-set screen, then the holdout replay with the paired test from section 4, against the contemporaneous control from section 5.

### Does it work?

A controlled experiment, since the fault is known in advance: take the JevBench
`intent` schema (k=5, one schema shared by 12 decisions) and rotate each label's
description onto its neighbour — same keys, same `k`, same instructions, only the
rubric now points at the wrong option.

```
broken schema accuracy       3/12
dev 5 / holdout 7, dev failures 2

candidate 1  screen (dev n=5): +2 -0
             verify (holdout n=7): +7 -0  discordant=7  p=0.0156  -> improvement
```

All three proposals recovered it from two dev failures, and the rationale named the
mechanism: *"the original descriptions were incorrectly mapped to their keys; this
version realigns each description with its semantically correct action key."* The
recovered criteria are semantically correct but reworded rather than identical to the
original — it re-derived the rule instead of reproducing an answer key, which is what
the prompt asks for.

Cost: $0.008 of Jev across 48 replays, plus one generator call. Note that 7 discordant
pairs is barely past the minimum of 6 — a 12-decision schema is near the smallest
sample on which the *decision* gate can conclude anything at all. A rerun that split
6/6 instead of 5/7 had only 3 flippable items in the holdout and could not conclude,
while the probability test on the same six items returned p=0.0312. That asymmetry is
why both are reported.

## The monitoring page

Everything above is a Python API. The page is those same numbers, rendered for someone
who would rather not call them — an engineer on call, a PM, whoever owns the workflow.

```bash
pip install "git+https://github.com/xxlya/evaljev"   # not on PyPI yet

evaljev report traces.jsonl -o report.html   # one self-contained file; open it anywhere
evaljev serve traces.jsonl                   # same page, re-reading the file as traffic arrives
evaljev demo                                 # the page on the recorded run bundled with the package
```

Live example: **<https://xxlya.github.io/evaljev/>** — 156 real decisions, generated by
the same code path `evaljev report` runs for you.

It answers eight questions, each with the number behind it and the next step:

| Check | The question, as asked on the page |
| --- | --- |
| Valid answers | Did every answer use your labels, with probabilities adding up to 1? |
| Certainty | How many decisions landed below your review line? |
| Drift | Has anything started answering differently than it used to? |
| Consistency | When the identical input came in twice, did it decide the same way? |
| Calibration | When it says it is 80% sure, is it right about 80% of the time? |
| Outcomes | Do you know how these decisions actually turned out? |
| Wording | Is anything in the schema vague, subjective or double-barrelled? |
| Configuration | Did the model, wording, options or policy change mid-stream? |

Plus certainty and traffic over time, the branch mix, a per-decision-point table, the
exact diff of what changed, and the recent decisions with their full probability spread.

Four properties are deliberate, because they are what make the page worth trusting:

- **"Not measured" never renders as "healthy".** A check with too little evidence says
  so and is left out of the score — [too little evidence means nothing was shown](#rates-carry-intervals-comparisons-carry-verdicts).
- **Every rate carries its interval.** 1 flip in 36 appears as 2.8% *with* its 0.5–14.2% range.
- **Nothing on the page is written by a model.** Every number is computed by the library
  from the traces; the prose is fixed text chosen by the verdict.
- **Nothing leaves your machine.** One HTML file, no CDN, no telemetry, no account.

Useful flags (`report`, `serve` and `demo` all accept them):

| Flag | Default | What it does |
| --- | --- | --- |
| `--unsure-below P` | `0.6` | probability under which a decision counts as one a human should see. A probability, never `confidence` — see [below](#confidence-is-a-margin-not-a-probability) |
| `--workflow ID` | all | restrict to one `workflow_id` |
| `--window N` | a quarter of each node's traffic | decisions per drift-comparison window |
| `--recent N` | `60` | how many decisions to list |
| `--json FILE` | — | also write the report as JSON, for a CI check or your own renderer |
| `--open` | off | open the page in a browser |

From Python, for a custom renderer or an assertion in CI:

```python
from evaljev import JsonlTraceStore, build_report, render_html

report = build_report(JsonlTraceStore("traces.jsonl").list(), unsure_below=0.6)
assert report["headline"]["status"] != "problem", report["headline"]["summary"]
Path("report.html").write_text(render_html(report))
```

## Published site

`docs/` is a dependency-free static site — GitHub Pages serves it directly. Enable it
under **Settings → Pages → Source: `main` / `/docs`**; there is no build step and no
workflow to configure. It is two pages: the dashboard (`index.html`) and the evidence
behind it (`how-it-works.html`).

```bash
python benchmarks/export_demo_data.py   # regenerates both pages from the stored traces
python -m http.server -d docs 8765      # preview at http://localhost:8765
```

Every number on both pages is computed by the library from `benchmarks/*.jsonl`, so the
site cannot drift from the run that produced it — rerun the benchmark and the export, and
the pages follow. Charts are hand-built SVG with no dependencies; the categorical palette
is validated for colour-vision deficiency against both the light and dark surfaces, and
every chart carries direct labels or a table view so identity is never colour-alone.

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
2. SQLite/Postgres trace store with immutable events and outcome joins, so drift windows do not have to fit in memory.
3. Workflow graph representation, so attribution can cross node boundaries instead of stopping at one decision.
4. Dataset runner with semantic perturbation generators.
5. Threshold repair to sit alongside the schema repair, and a multi-round loop that re-proposes from what the last gate rejected.
6. A hosted dashboard with history and shared access — the self-contained page ships today (`evaljev report`), but it is a snapshot of a file, not a service with retention.

## Safety of auto-fix

EvalJev should use `propose -> replay -> verify -> approve`, not silently mutate production policy. A proposed repair is only useful if it improves held-out or replayed outcomes without introducing unacceptable regressions.

## Status

This is an MVP library skeleton intended for experimentation. The Jev adapter posts the documented direct HTTP contract (`state`, `questions`) to `JEV_BASE_URL` (default `https://jevtypesafeai.com/api/v1/decide`), so the monitoring layer is not tightly coupled to one SDK release.
