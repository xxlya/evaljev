# EvalJev

**Know whether your Jev agent is still deciding well — and which change broke it.**

A Jev workflow makes typed decisions: state in, a probability over your options out, your
code picks a branch. EvalJev records those decisions as they happen and audits them, so you
can answer the two questions a deployed agent actually raises. *Which decisions should not
have been made without a person?* And *the change I shipped on Tuesday — did it help?*

**Live example: <https://xxlya.github.io/evaljev/>** — a three-step support assistant, 351
real decisions, four shipped versions of one question, and the audit that says which of
them broke it.

## Start here

Three steps, about five minutes. Nothing leaves your machine and there is no service to
sign up for.

### 1. Wrap the client you already have

```python
from evaljev import watch

client = watch(client, workflow_id="support-assistant")   # your JevHTTPClient, wrapped
```

Your call sites do not change. `client.decide(state=..., questions=...)` returns exactly
what it returned before, and every decision is appended to `traces.jsonl` as it happens.

Optional, and worth it — name the steps and group a request, so the audit can show the
path one request took through your workflow:

```python
with client.request(request_id):
    with client.step("classify_request", question_version="category-v4"):
        response, _ = client.decide(state=state, questions=questions)
    with client.step("route_handoff"):
        response, _ = client.decide(state=state, questions=handoff)
```

And when you learn what actually happened — a human corrected it, the refund went through:

```python
client.record_outcome({"reviewer": "agreed"}, correct=True)
```

### 2. Ship as usual

Change a question's wording, add an option, swap the model, edit the policy. Each
configuration you run on becomes a **run** the audit can judge. You do not need to tell it
that you shipped; it reads that off the traces.

### 3. Audit

```bash
evaljev console traces.jsonl --open    # the screen
evaljev serve traces.jsonl             # the same screen, refreshing as traffic arrives
evaljev demo                           # it, on a recorded run, before you have traces
```

You get, per run: what changed, what it did to real requests, the same input answered
before and after, a verdict — and the decisions the workflow made alone that it should
not have, each with the path it took.

```
RUN 4 · category v4 · 24 requests · measured against the 93 requests before it

  what changed   the instructions for "category" were rewritten
                 6 option descriptions reworded — the labels themselves are unchanged
  what it did    answers needing a second look: 0% → 20.8%     (p<0.001)
                 typical certainty: 1.00 → 0.84                (p<0.001)
                 accuracy on reviewed decisions: 100% → 62.5%  (p=0.006)
  evidence       13 of 24 inputs that arrived in both runs were answered differently
  verdict        HARMFUL — roll it back, or fix what it changed
```

## What it needs from you, and what it does not

| Needs | Does not need |
| --- | --- |
| One wrapped client | Labels or ground truth. Four of the five flags work without them |
| A trace file | A service, an account, a key, or a network call at audit time |
| Your review line, if 0.6 is not yours | Your prompts or your customers' data leaving the machine |
| Outcomes, *if* you want accuracy and calibration | Anything at all for drift, certainty, schema and consistency |

## Why this exists

A Jev-style workflow is not just `prompt -> output`. It is:

```text
state -> typed decision -> probability distribution -> policy/threshold
      -> action -> next state -> downstream outcome
```

Failures can come from the model, question schema, state construction, thresholds, policy code, version drift, or interactions between multiple decision nodes. Traditional LLM observability mostly traces requests; EvalJev evaluates the **decision semantics and workflow outcome**.

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

## The console

The console audits **runs**. A run is a stretch of traffic your workflow answered with one
configuration — the unit you actually ship and roll back. Every time a question's wording,
its options, the policy or the model changes, that is a new run, and it gets measured
against the last stretch that held still.

It is one file, no service, no key, no network call.

```bash
pip install "git+https://github.com/xxlya/evaljev"   # not on PyPI yet

evaljev console traces.jsonl --open   # the triage screen
evaljev serve traces.jsonl            # the same screen, re-reading the file as traffic arrives
evaljev demo                          # the screen, on the recorded example run
```

Live example: **<https://xxlya.github.io/evaljev/>**

Each run gets four lines and a verdict:

```
RUN 4 · category v4 · 24 requests · measured against the 93 requests before it

  what changed   the instructions for "category" were rewritten
                 6 option descriptions reworded — the labels themselves are unchanged
  what it did    answers needing a second look: 0% → 16.7%   (p=0.001)
                 typical certainty: 1.00 → 0.84              (p<0.001)
                 accuracy on reviewed decisions: 100% → 62.5% (p=0.006)
  evidence       13 of 24 inputs that arrived in both runs were answered differently
  verdict        HARMFUL — roll it back, or fix what it changed
```

Verdicts are `harmful`, `improvement`, `no effect shown`, `too early to tell`, and
`baseline` for the first one. **"No effect shown" is never rendered as "safe"** — it says
how much traffic it had and that a change would have to be large to show up at that size.

The baseline is not the previous run but **the last stretch that held still**: every run
since the last one that moved something, pooled. Two dozen requests against two dozen
cannot establish a rate change that ninety against two dozen can, and runs that changed
nothing are part of the same regime.

Inside a run, the requests it **answered alone and should not have** are listed with the
path each one took. Requests the workflow escalated itself are counted but not queued: a
flag only asks something of you when it contradicts what the workflow did.

Which branches mean "a person is involved" is matched by name (`human`, `needs_review`,
`escalate`, …) and the page says the match was inferred. `--human-actions refund,escalate`
sets it yourself.

A decision is flagged when EvalJev can say something concrete about it:

| Flag | What it means |
| --- | --- |
| answer broke the schema | the distribution was not over the options you declared — code downstream acted on garbage, whoever was holding the request |
| a reviewer disagreed | the recorded outcome contradicts what was decided |
| same input decided differently | an identical input took another branch elsewhere in the stream |
| model was unsure | the top answer fell under your `--unsure-below` review line |
| decided after this step changed | the decision point started answering differently in this window |

Below that, your workflow as a line of steps — each with its certainty *before → after*
rather than a bare level — and then the queue. Pick one and you get the path it took:
every decision, the probability behind it, and the step that raised the flag.

Give it `metadata={"request_id": ...}` at every `Monitor.run` in a request and it triages
whole requests with their path. Without one it triages each decision on its own and says
so — a missing path is not a missing console.

## What is behind it

The console shows one screen. Everything else `build_report()` computes — eight health
checks, calibration, drift windows, the attribution diff, per-decision rows — is written
by `--json` and is the interface for anything else you want to do with it, including a CI
assertion:

```python
from evaljev import JsonlTraceStore, build_report

report = build_report(JsonlTraceStore("traces.jsonl").list(), unsure_below=0.6)
assert report["queue"]["counts"]["acted_alone"] == 0, report["queue"]["headline"]
```

Every metric had to answer "what decision does this change, and can someone who did not
build it read it?". The ones that could not are absent from the page and remain in that
JSON: expected calibration error as a number, Jensen-Shannon distances, composition shift,
the names of the statistical tests, and a 0–100 health score that was an invented formula.

Four properties are deliberate, because they are what make the screen worth trusting:

- **"Not measured" never renders as "healthy".** A check with too little evidence says
  so — [too little evidence means nothing was shown](#rates-carry-intervals-comparisons-carry-verdicts).
- **Every rate carries its interval.** 1 flip in 36 appears as 2.8% *with* its 0.5–14.2% range.
- **Nothing on the page is written by a model.** Every number is computed by the library
  from the traces; the prose is fixed text chosen by the verdict.
- **Nothing leaves your machine.** One HTML file, no CDN, no telemetry, no account.

Useful flags:

| Flag | Default | What it does |
| --- | --- | --- |
| `--unsure-below P` | `0.6` | the review line: probability under which a decision needs a person. A probability, never `confidence` — see [below](#confidence-is-a-margin-not-a-probability) |
| `--human-actions A,B` | matched by name | the branches that mean a person is now involved |
| `--workflow ID` | all | restrict to one `workflow_id` |
| `--window N` | a quarter of each node's traffic | decisions per drift-comparison window |
| `--json FILE` | — | write the whole analysis as JSON |
| `--open` | off | open the page in a browser |

## The example workflow

`examples/support_assistant.py` is a three-node support assistant — classify, rate
urgency, decide whether a person is needed — instrumented end to end. It exists to be
copied: swap the questions and the policy for yours and the dashboard works the same way.

It also records the demo, in four phases, which is where the published page comes from:

```bash
for phase in v1 v2 v3 v4; do
  python examples/support_assistant.py --phase $phase --out examples/support-$phase.jsonl
done
evaljev console examples/support-v*.jsonl -o console.html
```

The phases are two harmless edits and one that is not, and knowing which is which in
advance is the point:

| Phase | The edit | What the monitor said |
| --- | --- | --- |
| `v1` | — | baseline, 45 requests |
| `v2` | a `billing` option is added for a new product line | config changed; no behaviour change established |
| `v3` | the question is "tidied up" — shorter instructions, shorter descriptions | still nothing established |
| `v4` | the labels get sorted in one file while their descriptions stay in insertion order, so every description lands on the wrong label | caught: `needs_review` 0% → 14%, typical certainty 1.00 → 0.86, 13 of 24 repeated inputs answered differently |

`v4` is a `zip()` of two lists that fell out of step. Same six labels, same `k`, same
instructions — a schema validator sees nothing, every answer stays inside the option set,
and accuracy on reviewed decisions falls from 100% to 70%.

## Published site

`docs/` is a dependency-free static site — GitHub Pages serves it directly, from
**Settings → Pages → Source: `main` / `/docs`**; there is no build step and no workflow
to configure. Two pages: the dashboard for the example workflow (`index.html`) and the
evidence behind the library (`how-it-works.html`).

```bash
python benchmarks/export_demo_data.py   # regenerates the pages from the recorded traces
python -m http.server -d docs 8765      # preview at http://localhost:8765
```

Every number on both pages is computed by the library from the recorded traces, so the
site cannot drift from the run that produced it — rerun the benchmark and the export, and
the pages follow. Charts are hand-built SVG with no dependencies; the categorical palette
is validated for colour-vision deficiency against both the light and dark surfaces, and
every chart carries direct labels or a table view so identity is never colour-alone.

## Auditing a benchmark board

A ranked board is a list of sample statistics printed as exact numbers.
[JevBench](https://github.com/fstandhartinger/jevbench) ranks 52 systems by a score over
Intelligence, Calibration, Speed and Cost — and Intelligence is an accuracy over a finite
set of items, so two systems half a point apart may be one system as far as the evidence
goes. It publishes per-item outcomes for 231 public items, which makes the **paired** test
possible: every system saw the same items, so the comparison is exact McNemar on the items
they disagreed about, not two accuracies quoted side by side.

```bash
git clone https://github.com/fstandhartinger/jevbench /tmp/jevbench
python benchmarks/audit_jevbench.py --repo /tmp/jevbench --top 12
```

Three things it found in the v1.3.0 artifacts, none of which dispute a measurement:

**Half the ordering is not established.** Of the 11 adjacent pairs in the top 12, 6 are
separated on accuracy and 5 are not — including ranks 2, 3 and 4, which sit within 2.0
points of composite score and are statistically one group.

**Two pairs are ranked the wrong way round on accuracy.**

```
 7 vs 8   decision-machine-1   decider-35b-a3b   #7 wins 9   #8 wins 45  p=0.000
          #8 beats #7 on accuracy, and ranks below it: speed 81 vs 93, cost 45 vs 54
10 vs 11  system-one-open      OpenJev 26B       #10 wins 18 #11 wins 38 p=0.010
          #11 beats #10 on accuracy, and ranks below it: cost 45 vs 65
```

That is the composite working as designed — Speed and Cost are half the score — but a
reader takes a ranking for an ordering of ability, so it is worth saying out loud.

**A column the board does not have.** The standard tier is 36 decisions each written two
ways with the same right answer. Counting the pairs where a system got one right and the
other wrong costs nothing extra and is already in their data: 1 of 36 for the top four, 6
of 36 for rank 9. That gap appears in none of the four axes — and unlike every one of them,
it needs no ground truth, so it is the one measurement here that also works on live traffic.

## Reference

The sections below are the library underneath the console: every primitive, with the
reasoning behind it. Start above; come here when you want to build something else on it.

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
- **Decision console**: one self-contained HTML file — which decisions your workflow made alone that it should not have, why, and the path through your workflow where each was flagged. Built from a trace file by `evaljev console`, with no service, no key and no network call; `--json` writes the whole analysis behind it.
- **Claude/Gemini example**: Jev routes work to Claude, Gemini, or both; EvalJev records the decision path.

### 1. Record a decision by hand

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

### 2. Calibration report

```python
from evaljev import calibration_report

report = calibration_report(monitor.store.list("research-agent"))
# {"route": {"n": ..., "accuracy": ..., "mean_probability": ..., "ece": ..., "brier": ...}}
print(report)
```

### 3. Stability test

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

### 4. Replay production traces

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

### 5. Measure a candidate against a contemporaneous control

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

### 6. Cap what an evaluation run can spend

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

### 7. Watch for drift over time

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

Three signals are tested, and the one that matters most in production needs no labels:

- **Certainty is tested as a distribution.** When a question's wording breaks, the
  model usually keeps answering inside its schema and keeps picking a plausible
  label — what collapses is the probability on the winner. A rank test on `p_max`
  across the two windows sees that; counting how many answers crossed a review line
  does not. On the recorded incident below it fires at `p = 2e-06` with no ground
  truth, and the labelled accuracy signal only confirms it afterwards.
  `rank_sum_test` is the one **approximate** test in `stats` — no exact null exists
  once ranks are tied, and decision APIs tie constantly at 1.00 — so it is gated on
  a minimum sample (8 a side) and a minimum move (`certainty_shift_threshold`,
  default 0.05). Significance alone will report a 0.01 move on a distribution that
  sits at 1.00.

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

### 8. Attribute a change to a component

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

### 9. Propose a repair, and refuse to believe it without evidence

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
