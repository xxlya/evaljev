# EvalJev — working notes

Runtime assurance for Jev-class typed decision workflows. Read `README.md` for the
API; this file is the context a new session needs that the code does not state.

## Dev environment

pytest and ruff are **not** on the default python. Use the conda env directly:

```bash
~/miniconda3/envs/evaljev/bin/pytest -q
~/miniconda3/envs/evaljev/bin/ruff check .
~/miniconda3/envs/evaljev/bin/python -m evaljev.cli demo -o /tmp/demo.html   # the dashboard
```

`.env` (gitignored) holds `JEV_API_KEY`, `VECTOR_API_KEY`, `GEMINI_API_KEY`. Scripts
load it via `python-dotenv` pinned to the repo root; the library itself only reads
`os.environ`.

## What this project is, and is not

Three different questions, often conflated:

| | Question | Needs labels | When |
|---|---|---|---|
| JevBench | Is this decision **model** good? | yes | once, to choose |
| [Archer's reverse-engineering](https://archerhume.com/posts/jevs-architecture-unmasked/) | What **is** it, mechanically? | no | once, version-bound |
| **EvalJev** | Is my **deployed workflow** still healthy, and what broke? | no | continuously |

The real dividing line is **label availability**, not "model vs workflow" — both
measure the same primitives. JevBench is a **fixture** (labelled data to check that
label-free detectors fire); the architecture write-up is a **prior** (tells us what
to perturb). Neither is a competitor. Do not turn this into another benchmark: no
leaderboards, no cross-model ranking, no re-implementing JevBench scoring.

A deployed decision has five moving parts — state construction, schema, model,
policy/threshold, downstream action. Benchmarks hold four constant. In production the
model is usually the *least* likely thing to have broken.

## Verified facts about the live API

Each was measured, not assumed. Re-verify before trusting after a version bump.

- **A certainty collapse is the label-free signal that fires when wording breaks.**
  On the rotated-criteria run: `p_max` median 1.00 → 0.79, rank-sum p = 2e-06, while
  no action's share change reached significance and every answer stayed valid.
  `rank_sum_test` is the only **approximate** test in `stats.py` (no exact null under
  ties, and answers tie constantly at 1.00); `drift_report` gates it on 8 values a
  side and a median move ≥ 0.05, because on a distribution piled at 1.00 a 0.01 move
  is significant and worthless.
- **`confidence` is a margin, not a probability**: `c = (p_max − 1/k)/(1 − 1/k)`.
  Checked on 48 live answers, max error 0.0100 (the 2-decimal reporting granularity).
  Consequences: score calibration on `p_max`; a confidence threshold is **not
  portable** across `k`, so adding one option to a `criteria` map silently moves
  every threshold keyed on it.
- **`noul` returns a bare scalar** `P(true)` with no `confidence` and no
  `probabilities`. This caused three separate silent bugs (adherence false alarm,
  0.0 flip rate, `None` distribution shift). Use `answer_branch` /
  `answer_distribution` from `models.py` — never `selected` directly.
- **Questions do not contaminate each other.** A question's distribution moves no
  more when a second shares the request (mean JS 0.0058) than between two identical
  requests (0.0035); maxima identical. State is encoded once, so one 2-question
  request costs ~35% less and returns ~49% faster than two.
- **Per-call jitter is ~0.005–0.011 JS.** That is the noise floor any instability
  claim must clear.
- **Schemas are validated server-side**: a choice question with no `criteria` is a
  400, not silent degradation.
- Cost is ~**$0.00015/decision**, and each response carries exact `usage.cost_usd`
  plus `credits_remaining_usd`.

## Conventions that are load-bearing

Breaking these silently undoes the point of the library.

1. **Every rate carries its interval.** 1 flip in 36 is `[0.005, 0.142]`, not 0.028.
2. **A verdict is never "no difference".** Too little evidence means *nothing was
   shown*, not equivalence. Check `min_discordant_needed` / `min_samples_needed`
   before paying for a comparison that cannot conclude.
3. **Replay comparisons are paired; window comparisons are not.** McNemar/signed-rank
   for replay, Fisher for drift windows. Do not mix them up.
4. **Attribution is deterministic and auditable — no LLM.** Its output justifies
   changing production config. An LLM is used in exactly one place: proposing
   candidate schema wordings.
5. **Repair proposes, never applies.** The proposer never sees the holdout, and a
   candidate may not change the label set.
6. **Threshold on `p_max`, never on `confidence`.**
7. **Every paid call goes through `SpendBudget`.** Default cap $5/API.
8. **Three bugs the realistic example exposed — do not reintroduce them.**
   `count_windows` chops from the oldest forward, so the newest window used to be the
   *remainder* (117 traces at size 29 left a window of one, and nothing could fire);
   `drift_report` now windows with `align="end"`. `schema_adherence` held every answer
   to the newest label set, so adding an option mid-stream reported every earlier
   answer as a schema failure; with no `labels` argument it now derives each trace's
   own. And an ordinal policy of `int(expected_value)` can pick a level the model gave
   no probability to — use `answer_branch`, which is the argmax.
9. **A count without its denominator is not a number, and a metric nobody can read is
   not worth page space.** Every figure carries what it is a share of. Metrics were
   audited by "what decision does this change": ECE as a number, JS distance,
   composition shift, the test names (McNemar, Fisher, signed-rank, Wilson), the
   JevBench family table, the repair funnel and the batching measurements are all gone
   from the pages — several remain in the library and in the evidence JSON, which is
   where a reader who wants them will look. `docs/how-it-works.html` is now five rules,
   three self-imposed constraints, and a section on what the tool cannot see.
10. **The dashboard never renders "unknown" as healthy.** A check with too little
   evidence reports `unknown` and is listed under the verdict. (There was a 0–100
   health score; it was an invented formula that changed no decision, so it is gone.) `report.py` translates; it computes nothing of its own — every number on
   the page comes from a library function, and no prose on it is model-written. The
   narrative at the top (`_story`) is assembled in `report.py`, not in JavaScript, so
   the claims it makes are testable.

## Current state

Branch `monitoring-dashboard`, off `main` (the previous branch is merged).
240 tests, ruff clean. Total spend across all development so far: ~$0.05.

Done: budget ceilings · outcome persistence (append-only, latest-wins) · schema
adherence · calibration on `p_max` · intervals + paired verdicts · noise floors ·
attribution + vendor-drift check · drift windows with comparability guard · gated
repair cycle · JevBench harness · **monitoring dashboard + `evaljev` CLI** · published
site (`docs/`, GitHub Pages from `/docs`).

The repo is **public** and Pages is live at <https://xxlya.github.io/evaljev/>
(source: `main` / `/docs`). Generated by `benchmarks/export_demo_data.py` — do not
hand-edit `index.html`:

- `docs/index.html` — the **run audit** for the example support assistant. The unit is
  a *run*: a stretch of traffic answered with one configuration, cut wherever a node's
  version tuple changes (`_split_runs`). Per run: what changed (per-node `attribute`),
  what it did (tested effects), same-input before/after evidence, a verdict, and the
  requests it answered alone that it should not have.

  A run is whatever the caller labelled (`run_key`, default `metadata["run_id"]`) and,
  absent a label, a stretch of unchanged configuration. The baseline for a run is
  **the last stretch that held still** — runs since the last
  one that moved something, pooled (`baseline_from` in `_audit_runs`). Comparing only
  against the immediately preceding run threw away the power that made the effects
  visible: 24 vs 24 could not establish the rate change that 24 vs 93 could.

  Inside a run, the number that leads is **decisions the workflow made alone that it
  should not have**, not "requests that need a person". The first version counted every flagged
  request and read 48 on the example workflow — meaningless, because that workflow
  already escalates 92 of 117 by its own policy, so the count restated the
  application's own decision. A flag only asks something of a reader when it
  contradicts what the workflow did. `human_actions` (inferred from a name
  vocabulary, overridable) is what makes that split possible, and the page always
  says which names it matched.

  Two more things about the console are deliberate. It opens with a **sentence**, not a row
  of KPI cards — the counters are on the page anyway, and what a reader needs first is
  which of them matters and why. And **no identifier is printed in snake_case**:
  `classify_request` reads as "classify request" everywhere, with the exact id kept in
  a `title`. Both pages run the same `words()` helper for that.
- `docs/how-it-works.html` — the essay page, reading `docs/data.js` (JevBench).
- `docs/incident.html` — a redirect to the console, kept so an earlier link works.

Both views render from one `build_report()` dict through `render_html(report, view)`
(`VIEWS` in `report.py`), so they cannot disagree about a number. The queue lives in
`report["queue"]`: per-request flags with a fixed priority
(`invalid > disagreed > inconsistent > unsure > changed`), each carrying its own
one-line advice. `needs a person` is not a guess at the app's escalation rules — it is
the caller's `unsure_below` line plus the three failures that make an answer unusable
whatever the threshold was.

`examples/support_assistant.py` records that run and is the copy-me template:
three nodes (`classify_request` → `rate_urgency` → `route_handoff`), a review
threshold on `p_max`, outcomes for one request in three, and
`metadata={"request_id": ...}` at every node — which is what lets the page show a
request's path. `examples/support-v*.jsonl` is committed (the `benchmarks/*.jsonl`
gitignore does not cover it) and is also bundled gzipped for `evaljev demo`.

The four phases are two edits that turned out harmless (add an option; shorten the
wording) and one that did not: `v4` sorts the labels while their descriptions stay in
insertion order, so `zip()` puts every description on the wrong label. Caught without
labels — `needs_review` 0% → 14%, typical certainty 1.00 → 0.86, 13 of 24 repeated
inputs answered differently. Re-record with
`python examples/support_assistant.py --phase v4` (~$0.012 a phase).

The dashboard itself lives in `src/evaljev/report.py` (the report dict) and
`src/evaljev/assets/dashboard.html` (the template, with `"__EVALJEV_REPORT__"`
replaced by the JSON). `src/evaljev/assets/sample-traces.jsonl.gz` is the real
baseline+vague+stripped run, bundled so `evaljev demo` works after a bare install.
Drift in the report is tested **per `node_id`**, never across the whole stream: two
windows with different node mixes are not comparable, so the stream-level guard would
fire on nearly every real workflow and say nothing.

`jevbench-rotated.jsonl` (12 decisions, $0.0019, recorded 2026-09-21) is the same fault
on the benchmark: `--degrade rotate-criteria`. Accuracy 12/12 → 2/12, schema adherence
still 1.000. Kept as the labelled fixture that proves the detector fires.

## Auditing the benchmark board

`benchmarks/audit_jevbench.py` audits the Benchmark Heaven board from its own
artifacts (`results/v1.2/jevbench-v1.2-{results,per-task}.json`; the per-task file
carries outcome codes for 231 public items × 52 systems, so comparisons are **paired**
— exact McNemar, not two accuracies). Findings on v1.3.0, top 12: 6 of 11 adjacent
pairs separated; two inversions where the lower-ranked system is the more accurate one
(#8 over #7 at p<0.001, #11 over #10 at p=0.010 — speed and cost are half the composite);
5 systems not separated from the leader. It also computes a column the board lacks:
the standard tier is 36 paraphrase pairs, so "got one right and the other wrong" is a
rephrasing-robustness count (1/36 for the top four, 6/36 for rank 9) — a **lower bound**,
since outcomes record right/wrong and not which wrong answer.

## Known gaps

- **`workflow_id`/`node_id` are strings, not a graph.** This is currently *node*
  health, not workflow health. Attribution cannot cross node boundaries. Biggest
  name-vs-reality gap; also the largest piece of work.
- **`state_diff` is shallow by construction** — catches shape changes, not semantic
  drift in traffic. A clean state report is *not* evidence inputs are unchanged.
- `lint_questions` is five regex heuristics.
- **The dashboard is a snapshot of a file**, not a service: no retention, no auth, no
  history beyond the traces you pass it. `evaljev serve` re-reads the file; it does not
  store anything.
- Not built: schema-level perturbation (criteria ablation, option-order — the
  architecture write-up reports position sensitivity of ~50% vs 100%), threshold
  repair, a vendor-drift probe that continuously asserts the confidence formula,
  paraphrase generation, OTel/SQL store.

## Known environment issue

`VECTOR_API_KEY` is **over its account budget** (402, $301.77 of $300) — every
commercial model on that proxy fails. Free open-weight models (`gpt-oss-120b`,
`Qwen3_8-27B`) still work; pass `CLAUDE_MODEL=gpt-oss-120b` to the demo. Gemini goes
direct to Google and is unaffected. Verify before debugging any Claude-route failure.

## Reproducing the benchmark

```bash
git clone https://github.com/fstandhartinger/jevbench.git /tmp/jevbench
~/miniconda3/envs/evaljev/bin/python benchmarks/run_jevbench.py \
    --data /tmp/jevbench/datasets/public/original.jsonl --cap 5.00
~/miniconda3/envs/evaljev/bin/python benchmarks/export_demo_data.py   # both docs pages
```

`original.jsonl` is the split to use: 72 decisions = 36 paraphrase pairs, one shared
schema per family. The `hard` split has a distinct schema per item, so schema repair
is meaningless there.
