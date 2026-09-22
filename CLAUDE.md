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
8. **The dashboard never renders "unknown" as healthy.** A check with too little
   evidence reports `unknown`, is excluded from the score, and is listed under the
   verdict. `report.py` translates; it computes nothing of its own — every number on
   the page comes from a library function, and no prose on it is model-written.

## Current state

Branch `monitoring-dashboard`, off `main` (the previous branch is merged).
240 tests, ruff clean. Total spend across all development so far: ~$0.05.

Done: budget ceilings · outcome persistence (append-only, latest-wins) · schema
adherence · calibration on `p_max` · intervals + paired verdicts · noise floors ·
attribution + vendor-drift check · drift windows with comparability guard · gated
repair cycle · JevBench harness · **monitoring dashboard + `evaljev` CLI** · published
site (`docs/`, GitHub Pages from `/docs`).

The published site is two pages, both generated by
`benchmarks/export_demo_data.py`:

- `docs/index.html` — the dashboard, rendered from the recorded run by exactly the
  code path `evaljev report` runs for a user. Do not hand-edit it.
- `docs/how-it-works.html` — the earlier essay page, reading `docs/data.js`.

The dashboard itself lives in `src/evaljev/report.py` (the report dict) and
`src/evaljev/assets/dashboard.html` (the template, with `"__EVALJEV_REPORT__"`
replaced by the JSON). `src/evaljev/assets/sample-traces.jsonl.gz` is the real
baseline+vague+stripped run, bundled so `evaljev demo` works after a bare install.
Drift in the report is tested **per `node_id`**, never across the whole stream: two
windows with different node mixes are not comparable, so the stream-level guard would
fire on nearly every real workflow and say nothing.

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
