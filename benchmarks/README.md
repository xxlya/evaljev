# JevBench as an EvalJev fixture

[JevBench](https://github.com/fstandhartinger/jevbench) ships labelled typed decisions
for Jev-class models. EvalJev uses its public splits as a **fixture**, not as a scoreboard:
labelled data is how you check that a detector fires when it should.

The two projects answer different questions, and the report is split along that line:

| | question | needs labels | when |
| --- | --- | --- | --- |
| JevBench | Is this decision model good? | yes | before you deploy, to choose |
| EvalJev | Is this deployed decision workflow still healthy — and which part broke? | no | continuously, on live traffic |

## Running it

```bash
git clone https://github.com/fstandhartinger/jevbench.git /tmp/jevbench
python benchmarks/run_jevbench.py --data /tmp/jevbench/datasets/public/original.jsonl --cap 5.00
python benchmarks/run_jevbench.py --data ... --degrade vague-instructions --out vague.jsonl
python benchmarks/run_jevbench.py --data ... --compare baseline.jsonl vague.jsonl
```

`original.jsonl` is the split worth starting from: 72 decisions = **36 paraphrase pairs**,
covering all six families and all three question types (`noul`, `choice`, `score`).
Every call is metered by `SpendBudget`; the full split costs about **$0.011**.

Label mapping follows JevBench's `IMPLEMENTATION.md` (`noul` → `{yes: p, no: 1-p}`;
`choice`/`score` → argmax over the model's own distribution, ties to the smallest label).
EvalJev does not re-derive JevBench's scoring beyond that.

## Results, jev-1.13.0, `original` split, 2026-09-20

```
accuracy                 71/72 = 0.986      ($0.000155/decision, p50 888ms, p95 1140ms)
schema adherence         72/72 = 1.000      (all six families)
paraphrase branch flips  1/36  = 0.028      (mean JS shift 0.060)
ordinal MAE              0.003
```

Per family (ECE on `p_max`, see "Confidence is not a probability" below):

```
adequacy   [noul  ] rate 0.500  p̄ 0.513  ECE 0.0875   policy  [noul  ] rate 0.500  p̄ 0.381  ECE 0.1742
extraction [choice] rate 1.000  p̄ 0.999  ECE 0.0008   routing [choice] rate 1.000  p̄ 0.997  ECE 0.0033
intent     [choice] rate 1.000  p̄ 0.990  ECE 0.0100   ordinal [score ] rate 1.000  p̄ 0.998  ECE 0.0017
```

For `noul` families the rate column is the **base rate** of the proposition, not
decision accuracy — `P(true)` is paired against whether the proposition holds.

### The one finding worth the whole run

The single incorrect decision and the single paraphrase flip are **the same item**,
`original-policy-06` — an inclusive-boundary policy question:

```
-06-0  "cancel free until 24 hours before departure, including exactly 24 hours.
        Departure is 24 hours away."          P(true)=0.86 -> yes   correct
-06-1  "There are exactly 24 hours until departure. Free cancellation is allowed
        at or before that 24-hour cutoff."    P(true)=0.33 -> no    WRONG
```

Resampling each phrasing five times gives P = 0.85/0.86/0.85/0.85/0.86 and
0.32/0.32/0.32/0.33/0.32. Each input is **near-deterministic** (σ≈0.005), but the two
phrasings sit on opposite sides of the 0.5 threshold.

Two consequences:

1. The flip is a reproducible phrasing sensitivity, not sampling noise. Repeating the
   same input cannot find it; only perturbing the input can.
2. The label-free signal located the labelled error. In production there is no
   `expected` column — but the paraphrase flip is observable, and here it pointed at
   the one decision out of 72 that was wrong.

### Fault injection

`--degrade strip-criteria` is rejected by the API: `Choice question "intent" needs a
"criteria" map of options.` (HTTP 400). That fault class fails fast at the boundary
rather than degrading silently, so it needs no detector.

`--degrade vague-instructions` replaces the instruction text but keeps the rubric. It did
**not** reduce accuracy (72/72 vs 71/72 — one item changed, in the *right* direction),
which says the per-label criteria carry the semantics on this split. It did move
calibration: `adequacy` ECE 0.088 → 0.189, `extraction` 0.001 → 0.020, `intent`
0.010 → 0.019. A workflow edit can leave accuracy untouched while degrading the
probability your policy threshold keys on.

One run of n=72, so single-item differences are not significant. The calibration shift
is the more trustworthy of the two observations.

## Confidence is not a probability

The API's Choice `confidence` is a normalized margin over a uniform prior, not a
second probability estimate:

    c = (p_max - 1/k) / (1 - 1/k)

Checked against the 48 choice/score answers in this run: max absolute error **0.0100**,
which is the reporting granularity of the two-decimal probabilities. So `confidence`
carries no information beyond `p_max` and `k`.

Two consequences, both of which the library now handles:

1. **ECE and Brier must be computed on `p_max`, not on `confidence`.** Feeding a
   rescaled margin to a calibration metric scores the wrong quantity. The error is
   small when `p_max ≈ 1` (the transform is near-identity there) and largest when the
   decision is uncertain — exactly the regime a threshold cares about. `k=2, p_max=0.6`
   reports `confidence=0.2`.
2. **A confidence threshold is not portable across schemas.** `confidence < 0.55`
   escalates below `p_max` 0.775 at k=2 but 0.595 at k=10. Add one option to a criteria
   map and every threshold keyed on confidence silently moves — no model change, no code
   change, no accuracy change. Set thresholds on `p_max`, or convert with
   `confidence_to_pmax(c, k)`.

Source: [Jev's Architecture, Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked/),
verified here against live traces.

## Reproducing

Trace files (`benchmarks/*.jsonl`) are gitignored. Each run writes an append-only log:
one event per decision, one more when its outcome is recorded.
