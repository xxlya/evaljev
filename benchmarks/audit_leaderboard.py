"""Audit a published win/loss board the way EvalJev audits anything else.

A leaderboard is a list of rates, and a rate from a dozen runs is an interval, not
a number. This says which rows of a board are actually distinguishable from each
other and how many runs the indistinguishable ones would need.

    python benchmarks/audit_leaderboard.py

Rows are transcribed from a published board; pass ``--rows name:wins/runs`` to
audit your own. Nothing here is fetched at runtime, so the numbers below are
whatever was transcribed and dated — check them against the source before quoting.
"""

from __future__ import annotations

import argparse

from evaljev import compare_rates, fisher_exact_2x2, rate_with_ci

# jevbench.dev/leaderboard, "Gaming Harness Results", read 22 Sep 2026.
PUBLISHED = [
    ("Jev 1.13 · typesafe wire", 9, 12),
    ("Jev 1.13 · openrouter", 5, 6),
    ("OpenJev wire", 0, 9),
    ("Untagged early harness", 0, 51),
]
SOURCE = "jevbench.dev/leaderboard · StarCraft II · Liberation Day · read 22 Sep 2026"


def runs_needed(rate_a: float, rate_b: float, *, alpha: float = 0.05, cap: int = 5000) -> int | None:
    """Smallest balanced arm size at which these two rates could clear ``alpha``.

    Optimistic by construction: it asks when the *most likely* split would be
    significant, not when 80% of splits would. The real number is larger, which
    only strengthens the point when the answer is already hundreds.
    """
    for n in range(4, cap):
        a, b = round(rate_a * n), round(rate_b * n)
        if fisher_exact_2x2(a, n - a, b, n - b) < alpha:
            return n
    return None


def audit(rows: list[tuple[str, int, int]]) -> None:
    print(f"\n{SOURCE}\n")
    print(f"{'row':<28}{'published':>12}{'measured':>10}   95% interval")
    for name, wins, runs in rows:
        r = rate_with_ci(wins, runs)
        print(
            f"{name:<28}{f'{wins}/{runs}':>12}{r['rate'] * 100:9.1f}%   "
            f"[{r['ci_low'] * 100:.1f}%, {r['ci_high'] * 100:.1f}%]"
        )

    print("\nevery pair the board ranks against each other")
    for i, (name_a, wins_a, runs_a) in enumerate(rows):
        for name_b, wins_b, runs_b in rows[i + 1 :]:
            test = compare_rates(wins_a, runs_a, wins_b, runs_b)
            if test["changed"]:
                verdict = "separated"
            else:
                need = runs_needed(wins_a / runs_a, wins_b / runs_b)
                verdict = (
                    f"not separated — needs ~{need} runs each"
                    if need
                    else "not separated — no run count would separate these rates"
                )
            print(f"  {name_a:<26} vs {name_b:<26} p={test['p_value']:.4f}  {verdict}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        nargs="*",
        metavar="NAME:WINS/RUNS",
        help="audit your own board instead of the transcribed one",
    )
    args = parser.parse_args()
    if args.rows:
        rows = []
        for item in args.rows:
            name, _, score = item.rpartition(":")
            wins, _, runs = score.partition("/")
            rows.append((name or score, int(wins), int(runs)))
    else:
        rows = PUBLISHED
    audit(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
