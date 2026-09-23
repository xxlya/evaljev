"""Audit a JevBench ranking: which positions on the board are actually separated?

JevBench publishes a ranked table of systems scored on Intelligence, Calibration,
Speed and Cost. Intelligence is a **sample statistic** — an accuracy over a finite
set of items — and a ranking prints it as an exact number. Two systems half a point
apart may be one system as far as the evidence goes.

This does not re-score anything and does not dispute a measurement. It reads the
published artifacts and asks one question of each adjacent pair on the board: on the
items both systems answered, is one of them actually better?

The test is **paired** (exact McNemar), because every system saw the same items.
That is far stronger than comparing two accuracies, and it is why items both got
right or both got wrong are excluded by construction: they carry no information
about which is better.

    git clone https://github.com/fstandhartinger/jevbench /tmp/jevbench
    python benchmarks/audit_jevbench.py --repo /tmp/jevbench
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaljev import min_discordant_for_significance, paired_comparison, rate_with_ci

CORRECT, ATTEMPTED = "c", {"c", "w", "f"}


def load(repo: Path, revision: str) -> tuple[list[dict], dict, list[str], str]:
    results = json.loads((repo / f"results/{revision}/jevbench-{revision}-results.json").read_text())
    per_task = json.loads(
        (repo / f"results/{revision}/jevbench-{revision}-per-task.json").read_text()
    )
    ranked = [s for s in results["systems"] if s.get("ranked")]
    ranked.sort(key=lambda s: -s["jevbench_score"])
    tasks = [t["id"] for t in per_task["tasks"] if t.get("public")]
    # The artifact's own revision, not the directory it lives in: v1.3.0 is a
    # scoring-only release over the v1.2 measurements.
    return ranked, per_task["systems"], tasks, results.get("revision", revision)


def outcomes(per_task_system: dict, tasks: list[str]) -> list[bool | None]:
    """Correct / wrong / not attempted, in one fixed item order.

    ``None`` for an item a system never attempted, so the paired test drops that
    pair rather than scoring an absence as a loss.
    """
    rows = per_task_system.get("public_tasks", {})
    out: list[bool | None] = []
    for task in tasks:
        entry = rows.get(task)
        code = entry[0] if isinstance(entry, list) else entry
        out.append(code == CORRECT if code in ATTEMPTED else None)
    return out


def paraphrase_pairs(repo: Path, tasks: list[str]) -> dict[str, list[str]]:
    """Public items that are two phrasings of the same decision.

    JevBench's standard tier is the `original` split: 36 decisions, each written
    twice, same expected answer. The pairing is in the dataset's ``group`` field.
    """
    groups: dict[str, list[str]] = {}
    known = set(tasks)
    for name in ("original", "easy", "hard", "standard"):
        path = repo / f"datasets/public/{name}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("group") and row["id"] in known:
                groups.setdefault(row["group"], []).append(row["id"])
    return {g: ids for g, ids in groups.items() if len(ids) == 2}


def report_rephrasing(repo: Path, board: list[dict], marks: dict, tasks: list[str]) -> None:
    """A column the board does not have, from data it already published.

    Two phrasings of one decision, with the same right answer. A system that gets
    one right and the other wrong has been moved by the wording — and unlike every
    axis on the board, this needs no ground truth to notice in production, which is
    the whole reason to care about it. It is a **lower bound**: the outcomes record
    right or wrong, not which wrong answer, so two different wrong answers to the
    same question count here as consistent.
    """
    pairs = paraphrase_pairs(repo, tasks)
    if not pairs:
        return
    index = {task: i for i, task in enumerate(tasks)}
    print(f"\nRephrasing: {len(pairs)} decisions written two ways, same right answer\n")
    print(f"{'#':>3}  {'system':<36}{'split pairs':>12}   95% interval")
    rows = []
    for rank, system in enumerate(board, 1):
        outcome = marks[system["key"]]
        split = both = 0
        for ids in pairs.values():
            a, b = (outcome[index[i]] for i in ids)
            if a is None or b is None:
                continue
            both += 1
            split += a != b
        if both:
            rows.append((rank, system, split, both))
    for rank, system, split, both in rows:
        r = rate_with_ci(split, both)
        print(
            f"{rank:>3}  {system['display'][:36]:<36}{f'{split}/{both}':>12}   "
            f"[{r['ci_low'] * 100:.0f}%, {r['ci_high'] * 100:.0f}%]"
        )
    best = min(rows, key=lambda r: r[2])
    worst = max(rows, key=lambda r: r[2])
    print(
        f"\n{worst[1]['display']} was moved by the wording on {worst[2]} of {worst[3]} pairs; "
        f"{best[1]['display']} on {best[2]}.\nNeither number is on the board, and the gap "
        "between them is not in any of its four axes."
    )


def audit(repo: Path, revision: str, top: int) -> int:
    ranked, per_task, tasks, artifact_revision = load(repo, revision)
    board = [s for s in ranked if s["key"] in per_task][:top]
    if not board:
        print("No ranked system has per-task outcomes in this artifact.")
        return 1

    print(f"\nJevBench {artifact_revision} · {len(ranked)} ranked systems · {len(tasks)} public items")
    print(
        "The board ranks by JevBench Score, a geometric mean over Intelligence, Calibration,\n"
        "Speed and Cost. Only Intelligence is a sample statistic; this audits that axis, on the\n"
        "public items alone (the held-out and imported items appear only in aggregates).\n"
    )

    print(f"{'#':>3}  {'system':<34}{'score':>7}{'intel':>7}   accuracy on public items")
    marks = {}
    for i, system in enumerate(board, 1):
        marks[system["key"]] = outcomes(per_task[system["key"]], tasks)
        attempted = [m for m in marks[system["key"]] if m is not None]
        rate = rate_with_ci(sum(attempted), len(attempted))
        print(
            f"{i:>3}  {system['display'][:34]:<34}{system['jevbench_score']:>7.1f}"
            f"{system['axes']['intelligence']:>7.1f}   {rate['rate']:.3f}"
            f"  [{rate['ci_low']:.3f}, {rate['ci_high']:.3f}]  n={len(attempted)}"
        )

    floor = min_discordant_for_significance()
    print(
        f"\nEach adjacent pair, tested on the items both answered "
        f"(exact McNemar; below {floor} disagreements no split can reach p<0.05)\n"
    )
    separated, inversions = 0, []
    for i in range(len(board) - 1):
        upper, lower = board[i], board[i + 1]
        # improvements = items the higher-ranked system got right and the lower one
        # did not; regressions the other way round.
        test = paired_comparison(marks[lower["key"]], marks[upper["key"]])
        discordant = test["discordant"]
        if test["p_value"] < 0.05:
            separated += 1
            if test["regressions"] > test["improvements"]:
                verdict = "SEPARATED — the lower-ranked system is the more accurate one"
                inversions.append((i + 1, upper, lower, test))
            else:
                verdict = "separated"
        elif discordant < floor:
            verdict = f"cannot conclude — only {discordant} disagreements"
        else:
            verdict = "not separated"
        print(
            f"{i + 1:>3} vs {i + 2:<3} {upper['display'][:24]:<25} {lower['display'][:22]:<23}"
            f"#{i + 1} wins {test['improvements']:<3} #{i + 2} wins {test['regressions']:<3}"
            f"p={test['p_value']:.3f}  {verdict}"
        )

    print(f"\nSeparated on accuracy: {separated} of {len(board) - 1} adjacent pairs.")
    if inversions:
        print(
            "\nRanked one way, more accurate the other way — the composite is doing this, and\n"
            "it is doing it on purpose (Speed and Cost are a full half of the score). Worth\n"
            "saying out loud, because a reader takes a ranking for an ordering of ability:"
        )
        for rank, upper, lower, test in inversions:
            u_axes, l_axes = upper["axes"], lower["axes"]
            print(
                f"  #{rank + 1} {lower['display'][:34]:<35} beats #{rank} "
                f"{upper['display'][:30]} on accuracy (p={test['p_value']:.3f})"
            )
            print(
                f"      but ranks below it: speed {l_axes['speed']:.0f} vs {u_axes['speed']:.0f}, "
                f"cost {l_axes['cost']:.0f} vs {u_axes['cost']:.0f}, "
                f"intelligence {l_axes['intelligence']:.0f} vs {u_axes['intelligence']:.0f}"
            )

    # How far down the board is indistinguishable from the top system.
    leader = board[0]
    band = [
        other["display"]
        for other in board[1:]
        if paired_comparison(marks[other["key"]], marks[leader["key"]])["p_value"] >= 0.05
    ]
    print(
        f"\nAnywhere on this board, {len(band)} system(s) are not separated from the "
        f"top one ({leader['display']}) on these items:"
    )
    for name in band or ["(none)"]:
        print(f"    {name}")
    report_rephrasing(repo, board, marks, tasks)

    print(
        "\nThis is a statement about the evidence, not about the systems: it says the public\n"
        "items cannot tell these apart, not that they are equally good. The held-out items,\n"
        "and the other three axes, are outside what this can test.\n"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="/tmp/jevbench", help="a clone of fstandhartinger/jevbench")
    parser.add_argument("--revision", default="v1.2", help="which results artifact to read")
    parser.add_argument("--top", type=int, default=12, help="how many ranked systems to audit")
    args = parser.parse_args()
    return audit(Path(args.repo), args.revision, args.top)


if __name__ == "__main__":
    raise SystemExit(main())
