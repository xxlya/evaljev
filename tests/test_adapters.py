"""Adapter tests against the log shape the StarCraft harness actually writes.

The fixture mirrors `jev_sc2`'s own writer: `{'time', 'event', **fields}` rows in
`runs/<stamp>/events.jsonl`, with the outcome in `result.json`.
"""

import json

import pytest

from evaljev import build_report, load_sc2_run, load_sc2_runs, schema_fingerprint

PRIORITY = {
    "type": "choice",
    "instructions": "What should the army do this tick?",
    "criteria": {"attack": "Push the objective.", "hold": "Hold position.", "retreat": "Fall back."},
}
MARINE = {
    "type": "choice",
    "instructions": "What should marines do?",
    "criteria": {"focus": "Focus fire.", "spread": "Spread out."},
}


def jev_event(t, revision, *, priority, marine, latency=180, cost=0.00004):
    return [
        {"time": t, "event": "tick", "loop": int(t), "revision": revision},
        {
            "time": t + 0.1,
            "event": "jev",
            "latency_ms": latency,
            "via": "typesafe",
            "state": {"loop": int(t), "own_units": 8},
            "questions": {"priority": PRIORITY, "marine": MARINE},
            "response": {
                "model": "jev-1.13.0",
                "usage": {"cost": cost},
                "answers": {
                    "priority": {"choice": priority[0], "probabilities": priority[1]},
                    "marine": {"choice": marine[0], "probabilities": marine[1]},
                },
            },
        },
    ]


@pytest.fixture
def runs_dir(tmp_path):
    def write(name, rows, status):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "events.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        (directory / "result.json").write_text(
            json.dumps({"status": status, "calls": len(rows) // 2}), encoding="utf-8"
        )

    sure = ({"attack": 0.95, "hold": 0.03, "retreat": 0.02}, {"focus": 0.9, "spread": 0.1})
    unsure = ({"attack": 0.4, "hold": 0.35, "retreat": 0.25}, {"focus": 0.52, "spread": 0.48})
    write(
        "20260922T120000Z",
        [e for i in range(6) for e in jev_event(1000 + i, "abc1234",
                                                priority=("attack", sure[0]), marine=("focus", sure[1]))],
        "victory",
    )
    write(
        "20260922T130000Z",
        [e for i in range(6) for e in jev_event(2000 + i, "def5678",
                                                priority=("hold", unsure[0]), marine=("focus", unsure[1]))],
        "incomplete",
    )
    return tmp_path


def test_one_call_becomes_one_trace_per_question(runs_dir):
    traces = load_sc2_run(runs_dir / "20260922T120000Z")
    assert len(traces) == 12                       # 6 calls, 2 questions each
    assert {t.node_id for t in traces} == {"priority", "marine"}
    first = traces[0]
    assert first.model == "jev-1.13.0"
    assert first.policy_version == "abc1234"       # the git revision of player.py
    assert first.latency_ms == 180
    assert first.metadata["cost_usd"] == 0.00004
    assert first.metadata["run_id"] == "20260922T120000Z"
    assert first.metadata["run_outcome"] == "victory"
    assert first.answers[0].probabilities == {"attack": 0.95, "hold": 0.03, "retreat": 0.02}


def test_questions_asked_at_one_loop_share_a_request(runs_dir):
    traces = load_sc2_run(runs_dir / "20260922T120000Z")
    ids = {t.metadata["request_id"] for t in traces}
    assert len(ids) == 6
    assert all(i.startswith("20260922T120000Z:") for i in ids)


def test_a_reworded_question_gets_a_new_version_without_anyone_remembering():
    """Harnesses build questions in code and rarely version them."""
    a = schema_fingerprint(PRIORITY)
    b = schema_fingerprint({**PRIORITY, "instructions": "What should the army do now?"})
    c = schema_fingerprint({**PRIORITY, "criteria": {**PRIORITY["criteria"], "flank": "Go around."}})
    assert a != b and a != c and b != c
    assert schema_fingerprint(dict(reversed(list(PRIORITY.items())))) == a  # key order is not a change


def test_each_game_is_a_run_and_carries_its_result(runs_dir):
    report = build_report(load_sc2_runs(runs_dir))
    runs = report["runs"]
    assert [r["requests"] for r in runs] == [6, 6]
    assert runs[0]["outcome"] == "incomplete" and runs[1]["outcome"] == "victory"
    assert runs[1]["verdict"] == "baseline"
    # The second game answered the same questions far less surely, and said so —
    # naming the player.py revision that shipped between the two games.
    assert runs[0]["verdict"] == "harmful"
    assert any("second look" in e["plain"] for e in runs[0]["effects"])
    assert any("policy version is now def5678" in c["plain"] for c in runs[0]["changes"])
    assert len(runs[0]["changes"]) == 1          # one change, both nodes, not two entries
    assert runs[0]["changes"][0]["node_id"] == "marine, priority"


def test_a_missing_run_directory_is_empty_not_an_error(tmp_path):
    assert load_sc2_run(tmp_path / "nope") == []
    (tmp_path / "half").mkdir()
    (tmp_path / "half" / "events.jsonl").write_text('{"time":1,"event":"tick","loop":1}\n{"time":2,"ev')
    assert load_sc2_run(tmp_path / "half") == []
