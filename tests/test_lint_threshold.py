from evaljev import QuestionSpec, lint_questions, optimize_binary_threshold


def test_linter_and_threshold():
    q = QuestionSpec(name="x", type="choice", instructions="What is important?", criteria={"a": "same", "b": "same"})
    issues = lint_questions([q])
    codes = {i.code for i in issues}
    assert "subjective" in codes
    assert "no_fallback" in codes
    assert "duplicate_criteria" in codes
    best = optimize_binary_threshold([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert 0.2 <= best["threshold"] <= 0.8
