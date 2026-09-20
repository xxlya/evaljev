from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .models import QuestionSpec


@dataclass
class LintIssue:
    question: str
    severity: str
    code: str
    message: str


def lint_questions(questions: Iterable[QuestionSpec]) -> list[LintIssue]:
    issues: list[LintIssue] = []
    subjective = re.compile(r"\b(important|good|bad|reasonable|appropriate|normal)\b", re.I)
    composite = re.compile(r"\b(and|or)\b", re.I)

    for q in questions:
        text = str(q.instructions)
        if len(text.strip()) < 12:
            issues.append(LintIssue(q.name, "warning", "too_short", "Instructions may be underspecified."))
        if subjective.search(text):
            issues.append(LintIssue(q.name, "warning", "subjective", "Instructions contain subjective language."))
        if composite.search(text) and q.type == "noul":
            issues.append(LintIssue(q.name, "warning", "composite_noul", "Noul question may combine multiple judgments."))
        if q.type == "choice" and isinstance(q.criteria, dict):
            labels = {str(k).lower() for k in q.criteria}
            if not labels.intersection({"other", "none", "unknown", "none_of_the_above"}):
                issues.append(LintIssue(q.name, "info", "no_fallback", "Consider an explicit no-match/other option."))
            descriptions = [str(v).strip().lower() for v in q.criteria.values() if v is not None]
            if len(descriptions) != len(set(descriptions)):
                issues.append(LintIssue(q.name, "warning", "duplicate_criteria", "Two or more choice descriptions are identical."))
    return issues
