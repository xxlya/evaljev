from evaljev import InMemoryTraceStore, Monitor, replay, summarize_replay


class FakeClient:
    model = "fake-jev"

    def __init__(self, answer="claude", confidence=0.8):
        self.answer = answer
        self.confidence = confidence

    def decide(self, *, state, questions):
        return {
            "model": self.model,
            "answers": {
                "route": {
                    "choice": self.answer,
                    "confidence": self.confidence,
                    "probabilities": {"claude": self.confidence, "gemini": 1 - self.confidence},
                }
            },
        }, 3.2


QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which model should handle the request?",
        "criteria": {"claude": "Claude", "gemini": "Gemini", "other": "No match"},
    }
}


def policy(a):
    return a["route"].selected


store = InMemoryTraceStore()
monitor = Monitor(store)
for i in range(5):
    _, trace = monitor.run(
        FakeClient("claude"), workflow_id="demo", node_id="router", state={"id": i},
        questions=QUESTIONS, policy=policy,
    )
    trace.outcome_correct = i % 2 == 0

results = replay(store.list(), FakeClient("gemini"), questions=QUESTIONS, policy=policy)
print(summarize_replay(results))
