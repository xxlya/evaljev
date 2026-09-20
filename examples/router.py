"""Jev -> Claude/Gemini router instrumented by EvalJev."""
from evaljev import JevHTTPClient, JsonlTraceStore, Monitor
from evaljev.integrations import execute_model_route

QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which execution strategy should handle this request?",
        "criteria": {
            "claude": "Use Claude for the request.",
            "gemini": "Use Gemini for the request.",
            "both": "Use both independently when a second opinion materially improves reliability.",
            "other": "None of these strategies is suitable.",
        },
    }
}


def policy(answers):
    ans = answers["route"]
    if ans.confidence is not None and ans.confidence < 0.55:
        return "both"
    return ans.selected


if __name__ == "__main__":
    prompt = "Explain why a Python async task can silently swallow an exception."
    client = JevHTTPClient()
    monitor = Monitor(JsonlTraceStore("traces.jsonl"))
    _, trace = monitor.run(
        client,
        workflow_id="dual-model-router",
        node_id="route-model",
        state={"user_request": prompt, "available_models": ["claude", "gemini"]},
        questions=QUESTIONS,
        policy=policy,
        question_version="q1",
        policy_version="p1",
        workflow_version="w1",
    )
    result = execute_model_route(trace.action, prompt)
    monitor.record_outcome(trace, outcome=result)
    print(trace.model_dump_json(indent=2))
