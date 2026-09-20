"""Jev -> Claude/Gemini router instrumented by EvalJev.

Needs TYPESAFE_API_KEY, plus VECTOR_API_KEY and/or GEMINI_API_KEY depending on
which route Jev picks. Copy .env.example to .env and fill it in, or export them.
"""
from pathlib import Path

from evaljev import JevHTTPClient, JsonlTraceStore, Monitor
from evaljev.integrations import execute_model_route

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_env() -> None:
    """Load this repo's .env into the environment, if python-dotenv is available.

    Kept optional and script-local: the library itself only reads os.environ and
    never mutates it, so exported variables work whether or not this runs.

    The path is pinned to the repo root deliberately. A bare ``load_dotenv()``
    searches parent directories and will happily load an unrelated ~/.env,
    pulling in credentials from outside the project. Already-exported variables
    take precedence (``override=False``).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("python-dotenv not installed; using already-exported environment")
        return
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=False)
        print(f"loaded credentials from {ENV_PATH}")
    else:
        print(f"no {ENV_PATH} found; using already-exported environment")

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
    load_env()
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
