"""Offline tests for JevHTTPClient.

httpx.post is patched in every test, so the suite makes no network call and
needs no API key.
"""

from unittest.mock import MagicMock, patch

import pytest

from evaljev.client import JEV_BASE_URL, JevHTTPClient


@pytest.fixture
def jev_key(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "jv_live_fake")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_BASE_URL", raising=False)


def mock_post(payload=None):
    """Patch httpx.post, returning the mock for call inspection."""
    patcher = patch("evaljev.client.httpx.post")
    post = patcher.start()
    post.return_value = MagicMock(**{"json.return_value": payload or {"answers": {}}})
    return patcher, post


def test_reads_jev_api_key(jev_key):
    assert JevHTTPClient().api_key == "jv_live_fake"


def test_falls_back_to_typesafe_api_key(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "legacy_fake")
    assert JevHTTPClient().api_key == "legacy_fake"


def test_explicit_key_wins_over_environment(jev_key):
    assert JevHTTPClient("passed_in").api_key == "passed_in"


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="JEV_API_KEY"):
        JevHTTPClient()


def test_base_url_defaults_and_is_overridable(jev_key, monkeypatch):
    assert JevHTTPClient().base_url == JEV_BASE_URL
    monkeypatch.setenv("JEV_BASE_URL", "https://example.test/decide")
    assert JevHTTPClient().base_url == "https://example.test/decide"
    # An explicit argument beats the environment.
    assert JevHTTPClient(base_url="https://arg.test/decide").base_url == "https://arg.test/decide"


def test_decide_posts_state_and_questions(jev_key):
    patcher, post = mock_post({"answers": {"route": {"choice": "claude"}}})
    try:
        questions = {"route": {"type": "choice"}}
        response, latency_ms = JevHTTPClient().decide(state={"q": 1}, questions=questions)
        assert response == {"answers": {"route": {"choice": "claude"}}}
        assert latency_ms >= 0
        assert post.call_args.args[0] == JEV_BASE_URL
        kwargs = post.call_args.kwargs
        assert kwargs["headers"] == {
            "Authorization": "Bearer jv_live_fake",
            "Content-Type": "application/json",
        }
        # The model is inferred server-side and must not be sent.
        assert kwargs["json"] == {"state": {"q": 1}, "questions": questions}
        assert kwargs["timeout"] == 30.0
        post.return_value.raise_for_status.assert_called_once()
    finally:
        patcher.stop()


def test_decide_propagates_http_errors(jev_key):
    patcher, post = mock_post()
    post.return_value.raise_for_status.side_effect = RuntimeError("401")
    try:
        with pytest.raises(RuntimeError, match="401"):
            JevHTTPClient().decide(state={}, questions={})
    finally:
        patcher.stop()


def test_model_is_kept_for_trace_labelling_only(jev_key):
    patcher, post = mock_post()
    try:
        client = JevHTTPClient(model="jev-experimental")
        assert client.model == "jev-experimental"
        client.decide(state={}, questions={})
        assert "model" not in post.call_args.kwargs["json"]
    finally:
        patcher.stop()
