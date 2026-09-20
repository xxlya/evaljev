"""Offline tests for the provider integrations.

Every test here mocks the SDK client, so the suite never makes a network
call and needs no API keys.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from evaljev.integrations import VECTOR_BASE_URL, call_claude, call_gemini, execute_model_route


@pytest.fixture
def claude_key(monkeypatch):
    monkeypatch.setenv("VECTOR_API_KEY", "vp_fake")
    monkeypatch.delenv("VECTOR_BASE_URL", raising=False)


@pytest.fixture
def gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gm_fake")


def mock_openai(text="claude says hi"):
    """Patch openai.OpenAI, returning the mock class for call inspection."""
    patcher = patch("openai.OpenAI")
    cls = patcher.start()
    cls.return_value.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=text))]
    )
    return patcher, cls


def mock_genai(text="gemini says hi"):
    patcher = patch("google.genai.Client")
    cls = patcher.start()
    cls.return_value.models.generate_content.return_value = MagicMock(text=text)
    return patcher, cls


def test_claude_targets_vector_proxy(claude_key):
    patcher, cls = mock_openai()
    try:
        assert call_claude("ping") == "claude says hi"
        kwargs = cls.call_args.kwargs
        assert kwargs["base_url"] == VECTOR_BASE_URL
        assert kwargs["api_key"] == "vp_fake"
        sent = cls.return_value.chat.completions.create.call_args.kwargs
        assert sent["model"] == "claude-sonnet-4-6"
        assert sent["max_tokens"] == 1200
        assert sent["messages"] == [{"role": "user", "content": "ping"}]
    finally:
        patcher.stop()


def test_claude_base_url_is_overridable(claude_key, monkeypatch):
    monkeypatch.setenv("VECTOR_BASE_URL", "https://example.test/v1")
    patcher, cls = mock_openai()
    try:
        call_claude("ping")
        assert cls.call_args.kwargs["base_url"] == "https://example.test/v1"
    finally:
        patcher.stop()


def test_claude_passes_model_and_max_tokens(claude_key):
    patcher, cls = mock_openai()
    try:
        call_claude("ping", model="claude-opus-4-1", max_tokens=7)
        sent = cls.return_value.chat.completions.create.call_args.kwargs
        assert sent["model"] == "claude-opus-4-1"
        assert sent["max_tokens"] == 7
    finally:
        patcher.stop()


def test_claude_empty_content_becomes_empty_string(claude_key):
    patcher, _ = mock_openai(text=None)
    try:
        assert call_claude("ping") == ""
    finally:
        patcher.stop()


def test_gemini_calls_google_directly(gemini_key):
    patcher, cls = mock_genai()
    try:
        assert call_gemini("ping") == "gemini says hi"
        assert cls.call_args.kwargs == {"api_key": "gm_fake"}
        sent = cls.return_value.models.generate_content.call_args.kwargs
        assert sent["model"] == "gemini-2.5-pro"
        assert sent["contents"] == "ping"
    finally:
        patcher.stop()


def test_gemini_empty_text_becomes_empty_string(gemini_key):
    patcher, _ = mock_genai(text=None)
    try:
        assert call_gemini("ping") == ""
    finally:
        patcher.stop()


@pytest.mark.parametrize(
    "fn,var",
    [(call_claude, "VECTOR_API_KEY"), (call_gemini, "GEMINI_API_KEY")],
)
def test_missing_key_raises_before_any_client_call(fn, var, monkeypatch):
    monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match=var):
        fn("ping")


def block_module(monkeypatch, name):
    """Make ``import name`` raise ImportError for the duration of a test.

    Poisoning sys.modules alone is not enough for a submodule: ``from google
    import genai`` resolves via the attribute already bound on the parent
    package and never consults sys.modules, so the attribute must go too.
    """
    monkeypatch.setitem(sys.modules, name, None)
    parent, _, child = name.rpartition(".")
    if parent and (mod := sys.modules.get(parent)) is not None:
        monkeypatch.delattr(mod, child, raising=False)


@pytest.mark.parametrize(
    "fn,module,env",
    [
        (call_claude, "openai", "VECTOR_API_KEY"),
        (call_gemini, "google.genai", "GEMINI_API_KEY"),
    ],
)
def test_missing_sdk_raises_install_hint(fn, module, env, monkeypatch):
    monkeypatch.setenv(env, "fake")
    block_module(monkeypatch, module)
    with pytest.raises(RuntimeError, match=r"evaljev\[llm\]"):
        fn("ping")


def test_execute_model_route_dispatches(claude_key, gemini_key):
    cp, _ = mock_openai()
    gp, _ = mock_genai()
    try:
        assert execute_model_route("claude", "ping") == {
            "provider": "claude",
            "text": "claude says hi",
        }
        assert execute_model_route("gemini", "ping") == {
            "provider": "gemini",
            "text": "gemini says hi",
        }
        assert execute_model_route("both", "ping") == {
            "provider": "both",
            "claude": "claude says hi",
            "gemini": "gemini says hi",
        }
    finally:
        cp.stop()
        gp.stop()


def test_execute_model_route_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown route"):
        execute_model_route("bogus", "ping")
