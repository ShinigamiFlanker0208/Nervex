"""Provider switch: no network, constructors only."""
import pytest

from agents import llm_factory
from config import settings


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL", "NVIDIA_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setattr(settings, k, None if k != "LLM_PROVIDER" else "nvidia", raising=False)
    for k in ("NVIDIA_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)


def test_default_is_nvidia_with_thinking(monkeypatch):
    monkeypatch.setattr(settings, "NVIDIA_API_KEY", "nvapi-x")
    llm = llm_factory.build_llm("coder")
    assert type(llm).__name__ == "ChatNVIDIA" and llm.model == "nvidia/nemotron-3-ultra-550b-a55b"


def test_missing_key_fails_immediately_and_names_the_env_var():
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        llm_factory.build_llm("coder")


def test_google_provider(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "google")
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "g-key")
    llm = llm_factory.build_llm("extractor")
    assert type(llm).__name__ == "ChatGoogleGenerativeAI" and llm.model.endswith("gemini-2.5-flash")


def test_google_key_from_gemini_env(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "google")
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    assert llm_factory.resolve_api_key("google") == "g-key"


def test_ollama_needs_no_key_and_defaults_to_local_url(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    llm = llm_factory.build_llm("chat")
    assert type(llm).__name__ == "ChatOpenAI" and "11434" in str(llm.openai_api_base)


def test_openai_compatible_requires_base_url(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai_compatible")
    with pytest.raises(RuntimeError, match="LLM_BASE_URL"):
        llm_factory.build_llm("chat")
    monkeypatch.setattr(settings, "LLM_BASE_URL", "http://localhost:8001/v1")
    assert llm_factory.build_llm("chat").model_name == "local-model"


def test_unknown_provider_and_role(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "bard")
    with pytest.raises(ValueError, match="not supported"):
        llm_factory.build_llm("chat")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    with pytest.raises(ValueError):
        llm_factory.build_llm("nope")


def test_describe_never_leaks_key(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "google")
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "SECRET")
    d = llm_factory.describe()
    assert "SECRET" not in str(d) and d["has_key"] is True


@pytest.mark.parametrize("msg,kind", [
    ("429 RESOURCE_EXHAUSTED. quota exceeded", "rate_limited"),
    ("Error code: 429 - {'error': 'rate'}", "rate_limited"),
    ("Error code: 503 - overloaded", "upstream_error"),
    ("400 API key not valid. Please pass a valid API key.", "upstream_auth"),
    ("[429] Too Many Requests", "rate_limited"),
])
def test_other_providers_errors_are_classified(msg, kind):
    from agents.llm_retry import classify
    assert classify(Exception(msg))[1] == kind
