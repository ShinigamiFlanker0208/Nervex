"""
Provider-neutral LLM construction (plug-and-play models).

Why this exists: every agent used to instantiate ChatNVIDIA directly, so Nervex only worked with one
provider's key. Now ONE place decides which chat model backs each role, driven by .env:

    LLM_PROVIDER = nvidia | google | openai | openai_compatible | ollama
    LLM_MODEL    = optional override (each provider has a default)
    LLM_API_KEY  = optional; otherwise the provider's usual key is used
    LLM_BASE_URL = for openai_compatible (vLLM, LM Studio, llama.cpp server, OpenRouter, ...) / ollama

Catastrophe prevented: a missing/wrong provider silently producing a 401 deep inside a 10-minute
ingestion. `build_llm` fails immediately with a message that names the exact env var to set.

The NVIDIA profile keeps the exact kwargs the agents used before (thinking on, reasoning budget), so
behaviour with the default provider is unchanged. Other providers only get portable kwargs.
"""
import os
from typing import Any, Dict, Optional

from config import settings

PROVIDERS = ("nvidia", "google", "openai", "openai_compatible", "ollama")

DEFAULT_MODELS = {
    "nvidia": "nvidia/nemotron-3-ultra-550b-a55b",
    "google": "gemini-3.5-flash",       # 2.5-flash was retired for new keys (404 on Nov 8 2026)
    "openai": "gpt-4o-mini",
    "openai_compatible": "local-model",
    "ollama": "qwen2.5-coder:7b",
}
KEY_ENV = {
    "nvidia": ("NVIDIA_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "openai_compatible": ("OPENAI_API_KEY",),
    "ollama": (),
}
OLLAMA_DEFAULT_URL = "http://localhost:11434/v1"

# role -> (temperature, max_tokens, thinking). `thinking` is honoured by NVIDIA only.
ROLES: Dict[str, Dict[str, Any]] = {
    "extractor":  {"temperature": 0.4, "max_tokens": 16384, "thinking": True, "reasoning_budget": 16384, "top_p": 0.95},
    "architect":  {"temperature": 0.1, "max_tokens": 16384, "thinking": True, "top_p": 0.95},
    "coder":      {"temperature": 0.1, "max_tokens": 16384, "thinking": True, "top_p": 0.95},
    "critic":     {"temperature": 0.1, "max_tokens": 16384, "thinking": True, "top_p": 0.95},
    "chat":       {"temperature": 0.2, "max_tokens": 4096, "thinking": False},
    "transpiler": {"temperature": 0.1, "max_tokens": 8192, "thinking": False},
    "probe":      {"temperature": 0.0, "max_tokens": 16, "thinking": False},     # key check when switching provider
}


def resolve_provider() -> str:
    p = (settings.LLM_PROVIDER or "nvidia").strip().lower()
    if p not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER={p!r} is not supported. Choose one of: {', '.join(PROVIDERS)}.")
    return p


def resolve_api_key(provider: str) -> Optional[str]:
    if settings.LLM_API_KEY:
        return settings.LLM_API_KEY
    for name in KEY_ENV[provider]:
        v = os.environ.get(name) or getattr(settings, name, None)
        if v:
            return v
    return None


def build_llm(role: str) -> Any:
    """Return a LangChain chat model for `role` using the configured provider."""
    if role not in ROLES:
        raise ValueError(f"unknown LLM role {role!r}")
    prof = ROLES[role]
    provider = resolve_provider()
    model = settings.LLM_MODEL or DEFAULT_MODELS[provider]
    key = resolve_api_key(provider)
    if provider in ("nvidia", "google", "openai") and not key:
        need = " or ".join(KEY_ENV[provider])
        raise RuntimeError(f"LLM_PROVIDER={provider} needs an API key: set {need} (or LLM_API_KEY) in .env.")

    if provider == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        kw: Dict[str, Any] = dict(model=model, api_key=key, temperature=prof["temperature"],
                                  max_tokens=prof["max_tokens"])
        if "top_p" in prof:
            kw["top_p"] = prof["top_p"]
        if "reasoning_budget" in prof:
            kw["reasoning_budget"] = prof["reasoning_budget"]
        if prof["thinking"]:
            kw["chat_template_kwargs"] = {"enable_thinking": True}
        if settings.LLM_BASE_URL:
            kw["base_url"] = settings.LLM_BASE_URL          # self-hosted NIM
        return ChatNVIDIA(**kw)

    if provider == "google":
        return _gemini_class()(model=model, google_api_key=key, temperature=prof["temperature"],
                                      max_output_tokens=prof["max_tokens"], max_retries=1)

    # openai / openai_compatible / ollama all speak the OpenAI chat protocol
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:                                # pragma: no cover
        raise RuntimeError("Provider needs `langchain-openai` (uv sync).") from exc
    base = settings.LLM_BASE_URL or (OLLAMA_DEFAULT_URL if provider == "ollama" else None)
    if provider == "openai_compatible" and not base:
        raise RuntimeError("LLM_PROVIDER=openai_compatible needs LLM_BASE_URL (e.g. http://localhost:8001/v1).")
    return ChatOpenAI(model=model, api_key=key or "not-needed", base_url=base,
                      temperature=prof["temperature"], max_tokens=prof["max_tokens"], max_retries=1)


def _flatten(content: Any) -> Any:
    """Gemini 3 returns content as a list of parts ([{"type": "text", "text": ...}, {"type": "thinking", ...}]).
    Every agent here expects a string (regex over ```python blocks), so keep only the text parts."""
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) and p.get("type", "text") == "text" else (p if isinstance(p, str) else "") for p in content)
    return content


_GEMINI = None
def _gemini_class():
    """ChatGoogleGenerativeAI whose replies always carry string content. Done at the generation level (not by
    wrapping the model in a Runnable) so bind_tools / with_structured_output keep working."""
    global _GEMINI
    if _GEMINI is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        class NervexGemini(ChatGoogleGenerativeAI):
            def _fix(self, result):
                for g in result.generations:
                    g.message.content = _flatten(g.message.content)
                    if hasattr(g, "text"): g.text = g.message.content if isinstance(g.message.content, str) else g.text
                return result
            def _generate(self, *a, **k): return self._fix(super()._generate(*a, **k))
            async def _agenerate(self, *a, **k): return self._fix(await super()._agenerate(*a, **k))
        _GEMINI = NervexGemini
    return _GEMINI


def describe() -> Dict[str, Any]:
    """Safe-to-expose summary (never the key) for health checks / the UI."""
    p = resolve_provider()
    return {"provider": p, "model": settings.LLM_MODEL or DEFAULT_MODELS[p],
            "base_url": settings.LLM_BASE_URL, "has_key": bool(resolve_api_key(p)) or p in ("ollama", "openai_compatible")}
